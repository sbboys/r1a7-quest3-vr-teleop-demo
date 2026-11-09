#!/usr/bin/env python3

import argparse
import copy
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np


CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)

HEADER_H = 100
DISPLAY_W = 1200

COLORS = {
    1: (50, 190, 100),
    2: (40, 165, 255),
    255: (255, 0, 255),
}


class HoleBoardAnnotator:

    def __init__(
        self,
        dataset,
        session,
        limit_groups=12,
        start_sample=None,
    ):

        self.root = Path(dataset)
        self.session = Path(session)

        camera_ids = []

        for camera in CAMERAS:

            folder = self.session / camera

            ids = {
                p.stem
                for p in folder.glob("*.png")
            }

            camera_ids.append(ids)

        common_ids = sorted(
            set.intersection(*camera_ids)
        )

        if not common_ids:
            raise RuntimeError(
                "No synchronized three-camera samples found"
            )

        if start_sample is not None:

            if start_sample not in common_ids:
                raise RuntimeError(
                    f"Sample not found: {start_sample}"
                )

            common_ids = [
                sid for sid in common_ids
                if sid >= start_sample
            ]

        common_ids = common_ids[:limit_groups]

        self.items = []

        for sid in common_ids:
            for camera in CAMERAS:

                self.items.append({
                    "sample_id": sid,
                    "camera": camera,
                    "image": (
                        self.session
                        / camera
                        / f"{sid}.png"
                    ),
                    "mask": (
                        self.root
                        / "masks"
                        / camera
                        / f"{sid}.png"
                    ),
                })

        self.geometry_file = (
            self.root
            / "annotations"
            / "hole_geometry.jsonl"
        )

        self.log_file = (
            self.root
            / "annotations"
            / "annotation_log.jsonl"
        )

        self.geometry_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.geometry_records = {}

        if self.geometry_file.exists():

            for line in self.geometry_file.read_text(
                encoding="utf-8"
            ).splitlines():

                if not line.strip():
                    continue

                record = json.loads(line)

                key = (
                    record["sample_id"],
                    record["camera"],
                )

                self.geometry_records[key] = record

        self.index = 0

        self.mode = "idle"
        self.points = []

        self.undo_stack = []

        self.image = None
        self.mask = None
        self.saved_mask = None

        self.holes = []
        self.board_visible = False
        self.current_hole_index = None

        self.dirty = False
        self.overlay = True

        self.zoom = 1.0
        self.cx = 0.0
        self.cy = 0.0

        self.display_width = DISPLAY_W
        self.window = "R1-A7 HOLE BOARD ANNOTATOR V1"

        self.load_current()

    def current(self):
        return self.items[self.index]

    def load_current(self):

        item = self.current()

        self.image = cv2.imread(
            str(item["image"]),
            cv2.IMREAD_COLOR,
        )

        if self.image is None:
            raise RuntimeError(
                f"Cannot read {item['image']}"
            )

        h, w = self.image.shape[:2]

        if item["mask"].exists():

            mask = cv2.imread(
                str(item["mask"]),
                cv2.IMREAD_UNCHANGED,
            )

            if (
                mask is None
                or mask.ndim != 2
                or mask.dtype != np.uint8
                or mask.shape != (h, w)
            ):
                raise RuntimeError(
                    f"Invalid mask: {item['mask']}"
                )

            if not set(
                int(x) for x in np.unique(mask)
            ).issubset({0, 1, 2, 255}):

                raise RuntimeError(
                    f"Invalid mask labels: {item['mask']}"
                )

            self.mask = mask.copy()

        else:

            self.mask = np.zeros(
                (h, w),
                dtype=np.uint8,
            )

        self.saved_mask = self.mask.copy()

        key = (
            item["sample_id"],
            item["camera"],
        )

        record = self.geometry_records.get(
            key,
            {},
        )

        self.holes = copy.deepcopy(
            record.get("holes", [])
        )

        self.board_visible = bool(
            record.get("board_visible", False)
        )

        self.current_hole_index = (
            len(self.holes) - 1
            if self.holes
            else None
        )

        self.points = []
        self.undo_stack = []

        self.mode = "idle"
        self.dirty = False

        self.zoom = 1.0
        self.cx = w / 2.0
        self.cy = h / 2.0

        print()
        print("=" * 65)
        print(
            "SAMPLE:",
            item["sample_id"],
            "| CAMERA:",
            item["camera"],
        )
        print(
            "MASK:",
            "EXISTING"
            if item["mask"].exists()
            else "NEW",
        )
        print(
            "HOLES:",
            len(self.holes),
        )
        print("=" * 65)

    def push_undo(self):

        self.undo_stack.append({
            "mask": self.mask.copy(),
            "holes": copy.deepcopy(self.holes),
            "board_visible": self.board_visible,
        })

        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    def undo(self):

        if not self.undo_stack:
            print("[UNDO] nothing to undo")
            return

        state = self.undo_stack.pop()

        self.mask = state["mask"]
        self.holes = state["holes"]
        self.board_visible = state["board_visible"]

        self.current_hole_index = (
            len(self.holes) - 1
            if self.holes
            else None
        )

        self.points = []
        self.dirty = True

        print("[UNDO] restored previous operation")

    def geometry(self):

        h, w = self.image.shape[:2]

        cw = max(
            1,
            min(w, int(round(w / self.zoom))),
        )

        ch = max(
            1,
            min(h, int(round(h / self.zoom))),
        )

        x0 = int(round(self.cx - cw / 2))
        y0 = int(round(self.cy - ch / 2))

        x0 = max(0, min(w - cw, x0))
        y0 = max(0, min(h - ch, y0))

        self.cx = x0 + cw / 2
        self.cy = y0 + ch / 2

        dh = int(round(
            self.display_width * ch / cw
        ))

        return x0, y0, cw, ch, dh

    def screen_to_image(self, sx, sy):

        x0, y0, cw, ch, dh = self.geometry()

        sy -= HEADER_H

        if (
            sx < 0
            or sx >= self.display_width
            or sy < 0
            or sy >= dh
        ):
            return None

        ix = x0 + (
            sx / max(1, self.display_width - 1)
        ) * (cw - 1)

        iy = y0 + (
            sy / max(1, dh - 1)
        ) * (ch - 1)

        return (
            int(round(ix)),
            int(round(iy)),
        )

    def image_to_screen(self, ix, iy):

        x0, y0, cw, ch, dh = self.geometry()

        sx = int(
            (ix - x0)
            * self.display_width
            / max(1, cw)
        )

        sy = HEADER_H + int(
            (iy - y0)
            * dh
            / max(1, ch)
        )

        return sx, sy

    def mouse(self, event, x, y, flags, userdata):

        point = self.screen_to_image(x, y)

        if point is None:
            return

        if event == cv2.EVENT_RBUTTONDOWN:

            if self.points:
                self.points.pop()
                print("[POINT] removed last point")

            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.mode == "center":

            if self.current_hole_index is None:
                return

            self.push_undo()

            hole = self.holes[
                self.current_hole_index
            ]

            hole["center_uv"] = [
                float(point[0]),
                float(point[1]),
            ]

            hole["center_valid"] = True

            self.mode = "idle"
            self.dirty = True

            print(
                "[CENTER]",
                hole["hole_id"],
                hole["center_uv"],
            )

            return

        if self.mode in (
            "board",
            "hole",
            "ignore",
            "background",
        ):

            self.points.append(point)

            print(
                "[POINT]",
                len(self.points),
                point,
            )

    def commit_polygon(self):

        if self.mode not in (
            "board",
            "hole",
            "ignore",
            "background",
        ):
            return

        if len(self.points) < 3:

            print(
                "[POLYGON] at least 3 points required"
            )

            return

        self.push_undo()

        polygon = np.asarray(
            self.points,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        labels = {
            "board": 1,
            "hole": 2,
            "ignore": 255,
            "background": 0,
        }

        label = labels[self.mode]

        cv2.fillPoly(
            self.mask,
            [polygon],
            int(label),
        )

        if self.mode == "board":

            self.board_visible = True

            print(
                "[BOARD COMMITTED]",
                len(self.points),
                "points",
            )

        elif self.mode == "hole":

            raw_id = input(
                "Enter physical hole ID "
                "(example R01_C01): "
            ).strip().upper()

            if not raw_id:
                raw_id = (
                    f"H{len(self.holes) + 1:03d}"
                )

            match = re.fullmatch(
                r"R(\d+)_C(\d+)",
                raw_id,
            )

            row = (
                int(match.group(1))
                if match
                else None
            )

            col = (
                int(match.group(2))
                if match
                else None
            )

            if any(
                h["hole_id"] == raw_id
                for h in self.holes
            ):
                print(
                    "[WARNING] duplicate hole ID:",
                    raw_id,
                )

            self.holes.append({
                "hole_id": raw_id,
                "row": row,
                "col": col,
                "polygon_uv": [
                    [int(x), int(y)]
                    for x, y in self.points
                ],
                "center_uv": None,
                "visible": True,
                "center_valid": False,
                "target": False,
            })

            self.current_hole_index = (
                len(self.holes) - 1
            )

            print(
                "[HOLE COMMITTED]",
                raw_id,
            )

            print(
                "Press P and click its center."
            )

        else:

            print(
                "[MASK COMMITTED]",
                "class=",
                label,
            )

        self.points = []
        self.mode = "idle"
        self.dirty = True

    def mark_target(self):

        if self.current_hole_index is None:
            print("[TARGET] no current hole")
            return

        self.push_undo()

        hole = self.holes[
            self.current_hole_index
        ]

        hole["target"] = not bool(
            hole.get("target", False)
        )

        self.dirty = True

        print(
            "[TARGET]",
            hole["hole_id"],
            hole["target"],
        )

    def save(self):

        item = self.current()

        mask_path = item["mask"]

        mask_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not cv2.imwrite(
            str(mask_path),
            self.mask,
        ):
            raise RuntimeError(
                f"Failed to save mask: {mask_path}"
            )

        record = {
            "sample_id": item["sample_id"],
            "camera": item["camera"],
            "board_visible": self.board_visible,
            "holes": self.holes,
            "annotation_status": "annotated",
            "updated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        }

        key = (
            item["sample_id"],
            item["camera"],
        )

        self.geometry_records[key] = record

        ordered = sorted(
            self.geometry_records.values(),
            key=lambda r: (
                r["sample_id"],
                CAMERAS.index(r["camera"]),
            ),
        )

        tmp = Path(
            str(self.geometry_file) + ".tmp"
        )

        with tmp.open(
            "w",
            encoding="utf-8",
        ) as f:

            for r in ordered:
                f.write(
                    json.dumps(
                        r,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        tmp.replace(
            self.geometry_file
        )

        log_record = {
            "sample_id": item["sample_id"],
            "camera": item["camera"],
            "mask": str(mask_path),
            "hole_count": len(self.holes),
            "pixel_counts": {
                str(v): int(
                    np.count_nonzero(
                        self.mask == v
                    )
                )
                for v in (0, 1, 2, 255)
            },
            "status": "annotated",
            "time": time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        }

        with self.log_file.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    log_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        preview = self.render()

        preview_dir = (
            self.root
            / "mask_previews"
        )

        preview_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        preview_path = (
            preview_dir
            /
            f"{item['sample_id']}_{item['camera']}.jpg"
        )

        cv2.imwrite(
            str(preview_path),
            preview,
        )

        self.saved_mask = self.mask.copy()
        self.dirty = False

        print()
        print("[SAVED]", item["sample_id"], item["camera"])
        print("MASK:", mask_path)
        print("HOLES:", len(self.holes))
        print("PREVIEW:", preview_path)

    def render(self):

        x0, y0, cw, ch, dh = self.geometry()

        image = self.image[
            y0:y0+ch,
            x0:x0+cw,
        ]

        mask = self.mask[
            y0:y0+ch,
            x0:x0+cw,
        ]

        image = cv2.resize(
            image,
            (
                self.display_width,
                dh,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (
                self.display_width,
                dh,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        overlay = image.copy()

        for label, color in COLORS.items():
            overlay[mask == label] = color

        if self.overlay:
            shown = cv2.addWeighted(
                image,
                0.65,
                overlay,
                0.35,
                0,
            )
        else:
            shown = image.copy()

        for i, point in enumerate(self.points):

            sx, sy = self.image_to_screen(
                point[0],
                point[1],
            )

            cv2.circle(
                shown,
                (sx, sy - HEADER_H),
                4,
                (0, 255, 255),
                -1,
            )

            if i > 0:

                p0 = self.image_to_screen(
                    *self.points[i-1]
                )

                cv2.line(
                    shown,
                    (
                        p0[0],
                        p0[1] - HEADER_H,
                    ),
                    (
                        sx,
                        sy - HEADER_H,
                    ),
                    (0, 255, 255),
                    1,
                )

        for hole in self.holes:

            if hole.get("center_valid"):

                u, v = hole["center_uv"]

                sx, sy = self.image_to_screen(
                    u,
                    v,
                )

                cv2.drawMarker(
                    shown,
                    (sx, sy - HEADER_H),
                    (255, 255, 0),
                    cv2.MARKER_CROSS,
                    16,
                    2,
                )

        canvas = np.zeros(
            (
                HEADER_H + dh,
                self.display_width,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            HEADER_H:
        ] = shown

        item = self.current()

        status = (
            "UNSAVED"
            if self.dirty
            else "SAVED"
        )

        cv2.putText(
            canvas,
            (
                f"{item['sample_id']} "
                f"{item['camera'].upper()} "
                f"{self.index+1}/{len(self.items)} "
                f"MODE={self.mode.upper()} "
                f"HOLES={len(self.holes)} "
                f"{status}"
            ),
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )

        cv2.putText(
            canvas,
            "B Board | H Hole | ENTER Commit | P Center | T Target | I Ignore",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            canvas,
            "0 Background | U Undo | K Save | N Next | J Previous | V Overlay",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            canvas,
            "Z/X Zoom | W/A/S/D Pan | Right Click Remove Point | Q Quit",
            (10, 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
        )

        return canvas

    def navigate(self, direction):

        if self.dirty:

            print(
                "[BLOCKED] Save with K before navigation."
            )

            return

        next_index = (
            self.index + direction
        )

        if not 0 <= next_index < len(self.items):
            print("[END] no more images")
            return

        self.index = next_index
        self.load_current()

    def pan(self, dx, dy):

        x0, y0, cw, ch, _ = self.geometry()

        self.cx += dx * cw * 0.12
        self.cy += dy * ch * 0.12

    def run(self):

        cv2.namedWindow(
            self.window,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            self.window,
            1200,
            900,
        )

        cv2.setMouseCallback(
            self.window,
            self.mouse,
        )

        print()
        print("=" * 65)
        print("R1-A7 HOLE BOARD ANNOTATOR V1")
        print("=" * 65)
        print("B: Board polygon -> Class 1")
        print("H: Hole polygon -> Class 2")
        print("P: Click center of latest hole")
        print("T: Toggle target hole")
        print("I: Ignore polygon -> 255")
        print("0: Background correction -> 0")
        print("ENTER: Commit polygon")
        print("K: Save mask + geometry JSON")
        print("U: Undo")
        print("N/J: Next/Previous")
        print("Q: Quit after saving")
        print("=" * 65)

        while True:

            cv2.imshow(
                self.window,
                self.render(),
            )

            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            if key in (13, 10):
                self.commit_polygon()
                continue

            ch = chr(key).lower()

            if ch == "b":
                self.mode = "board"
                self.points = []
                print("[MODE] BOARD")

            elif ch == "h":
                self.mode = "hole"
                self.points = []
                print("[MODE] HOLE")

            elif ch == "i":
                self.mode = "ignore"
                self.points = []
                print("[MODE] IGNORE")

            elif ch == "0":
                self.mode = "background"
                self.points = []
                print("[MODE] BACKGROUND")

            elif ch == "p":

                if self.current_hole_index is None:
                    print(
                        "[CENTER] First commit a hole polygon."
                    )
                else:
                    self.mode = "center"
                    print(
                        "[CENTER] Click center of",
                        self.holes[self.current_hole_index]["hole_id"],
                    )

            elif ch == "t":
                self.mark_target()

            elif ch == "k":
                self.save()

            elif ch == "u":
                self.undo()

            elif ch == "v":
                self.overlay = not self.overlay

            elif ch == "n":
                self.navigate(+1)

            elif ch == "j":
                self.navigate(-1)

            elif ch == "z":
                self.zoom = min(
                    5.0,
                    self.zoom * 1.25,
                )

            elif ch == "x":
                self.zoom = max(
                    1.0,
                    self.zoom / 1.25,
                )

            elif ch == "w":
                self.pan(0, -1)

            elif ch == "s":
                self.pan(0, 1)

            elif ch == "a":
                self.pan(-1, 0)

            elif ch == "d":
                self.pan(1, 0)

            elif ch == "f":

                self.zoom = 1.0
                self.cx = self.image.shape[1] / 2
                self.cy = self.image.shape[0] / 2

            elif ch == "r":

                if self.dirty:
                    print(
                        "[BLOCKED] Save before reload."
                    )
                else:
                    self.load_current()

            elif ch == "q":

                if self.dirty:
                    print(
                        "[BLOCKED] Unsaved changes. Press K first."
                    )
                else:
                    break

            elif key == 27:

                self.points = []
                self.mode = "idle"
                print("[CANCEL] pending polygon cleared")

        cv2.destroyAllWindows()

        print()
        print("ANNOTATION SESSION FINISHED")
        print("Geometry:", self.geometry_file)
        print("Masks:", self.root / "masks")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--session",
        required=True,
    )

    parser.add_argument(
        "--limit-groups",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--start-sample",
        default=None,
    )

    args = parser.parse_args()

    annotator = HoleBoardAnnotator(
        dataset=args.dataset,
        session=args.session,
        limit_groups=args.limit_groups,
        start_sample=args.start_sample,
    )

    annotator.run()


if __name__ == "__main__":
    main()
