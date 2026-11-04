#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("external", "right_wrist", "left_wrist")

LABEL_NAME = {
    0: "background",
    1: "wire",
    2: "tool",
    255: "ignore",
}

LABEL_COLOR = {
    1: (0, 255, 0),
    2: (0, 0, 255),
    255: (255, 0, 255),
}

HEADER_H = 92


class Annotator:
    def __init__(self, root, split_name, limit=None):
        self.root = Path(root)
        self.split_name = split_name

        split = json.loads(
            (self.root / "splits_sample_level_seed42.json")
            .read_text(encoding="utf-8")
        )

        self.items = []

        for sid in split[split_name]:
            for camera in CAMERAS:
                image = self.root / "images" / camera / f"{sid}.png"
                mask = self.root / "masks" / camera / f"{sid}.png"

                if not image.exists():
                    raise RuntimeError(f"Missing image: {image}")

                self.items.append({
                    "sample_id": sid,
                    "camera": camera,
                    "image": image,
                    "mask": mask,
                })

        if limit is not None:
            self.items = self.items[:limit]

        self.index = 0

        # Modes:
        # wire = polyline -> class 1
        # tool = polygon  -> class 2
        # brush = manual correction
        self.mode = "wire"
        self.brush_label = 1

        self.wire_width = 1
        self.brush_radius = 1

        self.points = []

        self.image = None
        self.mask = None
        self.saved_mask = None

        self.zoom = 1.0
        self.cx = 0.0
        self.cy = 0.0

        self.display_width = 1200

        self.dirty = False
        self.drawing = False
        self.last_point = None
        self.undo_stack = []

        self.overlay_enabled = True
        self.window = "R1A7 U-Net Semi-Auto Annotator"

        (self.root / "mask_previews").mkdir(
            parents=True,
            exist_ok=True
        )

        (self.root / "annotation_work" / "backups").mkdir(
            parents=True,
            exist_ok=True
        )

        self.load_current()

    def current(self):
        return self.items[self.index]

    def load_current(self):
        item = self.current()

        self.image = cv2.imread(
            str(item["image"]),
            cv2.IMREAD_COLOR
        )

        if self.image is None:
            raise RuntimeError(
                f"Cannot read image: {item['image']}"
            )

        h, w = self.image.shape[:2]

        if item["mask"].exists():
            mask = cv2.imread(
                str(item["mask"]),
                cv2.IMREAD_UNCHANGED
            )

            if (
                mask is None or
                mask.dtype != np.uint8 or
                mask.ndim != 2 or
                mask.shape != (h, w)
            ):
                raise RuntimeError(
                    f"Invalid mask: {item['mask']}"
                )

            values = set(
                int(v) for v in np.unique(mask)
            )

            if not values.issubset({0, 1, 2, 255}):
                raise RuntimeError(
                    f"Invalid mask values: {values}"
                )

            self.mask = mask.copy()

        else:
            # In memory only.
            # It is NOT a confirmed negative until K is pressed.
            self.mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )

        self.saved_mask = self.mask.copy()

        self.zoom = 1.0
        self.cx = w / 2.0
        self.cy = h / 2.0

        self.points.clear()
        self.undo_stack.clear()

        self.dirty = False
        self.drawing = False
        self.last_point = None

    def geometry(self):
        h, w = self.image.shape[:2]

        cw = max(
            1,
            min(w, int(round(w / self.zoom)))
        )

        ch = max(
            1,
            min(h, int(round(h / self.zoom)))
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

        if sy < 0:
            return None

        ix = x0 + (
            sx / max(1, self.display_width - 1)
        ) * (cw - 1)

        iy = y0 + (
            sy / max(1, dh - 1)
        ) * (ch - 1)

        h, w = self.image.shape[:2]

        ix = int(max(0, min(w - 1, round(ix))))
        iy = int(max(0, min(h - 1, round(iy))))

        return ix, iy

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

    def push_undo(self):
        self.undo_stack.append(
            self.mask.copy()
        )

        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    def brush_paint(self, p0, p1):
        value = int(self.brush_label)

        thickness = max(
            1,
            self.brush_radius * 2 - 1
        )

        cv2.line(
            self.mask,
            p0,
            p1,
            value,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )

        cv2.circle(
            self.mask,
            p1,
            self.brush_radius,
            value,
            -1,
            lineType=cv2.LINE_8,
        )

        self.dirty = not np.array_equal(
            self.mask,
            self.saved_mask
        )

    def mouse(self, event, x, y, flags, userdata):
        point = self.screen_to_image(x, y)

        if point is None:
            return

        if self.mode == "brush":

            if event == cv2.EVENT_LBUTTONDOWN:
                self.push_undo()

                self.drawing = True
                self.last_point = point

                self.brush_paint(
                    point,
                    point
                )

            elif (
                event == cv2.EVENT_MOUSEMOVE
                and self.drawing
            ):
                self.brush_paint(
                    self.last_point,
                    point
                )

                self.last_point = point

            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.last_point = None

        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append(point)

            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.points:
                    removed = self.points.pop()
                    print(
                        "[POINT REMOVED]",
                        removed
                    )

    def commit_geometry(self):
        if self.mode == "wire":

            if len(self.points) < 2:
                print(
                    "[WIRE] Need at least 2 points."
                )
                return

            self.push_undo()

            pts = np.array(
                self.points,
                dtype=np.int32
            ).reshape((-1, 1, 2))

            cv2.polylines(
                self.mask,
                [pts],
                False,
                1,
                thickness=self.wire_width,
                lineType=cv2.LINE_8,
            )

            print(
                "[WIRE COMMITTED]",
                len(self.points),
                "points, width=",
                self.wire_width,
            )

        elif self.mode == "tool":

            if len(self.points) < 3:
                print(
                    "[TOOL] Need at least 3 points."
                )
                return

            self.push_undo()

            pts = np.array(
                self.points,
                dtype=np.int32
            ).reshape((-1, 1, 2))

            cv2.fillPoly(
                self.mask,
                [pts],
                2,
                lineType=cv2.LINE_8,
            )

            print(
                "[TOOL COMMITTED]",
                len(self.points),
                "points"
            )

        else:
            return

        self.points.clear()

        self.dirty = not np.array_equal(
            self.mask,
            self.saved_mask
        )

    def render(self):
        x0, y0, cw, ch, dh = self.geometry()

        image = self.image[
            y0:y0 + ch,
            x0:x0 + cw
        ].copy()

        mask = self.mask[
            y0:y0 + ch,
            x0:x0 + cw
        ]

        if self.overlay_enabled:
            overlay = image.copy()

            for value, color in LABEL_COLOR.items():
                overlay[mask == value] = color

            image = cv2.addWeighted(
                image,
                0.70,
                overlay,
                0.30,
                0
            )

        interpolation = (
            cv2.INTER_NEAREST
            if self.zoom >= 4.0
            else cv2.INTER_LINEAR
        )

        image = cv2.resize(
            image,
            (self.display_width, dh),
            interpolation=interpolation,
        )

        header = np.zeros(
            (
                HEADER_H,
                self.display_width,
                3,
            ),
            dtype=np.uint8
        )

        display = np.vstack(
            [header, image]
        )

        item = self.current()

        state = (
            "SAVED"
            if item["mask"].exists()
            else "UNLABELED"
        )

        if self.dirty:
            state += " MODIFIED"

        size_text = ""

        if self.mode == "wire":
            size_text = (
                f"wire_width={self.wire_width}px"
            )

        elif self.mode == "brush":
            size_text = (
                f"brush={self.brush_radius}px "
                f"class={self.brush_label}:"
                f"{LABEL_NAME[self.brush_label]}"
            )

        else:
            size_text = "polygon fill class=2"

        cv2.putText(
            display,
            (
                f"{self.index + 1}/{len(self.items)} | "
                f"{self.split_name} | "
                f"{item['sample_id']} | "
                f"{item['camera']} | {state}"
            ),
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            (
                f"MODE={self.mode.upper()} | "
                f"{size_text} | "
                f"zoom={self.zoom:.1f}x | "
                f"pending_points={len(self.points)}"
            ),
            (12, 59),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.59,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            (
                "L=wire  G=tool  B=brush  "
                "ENTER=commit  K=save  N=next  J=prev"
            ),
            (12, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Show pending polyline/polygon points.
        if self.points:
            screen_points = [
                self.image_to_screen(x, y)
                for x, y in self.points
            ]

            for sx, sy in screen_points:
                if (
                    0 <= sx < self.display_width
                    and HEADER_H <= sy < display.shape[0]
                ):
                    cv2.circle(
                        display,
                        (sx, sy),
                        5,
                        (0, 255, 255),
                        -1,
                    )

            for a, b in zip(
                screen_points[:-1],
                screen_points[1:]
            ):
                cv2.line(
                    display,
                    a,
                    b,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if (
                self.mode == "tool"
                and len(screen_points) >= 3
            ):
                cv2.line(
                    display,
                    screen_points[-1],
                    screen_points[0],
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        return display

    def save(self):
        if self.points:
            print(
                "[BLOCKED] Pending points exist. "
                "ENTER=commit or C=clear first."
            )
            return

        item = self.current()
        target = item["mask"]

        target.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        values = set(
            int(v)
            for v in np.unique(self.mask)
        )

        if not values.issubset(
            {0, 1, 2, 255}
        ):
            raise RuntimeError(
                f"Invalid mask values: {values}"
            )

        if target.exists():
            old = cv2.imread(
                str(target),
                cv2.IMREAD_UNCHANGED
            )

            if (
                old is not None
                and not np.array_equal(
                    old,
                    self.mask
                )
            ):
                backup_dir = (
                    self.root /
                    "annotation_work" /
                    "backups" /
                    item["camera"]
                )

                backup_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                stamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                shutil.copy2(
                    target,
                    backup_dir /
                    f"{item['sample_id']}_{stamp}.png"
                )

        temp = target.with_name(
            target.stem + ".tmp.png"
        )

        if not cv2.imwrite(
            str(temp),
            self.mask
        ):
            raise RuntimeError(
                "Failed to write mask"
            )

        check = cv2.imread(
            str(temp),
            cv2.IMREAD_UNCHANGED
        )

        if (
            check is None
            or check.dtype != np.uint8
            or check.ndim != 2
            or check.shape != self.mask.shape
        ):
            temp.unlink(missing_ok=True)

            raise RuntimeError(
                "Mask verification failed"
            )

        check_values = set(
            int(v)
            for v in np.unique(check)
        )

        if not check_values.issubset(
            {0, 1, 2, 255}
        ):
            temp.unlink(missing_ok=True)

            raise RuntimeError(
                f"Bad values after write: {check_values}"
            )

        os.replace(
            temp,
            target
        )

        preview = self.image.copy()
        overlay = preview.copy()

        for value, color in LABEL_COLOR.items():
            overlay[self.mask == value] = color

        preview = cv2.addWeighted(
            preview,
            0.65,
            overlay,
            0.35,
            0
        )

        cv2.imwrite(
            str(
                self.root /
                "mask_previews" /
                (
                    f"{item['camera']}_"
                    f"{item['sample_id']}.jpg"
                )
            ),
            preview,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                92,
            ],
        )

        counts = {
            str(value): int(
                np.sum(self.mask == value)
            )
            for value in (
                0,
                1,
                2,
                255,
            )
        }

        status = (
            "confirmed_background_only"
            if (
                counts["1"] == 0
                and counts["2"] == 0
                and counts["255"] == 0
            )
            else "annotated"
        )

        record = {
            "time": datetime.now().isoformat(),
            "sample_id": item["sample_id"],
            "camera": item["camera"],
            "split": self.split_name,
            "status": status,
            "pixel_counts": counts,
            "mask": str(target),
            "annotator": "semi_auto_v2",
        }

        with (
            self.root /
            "annotation_log.jsonl"
        ).open(
            "a",
            encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

        self.saved_mask = self.mask.copy()
        self.dirty = False

        print(
            "[SAVED]",
            item["sample_id"],
            item["camera"],
            counts,
        )

    def move(self, delta):
        if self.points:
            print(
                "[BLOCKED] Pending geometry. "
                "ENTER=commit or C=clear."
            )
            return

        if self.dirty:
            print(
                "[BLOCKED] Unsaved mask. "
                "K=save or R=discard."
            )
            return

        new_index = self.index + delta

        if not (
            0 <= new_index < len(self.items)
        ):
            print("[END] queue boundary")
            return

        self.index = new_index
        self.load_current()

    def run(self):
        cv2.namedWindow(
            self.window,
            cv2.WINDOW_NORMAL
        )

        cv2.setMouseCallback(
            self.window,
            self.mouse
        )

        print("""
============================================================
R1-A7 U-NET SEMI-AUTO ANNOTATOR V2
============================================================

MODES
  L : WIRE POLYLINE
      left click visible wire center points
      ENTER commits current visible segment as class 1

  G : TOOL POLYGON
      left click tool/tweezer boundary points
      ENTER closes + fills polygon as class 2

  B : BRUSH correction
      1 = wire
      2 = tool
      0 = background/erase
      I = ignore 255
      left drag = paint

GEOMETRY
  Right mouse = remove last pending point
  C           = clear pending points
  ENTER       = commit current polyline/polygon

SIZE
  [ / ] = decrease/increase wire width or brush radius

VIEW
  Z / X   = zoom in/out
  W A S D = pan
  F       = fit full image
  V       = overlay on/off

EDIT
  U = undo last committed/drawn operation
  K = SAVE mask
  N = next image
  J = previous image
  R = discard unsaved changes/reload
  Q = quit

IMPORTANT
  * Only visible wire is class 1.
  * NEVER bridge wire through an occluding tool.
  * Commit separate wire segments on each side of an occlusion.
  * Polygon fills visible tool area.
  * Brush is for corrections and ambiguous boundaries.
============================================================
""")

        while True:
            cv2.imshow(
                self.window,
                self.render()
            )

            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            if key in (ord("l"), ord("L")):
                self.points.clear()
                self.mode = "wire"
                print("[MODE] WIRE POLYLINE")

            elif key in (ord("g"), ord("G")):
                self.points.clear()
                self.mode = "tool"
                print("[MODE] TOOL POLYGON")

            elif key in (ord("b"), ord("B")):
                self.points.clear()
                self.mode = "brush"
                print("[MODE] BRUSH")

            elif key == ord("1"):
                self.mode = "brush"
                self.points.clear()
                self.brush_label = 1
                print("[BRUSH] wire=1")

            elif key == ord("2"):
                self.mode = "brush"
                self.points.clear()
                self.brush_label = 2
                print("[BRUSH] tool=2")

            elif key == ord("0"):
                self.mode = "brush"
                self.points.clear()
                self.brush_label = 0
                print("[BRUSH] background=0")

            elif key in (ord("i"), ord("I")):
                self.mode = "brush"
                self.points.clear()
                self.brush_label = 255
                print("[BRUSH] ignore=255")

            elif key in (10, 13):
                self.commit_geometry()

            elif key in (ord("c"), ord("C")):
                self.points.clear()
                print("[POINTS] cleared")

            elif key == ord("["):
                if self.mode == "wire":
                    self.wire_width = max(
                        1,
                        self.wire_width - 1
                    )
                    print(
                        "[WIRE WIDTH]",
                        self.wire_width
                    )
                elif self.mode == "brush":
                    self.brush_radius = max(
                        1,
                        self.brush_radius - 1
                    )
                    print(
                        "[BRUSH RADIUS]",
                        self.brush_radius
                    )

            elif key == ord("]"):
                if self.mode == "wire":
                    self.wire_width = min(
                        20,
                        self.wire_width + 1
                    )
                    print(
                        "[WIRE WIDTH]",
                        self.wire_width
                    )
                elif self.mode == "brush":
                    self.brush_radius = min(
                        50,
                        self.brush_radius + 1
                    )
                    print(
                        "[BRUSH RADIUS]",
                        self.brush_radius
                    )

            elif key in (ord("z"), ord("Z")):
                self.zoom = min(
                    16.0,
                    self.zoom * 2.0
                )

            elif key in (ord("x"), ord("X")):
                self.zoom = max(
                    1.0,
                    self.zoom / 2.0
                )

            elif key in (ord("f"), ord("F")):
                h, w = self.image.shape[:2]
                self.zoom = 1.0
                self.cx = w / 2
                self.cy = h / 2

            elif key in (
                ord("w"), ord("W"),
                ord("a"), ord("A"),
                ord("s"), ord("S"),
                ord("d"), ord("D"),
            ):
                _, _, cw, ch, _ = (
                    self.geometry()
                )

                if key in (ord("w"), ord("W")):
                    self.cy -= ch * 0.20

                elif key in (ord("s"), ord("S")):
                    self.cy += ch * 0.20

                elif key in (ord("a"), ord("A")):
                    self.cx -= cw * 0.20

                elif key in (ord("d"), ord("D")):
                    self.cx += cw * 0.20

            elif key in (ord("u"), ord("U")):
                if self.undo_stack:
                    self.mask = (
                        self.undo_stack.pop()
                    )

                    self.points.clear()

                    self.dirty = not np.array_equal(
                        self.mask,
                        self.saved_mask
                    )

                    print("[UNDO]")

            elif key in (ord("v"), ord("V")):
                self.overlay_enabled = (
                    not self.overlay_enabled
                )

            elif key in (ord("k"), ord("K")):
                self.save()

            elif key in (ord("n"), ord("N")):
                self.move(+1)

            elif key in (ord("j"), ord("J")):
                self.move(-1)

            elif key in (ord("r"), ord("R")):
                self.load_current()
                print("[RELOAD] unsaved changes discarded")

            elif key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                if self.points:
                    print(
                        "[BLOCKED] pending points: "
                        "ENTER=commit or C=clear"
                    )
                    continue

                if self.dirty:
                    print(
                        "[BLOCKED] unsaved mask: "
                        "K=save or R=discard"
                    )
                    continue

                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/pilot_001"
        ),
    )

    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    Annotator(
        args.dataset,
        args.split,
        args.limit,
    ).run()


if __name__ == "__main__":
    main()
