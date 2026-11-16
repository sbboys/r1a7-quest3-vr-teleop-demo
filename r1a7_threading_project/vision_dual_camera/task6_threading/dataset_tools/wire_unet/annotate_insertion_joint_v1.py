#!/usr/bin/env python3

import argparse
import copy
import json
import math
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)

HEADER_H = 122
DISPLAY_W = 1200

MODE_NAME = {
    "wire": "L WIRE",
    "tool": "T TWEEZER",
    "board": "B BOARD",
    "ref_hole": "R REF-HOLE",
    "hole": "H AUTO-HOLE",
    "manual_hole": "E MANUAL-HOLE",
}

WIRE_COLOR = (0, 255, 0)
TOOL_COLOR = (0, 0, 255)
BOARD_COLOR = (255, 180, 0)
HOLE_COLOR = (0, 180, 255)
PENDING_COLOR = (255, 255, 255)


class JointAnnotator:

    def __init__(
        self,
        dataset,
        session,
        start_sample=None,
        wire_width=2,
    ):
        self.root = Path(dataset)
        self.session = Path(session)
        self.wire_width = int(wire_width)

        ids = []

        for camera in CAMERAS:
            folder = self.session / camera

            if not folder.is_dir():
                raise RuntimeError(
                    f"Missing camera folder: {folder}"
                )

            ids.append({
                p.stem
                for p in folder.glob("*.png")
            })

        common = sorted(
            set.intersection(*ids)
        )

        if not common:
            raise RuntimeError(
                "No synchronized samples found"
            )

        if start_sample is not None:
            start_sample = str(
                start_sample
            ).zfill(6)

            if start_sample not in common:
                raise RuntimeError(
                    f"Start sample not found: {start_sample}"
                )

            common = [
                sid
                for sid in common
                if sid >= start_sample
            ]

        self.items = []

        for sid in common:
            for camera in CAMERAS:
                self.items.append({
                    "sid": sid,
                    "camera": camera,
                    "image": (
                        self.session
                        / camera
                        / f"{sid}.png"
                    ),
                    "wire_mask": (
                        self.root
                        / "masks_wire"
                        / camera
                        / f"{sid}.png"
                    ),
                    "hole_mask": (
                        self.root
                        / "masks_hole"
                        / camera
                        / f"{sid}.png"
                    ),
                })

        for camera in CAMERAS:
            (
                self.root
                / "masks_wire"
                / camera
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                self.root
                / "masks_hole"
                / camera
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        self.annotation_dir = (
            self.root
            / "annotations_joint"
        )

        self.annotation_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_path = (
            self.annotation_dir
            / "annotation_log.jsonl"
        )

        self.geometry_path = (
            self.annotation_dir
            / "hole_geometry.jsonl"
        )

        self.geometry_records = {}

        if self.geometry_path.exists():
            for line in (
                self.geometry_path
                .read_text(
                    encoding="utf-8"
                )
                .splitlines()
            ):
                if not line.strip():
                    continue

                r = json.loads(line)

                self.geometry_records[
                    (
                        r["sample_id"],
                        r["camera"],
                    )
                ] = r

        self.index = 0
        self.mode = "wire"
        self.points = []

        self.image = None

        self.wire_mask = None
        self.hole_mask = None

        self.saved_wire_mask = None
        self.saved_hole_mask = None

        self.ellipse_refs = []
        self.hole_entries = []

        self.undo_stack = []
        self.dirty = False

        self.zoom = 1.0
        self.cx = 0.0
        self.cy = 0.0

        self.window = (
            "R1-A7 INSERTION JOINT ANNOTATOR"
        )

        self.load_current()

    def current(self):
        return self.items[self.index]

    def validate_mask(
        self,
        mask,
        shape,
        path,
    ):
        if (
            mask is None
            or mask.dtype != np.uint8
            or mask.ndim != 2
            or mask.shape != shape
        ):
            raise RuntimeError(
                f"Invalid mask: {path}"
            )

        values = {
            int(v)
            for v in np.unique(mask)
        }

        if not values.issubset(
            {0, 1, 2, 255}
        ):
            raise RuntimeError(
                f"Invalid mask values "
                f"{values}: {path}"
            )

    def load_current(self):

        item = self.current()

        self.image = cv2.imread(
            str(item["image"]),
            cv2.IMREAD_COLOR,
        )

        if self.image is None:
            raise RuntimeError(
                f"Cannot read image: "
                f"{item['image']}"
            )

        h, w = self.image.shape[:2]

        wp = item["wire_mask"]
        hp = item["hole_mask"]

        if wp.exists() != hp.exists():
            raise RuntimeError(
                "Paired mask mismatch:\n"
                f"wire={wp.exists()} {wp}\n"
                f"hole={hp.exists()} {hp}"
            )

        if wp.exists():

            self.wire_mask = cv2.imread(
                str(wp),
                cv2.IMREAD_UNCHANGED,
            )

            self.hole_mask = cv2.imread(
                str(hp),
                cv2.IMREAD_UNCHANGED,
            )

            self.validate_mask(
                self.wire_mask,
                (h, w),
                wp,
            )

            self.validate_mask(
                self.hole_mask,
                (h, w),
                hp,
            )

        else:

            self.wire_mask = np.zeros(
                (h, w),
                dtype=np.uint8,
            )

            self.hole_mask = np.zeros(
                (h, w),
                dtype=np.uint8,
            )

        key = (
            item["sid"],
            item["camera"],
        )

        record = self.geometry_records.get(
            key,
            {},
        )

        self.ellipse_refs = copy.deepcopy(
            record.get(
                "ellipse_refs",
                [],
            )
        )

        self.hole_entries = copy.deepcopy(
            record.get(
                "holes",
                [],
            )
        )

        self.saved_wire_mask = (
            self.wire_mask.copy()
        )

        self.saved_hole_mask = (
            self.hole_mask.copy()
        )

        self.points = []
        self.undo_stack = []
        self.dirty = False

        self.zoom = 1.0
        self.cx = w / 2.0
        self.cy = h / 2.0

        print()
        print(
            "[LOAD]",
            f"{self.index + 1}/"
            f"{len(self.items)}",
            item["sid"],
            item["camera"],
            f"refs={len(self.ellipse_refs)}",
            f"holes={len(self.hole_entries)}",
        )

    def push_undo(self):

        self.undo_stack.append((
            self.wire_mask.copy(),
            self.hole_mask.copy(),
            copy.deepcopy(
                self.ellipse_refs
            ),
            copy.deepcopy(
                self.hole_entries
            ),
        ))

        if len(self.undo_stack) > 40:
            self.undo_stack.pop(0)

    def undo(self):

        if self.points:

            self.points.pop()

            print(
                "[UNDO POINT]",
                len(self.points),
                "points remain",
            )

            return

        if not self.undo_stack:
            print(
                "[UNDO] nothing to undo"
            )
            return

        (
            self.wire_mask,
            self.hole_mask,
            self.ellipse_refs,
            self.hole_entries,
        ) = self.undo_stack.pop()

        self.dirty = not (
            np.array_equal(
                self.wire_mask,
                self.saved_wire_mask,
            )
            and
            np.array_equal(
                self.hole_mask,
                self.saved_hole_mask,
            )
        )

        print(
            "[UNDO] last committed object removed"
        )

    def set_mode(
        self,
        mode,
    ):

        if self.points:
            print(
                "[BLOCKED] ENTER or C "
                "before changing mode"
            )
            return

        self.mode = mode

        print(
            "[MODE]",
            MODE_NAME[self.mode],
        )

    def geometry(self):

        h, w = self.image.shape[:2]

        self.zoom = max(
            1.0,
            min(
                8.0,
                float(self.zoom),
            ),
        )

        cw = max(
            2,
            min(
                w,
                int(round(
                    w / self.zoom
                )),
            ),
        )

        ch = max(
            2,
            min(
                h,
                int(round(
                    h / self.zoom
                )),
            ),
        )

        x0 = int(round(
            self.cx - cw / 2.0
        ))

        y0 = int(round(
            self.cy - ch / 2.0
        ))

        x0 = max(
            0,
            min(
                w - cw,
                x0,
            ),
        )

        y0 = max(
            0,
            min(
                h - ch,
                y0,
            ),
        )

        self.cx = x0 + cw / 2.0
        self.cy = y0 + ch / 2.0

        dw = max(
            2,
            min(
                DISPLAY_W,
                w,
            ),
        )

        dh = max(
            2,
            int(round(
                dw * ch / float(cw)
            )),
        )

        return (
            x0,
            y0,
            cw,
            ch,
            dw,
            dh,
        )

    def zoom_view(
        self,
        factor,
    ):

        old = self.zoom

        self.zoom = max(
            1.0,
            min(
                8.0,
                self.zoom * factor,
            ),
        )

        self.geometry()

        print(
            "[ZOOM]",
            f"{old:.2f} -> "
            f"{self.zoom:.2f}",
        )

    def pan_view(
        self,
        dx,
        dy,
    ):

        (
            _,
            _,
            cw,
            ch,
            _,
            _,
        ) = self.geometry()

        self.cx += (
            dx
            * max(
                1.0,
                cw * 0.12,
            )
        )

        self.cy += (
            dy
            * max(
                1.0,
                ch * 0.12,
            )
        )

        self.geometry()

        print(
            "[PAN]",
            round(
                self.cx,
                1,
            ),
            round(
                self.cy,
                1,
            ),
        )

    def screen_to_image(
        self,
        x,
        y,
    ):

        y -= HEADER_H

        if y < 0:
            return None

        (
            x0,
            y0,
            cw,
            ch,
            dw,
            dh,
        ) = self.geometry()

        if (
            x < 0
            or x >= dw
            or y >= dh
        ):
            return None

        ix = x0 + (
            x
            / max(
                1,
                dw - 1,
            )
        ) * max(
            1,
            cw - 1,
        )

        iy = y0 + (
            y
            / max(
                1,
                dh - 1,
            )
        ) * max(
            1,
            ch - 1,
        )

        h, w = self.image.shape[:2]

        ix = max(
            0,
            min(
                w - 1,
                int(round(ix)),
            ),
        )

        iy = max(
            0,
            min(
                h - 1,
                int(round(iy)),
            ),
        )

        return ix, iy

    def mouse(
        self,
        event,
        x,
        y,
        flags,
        userdata,
    ):

        if (
            event
            != cv2.EVENT_LBUTTONDOWN
        ):
            return

        p = self.screen_to_image(
            x,
            y,
        )

        if p is None:
            return

        if (
            self.mode == "hole"
            and len(self.points) >= 1
        ):
            print(
                "[AUTO-HOLE] "
                "one center only; "
                "ENTER or C first"
            )
            return

        if (
            self.mode
            in (
                "ref_hole",
                "manual_hole",
            )
            and len(self.points) >= 3
        ):
            print(
                "[HOLE] "
                "three points already set; "
                "ENTER or C first"
            )
            return

        self.points.append(p)

        print(
            "[POINT]",
            MODE_NAME[self.mode],
            len(self.points),
            p,
        )

    def commit_wire(self):

        if len(self.points) < 2:
            print(
                "[WIRE] "
                "need at least 2 points"
            )
            return False

        self.push_undo()

        pts = np.array(
            self.points,
            dtype=np.int32,
        )

        cv2.polylines(
            self.wire_mask,
            [pts],
            False,
            1,
            thickness=self.wire_width,
            lineType=cv2.LINE_8,
        )

        return True

    def commit_tool(self):

        if len(self.points) < 3:
            print(
                "[TWEEZER] "
                "need at least 3 points"
            )
            return False

        self.push_undo()

        region = np.zeros_like(
            self.wire_mask
        )

        pts = np.array(
            self.points,
            dtype=np.int32,
        )

        cv2.fillPoly(
            region,
            [pts],
            1,
        )

        sel = (
            (region == 1)
            &
            (self.wire_mask != 1)
        )

        self.wire_mask[sel] = 2

        return True

    def commit_board(self):

        if len(self.points) < 3:
            print(
                "[BOARD] "
                "need at least 3 points"
            )
            return False

        self.push_undo()

        region = np.zeros_like(
            self.hole_mask
        )

        pts = np.array(
            self.points,
            dtype=np.int32,
        )

        cv2.fillPoly(
            region,
            [pts],
            1,
        )

        sel = (
            (region == 1)
            &
            (self.hole_mask != 2)
        )

        self.hole_mask[sel] = 1

        return True

    def three_point_ellipse(
        self,
        points,
    ):

        if len(points) != 3:
            return None

        c = points[0]
        p1 = points[1]
        p2 = points[2]

        a = float(
            math.hypot(
                p1[0] - c[0],
                p1[1] - c[1],
            )
        )

        b = float(
            math.hypot(
                p2[0] - c[0],
                p2[1] - c[1],
            )
        )

        if (
            a < 2.0
            or b < 2.0
        ):
            return None

        angle = math.degrees(
            math.atan2(
                p1[1] - c[1],
                p1[0] - c[0],
            )
        )

        return {
            "center_uv": [
                int(c[0]),
                int(c[1]),
            ],
            "axes_px": [
                float(a),
                float(b),
            ],
            "angle_deg":
                float(angle),
        }

    def draw_hole(
        self,
        ellipse,
    ):

        c = tuple(
            int(round(v))
            for v
            in ellipse["center_uv"]
        )

        a = max(
            2,
            int(round(
                ellipse[
                    "axes_px"
                ][0]
            )),
        )

        b = max(
            2,
            int(round(
                ellipse[
                    "axes_px"
                ][1]
            )),
        )

        angle = float(
            ellipse[
                "angle_deg"
            ]
        )

        cv2.ellipse(
            self.hole_mask,
            c,
            (a, b),
            angle,
            0,
            360,
            2,
            thickness=-1,
            lineType=cv2.LINE_8,
        )

    def commit_reference_hole(self):

        ellipse = self.three_point_ellipse(
            self.points
        )

        if ellipse is None:
            print(
                "[REF-HOLE] need 3 "
                "valid points: "
                "center, major edge, "
                "minor edge"
            )
            return False

        self.push_undo()

        self.draw_hole(
            ellipse
        )

        ref = copy.deepcopy(
            ellipse
        )

        ref["method"] = (
            "reference_3point"
        )

        self.ellipse_refs.append(
            ref
        )

        hole = copy.deepcopy(
            ellipse
        )

        hole["method"] = (
            "reference_3point"
        )

        self.hole_entries.append(
            hole
        )

        print(
            "[REF-HOLE] refs =",
            len(
                self.ellipse_refs
            ),
        )

        return True

    def predict_ellipse(
        self,
        center,
    ):

        if len(
            self.ellipse_refs
        ) < 3:
            return None

        weighted = []

        for ref in self.ellipse_refs:

            rc = ref[
                "center_uv"
            ]

            d = math.hypot(
                center[0] - rc[0],
                center[1] - rc[1],
            )

            if d < 1e-6:

                out = copy.deepcopy(
                    ref
                )

                out[
                    "center_uv"
                ] = [
                    int(center[0]),
                    int(center[1]),
                ]

                return out

            weight = (
                1.0
                /
                (
                    d * d
                    + 1.0
                )
            )

            weighted.append(
                (
                    weight,
                    ref,
                )
            )

        sw = sum(
            weight
            for weight, _
            in weighted
        )

        a = sum(
            weight
            * float(
                ref[
                    "axes_px"
                ][0]
            )
            for weight, ref
            in weighted
        ) / sw

        b = sum(
            weight
            * float(
                ref[
                    "axes_px"
                ][1]
            )
            for weight, ref
            in weighted
        ) / sw

        c2 = sum(
            weight
            * math.cos(
                2.0
                * math.radians(
                    float(
                        ref[
                            "angle_deg"
                        ]
                    )
                )
            )
            for weight, ref
            in weighted
        ) / sw

        s2 = sum(
            weight
            * math.sin(
                2.0
                * math.radians(
                    float(
                        ref[
                            "angle_deg"
                        ]
                    )
                )
            )
            for weight, ref
            in weighted
        ) / sw

        angle = (
            0.5
            * math.degrees(
                math.atan2(
                    s2,
                    c2,
                )
            )
        )

        return {
            "center_uv": [
                int(center[0]),
                int(center[1]),
            ],
            "axes_px": [
                float(a),
                float(b),
            ],
            "angle_deg":
                float(angle),
        }

    def commit_auto_hole(self):

        if len(self.points) != 1:
            print(
                "[AUTO-HOLE] "
                "click exactly one "
                "hole center"
            )
            return False

        if len(
            self.ellipse_refs
        ) < 3:
            print(
                "[AUTO-HOLE] "
                "need at least "
                "3 R reference holes first"
            )
            return False

        ellipse = self.predict_ellipse(
            self.points[0]
        )

        if ellipse is None:
            print(
                "[AUTO-HOLE] "
                "prediction failed"
            )
            return False

        self.push_undo()

        self.draw_hole(
            ellipse
        )

        hole = copy.deepcopy(
            ellipse
        )

        hole["method"] = (
            "interpolated_center"
        )

        self.hole_entries.append(
            hole
        )

        return True

    def commit_manual_hole(self):

        ellipse = self.three_point_ellipse(
            self.points
        )

        if ellipse is None:
            print(
                "[MANUAL-HOLE] "
                "need 3 valid points: "
                "center, major edge, "
                "minor edge"
            )
            return False

        self.push_undo()

        self.draw_hole(
            ellipse
        )

        hole = copy.deepcopy(
            ellipse
        )

        hole["method"] = (
            "manual_3point"
        )

        self.hole_entries.append(
            hole
        )

        return True

    def commit(self):

        ok = False

        if self.mode == "wire":
            ok = self.commit_wire()

        elif self.mode == "tool":
            ok = self.commit_tool()

        elif self.mode == "board":
            ok = self.commit_board()

        elif self.mode == "ref_hole":
            ok = (
                self.commit_reference_hole()
            )

        elif self.mode == "hole":
            ok = (
                self.commit_auto_hole()
            )

        elif self.mode == "manual_hole":
            ok = (
                self.commit_manual_hole()
            )

        if ok:

            print(
                "[COMMIT]",
                MODE_NAME[
                    self.mode
                ],
            )

            self.points = []
            self.dirty = True

    def cancel(self):

        if self.points:

            self.points = []

            print(
                "[CANCEL] "
                "current object cleared"
            )

        else:
            print(
                "[CANCEL] "
                "no pending object"
            )

    def write_geometry(
        self,
        item,
    ):

        record = {
            "sample_id":
                item["sid"],
            "camera":
                item["camera"],
            "board_visible":
                bool(
                    np.any(
                        self.hole_mask
                        == 1
                    )
                ),
            "ellipse_refs":
                copy.deepcopy(
                    self.ellipse_refs
                ),
            "holes":
                copy.deepcopy(
                    self.hole_entries
                ),
        }

        self.geometry_records[
            (
                item["sid"],
                item["camera"],
            )
        ] = record

        rows = [
            self.geometry_records[
                key
            ]
            for key
            in sorted(
                self.geometry_records
            )
        ]

        tmp = (
            self.geometry_path
            .with_suffix(
                ".jsonl.tmp"
            )
        )

        with tmp.open(
            "w",
            encoding="utf-8",
        ) as fp:

            for row in rows:

                fp.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        os.replace(
            tmp,
            self.geometry_path,
        )

    def save(self):

        if self.points:
            print(
                "[BLOCKED] "
                "ENTER or C before saving"
            )
            return

        item = self.current()

        wp = item[
            "wire_mask"
        ]

        hp = item[
            "hole_mask"
        ]

        wp.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        hp.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        wtmp = wp.with_name(
            wp.stem
            + ".tmp.png"
        )

        htmp = hp.with_name(
            hp.stem
            + ".tmp.png"
        )

        for tmp in (
            wtmp,
            htmp,
        ):
            if tmp.exists():
                tmp.unlink()

        if not cv2.imwrite(
            str(wtmp),
            self.wire_mask,
        ):
            raise RuntimeError(
                f"Failed to write: "
                f"{wtmp}"
            )

        if not cv2.imwrite(
            str(htmp),
            self.hole_mask,
        ):
            if wtmp.exists():
                wtmp.unlink()

            raise RuntimeError(
                f"Failed to write: "
                f"{htmp}"
            )

        wcheck = cv2.imread(
            str(wtmp),
            cv2.IMREAD_UNCHANGED,
        )

        hcheck = cv2.imread(
            str(htmp),
            cv2.IMREAD_UNCHANGED,
        )

        self.validate_mask(
            wcheck,
            self.wire_mask.shape,
            wtmp,
        )

        self.validate_mask(
            hcheck,
            self.hole_mask.shape,
            htmp,
        )

        os.replace(
            wtmp,
            wp,
        )

        os.replace(
            htmp,
            hp,
        )

        self.write_geometry(
            item
        )

        record = {
            "timestamp":
                datetime.now()
                .isoformat(
                    timespec="seconds"
                ),
            "sample_id":
                item["sid"],
            "camera":
                item["camera"],
            "wire_mask":
                str(wp),
            "hole_mask":
                str(hp),
            "wire_pixels":
                int(
                    np.sum(
                        self.wire_mask
                        == 1
                    )
                ),
            "tool_pixels":
                int(
                    np.sum(
                        self.wire_mask
                        == 2
                    )
                ),
            "board_pixels":
                int(
                    np.sum(
                        self.hole_mask
                        == 1
                    )
                ),
            "hole_pixels":
                int(
                    np.sum(
                        self.hole_mask
                        == 2
                    )
                ),
            "reference_holes":
                len(
                    self.ellipse_refs
                ),
            "hole_count":
                len(
                    self.hole_entries
                ),
        }

        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as fp:

            fp.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        self.saved_wire_mask = (
            self.wire_mask.copy()
        )

        self.saved_hole_mask = (
            self.hole_mask.copy()
        )

        self.dirty = False

        print(
            "[SAVED]",
            item["sid"],
            item["camera"],
        )

    def navigation_blocked(self):

        if self.points:
            print(
                "[BLOCKED] "
                "pending object. "
                "ENTER or C first."
            )
            return True

        if self.dirty:
            print(
                "[BLOCKED] "
                "unsaved changes. "
                "Press K first."
            )
            return True

        return False

    def navigate(
        self,
        step,
    ):

        if self.navigation_blocked():
            return

        new_index = max(
            0,
            min(
                len(self.items) - 1,
                self.index + step,
            ),
        )

        if new_index == self.index:
            print(
                "[NAV] dataset boundary"
            )
            return

        self.index = new_index

        self.load_current()

    def render_overlay(self):

        # HOLE_OUTLINE_DISPLAY_ONLY_V1
        # Board and Wire/Tweezer retain their existing overlays.
        # Hole class 2 is NOT filled on screen.
        # Its contour is displayed as a thin outline only.

        out = self.image.copy()

        overlay = out.copy()

        overlay[
            self.hole_mask == 1
        ] = BOARD_COLOR

        overlay[
            self.wire_mask == 2
        ] = TOOL_COLOR

        overlay[
            self.wire_mask == 1
        ] = WIRE_COLOR

        # Exclude Hole class 2 from filled overlay.
        mask_any = (
            (self.wire_mask != 0)
            |
            (self.hole_mask == 1)
        )

        blended = cv2.addWeighted(
            out,
            0.55,
            overlay,
            0.45,
            0,
        )

        out[
            mask_any
        ] = blended[
            mask_any
        ]

        # Draw only the boundary of the actual Hole GT mask.
        hole_binary = (
            (self.hole_mask == 2)
            .astype("uint8")
            * 255
        )

        contours, _ = cv2.findContours(
            hole_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if contours:

            cv2.drawContours(
                out,
                contours,
                -1,
                HOLE_COLOR,
                1,
                cv2.LINE_AA,
            )

        return out


    def render_pending_ellipse(
        self,
        full,
    ):

        ellipse = None

        if (
            self.mode == "hole"
            and len(self.points) == 1
            and len(
                self.ellipse_refs
            ) >= 3
        ):

            ellipse = (
                self.predict_ellipse(
                    self.points[0]
                )
            )

        elif (
            self.mode
            in (
                "ref_hole",
                "manual_hole",
            )
            and len(self.points) == 3
        ):

            ellipse = (
                self.three_point_ellipse(
                    self.points
                )
            )

        if ellipse is None:
            return

        c = tuple(
            int(round(v))
            for v
            in ellipse[
                "center_uv"
            ]
        )

        a = max(
            2,
            int(round(
                ellipse[
                    "axes_px"
                ][0]
            )),
        )

        b = max(
            2,
            int(round(
                ellipse[
                    "axes_px"
                ][1]
            )),
        )

        cv2.ellipse(
            full,
            c,
            (a, b),
            float(
                ellipse[
                    "angle_deg"
                ]
            ),
            0,
            360,
            PENDING_COLOR,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    def render(self):

        item = self.current()

        full = self.render_overlay()

        self.render_pending_ellipse(
            full
        )

        (
            x0,
            y0,
            cw,
            ch,
            dw,
            dh,
        ) = self.geometry()

        crop = full[
            y0:y0 + ch,
            x0:x0 + cw,
        ]

        image = cv2.resize(
            crop,
            (
                dw,
                dh,
            ),
            interpolation=(
                cv2.INTER_NEAREST
                if self.zoom > 1.0
                else cv2.INTER_AREA
            ),
        )

        canvas = np.zeros(
            (
                HEADER_H + dh,
                dw,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            HEADER_H:
            HEADER_H + dh,
            :dw,
        ] = image

        state = (
            "DIRTY"
            if self.dirty
            else "SAVED/CLEAN"
        )

        line1 = (
            f"{self.index + 1}/"
            f"{len(self.items)}  "
            f"{item['sid']}  "
            f"{item['camera']}  "
            f"{MODE_NAME[self.mode]}  "
            f"{state}  "
            f"ZOOM={self.zoom:.2f}x  "
            f"REF={len(self.ellipse_refs)} "
            f"HOLES={len(self.hole_entries)}"
        )

        line2 = (
            "L Wire | T Tweezer | "
            "B Board | R RefHole | "
            "H AutoHole | E ManualHole"
        )

        line3 = (
            "ENTER confirm | C cancel | "
            "U undo | K save | "
            "N/J image | Z/X zoom | "
            "WASD pan | Q quit"
        )

        cv2.putText(
            canvas,
            line1,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            line2,
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            line3,
            (12, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        hint = ""

        if self.mode == "ref_hole":

            hint = (
                "R: CENTER -> MAJOR edge -> "
                "MINOR edge -> ENTER "
                "(make >=3 refs)"
            )

        elif self.mode == "hole":

            hint = (
                "H: click CENTER only -> "
                "inspect white ellipse -> ENTER"
            )

        elif (
            self.mode
            == "manual_hole"
        ):

            hint = (
                "E: CENTER -> MAJOR edge -> "
                "MINOR edge -> ENTER"
            )

        if hint:

            cv2.putText(
                canvas,
                hint,
                (12, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                HOLE_COLOR,
                1,
                cv2.LINE_AA,
            )

        if self.points:

            sp = []

            for p in self.points:

                sx = int(round(
                    (
                        p[0] - x0
                    )
                    * dw
                    / max(
                        1,
                        cw - 1,
                    )
                ))

                sy = (
                    HEADER_H
                    +
                    int(round(
                        (
                            p[1] - y0
                        )
                        * dh
                        / max(
                            1,
                            ch - 1,
                        )
                    ))
                )

                sp.append(
                    (
                        sx,
                        sy,
                    )
                )

                if (
                    0 <= sx < dw
                    and
                    HEADER_H
                    <= sy
                    < HEADER_H + dh
                ):

                    cv2.circle(
                        canvas,
                        (
                            sx,
                            sy,
                        ),
                        4,
                        PENDING_COLOR,
                        -1,
                        cv2.LINE_AA,
                    )

            if len(sp) >= 2:

                for a, b in zip(
                    sp[:-1],
                    sp[1:],
                ):

                    cv2.line(
                        canvas,
                        a,
                        b,
                        PENDING_COLOR,
                        1,
                        cv2.LINE_AA,
                    )

            if (
                self.mode
                in (
                    "tool",
                    "board",
                )
                and len(sp) >= 3
            ):

                cv2.line(
                    canvas,
                    sp[-1],
                    sp[0],
                    (
                        180,
                        180,
                        180,
                    ),
                    1,
                    cv2.LINE_AA,
                )

        return canvas

    def run(self):

        cv2.namedWindow(
            self.window,
            cv2.WINDOW_AUTOSIZE,
        )

        cv2.setMouseCallback(
            self.window,
            self.mouse,
        )

        print()
        print(
            "========================================"
        )
        print(
            " R1-A7 JOINT INSERTION ANNOTATOR V2"
        )
        print(
            "========================================"
        )

        print(
            "L = Wire / Line"
        )

        print(
            "T = Tweezer / Tool"
        )

        print(
            "B = Board"
        )

        print(
            "R = Reference Hole "
            "(3 points)"
        )

        print(
            "H = Auto Hole "
            "(center only, requires >=3 R refs)"
        )

        print(
            "E = Manual Hole "
            "(3 points)"
        )

        print(
            "ENTER = confirm current object"
        )

        print(
            "C / ESC = cancel current object"
        )

        print(
            "U = undo"
        )

        print(
            "K = save BOTH masks "
            "+ hole geometry"
        )

        print(
            "N/J = next / previous image"
        )

        print(
            "Z/X = zoom in / out"
        )

        print(
            "W/A/S/D = "
            "pan up / left / down / right"
        )

        print(
            "Q = quit"
        )

        print()

        while True:

            cv2.imshow(
                self.window,
                self.render(),
            )

            key = cv2.waitKeyEx(
                20
            )

            if key < 0:
                continue

            low = key & 0xFF

            if low == ord("l"):

                self.set_mode(
                    "wire"
                )

            elif low == ord("t"):

                self.set_mode(
                    "tool"
                )

            elif low == ord("b"):

                self.set_mode(
                    "board"
                )

            elif low == ord("r"):

                self.set_mode(
                    "ref_hole"
                )

            elif low == ord("h"):

                self.set_mode(
                    "hole"
                )

            elif low == ord("e"):

                self.set_mode(
                    "manual_hole"
                )

            elif key in (
                10,
                13,
            ):

                self.commit()

            elif (
                low == ord("c")
                or key == 27
            ):

                self.cancel()

            elif low == ord("u"):

                self.undo()

            elif low == ord("k"):

                self.save()

            elif low == ord("n"):

                self.navigate(
                    +1
                )

            elif low == ord("j"):

                self.navigate(
                    -1
                )

            elif low == ord("z"):

                self.zoom_view(
                    1.25
                )

            elif low == ord("x"):

                self.zoom_view(
                    1.0 / 1.25
                )

            elif low == ord("w"):

                self.pan_view(
                    0,
                    -1,
                )

            elif low == ord("s"):

                self.pan_view(
                    0,
                    +1,
                )

            elif low == ord("a"):

                self.pan_view(
                    -1,
                    0,
                )

            elif low == ord("d"):

                self.pan_view(
                    +1,
                    0,
                )

            elif low == ord("q"):

                if (
                    self.navigation_blocked()
                ):

                    print(
                        "[BLOCKED] "
                        "Save with K first, "
                        "or cancel/undo changes."
                    )

                else:
                    break

        cv2.destroyAllWindows()


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
        "--start-sample",
        default=None,
    )

    parser.add_argument(
        "--wire-width",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    JointAnnotator(
        dataset=args.dataset,
        session=args.session,
        start_sample=args.start_sample,
        wire_width=args.wire_width,
    ).run()


if __name__ == "__main__":
    main()
