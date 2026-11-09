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

HEADER = 115
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
    ):

        self.root = Path(dataset)
        self.session = Path(session)

        ids = []

        for camera in CAMERAS:
            folder = self.session / camera

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

        common = common[:limit_groups]

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
                    "mask": (
                        self.root
                        / "masks_v2"
                        / camera
                        / f"{sid}.png"
                    ),
                })

        self.geometry_path = (
            self.root
            / "annotations_v2"
            / "hole_geometry.jsonl"
        )

        self.log_path = (
            self.root
            / "annotations_v2"
            / "annotation_log.jsonl"
        )

        self.geometry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.records = {}

        if self.geometry_path.exists():
            for line in self.geometry_path.read_text(
                encoding="utf-8"
            ).splitlines():

                if line.strip():
                    r = json.loads(line)

                    self.records[
                        (r["sample_id"], r["camera"])
                    ] = r

        self.index = 0

        self.mode = "idle"
        self.points = []
        self.pending = None

        self.undo_stack = []

        self.image = None
        self.mask = None
        self.saved_mask = None

        self.holes = []
        self.board_visible = False
        self.current_hole = None

        self.dirty = False
        self.overlay = True

        self.zoom = 1.0
        self.cx = 0
        self.cy = 0

        self.roi_radius = 24
        self.gray_tolerance = 28

        self.brush_label = 2
        self.brush_radius = 3
        self.drawing = False
        self.last_point = None

        self.window = "R1-A7 HOLE BOARD ANNOTATOR V2"

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
                or not set(
                    int(v) for v in np.unique(mask)
                ).issubset({0, 1, 2, 255})
            ):
                raise RuntimeError(
                    f"Invalid mask: {item['mask']}"
                )

            self.mask = mask.copy()

        else:
            self.mask = np.zeros(
                (h, w),
                dtype=np.uint8,
            )

        self.saved_mask = self.mask.copy()

        record = self.records.get(
            (
                item["sid"],
                item["camera"],
            ),
            {},
        )

        self.holes = copy.deepcopy(
            record.get("holes", [])
        )

        self.board_visible = bool(
            record.get(
                "board_visible",
                np.any(self.mask == 1),
            )
        )

        self.current_hole = (
            len(self.holes) - 1
            if self.holes
            else None
        )

        self.mode = "idle"
        self.points = []
        self.pending = None
        self.undo_stack = []

        self.dirty = False
        self.drawing = False

        self.zoom = 1.0
        self.cx = w / 2
        self.cy = h / 2

        print()
        print("=" * 65)
        print(
            "SAMPLE:",
            item["sid"],
            "CAMERA:",
            item["camera"],
        )
        print("EXISTING HOLES:", len(self.holes))
        print("=" * 65)

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

        dh = int(
            round(DISPLAY_W * ch / cw)
        )

        return x0, y0, cw, ch, dh

    def screen_to_image(self, sx, sy):

        x0, y0, cw, ch, dh = self.geometry()

        sy -= HEADER

        if (
            sx < 0
            or sx >= DISPLAY_W
            or sy < 0
            or sy >= dh
        ):
            return None

        u = x0 + sx * (cw - 1) / max(
            DISPLAY_W - 1, 1
        )

        v = y0 + sy * (ch - 1) / max(
            dh - 1, 1
        )

        return (
            int(round(u)),
            int(round(v)),
        )

    def image_to_screen(self, u, v):

        x0, y0, cw, ch, dh = self.geometry()

        sx = int(
            (u - x0)
            * DISPLAY_W
            / max(cw, 1)
        )

        sy = HEADER + int(
            (v - y0)
            * dh
            / max(ch, 1)
        )

        return sx, sy

    def push_undo(self):

        self.undo_stack.append({
            "mask": self.mask.copy(),
            "holes": copy.deepcopy(self.holes),
            "board": self.board_visible,
        })

        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    def undo(self):

        if not self.undo_stack:
            print("[UNDO] empty")
            return

        s = self.undo_stack.pop()

        self.mask = s["mask"]
        self.holes = s["holes"]
        self.board_visible = s["board"]

        self.current_hole = (
            len(self.holes) - 1
            if self.holes
            else None
        )

        self.dirty = True

        print("[UNDO] restored")

    # ========================================================
    # CLICK-BASED LOCAL HOLE SEGMENTATION
    # ========================================================

    def auto_segment(self, point):

        u, v = point

        # Board + already accepted holes are valid search area.
        board = (
            (self.mask == 1)
            |
            (self.mask == 2)
        )

        if not board[v, u]:

            print(
                "[AUTO] Click inside annotated board area."
            )

            return None

        r = self.roi_radius

        x0 = max(
            0,
            u - r,
        )

        x1 = min(
            self.image.shape[1],
            u + r + 1,
        )

        y0 = max(
            0,
            v - r,
        )

        y1 = min(
            self.image.shape[0],
            v + r + 1,
        )

        gray = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY,
        )

        roi = gray[
            y0:y1,
            x0:x1,
        ]

        board_roi = board[
            y0:y1,
            x0:x1,
        ]

        sx = u - x0
        sy = v - y0

        # ----------------------------------------------------
        # Seed intensity
        # ----------------------------------------------------

        patch = gray[
            max(0, v - 2):
            min(gray.shape[0], v + 3),

            max(0, u - 2):
            min(gray.shape[1], u + 3),
        ]

        seed_gray = float(
            np.median(
                patch
            )
        )

        # Small blur removes isolated sensor / reflection noise.
        blur = cv2.GaussianBlur(
            roi,
            (3, 3),
            0,
        )

        # ----------------------------------------------------
        # Click-seeded local dark-region extraction.
        #
        # Do NOT union with local Otsu here:
        # on this metal hole board, shadows and dark board
        # texture can cause Otsu to select almost the whole ROI.
        #
        # We only use the intensity around the clicked hole.
        # Missing highlight pixels are repaired later by
        # morphology + convex-hull completion.
        # ----------------------------------------------------

        seed_threshold = min(
            255,
            int(
                seed_gray
                +
                self.gray_tolerance
            ),
        )

        candidate = (
            (
                blur
                <=
                seed_threshold
            )
            &
            board_roi
        ).astype(
            np.uint8
        )

        # Make sure clicked seed belongs to candidate.
        candidate[
            sy,
            sx
        ] = 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        # Fill small gaps caused by highlights inside the hole.
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        # ----------------------------------------------------
        # Keep only connected component containing click point.
        # ----------------------------------------------------

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                candidate,
                connectivity=8,
            )
        )

        label = int(
            labels[
                sy,
                sx
            ]
        )

        if label == 0:

            print(
                "[AUTO] No connected hole region at click."
            )

            return None

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        left = int(
            stats[
                label,
                cv2.CC_STAT_LEFT,
            ]
        )

        top = int(
            stats[
                label,
                cv2.CC_STAT_TOP,
            ]
        )

        width = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH,
            ]
        )

        height = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        roi_h, roi_w = candidate.shape

        roi_area = int(
            roi_h
            *
            roi_w
        )

        if area < 6:

            print(
                "[AUTO REJECT] region too small"
            )

            return None

        # Prevent huge board-shadow regions from becoming holes.
        if area > roi_area * 0.30:

            print(
                "[AUTO REJECT] region too large:",
                "area=",
                area,
                "roi_area=",
                roi_area,
            )

            print(
                "Try [ to reduce tolerance "
                "or - to reduce ROI."
            )

            return None

        max_allowed_dim = int(
            max(
                12,
                self.roi_radius * 1.6,
            )
        )

        if (
            width > max_allowed_dim
            or
            height > max_allowed_dim
        ):

            print(
                "[AUTO REJECT] component too large:",
                "width=",
                width,
                "height=",
                height,
            )

            return None

        # A real local hole should normally close inside ROI.
        touches_border = (
            left <= 0
            or
            top <= 0
            or
            left + width >= roi_w
            or
            top + height >= roi_h
        )

        if touches_border:

            print(
                "[AUTO REJECT] component touches ROI border"
            )

            print(
                "Try smaller ROI (-) or lower tolerance ([)."
            )

            return None

        local = (
            labels
            ==
            label
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Complete partially extracted aperture.
        #
        # Small holes are approximately convex under camera
        # perspective. Convex hull repairs missing pieces caused
        # by highlight / texture inside the aperture without
        # forcing every hole to be a perfect circle.
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            local,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:

            print(
                "[AUTO] no contour"
            )

            return None

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        if len(contour) < 3:

            print(
                "[AUTO] contour too small"
            )

            return None

        hull = cv2.convexHull(
            contour
        )

        completed = np.zeros_like(
            local
        )

        cv2.drawContours(
            completed,
            [hull],
            -1,
            1,
            thickness=-1,
        )

        # One final gentle close only.
        completed = cv2.morphologyEx(
            completed,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        completed_area = int(
            np.count_nonzero(
                completed
            )
        )

        # Safety check again after completion.
        if completed_area > roi_area * 0.32:

            print(
                "[AUTO REJECT] completed region too large:",
                completed_area,
            )

            return None

        full = np.zeros(
            self.mask.shape,
            dtype=np.uint8,
        )

        full[
            y0:y1,
            x0:x1
        ] = completed

        contours2, _ = cv2.findContours(
            completed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours2:
            return None

        contour2 = max(
            contours2,
            key=cv2.contourArea,
        )

        approx = cv2.approxPolyDP(
            contour2,
            1.0,
            True,
        )

        contour_uv = [
            [
                int(
                    p[0][0]
                    +
                    x0
                ),
                int(
                    p[0][1]
                    +
                    y0
                ),
            ]
            for p in approx
        ]

        # Use completed mask centroid as automatic hole center.
        moments = cv2.moments(
            completed,
            binaryImage=True,
        )

        if moments["m00"] > 0:

            center_u = (
                moments["m10"]
                /
                moments["m00"]
                +
                x0
            )

            center_v = (
                moments["m01"]
                /
                moments["m00"]
                +
                y0
            )

        else:

            center_u = float(u)
            center_v = float(v)

        print(
            "[AUTO CANDIDATE]",
            "raw_area=",
            area,
            "completed_area=",
            completed_area,
            "seed_gray=",
            round(seed_gray, 1),
            "seed_threshold=",
            seed_threshold,
            "center=",
            (
                round(center_u, 1),
                round(center_v, 1),
            ),
        )

        return {
            "mask":
                full.astype(bool),

            "center_uv": [
                float(center_u),
                float(center_v),
            ],

            "clicked_uv": [
                float(u),
                float(v),
            ],

            "method":
                "click_auto_completed",

            "area_px":
                completed_area,

            "contour_uv":
                contour_uv,
        }


    # ========================================================
    # ELLIPSE FALLBACK
    # ========================================================

    def create_ellipse(self):

        if len(self.points) != 3:
            return

        center = self.points[0]
        major_point = self.points[1]
        minor_point = self.points[2]

        dx = major_point[0] - center[0]
        dy = major_point[1] - center[1]

        major = int(
            round(np.hypot(dx, dy))
        )

        minor = int(
            round(
                np.hypot(
                    minor_point[0] - center[0],
                    minor_point[1] - center[1],
                )
            )
        )

        if major < 2 or minor < 2:
            print("[ELLIPSE] Axis too small.")
            self.points = []
            return

        angle = float(
            np.degrees(
                np.arctan2(dy, dx)
            )
        )

        candidate = np.zeros(
            self.mask.shape,
            dtype=np.uint8,
        )

        cv2.ellipse(
            candidate,
            center,
            (major, minor),
            angle,
            0,
            360,
            1,
            -1,
        )

        board = (
            (self.mask == 1)
            |
            (self.mask == 2)
        )

        candidate = (
            (candidate > 0)
            &
            board
        )

        if np.count_nonzero(candidate) < 4:
            print("[ELLIPSE] No valid board overlap.")
            self.points = []
            return

        self.pending = {
            "mask": candidate,
            "center_uv": [
                float(center[0]),
                float(center[1]),
            ],
            "method": "three_point_ellipse",
            "area_px": int(
                np.count_nonzero(candidate)
            ),
            "contour_uv": [],
        }

        self.mode = "candidate"
        self.points = []

        print(
            "[ELLIPSE] Candidate ready. ENTER to accept."
        )

    # ========================================================
    # ACCEPT / REJECT
    # ========================================================

    def accept_candidate(self):

        if self.pending is None:
            return

        self.push_undo()

        self.mask[
            self.pending["mask"]
        ] = 2

        hole = {
            "center_uv":
                self.pending["center_uv"],

            "clicked_uv":
                self.pending.get(
                    "clicked_uv"
                ),

            "center_valid":
                True,

            "visible":
                True,

            "target":
                False,

            "method":
                self.pending["method"],

            "area_px":
                self.pending["area_px"],

            "contour_uv":
                self.pending[
                    "contour_uv"
                ],
        }

        self.holes.append(
            hole
        )

        self.current_hole = (
            len(self.holes)
            -
            1
        )

        hole_number = len(
            self.holes
        )

        self.pending = None

        # Stay in ellipse mode after every accepted hole.
        # This allows continuous annotation without pressing H again.
        self.mode = "ellipse"
        self.points = []

        self.dirty = True

        print(
            "[HOLE ACCEPTED]",
            "number=",
            hole_number,
            "center=",
            hole["center_uv"],
            "area=",
            hole["area_px"],
        )

        print(
            "[CONTINUOUS ELLIPSE] "
            "Next hole: click center -> major axis -> minor axis"
        )


    def reject_candidate(self):

        self.pending = None
        self.mode = "idle"

        print("[CANDIDATE] rejected")

    # ========================================================
    # MOUSE
    # ========================================================

    def paint(self, p0, p1):

        cv2.line(
            self.mask,
            p0,
            p1,
            int(self.brush_label),
            thickness=max(
                1,
                self.brush_radius * 2,
            ),
        )

        cv2.circle(
            self.mask,
            p1,
            self.brush_radius,
            int(self.brush_label),
            -1,
        )

        self.dirty = True

    def mouse(self, event, x, y, flags, userdata):

        point = self.screen_to_image(x, y)

        if event == cv2.EVENT_RBUTTONDOWN:

            if self.pending is not None:
                self.reject_candidate()

            elif self.points:
                self.points.pop()
                print("[POINT] removed")

            return

        if self.mode == "brush":

            if event == cv2.EVENT_LBUTTONDOWN:

                if point is None:
                    return

                self.push_undo()

                self.drawing = True
                self.last_point = point

                self.paint(
                    point,
                    point,
                )

            elif (
                event == cv2.EVENT_MOUSEMOVE
                and self.drawing
                and point is not None
            ):

                self.paint(
                    self.last_point,
                    point,
                )

                self.last_point = point

            elif event == cv2.EVENT_LBUTTONUP:

                self.drawing = False
                self.last_point = None

            return

        if (
            event != cv2.EVENT_LBUTTONDOWN
            or point is None
        ):
            return

        if self.mode == "board":
            self.points.append(point)
            print("[BOARD POINT]", point)

        elif self.mode == "ignore":
            self.points.append(point)

        elif self.mode == "background":
            self.points.append(point)

        elif self.mode == "auto":

            if not self.board_visible:
                print(
                    "[AUTO] Annotate board first."
                )
                return

            candidate = self.auto_segment(point)

            if candidate is not None:
                self.pending = candidate
                self.mode = "candidate"

                print(
                    "[AUTO] Preview ready. "
                    "ENTER accept / ESC reject."
                )

        elif self.mode == "center":

            if self.current_hole is None:
                return

            self.push_undo()

            self.holes[
                self.current_hole
            ]["center_uv"] = [
                float(point[0]),
                float(point[1]),
            ]

            self.holes[
                self.current_hole
            ]["center_valid"] = True

            self.mode = "idle"
            self.dirty = True

            print(
                "[CENTER UPDATED]",
                point,
            )

        elif self.mode == "ellipse":

            self.points.append(point)

            print(
                "[ELLIPSE POINT]",
                len(self.points),
                point,
            )

            if len(self.points) == 3:
                self.create_ellipse()

    # ========================================================
    # POLYGON
    # ========================================================

    def commit_polygon(self):

        if len(self.points) < 3:
            print("[POLYGON] Need 3 points.")
            return

        self.push_undo()

        polygon = np.asarray(
            self.points,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        fill = np.zeros(
            self.mask.shape,
            dtype=np.uint8,
        )

        cv2.fillPoly(
            fill,
            [polygon],
            1,
        )

        label = {
            "board": 1,
            "ignore": 255,
            "background": 0,
        }[self.mode]

        if label == 1:

            # Do not overwrite already annotated holes.
            selected = (
                (fill > 0)
                &
                (self.mask != 2)
            )

            self.mask[selected] = 1
            self.board_visible = True

        else:

            self.mask[fill > 0] = label

        print(
            "[POLYGON COMMITTED]",
            self.mode,
            "points=",
            len(self.points),
        )

        self.points = []
        self.mode = "idle"
        self.dirty = True

    # ========================================================
    # HOLE CENTER / TARGET
    # ========================================================

    def mark_target(self):

        if self.current_hole is None:

            print(
                "[TARGET] no hole selected"
            )

            return

        self.push_undo()

        for hole in self.holes:

            hole["target"] = False

        self.holes[
            self.current_hole
        ]["target"] = True

        self.dirty = True

        print(
            "[TARGET] hole number =",
            self.current_hole + 1,
        )


    # ========================================================
    # SAVE
    # ========================================================

    def save(self):

        if (
            self.pending is not None
            or self.points
        ):
            print(
                "[BLOCKED] Accept or cancel pending operation first."
            )
            return

        item = self.current()

        path = item["mask"]
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp_mask = Path(
            str(path) + ".tmp.png"
        )

        if not cv2.imwrite(
            str(tmp_mask),
            self.mask,
        ):
            raise RuntimeError(
                "Mask save failed"
            )

        tmp_mask.replace(path)

        record = {
            "sample_id": item["sid"],
            "camera": item["camera"],
            "board_visible": self.board_visible,
            "holes": self.holes,
            "annotation_status": "annotated",
            "updated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        }

        self.records[
            (
                item["sid"],
                item["camera"],
            )
        ] = record

        ordered = sorted(
            self.records.values(),
            key=lambda r: (
                r["sample_id"],
                CAMERAS.index(r["camera"]),
            ),
        )

        tmp_json = Path(
            str(self.geometry_path) + ".tmp"
        )

        with tmp_json.open(
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

        tmp_json.replace(
            self.geometry_path
        )

        log = {
            "sample_id": item["sid"],
            "camera": item["camera"],
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

        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    log,
                    ensure_ascii=False,
                )
                + "\n"
            )

        preview_dir = (
            self.root
            / "mask_previews_v2"
        )

        preview_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        preview_path = (
            preview_dir
            /
            f"{item['sid']}_{item['camera']}.jpg"
        )

        cv2.imwrite(
            str(preview_path),
            self.render(),
        )

        self.saved_mask = self.mask.copy()
        self.dirty = False

        print()
        print(
            "[SAVED]",
            item["sid"],
            item["camera"],
        )

        print(
            "HOLES:",
            len(self.holes),
        )

        print(
            "MASK:",
            path,
        )

    # ========================================================
    # DISPLAY
    # ========================================================

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
            (DISPLAY_W, dh),
        )

        mask = cv2.resize(
            mask,
            (DISPLAY_W, dh),
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

        if self.pending is not None:

            candidate = self.pending["mask"][
                y0:y0+ch,
                x0:x0+cw,
            ]

            candidate = cv2.resize(
                candidate.astype(np.uint8),
                (DISPLAY_W, dh),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            shown[candidate] = (
                0.5 * shown[candidate]
                +
                0.5 * np.array(
                    [255, 255, 0],
                    dtype=np.float32,
                )
            ).astype(np.uint8)

        # --------------------------------------------------------
        # Polygon preview drawn directly on the displayed image.
        # This avoids HEADER / canvas coordinate mismatch.
        # --------------------------------------------------------
        if (
            self.mode in (
                "board",
                "ignore",
                "background",
            )
            and
            len(self.points) >= 1
        ):

            preview_points = []

            for pu, pv in self.points:

                sx = int(
                    (pu - x0)
                    * DISPLAY_W
                    / max(cw, 1)
                )

                sy = int(
                    (pv - y0)
                    * dh
                    / max(ch, 1)
                )

                preview_points.append(
                    (sx, sy)
                )

            # Draw vertices.
            for sx, sy in preview_points:

                cv2.circle(
                    shown,
                    (sx, sy),
                    3,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )

            # Draw open polygon edges.
            if len(preview_points) >= 2:

                pts = np.asarray(
                    preview_points,
                    dtype=np.int32,
                ).reshape(
                    (-1, 1, 2)
                )

                cv2.polylines(
                    shown,
                    [pts],
                    False,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Thin closing-line preview.
            if len(preview_points) >= 3:

                cv2.line(
                    shown,
                    preview_points[-1],
                    preview_points[0],
                    (0, 180, 255),
                    1,
                    cv2.LINE_AA,
                )

        canvas = np.zeros(
            (
                HEADER + dh,
                DISPLAY_W,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[HEADER:] = shown

        for hole in self.holes:

            if hole.get("center_valid"):

                u, v = hole["center_uv"]

                sx, sy = self.image_to_screen(
                    u,
                    v,
                )

                cv2.drawMarker(
                    canvas,
                    (sx, sy),
                    (255, 255, 0),
                    cv2.MARKER_CROSS,
                    15,
                    2,
                )

        for point in self.points:

            sx, sy = self.image_to_screen(
                *point
            )

            cv2.circle(
                canvas,
                (sx, sy),
                4,
                (0, 255, 255),
                -1,
            )

        item = self.current()

        status = (
            "UNSAVED"
            if self.dirty
            else "SAVED"
        )

        lines = [
            (
                f"{item['sid']} "
                f"{item['camera'].upper()} "
                f"{self.index+1}/{len(self.items)} "
                f"MODE={self.mode.upper()} "
                f"HOLES={len(self.holes)} "
                f"{status}"
            ),
            "B Board polygon | H Ellipse hole | A Click-auto fallback | ENTER Accept/Commit",
            "P Center correction | T Target | C Brush | 1/2/0/9 Brush label",
            "I Ignore polygon | U Undo | K Save | N/J Next/Previous | V Overlay",
            (
                f"Z/X Zoom | W/A/S/D Pan | "
                f"[ ] Gray tolerance={self.gray_tolerance} | "
                f"-/= ROI={self.roi_radius} | Q Quit"
            ),
        ]

        for i, text in enumerate(lines):

            cv2.putText(
                canvas,
                text,
                (8, 21 + i * 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas

    # ========================================================
    # NAVIGATION
    # ========================================================

    def navigate(self, step):

        if (
            self.dirty
            or self.pending is not None
            or self.points
        ):
            print(
                "[BLOCKED] Save or cancel before navigation."
            )
            return

        ni = self.index + step

        if not 0 <= ni < len(self.items):
            print("[END]")
            return

        self.index = ni
        self.load_current()

    def pan(self, dx, dy):

        _, _, cw, ch, _ = self.geometry()

        self.cx += dx * cw * 0.12
        self.cy += dy * ch * 0.12

    # ========================================================
    # MAIN LOOP
    # ========================================================

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
        print("R1-A7 HOLE BOARD ANNOTATOR V2")
        print("=" * 65)
        print("B: Board polygon")
        print("H: Three-point ellipse hole (PRIMARY)")
        print("A: Click-auto hole segmentation (FALLBACK)")
        print("ENTER: Accept candidate / commit polygon")
        print("P: Correct latest hole center")
        print("T: Set target hole")
        print("C: Brush mode")
        print("1/2/0/9: Brush class 1/2/0/255")
        print("I: Ignore polygon")
        print("U: Undo")
        print("K: Save")
        print("N/J: Next/Previous")
        print("V: Overlay")
        print("Z/X: Zoom")
        print("[ / ]: Change gray tolerance")
        print("- / =: Change ROI radius")
        print("Q: Quit")
        print("=" * 65)

        while True:

            cv2.imshow(
                self.window,
                self.render(),
            )

            key = cv2.waitKeyEx(20)

            if key < 0:
                continue

            k = key & 0xFF

            if k in (13, 10):

                if self.pending is not None:
                    self.accept_candidate()

                elif self.mode in (
                    "board",
                    "ignore",
                    "background",
                ):
                    self.commit_polygon()

                continue

            ch = chr(k).lower()

            if ch == "b":

                self.mode = "board"
                self.points = []
                self.pending = None
                print("[MODE] BOARD")

            elif ch == "h":

                if not self.board_visible:
                    print(
                        "[WARNING] Mark board first."
                    )
                    continue

                self.mode = "ellipse"
                self.points = []

                print(
                    "[MODE] HOLE ELLIPSE"
                )

                print(
                    "Click 1: hole center"
                )

                print(
                    "Click 2: major-axis boundary"
                )

                print(
                    "Click 3: minor-axis boundary"
                )

            elif ch == "a":

                if not self.board_visible:
                    print(
                        "[WARNING] Mark board first."
                    )
                    continue

                self.mode = "auto"
                self.points = []

                print(
                    "[MODE] CLICK-AUTO HOLE (FALLBACK)"
                )

                print(
                    "Click inside the hole."
                )

            elif ch == "i":

                self.mode = "ignore"
                self.points = []
                print("[MODE] IGNORE")

            elif ch == "p":

                if self.current_hole is None:
                    print("[CENTER] no hole")
                else:
                    self.mode = "center"
                    print(
                        "[CENTER] click corrected center"
                    )

            elif ch == "t":
                self.mark_target()

            elif ch == "c":

                self.mode = "brush"
                self.pending = None
                print(
                    "[BRUSH] current label=",
                    self.brush_label,
                )

            elif ch in ("1", "2", "0", "9"):

                self.brush_label = {
                    "1": 1,
                    "2": 2,
                    "0": 0,
                    "9": 255,
                }[ch]

                print(
                    "[BRUSH LABEL]",
                    self.brush_label,
                )

            elif ch == "u":
                self.undo()

            elif ch == "k":
                self.save()

            elif ch == "v":
                self.overlay = not self.overlay

            elif ch == "n":
                self.navigate(1)

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

            elif ch == "[":

                self.gray_tolerance = max(
                    2,
                    self.gray_tolerance - 2,
                )

                print(
                    "[TOLERANCE]",
                    self.gray_tolerance,
                )

                if (
                    self.pending is not None
                    and self.pending["method"] == "click_auto"
                ):

                    point = tuple(
                        int(v)
                        for v in self.pending["center_uv"]
                    )

                    candidate = self.auto_segment(point)

                    if candidate is not None:
                        self.pending = candidate

            elif ch == "]":

                self.gray_tolerance = min(
                    80,
                    self.gray_tolerance + 2,
                )

                print(
                    "[TOLERANCE]",
                    self.gray_tolerance,
                )

                if (
                    self.pending is not None
                    and self.pending["method"] == "click_auto"
                ):

                    point = tuple(
                        int(v)
                        for v in self.pending["center_uv"]
                    )

                    candidate = self.auto_segment(point)

                    if candidate is not None:
                        self.pending = candidate

            elif ch == "-":

                self.roi_radius = max(
                    12,
                    self.roi_radius - 4,
                )

                print(
                    "[ROI]",
                    self.roi_radius,
                )

            elif ch == "=":

                self.roi_radius = min(
                    100,
                    self.roi_radius + 4,
                )

                print(
                    "[ROI]",
                    self.roi_radius,
                )

            elif ch == "q":

                if (
                    self.dirty
                    or self.pending is not None
                    or self.points
                ):
                    print(
                        "[BLOCKED] Save or cancel pending changes first."
                    )
                else:
                    break

            elif k == 27:

                self.points = []
                self.pending = None
                self.mode = "idle"
                print("[CANCEL]")

        cv2.destroyAllWindows()

        print()
        print("ANNOTATION FINISHED")
        print(
            "Masks:",
            self.root / "masks_v2",
        )
        print(
            "Geometry:",
            self.geometry_path,
        )


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

    args = parser.parse_args()

    app = HoleBoardAnnotator(
        args.dataset,
        args.session,
        args.limit_groups,
    )

    app.run()


if __name__ == "__main__":
    main()
