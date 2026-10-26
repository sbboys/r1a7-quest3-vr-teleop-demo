#!/usr/bin/env python3
# Left wrist 007Z hole-mouth tracker.
# Manual geometric initialization + local ellipse fitting.
# Image-space perception only; no robot control.

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


class LeftWristHoleTracker:
    def __init__(self, calibration_path, camera_serial):
        self.calibration_path = Path(calibration_path)
        self.camera_serial = str(camera_serial)

        self.calibration = None
        self.reference_center = None
        self.filtered_center = None
        self.reference_axes = None
        self.current_axes = None
        self.reference_angle = 0.0
        self.current_angle = 0.0
        self.axis_point = None
        self.roi = None

        self.state = "NEED_CALIBRATION"
        self.last_result = self._empty_result()

        print("[HOLE] Startup calibration required; previous calibration will not be loaded.")

    def _empty_result(self):
        return {
            "hole_calibrated": False,
            "hole_valid": False,
            "hole_state": "NEED_CALIBRATION",
            "hole_source": None,
            "hole_center_uv": None,
            "hole_ellipse": None,
            "hole_edge_points_uv": [],
            "hole_axis_uv": None,
            "hole_axis_direction_uv": None,
            "hole_roi_xywh": None,
        }

    def _load_calibration(self):
        if not self.calibration_path.is_file():
            print(
                "[HOLE] No calibration yet. "
                "Press H in the live window to initialize."
            )
            return

        try:
            data = json.loads(
                self.calibration_path.read_text(encoding="utf-8")
            )

            if data.get("camera_serial") != self.camera_serial:
                print("[HOLE] Calibration camera serial mismatch; ignored.")
                return

            center = np.asarray(
                data["hole_center_uv"], dtype=np.float64
            )
            axes = np.asarray(
                data["ellipse_diameters_px"], dtype=np.float64
            )
            axis_point = np.asarray(
                data["axis_point_uv"], dtype=np.float64
            )
            roi = tuple(int(v) for v in data["roi_xywh"])
            angle = float(data["ellipse_angle_deg"])

            if (
                center.shape != (2,)
                or axes.shape != (2,)
                or axis_point.shape != (2,)
                or not np.isfinite(center).all()
                or not np.isfinite(axes).all()
                or not np.isfinite(axis_point).all()
                or min(axes) < 2
                or roi[2] <= 0
                or roi[3] <= 0
            ):
                raise ValueError("Invalid calibration geometry.")

            self.calibration = data
            self.reference_center = center
            self.filtered_center = center.copy()
            self.reference_axes = axes
            self.current_axes = axes.copy()
            self.reference_angle = angle
            self.current_angle = angle
            self.axis_point = axis_point
            self.roi = roi
            self.state = "REFERENCE_ONLY"

            print("[HOLE] Loaded calibration:", self.calibration_path)
            print("[HOLE] Center:", center.tolist())
            print("[HOLE] Ellipse diameters:", axes.tolist())
            print("[HOLE] Axis point:", axis_point.tolist())

        except Exception as exc:
            self.calibration = None
            self.state = "CALIBRATION_ERROR"
            print("[HOLE] Calibration load failed:", repr(exc))

    def calibrate(self, frame):
        roi_window = "HOLE CALIBRATION - SELECT ROI"

        print(
            "[HOLE] Calibration requested. Opening ROI selection window.",
            flush=True,
        )
        print(
            "[HOLE] If the window is hidden, use Alt+Tab to locate it.",
            flush=True,
        )

        cv2.namedWindow(roi_window, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(
                roi_window, cv2.WND_PROP_TOPMOST, 1
            )
        except cv2.error:
            pass

        roi = cv2.selectROI(
            roi_window,
            frame,
            False,
            False,
        )

        try:
            cv2.setWindowProperty(
                roi_window, cv2.WND_PROP_TOPMOST, 0
            )
        except cv2.error:
            pass

        cv2.destroyWindow(roi_window)

        x, y, w, h = [int(v) for v in roi]
        if w < 8 or h < 8:
            print("[HOLE] ROI selection cancelled.")
            return False

        points = []
        window = "HOLE CALIBRATION - MARK GEOMETRY"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        def on_mouse(event, px, py, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
                points.append((int(px), int(py)))
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                points.pop()

        cv2.setMouseCallback(window, on_mouse)

        instructions = [
            "1: click HOLE CENTER",
            "2: click major-axis rim end A",
            "3: click opposite major-axis rim end B",
            "4: click minor-axis rim end A",
            "5: click opposite minor-axis rim end B",
            "6: click a point along the PIPE CENTERLINE",
        ]

        print("[HOLE] Selected ROI:", (x, y, w, h))
        print("[HOLE] Click 6 points in the specified order.")
        print("[HOLE] Right-click or U=undo; Enter=confirm; Esc=cancel.")
        print("[HOLE] Keep the hole mouth and a visible pipe-axis segment inside ROI.")

        cancelled = False
        try:
            while True:
                vis = frame.copy()
                cv2.rectangle(
                    vis, (x, y), (x + w, y + h), (255, 200, 0), 2
                )

                next_index = min(len(points), 5)
                cv2.putText(
                    vis,
                    instructions[next_index],
                    (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    vis,
                    f"points={len(points)}/6 | right-click/U undo | ENTER confirm",
                    (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 255),
                    1,
                )

                for i, point in enumerate(points):
                    cv2.circle(vis, point, 5, (0, 0, 255), -1)
                    cv2.putText(
                        vis, str(i + 1),
                        (point[0] + 6, point[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

                if len(points) >= 3:
                    cv2.line(vis, points[1], points[2], (255, 0, 255), 2)
                if len(points) >= 5:
                    cv2.line(vis, points[3], points[4], (255, 0, 255), 2)
                if len(points) >= 6:
                    cv2.line(vis, points[0], points[5], (255, 255, 0), 2)

                cv2.imshow(window, vis)
                key = cv2.waitKey(20) & 0xFF

                if key in (ord("u"), ord("U")) and points:
                    points.pop()

                elif key in (10, 13):
                    if len(points) == 6:
                        break
                    print(f"[HOLE] Need 6 points; current={len(points)}")

                elif key == 27:
                    cancelled = True
                    break

        finally:
            cv2.destroyWindow(window)

        if cancelled or len(points) != 6:
            print("[HOLE] Calibration cancelled.")
            return False

        pts = np.asarray(points, dtype=np.float64)
        center = pts[0]
        major_pair = pts[1:3]
        minor_pair = pts[3:5]
        axis_point = pts[5]

        major_vec = major_pair[1] - major_pair[0]
        minor_vec = minor_pair[1] - minor_pair[0]

        major_len = float(np.linalg.norm(major_vec))
        minor_len = float(np.linalg.norm(minor_vec))

        if major_len < 2 or minor_len < 2:
            print("[HOLE] Rim points too close; calibration rejected.")
            return False

        # Store the longer measured diameter as major axis.
        if major_len >= minor_len:
            major = major_len
            minor = minor_len
            angle = math.degrees(
                math.atan2(major_vec[1], major_vec[0])
            )
        else:
            major = minor_len
            minor = major_len
            angle = math.degrees(
                math.atan2(minor_vec[1], minor_vec[0])
            )

        if minor / major < 0.20:
            print("[HOLE] Ellipse is too elongated; calibration rejected.")
            return False

        # Ensure all clicked geometry lies inside the selected ROI.
        if not (
            x <= center[0] < x + w
            and y <= center[1] < y + h
            and x <= axis_point[0] < x + w
            and y <= axis_point[1] < y + h
        ):
            print(
                "[HOLE] Center or axis point is outside the ROI. "
                "Select a larger ROI and retry."
            )
            return False

        data = {
            "camera_serial": self.camera_serial,
            "method": "manual_hole_ellipse_and_pipe_axis",
            "roi_xywh": [x, y, w, h],
            "hole_center_uv": center.tolist(),
            "ellipse_diameters_px": [major, minor],
            "ellipse_angle_deg": angle,
            "axis_point_uv": axis_point.tolist(),
            "point_order": [
                "hole_center",
                "major_axis_rim_A",
                "major_axis_rim_B",
                "minor_axis_rim_A",
                "minor_axis_rim_B",
                "pipe_axis_point",
            ],
        }

        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        self.calibration_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.calibration = data
        self.reference_center = center
        self.filtered_center = center.copy()
        self.reference_axes = np.asarray([major, minor], dtype=np.float64)
        self.current_axes = self.reference_axes.copy()
        self.reference_angle = angle
        self.current_angle = angle
        self.axis_point = axis_point
        self.roi = (x, y, w, h)
        self.state = "REFERENCE_ONLY"

        print("[HOLE] Calibration saved:", self.calibration_path)
        print("[HOLE] Center:", center.tolist())
        print("[HOLE] Ellipse diameters:", [major, minor])
        print("[HOLE] Axis angle:", angle)
        print("[HOLE] Axis point:", axis_point.tolist())
        return True

    @staticmethod
    def _ellipse_points(center, axes, angle_deg, count=48):
        cx, cy = float(center[0]), float(center[1])
        major, minor = float(axes[0]), float(axes[1])
        angle = math.radians(float(angle_deg))
        ca, sa = math.cos(angle), math.sin(angle)

        points = []
        for theta in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False):
            u = (major / 2.0) * math.cos(theta)
            v = (minor / 2.0) * math.sin(theta)
            px = cx + u * ca - v * sa
            py = cy + u * sa + v * ca
            points.append([round(px, 2), round(py, 2)])
        return points

    def update(self, frame):
        # The hole tube is rigidly fixed relative to the wrist camera.
        # Use the manually calibrated image geometry directly.
        # No per-frame edge detection or ellipse fitting is performed.
        if (
            self.calibration is None
            or self.reference_center is None
            or self.reference_axes is None
            or self.axis_point is None
        ):
            self.last_result = self._empty_result()
            return self.last_result

        height, width = frame.shape[:2]
        center = self.reference_center.copy()
        axes = self.reference_axes.copy()
        axis_point = self.axis_point.copy()

        in_frame = bool(
            0 <= center[0] < width
            and 0 <= center[1] < height
        )

        direction = axis_point - center
        norm = float(np.linalg.norm(direction))
        direction_uv = (
            (direction / norm).tolist()
            if norm > 1e-6 else None
        )

        result = {
            "hole_calibrated": True,
            "hole_valid": in_frame,
            "hole_detected": False,
            "hole_reference_valid": in_frame,
            "hole_state": (
                "STATIC_CALIBRATED" if in_frame
                else "CALIBRATION_OUTSIDE_IMAGE"
            ),
            "hole_source": "STATIC_CALIBRATION",
            "hole_center_uv": center.tolist(),
            "hole_ellipse": {
                "center_uv": center.tolist(),
                "diameters_px": axes.tolist(),
                "angle_deg": float(self.reference_angle),
            },
            "hole_edge_points_uv": self._ellipse_points(
                center, axes, self.reference_angle
            ),
            "hole_axis_uv": [
                center.tolist(),
                axis_point.tolist(),
            ],
            "hole_axis_direction_uv": direction_uv,
            "hole_roi_xywh": list(self.roi),
            "hole_debug": {
                "mode": "STATIC_CALIBRATION",
                "per_frame_detection": False,
            },
        }

        self.filtered_center = center.copy()
        self.current_axes = axes.copy()
        self.current_angle = float(self.reference_angle)
        self.state = result["hole_state"]
        self.last_result = result
        return result

    def draw(self, image, result=None):
        if result is None:
            result = self.last_result

        if not result.get("hole_calibrated", False):
            cv2.putText(
                image,
                "HOLE: press H to calibrate",
                (15, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
            )
            return image

        x, y, w, h = result["hole_roi_xywh"]
        cv2.rectangle(
            image, (x, y), (x + w, y + h), (255, 200, 0), 1
        )

        ellipse = result["hole_ellipse"]
        center = tuple(np.rint(ellipse["center_uv"]).astype(int))
        major, minor = ellipse["diameters_px"]
        axes = (
            max(1, int(round(major / 2.0))),
            max(1, int(round(minor / 2.0))),
        )

        valid = bool(result.get("hole_valid", False))
        source = result.get("hole_source")

        if source == "STATIC_CALIBRATION":
            color = (255, 255, 0)
            label = "HOLE STATIC CALIBRATION"
        elif valid:
            color = (0, 255, 0)
            label = "HOLE AUTO TRACK"
        else:
            color = (0, 165, 255)
            label = "HOLE REFERENCE ONLY"

        cv2.ellipse(
            image,
            center,
            axes,
            float(ellipse["angle_deg"]),
            0,
            360,
            color,
            2,
        )

        debug = result.get("hole_debug", {})
        for candidate in debug.get("nearest_candidates", [])[:5]:
            cc = tuple(
                np.rint(candidate["center_uv"]).astype(int)
            )
            dd = candidate["diameters_px"]
            aa = (
                max(1, int(round(float(dd[0]) / 2.0))),
                max(1, int(round(float(dd[1]) / 2.0))),
            )
            candidate_color = (
                (0, 255, 255)
                if candidate["reason"] == "PASS"
                else (0, 0, 255)
            )
            cv2.ellipse(
                image,
                cc,
                aa,
                float(candidate["angle_deg"]),
                0,
                360,
                candidate_color,
                1,
            )
            cv2.putText(
                image,
                candidate["reason"],
                (cc[0] + 3, cc[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                candidate_color,
                1,
            )
        cv2.drawMarker(
            image,
            center,
            color,
            cv2.MARKER_CROSS,
            18,
            2,
        )

        axis_uv = result.get("hole_axis_uv")
        if axis_uv and len(axis_uv) == 2:
            p0 = tuple(np.rint(axis_uv[0]).astype(int))
            p1 = tuple(np.rint(axis_uv[1]).astype(int))
            cv2.line(image, p0, p1, (255, 255, 0), 2)

        cv2.putText(
            image,
            label,
            (max(5, x), max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
        )
        return image
