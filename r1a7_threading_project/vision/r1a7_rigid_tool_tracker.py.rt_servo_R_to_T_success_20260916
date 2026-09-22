from __future__ import annotations

import cv2
import numpy as np


MAX_CORNERS = 40
MIN_FEATURES = 6


def detect_features(gray, roi):
    x, y, w, h = [
        int(round(v))
        for v in roi
    ]

    x = max(0, x)
    y = max(0, y)

    x2 = min(gray.shape[1], x + max(1, w))
    y2 = min(gray.shape[0], y + max(1, h))

    mask = np.zeros_like(gray)
    mask[y:y2, x:x2] = 255

    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=5,
    )


def track_features(prev_gray, gray, prev_pts):
    if prev_pts is None or len(prev_pts) == 0:
        return None, None, None

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_pts,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if next_pts is None or status is None:
        return None, None, None

    back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        prev_gray,
        next_pts,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if back_pts is None or back_status is None:
        return None, None, None

    p0 = prev_pts.reshape(-1, 2)
    p1 = next_pts.reshape(-1, 2)
    pb = back_pts.reshape(-1, 2)

    st1 = status.reshape(-1).astype(bool)
    st2 = back_status.reshape(-1).astype(bool)

    fb_error = np.linalg.norm(
        p0 - pb,
        axis=1,
    )

    finite = (
        np.all(np.isfinite(p0), axis=1)
        & np.all(np.isfinite(p1), axis=1)
        & np.isfinite(fb_error)
    )

    good = (
        st1
        & st2
        & finite
        & (fb_error < 1.0)
    )

    p0_good = p0[good]
    p1_good = p1[good]

    if len(p0_good) < MIN_FEATURES:
        return None, None, None

    displacement = (
        p1_good - p0_good
    )

    median_delta = np.median(
        displacement,
        axis=0,
    )

    return (
        p1_good.reshape(
            -1, 1, 2
        ).astype(np.float32),
        median_delta,
        p1_good,
    )


class RigidRoiTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.lost = False

        self.prev_gray = None
        self.prev_pts = None

        self.roi = None
        self.roi_size = None

        self.center = None
        self.initial_center = None

        self.visible_points = None

        self.good_count = 0
        self.failure_count = 0

        self.frame_delta = np.zeros(
            2,
            dtype=float,
        )

        self.total_delta = np.zeros(
            2,
            dtype=float,
        )

        self.status = "IDLE"

    @staticmethod
    def _gray(frame):
        if frame.ndim == 2:
            return frame

        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

    def _update_roi(self, shape):
        w, h = self.roi_size

        x = self.center[0] - w / 2.0
        y = self.center[1] - h / 2.0

        x = np.clip(
            x,
            0.0,
            max(0.0, shape[1] - w),
        )

        y = np.clip(
            y,
            0.0,
            max(0.0, shape[0] - h),
        )

        self.roi = np.asarray(
            [x, y, w, h],
            dtype=float,
        )

    def initialize(self, frame, roi):
        self.reset()

        gray = self._gray(frame)

        roi = np.asarray(
            roi,
            dtype=float,
        ).reshape(4)

        pts = detect_features(
            gray,
            roi,
        )

        count = (
            0
            if pts is None
            else len(pts)
        )

        if count < MIN_FEATURES:
            self.status = (
                f"INIT_FAIL N={count}"
            )
            return False

        x, y, w, h = roi

        self.roi = roi.copy()

        self.roi_size = np.asarray(
            [w, h],
            dtype=float,
        )

        self.center = np.asarray(
            [
                x + w / 2.0,
                y + h / 2.0,
            ],
            dtype=float,
        )

        self.initial_center = (
            self.center.copy()
        )

        self.prev_gray = gray
        self.prev_pts = pts

        self.visible_points = (
            pts.reshape(-1, 2).copy()
        )

        self.good_count = int(
            len(pts)
        )

        self.active = True
        self.lost = False
        self.status = "TRACKING"

        return True

    def update(self, frame):
        if not self.active:
            return False

        gray = self._gray(frame)

        new_points, delta, visible_points = (
            track_features(
                self.prev_gray,
                gray,
                self.prev_pts,
            )
        )

        if new_points is None:
            self.failure_count += 1

            self.status = (
                f"TRANSIENT_FAIL "
                f"{self.failure_count}/3"
            )

            if self.failure_count >= 3:
                self.active = False
                self.lost = True
                self.status = "LOST"

            return False

        self.failure_count = 0

        self.prev_gray = gray
        self.prev_pts = new_points

        self.visible_points = (
            visible_points.copy()
        )

        self.good_count = int(
            len(new_points)
        )

        self.frame_delta = np.asarray(
            delta,
            dtype=float,
        )

        self.total_delta += (
            self.frame_delta
        )

        self.center = (
            self.initial_center
            + self.total_delta
        )

        self.center[0] = np.clip(
            self.center[0],
            0.0,
            gray.shape[1] - 1,
        )

        self.center[1] = np.clip(
            self.center[1],
            0.0,
            gray.shape[0] - 1,
        )

        self._update_roi(
            gray.shape
        )

        self.status = "TRACKING"

        return True
