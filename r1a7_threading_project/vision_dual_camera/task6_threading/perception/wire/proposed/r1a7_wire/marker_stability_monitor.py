#!/usr/bin/env python3

import json
import math
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


OUTPUT = Path(
    "/tmp/r1a7_external_marker_stability.json"
)

DICT_NAME = "DICT_5X5_50"

# New tabletop stability marker.
TARGET_ID = 1

WINDOW = 30

MIN_DETECTION_RATIO = 0.90
MAX_LAST_SEEN_AGE_S = 0.20

MAX_SIGMA_U_PX = 1.5
MAX_SIGMA_V_PX = 1.5
MAX_SIGMA_SIDE_PX = 1.5
MAX_SIGMA_ANGLE_DEG = 1.0

# If marker disappears for a long time, do not mix
# old position statistics with a later reacquisition.
RESET_AFTER_LOST_S = 0.50


def atomic_json(path, data):

    tmp = Path(str(path) + ".tmp")

    tmp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


class MarkerStabilityMonitor:

    def __init__(self):

        dictionary_id = getattr(
            cv2.aruco,
            DICT_NAME,
        )

        self.dictionary = (
            cv2.aruco.getPredefinedDictionary(
                dictionary_id
            )
        )

        self.parameters = (
            cv2.aruco.DetectorParameters()
        )

        self.detector = (
            cv2.aruco.ArucoDetector(
                self.dictionary,
                self.parameters,
            )
        )

        self.presence = deque(
            maxlen=WINDOW
        )

        self.samples = deque(
            maxlen=WINDOW
        )

        self.last_seen_time = None

        print(
            "[MARKER STABILITY] "
            f"target ID={TARGET_ID}",
            flush=True,
        )

        print(
            "[MARKER STABILITY] "
            "diagnostic only - NO localization/control",
            flush=True,
        )


    def clear_samples(self):

        self.samples.clear()


    def update(self, frame):

        now = time.time()

        corners, ids, _ = (
            self.detector.detectMarkers(
                frame
            )
        )

        seen_ids = []

        target_pts = None

        if ids is not None:

            ids_flat = ids.reshape(-1)

            for marker_id, marker_corners in zip(
                ids_flat,
                corners,
            ):

                marker_id = int(
                    marker_id
                )

                seen_ids.append(
                    marker_id
                )

                if marker_id == TARGET_ID:

                    target_pts = np.asarray(
                        marker_corners,
                        dtype=np.float64,
                    ).reshape(4, 2)


        currently_seen = (
            target_pts is not None
        )

        self.presence.append(
            1 if currently_seen else 0
        )


        if currently_seen:

            if (
                self.last_seen_time is not None
                and
                now - self.last_seen_time
                >
                RESET_AFTER_LOST_S
            ):

                self.clear_samples()

            center = np.mean(
                target_pts,
                axis=0,
            )

            edge = (
                target_pts[1]
                -
                target_pts[0]
            )

            angle_deg = math.degrees(
                math.atan2(
                    edge[1],
                    edge[0],
                )
            )

            sides = []

            for i in range(4):

                sides.append(
                    float(
                        np.linalg.norm(
                            target_pts[
                                (i + 1) % 4
                            ]
                            -
                            target_pts[i]
                        )
                    )
                )

            side = float(
                np.mean(
                    sides
                )
            )

            self.samples.append(
                (
                    float(center[0]),
                    float(center[1]),
                    side,
                    angle_deg,
                )
            )

            self.last_seen_time = now


        if self.last_seen_time is None:

            last_seen_age = math.inf

        else:

            last_seen_age = (
                now - self.last_seen_time
            )


        if len(self.presence) > 0:

            detection_ratio = float(
                sum(self.presence)
                /
                len(self.presence)
            )

        else:

            detection_ratio = 0.0


        stats = None

        if len(self.samples) >= 2:

            a = np.asarray(
                self.samples,
                dtype=np.float64,
            )

            u = a[:, 0]
            v = a[:, 1]
            side = a[:, 2]

            angles = np.rad2deg(
                np.unwrap(
                    np.deg2rad(
                        a[:, 3]
                    )
                )
            )

            sigma_u = float(
                np.std(u)
            )

            sigma_v = float(
                np.std(v)
            )

            sigma_side = float(
                np.std(side)
            )

            sigma_angle = float(
                np.std(angles)
            )

            stable = bool(
                currently_seen
                and
                detection_ratio
                >=
                MIN_DETECTION_RATIO
                and
                last_seen_age
                <=
                MAX_LAST_SEEN_AGE_S
                and
                sigma_u
                <=
                MAX_SIGMA_U_PX
                and
                sigma_v
                <=
                MAX_SIGMA_V_PX
                and
                sigma_side
                <=
                MAX_SIGMA_SIDE_PX
                and
                sigma_angle
                <=
                MAX_SIGMA_ANGLE_DEG
            )

            stats = {
                "sample_count":
                    len(self.samples),

                "center_uv": [
                    float(np.mean(u)),
                    float(np.mean(v)),
                ],

                "sigma_u_px":
                    sigma_u,

                "sigma_v_px":
                    sigma_v,

                "sigma_center_px":
                    float(
                        math.sqrt(
                            sigma_u * sigma_u
                            +
                            sigma_v * sigma_v
                        )
                    ),

                "sigma_side_px":
                    sigma_side,

                "sigma_angle_deg":
                    sigma_angle,

                "stable":
                    stable,
            }



        out = {
            "timestamp":
                now,

            "dictionary":
                DICT_NAME,

            "target_id":
                TARGET_ID,

            "role":
                "stability_only",

            "used_for_localization":
                False,

            "used_for_control":
                False,

            "seen_ids":
                sorted(
                    seen_ids
                ),

            "target_currently_seen":
                currently_seen,

            "detection_ratio":
                detection_ratio,

            "last_seen_age_s":
                (
                    last_seen_age
                    if math.isfinite(
                        last_seen_age
                    )
                    else None
                ),

            "target_corners_uv":
                (
                    target_pts.tolist()
                    if currently_seen
                    else None
                ),

            "target_bbox_xyxy":
                (
                    [
                        int(np.floor(target_pts[:, 0].min())),
                        int(np.floor(target_pts[:, 1].min())),
                        int(np.ceil(target_pts[:, 0].max())),
                        int(np.ceil(target_pts[:, 1].max())),
                    ]
                    if currently_seen
                    else None
                ),

            "target":
                stats,
        }

        atomic_json(
            OUTPUT,
            out,
        )

        return {
            "detected": bool(currently_seen),
            "target_id": int(TARGET_ID),
            "corners": (
                target_pts.tolist()
                if currently_seen and target_pts is not None
                else None
            ),
            "stats": stats,
            "detection_ratio": float(detection_ratio),
        }


_monitor = None


def monitor_frame(frame):

    global _monitor

    if _monitor is None:

        _monitor = (
            MarkerStabilityMonitor()
        )

    try:

        return _monitor.update(
            frame
        )

    except Exception as exc:

        atomic_json(
            OUTPUT,
            {
                "timestamp":
                    time.time(),

                "target_id":
                    TARGET_ID,

                "role":
                    "stability_only",

                "used_for_localization":
                    False,

                "used_for_control":
                    False,

                "error":
                    type(exc).__name__
                    +
                    ":"
                    +
                    str(exc),
            },
        )


def draw_marker_overlay(frame, status):
    """Draw tabletop ID 1 on an existing camera display image only."""
    if frame is None:
        return frame

    if status is None:
        return frame

    corners = status.get("corners")
    target_id = status.get("target_id", TARGET_ID)

    if corners is None:
        cv2.putText(
            frame,
            f"TABLE MARKER ID {target_id}: LOST",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    pts_i = np.round(pts).astype(np.int32)

    cv2.polylines(
        frame,
        [pts_i.reshape((-1, 1, 2))],
        True,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    center = tuple(
        np.round(pts.mean(axis=0)).astype(int)
    )

    cv2.circle(
        frame,
        center,
        5,
        (0, 255, 255),
        -1,
    )

    stats = status.get("stats") or {}
    sigma = stats.get("sigma_center_px")
    sigma_text = (
        f"{sigma:.2f}px"
        if sigma is not None
        else "warming"
    )

    ratio = status.get("detection_ratio", 0.0)

    cv2.putText(
        frame,
        f"TABLE ID {target_id} sigma={sigma_text} det={ratio:.2f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame
