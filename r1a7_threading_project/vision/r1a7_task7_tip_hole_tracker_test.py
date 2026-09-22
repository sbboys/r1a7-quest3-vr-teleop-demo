#!/usr/bin/env python3

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from r1a7_rigid_tool_tracker import RigidRoiTracker


# ============================================================
# TASK 7 - TIP / HOLE DUAL TRACKING
# Visual-only test.
#
# NO ROBOT COMMAND IS SENT BY THIS PROGRAM.
# ============================================================

CAMERA_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-ALP_USB_ZOOM_Camera_20685204b12d5283-video-index0"
)

ALIGN_THRESHOLD_PX = 15.0
ALIGN_STABLE_FRAMES = 10


def select_roi(title, frame):
    roi = cv2.selectROI(
        title,
        frame,
        fromCenter=False,
        showCrosshair=True,
    )

    cv2.destroyWindow(title)

    x, y, w, h = [int(v) for v in roi]

    if w <= 0 or h <= 0:
        return None

    return np.asarray(
        [x, y, w, h],
        dtype=float,
    )


def draw_tracker(
    frame,
    tracker,
    name,
):
    if tracker.roi is None:
        return

    x, y, w, h = [
        int(round(v))
        for v in tracker.roi
    ]

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (255, 255, 255),
        2,
    )

    if tracker.center is not None:
        u = int(round(tracker.center[0]))
        v = int(round(tracker.center[1]))

        cv2.drawMarker(
            frame,
            (u, v),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            20,
            2,
        )

        cv2.putText(
            frame,
            name,
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main():

    print()
    print("========================================")
    print("R1-A7 TASK7 TIP-HOLE TRACKER TEST")
    print("CAMERA =", CAMERA_DEVICE)
    print("ROBOT CONTROL = DISABLED")
    print("========================================")
    print()

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open fixed USB camera."
        )

    # Give camera a little time to settle.
    time.sleep(0.5)

    frame = None

    for _ in range(20):
        ok, candidate = cap.read()

        if ok:
            frame = candidate

    if frame is None:
        raise RuntimeError(
            "Cannot obtain initial camera frame."
        )

    print(
        "Camera frame:",
        frame.shape,
        flush=True,
    )

    # --------------------------------------------------------
    # 1. Manually select tweezer TIP.
    # --------------------------------------------------------

    print()
    print("STEP 1")
    print("Select a SMALL ROI around the tweezer TIP.")
    print("Press ENTER / SPACE to confirm.")

    tip_roi = select_roi(
        "TASK7 - Select TWEEZER TIP",
        frame,
    )

    if tip_roi is None:
        raise RuntimeError(
            "TIP ROI selection cancelled."
        )

    # --------------------------------------------------------
    # 2. Manually select HOLE.
    # --------------------------------------------------------

    print()
    print("STEP 2")
    print("Select the target HOLE ROI.")
    print("Press ENTER / SPACE to confirm.")

    hole_roi = select_roi(
        "TASK7 - Select HOLE",
        frame,
    )

    if hole_roi is None:
        raise RuntimeError(
            "HOLE ROI selection cancelled."
        )

    # --------------------------------------------------------
    # 3. Initialize two independent verified trackers.
    # --------------------------------------------------------

    tip_tracker = RigidRoiTracker()
    hole_tracker = RigidRoiTracker()

    tip_ok = tip_tracker.initialize(
        frame,
        tip_roi,
    )

    hole_ok = hole_tracker.initialize(
        frame,
        hole_roi,
    )

    if not tip_ok:
        raise RuntimeError(
            "TIP tracker initialization failed: "
            + tip_tracker.status
        )

    if not hole_ok:
        raise RuntimeError(
            "HOLE tracker initialization failed: "
            + hole_tracker.status
        )

    print()
    print(
        "[TIP] initialized features =",
        tip_tracker.good_count,
    )

    print(
        "[HOLE] initialized features =",
        hole_tracker.good_count,
    )

    # --------------------------------------------------------
    # Logging.
    # --------------------------------------------------------

    log_dir = Path(
        "/home/robot/unitree_sim_isaaclab_threading/"
        "r1a7_threading_project/vision/task7_logs"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        log_dir
        / datetime.now().strftime(
            "tip_hole_%Y%m%d_%H%M%S.csv"
        )
    )

    log_file = log_path.open(
        "w",
        newline="",
    )

    writer = csv.writer(log_file)

    writer.writerow(
        [
            "time",
            "tip_u",
            "tip_v",
            "hole_u",
            "hole_v",
            "eu",
            "ev",
            "pixel_error",
            "tip_features",
            "hole_features",
            "tip_status",
            "hole_status",
        ]
    )

    aligned_frames = 0
    last_print = 0.0

    print()
    print("========================================")
    print("TRACKING ACTIVE")
    print("Q = quit")
    print("Robot remains DISABLED.")
    print("========================================")

    while True:

        ok, frame = cap.read()

        if not ok:
            print("[CAMERA] frame read failed")
            break

        # ----------------------------------------------------
        # Update BOTH trackers using same camera frame.
        # ----------------------------------------------------

        tip_updated = tip_tracker.update(
            frame
        )

        hole_updated = hole_tracker.update(
            frame
        )

        display = frame.copy()

        draw_tracker(
            display,
            tip_tracker,
            "TIP",
        )

        draw_tracker(
            display,
            hole_tracker,
            "HOLE",
        )

        # ----------------------------------------------------
        # Safety / tracking state.
        # ----------------------------------------------------

        if (
            not tip_tracker.active
            or not hole_tracker.active
            or tip_tracker.center is None
            or hole_tracker.center is None
        ):

            aligned_frames = 0

            cv2.putText(
                display,
                "TRACK LOST - ROBOT MUST STOP",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        else:

            tip = tip_tracker.center.copy()
            hole = hole_tracker.center.copy()

            # =================================================
            # CORE VISUAL ERROR
            # =================================================

            eu = float(
                hole[0] - tip[0]
            )

            ev = float(
                hole[1] - tip[1]
            )

            error = math.hypot(
                eu,
                ev,
            )

            # Line: TIP -> HOLE
            cv2.line(
                display,
                (
                    int(round(tip[0])),
                    int(round(tip[1])),
                ),
                (
                    int(round(hole[0])),
                    int(round(hole[1])),
                ),
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f"eu = {eu:+.1f} px",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f"ev = {ev:+.1f} px",
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f"error = {error:.1f} px",
                (20, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                (
                    f"TIP features="
                    f"{tip_tracker.good_count}"
                ),
                (20, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                display,
                (
                    f"HOLE features="
                    f"{hole_tracker.good_count}"
                ),
                (20, 137),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # -------------------------------------------------
            # Alignment observation only.
            # NO robot motion yet.
            # -------------------------------------------------

            if error <= ALIGN_THRESHOLD_PX:
                aligned_frames += 1
            else:
                aligned_frames = 0

            if (
                aligned_frames
                >= ALIGN_STABLE_FRAMES
            ):
                cv2.putText(
                    display,
                    "APPROACH ALIGNMENT OK",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                display,
                "ROBOT: DISABLED",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            now = time.monotonic()

            if now - last_print >= 0.5:

                print(
                    "[VISION]"
                    f" tip=({tip[0]:.1f},"
                    f"{tip[1]:.1f})"
                    f" hole=({hole[0]:.1f},"
                    f"{hole[1]:.1f})"
                    f" eu={eu:+.1f}"
                    f" ev={ev:+.1f}"
                    f" error={error:.1f}"
                    f" tipN={tip_tracker.good_count}"
                    f" holeN={hole_tracker.good_count}",
                    flush=True,
                )

                last_print = now

            writer.writerow(
                [
                    time.time(),
                    float(tip[0]),
                    float(tip[1]),
                    float(hole[0]),
                    float(hole[1]),
                    eu,
                    ev,
                    error,
                    tip_tracker.good_count,
                    hole_tracker.good_count,
                    tip_tracker.status,
                    hole_tracker.status,
                ]
            )

        cv2.imshow(
            "R1-A7 TASK7 TIP-HOLE TRACKING",
            display,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    log_file.close()
    cap.release()
    cv2.destroyAllWindows()

    print()
    print("Stopped.")
    print("Log saved to:")
    print(log_path)


if __name__ == "__main__":
    main()
