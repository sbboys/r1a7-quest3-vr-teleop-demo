#!/usr/bin/env python3

import math
import os
import time

import cv2
import numpy as np

from r1a7_rigid_tool_tracker import RigidRoiTracker


CAMERA_BY_ID = (
    "/dev/v4l/by-id/"
    "usb-ALP_USB_ZOOM_Camera_20685204b12d5283-video-index0"
)

ALIGN_THRESHOLD_PX = 15.0
ALIGN_STABLE_FRAMES = 10


def select_roi(title, frame):
    selected = cv2.selectROI(
        title,
        frame,
        fromCenter=False,
        showCrosshair=True,
    )

    cv2.destroyWindow(title)

    x, y, w, h = [
        int(v)
        for v in selected
    ]

    if w <= 0 or h <= 0:
        return None

    return np.asarray(
        [x, y, w, h],
        dtype=float,
    )


def main():

    print()
    print("========================================")
    print("TASK7 TIP -> FIXED HOLE VISUAL TEST")
    print("ROBOT CONTROL = DISABLED")
    print("========================================")

    camera_device = os.path.realpath(
        CAMERA_BY_ID
    )

    print(
        "Camera =",
        camera_device,
    )

    cap = cv2.VideoCapture(
        camera_device
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open camera: "
            + camera_device
        )

    time.sleep(0.5)

    frame = None

    for _ in range(20):
        ok, candidate = cap.read()

        if ok:
            frame = candidate

    if frame is None:
        raise RuntimeError(
            "Cannot read camera frame."
        )

    print(
        "Frame =",
        frame.shape,
    )

    # ========================================================
    # STEP 1 - TIP
    # ========================================================

    print()
    print("STEP 1")
    print("Select TWEEZER TIP ROI")
    print(
        "Include the tip and some visible "
        "edges / texture."
    )

    tip_roi = select_roi(
        "Select TWEEZER TIP",
        frame,
    )

    if tip_roi is None:
        raise RuntimeError(
            "TIP selection cancelled."
        )

    tip_tracker = RigidRoiTracker()

    if not tip_tracker.initialize(
        frame,
        tip_roi,
    ):
        raise RuntimeError(
            "TIP initialization failed: "
            + tip_tracker.status
        )

    print(
        "[TIP] initialized features =",
        tip_tracker.good_count,
    )

    # ========================================================
    # STEP 2 - HOLE
    #
    # IMPORTANT:
    # Hole is fixed in the world and the camera is fixed.
    # Therefore we only save its image-space center.
    # No optical-flow tracker is needed.
    # ========================================================

    print()
    print("STEP 2")
    print("Select TARGET HOLE region")
    print(
        "The center of this ROI will become "
        "the fixed visual target T."
    )

    hole_roi = select_roi(
        "Select TARGET HOLE",
        frame,
    )

    if hole_roi is None:
        raise RuntimeError(
            "HOLE selection cancelled."
        )

    hx, hy, hw, hh = hole_roi

    hole_center = np.asarray(
        [
            hx + hw / 2.0,
            hy + hh / 2.0,
        ],
        dtype=float,
    )

    print(
        "[HOLE] fixed target =",
        np.round(
            hole_center,
            2,
        ).tolist(),
    )

    print()
    print("========================================")
    print("VISUAL TRACKING ACTIVE")
    print("TIP = live tracker")
    print("HOLE = fixed target")
    print("ROBOT = DISABLED")
    print("Q = quit")
    print("========================================")

    aligned_frames = 0
    last_print = 0.0

    while True:

        ok, frame = cap.read()

        if not ok:
            print(
                "[CAMERA] frame read failed"
            )
            break

        # ----------------------------------------------------
        # Update moving tweezer tip only.
        # ----------------------------------------------------

        tip_tracker.update(
            frame
        )

        display = frame.copy()

        if (
            not tip_tracker.active
            or tip_tracker.center is None
        ):

            cv2.putText(
                display,
                "TIP TRACK LOST - STOP",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            aligned_frames = 0

        else:

            tip = (
                tip_tracker.center.copy()
            )

            hole = hole_center

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

            # ------------------------------------------------
            # Draw TIP ROI.
            # ------------------------------------------------

            x, y, w, h = [
                int(round(v))
                for v in tip_tracker.roi
            ]

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2,
            )

            tip_px = (
                int(round(tip[0])),
                int(round(tip[1])),
            )

            hole_px = (
                int(round(hole[0])),
                int(round(hole[1])),
            )

            cv2.drawMarker(
                display,
                tip_px,
                (255, 255, 255),
                cv2.MARKER_CROSS,
                18,
                2,
            )

            cv2.putText(
                display,
                "TIP",
                (
                    tip_px[0] + 10,
                    tip_px[1] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # ------------------------------------------------
            # Draw fixed HOLE target.
            # ------------------------------------------------

            cv2.drawMarker(
                display,
                hole_px,
                (255, 255, 255),
                cv2.MARKER_CROSS,
                24,
                2,
            )

            cv2.circle(
                display,
                hole_px,
                10,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                "HOLE TARGET",
                (
                    hole_px[0] + 12,
                    hole_px[1] - 12,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # TIP -> HOLE error vector
            cv2.line(
                display,
                tip_px,
                hole_px,
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
                    "TIP features = "
                    f"{tip_tracker.good_count}"
                ),
                (20, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                display,
                "ROBOT = DISABLED",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

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
                    (20, 178),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
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
                    f" tipN={tip_tracker.good_count}",
                    flush=True,
                )

                last_print = now

        cv2.imshow(
            "TASK7 TIP -> FIXED HOLE",
            display,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Stopped.")


if __name__ == "__main__":
    main()
