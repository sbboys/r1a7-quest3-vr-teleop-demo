#!/usr/bin/env python3

import sys
import cv2
import json
import time
import numpy as np
from pathlib import Path


# ============================================================
# Import the already-verified right-wrist Orbbec helpers.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HANDEYE_DIR = (
    PROJECT_ROOT
    /
    "vision_dual_camera"
    /
    "handeye"
)

sys.path.insert(
    0,
    str(HANDEYE_DIR)
)

import collect_handeye_samples as hc


# ============================================================
# Global UI / tracking state
# ============================================================

hole = None

tip_center = None
tip_roi = None

feature_points = None

frame_now = None

tracking_enabled = False


ROI_HALF_SIZE = 50


# ============================================================
# Mouse
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global hole
    global tip_center
    global tip_roi
    global feature_points
    global frame_now
    global tracking_enabled

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # --------------------------------------------------------
    # First click: hole center
    # --------------------------------------------------------

    if hole is None:

        hole = (
            int(x),
            int(y)
        )

        print(
            f"[WRIST SELECT] HOLE "
            f"u={hole[0]} v={hole[1]}",
            flush=True
        )

        return

    # --------------------------------------------------------
    # Any later left click:
    # manually define / reseed tweezer feature center.
    #
    # Hole remains unchanged.
    # --------------------------------------------------------

    if frame_now is None:
        return

    tip_center = (
        int(x),
        int(y)
    )

    tip_roi = (
        int(x - ROI_HALF_SIZE),
        int(y - ROI_HALF_SIZE),
        int(x + ROI_HALF_SIZE),
        int(y + ROI_HALF_SIZE)
    )

    gray = cv2.cvtColor(
        frame_now,
        cv2.COLOR_BGR2GRAY
    )

    mask = np.zeros_like(
        gray
    )

    x0, y0, x1, y1 = tip_roi

    x0 = max(
        0,
        x0
    )

    y0 = max(
        0,
        y0
    )

    x1 = min(
        gray.shape[1],
        x1
    )

    y1 = min(
        gray.shape[0],
        y1
    )

    mask[
        y0:y1,
        x0:x1
    ] = 255

    feature_points = (
        cv2.goodFeaturesToTrack(
            gray,
            maxCorners=50,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=7,
            mask=mask
        )
    )

    if feature_points is None:

        tracking_enabled = False

        print(
            "[WRIST TRACK] "
            "No Shi-Tomasi features found in ROI.",
            flush=True
        )

        return

    tracking_enabled = True

    print(
        f"[WRIST SELECT] TWEEZER "
        f"u={tip_center[0]} "
        f"v={tip_center[1]} "
        f"features={len(feature_points)}",
        flush=True
    )


# ============================================================
# Forward / backward LK tracker
# ============================================================

def track_tip(
    prev_gray,
    gray
):

    global feature_points
    global tip_center
    global tracking_enabled

    if (
        not tracking_enabled
        or
        feature_points is None
        or
        tip_center is None
    ):
        return

    p0 = np.asarray(
        feature_points,
        dtype=np.float32
    ).reshape(
        -1,
        1,
        2
    )

    # Forward
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        p0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if (
        p1 is None
        or
        st1 is None
    ):

        tracking_enabled = False

        print(
            "[WRIST TRACK] LOST "
            "(forward LK failed)",
            flush=True
        )

        return

    # Backward
    p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        prev_gray,
        p1,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if (
        p0_back is None
        or
        st2 is None
    ):

        tracking_enabled = False

        print(
            "[WRIST TRACK] LOST "
            "(backward LK failed)",
            flush=True
        )

        return

    p0_xy = p0.reshape(
        -1,
        2
    )

    p1_xy = p1.reshape(
        -1,
        2
    )

    pb_xy = p0_back.reshape(
        -1,
        2
    )

    valid = (
        st1.reshape(-1).astype(bool)
        &
        st2.reshape(-1).astype(bool)
    )

    fb_error = np.linalg.norm(
        p0_xy - pb_xy,
        axis=1
    )

    good = (
        valid
        &
        np.isfinite(fb_error)
        &
        (fb_error < 1.5)
    )

    number_good = int(
        np.count_nonzero(
            good
        )
    )

    if number_good < 3:

        tracking_enabled = False

        print(
            "[WRIST TRACK] LOST "
            f"reliable_features={number_good}. "
            "Click tweezer again to reseed.",
            flush=True
        )

        return

    good_old = p0_xy[
        good
    ]

    good_new = p1_xy[
        good
    ]

    displacement = (
        good_new
        -
        good_old
    )

    median_motion = np.median(
        displacement,
        axis=0
    )

    dx = float(
        median_motion[0]
    )

    dy = float(
        median_motion[1]
    )

    tip_center = (
        int(
            round(
                tip_center[0]
                +
                dx
            )
        ),
        int(
            round(
                tip_center[1]
                +
                dy
            )
        )
    )

    feature_points = (
        good_new
        .reshape(-1, 1, 2)
        .astype(np.float32)
    )


# ============================================================
# Main
# ============================================================

def main():

    global hole
    global tip_center
    global tip_roi
    global feature_points
    global frame_now
    global tracking_enabled

    from pyorbbecsdk import (
        Config,
        OBFormat,
        OBSensorType,
        Pipeline,
    )

    print(
        "================================="
    )

    print(
        "TASK6 RIGHT WRIST ROI LK TRACKER"
    )

    print(
        "NO ROBOT CONTROL"
    )

    print(
        "================================="
    )

    print(
        f"Right wrist SN : "
        f"{hc.RIGHT_WRIST_SN}"
    )

    print(
        f"RGB profile    : "
        f"{hc.COLOR_WIDTH}x"
        f"{hc.COLOR_HEIGHT}"
        f"@{hc.COLOR_FPS}"
    )

    print()

    # --------------------------------------------------------
    # Lock exact right wrist camera by SN.
    # --------------------------------------------------------

    context, device = (
        hc.find_orbbec_device(
            hc.RIGHT_WRIST_SN
        )
    )

    pipeline = Pipeline(
        device
    )

    config = Config()

    profiles = (
        pipeline
        .get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
    )

    try:

        color_profile = (
            profiles
            .get_video_stream_profile(
                hc.COLOR_WIDTH,
                hc.COLOR_HEIGHT,
                OBFormat.RGB,
                hc.COLOR_FPS
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Required right-wrist RGB profile "
            f"{hc.COLOR_WIDTH}x"
            f"{hc.COLOR_HEIGHT}"
            f"@{hc.COLOR_FPS} RGB unavailable."
        ) from exc

    config.enable_stream(
        color_profile
    )

    pipeline.start(
        config
    )

    print(
        "[WRIST CAMERA] RGB started: "
        f"{hc.COLOR_WIDTH}x"
        f"{hc.COLOR_HEIGHT}"
        f"@{hc.COLOR_FPS}",
        flush=True
    )

    print()
    print("Mouse:")
    print("  first left click = HOLE")
    print("  later left click = TWEEZER / reseed ROI")
    print()
    print("Keys:")
    print("  C = clear all")
    print("  S = save current state")
    print("  Q = quit")
    print()

    window = (
        "TASK6 RIGHT WRIST IBVS"
    )

    cv2.namedWindow(
        window
    )

    cv2.setMouseCallback(
        window,
        mouse_callback
    )

    prev_gray = None

    last_print = 0.0

    save_path = (
        PROJECT_ROOT
        /
        "vision_dual_camera"
        /
        "task6_threading"
        /
        "wrist_ibvs_state.json"
    )

    try:

        while True:

            frames = (
                pipeline.wait_for_frames(
                    1000
                )
            )

            if frames is None:
                continue

            color_frame = (
                frames.get_color_frame()
            )

            if color_frame is None:
                continue

            width = int(
                color_frame.get_width()
            )

            height = int(
                color_frame.get_height()
            )

            if (
                width != hc.COLOR_WIDTH
                or
                height != hc.COLOR_HEIGHT
            ):

                raise RuntimeError(
                    "Unexpected right-wrist RGB resolution: "
                    f"{width}x{height}"
                )

            frame = (
                hc.color_frame_to_bgr(
                    color_frame
                )
            )

            if frame is None:
                continue

            frame_now = (
                frame.copy()
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            if (
                prev_gray is not None
                and
                tracking_enabled
            ):

                track_tip(
                    prev_gray,
                    gray
                )

            image = (
                frame.copy()
            )

            # ------------------------------------------------
            # Hole marker
            # ------------------------------------------------

            if hole is not None:

                cv2.drawMarker(
                    image,
                    hole,
                    (0, 255, 0),
                    cv2.MARKER_CROSS,
                    30,
                    2
                )

                cv2.putText(
                    image,
                    "HOLE",
                    (
                        hole[0] + 15,
                        hole[1] - 15
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # ------------------------------------------------
            # Tweezer visual point
            # ------------------------------------------------

            if tip_center is not None:

                cv2.drawMarker(
                    image,
                    tip_center,
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    30,
                    2
                )

                cv2.putText(
                    image,
                    "TWEEZER",
                    (
                        tip_center[0] + 15,
                        tip_center[1] - 15
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # Draw current LK features.
            if (
                feature_points is not None
                and
                tracking_enabled
            ):

                for point in (
                    feature_points
                    .reshape(-1, 2)
                ):

                    px = int(
                        round(
                            point[0]
                        )
                    )

                    py = int(
                        round(
                            point[1]
                        )
                    )

                    cv2.circle(
                        image,
                        (px, py),
                        2,
                        (255, 0, 255),
                        -1
                    )

            # ------------------------------------------------
            # Image error
            # ------------------------------------------------

            if (
                hole is not None
                and
                tip_center is not None
            ):

                du = float(
                    hole[0]
                    -
                    tip_center[0]
                )

                dv = float(
                    hole[1]
                    -
                    tip_center[1]
                )

                error = float(
                    np.hypot(
                        du,
                        dv
                    )
                )

                cv2.line(
                    image,
                    tip_center,
                    hole,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    (
                        f"du={du:+.1f}px "
                        f"dv={dv:+.1f}px "
                        f"err={error:.1f}px"
                    ),
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    "Wrist mapping: U -> Y, V -> Z",
                    (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                now = time.time()

                if (
                    now
                    -
                    last_print
                    >= 0.2
                ):

                    feature_count = (
                        0
                        if feature_points is None
                        else
                        len(feature_points)
                    )

                    print(
                        "[WRIST IBVS] "
                        f"R={tip_center} "
                        f"T={hole} "
                        f"du={du:+.1f} "
                        f"dv={dv:+.1f} "
                        f"err={error:.1f} "
                        f"features={feature_count}",
                        flush=True
                    )

                    last_print = now

            status = (
                "TRACKING"
                if tracking_enabled
                else
                "NOT TRACKING"
            )

            cv2.putText(
                image,
                status,
                (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    (0, 255, 0)
                    if tracking_enabled
                    else
                    (0, 0, 255)
                ),
                2
            )

            cv2.imshow(
                window,
                image
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # Q / ESC
            if (
                key == ord("q")
                or
                key == 27
            ):
                break

            # Clear all
            if key == ord("c"):

                hole = None
                tip_center = None
                tip_roi = None
                feature_points = None
                tracking_enabled = False

                print(
                    "[WRIST] CLEAR",
                    flush=True
                )

            # Save current state
            if key == ord("s"):

                if (
                    hole is None
                    or
                    tip_center is None
                ):

                    print(
                        "[WRIST SAVE] "
                        "hole/tweezer not selected",
                        flush=True
                    )

                else:

                    du = float(
                        hole[0]
                        -
                        tip_center[0]
                    )

                    dv = float(
                        hole[1]
                        -
                        tip_center[1]
                    )

                    data = {
                        "timestamp":
                            time.time(),

                        "camera_serial":
                            hc.RIGHT_WRIST_SN,

                        "resolution":
                            [
                                hc.COLOR_WIDTH,
                                hc.COLOR_HEIGHT
                            ],

                        "hole_pixel":
                            list(hole),

                        "tip_pixel":
                            list(tip_center),

                        "error_pixel":
                            [
                                du,
                                dv
                            ],

                        "mapping":
                            {
                                "u":
                                    "robot_y",
                                "v":
                                    "robot_z"
                            }
                    }

                    save_path.write_text(
                        json.dumps(
                            data,
                            indent=2
                        )
                    )

                    print(
                        f"[WRIST SAVE] "
                        f"{save_path}",
                        flush=True
                    )

            prev_gray = (
                gray.copy()
            )

    finally:

        pipeline.stop()

        cv2.destroyAllWindows()

        print(
            "[WRIST CAMERA] stopped",
            flush=True
        )


if __name__ == "__main__":
    main()
