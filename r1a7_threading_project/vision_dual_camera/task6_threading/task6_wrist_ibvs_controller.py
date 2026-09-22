#!/usr/bin/env python3

import sys
import cv2
import json
import time
import numpy as np
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(
        ROOT /
        "vision_dual_camera" /
        "handeye"
    )
)

import collect_handeye_samples as hc

from pyorbbecsdk import (
    Config,
    OBFormat,
    OBSensorType,
    Pipeline,
)


# ============================================================
# Files
# ============================================================

TIP_FILE = (
    ROOT /
    "vision_dual_camera" /
    "task6_threading" /
    "wrist_tip_calibration.json"
)

OUTPUT_FILE = Path(
    "/tmp/r1a7_wrist_ibvs_velocity.json"
)


# ============================================================
# Wrist PID
#
# Wrist image mapping already determined:
#
#   image U -> robot Y
#   image V -> robot Z
#
# Current Y sign has already been experimentally corrected:
#
#   du > 0 -> vy < 0
#
# Current Z:
#
#   dv < 0 -> vz > 0
# ============================================================

KP_Y = 0.015
KP_Z = 0.015

MAX_SPEED = 5.0


def limit(v):

    return max(
        -MAX_SPEED,
        min(
            MAX_SPEED,
            v
        )
    )


# ============================================================
# Global state
# ============================================================

hole_roi = None

hole_click = None

select_done = False

hole_center = None

hole_radius = None

tip_pixel = None

last_print = 0.0

reselect_mode = True

tip_reselect_mode = False


# ============================================================
# Output helpers
# ============================================================

def write_disabled_output():

    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "enabled": False,
                "vy_mm_s": 0.0,
                "vz_mm_s": 0.0,
                "du": None,
                "dv": None,
                "timestamp": time.time()
            },
            indent=2
        )
    )


def write_velocity_output(
    vy,
    vz,
    du,
    dv
):

    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "enabled": True,
                "vy_mm_s": float(vy),
                "vz_mm_s": float(vz),
                "du": float(du),
                "dv": float(dv),
                "timestamp": time.time()
            },
            indent=2
        )
    )


# ============================================================
# Enter target re-selection mode
# ============================================================

def enter_reselect_mode():

    global hole_roi
    global hole_click
    global select_done
    global hole_center
    global hole_radius
    global reselect_mode

    hole_roi = None
    hole_click = None

    select_done = False

    hole_center = None
    hole_radius = None

    reselect_mode = True

    write_disabled_output()

    print(
        "",
        flush=True
    )

    print(
        "==========================================",
        flush=True
    )

    print(
        "[WRIST TARGET RESELECT]",
        flush=True
    )

    print(
        "Click the NEW target hole in wrist image.",
        flush=True
    )

    print(
        "Robot wrist IBVS output is temporarily zero.",
        flush=True
    )

    print(
        "==========================================",
        flush=True
    )


# ============================================================
# Mouse callback
# ============================================================


def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global select_done
    global hole_roi
    global hole_click

    global tip_pixel
    global tip_reselect_mode


    if event != cv2.EVENT_LBUTTONDOWN:
        return


    # ========================================================
    # Priority 1:
    # Tip recalibration
    # ========================================================

    if tip_reselect_mode:


        tip_pixel = (
            int(x),
            int(y)
        )


        data = {

            "tip_pixel":
                list(tip_pixel),

            "camera":
                hc.RIGHT_WRIST_SN,

            "timestamp":
                time.time()

        }


        TIP_FILE.write_text(
            json.dumps(
                data,
                indent=2
            )
        )


        tip_reselect_mode=False


        print(
            "[TIP UPDATED]",
            tip_pixel,
            flush=True
        )


        return



    # ========================================================
    # Priority 2:
    # Hole selection
    # ========================================================


    if select_done:
        return


    size=60


    hole_click=(
        int(x),
        int(y)
    )


    hole_roi=(

        int(x-size),
        int(y-size),
        int(x+size),
        int(y+size)

    )


    select_done=True


    print(
        f"[SELECT NEW HOLE] "
        f"click=({x},{y}) "
        f"roi={hole_roi}",
        flush=True
    )



# ============================================================
# Load fixed tweezer tip
# ============================================================

def load_tip():

    global tip_pixel

    if not TIP_FILE.exists():

        raise RuntimeError(
            f"Tip calibration file not found: "
            f"{TIP_FILE}"
        )

    data = json.loads(
        TIP_FILE.read_text()
    )

    tip_pixel = tuple(
        int(v)
        for v in data["tip_pixel"]
    )

    print(
        "[TIP FIXED]",
        tip_pixel,
        flush=True
    )


# ============================================================
# Hole initialization
#
# Important:
# If several holes are detected inside the ROI,
# choose the one nearest to the mouse click.
# ============================================================

def init_hole(gray):

    global hole_center
    global hole_radius
    global reselect_mode

    if hole_roi is None:
        return False

    if hole_click is None:
        return False

    x0, y0, x1, y1 = hole_roi

    h, w = gray.shape

    x0 = max(
        0,
        x0
    )

    y0 = max(
        0,
        y0
    )

    x1 = min(
        w,
        x1
    )

    y1 = min(
        h,
        y1
    )

    if x1 <= x0 or y1 <= y0:
        return False

    crop = gray[
        y0:y1,
        x0:x1
    ]

    if crop.size == 0:
        return False

    blur = cv2.GaussianBlur(
        crop,
        (5, 5),
        0
    )

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=80,
        param2=12,
        minRadius=5,
        maxRadius=40
    )

    if circles is None:
        return False

    candidates = np.round(
        circles[0]
    ).astype(int)

    click_x, click_y = hole_click

    best_circle = None
    best_dist2 = None

    for c in candidates:

        cx = int(
            c[0] + x0
        )

        cy = int(
            c[1] + y0
        )

        r = int(
            c[2]
        )

        dist2 = (
            (cx - click_x) ** 2
            +
            (cy - click_y) ** 2
        )

        if (
            best_dist2 is None
            or
            dist2 < best_dist2
        ):

            best_dist2 = dist2

            best_circle = (
                cx,
                cy,
                r
            )

    if best_circle is None:
        return False

    hole_center = (
        int(best_circle[0]),
        int(best_circle[1])
    )

    hole_radius = int(
        best_circle[2]
    )

    reselect_mode = False

    print(
        "[HOLE INIT] "
        f"H={hole_center} "
        f"radius={hole_radius} "
        f"click={hole_click}",
        flush=True
    )

    print(
        "[WRIST IBVS] NEW TARGET LOCKED",
        flush=True
    )

    return True


# ============================================================
# Local target-hole tracking
#
# Search only around previous hole.
# If multiple circles are found, choose the one nearest to
# the previous target position.
#
# If no circle is detected in the current frame:
# return None and output zero wrist velocity.
# ============================================================

def track_hole(gray):

    global hole_center

    if hole_center is None:
        return None

    if hole_radius is None:
        return None

    prev_x, prev_y = hole_center

    search = 40

    x0 = max(
        0,
        prev_x - search
    )

    y0 = max(
        0,
        prev_y - search
    )

    x1 = min(
        gray.shape[1],
        prev_x + search
    )

    y1 = min(
        gray.shape[0],
        prev_y + search
    )

    if x1 <= x0 or y1 <= y0:
        return None

    crop = gray[
        y0:y1,
        x0:x1
    ]

    if crop.size == 0:
        return None

    blur = cv2.GaussianBlur(
        crop,
        (5, 5),
        0
    )

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,
        param1=80,
        param2=12,
        minRadius=max(
            5,
            hole_radius - 5
        ),
        maxRadius=hole_radius + 5
    )

    if circles is None:
        return None

    candidates = np.round(
        circles[0]
    ).astype(int)

    best = None
    best_dist2 = None

    for c in candidates:

        cx = int(
            c[0] + x0
        )

        cy = int(
            c[1] + y0
        )

        dist2 = (
            (cx - prev_x) ** 2
            +
            (cy - prev_y) ** 2
        )

        if (
            best_dist2 is None
            or
            dist2 < best_dist2
        ):

            best_dist2 = dist2

            best = (
                cx,
                cy
            )

    if best is None:
        return None

    hole_center = (
        int(best[0]),
        int(best[1])
    )

    return hole_center


# ============================================================
# Main
# ============================================================

def main():

    global last_print
    global reselect_mode

    load_tip()

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

    profile = (
        profiles
        .get_video_stream_profile(
            hc.COLOR_WIDTH,
            hc.COLOR_HEIGHT,
            OBFormat.RGB,
            hc.COLOR_FPS
        )
    )

    config.enable_stream(
        profile
    )

    pipeline.start(
        config
    )

    print(
        "",
        flush=True
    )

    print(
        "==========================================",
        flush=True
    )

    print(
        "TASK6 WRIST IBVS CONTROLLER",
        flush=True
    )

    print(
        "==========================================",
        flush=True
    )

    print(
        f"Fixed tip : {tip_pixel}",
        flush=True
    )

    print(
        "",
        flush=True
    )

    print(
        "Controls:",
        flush=True
    )

    print(
        "  Left click : select target hole",
        flush=True
    )

    print(
        "  N          : change/reselect target hole",
        flush=True
    )

    print(
        "  Q / ESC    : quit",
        flush=True
    )

    print(
        "",
        flush=True
    )

    print(
        "Initial state: waiting for target hole.",
        flush=True
    )

    write_disabled_output()

    win = "WRIST IBVS"

    cv2.namedWindow(
        win
    )

    cv2.setMouseCallback(
        win,
        mouse_callback
    )

    initialized = False

    try:

        while True:

            frames = pipeline.wait_for_frames(
                1000
            )

            if frames is None:
                continue

            color = frames.get_color_frame()

            if color is None:
                continue

            frame = hc.color_frame_to_bgr(
                color
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # ------------------------------------------------
            # New target selected -> initialize target
            # ------------------------------------------------

            if (
                select_done
                and
                not initialized
            ):

                initialized = init_hole(
                    gray
                )

                if not initialized:

                    write_disabled_output()

            # ------------------------------------------------
            # Track selected hole
            # ------------------------------------------------

            if initialized:

                H = track_hole(
                    gray
                )

            else:

                H = None

            img = frame.copy()

            # ------------------------------------------------
            # Draw fixed tweezer tip
            # ------------------------------------------------

            if tip_pixel is not None:

                cv2.drawMarker(
                    img,
                    tip_pixel,
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    30,
                    2
                )

                cv2.putText(
                    img,
                    "TIP",
                    (
                        tip_pixel[0] + 15,
                        tip_pixel[1] - 15
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            # ------------------------------------------------
            # Draw target hole
            # ------------------------------------------------

            if H is not None:

                cv2.drawMarker(
                    img,
                    H,
                    (0, 255, 0),
                    cv2.MARKER_CROSS,
                    30,
                    2
                )

                if hole_radius is not None:

                    cv2.circle(
                        img,
                        H,
                        int(hole_radius),
                        (0, 255, 0),
                        2
                    )

                cv2.putText(
                    img,
                    "TARGET",
                    (
                        H[0] + 15,
                        H[1] - 15
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # ------------------------------------------------
            # Waiting for a new target
            # ------------------------------------------------

            if not initialized:

                cv2.putText(
                    img,
                    "CLICK TARGET HOLE",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    img,
                    "N = reselect target",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            # ------------------------------------------------
            # Target tracking valid -> PID
            # ------------------------------------------------

            if (
                H is not None
                and
                tip_pixel is not None
            ):

                du = float(
                    H[0]
                    -
                    tip_pixel[0]
                )

                dv = float(
                    H[1]
                    -
                    tip_pixel[1]
                )

                # ------------------------------------------------
                # Wrist PID
                #
                # Y direction sign already experimentally fixed:
                #
                #   du > 0 -> vy < 0
                #
                # Z:
                #
                #   dv < 0 -> vz > 0
                # ------------------------------------------------

                vy = limit(
                    -KP_Y * du
                )

                vz = limit(
                    KP_Z * (-dv)
                )

                write_velocity_output(
                    vy,
                    vz,
                    du,
                    dv
                )

                cv2.line(
                    img,
                    tip_pixel,
                    H,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    (
                        f"du={du:+.0f}px "
                        f"dv={dv:+.0f}px"
                    ),
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    (
                        f"vy={vy:+.3f} "
                        f"vz={vz:+.3f} mm/s"
                    ),
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    "N = CHANGE TARGET",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                now = time.time()

                if now - last_print > 0.2:

                    print(
                        "[WRIST IBVS] "
                        f"H={H} "
                        f"T={tip_pixel} "
                        f"du={du:+.1f} "
                        f"dv={dv:+.1f} "
                        f"vy={vy:+.3f} "
                        f"vz={vz:+.3f}",
                        flush=True
                    )

                    last_print = now

            # ------------------------------------------------
            # Tracking lost
            # ------------------------------------------------

            elif initialized:

                write_disabled_output()

                cv2.putText(
                    img,
                    "TARGET TEMPORARILY NOT DETECTED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    img,
                    "N = SELECT ANOTHER HOLE",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            # ------------------------------------------------
            # Show
            # ------------------------------------------------

            cv2.imshow(
                win,
                img
            )

            key = cv2.waitKey(1) & 0xFF

            # ------------------------------------------------
            # N -> switch target
            # ------------------------------------------------

            if (
                key == ord("t")
                or
                key == ord("T")
            ):

                global tip_reselect_mode

                tip_reselect_mode=True


                print(
                    "",
                    flush=True
                )

                print(
                    "[TIP RESELECT]",
                    flush=True
                )

                print(
                    "Click fixed tweezer tip",
                    flush=True
                )


            elif (
                key == ord("n")
                or
                key == ord("N")
            ):

                initialized = False

                enter_reselect_mode()

            # ------------------------------------------------
            # Quit
            # ------------------------------------------------

            elif (
                key == ord("q")
                or
                key == ord("Q")
                or
                key == 27
            ):

                break

    except KeyboardInterrupt:

        print(
            "",
            flush=True
        )

        print(
            "[WRIST IBVS] Ctrl+C",
            flush=True
        )

    finally:

        write_disabled_output()

        pipeline.stop()

        cv2.destroyAllWindows()

        print(
            "[WRIST IBVS] stopped",
            flush=True
        )


if __name__ == "__main__":
    main()
