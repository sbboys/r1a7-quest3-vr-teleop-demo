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
# Live tweezer-tip perception
# ============================================================

TWEEZER_DIR = (
    ROOT
    /
    "vision_dual_camera"
    /
    "task6_threading"
    /
    "perception"
    /
    "tweezer_tip"
)

SAM2_REPO = (
    ROOT
    /
    "third_party"
    /
    "sam2"
)

sys.path.insert(
    0,
    str(TWEEZER_DIR)
)

sys.path.insert(
    0,
    str(SAM2_REPO)
)

from sam2_pca_detector import (
    Sam2PcaTweezerDetector,
)

from tip_refiner_v2 import (
    TweezerTipRefinerV2,
)


SAM2_CHECKPOINT = (
    ROOT
    /
    "third_party"
    /
    "sam2"
    /
    "checkpoints"
    /
    "sam2.1_hiera_small.pt"
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

# Conservative settings for first live-tip IBVS run.
KP_Y = 0.010
KP_Z = 0.010

MAX_SPEED = 5.0

DEADBAND_U_PX = 3.0
DEADBAND_V_PX = 3.0

TIP_EMA_ALPHA = 0.4

# Strict normal tracking gate.
TIP_MAX_JUMP_PX = 8.0

# Automatic relock:
# A candidate outside the strict 8 px gate is never used
# immediately for robot control. It must remain spatially
# consistent for 3 consecutive valid perception frames.
TIP_RELOCK_FRAMES = 3
TIP_RELOCK_CLUSTER_RADIUS_PX = 5.0

# Wrist tip short-time hold
TIP_HOLD_S = 0.30



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
hole_ema = None

# Wrist hole motion hold state
hole_last_valid_center = None
hole_last_valid_time = None
hole_velocity = np.zeros(2, dtype=np.float64)

# Brief Hough-circle misses must not immediately disable
# the whole wrist visual-servo source.
#
# Reuse only the most recent CONFIRMED hole center and only
# for this bounded grace interval.
HOLE_HOLD_S = 0.15

# Wrist hole stability
HOLE_MAX_JUMP_PX = 20.0
HOLE_EMA_ALPHA = 0.25
hole_last_valid_time = None
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
    dv,
    tip_pixel=None,
    hole_pixel=None,
    tip_status="LOST",
    hole_status="LOST",
):

    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "enabled": True,

                "tweezer_tip_pixel":
                    (
                        list(tip_pixel)
                        if tip_pixel is not None
                        else None
                    ),

                "hole_pixel":
                    (
                        list(hole_pixel)
                        if hole_pixel is not None
                        else None
                    ),

                "tip_state":
                    str(tip_status),

                "hole_state":
                    str(hole_status),

                "tip_confidence":
                    (
                        1.0
                        if tip_pixel is not None
                        else 0.0
                    ),

                "hole_confidence":
                    (
                        1.0
                        if hole_pixel is not None
                        else 0.0
                    ),

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
    global hole_ema
    global hole_last_valid_center
    global hole_last_valid_time
    global hole_velocity
    global hole_radius
    global reselect_mode

    hole_roi = None
    hole_click = None

    select_done = False

    hole_center = None
    hole_ema = None
    hole_last_valid_center = None
    hole_last_valid_time = None
    hole_velocity = np.zeros(2, dtype=np.float64)
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

def detect_hole_ellipse(crop, offset_x=0, offset_y=0):

    edges = cv2.Canny(
        crop,
        40,
        120
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates=[]

    for cnt in contours:

        area=cv2.contourArea(cnt)

        if area < 2 or area > 500:
            continue

        if len(cnt) < 5:
            continue

        ellipse=cv2.fitEllipse(cnt)

        (cx,cy),(ma,mi),angle=ellipse

        if ma <=0 or mi <=0:
            continue

        ratio=min(ma,mi)/max(ma,mi)

        # keep approximately circular holes
        if ratio < 0.4:
            continue

        candidates.append(
            (
                cx+offset_x,
                cy+offset_y,
                (ma+mi)/4
            )
        )

    if not candidates:
        return None

    # choose smallest stable hole candidate
    candidates.sort(
        key=lambda x:x[2]
    )

    return candidates[0]


# If several holes are detected inside the ROI,
# choose the one nearest to the mouse click.
# ============================================================

def init_hole(gray):

    global hole_center
    global hole_last_valid_time
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

    ellipse_hole = detect_hole_ellipse(
        crop,
        x0,
        y0
    )

    if ellipse_hole is not None:

        hole_center = (
            int(round(ellipse_hole[0])),
            int(round(ellipse_hole[1]))
        )

        hole_radius = int(
            round(ellipse_hole[2])
        )

        hole_last_valid_time = time.time()

        reselect_mode = False

        print(
            "[HOLE INIT ELLIPSE] "
            f"H={hole_center} "
            f"radius={hole_radius}",
            flush=True
        )

        return True


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

    hole_last_valid_time = time.time()

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
    global hole_last_valid_center
    global hole_last_valid_time
    global hole_velocity
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

    ellipse_hole = detect_hole_ellipse(
        crop,
        x0,
        y0
    )

    if ellipse_hole is not None:

        global hole_ema

        candidate = np.array(
            [
                ellipse_hole[0],
                ellipse_hole[1]
            ],
            dtype=np.float64,
        )

        previous = np.array(
            [
                prev_x,
                prev_y
            ],
            dtype=np.float64,
        )

        jump = float(
            np.linalg.norm(
                candidate - previous
            )
        )

        # Reject false ellipse detections
        if jump > HOLE_MAX_JUMP_PX:

            return hole_center


        # Reject invalid ellipse results
        if not np.all(
            np.isfinite(candidate)
        ):
            return hole_center


        if (
            hole_ema is not None
            and
            not np.all(
                np.isfinite(hole_ema)
            )
        ):
            hole_ema = None


        if hole_ema is None:

            hole_ema = candidate.copy()

        else:

            hole_ema = (
                HOLE_EMA_ALPHA * candidate
                +
                (1.0 - HOLE_EMA_ALPHA)
                *
                hole_ema
            )


        if not np.all(
            np.isfinite(hole_ema)
        ):
            return hole_center


        hole_center = (
            int(round(hole_ema[0])),
            int(round(hole_ema[1]))
        )

        now_hole_time = time.time()

        if (
            hole_last_valid_center is not None
            and
            hole_last_valid_time is not None
        ):

            dt = max(
                now_hole_time - hole_last_valid_time,
                1e-3
            )

            hole_velocity = (
                np.array(
                    hole_center,
                    dtype=np.float64
                )
                -
                hole_last_valid_center
            ) / dt


        hole_last_valid_center = np.array(
            hole_center,
            dtype=np.float64
        )

        hole_last_valid_time = now_hole_time

        return hole_center


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

        if (
            hole_last_valid_center is not None
            and
            hole_last_valid_time is not None
        ):

            lost_time = (
                time.time()
                -
                hole_last_valid_time
            )

            if lost_time <= HOLE_HOLD_S:

                predicted = (
                    hole_last_valid_center
                    +
                    hole_velocity
                    *
                    lost_time
                )

                if np.all(
                    np.isfinite(predicted)
                ):

                    hole_center = (
                        int(round(predicted[0])),
                        int(round(predicted[1]))
                    )

                    return hole_center


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

        if (
            hole_last_valid_center is not None
            and
            hole_last_valid_time is not None
        ):

            lost_time = (
                time.time()
                -
                hole_last_valid_time
            )

            if lost_time <= HOLE_HOLD_S:

                predicted = (
                    hole_last_valid_center
                    +
                    hole_velocity
                    *
                    lost_time
                )

                if np.all(
                    np.isfinite(predicted)
                ):

                    hole_center = (
                        int(round(predicted[0])),
                        int(round(predicted[1]))
                    )

                    return hole_center


        return None
    hole_center = (
        int(best[0]),
        int(best[1])
    )

    hole_last_valid_time = time.time()

    return hole_center


# ============================================================
# Live tweezer-tip initialization
# ============================================================

def select_tip_positive_points(image):

    points = []

    win = "SELECT TWO TWEEZER SAM2 POINTS"

    cv2.namedWindow(win)

    def callback(
        event,
        x,
        y,
        flags,
        param,
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            if len(points) < 2:

                points.append(
                    (
                        int(x),
                        int(y),
                    )
                )

        elif event == cv2.EVENT_RBUTTONDOWN:

            if points:

                points.pop()

    cv2.setMouseCallback(
        win,
        callback,
    )

    while True:

        vis = image.copy()

        for i, pt in enumerate(points):

            cv2.circle(
                vis,
                pt,
                7,
                (0, 255, 255),
                -1,
            )

            cv2.putText(
                vis,
                f"P{i + 1}",
                (
                    pt[0] + 10,
                    pt[1] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            vis,
            "LEFT: P1/P2   RIGHT: undo",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            "ENTER: confirm   ESC: cancel",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow(
            win,
            vis,
        )

        key = cv2.waitKey(20) & 0xFF

        if (
            key in (13, 10)
            and
            len(points) == 2
        ):
            break

        if key == 27:

            cv2.destroyWindow(win)

            return None

    cv2.destroyWindow(win)

    return points


def initialize_live_tip(frame):

    roi_win = "SELECT TWEEZER ROI"

    roi = cv2.selectROI(
        roi_win,
        frame,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(
        roi_win
    )

    x, y, w, h = roi

    if (
        w <= 0
        or
        h <= 0
    ):

        return None

    box = [
        int(x),
        int(y),
        int(x + w),
        int(y + h),
    ]

    positive_points = (
        select_tip_positive_points(
            frame
        )
    )

    if positive_points is None:

        return None

    point_array = np.asarray(
        positive_points,
        dtype=np.float64,
    )

    hint_array = np.mean(
        point_array,
        axis=0,
    )

    tip_hint = (
        float(hint_array[0]),
        float(hint_array[1]),
    )

    return (
        box,
        positive_points,
        tip_hint,
    )


# ============================================================
# Main
# ============================================================

def main():

    global last_print
    global reselect_mode
    global tip_pixel

    # Never leave an old velocity active while perception starts.
    write_disabled_output()

    tip_pixel = None

    print(
        "[TIP] Loading SAM2 detector...",
        flush=True,
    )

    detector = (
        Sam2PcaTweezerDetector(
            checkpoint_path=
            SAM2_CHECKPOINT
        )
    )

    refiner = (
        TweezerTipRefinerV2()
    )

    tip_ema = None

    # Wrist tip hold state
    wrist_last_valid_tip = None
    wrist_last_valid_time = None
    wrist_tip_velocity = np.zeros(2, dtype=np.float64)

    # Candidates used only for automatic recovery after the
    # strict 8 px normal gate rejects the current detection.
    relock_candidates = []

    print(
        "[TIP] SAM2 + RefinerV2 ready",
        flush=True,
    )

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

    # --------------------------------------------------------
    # Initialize live tweezer tracking from one camera frame.
    # --------------------------------------------------------

    init_frame = None

    print(
        "",
        flush=True,
    )

    print(
        "[TIP INIT] Waiting for wrist image...",
        flush=True,
    )

    while init_frame is None:

        frames = (
            pipeline.wait_for_frames(
                1000
            )
        )

        if frames is None:
            continue

        color = (
            frames.get_color_frame()
        )

        if color is None:
            continue

        init_frame = (
            hc.color_frame_to_bgr(
                color
            )
        )

    live_tip_init = (
        initialize_live_tip(
            init_frame
        )
    )

    if live_tip_init is None:

        write_disabled_output()

        pipeline.stop()

        cv2.destroyAllWindows()

        print(
            "[TIP INIT] Cancelled.",
            flush=True,
        )

        return

    (
        tip_box,
        tip_positive_points,
        tip_hint,
    ) = live_tip_init

    print(
        "[TIP INIT] ROI =",
        tip_box,
        flush=True,
    )

    print(
        "[TIP INIT] SAM2 points =",
        tip_positive_points,
        flush=True,
    )

    print(
        "[TIP INIT] PCA hint =",
        tip_hint,
        flush=True,
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
        "TASK6 WRIST IBVS + LIVE TWEEZER TIP",
        flush=True
    )

    print(
        "==========================================",
        flush=True
    )

    print(
        "Tip mode  : LIVE SAM2 + PCA + RefinerV2",
        flush=True
    )

    print(
        f"Tip ROI   : {tip_box}",
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

                write_disabled_output()

                continue

            color = frames.get_color_frame()

            if color is None:

                write_disabled_output()

                continue

            frame = hc.color_frame_to_bgr(
                color
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # ------------------------------------------------
            # LIVE TWEEZER TIP
            # ------------------------------------------------

            tip_pixel = None
            tip_raw_pixel = None
            tip_status = "LOST"

            try:

                coarse_tip = (
                    detector.detect(
                        frame,
                        box_xyxy=
                        tip_box,
                        tip_hint=
                        tip_hint,
                        positive_points=
                        tip_positive_points,
                    )
                )

                if coarse_tip.get(
                    "valid",
                    False,
                ):

                    refined_tip = (
                        refiner.refine(
                            frame,
                            coarse_tip["mask"],
                            coarse_tip["tip_uv"],
                            coarse_tip["direction"],
                        )
                    )

                    if refined_tip.get(
                        "valid",
                        False,
                    ):

                        raw_tip = np.asarray(
                            refined_tip[
                                "refined_tip_uv"
                            ],
                            dtype=np.float64,
                        )

                        tip_raw_pixel = (
                            int(
                                round(
                                    raw_tip[0]
                                )
                            ),
                            int(
                                round(
                                    raw_tip[1]
                                )
                            ),
                        )

                        tip_accept = False

                        if tip_ema is None:

                            tip_ema = (
                                raw_tip.copy()
                            )

                            relock_candidates.clear()

                            tip_accept = True

                            tip_status = "INIT"

                        else:

                            tip_jump = float(
                                np.linalg.norm(
                                    raw_tip
                                    -
                                    tip_ema
                                )
                            )

                            # ====================================
                            # NORMAL TRACKING
                            #
                            # Keep the original strict 8 px gate.
                            # ====================================

                            if (
                                tip_jump
                                <=
                                TIP_MAX_JUMP_PX
                            ):

                                tip_ema = (
                                    TIP_EMA_ALPHA
                                    *
                                    raw_tip
                                    +
                                    (
                                        1.0
                                        -
                                        TIP_EMA_ALPHA
                                    )
                                    *
                                    tip_ema
                                )

                                relock_candidates.clear()

                                tip_accept = True

                                tip_status = (
                                    f"OK jump="
                                    f"{tip_jump:.1f}"
                                )

                            # ====================================
                            # AUTO RELOCK
                            #
                            # Do NOT use a single large jump.
                            # Require three consecutive refined
                            # candidates to form a tight cluster.
                            # ====================================

                            else:

                                relock_candidates.append(
                                    raw_tip.copy()
                                )

                                if (
                                    len(relock_candidates)
                                    >
                                    TIP_RELOCK_FRAMES
                                ):
                                    del relock_candidates[
                                        :-
                                        TIP_RELOCK_FRAMES
                                    ]

                                tip_status = (
                                    f"RELOCK WAIT "
                                    f"{len(relock_candidates)}/"
                                    f"{TIP_RELOCK_FRAMES} "
                                    f"jump={tip_jump:.1f}"
                                )

                                if (
                                    len(relock_candidates)
                                    ==
                                    TIP_RELOCK_FRAMES
                                ):

                                    relock_array = np.stack(
                                        relock_candidates,
                                        axis=0,
                                    )

                                    relock_center = np.mean(
                                        relock_array,
                                        axis=0,
                                    )

                                    relock_distances = (
                                        np.linalg.norm(
                                            relock_array
                                            -
                                            relock_center,
                                            axis=1,
                                        )
                                    )

                                    relock_spread = float(
                                        np.max(
                                            relock_distances
                                        )
                                    )

                                    if (
                                        relock_spread
                                        <=
                                        TIP_RELOCK_CLUSTER_RADIUS_PX
                                    ):

                                        tip_ema = (
                                            relock_center.copy()
                                        )

                                        relock_candidates.clear()

                                        tip_accept = True

                                        tip_status = (
                                            f"RELOCK "
                                            f"spread="
                                            f"{relock_spread:.1f}"
                                        )

                                        print(
                                            "[TIP RELOCK] "
                                            f"tip="
                                            f"({tip_ema[0]:.1f},"
                                            f"{tip_ema[1]:.1f}) "
                                            f"spread="
                                            f"{relock_spread:.1f}px",
                                            flush=True,
                                        )

                                    else:

                                        tip_status = (
                                            f"RELOCK WAIT "
                                            f"{TIP_RELOCK_FRAMES}/"
                                            f"{TIP_RELOCK_FRAMES} "
                                            f"spread="
                                            f"{relock_spread:.1f}"
                                        )

                        if tip_accept:

                            # Update wrist tip motion model
                            now_tip_time = time.time()

                            if (
                                wrist_last_valid_tip is not None
                                and
                                wrist_last_valid_time is not None
                            ):

                                dt = max(
                                    now_tip_time - wrist_last_valid_time,
                                    1e-3
                                )

                                wrist_tip_velocity = (
                                    tip_ema
                                    -
                                    wrist_last_valid_tip
                                ) / dt

                            wrist_last_valid_tip = tip_ema.copy()
                            wrist_last_valid_time = now_tip_time

                            # Feed the confirmed filtered tip back
                            # as the hint for the next frame.
                            tip_hint = (
                                float(tip_ema[0]),
                                float(tip_ema[1]),
                            )

                            tip_pixel = (
                                int(
                                    round(
                                        tip_ema[0]
                                    )
                                ),
                                int(
                                    round(
                                        tip_ema[1]
                                    )
                                ),
                            )

                    else:

                        relock_candidates.clear()

                        # ====================================
                        # WRIST TIP HOLD
                        #
                        # Refiner can lose the thin tweezer
                        # tip for a few frames. Use the last
                        # valid tip and motion prediction
                        # instead of immediately disabling IBVS.
                        # ====================================

                        if (
                            wrist_last_valid_tip is not None
                            and
                            wrist_last_valid_time is not None
                        ):

                            lost_time = (
                                time.time()
                                -
                                wrist_last_valid_time
                            )

                            if lost_time < TIP_HOLD_S:

                                hold_tip = (
                                    wrist_last_valid_tip
                                    +
                                    wrist_tip_velocity
                                    *
                                    lost_time
                                )

                                tip_ema = hold_tip.copy()

                                tip_pixel = (
                                    int(round(float(hold_tip[0]))),
                                    int(round(float(hold_tip[1])))
                                )

                                tip_hint = (
                                    float(hold_tip[0]),
                                    float(hold_tip[1]),
                                )

                                tip_status = (
                                    f"HOLD {lost_time:.3f}s"
                                )

                            else:

                                tip_status = (
                                    "REFINER LOST"
                                )

                        else:

                            tip_status = (
                                "REFINER LOST"
                            )

                else:

                    relock_candidates.clear()

                    tip_status = (
                        "SAM2 LOST"
                    )

            except Exception as exc:

                tip_pixel = None
                relock_candidates.clear()

                tip_status = (
                    "TIP ERROR: "
                    +
                    type(exc).__name__
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
            # Draw live tweezer tip
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
                    "LIVE TIP",
                    (
                        tip_pixel[0] + 15,
                        tip_pixel[1] - 15
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            # Raw V2 point for debugging only.
            if tip_raw_pixel is not None:

                cv2.circle(
                    img,
                    tip_raw_pixel,
                    4,
                    (255, 255, 0),
                    -1,
                )

            cv2.putText(
                img,
                f"TIP: {tip_status}",
                (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (
                    (0, 255, 0)
                    if tip_pixel is not None
                    else
                    (0, 0, 255)
                ),
                2,
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

                du_cmd = (
                    0.0
                    if
                    abs(du)
                    <=
                    DEADBAND_U_PX
                    else
                    du
                )

                dv_cmd = (
                    0.0
                    if
                    abs(dv)
                    <=
                    DEADBAND_V_PX
                    else
                    dv
                )

                vy = limit(
                    -KP_Y
                    *
                    du_cmd
                )

                vz = limit(
                    KP_Z
                    *
                    (-dv_cmd)
                )

                hole_status = (
                    "TRACKING"
                    if H is not None
                    else "LOST"
                )

                write_velocity_output(
                    vy,
                    vz,
                    du,
                    dv,
                    tip_pixel,
                    H,
                    tip_status,
                    hole_status
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

                now = time.time()

                if now - last_print > 0.2:

                    print(
                        "[WRIST LOST] "
                        f"hole={'OK' if H is not None else 'LOST'} "
                        f"tip_status={tip_status} "
                        f"tip_raw={tip_raw_pixel}",
                        flush=True,
                    )

                    last_print = now

                cv2.putText(
                    img,
                    "TIP OR TARGET TRACKING LOST",
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
            # M -> reinitialize LIVE tweezer tip
            # ------------------------------------------------

            if (
                key == ord("m")
                or
                key == ord("M")
            ):

                write_disabled_output()

                print(
                    "",
                    flush=True,
                )

                print(
                    "[TIP] Reinitializing live tip...",
                    flush=True,
                )

                new_tip_init = (
                    initialize_live_tip(
                        frame
                    )
                )

                if new_tip_init is not None:

                    (
                        tip_box,
                        tip_positive_points,
                        tip_hint,
                    ) = new_tip_init

                    tip_ema = None

                    wrist_last_valid_tip = None
                    wrist_last_valid_time = None
                    wrist_tip_velocity = np.zeros(2, dtype=np.float64)

                    relock_candidates.clear()
                    tip_pixel = None

                    print(
                        "[TIP] Reinitialized.",
                        flush=True,
                    )

                    print(
                        "[TIP] ROI =",
                        tip_box,
                        flush=True,
                    )

                else:

                    print(
                        "[TIP] Reinitialization cancelled.",
                        flush=True,
                    )

                continue

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
