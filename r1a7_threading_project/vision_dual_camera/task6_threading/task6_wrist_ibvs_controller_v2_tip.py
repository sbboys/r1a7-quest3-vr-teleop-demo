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
KP_Y = 0.150
KP_Z = 0.150

MAX_SPEED = 50.0

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

# A relock cluster must also remain close to the last reliable
# Tip trajectory.  This specifically rejects the observed
# false jump from roughly (840,575) toward (880,644).
TIP_RELOCK_MAX_ANCHOR_DIST_PX = 45.0

# Tip motion model is used only for short-term gate prediction.
# It is NOT itself treated as a visual observation.
TIP_VELOCITY_ALPHA = 0.25
TIP_PREDICT_MAX_SPEED_PX_S = 250.0

# Wrist tip short-time hold
TIP_HOLD_S = 0.30




# Heavy coarse Tip detector runs at reduced frequency.
# Intermediate frames use Tip motion prediction.
TIP_DETECT_EVERY_N_FRAMES = 2
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
HOLE_HOLD_S = 0.30

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
    global hole_ema

    if hole_center is None:
        return None

    if hole_radius is None:
        return None

    # ========================================================
    # Hole tracker configuration
    # ========================================================

    LOCAL_SEARCH = 80

    # Short local prediction period.
    # HOLE_HOLD_S itself remains unchanged.

    # --------------------------------------------------------
    # Global template reacquisition
    # --------------------------------------------------------

    GLOBAL_TM_SCALE = 0.75
    GLOBAL_TM_PERIOD_S = 0.10

    # Initial conservative template threshold.
    GLOBAL_TM_MIN_SCORE = 0.72

    # Local ellipse/Hough measurements must still look like
    # the manually initialized target Hole.
    #
    # Deliberately lower than GLOBAL_TM_MIN_SCORE because the
    # local target is moving and may have perspective change.
    LOCAL_TM_MIN_SCORE = 0.60

    # Do not require the geometric center to align perfectly
    # with the template center. Search around it locally.
    LOCAL_TM_SEARCH_MARGIN = 10

    # Hole image velocity is used ONLY to predict the search
    # position. Filter and clamp it to avoid runaway ROI
    # prediction after a single noisy measurement.
    HOLE_VELOCITY_ALPHA = 0.25
    HOLE_PREDICT_MAX_SPEED_PX_S = 300.0

    # --------------------------------------------------------
    # Adaptive RECENT Hole appearance
    #
    # INITIAL template remains immutable.
    # RECENT template follows slow viewpoint/illumination
    # changes only after reliable LOCAL visual measurements.
    # --------------------------------------------------------
    RECENT_TEMPLATE_ALPHA = 0.15
    RECENT_TEMPLATE_MIN_SCORE = 0.68

    # During the first second after loss, a global candidate
    # should stay reasonably close to the predicted trajectory.
    GLOBAL_NEAR_LOSS_S = 1.0
    GLOBAL_NEAR_MOTION_LIMIT_PX = 90.0

    # Radius-aware Hough range.
    #
    # Example:
    # manually initialized radius=3 px
    # -> search roughly 2..7 px
    #
    # The old max(5, radius-5) excluded a 3 px target.
    HOUGH_MIN_RADIUS = max(
        2,
        int(round(float(hole_radius))) - 2
    )

    HOUGH_MAX_RADIUS = max(
        HOUGH_MIN_RADIUS + 2,
        int(round(float(hole_radius))) + 4
    )

    # Inspect several local maxima instead of trusting only
    # the single global maximum.
    GLOBAL_TM_TOP_K = 8

    # Template candidate must also pass local geometry.
    GLOBAL_GEOM_MAX_OFFSET_PX = 30.0

    # Three geometrically verified template observations are
    # needed before relocking.
    GLOBAL_RELOCK_FRAMES = 3
    GLOBAL_RELOCK_MAX_STEP_PX = 30.0

    # Permit one intermittent failed global frame.
    GLOBAL_MAX_MISSES = 1

    # After global relock, require two LOCAL visual
    # measurements before Hole is released back to IBVS.
    POST_RELOCK_LOCAL_FRAMES = 2

    # --------------------------------------------------------
    # Persistent template/global state
    # --------------------------------------------------------

    if not hasattr(
        track_hole,
        "_tm_template_gray"
    ):
        track_hole._tm_template_gray = None

    if not hasattr(
        track_hole,
        "_tm_template_edge"
    ):
        track_hole._tm_template_edge = None

    if not hasattr(
        track_hole,
        "_tm_recent_gray"
    ):
        track_hole._tm_recent_gray = None

    if not hasattr(
        track_hole,
        "_tm_recent_edge"
    ):
        track_hole._tm_recent_edge = None

    if not hasattr(
        track_hole,
        "_tm_recent_updates"
    ):
        track_hole._tm_recent_updates = 0

    if not hasattr(
        track_hole,
        "_tm_candidates"
    ):
        track_hole._tm_candidates = []

    if not hasattr(
        track_hole,
        "_tm_misses"
    ):
        track_hole._tm_misses = 0

    if not hasattr(
        track_hole,
        "_tm_active"
    ):
        track_hole._tm_active = False

    if not hasattr(
        track_hole,
        "_tm_last_run"
    ):
        track_hole._tm_last_run = 0.0

    if not hasattr(
        track_hole,
        "_tm_last_print"
    ):
        track_hole._tm_last_print = 0.0

    if not hasattr(
        track_hole,
        "_post_relock_remaining"
    ):
        track_hole._post_relock_remaining = 0

    now_track = time.time()

    # ========================================================
    # Capture the MANUALLY INITIALIZED Hole appearance.
    #
    # This happens only once per visual process / selected
    # target. It uses a patch centered on the manually
    # initialized hole_center.
    # ========================================================

    if track_hole._tm_template_gray is None:

        cx = int(hole_center[0])
        cy = int(hole_center[1])

        desired_half = max(
            20,
            min(
                70,
                int(
                    round(
                        float(hole_radius)
                        *
                        2.5
                    )
                )
            )
        )

        max_half = min(
            cx,
            cy,
            gray.shape[1] - 1 - cx,
            gray.shape[0] - 1 - cy,
        )

        template_half = min(
            desired_half,
            max_half
        )

        if template_half >= 12:

            tx0 = cx - template_half
            ty0 = cy - template_half
            tx1 = cx + template_half + 1
            ty1 = cy + template_half + 1

            template = gray[
                ty0:ty1,
                tx0:tx1
            ].copy()

            if (
                template.size > 0
                and
                template.shape[0] >= 20
                and
                template.shape[1] >= 20
            ):

                template = cv2.GaussianBlur(
                    template,
                    (3, 3),
                    0
                )

                track_hole._tm_template_gray = (
                    template.copy()
                )

                track_hole._tm_template_edge = (
                    cv2.Canny(
                        template,
                        50,
                        150
                    )
                )

                # INITIAL template is never overwritten.
                # RECENT starts from the same manually
                # initialized appearance and then adapts.
                track_hole._tm_recent_gray = (
                    template.copy()
                )

                track_hole._tm_recent_edge = (
                    track_hole
                    ._tm_template_edge
                    .copy()
                )

                track_hole._tm_recent_updates = 0

                cv2.imwrite(
                    "/tmp/r1a7_hole_template.png",
                    template
                )

                print(
                    "[HOLE TEMPLATE] INIT "
                    f"center=({cx},{cy}) "
                    f"radius={hole_radius} "
                    f"size="
                    f"{template.shape[1]}x"
                    f"{template.shape[0]} "
                    "saved="
                    "/tmp/r1a7_hole_template.png",
                    flush=True,
                )

    # ========================================================
    # LOCAL appearance verification
    #
    # A geometric ellipse/circle alone is not enough to renew
    # hole_last_valid_time.  The surrounding appearance must
    # also resemble the manually initialized Hole template.
    # ========================================================

    def _template_score_near(
        candidate_center,
        template_local,
    ):

        if template_local is None:
            return -1.0

        candidate_center = np.array(
            candidate_center,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                candidate_center
            )
        ):
            return -1.0

        th = int(
            template_local.shape[0]
        )

        tw = int(
            template_local.shape[1]
        )

        margin = int(
            LOCAL_TM_SEARCH_MARGIN
        )

        cx = int(
            round(
                candidate_center[0]
            )
        )

        cy = int(
            round(
                candidate_center[1]
            )
        )

        x0_tm = (
            cx
            -
            tw // 2
            -
            margin
        )

        y0_tm = (
            cy
            -
            th // 2
            -
            margin
        )

        x1_tm = (
            cx
            +
            tw // 2
            +
            margin
            +
            1
        )

        y1_tm = (
            cy
            +
            th // 2
            +
            margin
            +
            1
        )

        if (
            x0_tm < 0
            or
            y0_tm < 0
            or
            x1_tm > gray.shape[1]
            or
            y1_tm > gray.shape[0]
        ):
            return -1.0

        search_patch = gray[
            y0_tm:y1_tm,
            x0_tm:x1_tm
        ]

        if (
            search_patch.shape[0]
            <
            th
            or
            search_patch.shape[1]
            <
            tw
        ):
            return -1.0

        search_patch = cv2.GaussianBlur(
            search_patch,
            (3, 3),
            0
        )

        result = cv2.matchTemplate(
            search_patch,
            template_local,
            cv2.TM_CCOEFF_NORMED,
        )

        (
            _min_val,
            max_val,
            _min_loc,
            _max_loc,
        ) = cv2.minMaxLoc(
            result
        )

        score = float(
            max_val
        )

        if not np.isfinite(score):
            return -1.0

        return score


    def local_template_score(
        candidate_center,
    ):

        initial_score = (
            _template_score_near(
                candidate_center,
                track_hole._tm_template_gray,
            )
        )

        recent_score = (
            _template_score_near(
                candidate_center,
                track_hole._tm_recent_gray,
            )
        )

        # Either identity anchor can validate LOCAL tracking.
        # INITIAL prevents unlimited appearance drift.
        # RECENT permits gradual viewpoint changes.
        return max(
            initial_score,
            recent_score,
        )


    def update_recent_template(
        candidate_center,
    ):

        initial_template = (
            track_hole._tm_template_gray
        )

        if initial_template is None:
            return

        # Never adapt immediately after a GLOBAL relock.
        # First require normal LOCAL verification.
        if (
            track_hole._post_relock_remaining
            >
            0
        ):
            return

        score = local_template_score(
            candidate_center
        )

        if (
            score
            <
            RECENT_TEMPLATE_MIN_SCORE
        ):
            return

        th = int(
            initial_template.shape[0]
        )

        tw = int(
            initial_template.shape[1]
        )

        cx = int(
            round(
                float(
                    candidate_center[0]
                )
            )
        )

        cy = int(
            round(
                float(
                    candidate_center[1]
                )
            )
        )

        x0 = cx - tw // 2
        y0 = cy - th // 2
        x1 = x0 + tw
        y1 = y0 + th

        if (
            x0 < 0
            or
            y0 < 0
            or
            x1 > gray.shape[1]
            or
            y1 > gray.shape[0]
        ):
            return

        patch = gray[
            y0:y1,
            x0:x1
        ].copy()

        if (
            patch.shape
            !=
            initial_template.shape
        ):
            return

        patch = cv2.GaussianBlur(
            patch,
            (3, 3),
            0
        )

        if track_hole._tm_recent_gray is None:

            track_hole._tm_recent_gray = (
                patch.copy()
            )

        else:

            recent_float = (
                track_hole
                ._tm_recent_gray
                .astype(np.float32)
            )

            patch_float = patch.astype(
                np.float32
            )

            updated = (
                (
                    1.0
                    -
                    RECENT_TEMPLATE_ALPHA
                )
                *
                recent_float
                +
                RECENT_TEMPLATE_ALPHA
                *
                patch_float
            )

            track_hole._tm_recent_gray = (
                np.clip(
                    updated,
                    0,
                    255
                )
                .astype(np.uint8)
            )

        track_hole._tm_recent_edge = (
            cv2.Canny(
                track_hole._tm_recent_gray,
                50,
                150
            )
        )

        track_hole._tm_recent_updates += 1


    # ========================================================
    # Sanitized velocity
    # ========================================================

    if hole_velocity is None:

        velocity = np.zeros(
            2,
            dtype=np.float64,
        )

    else:

        velocity = np.array(
            hole_velocity,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(velocity)
        ):

            velocity = np.zeros(
                2,
                dtype=np.float64,
            )

    # ========================================================
    # Time since last REAL Hole measurement
    # ========================================================

    if hole_last_valid_time is not None:

        lost_time = max(
            0.0,
            now_track
            -
            hole_last_valid_time
        )

    else:

        lost_time = float("inf")

    # ========================================================
    # Predicted reference for normal LOCAL tracking
    # ========================================================

    reference = np.array(
        hole_center,
        dtype=np.float64,
    )

    if (
        hole_last_valid_center is not None
        and
        hole_last_valid_time is not None
    ):

        predict_dt = min(
            lost_time,
            HOLE_HOLD_S
        )

        predicted_reference = (
            np.array(
                hole_last_valid_center,
                dtype=np.float64,
            )
            +
            velocity
            *
            predict_dt
        )

        if np.all(
            np.isfinite(
                predicted_reference
            )
        ):

            reference = (
                predicted_reference
            )

    # ========================================================
    # Helper: accept a REAL visual Hole measurement
    # ========================================================

    def accept_measurement(
        measured_center,
        reset_velocity=False,
    ):

        global hole_center
        global hole_last_valid_center
        global hole_last_valid_time
        global hole_velocity
        global hole_ema

        measured_center = np.array(
            measured_center,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                measured_center
            )
        ):
            return None

        now_measure = time.time()

        if reset_velocity:

            hole_velocity = np.zeros(
                2,
                dtype=np.float64,
            )

        elif (
            hole_last_valid_center is not None
            and
            hole_last_valid_time is not None
        ):

            dt = max(
                now_measure
                -
                hole_last_valid_time,
                1e-3
            )

            new_velocity = (
                measured_center
                -
                np.array(
                    hole_last_valid_center,
                    dtype=np.float64,
                )
            ) / dt

            if np.all(
                np.isfinite(
                    new_velocity
                )
            ):

                speed = float(
                    np.linalg.norm(
                        new_velocity
                    )
                )

                if (
                    speed
                    >
                    HOLE_PREDICT_MAX_SPEED_PX_S
                    and
                    speed > 1e-6
                ):

                    new_velocity = (
                        new_velocity
                        *
                        (
                            HOLE_PREDICT_MAX_SPEED_PX_S
                            /
                            speed
                        )
                    )

                if hole_velocity is None:

                    hole_velocity = (
                        new_velocity
                    )

                else:

                    previous_velocity = np.array(
                        hole_velocity,
                        dtype=np.float64,
                    )

                    if np.all(
                        np.isfinite(
                            previous_velocity
                        )
                    ):

                        hole_velocity = (
                            HOLE_VELOCITY_ALPHA
                            *
                            new_velocity
                            +
                            (
                                1.0
                                -
                                HOLE_VELOCITY_ALPHA
                            )
                            *
                            previous_velocity
                        )

                    else:

                        hole_velocity = (
                            new_velocity
                        )

        hole_center = (
            int(
                round(
                    measured_center[0]
                )
            ),
            int(
                round(
                    measured_center[1]
                )
            ),
        )

        hole_last_valid_center = (
            np.array(
                hole_center,
                dtype=np.float64,
            )
        )

        hole_last_valid_time = (
            now_measure
        )

        track_hole._tm_active = False
        track_hole._tm_candidates = []
        track_hole._tm_misses = 0

        return hole_center

    # ========================================================
    # Helper: after GLOBAL relock, do not immediately feed the
    # recovered target into IBVS.
    #
    # Require POST_RELOCK_LOCAL_FRAMES normal local visual
    # measurements first.
    # ========================================================

    def finish_local_measurement(
        accepted,
        source,
    ):

        if accepted is None:
            return None

        if (
            track_hole._post_relock_remaining
            >
            0
        ):

            track_hole._post_relock_remaining -= 1

            print(
                "[HOLE POST RELOCK] "
                f"source={source} "
                f"remaining="
                f"{track_hole._post_relock_remaining}",
                flush=True,
            )

            if (
                track_hole._post_relock_remaining
                >
                0
            ):
                return None

            print(
                "[HOLE POST RELOCK] "
                "LOCAL VERIFIED -> IBVS ENABLED",
                flush=True,
            )

        return accepted

    # ========================================================
    # LOCAL TRACKING
    # ========================================================

    ref_x = int(
        round(
            reference[0]
        )
    )

    ref_y = int(
        round(
            reference[1]
        )
    )

    x0 = max(
        0,
        ref_x - LOCAL_SEARCH
    )

    y0 = max(
        0,
        ref_y - LOCAL_SEARCH
    )

    x1 = min(
        gray.shape[1],
        ref_x + LOCAL_SEARCH
    )

    y1 = min(
        gray.shape[0],
        ref_y + LOCAL_SEARCH
    )

    local_valid = (
        x1 > x0
        and
        y1 > y0
    )

    if local_valid:

        crop = gray[
            y0:y1,
            x0:x1
        ]

        local_valid = (
            crop.size > 0
        )

    if local_valid:

        blur = cv2.GaussianBlur(
            crop,
            (5, 5),
            0
        )

        # ----------------------------------------------------
        # LOCAL ellipse
        # ----------------------------------------------------

        ellipse_hole = (
            detect_hole_ellipse(
                crop,
                x0,
                y0
            )
        )

        if ellipse_hole is not None:

            candidate = np.array(
                [
                    ellipse_hole[0],
                    ellipse_hole[1],
                ],
                dtype=np.float64,
            )

            if np.all(
                np.isfinite(candidate)
            ):

                jump = float(
                    np.linalg.norm(
                        candidate
                        -
                        reference
                    )
                )

                if (
                    jump
                    <=
                    HOLE_MAX_JUMP_PX
                    and
                    local_template_score(
                        candidate
                    )
                    >=
                    LOCAL_TM_MIN_SCORE
                ):

                    if (
                        hole_ema is None
                        or
                        not np.all(
                            np.isfinite(
                                hole_ema
                            )
                        )
                    ):

                        hole_ema = (
                            candidate.copy()
                        )

                    else:

                        hole_ema = (
                            HOLE_EMA_ALPHA
                            *
                            candidate
                            +
                            (
                                1.0
                                -
                                HOLE_EMA_ALPHA
                            )
                            *
                            hole_ema
                        )

                    if np.all(
                        np.isfinite(
                            hole_ema
                        )
                    ):

                        update_recent_template(
                            hole_ema
                        )

                        accepted = (
                            accept_measurement(
                                hole_ema,
                                reset_velocity=False,
                            )
                        )

                        return finish_local_measurement(
                            accepted,
                            "ELLIPSE",
                        )

        # ----------------------------------------------------
        # LOCAL Hough fallback
        # ----------------------------------------------------

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=15,
            param1=80,
            param2=12,
            minRadius=
            HOUGH_MIN_RADIUS,
            maxRadius=
            HOUGH_MAX_RADIUS
        )

        if circles is not None:

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
                    (
                        float(cx)
                        -
                        reference[0]
                    ) ** 2
                    +
                    (
                        float(cy)
                        -
                        reference[1]
                    ) ** 2
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

            if best is not None:

                best_array = np.array(
                    best,
                    dtype=np.float64,
                )

                best_jump = float(
                    np.linalg.norm(
                        best_array
                        -
                        reference
                    )
                )

                if (
                    np.all(
                        np.isfinite(
                            best_array
                        )
                    )
                    and
                    best_jump
                    <=
                    HOLE_MAX_JUMP_PX
                    and
                    local_template_score(
                        best_array
                    )
                    >=
                    LOCAL_TM_MIN_SCORE
                ):

                    hole_ema = (
                        best_array.copy()
                    )

                    update_recent_template(
                        best_array
                    )

                    accepted = (
                        accept_measurement(
                            best_array,
                            reset_velocity=False,
                        )
                    )

                    return finish_local_measurement(
                        accepted,
                        "HOUGH",
                    )

    # ========================================================
    # During post-global verification, prediction must NOT
    # enter IBVS.
    # ========================================================

    if (
        track_hole._post_relock_remaining
        >
        0
    ):

        return None

    # ========================================================
    # SHORT HOLD / PREDICTION
    # ========================================================

    if (
        hole_last_valid_center is not None
        and
        hole_last_valid_time is not None
        and
        lost_time <= HOLE_HOLD_S
    ):

        # Prediction is already used above to move the LOCAL
        # search reference.  It is intentionally NOT returned
        # as a valid Hole observation.
        #
        # This prevents predicted Hole coordinates from
        # generating Wrist IBVS velocity commands.
        return None

    # ========================================================
    # GLOBAL TEMPLATE REACQUIRE
    # ========================================================

    # Prefer the most recently validated appearance during
    # GLOBAL reacquisition.
    #
    # INITIAL template remains stored separately and is never
    # overwritten.
    if (
        track_hole._tm_recent_gray
        is not None
    ):

        template_gray = (
            track_hole._tm_recent_gray
        )

        template_edge = (
            track_hole._tm_recent_edge
        )

    else:

        template_gray = (
            track_hole._tm_template_gray
        )

        template_edge = (
            track_hole._tm_template_edge
        )

    if template_gray is None:

        if (
            now_track
            -
            track_hole._tm_last_print
            >
            0.5
        ):

            print(
                "[HOLE GLOBAL TM] "
                "NO TEMPLATE",
                flush=True,
            )

            track_hole._tm_last_print = (
                now_track
            )

        return None

    if not track_hole._tm_active:

        track_hole._tm_active = True
        track_hole._tm_candidates = []
        track_hole._tm_misses = 0

        print(
            "[HOLE GLOBAL TM] START "
            f"lost={lost_time:.3f}s "
            f"last={hole_last_valid_center}",
            flush=True,
        )

    if (
        now_track
        -
        track_hole._tm_last_run
        <
        GLOBAL_TM_PERIOD_S
    ):

        return None

    track_hole._tm_last_run = (
        now_track
    )

    # --------------------------------------------------------
    # Search image and template at 0.75 scale.
    # --------------------------------------------------------

    search_small = cv2.resize(
        gray,
        None,
        fx=GLOBAL_TM_SCALE,
        fy=GLOBAL_TM_SCALE,
        interpolation=cv2.INTER_AREA,
    )

    template_small = cv2.resize(
        template_gray,
        None,
        fx=GLOBAL_TM_SCALE,
        fy=GLOBAL_TM_SCALE,
        interpolation=cv2.INTER_AREA,
    )

    if (
        template_small.shape[0] < 8
        or
        template_small.shape[1] < 8
        or
        template_small.shape[0]
        >=
        search_small.shape[0]
        or
        template_small.shape[1]
        >=
        search_small.shape[1]
    ):

        print(
            "[HOLE GLOBAL TM] "
            "INVALID TEMPLATE SIZE",
            flush=True,
        )

        return None

    search_small = cv2.GaussianBlur(
        search_small,
        (3, 3),
        0
    )

    gray_response = cv2.matchTemplate(
        search_small,
        template_small,
        cv2.TM_CCOEFF_NORMED,
    )

    # --------------------------------------------------------
    # Add edge appearance to reduce confusion with unrelated
    # circular objects having different local texture.
    # --------------------------------------------------------

    use_edge = False
    combined = gray_response

    if template_edge is not None:

        template_edge_small = cv2.resize(
            template_edge,
            (
                template_small.shape[1],
                template_small.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        if (
            float(
                np.std(
                    template_edge_small
                )
            )
            >
            1.0
        ):

            search_edge_small = cv2.Canny(
                search_small,
                50,
                150
            )

            edge_response = cv2.matchTemplate(
                search_edge_small,
                template_edge_small,
                cv2.TM_CCORR_NORMED,
            )

            combined = (
                0.70
                *
                gray_response
                +
                0.30
                *
                edge_response
            )

            use_edge = True

    # --------------------------------------------------------
    # Get several spatially separated template peaks.
    # --------------------------------------------------------

    score_work = (
        combined.copy()
    )

    th = template_small.shape[0]
    tw = template_small.shape[1]

    peak_candidates = []

    nms_radius = max(
        10,
        int(
            round(
                0.75
                *
                max(
                    th,
                    tw
                )
            )
        )
    )

    best_raw_score = -1.0

    for _ in range(
        GLOBAL_TM_TOP_K
    ):

        (
            _min_val,
            max_val,
            _min_loc,
            max_loc,
        ) = cv2.minMaxLoc(
            score_work
        )

        score = float(
            max_val
        )

        if score > best_raw_score:
            best_raw_score = score

        if (
            score
            <
            GLOBAL_TM_MIN_SCORE
        ):
            break

        cx_small = (
            float(max_loc[0])
            +
            0.5
            *
            float(tw)
        )

        cy_small = (
            float(max_loc[1])
            +
            0.5
            *
            float(th)
        )

        candidate_full = np.array(
            [
                cx_small
                /
                GLOBAL_TM_SCALE,

                cy_small
                /
                GLOBAL_TM_SCALE,
            ],
            dtype=np.float64,
        )

        peak_candidates.append(
            (
                score,
                candidate_full,
            )
        )

        sx0 = max(
            0,
            max_loc[0]
            -
            nms_radius
        )

        sy0 = max(
            0,
            max_loc[1]
            -
            nms_radius
        )

        sx1 = min(
            score_work.shape[1],
            max_loc[0]
            +
            nms_radius
            +
            1
        )

        sy1 = min(
            score_work.shape[0],
            max_loc[1]
            +
            nms_radius
            +
            1
        )

        score_work[
            sy0:sy1,
            sx0:sx1
        ] = -1.0

    if not peak_candidates:

        track_hole._tm_misses += 1

        if (
            track_hole._tm_misses
            >
            GLOBAL_MAX_MISSES
        ):

            track_hole._tm_candidates = []
            track_hole._tm_misses = 0

        if (
            now_track
            -
            track_hole._tm_last_print
            >
            0.30
        ):

            print(
                "[HOLE GLOBAL TM] "
                f"score={best_raw_score:.3f} "
                f"< {GLOBAL_TM_MIN_SCORE:.2f} "
                f"edge={int(use_edge)}",
                flush=True,
            )

            track_hole._tm_last_print = (
                now_track
            )

        return None

    # ========================================================
    # Helper: geometry validation around template candidate
    # ========================================================

    def verify_global_geometry(
        template_center,
    ):

        tcx = int(
            round(
                template_center[0]
            )
        )

        tcy = int(
            round(
                template_center[1]
            )
        )

        verify_half = max(
            45,
            min(
                90,
                int(
                    round(
                        float(hole_radius)
                        *
                        4.0
                    )
                )
            )
        )

        gx0 = max(
            0,
            tcx - verify_half
        )

        gy0 = max(
            0,
            tcy - verify_half
        )

        gx1 = min(
            gray.shape[1],
            tcx + verify_half
        )

        gy1 = min(
            gray.shape[0],
            tcy + verify_half
        )

        if (
            gx1 <= gx0
            or
            gy1 <= gy0
        ):
            return None, None

        gcrop = gray[
            gy0:gy1,
            gx0:gx1
        ]

        if gcrop.size == 0:
            return None, None

        geom_candidates = []

        # ----------------------------------------------------
        # Geometry A: ellipse detector
        # ----------------------------------------------------

        gellipse = detect_hole_ellipse(
            gcrop,
            gx0,
            gy0
        )

        if gellipse is not None:

            ep = np.array(
                [
                    gellipse[0],
                    gellipse[1],
                ],
                dtype=np.float64,
            )

            if np.all(
                np.isfinite(ep)
            ):

                ed = float(
                    np.linalg.norm(
                        ep
                        -
                        template_center
                    )
                )

                if (
                    ed
                    <=
                    GLOBAL_GEOM_MAX_OFFSET_PX
                ):

                    geom_candidates.append(
                        (
                            ed,
                            ep,
                            "ELLIPSE",
                        )
                    )

        # ----------------------------------------------------
        # Geometry B: radius-constrained Hough
        # ----------------------------------------------------

        gblur = cv2.GaussianBlur(
            gcrop,
            (5, 5),
            0
        )

        gcircles = cv2.HoughCircles(
            gblur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=15,
            param1=80,
            param2=12,
            minRadius=
            HOUGH_MIN_RADIUS,
            maxRadius=
            HOUGH_MAX_RADIUS,
        )

        if gcircles is not None:

            for c in np.round(
                gcircles[0]
            ).astype(int):

                hp = np.array(
                    [
                        float(
                            c[0]
                            +
                            gx0
                        ),
                        float(
                            c[1]
                            +
                            gy0
                        ),
                    ],
                    dtype=np.float64,
                )

                hd = float(
                    np.linalg.norm(
                        hp
                        -
                        template_center
                    )
                )

                if (
                    hd
                    <=
                    GLOBAL_GEOM_MAX_OFFSET_PX
                ):

                    geom_candidates.append(
                        (
                            hd,
                            hp,
                            "HOUGH",
                        )
                    )

        if not geom_candidates:
            return None, None

        geom_candidates.sort(
            key=lambda item: item[0]
        )

        (
            _dist,
            best_geom,
            source,
        ) = geom_candidates[0]

        return (
            best_geom.copy(),
            source,
        )

    # ========================================================
    # Evaluate template peaks.
    #
    # Also use a very conservative motion plausibility gate.
    # This is NOT a ROI: the whole frame is searched.
    # It only prevents a target from teleporting hundreds of
    # pixels immediately after a short 0.30 s loss.
    # ========================================================

    if hole_last_valid_center is not None:

        motion_anchor = np.array(
            hole_last_valid_center,
            dtype=np.float64,
        )

        # Predict only a short horizon. The filtered velocity
        # cannot recursively update itself because prediction
        # is not treated as a visual measurement.
        if np.all(
            np.isfinite(
                velocity
            )
        ):

            motion_anchor = (
                motion_anchor
                +
                velocity
                *
                min(
                    lost_time,
                    0.50
                )
            )

    else:

        motion_anchor = reference.copy()

    if (
        lost_time
        <=
        GLOBAL_NEAR_LOSS_S
    ):

        motion_limit = (
            GLOBAL_NEAR_MOTION_LIMIT_PX
        )

    else:

        # If the target has been missing for longer, gradually
        # permit a true global recovery.
        motion_limit = min(
            700.0,
            90.0
            +
            180.0
            *
            (
                lost_time
                -
                GLOBAL_NEAR_LOSS_S
            )
        )

    verified = []

    for (
        tm_score,
        tm_center,
    ) in peak_candidates:

        motion_dist = float(
            np.linalg.norm(
                tm_center
                -
                motion_anchor
            )
        )

        if (
            motion_dist
            >
            motion_limit
        ):

            continue

        (
            geom_center,
            geom_source,
        ) = verify_global_geometry(
            tm_center
        )

        if geom_center is None:
            continue

        # Small spatial prior is only a ranking term.
        # It does not define the search region.
        rank_score = (
            float(tm_score)
            -
            0.00025
            *
            motion_dist
        )

        verified.append(
            (
                rank_score,
                float(tm_score),
                motion_dist,
                geom_center,
                geom_source,
                tm_center,
            )
        )

    if not verified:

        track_hole._tm_misses += 1

        if (
            track_hole._tm_misses
            >
            GLOBAL_MAX_MISSES
        ):

            track_hole._tm_candidates = []
            track_hole._tm_misses = 0

        if (
            now_track
            -
            track_hole._tm_last_print
            >
            0.30
        ):

            print(
                "[HOLE GLOBAL TM] "
                f"NO GEOM "
                f"best_score="
                f"{best_raw_score:.3f} "
                f"peaks="
                f"{len(peak_candidates)} "
                f"motion_limit="
                f"{motion_limit:.1f}",
                flush=True,
            )

            track_hole._tm_last_print = (
                now_track
            )

        return None

    verified.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    (
        _rank_score,
        tm_score,
        motion_dist,
        best_global,
        geom_source,
        raw_tm_center,
    ) = verified[0]

    track_hole._tm_misses = 0

    # ========================================================
    # Temporal consistency check
    # ========================================================

    if track_hole._tm_candidates:

        previous_global = (
            track_hole
            ._tm_candidates[-1]
        )

        global_step = float(
            np.linalg.norm(
                best_global
                -
                previous_global
            )
        )

        if (
            global_step
            >
            GLOBAL_RELOCK_MAX_STEP_PX
        ):

            track_hole._tm_candidates = [
                best_global.copy()
            ]

            print(
                "[HOLE GLOBAL TM] RESET "
                f"score={tm_score:.3f} "
                f"geom={geom_source} "
                f"H=("
                f"{best_global[0]:.1f},"
                f"{best_global[1]:.1f}) "
                f"step={global_step:.1f}px "
                f"motion={motion_dist:.1f}px",
                flush=True,
            )

            return None

    else:

        track_hole._tm_candidates = []

    track_hole._tm_candidates.append(
        best_global.copy()
    )

    if (
        len(
            track_hole._tm_candidates
        )
        >
        GLOBAL_RELOCK_FRAMES
    ):

        track_hole._tm_candidates = (
            track_hole
            ._tm_candidates[
                -GLOBAL_RELOCK_FRAMES:
            ]
        )

    confirm_count = len(
        track_hole._tm_candidates
    )

    print(
        "[HOLE GLOBAL TM] candidate "
        f"score={tm_score:.3f} "
        f"geom={geom_source} "
        f"TM=("
        f"{raw_tm_center[0]:.1f},"
        f"{raw_tm_center[1]:.1f}) "
        f"H=("
        f"{best_global[0]:.1f},"
        f"{best_global[1]:.1f}) "
        f"motion={motion_dist:.1f}px "
        f"confirm="
        f"{confirm_count}/"
        f"{GLOBAL_RELOCK_FRAMES}",
        flush=True,
    )

    if (
        confirm_count
        <
        GLOBAL_RELOCK_FRAMES
    ):
        return None

    # ========================================================
    # Use the mean of the verified sequence, not one frame.
    # ========================================================

    relock_stack = np.stack(
        track_hole._tm_candidates,
        axis=0,
    )

    relock_center = np.mean(
        relock_stack,
        axis=0,
    )

    print(
        "[HOLE GLOBAL TM RELOCK] "
        f"H=("
        f"{relock_center[0]:.1f},"
        f"{relock_center[1]:.1f}) "
        f"score={tm_score:.3f} "
        "waiting for local verification",
        flush=True,
    )

    # --------------------------------------------------------
    # Establish the new local-search center but DO NOT return
    # it to IBVS yet.
    #
    # Velocity is reset so the long LOST interval cannot
    # create an enormous prediction velocity.
    # --------------------------------------------------------

    hole_center = (
        int(
            round(
                relock_center[0]
            )
        ),
        int(
            round(
                relock_center[1]
            )
        ),
    )

    hole_last_valid_center = np.array(
        hole_center,
        dtype=np.float64,
    )

    hole_last_valid_time = time.time()

    hole_velocity = np.zeros(
        2,
        dtype=np.float64,
    )

    hole_ema = np.array(
        hole_center,
        dtype=np.float64,
    )

    track_hole._tm_active = False
    track_hole._tm_candidates = []
    track_hole._tm_misses = 0

    track_hole._post_relock_remaining = (
        POST_RELOCK_LOCAL_FRAMES
    )

    # Critical:
    # no IBVS output from the global candidate itself.
    return None


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

    perf_last_report = time.perf_counter()
    perf_frames = 0
    perf_sum_wait_ms = 0.0
    perf_sum_convert_ms = 0.0
    perf_sum_tip_detect_ms = 0.0
    perf_sum_tip_refine_ms = 0.0
    perf_sum_hole_ms = 0.0
    perf_sum_rest_ms = 0.0
    perf_sum_total_ms = 0.0
    perf_max_total_ms = 0.0
    perf_refine_calls = 0
    perf_hole_calls = 0

    tip_detect_frame_counter = 0

    try:

        while True:

            perf_loop_t0 = time.perf_counter()
            perf_tip_detect_ms = 0.0
            perf_tip_refine_ms = 0.0
            perf_hole_ms = 0.0
            perf_refine_called = False
            perf_hole_called = False

            perf_t0 = time.perf_counter()

            frames = pipeline.wait_for_frames(
                1000
            )

            perf_wait_ms = (
                time.perf_counter() - perf_t0
            ) * 1000.0

            if frames is None:

                write_disabled_output()

                continue

            color = frames.get_color_frame()

            if color is None:

                write_disabled_output()

                continue

            perf_t0 = time.perf_counter()

            frame = hc.color_frame_to_bgr(
                color
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            perf_convert_ms = (
                time.perf_counter() - perf_t0
            ) * 1000.0

            tip_detect_frame_counter += 1

            run_tip_detector = (
                tip_ema is None
                or
                wrist_last_valid_tip is None
                or
                wrist_last_valid_time is None
                or
                (
                    tip_detect_frame_counter
                    %
                    TIP_DETECT_EVERY_N_FRAMES
                    ==
                    0
                )
            )

            # ------------------------------------------------
            # LIVE TWEEZER TIP
            # ------------------------------------------------

            tip_pixel = None
            tip_raw_pixel = None
            tip_status = "LOST"

            if run_tip_detector:
                try:

                    perf_t0 = time.perf_counter()

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

                    perf_tip_detect_ms = (
                        time.perf_counter() - perf_t0
                    ) * 1000.0

                    if coarse_tip.get(
                        "valid",
                        False,
                    ):

                        perf_t0 = time.perf_counter()
                        perf_refine_called = True

                        refined_tip = (
                            refiner.refine(
                                frame,
                                coarse_tip["mask"],
                                coarse_tip["tip_uv"],
                                coarse_tip["direction"],
                            )
                        )

                        perf_tip_refine_ms = (
                            time.perf_counter() - perf_t0
                        ) * 1000.0

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

                                # ====================================
                                # MOTION-AWARE VISUAL GATE
                                #
                                # tip_ema remains the last accepted visual
                                # state.  A short prediction is used ONLY
                                # as the expected position for jump testing.
                                # ====================================

                                tip_gate_reference = np.array(
                                    tip_ema,
                                    dtype=np.float64,
                                )

                                if (
                                    wrist_last_valid_tip is not None
                                    and
                                    wrist_last_valid_time is not None
                                    and
                                    wrist_tip_velocity is not None
                                ):

                                    gate_dt = max(
                                        0.0,
                                        time.time()
                                        -
                                        wrist_last_valid_time
                                    )

                                    gate_dt = min(
                                        gate_dt,
                                        TIP_HOLD_S
                                    )

                                    gate_velocity = np.array(
                                        wrist_tip_velocity,
                                        dtype=np.float64,
                                    )

                                    if np.all(
                                        np.isfinite(
                                            gate_velocity
                                        )
                                    ):

                                        predicted_gate = (
                                            np.array(
                                                wrist_last_valid_tip,
                                                dtype=np.float64,
                                            )
                                            +
                                            gate_velocity
                                            *
                                            gate_dt
                                        )

                                        if np.all(
                                            np.isfinite(
                                                predicted_gate
                                            )
                                        ):

                                            tip_gate_reference = (
                                                predicted_gate
                                            )

                                tip_jump = float(
                                    np.linalg.norm(
                                        raw_tip
                                        -
                                        tip_gate_reference
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

                                        # --------------------------------
                                        # Relock identity / motion continuity
                                        #
                                        # Do not trust "3 stable points" alone.
                                        # Compare the cluster center against
                                        # the most recent reliable Tip
                                        # trajectory.
                                        # --------------------------------

                                        relock_anchor_dist = 0.0

                                        if (
                                            wrist_last_valid_tip
                                            is not None
                                        ):

                                            relock_anchor = np.array(
                                                wrist_last_valid_tip,
                                                dtype=np.float64,
                                            )

                                            if (
                                                wrist_last_valid_time
                                                is not None
                                                and
                                                wrist_tip_velocity
                                                is not None
                                            ):

                                                relock_age = max(
                                                    0.0,
                                                    time.time()
                                                    -
                                                    wrist_last_valid_time
                                                )

                                                relock_velocity = np.array(
                                                    wrist_tip_velocity,
                                                    dtype=np.float64,
                                                )

                                                if np.all(
                                                    np.isfinite(
                                                        relock_velocity
                                                    )
                                                ):

                                                    relock_anchor = (
                                                        relock_anchor
                                                        +
                                                        relock_velocity
                                                        *
                                                        min(
                                                            relock_age,
                                                            TIP_HOLD_S
                                                        )
                                                    )

                                            relock_anchor_dist = float(
                                                np.linalg.norm(
                                                    relock_center
                                                    -
                                                    relock_anchor
                                                )
                                            )

                                        if (
                                            relock_spread
                                            <=
                                            TIP_RELOCK_CLUSTER_RADIUS_PX
                                            and
                                            relock_anchor_dist
                                            <=
                                            TIP_RELOCK_MAX_ANCHOR_DIST_PX
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

                                            if (
                                                relock_anchor_dist
                                                >
                                                TIP_RELOCK_MAX_ANCHOR_DIST_PX
                                            ):

                                                tip_status = (
                                                    f"RELOCK REJECT "
                                                    f"anchor="
                                                    f"{relock_anchor_dist:.1f} "
                                                    f"spread="
                                                    f"{relock_spread:.1f}"
                                                )

                                                print(
                                                    "[TIP RELOCK REJECT] "
                                                    f"candidate="
                                                    f"({relock_center[0]:.1f},"
                                                    f"{relock_center[1]:.1f}) "
                                                    f"anchor_dist="
                                                    f"{relock_anchor_dist:.1f}px "
                                                    f"spread="
                                                    f"{relock_spread:.1f}px",
                                                    flush=True,
                                                )

                                                # Require a completely new
                                                # candidate sequence.
                                                relock_candidates.clear()

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

                                    if tip_status.startswith(
                                        "RELOCK"
                                    ):

                                        # A relock establishes a new visual
                                        # state. Do not interpret the relock
                                        # displacement as physical velocity.
                                        wrist_tip_velocity = np.zeros(
                                            2,
                                            dtype=np.float64,
                                        )

                                    else:

                                        measured_tip_velocity = (
                                            tip_ema
                                            -
                                            wrist_last_valid_tip
                                        ) / dt

                                        if np.all(
                                            np.isfinite(
                                                measured_tip_velocity
                                            )
                                        ):

                                            measured_speed = float(
                                                np.linalg.norm(
                                                    measured_tip_velocity
                                                )
                                            )

                                            if (
                                                measured_speed
                                                >
                                                TIP_PREDICT_MAX_SPEED_PX_S
                                                and
                                                measured_speed > 1e-6
                                            ):

                                                measured_tip_velocity = (
                                                    measured_tip_velocity
                                                    *
                                                    (
                                                        TIP_PREDICT_MAX_SPEED_PX_S
                                                        /
                                                        measured_speed
                                                    )
                                                )

                                            if wrist_tip_velocity is None:

                                                wrist_tip_velocity = (
                                                    measured_tip_velocity
                                                )

                                            else:

                                                previous_tip_velocity = (
                                                    np.array(
                                                        wrist_tip_velocity,
                                                        dtype=np.float64,
                                                    )
                                                )

                                                if np.all(
                                                    np.isfinite(
                                                        previous_tip_velocity
                                                    )
                                                ):

                                                    wrist_tip_velocity = (
                                                        TIP_VELOCITY_ALPHA
                                                        *
                                                        measured_tip_velocity
                                                        +
                                                        (
                                                            1.0
                                                            -
                                                            TIP_VELOCITY_ALPHA
                                                        )
                                                        *
                                                        previous_tip_velocity
                                                    )

                                                else:

                                                    wrist_tip_velocity = (
                                                        measured_tip_velocity
                                                    )

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

                                    # IMPORTANT:
                                    # Keep tip_ema at the last accepted
                                    # VISUAL Tip.  hold_tip is only a short
                                    # prediction and must not move the
                                    # tracking reference.

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

            else:

                # --------------------------------------------
                # Lightweight intermediate-frame Tip estimate
                # --------------------------------------------
                now_tip_time = time.time()

                if (
                    wrist_last_valid_tip is not None
                    and
                    wrist_last_valid_time is not None
                ):
                    predict_dt = max(
                        0.0,
                        now_tip_time
                        -
                        wrist_last_valid_time
                    )

                    # ============================================
                    # SAFE SKIPPED-FRAME PREDICTION
                    #
                    # Prediction is allowed only for the same
                    # short horizon as TIP_HOLD_S.
                    #
                    # IMPORTANT:
                    # prediction must NOT overwrite tip_ema.
                    # tip_ema remains the last accepted VISUAL
                    # tracking reference.
                    # ============================================

                    if predict_dt < TIP_HOLD_S:

                        predicted_tip = (
                            wrist_last_valid_tip
                            +
                            wrist_tip_velocity
                            *
                            predict_dt
                        )

                        if np.all(
                            np.isfinite(
                                predicted_tip
                            )
                        ):

                            tip_pixel = (
                                int(
                                    round(
                                        float(
                                            predicted_tip[0]
                                        )
                                    )
                                ),
                                int(
                                    round(
                                        float(
                                            predicted_tip[1]
                                        )
                                    )
                                ),
                            )

                            # Short-lived prediction may still
                            # help the next heavy detector frame.
                            tip_hint = (
                                float(predicted_tip[0]),
                                float(predicted_tip[1]),
                            )

                            tip_status = (
                                f"PREDICT "
                                f"{predict_dt:.3f}s"
                            )

                        else:

                            tip_pixel = None

                            tip_status = (
                                "PREDICT INVALID"
                            )

                    else:

                        # Prediction has expired.
                        #
                        # Do not generate a Tip observation and
                        # therefore do not allow Wrist IBVS to
                        # use an arbitrarily old motion model.
                        tip_pixel = None

                        tip_status = (
                            f"PREDICT EXPIRED "
                            f"{predict_dt:.3f}s"
                        )

                else:

                    tip_pixel = None

                    tip_status = (
                        "PREDICT NO STATE"
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

                perf_t0 = time.perf_counter()
                perf_hole_called = True

                H = track_hole(
                    gray
                )

                perf_hole_ms = (
                    time.perf_counter() - perf_t0
                ) * 1000.0

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

            perf_total_ms = (
                time.perf_counter() - perf_loop_t0
            ) * 1000.0

            perf_rest_ms = max(
                0.0,
                perf_total_ms
                - perf_wait_ms
                - perf_convert_ms
                - perf_tip_detect_ms
                - perf_tip_refine_ms
                - perf_hole_ms
            )

            perf_frames += 1
            perf_sum_wait_ms += perf_wait_ms
            perf_sum_convert_ms += perf_convert_ms
            perf_sum_tip_detect_ms += perf_tip_detect_ms
            perf_sum_tip_refine_ms += perf_tip_refine_ms
            perf_sum_hole_ms += perf_hole_ms
            perf_sum_rest_ms += perf_rest_ms
            perf_sum_total_ms += perf_total_ms
            perf_max_total_ms = max(
                perf_max_total_ms,
                perf_total_ms
            )

            if perf_refine_called:
                perf_refine_calls += 1

            if perf_hole_called:
                perf_hole_calls += 1

            perf_now = time.perf_counter()
            perf_elapsed = perf_now - perf_last_report

            if perf_elapsed >= 1.0 and perf_frames > 0:
                n = float(perf_frames)

                refine_call_ms = (
                    perf_sum_tip_refine_ms / perf_refine_calls
                    if perf_refine_calls > 0 else 0.0
                )

                hole_call_ms = (
                    perf_sum_hole_ms / perf_hole_calls
                    if perf_hole_calls > 0 else 0.0
                )

                print(
                    "[WRIST PERF] "
                    f"fps={perf_frames / perf_elapsed:.2f} "
                    f"total={perf_sum_total_ms / n:.2f}ms "
                    f"max={perf_max_total_ms:.2f}ms "
                    f"wait={perf_sum_wait_ms / n:.2f}ms "
                    f"convert={perf_sum_convert_ms / n:.2f}ms "
                    f"tip_detect={perf_sum_tip_detect_ms / n:.2f}ms "
                    f"tip_refine_frame={perf_sum_tip_refine_ms / n:.2f}ms "
                    f"tip_refine_call={refine_call_ms:.2f}ms "
                    f"hole_frame={perf_sum_hole_ms / n:.2f}ms "
                    f"hole_call={hole_call_ms:.2f}ms "
                    f"rest={perf_sum_rest_ms / n:.2f}ms "
                    f"refine={perf_refine_calls}/{perf_frames} "
                    f"hole={perf_hole_calls}/{perf_frames}",
                    flush=True,
                )

                perf_last_report = perf_now
                perf_frames = 0
                perf_sum_wait_ms = 0.0
                perf_sum_convert_ms = 0.0
                perf_sum_tip_detect_ms = 0.0
                perf_sum_tip_refine_ms = 0.0
                perf_sum_hole_ms = 0.0
                perf_sum_rest_ms = 0.0
                perf_sum_total_ms = 0.0
                perf_max_total_ms = 0.0
                perf_refine_calls = 0
                perf_hole_calls = 0

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
