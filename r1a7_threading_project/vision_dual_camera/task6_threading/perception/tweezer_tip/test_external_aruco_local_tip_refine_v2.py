#!/usr/bin/env python3

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

CAMERA_DEVICE = "/dev/video16"

INTRINSICS_JSON = Path(
    "calibration/fixed_camera_intrinsics/results/"
    "usb_zoom_camera_intrinsics.json"
)

TIP_CALIB_JSON = Path(
    "calibration/tweezer_marker_tip_calibration_median30.json"
)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30

ARUCO_ID = 20
MARKER_SIZE_M = 0.020

# Local search region around coarse ArUco prediction
ROI_HALF = 55

# Search corridor relative to predicted tool axis
CORRIDOR_HALF_WIDTH = 22.0
S_BACK = 70.0
S_FORWARD = 45.0

# Local refinement cannot move farther than this
MAX_REFINE_DIST = 50.0

# Only for visual display.
# Raw GREEN point is still shown independently.
EMA_ALPHA = 0.35

WINDOW = "ARUCO COARSE + LOCAL TIP REFINE V2"
DEBUG_WINDOW = "LOCAL TIP ROI V2"


# ============================================================
# Marker model
# ============================================================

half = MARKER_SIZE_M / 2.0

OBJECT_POINTS = np.asarray(
    [
        [-half, +half, 0.0],
        [+half, +half, 0.0],
        [+half, -half, 0.0],
        [-half, -half, 0.0],
    ],
    dtype=np.float64,
)

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_5X5_50
)

ARUCO_PARAMS = cv2.aruco.DetectorParameters()

ARUCO_DETECTOR = cv2.aruco.ArucoDetector(
    ARUCO_DICT,
    ARUCO_PARAMS,
)


# ============================================================
# Load calibration
# ============================================================

def load_intrinsics():

    with open(INTRINSICS_JSON, "r") as f:
        data = json.load(f)

    cal = data["calibration"]

    K = np.asarray(
        cal["camera_matrix"],
        dtype=np.float64,
    )

    dist = np.asarray(
        cal["dist_coeffs"],
        dtype=np.float64,
    ).reshape(-1, 1)

    return K, dist


def load_coarse_tip_marker():

    fallback = np.asarray(
        [
            0.004,
            0.080,
            -0.004,
        ],
        dtype=np.float64,
    )

    if not TIP_CALIB_JSON.exists():

        print(
            "[WARN] calibration file missing."
        )

        print(
            "[WARN] fallback P_tip_marker = "
            "[4,80,-4] mm"
        )

        return fallback

    with open(TIP_CALIB_JSON, "r") as f:
        data = json.load(f)

    if "robust_tip_marker_m" in data:

        p = np.asarray(
            data["robust_tip_marker_m"],
            dtype=np.float64,
        )

    elif "robust_tip_marker_mm" in data:

        p = (
            np.asarray(
                data["robust_tip_marker_mm"],
                dtype=np.float64,
            )
            /
            1000.0
        )

    else:

        print(
            "[WARN] robust tip result missing."
        )

        return fallback

    print(
        "[TIP] coarse P_tip_marker [mm] =",
        p * 1000.0,
    )

    return p


# ============================================================
# ArUco detection
# ============================================================

def detect_marker(gray):

    corners, ids, _ = (
        ARUCO_DETECTOR.detectMarkers(
            gray
        )
    )

    if ids is None:
        return None

    ids = ids.reshape(-1)

    for i, marker_id in enumerate(ids):

        if int(marker_id) != ARUCO_ID:
            continue

        pts = np.asarray(
            corners[i],
            dtype=np.float32,
        ).reshape(4, 2)

        refined = pts.reshape(
            4,
            1,
            2,
        ).copy()

        cv2.cornerSubPix(
            gray,
            refined,
            (5, 5),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS
                |
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.01,
            ),
        )

        return (
            refined
            .reshape(4, 2)
            .astype(np.float64)
        )

    return None


# ============================================================
# IPPE candidates
# ============================================================

def solve_pose_candidates(
    image_points,
    K,
    dist,
):

    result = cv2.solvePnPGeneric(
        OBJECT_POINTS,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not bool(result[0]):
        return []

    candidates = []

    for branch_id, (
        rvec,
        tvec,
    ) in enumerate(
        zip(
            result[1],
            result[2],
        )
    ):

        rvec = np.asarray(
            rvec,
            dtype=np.float64,
        ).reshape(3, 1)

        tvec = np.asarray(
            tvec,
            dtype=np.float64,
        ).reshape(3, 1)

        if float(tvec[2, 0]) <= 0:
            continue

        projected, _ = cv2.projectPoints(
            OBJECT_POINTS,
            rvec,
            tvec,
            K,
            dist,
        )

        projected = projected.reshape(
            4,
            2,
        )

        reproj = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            projected
                            -
                            image_points
                        ) ** 2,
                        axis=1,
                    )
                )
            )
        )

        candidates.append(
            {
                "branch_id":
                    int(branch_id),

                "rvec":
                    rvec,

                "tvec":
                    tvec,

                "marker_reproj_px":
                    reproj,
            }
        )

    return candidates


def project_point(
    p_marker,
    pose,
    K,
    dist,
):

    uv, _ = cv2.projectPoints(
        np.asarray(
            p_marker,
            dtype=np.float64,
        ).reshape(1, 3),

        pose["rvec"],
        pose["tvec"],
        K,
        dist,
    )

    return uv.reshape(2)


# ============================================================
# Local appearance model
# ============================================================

def make_border_mask(
    h,
    w,
    border=8,
):

    mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    b = min(
        border,
        max(
            1,
            min(h, w) // 4,
        ),
    )

    mask[:b, :] = 1
    mask[-b:, :] = 1
    mask[:, :b] = 1
    mask[:, -b:] = 1

    return mask.astype(bool)


# ============================================================
# Local real-tip refinement
# ============================================================

def local_refine(
    frame,
    coarse_uv,
    marker_center_uv,
    branch_id,
):

    h_img, w_img = frame.shape[:2]

    cu = float(coarse_uv[0])
    cv = float(coarse_uv[1])

    if not (
        0 <= cu < w_img
        and
        0 <= cv < h_img
    ):
        return None

    # --------------------------------------------------------
    # Tool image axis:
    # Marker center -> coarse predicted Tip
    # --------------------------------------------------------

    axis = np.asarray(
        [
            cu - marker_center_uv[0],
            cv - marker_center_uv[1],
        ],
        dtype=np.float64,
    )

    axis_norm = float(
        np.linalg.norm(axis)
    )

    if axis_norm < 10.0:
        return None

    axis /= axis_norm

    perp = np.asarray(
        [
            -axis[1],
            axis[0],
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # ROI around coarse prediction
    # --------------------------------------------------------

    x0 = max(
        0,
        int(
            math.floor(
                cu - ROI_HALF
            )
        ),
    )

    y0 = max(
        0,
        int(
            math.floor(
                cv - ROI_HALF
            )
        ),
    )

    x1 = min(
        w_img,
        int(
            math.ceil(
                cu + ROI_HALF + 1
            )
        ),
    )

    y1 = min(
        h_img,
        int(
            math.ceil(
                cv + ROI_HALF + 1
            )
        ),
    )

    if (
        x1 - x0 < 30
        or
        y1 - y0 < 30
    ):
        return None

    roi = frame[
        y0:y1,
        x0:x1,
    ].copy()

    rh, rw = roi.shape[:2]

    yy, xx = np.mgrid[
        0:rh,
        0:rw,
    ]

    gx = (
        xx.astype(np.float64)
        +
        x0
    )

    gy = (
        yy.astype(np.float64)
        +
        y0
    )

    rx = gx - cu
    ry = gy - cv

    # Longitudinal coordinate along tool
    s = (
        rx * axis[0]
        +
        ry * axis[1]
    )

    # Lateral coordinate
    d_signed = (
        rx * perp[0]
        +
        ry * perp[1]
    )

    d = np.abs(
        d_signed
    )

    # --------------------------------------------------------
    # Search corridor
    # --------------------------------------------------------

    corridor = (
        (s >= -70.0)
        &
        (s <= +42.0)
        &
        (d <= 24.0)
    )

    # --------------------------------------------------------
    # Learn the appearance of the TWEEZER ITSELF.
    #
    # We deliberately sample 20~48 px BEHIND the predicted Tip.
    # This region should belong to the metal tweezer, not the
    # fixture in front of the Tip.
    # --------------------------------------------------------

    seed_region = (
        (s >= -48.0)
        &
        (s <= -20.0)
        &
        (d <= 11.0)
    )

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB,
    ).astype(
        np.float32
    )

    seed_pixels = lab[
        seed_region
    ]

    if seed_pixels.shape[0] < 25:
        return None

    # Prefer the darker half of the seed strip.
    # This helps reject the bright tabletop/background while
    # retaining the grey metal tweezer.
    seed_L = seed_pixels[:, 0]

    L_cut = np.percentile(
        seed_L,
        50.0,
    )

    darker_seed = seed_pixels[
        seed_L <= L_cut
    ]

    if darker_seed.shape[0] < 12:
        darker_seed = seed_pixels

    tool_lab = np.median(
        darker_seed,
        axis=0,
    )

    seed_dist = np.linalg.norm(
        darker_seed
        -
        tool_lab.reshape(1, 3),
        axis=1,
    )

    seed_mad = float(
        np.median(
            seed_dist
        )
    )

    color_threshold = float(
        np.clip(
            max(
                18.0,
                3.0 * seed_mad + 8.0,
            ),
            18.0,
            38.0,
        )
    )

    # --------------------------------------------------------
    # Tool-color similarity
    # --------------------------------------------------------

    color_dist = np.linalg.norm(
        lab
        -
        tool_lab.reshape(
            1,
            1,
            3,
        ),
        axis=2,
    )

    tool_mask = (
        (color_dist <= color_threshold)
        &
        corridor
    ).astype(
        np.uint8
    ) * 255

    # --------------------------------------------------------
    # Remove obvious bright background.
    #
    # Allow some brightness change relative to seed, but not
    # arbitrary white tabletop regions.
    # --------------------------------------------------------

    L = lab[:, :, 0]

    seed_L_med = float(
        tool_lab[0]
    )

    brightness_ok = (
        np.abs(
            L - seed_L_med
        )
        <= 48.0
    )

    tool_mask[
        ~brightness_ok
    ] = 0

    # --------------------------------------------------------
    # Small morphology only.
    # We do NOT want large dilation that could connect Tip
    # to the fixture.
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    tool_mask = cv2.morphologyEx(
        tool_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    tool_mask = cv2.morphologyEx(
        tool_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    (
        nlabels,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        tool_mask,
        connectivity=8,
    )

    best = None

    for label in range(
        1,
        nlabels,
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if (
            area < 18
            or
            area > 1800
        ):
            continue

        ys, xs = np.where(
            labels == label
        )

        if len(xs) < 18:
            continue

        global_pts = np.column_stack(
            [
                xs + x0,
                ys + y0,
            ]
        ).astype(
            np.float64
        )

        rel = (
            global_pts
            -
            np.asarray(
                [cu, cv],
                dtype=np.float64,
            )
        )

        svals = (
            rel
            @
            axis
        )

        dvals_signed = (
            rel
            @
            perp
        )

        dvals = np.abs(
            dvals_signed
        )

        min_s = float(
            np.min(svals)
        )

        max_s = float(
            np.max(svals)
        )

        span = (
            max_s
            -
            min_s
        )

        if span < 18.0:
            continue

        # ----------------------------------------------------
        # Strong anchor condition:
        #
        # Candidate must actually extend through the part of
        # the image where the tweezer shaft should exist.
        # ----------------------------------------------------

        anchor_mask = (
            (svals >= -46.0)
            &
            (svals <= -18.0)
            &
            (dvals <= 12.0)
        )

        anchor_count = int(
            np.sum(
                anchor_mask
            )
        )

        if anchor_count < 15:
            continue

        # ----------------------------------------------------
        # PCA orientation
        # ----------------------------------------------------

        centered = (
            global_pts
            -
            np.mean(
                global_pts,
                axis=0,
                keepdims=True,
            )
        )

        cov = (
            centered.T
            @
            centered
        ) / max(
            len(global_pts) - 1,
            1,
        )

        evals, evecs = np.linalg.eigh(
            cov
        )

        major = evecs[
            :,
            int(
                np.argmax(
                    evals
                )
            ),
        ]

        alignment = float(
            abs(
                np.dot(
                    major,
                    axis,
                )
            )
        )

        if alignment < 0.72:
            continue

        # ----------------------------------------------------
        # Thin-front test
        #
        # The real metal Tip should become THIN near its
        # forward endpoint.
        #
        # A fixture / plate component usually becomes broad.
        # ----------------------------------------------------

        front_zone = (
            svals
            >=
            (
                max_s
                -
                12.0
            )
        )

        front_d = (
            dvals_signed[
                front_zone
            ]
        )

        if front_d.size < 4:
            continue

        front_width = float(
            np.percentile(
                front_d,
                90.0,
            )
            -
            np.percentile(
                front_d,
                10.0,
            )
        )

        if front_width > 17.0:
            continue

        # ----------------------------------------------------
        # Reject a component that suddenly becomes very wide
        # anywhere near/after coarse prediction.
        # ----------------------------------------------------

        forward_region = (
            svals >= -8.0
        )

        if np.sum(
            forward_region
        ) >= 8:

            fw_d = (
                dvals_signed[
                    forward_region
                ]
            )

            forward_width = float(
                np.percentile(
                    fw_d,
                    95.0,
                )
                -
                np.percentile(
                    fw_d,
                    5.0,
                )
            )

            if forward_width > 24.0:
                continue

        else:

            forward_width = front_width

        # ----------------------------------------------------
        # Forward-most thin pixels = physical Tip
        # ----------------------------------------------------

        front_pixels = (
            svals
            >=
            (
                max_s
                -
                1.5
            )
        )

        tip_candidates = (
            global_pts[
                front_pixels
            ]
        )

        if (
            tip_candidates.shape[0]
            < 1
        ):
            continue

        tip = np.median(
            tip_candidates,
            axis=0,
        )

        refine_dist = float(
            np.linalg.norm(
                tip
                -
                np.asarray(
                    [cu, cv],
                    dtype=np.float64,
                )
            )
        )

        if (
            refine_dist
            >
            MAX_REFINE_DIST
        ):
            continue

        mean_perp = float(
            np.mean(
                dvals
            )
        )

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        score = (
            1.7 * span
            +
            60.0 * alignment
            +
            0.10 * anchor_count
            +
            0.018 * area
            -
            0.35 * refine_dist
            -
            0.55 * front_width
            -
            0.18 * forward_width
            -
            0.15 * mean_perp
        )

        candidate = {
            "valid":
                True,

            "branch_id":
                branch_id,

            "tip_uv":
                tip,

            "coarse_uv":
                np.asarray(
                    [cu, cv],
                    dtype=np.float64,
                ),

            "axis":
                axis,

            "score":
                float(score),

            "alignment":
                alignment,

            "area":
                area,

            "span":
                span,

            "anchor_count":
                anchor_count,

            "front_width":
                front_width,

            "forward_width":
                forward_width,

            "refine_dist":
                refine_dist,

            "threshold":
                color_threshold,

            "tool_lab":
                tool_lab,

            "roi_rect":
                (
                    x0,
                    y0,
                    x1,
                    y1,
                ),

            "roi":
                roi,

            "mask":
                tool_mask,
        }

        if (
            best is None
            or
            candidate["score"]
            >
            best["score"]
        ):

            best = candidate

    return best



# ============================================================
# Display helpers
# ============================================================

def draw_cross(
    img,
    uv,
    color,
    size=14,
    thickness=2,
):

    u = int(
        round(
            float(
                uv[0]
            )
        )
    )

    v = int(
        round(
            float(
                uv[1]
            )
        )
    )

    cv2.drawMarker(
        img,
        (u, v),
        color,
        cv2.MARKER_CROSS,
        size,
        thickness,
    )


def build_debug_view(
    result,
):

    if result is None:

        return np.zeros(
            (
                360,
                360,
                3,
            ),
            dtype=np.uint8,
        )

    roi = result[
        "roi"
    ].copy()

    mask = result[
        "mask"
    ]

    overlay = roi.copy()

    overlay[
        mask > 0
    ] = (
        0,
        255,
        255,
    )

    view = cv2.addWeighted(
        roi,
        0.65,
        overlay,
        0.35,
        0.0,
    )

    x0, y0, _, _ = (
        result[
            "roi_rect"
        ]
    )

    tip_local = (
        result["tip_uv"]
        -
        np.asarray(
            [x0, y0],
            dtype=np.float64,
        )
    )

    coarse_local = (
        result["coarse_uv"]
        -
        np.asarray(
            [x0, y0],
            dtype=np.float64,
        )
    )

    draw_cross(
        view,
        coarse_local,
        (255, 0, 255),
        16,
        1,
    )

    draw_cross(
        view,
        tip_local,
        (0, 255, 0),
        22,
        2,
    )

    text = (
        f"B{result['branch_id']} "
        f"score={result['score']:.1f} "
        f"d={result['refine_dist']:.1f}px "
        f"align={result['alignment']:.2f}"
    )

    cv2.putText(
        view,
        text,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1,
    )

    return cv2.resize(
        view,
        (440, 440),
        interpolation=cv2.INTER_NEAREST,
    )


# ============================================================
# Main
# ============================================================

def main():

    K, dist = load_intrinsics()

    p_tip_marker = (
        load_coarse_tip_marker()
    )

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2,
    )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FRAME_FPS,
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Cannot open camera"
        )

    print()
    print(
        "=============================================="
    )
    print(
        "ARUCO COARSE + LOCAL TIP REFINE V2"
    )
    print(
        "=============================================="
    )

    print(
        "Marker = 20 mm ID20"
    )

    print(
        "coarse tip [mm] =",
        p_tip_marker * 1000.0,
    )

    print()
    print(
        "MAGENTA = ArUco coarse prediction"
    )
    print(
        "GREEN   = local real-tip refinement"
    )
    print(
        "CYAN    = EMA display point"
    )
    print(
        "YELLOW  = local search ROI"
    )
    print()
    print(
        "S = save debug images"
    )
    print(
        "Q/ESC = quit"
    )
    print()

    ema_tip = None
    last_valid_branch = None

    t_prev = time.time()
    fps = 0.0

    while True:

        ok, frame = cap.read()

        if (
            not ok
            or
            frame is None
        ):
            continue

        display = frame.copy()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        corners = detect_marker(
            gray
        )

        best_result = None

        candidate_results = []

        if corners is not None:

            cv2.polylines(
                display,
                [
                    corners
                    .astype(np.int32)
                    .reshape(
                        -1,
                        1,
                        2,
                    )
                ],
                True,
                (0, 255, 255),
                2,
            )

            marker_center = np.mean(
                corners,
                axis=0,
            )

            pose_candidates = (
                solve_pose_candidates(
                    corners,
                    K,
                    dist,
                )
            )

            # ------------------------------------------------
            # IMPORTANT:
            # do not trust only one IPPE branch.
            # Test BOTH branches through local real-tip vision.
            # ------------------------------------------------

            for pose in pose_candidates:

                coarse_uv = project_point(
                    p_tip_marker,
                    pose,
                    K,
                    dist,
                )

                if not (
                    0
                    <=
                    coarse_uv[0]
                    <
                    FRAME_WIDTH
                    and
                    0
                    <=
                    coarse_uv[1]
                    <
                    FRAME_HEIGHT
                ):
                    continue

                draw_cross(
                    display,
                    coarse_uv,
                    (255, 0, 255),
                    14,
                    1,
                )

                result = local_refine(
                    frame,
                    coarse_uv,
                    marker_center,
                    pose[
                        "branch_id"
                    ],
                )

                if result is None:
                    continue

                result[
                    "marker_reproj_px"
                ] = pose[
                    "marker_reproj_px"
                ]

                # Small temporal preference only.
                if (
                    last_valid_branch
                    is not None
                    and
                    result["branch_id"]
                    ==
                    last_valid_branch
                ):

                    result["score"] += 8.0

                candidate_results.append(
                    result
                )

            if candidate_results:

                candidate_results.sort(
                    key=lambda x:
                    x["score"],
                    reverse=True,
                )

                best_result = (
                    candidate_results[0]
                )

        # ====================================================
        # Valid local Tip
        # ====================================================

        if best_result is not None:

            (
                x0,
                y0,
                x1,
                y1,
            ) = best_result[
                "roi_rect"
            ]

            cv2.rectangle(
                display,
                (x0, y0),
                (
                    x1 - 1,
                    y1 - 1,
                ),
                (0, 255, 255),
                1,
            )

            # ArUco prediction
            draw_cross(
                display,
                best_result[
                    "coarse_uv"
                ],
                (255, 0, 255),
                18,
                1,
            )

            # Actual locally detected Tip
            draw_cross(
                display,
                best_result[
                    "tip_uv"
                ],
                (0, 255, 0),
                24,
                2,
            )

            # EMA visualization
            if ema_tip is None:

                ema_tip = (
                    best_result[
                        "tip_uv"
                    ].copy()
                )

            else:

                ema_tip = (
                    EMA_ALPHA
                    *
                    best_result[
                        "tip_uv"
                    ]
                    +
                    (
                        1.0
                        -
                        EMA_ALPHA
                    )
                    *
                    ema_tip
                )

            draw_cross(
                display,
                ema_tip,
                (255, 255, 0),
                16,
                2,
            )

            last_valid_branch = (
                best_result[
                    "branch_id"
                ]
            )

            cv2.putText(
                display,
                (
                    f"VALID "
                    f"B{best_result['branch_id']} "
                    f"score="
                    f"{best_result['score']:.1f} "
                    f"coarse->tip="
                    f"{best_result['refine_dist']:.1f}px "
                    f"align="
                    f"{best_result['alignment']:.2f}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        # ====================================================
        # Invalid:
        # DO NOT reuse stale Tip
        # ====================================================

        else:

            ema_tip = None
            last_valid_branch = None

            cv2.putText(
                display,
                "TIP REFINE INVALID",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        # ====================================================
        # FPS
        # ====================================================

        now = time.time()

        dt = now - t_prev

        if dt > 0:

            instant_fps = 1.0 / dt

            if fps == 0.0:
                fps = instant_fps

            else:

                fps = (
                    0.1
                    *
                    instant_fps
                    +
                    0.9
                    *
                    fps
                )

        t_prev = now

        cv2.putText(
            display,
            (
                f"fps={fps:.1f} "
                f"candidates="
                f"{len(candidate_results)}"
            ),
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            display,
            (
                "MAGENTA coarse | "
                "GREEN refined | "
                "CYAN EMA"
            ),
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )

        cv2.imshow(
            WINDOW,
            display,
        )

        cv2.imshow(
            DEBUG_WINDOW,
            build_debug_view(
                best_result
            ),
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            ord("Q"),
            27,
        ):
            break

        if key in (
            ord("s"),
            ord("S"),
        ):

            cv2.imwrite(
                "/tmp/r1a7_tip_refine_full.png",
                display,
            )

            cv2.imwrite(
                "/tmp/r1a7_tip_refine_roi.png",
                build_debug_view(
                    best_result
                ),
            )

            print(
                "[SAVE] "
                "/tmp/r1a7_tip_refine_full.png"
            )

            print(
                "[SAVE] "
                "/tmp/r1a7_tip_refine_roi.png"
            )

    cap.release()

    cv2.destroyAllWindows()

    print(
        "[TIP] stopped"
    )


if __name__ == "__main__":
    main()
