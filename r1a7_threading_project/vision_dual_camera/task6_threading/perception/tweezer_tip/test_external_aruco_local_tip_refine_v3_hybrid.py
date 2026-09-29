#!/usr/bin/env python3

import sys
import time
import math
from pathlib import Path

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(THIS_DIR),
    )

import test_external_aruco_local_tip_refine_v2 as v2


WINDOW = "ARUCO LOCAL TIP REFINE V3 HYBRID"
DEBUG_WINDOW = "TIP V3 DEBUG"

# Edge fallback settings
EDGE_ROI_HALF = 55

EDGE_CORRIDOR_HALF = 17.0

EDGE_S_BACK = 65.0
EDGE_S_FORWARD = 38.0

EDGE_MIN_ALIGNMENT = 0.86
EDGE_MIN_LENGTH = 11.0

EDGE_MAX_COARSE_DIST = 45.0

# If we have recent Tip history, current-frame fallback
# observation must remain temporally plausible.
EDGE_MAX_TEMPORAL_JUMP = 12.0

# Keep temporal prior through short INVALID gaps,
# but NEVER output stale Tip.
MAX_PRIOR_AGE = 10


def draw_cross(
    img,
    uv,
    color,
    size=18,
    thickness=2,
):

    u = int(
        round(
            float(uv[0])
        )
    )

    vv = int(
        round(
            float(uv[1])
        )
    )

    cv2.drawMarker(
        img,
        (u, vv),
        color,
        cv2.MARKER_CROSS,
        size,
        thickness,
    )


def edge_fallback(
    frame,
    coarse_uv,
    marker_center_uv,
    previous_tip=None,
):

    h_img, w_img = frame.shape[:2]

    cu = float(coarse_uv[0])
    cvv = float(coarse_uv[1])

    if not (
        0 <= cu < w_img
        and
        0 <= cvv < h_img
    ):
        return None

    # ========================================================
    # Tool image direction
    # ========================================================

    axis = np.asarray(
        [
            cu - marker_center_uv[0],
            cvv - marker_center_uv[1],
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

    # ========================================================
    # ROI
    # ========================================================

    x0 = max(
        0,
        int(
            math.floor(
                cu - EDGE_ROI_HALF
            )
        ),
    )

    y0 = max(
        0,
        int(
            math.floor(
                cvv - EDGE_ROI_HALF
            )
        ),
    )

    x1 = min(
        w_img,
        int(
            math.ceil(
                cu + EDGE_ROI_HALF + 1
            )
        ),
    )

    y1 = min(
        h_img,
        int(
            math.ceil(
                cvv + EDGE_ROI_HALF + 1
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

    # ========================================================
    # Edge extraction
    # ========================================================

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    median_intensity = float(
        np.median(gray)
    )

    lower = int(
        max(
            18,
            0.55 * median_intensity,
        )
    )

    upper = int(
        min(
            255,
            max(
                lower + 20,
                1.25 * median_intensity,
            ),
        )
    )

    edges = cv2.Canny(
        gray,
        lower,
        upper,
        apertureSize=3,
        L2gradient=True,
    )

    # ========================================================
    # Keep only expected tool corridor
    # ========================================================

    yy, xx = np.mgrid[
        0:rh,
        0:rw,
    ]

    gx = xx.astype(
        np.float64
    ) + x0

    gy = yy.astype(
        np.float64
    ) + y0

    rx = gx - cu
    ry = gy - cvv

    s = (
        rx * axis[0]
        +
        ry * axis[1]
    )

    d = np.abs(
        rx * perp[0]
        +
        ry * perp[1]
    )

    corridor = (
        (s >= -EDGE_S_BACK)
        &
        (s <= EDGE_S_FORWARD)
        &
        (d <= EDGE_CORRIDOR_HALF)
    )

    edges[
        ~corridor
    ] = 0

    # ========================================================
    # Hough segments
    # ========================================================

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=8,
        minLineLength=8,
        maxLineGap=6,
    )

    if lines is None:
        return None

    best = None

    debug = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR,
    )

    for item in lines:

        x_a, y_a, x_b, y_b = (
            item[0]
        )

        p1 = np.asarray(
            [
                x_a + x0,
                y_a + y0,
            ],
            dtype=np.float64,
        )

        p2 = np.asarray(
            [
                x_b + x0,
                y_b + y0,
            ],
            dtype=np.float64,
        )

        vec = p2 - p1

        length = float(
            np.linalg.norm(vec)
        )

        if length < EDGE_MIN_LENGTH:
            continue

        direction = vec / length

        alignment = float(
            abs(
                np.dot(
                    direction,
                    axis,
                )
            )
        )

        if alignment < EDGE_MIN_ALIGNMENT:
            continue

        rel1 = (
            p1
            -
            np.asarray(
                [cu, cvv],
                dtype=np.float64,
            )
        )

        rel2 = (
            p2
            -
            np.asarray(
                [cu, cvv],
                dtype=np.float64,
            )
        )

        s1 = float(
            np.dot(
                rel1,
                axis,
            )
        )

        s2 = float(
            np.dot(
                rel2,
                axis,
            )
        )

        d1 = abs(
            float(
                np.dot(
                    rel1,
                    perp,
                )
            )
        )

        d2 = abs(
            float(
                np.dot(
                    rel2,
                    perp,
                )
            )
        )

        # Segment must extend backwards into the tweezer shaft.
        if min(
            s1,
            s2,
        ) > -12.0:
            continue

        # And it must reach reasonably close to the predicted Tip.
        if max(
            s1,
            s2,
        ) < -8.0:
            continue

        if (
            d1 > EDGE_CORRIDOR_HALF
            or
            d2 > EDGE_CORRIDOR_HALF
        ):
            continue

        # Forward endpoint is Tip candidate.
        if s1 >= s2:
            tip = p1
            forward_s = s1
        else:
            tip = p2
            forward_s = s2

        coarse_dist = float(
            np.linalg.norm(
                tip
                -
                np.asarray(
                    [cu, cvv],
                    dtype=np.float64,
                )
            )
        )

        if (
            coarse_dist
            >
            EDGE_MAX_COARSE_DIST
        ):
            continue

        temporal_jump = None

        if previous_tip is not None:

            temporal_jump = float(
                np.linalg.norm(
                    tip
                    -
                    previous_tip
                )
            )

            if (
                temporal_jump
                >
                EDGE_MAX_TEMPORAL_JUMP
            ):
                continue

        lateral = 0.5 * (
            d1 + d2
        )

        score = (
            110.0 * alignment
            +
            1.3 * length
            +
            0.25 * forward_s
            -
            0.50 * coarse_dist
            -
            0.70 * lateral
        )

        if temporal_jump is not None:

            score -= (
                1.7
                *
                temporal_jump
            )

        candidate = {
            "tip_uv":
                tip,

            "coarse_uv":
                np.asarray(
                    [cu, cvv],
                    dtype=np.float64,
                ),

            "score":
                float(score),

            "alignment":
                alignment,

            "length":
                length,

            "coarse_dist":
                coarse_dist,

            "temporal_jump":
                temporal_jump,

            "debug":
                debug,

            "roi_rect":
                (
                    x0,
                    y0,
                    x1,
                    y1,
                ),
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


def build_debug_view(
    strict_result,
    edge_result,
):

    if strict_result is not None:

        roi = strict_result[
            "roi"
        ].copy()

        mask = strict_result[
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

        cv2.putText(
            view,
            "STRICT V2",
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            1,
        )

    elif edge_result is not None:

        view = edge_result[
            "debug"
        ].copy()

        cv2.putText(
            view,
            "EDGE FALLBACK",
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 165, 255),
            1,
        )

    else:

        view = np.zeros(
            (
                110,
                110,
                3,
            ),
            dtype=np.uint8,
        )

        cv2.putText(
            view,
            "INVALID",
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 255),
            1,
        )

    return cv2.resize(
        view,
        (440, 440),
        interpolation=cv2.INTER_NEAREST,
    )


def main():

    K, dist = (
        v2.load_intrinsics()
    )

    p_tip_marker = (
        v2.load_coarse_tip_marker()
    )

    cap = cv2.VideoCapture(
        v2.CAMERA_DEVICE,
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
        v2.FRAME_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        v2.FRAME_HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        v2.FRAME_FPS,
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open camera"
        )

    previous_tip = None
    previous_branch = None
    prior_age = 999

    strict_count = 0
    edge_count = 0
    invalid_count = 0

    print()
    print(
        "=============================================="
    )
    print(
        "TIP REFINE V3 HYBRID"
    )
    print(
        "=============================================="
    )

    print(
        "GREEN  = strict V2 Tip"
    )

    print(
        "ORANGE = current-frame EDGE fallback Tip"
    )

    print(
        "MAGENTA = ArUco coarse Tip"
    )

    print(
        "RED INVALID = neither detector found current Tip"
    )

    print()

    while True:

        ok, frame = cap.read()

        if not ok:
            continue

        display = frame.copy()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        corners = v2.detect_marker(
            gray
        )

        strict_candidates = []
        edge_candidates = []

        best_strict = None
        best_edge = None
        best_pose = None

        marker_center = None

        if corners is not None:

            cv2.polylines(
                display,
                [
                    corners
                    .astype(np.int32)
                    .reshape(-1, 1, 2)
                ],
                True,
                (0, 255, 255),
                2,
            )

            marker_center = np.mean(
                corners,
                axis=0,
            )

            poses = (
                v2.solve_pose_candidates(
                    corners,
                    K,
                    dist,
                )
            )

            # =================================================
            # Stage 1: strict V2 detector
            # =================================================

            for pose in poses:

                coarse_uv = v2.project_point(
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
                    v2.FRAME_WIDTH
                    and
                    0
                    <=
                    coarse_uv[1]
                    <
                    v2.FRAME_HEIGHT
                ):
                    continue

                result = v2.local_refine(
                    frame,
                    coarse_uv,
                    marker_center,
                    pose[
                        "branch_id"
                    ],
                )

                if result is None:
                    continue

                result = dict(
                    result
                )

                result[
                    "pose"
                ] = pose

                # Temporal continuity affects selection only.
                if (
                    previous_tip is not None
                    and
                    prior_age
                    <=
                    MAX_PRIOR_AGE
                ):

                    jump = float(
                        np.linalg.norm(
                            result["tip_uv"]
                            -
                            previous_tip
                        )
                    )

                    result[
                        "selection_score"
                    ] = (
                        result["score"]
                        -
                        2.0
                        *
                        jump
                    )

                else:

                    result[
                        "selection_score"
                    ] = (
                        result["score"]
                    )

                if (
                    previous_branch
                    is not None
                    and
                    result["branch_id"]
                    ==
                    previous_branch
                ):

                    result[
                        "selection_score"
                    ] += 8.0

                strict_candidates.append(
                    result
                )

            if strict_candidates:

                strict_candidates.sort(
                    key=lambda x:
                    x[
                        "selection_score"
                    ],
                    reverse=True,
                )

                best_strict = (
                    strict_candidates[0]
                )

            # =================================================
            # Stage 2: only if strict V2 completely failed
            # =================================================

            if best_strict is None:

                temporal_prior = None

                if (
                    previous_tip is not None
                    and
                    prior_age
                    <=
                    MAX_PRIOR_AGE
                ):

                    temporal_prior = (
                        previous_tip
                    )

                for pose in poses:

                    coarse_uv = v2.project_point(
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
                        v2.FRAME_WIDTH
                        and
                        0
                        <=
                        coarse_uv[1]
                        <
                        v2.FRAME_HEIGHT
                    ):
                        continue

                    result = edge_fallback(
                        frame,
                        coarse_uv,
                        marker_center,
                        previous_tip=
                            temporal_prior,
                    )

                    if result is None:
                        continue

                    result[
                        "branch_id"
                    ] = pose[
                        "branch_id"
                    ]

                    result[
                        "pose"
                    ] = pose

                    selection_score = (
                        result[
                            "score"
                        ]
                    )

                    if (
                        previous_branch
                        is not None
                        and
                        pose[
                            "branch_id"
                        ]
                        ==
                        previous_branch
                    ):

                        selection_score += 7.0

                    result[
                        "selection_score"
                    ] = selection_score

                    edge_candidates.append(
                        result
                    )

                if edge_candidates:

                    edge_candidates.sort(
                        key=lambda x:
                        x[
                            "selection_score"
                        ],
                        reverse=True,
                    )

                    best_edge = (
                        edge_candidates[0]
                    )

        # =====================================================
        # Output current-frame observation
        # =====================================================

        if best_strict is not None:

            tip = np.asarray(
                best_strict[
                    "tip_uv"
                ],
                dtype=np.float64,
            )

            coarse = np.asarray(
                best_strict[
                    "coarse_uv"
                ],
                dtype=np.float64,
            )

            draw_cross(
                display,
                coarse,
                (255, 0, 255),
                16,
                1,
            )

            draw_cross(
                display,
                tip,
                (0, 255, 0),
                24,
                2,
            )

            mode = "STRICT"

            previous_tip = tip.copy()
            previous_branch = int(
                best_strict[
                    "branch_id"
                ]
            )

            prior_age = 0

            strict_count += 1

            cv2.putText(
                display,
                (
                    f"STRICT VALID B"
                    f"{previous_branch} "
                    f"d="
                    f"{best_strict['refine_dist']:.1f}px "
                    f"align="
                    f"{best_strict['alignment']:.2f}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 0),
                2,
            )

        elif best_edge is not None:

            tip = np.asarray(
                best_edge[
                    "tip_uv"
                ],
                dtype=np.float64,
            )

            coarse = np.asarray(
                best_edge[
                    "coarse_uv"
                ],
                dtype=np.float64,
            )

            draw_cross(
                display,
                coarse,
                (255, 0, 255),
                16,
                1,
            )

            draw_cross(
                display,
                tip,
                (0, 165, 255),
                24,
                2,
            )

            mode = "EDGE"

            previous_tip = tip.copy()

            previous_branch = int(
                best_edge[
                    "branch_id"
                ]
            )

            prior_age = 0

            edge_count += 1

            temporal_text = ""

            if (
                best_edge[
                    "temporal_jump"
                ]
                is not None
            ):

                temporal_text = (
                    f" jump="
                    f"{best_edge['temporal_jump']:.1f}"
                )

            cv2.putText(
                display,
                (
                    f"EDGE VALID B"
                    f"{previous_branch} "
                    f"d="
                    f"{best_edge['coarse_dist']:.1f}px "
                    f"align="
                    f"{best_edge['alignment']:.2f}"
                    f"{temporal_text}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 165, 255),
                2,
            )

        else:

            mode = "INVALID"

            invalid_count += 1

            # Important:
            # retain old point only as short-term PRIOR.
            # It is NOT drawn/output as current Tip.
            prior_age += 1

            if prior_age > MAX_PRIOR_AGE:

                previous_tip = None
                previous_branch = None

            cv2.putText(
                display,
                (
                    f"INVALID "
                    f"prior_age={prior_age}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            display,
            (
                f"strict={strict_count} "
                f"edge={edge_count} "
                f"invalid={invalid_count}"
            ),
            (20, 68),
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
                best_strict,
                best_edge,
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
                "/tmp/r1a7_tip_refine_v3_hybrid.png",
                display,
            )

            cv2.imwrite(
                "/tmp/r1a7_tip_refine_v3_debug.png",
                build_debug_view(
                    best_strict,
                    best_edge,
                ),
            )

            print(
                "[SAVE] "
                "/tmp/r1a7_tip_refine_v3_hybrid.png"
            )

    cap.release()

    cv2.destroyAllWindows()

    print(
        "[V3] stopped"
    )


if __name__ == "__main__":
    main()
