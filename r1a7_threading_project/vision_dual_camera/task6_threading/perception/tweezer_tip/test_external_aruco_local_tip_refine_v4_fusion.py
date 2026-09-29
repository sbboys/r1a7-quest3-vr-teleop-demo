#!/usr/bin/env python3

import csv
import sys
import time
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
import test_external_aruco_local_tip_refine_v3_hybrid as v3


# ============================================================
# Configuration
# ============================================================

WINDOW = "TIP REFINE V4 FUSION"
DEBUG_WINDOW = "TIP V4 DEBUG"

LOG_FRAMES = 500

OUTPUT_CSV = Path(
    "/tmp/r1a7_tip_refine_v4_fusion_500.csv"
)

# STRICT and EDGE are assumed to refer to same physical Tip
# when their outputs are within this distance.
AGREE_PX = 14.0

# Temporal prior is NEVER used as current measurement.
# It is only used to reject implausible current-frame results.
MAX_PRIOR_AGE = 10

# EDGE-only acceptance
EDGE_ONLY_ALIGN_MIN = 0.94
EDGE_ONLY_COARSE_MAX = 42.0

# Cold start EDGE-only is stricter.
EDGE_COLD_ALIGN_MIN = 0.98
EDGE_COLD_COARSE_MAX = 25.0

# STRICT-only acceptance
STRICT_ONLY_ALIGN_MIN = 0.75
STRICT_ONLY_COARSE_MAX = 48.0


MODE_BASE_SCORE = {
    "BOTH": 100.0,
    "EDGE_ONLY": 72.0,
    "STRICT_ONLY": 55.0,
}


# ============================================================
# Drawing
# ============================================================

def draw_cross(
    image,
    uv,
    color,
    size=18,
    thickness=2,
):

    if uv is None:
        return

    u = int(round(float(uv[0])))
    vv = int(round(float(uv[1])))

    cv2.drawMarker(
        image,
        (u, vv),
        color,
        cv2.MARKER_CROSS,
        size,
        thickness,
    )


# ============================================================
# Fuse STRICT + EDGE for one IPPE branch
# ============================================================

def fuse_branch(
    strict_result,
    edge_result,
    branch_id,
    coarse_uv,
    previous_tip,
    previous_branch,
    prior_age,
):

    strict_valid = (
        strict_result is not None
    )

    edge_valid = (
        edge_result is not None
    )

    agree_dist = float("nan")

    # --------------------------------------------------------
    # BOTH valid
    # --------------------------------------------------------

    if strict_valid and edge_valid:

        strict_tip = np.asarray(
            strict_result["tip_uv"],
            dtype=np.float64,
        )

        edge_tip = np.asarray(
            edge_result["tip_uv"],
            dtype=np.float64,
        )

        agree_dist = float(
            np.linalg.norm(
                strict_tip
                -
                edge_tip
            )
        )

        # They see different physical structures.
        # Do not guess.
        if agree_dist > AGREE_PX:

            return {
                "valid": False,
                "conflict": True,
                "branch_id": branch_id,
                "strict": strict_result,
                "edge": edge_result,
                "agree_dist": agree_dist,
            }

        # Experimental observation:
        # EDGE corresponds better to true physical endpoint.
        final_tip = edge_tip.copy()

        mode = "BOTH"

        final_alignment = min(
            float(
                strict_result[
                    "alignment"
                ]
            ),
            float(
                edge_result[
                    "alignment"
                ]
            ),
        )

        coarse_dist = float(
            np.linalg.norm(
                final_tip
                -
                coarse_uv
            )
        )

        quality = (
            MODE_BASE_SCORE["BOTH"]
            +
            20.0 * final_alignment
            -
            1.5 * agree_dist
            -
            0.15 * coarse_dist
        )

    # --------------------------------------------------------
    # EDGE only
    # --------------------------------------------------------

    elif edge_valid:

        final_tip = np.asarray(
            edge_result["tip_uv"],
            dtype=np.float64,
        )

        alignment = float(
            edge_result[
                "alignment"
            ]
        )

        coarse_dist = float(
            edge_result[
                "coarse_dist"
            ]
        )

        have_prior = (
            previous_tip is not None
            and
            prior_age <= MAX_PRIOR_AGE
        )

        # With temporal history, EDGE can operate as fallback.
        if have_prior:

            if (
                alignment
                <
                EDGE_ONLY_ALIGN_MIN
                or
                coarse_dist
                >
                EDGE_ONLY_COARSE_MAX
            ):

                return None

        # Without history, require much stronger evidence.
        else:

            if (
                alignment
                <
                EDGE_COLD_ALIGN_MIN
                or
                coarse_dist
                >
                EDGE_COLD_COARSE_MAX
            ):

                return None

        mode = "EDGE_ONLY"

        final_alignment = alignment

        quality = (
            MODE_BASE_SCORE["EDGE_ONLY"]
            +
            20.0 * alignment
            -
            0.20 * coarse_dist
        )

    # --------------------------------------------------------
    # STRICT only
    # --------------------------------------------------------

    elif strict_valid:

        final_tip = np.asarray(
            strict_result["tip_uv"],
            dtype=np.float64,
        )

        alignment = float(
            strict_result[
                "alignment"
            ]
        )

        coarse_dist = float(
            strict_result[
                "refine_dist"
            ]
        )

        if (
            alignment
            <
            STRICT_ONLY_ALIGN_MIN
            or
            coarse_dist
            >
            STRICT_ONLY_COARSE_MAX
        ):

            return None

        mode = "STRICT_ONLY"

        final_alignment = alignment

        quality = (
            MODE_BASE_SCORE["STRICT_ONLY"]
            +
            20.0 * alignment
            -
            0.20 * coarse_dist
        )

    else:

        return None

    # ========================================================
    # Temporal consistency
    #
    # Previous Tip is ONLY a prior.
    # It is never used as current Tip.
    # ========================================================

    temporal_jump = float("nan")

    if (
        previous_tip is not None
        and
        prior_age <= MAX_PRIOR_AGE
    ):

        temporal_jump = float(
            np.linalg.norm(
                final_tip
                -
                previous_tip
            )
        )

        # Soft selection penalty.
        quality -= (
            1.8
            *
            temporal_jump
        )

    # Small branch continuity preference.
    if (
        previous_branch is not None
        and
        branch_id
        ==
        previous_branch
    ):

        quality += 5.0

    return {
        "valid": True,

        "conflict": False,

        "mode": mode,

        "branch_id": int(
            branch_id
        ),

        "tip_uv": final_tip,

        "coarse_uv": np.asarray(
            coarse_uv,
            dtype=np.float64,
        ),

        "coarse_dist": coarse_dist,

        "alignment": final_alignment,

        "quality": float(
            quality
        ),

        "temporal_jump":
            temporal_jump,

        "agree_dist":
            agree_dist,

        "strict":
            strict_result,

        "edge":
            edge_result,
    }


# ============================================================
# Debug view
# ============================================================

def build_debug_view(
    frame,
    result,
):

    if result is None:

        canvas = np.zeros(
            (440, 440, 3),
            dtype=np.uint8,
        )

        cv2.putText(
            canvas,
            "INVALID",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.90,
            (0, 0, 255),
            2,
        )

        return canvas

    coarse = np.asarray(
        result[
            "coarse_uv"
        ],
        dtype=np.float64,
    )

    half = 55

    h, w = frame.shape[:2]

    x0 = max(
        0,
        int(
            round(
                coarse[0]
                -
                half
            )
        ),
    )

    y0 = max(
        0,
        int(
            round(
                coarse[1]
                -
                half
            )
        ),
    )

    x1 = min(
        w,
        int(
            round(
                coarse[0]
                +
                half
            )
        ),
    )

    y1 = min(
        h,
        int(
            round(
                coarse[1]
                +
                half
            )
        ),
    )

    crop = frame[
        y0:y1,
        x0:x1,
    ].copy()

    if crop.size == 0:

        return np.zeros(
            (440, 440, 3),
            dtype=np.uint8,
        )

    offset = np.asarray(
        [x0, y0],
        dtype=np.float64,
    )

    # MAGENTA = coarse
    draw_cross(
        crop,
        coarse - offset,
        (255, 0, 255),
        16,
        1,
    )

    strict = result.get(
        "strict"
    )

    edge = result.get(
        "edge"
    )

    # GREEN = raw STRICT
    if strict is not None:

        draw_cross(
            crop,
            np.asarray(
                strict["tip_uv"]
            )
            -
            offset,
            (0, 255, 0),
            18,
            1,
        )

    # ORANGE = raw EDGE
    if edge is not None:

        draw_cross(
            crop,
            np.asarray(
                edge["tip_uv"]
            )
            -
            offset,
            (0, 165, 255),
            18,
            1,
        )

    # CYAN = FINAL
    draw_cross(
        crop,
        result["tip_uv"]
        -
        offset,
        (255, 255, 0),
        26,
        2,
    )

    cv2.putText(
        crop,
        (
            f"{result['mode']} "
            f"B{result['branch_id']}"
        ),
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    if np.isfinite(
        result[
            "agree_dist"
        ]
    ):

        cv2.putText(
            crop,
            (
                f"S-E="
                f"{result['agree_dist']:.1f}px"
            ),
            (5, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )

    return cv2.resize(
        crop,
        (440, 440),
        interpolation=
            cv2.INTER_NEAREST,
    )


# ============================================================
# Logging summary
# ============================================================

def print_summary(
    rows,
):

    total = len(rows)

    valid_rows = [
        r
        for r in rows
        if r["valid"] == 1
    ]

    print()
    print(
        "=============================================="
    )
    print(
        "V4 FUSION 500-FRAME RESULT"
    )
    print(
        "=============================================="
    )

    print(
        "frames =",
        total,
    )

    print(
        "valid =",
        len(valid_rows),
    )

    invalid = (
        total
        -
        len(valid_rows)
    )

    print(
        "invalid =",
        invalid,
    )

    if total:

        print(
            f"valid ratio = "
            f"{100.0*len(valid_rows)/total:.2f}%"
        )

    for mode in (
        "BOTH",
        "EDGE_ONLY",
        "STRICT_ONLY",
    ):

        count = sum(
            1
            for r in rows
            if r["mode"]
            ==
            mode
        )

        print(
            f"{mode:11s} = "
            f"{count}"
        )

    conflict_count = sum(
        int(
            r["conflict"]
        )
        for r in rows
    )

    print(
        "conflict frames =",
        conflict_count,
    )

    if valid_rows:

        coarse = np.asarray(
            [
                r[
                    "coarse_dist"
                ]
                for r in valid_rows
            ],
            dtype=np.float64,
        )

        jumps = np.asarray(
            [
                r["jump_px"]
                for r in valid_rows
                if np.isfinite(
                    r["jump_px"]
                )
            ],
            dtype=np.float64,
        )

        agreements = np.asarray(
            [
                r[
                    "agree_dist"
                ]
                for r in valid_rows
                if (
                    r["mode"]
                    ==
                    "BOTH"
                    and
                    np.isfinite(
                        r[
                            "agree_dist"
                        ]
                    )
                )
            ],
            dtype=np.float64,
        )

        print()
        print(
            "coarse -> FINAL:"
        )

        print(
            f" mean   = "
            f"{np.mean(coarse):.3f} px"
        )

        print(
            f" p95    = "
            f"{np.percentile(coarse,95):.3f} px"
        )

        print(
            f" max    = "
            f"{np.max(coarse):.3f} px"
        )

        if agreements.size:

            print()
            print(
                "STRICT vs EDGE agreement:"
            )

            print(
                f" mean   = "
                f"{np.mean(agreements):.3f} px"
            )

            print(
                f" p95    = "
                f"{np.percentile(agreements,95):.3f} px"
            )

            print(
                f" max    = "
                f"{np.max(agreements):.3f} px"
            )

        if jumps.size:

            print()
            print(
                "FINAL frame-to-frame jump:"
            )

            print(
                f" mean   = "
                f"{np.mean(jumps):.3f} px"
            )

            print(
                f" median = "
                f"{np.median(jumps):.3f} px"
            )

            print(
                f" p95    = "
                f"{np.percentile(jumps,95):.3f} px"
            )

            print(
                f" max    = "
                f"{np.max(jumps):.3f} px"
            )

            print(
                " >10 px =",
                int(
                    np.sum(
                        jumps
                        >
                        10.0
                    )
                ),
            )

            print(
                " >20 px =",
                int(
                    np.sum(
                        jumps
                        >
                        20.0
                    )
                ),
            )

    # --------------------------------------------------------
    # Branch switches
    # --------------------------------------------------------

    switches = 0

    last_branch = None

    for row in rows:

        if row["valid"] == 0:

            continue

        branch = row[
            "branch"
        ]

        if (
            last_branch is not None
            and
            branch
            !=
            last_branch
        ):

            switches += 1

        last_branch = branch

    print()
    print(
        "branch switches =",
        switches,
    )

    # --------------------------------------------------------
    # Longest INVALID sequence
    # --------------------------------------------------------

    longest = 0
    current = 0

    for row in rows:

        if row["valid"] == 0:

            current += 1
            longest = max(
                longest,
                current,
            )

        else:

            current = 0

    print(
        "max consecutive invalid =",
        longest,
        "frames",
    )

    print(
        "=============================================="
    )


def save_csv(
    rows,
):

    fields = [
        "index",
        "time_s",
        "valid",
        "mode",
        "branch",
        "conflict",
        "candidate_count",
        "final_u",
        "final_v",
        "coarse_u",
        "coarse_v",
        "coarse_dist",
        "strict_u",
        "strict_v",
        "edge_u",
        "edge_v",
        "agree_dist",
        "alignment",
        "quality",
        "jump_px",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )

    print(
        "[SAVE]",
        OUTPUT_CSV,
    )


# ============================================================
# Main
# ============================================================

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

    lifetime = {
        "BOTH": 0,
        "EDGE_ONLY": 0,
        "STRICT_ONLY": 0,
        "INVALID": 0,
        "CONFLICT": 0,
    }

    logging = False
    rows = []
    log_start = None

    print()
    print(
        "=============================================="
    )
    print(
        "TIP REFINE V4 FUSION"
    )
    print(
        "=============================================="
    )

    print(
        "MAGENTA = ArUco coarse"
    )

    print(
        "GREEN   = raw STRICT"
    )

    print(
        "ORANGE  = raw EDGE"
    )

    print(
        "CYAN    = FINAL fused Tip"
    )

    print()
    print(
        "L = start 500-frame logging"
    )

    print(
        "S = save screenshot"
    )

    print(
        "Q / ESC = quit"
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

        fused_candidates = []

        frame_conflict = False

        selected = None

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

            # =================================================
            # STRICT and EDGE run on EVERY branch EVERY frame
            # =================================================

            for pose in poses:

                branch_id = int(
                    pose[
                        "branch_id"
                    ]
                )

                coarse_uv = (
                    v2.project_point(
                        p_tip_marker,
                        pose,
                        K,
                        dist,
                    )
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

                strict_result = (
                    v2.local_refine(
                        frame,
                        coarse_uv,
                        marker_center,
                        branch_id,
                    )
                )

                edge_result = (
                    v3.edge_fallback(
                        frame,
                        coarse_uv,
                        marker_center,
                        previous_tip=
                            temporal_prior,
                    )
                )

                if edge_result is not None:

                    edge_result[
                        "branch_id"
                    ] = branch_id

                fused = fuse_branch(
                    strict_result,
                    edge_result,
                    branch_id,
                    coarse_uv,
                    previous_tip,
                    previous_branch,
                    prior_age,
                )

                if fused is None:
                    continue

                if fused.get(
                    "conflict",
                    False,
                ):

                    frame_conflict = True
                    continue

                if fused.get(
                    "valid",
                    False,
                ):

                    fused_candidates.append(
                        fused
                    )

            if fused_candidates:

                fused_candidates.sort(
                    key=lambda x:
                    x["quality"],
                    reverse=True,
                )

                selected = (
                    fused_candidates[0]
                )

        # =====================================================
        # Valid FINAL measurement
        # =====================================================

        jump_px = float("nan")

        if selected is not None:

            final_tip = np.asarray(
                selected["tip_uv"],
                dtype=np.float64,
            )

            if (
                previous_tip is not None
                and
                prior_age
                <=
                MAX_PRIOR_AGE
            ):

                jump_px = float(
                    np.linalg.norm(
                        final_tip
                        -
                        previous_tip
                    )
                )

            strict = selected.get(
                "strict"
            )

            edge = selected.get(
                "edge"
            )

            # Raw STRICT
            if strict is not None:

                draw_cross(
                    display,
                    strict[
                        "tip_uv"
                    ],
                    (0, 255, 0),
                    15,
                    1,
                )

            # Raw EDGE
            if edge is not None:

                draw_cross(
                    display,
                    edge[
                        "tip_uv"
                    ],
                    (0, 165, 255),
                    15,
                    1,
                )

            # Coarse ArUco prediction
            draw_cross(
                display,
                selected[
                    "coarse_uv"
                ],
                (255, 0, 255),
                15,
                1,
            )

            # FINAL:
            # always CYAN, regardless of underlying detector.
            draw_cross(
                display,
                final_tip,
                (255, 255, 0),
                28,
                2,
            )

            mode = selected[
                "mode"
            ]

            lifetime[
                mode
            ] += 1

            previous_tip = (
                final_tip.copy()
            )

            previous_branch = (
                selected[
                    "branch_id"
                ]
            )

            prior_age = 0

            text = (
                f"FINAL {mode} "
                f"B{selected['branch_id']} "
                f"d={selected['coarse_dist']:.1f}px "
                f"jump="
            )

            if np.isfinite(
                jump_px
            ):

                text += (
                    f"{jump_px:.1f}px"
                )

            else:

                text += "NA"

            if np.isfinite(
                selected[
                    "agree_dist"
                ]
            ):

                text += (
                    f" agree="
                    f"{selected['agree_dist']:.1f}px"
                )

            cv2.putText(
                display,
                text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (255, 255, 0),
                2,
            )

        # =====================================================
        # INVALID
        # =====================================================

        else:

            mode = "INVALID"

            lifetime[
                "INVALID"
            ] += 1

            if frame_conflict:

                lifetime[
                    "CONFLICT"
                ] += 1

            # No stale output.
            prior_age += 1

            if (
                prior_age
                >
                MAX_PRIOR_AGE
            ):

                previous_tip = None
                previous_branch = None

            cv2.putText(
                display,
                (
                    "FINAL INVALID"
                    +
                    (
                        " CONFLICT"
                        if frame_conflict
                        else ""
                    )
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
            )

        # =====================================================
        # Lifetime counters
        # =====================================================

        cv2.putText(
            display,
            (
                f"both={lifetime['BOTH']} "
                f"edge={lifetime['EDGE_ONLY']} "
                f"strict={lifetime['STRICT_ONLY']} "
                f"invalid={lifetime['INVALID']} "
                f"conflict={lifetime['CONFLICT']}"
            ),
            (20, 67),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
        )

        # =====================================================
        # Logging
        # =====================================================

        if logging:

            idx = (
                len(rows) + 1
            )

            if selected is not None:

                strict = selected.get(
                    "strict"
                )

                edge = selected.get(
                    "edge"
                )

                strict_uv = (
                    np.asarray(
                        strict["tip_uv"]
                    )
                    if strict is not None
                    else
                    np.asarray(
                        [np.nan, np.nan]
                    )
                )

                edge_uv = (
                    np.asarray(
                        edge["tip_uv"]
                    )
                    if edge is not None
                    else
                    np.asarray(
                        [np.nan, np.nan]
                    )
                )

                row = {
                    "index":
                        idx,

                    "time_s":
                        time.time()
                        -
                        log_start,

                    "valid":
                        1,

                    "mode":
                        selected[
                            "mode"
                        ],

                    "branch":
                        selected[
                            "branch_id"
                        ],

                    "conflict":
                        int(
                            frame_conflict
                        ),

                    "candidate_count":
                        len(
                            fused_candidates
                        ),

                    "final_u":
                        float(
                            selected[
                                "tip_uv"
                            ][0]
                        ),

                    "final_v":
                        float(
                            selected[
                                "tip_uv"
                            ][1]
                        ),

                    "coarse_u":
                        float(
                            selected[
                                "coarse_uv"
                            ][0]
                        ),

                    "coarse_v":
                        float(
                            selected[
                                "coarse_uv"
                            ][1]
                        ),

                    "coarse_dist":
                        float(
                            selected[
                                "coarse_dist"
                            ]
                        ),

                    "strict_u":
                        float(
                            strict_uv[0]
                        ),

                    "strict_v":
                        float(
                            strict_uv[1]
                        ),

                    "edge_u":
                        float(
                            edge_uv[0]
                        ),

                    "edge_v":
                        float(
                            edge_uv[1]
                        ),

                    "agree_dist":
                        float(
                            selected[
                                "agree_dist"
                            ]
                        ),

                    "alignment":
                        float(
                            selected[
                                "alignment"
                            ]
                        ),

                    "quality":
                        float(
                            selected[
                                "quality"
                            ]
                        ),

                    "jump_px":
                        float(
                            jump_px
                        ),
                }

            else:

                row = {
                    "index":
                        idx,

                    "time_s":
                        time.time()
                        -
                        log_start,

                    "valid":
                        0,

                    "mode":
                        "INVALID",

                    "branch":
                        -1,

                    "conflict":
                        int(
                            frame_conflict
                        ),

                    "candidate_count":
                        len(
                            fused_candidates
                        ),

                    "final_u":
                        np.nan,

                    "final_v":
                        np.nan,

                    "coarse_u":
                        np.nan,

                    "coarse_v":
                        np.nan,

                    "coarse_dist":
                        np.nan,

                    "strict_u":
                        np.nan,

                    "strict_v":
                        np.nan,

                    "edge_u":
                        np.nan,

                    "edge_v":
                        np.nan,

                    "agree_dist":
                        np.nan,

                    "alignment":
                        np.nan,

                    "quality":
                        np.nan,

                    "jump_px":
                        np.nan,
                }

            rows.append(
                row
            )

            cv2.putText(
                display,
                (
                    f"LOG "
                    f"{len(rows)}/"
                    f"{LOG_FRAMES}"
                ),
                (20, 97),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 165, 255),
                2,
            )

            if (
                len(rows)
                >=
                LOG_FRAMES
            ):

                logging = False

                save_csv(
                    rows
                )

                print_summary(
                    rows
                )

                print()
                print(
                    "[V4] 500-frame test complete."
                )

        else:

            cv2.putText(
                display,
                "L=500-frame test  S=save  Q=quit",
                (20, 97),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
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
                frame,
                selected,
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
                "/tmp/r1a7_tip_refine_v4_fusion.png",
                display,
            )

            cv2.imwrite(
                "/tmp/r1a7_tip_refine_v4_debug.png",
                build_debug_view(
                    frame,
                    selected,
                ),
            )

            print(
                "[SAVE] "
                "/tmp/r1a7_tip_refine_v4_fusion.png"
            )

        if key in (
            ord("l"),
            ord("L"),
        ):

            rows = []

            logging = True

            log_start = (
                time.time()
            )

            print()
            print(
                "[V4] Started 500-frame logging."
            )

            print(
                "[V4] Move robot slowly through "
                "normal + fixture poses."
            )

    cap.release()

    cv2.destroyAllWindows()

    print(
        "[V4] stopped"
    )


if __name__ == "__main__":
    main()
