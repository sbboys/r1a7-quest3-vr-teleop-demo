#!/usr/bin/env python3

import argparse
import cv2
import json
import os
import sys
import time
import numpy as np
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TIP_DIR = (
    ROOT
    / "vision_dual_camera"
    / "task6_threading"
    / "perception"
    / "tweezer_tip"
)

TASK_DIR = (
    ROOT
    / "vision_dual_camera"
    / "task6_threading"
)

sys.path.insert(
    0,
    str(TIP_DIR),
)

sys.path.insert(
    0,
    str(TASK_DIR),
)


# ============================================================
# Reuse the already-tested V4 perception modules
# ============================================================

import test_external_aruco_local_tip_refine_v2 as v2
import test_external_aruco_local_tip_refine_v3_hybrid as v3
import test_external_aruco_local_tip_refine_v4_fusion as v4


# Reuse target-hole selection/tracking only.
# This module is NOT executed as a wrist controller here.
import task6_wrist_ibvs_controller as hole_tracker


# ============================================================
# Output
# ============================================================

DEFAULT_OUT = Path(
    "/tmp/r1a7_ibvs_velocity.json"
)


# ============================================================
# External image -> robot mapping
#
# already experimentally determined:
#
# image +U -> robot +X
# image +V -> robot -Z
# ============================================================

KP_X = 0.010
KP_Z = 0.010

MAX_SPEED_MM_S = 5.0

DEADBAND_U_PX = 3.0
DEADBAND_V_PX = 3.0


def limit(v):

    return float(
        np.clip(
            float(v),
            -MAX_SPEED_MM_S,
            MAX_SPEED_MM_S,
        )
    )


# ============================================================
# JSON
# ============================================================

def atomic_write_json(
    path,
    data,
):

    path = Path(path)

    tmp = Path(
        str(path)
        +
        ".tmp"
    )

    tmp.write_text(
        json.dumps(
            data,
            indent=2,
        )
    )

    os.replace(
        tmp,
        path,
    )


def write_disabled(
    path,
    reason,
):

    atomic_write_json(
        path,
        {
            "timestamp":
                time.time(),

            "enabled":
                False,

            "source":
                "external_aruco_v4_edge",

            "reason":
                str(reason),

            "vx_mm_s":
                0.0,

            "vy_mm_s":
                0.0,

            "vz_mm_s":
                0.0,
        },
    )


# ============================================================
# Hole tracker reset
# ============================================================


def detect_external_hole_ellipse(gray, center, roi_size=30):

    if center is None:
        return None

    cx, cy = center

    h, w = gray.shape

    x0=max(0, cx-roi_size)
    y0=max(0, cy-roi_size)

    x1=min(w, cx+roi_size)
    y1=min(h, cy+roi_size)

    crop=gray[y0:y1, x0:x1]

    if crop.size == 0:
        return None


    edges=cv2.Canny(
        crop,
        40,
        120
    )


    contours,_=cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    candidates=[]


    for cnt in contours:

        area=cv2.contourArea(cnt)

        if area < 2 or area > 800:
            continue

        if len(cnt)<5:
            continue


        ellipse=cv2.fitEllipse(cnt)

        ex,ey=(ellipse[0])

        ma,mi=ellipse[1]

        if ma<=0 or mi<=0:
            continue


        ratio=min(ma,mi)/max(ma,mi)

        if ratio < 0.65:
            continue


        dx = ex + x0 - cx
        dy = ey + y0 - cy

        dist = (
            dx**2
            +
            dy**2
        )

        # Reject ellipse candidates too far from click point
        if dist > 25**2:
            continue


        candidates.append(
            (
                ex+x0,
                ey+y0,
                (ma+mi)/4,
                dist
            )
        )


    if not candidates:
        return None


    candidates.sort(
        key=lambda x:x[3]
    )


    best=candidates[0]

    return (
        int(round(best[0])),
        int(round(best[1])),
        int(round(best[2]))
    )


def reset_hole_tracker():

    hole_tracker.hole_roi = None
    hole_tracker.hole_click = None

    hole_tracker.select_done = False

    hole_tracker.hole_center = None
    hole_tracker.hole_radius = None

    hole_tracker.reselect_mode = True

    # Important:
    # external camera must never enter wrist-tip calibration.
    hole_tracker.tip_reselect_mode = False


# ============================================================
# V4 FINAL TIP
# ============================================================

# ============================================================
# External V4 temporal safety gate
#
# A single V4 detection is NOT allowed to move the robot if it
# jumps far away from the last trusted tip.
# ============================================================

EXT_TIP_NORMAL_GATE_PX = 25.0
EXT_TIP_RELOCK_MAX_JUMP_PX = 80.0
EXT_TIP_RELOCK_FRAMES = 3
EXT_TIP_RELOCK_CLUSTER_RADIUS_PX = 5.0

# Short term external tip hold
EXT_TIP_HOLD_S = 0.30

# External Tip short-time hold prediction



def get_v4_tip(
    frame,
    gray,
    K,
    dist,
    p_tip_marker,
    previous_tip,
    previous_branch,
    prior_age,
):

    corners = v2.detect_marker(
        gray
    )

    if corners is None:

        return (
            None,
            None,
        )

    marker_center = np.mean(
        corners,
        axis=0,
    )

    poses = v2.solve_pose_candidates(
        corners,
        K,
        dist,
    )

    temporal_prior = None

    if (
        previous_tip is not None
        and
        prior_age
        <=
        v4.MAX_PRIOR_AGE
    ):

        temporal_prior = (
            previous_tip
        )

    candidates = []

    h, w = frame.shape[:2]

    for pose in poses:

        branch_id = int(
            pose["branch_id"]
        )

        coarse_uv = v2.project_point(
            p_tip_marker,
            pose,
            K,
            dist,
        )

        if not (
            0
            <=
            float(coarse_uv[0])
            <
            w
            and
            0
            <=
            float(coarse_uv[1])
            <
            h
        ):

            continue

        # ----------------------------------------------------
        # STRICT local detector
        # ----------------------------------------------------

        strict_result = v2.local_refine(
            frame,
            coarse_uv,
            marker_center,
            branch_id,
        )

        # ----------------------------------------------------
        # EDGE endpoint detector
        # ----------------------------------------------------

        edge_result = v3.edge_fallback(
            frame,
            coarse_uv,
            marker_center,
            previous_tip=
                temporal_prior,
        )

        if edge_result is not None:

            edge_result = dict(
                edge_result
            )

            edge_result[
                "branch_id"
            ] = branch_id

        # ----------------------------------------------------
        # Existing V4 fusion logic
        # ----------------------------------------------------

        fused = v4.fuse_branch(
            strict_result,
            edge_result,
            branch_id,
            coarse_uv,
            previous_tip,
            previous_branch,
            prior_age,
        )

        if (
            fused is not None
            and
            fused.get(
                "valid",
                False,
            )
        ):

            fused = dict(
                fused
            )

            fused[
                "coarse_uv"
            ] = np.asarray(
                coarse_uv,
                dtype=np.float64,
            )

            fused[
                "marker_reproj_px"
            ] = float(
                pose.get(
                    "marker_reproj_px",
                    0.0,
                )
            )

            candidates.append(
                fused
            )

    if not candidates:

        return (
            None,
            corners,
        )

    candidates.sort(
        key=lambda x:
        float(
            x.get(
                "quality",
                0.0,
            )
        ),
        reverse=True,
    )

    return (
        candidates[0],
        corners,
    )


# ============================================================
# Drawing
# ============================================================

def draw_cross(
    img,
    uv,
    color,
    size=24,
    thickness=2,
):

    if uv is None:
        return

    p = (
        int(
            round(
                float(uv[0])
            )
        ),
        int(
            round(
                float(uv[1])
            )
        ),
    )

    cv2.drawMarker(
        img,
        p,
        color,
        cv2.MARKER_CROSS,
        size,
        thickness,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="/dev/video16",
    )

    parser.add_argument(
        "--command-file",
        default=str(
            DEFAULT_OUT
        ),
    )

    args = parser.parse_args()

    command_file = Path(
        args.command_file
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    K, dist = (
        v2.load_intrinsics()
    )

    p_tip_marker = (
        v2.load_coarse_tip_marker()
    )

    print(
        "[EXT V4] coarse P_tip_marker [mm] =",
        p_tip_marker,
        flush=True,
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        args.device,
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
        1280,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30,
    )

    if not cap.isOpened():

        raise RuntimeError(
            "External camera open failed: "
            +
            args.device
        )

    # --------------------------------------------------------
    # Runtime state
    # --------------------------------------------------------

    previous_tip = None
    previous_branch = None

    # External tip hold state
    external_last_valid_tip = None
    external_last_valid_time = None
    external_tip_velocity = np.zeros(2, dtype=np.float64)
    prior_age = 999

    # Last tip accepted by the OUTER safety gate.
    # This is intentionally independent of previous_tip because
    # the V4 internal temporal prior may expire after LOST frames.
    external_gate_anchor_tip = None
    external_gate_anchor_branch = None

    external_relock_candidates = []
    external_relock_branches = []

    reset_hole_tracker()

    hole_initialized = False

    # This is only a proposed external velocity source.
    # Start disabled.
    pid_enabled = False

    last_print = 0.0

    win = (
        "TASK6 EXTERNAL V4 "
        "ARUCO EDGE IBVS"
    )

    cv2.namedWindow(
        win,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        win,
        hole_tracker.mouse_callback,
    )

    write_disabled(
        command_file,
        "startup",
    )

    print()
    print(
        "=============================================="
    )
    print(
        "TASK6 EXTERNAL V4 ARUCO + EDGE IBVS"
    )
    print(
        "=============================================="
    )
    print(
        "Tip: 20mm ArUco ID20 -> coarse ROI -> V4 FINAL"
    )
    print()
    print(
        "Mapping:"
    )
    print(
        " image +U -> robot +X"
    )
    print(
        " image +V -> robot -Z"
    )
    print()
    print(
        "Controls:"
    )
    print(
        " Left click : select target hole"
    )
    print(
        " P          : proposed external velocity ON/OFF"
    )
    print(
        " N          : reselect target hole"
    )
    print(
        " R          : reset V4 temporal prior"
    )
    print(
        " Q / ESC    : quit"
    )
    print()
    print(
        "IMPORTANT: keep dual fusion SHADOW_MODE=True"
    )
    print()

    try:

        while True:

            ok, frame = cap.read()

            if not ok:

                write_disabled(
                    command_file,
                    "camera_frame_lost",
                )

                continue

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            img = frame.copy()

            # =================================================
            # V4 FINAL TIP
            # =================================================

            (
                v4_result,
                corners,
            ) = get_v4_tip(
                frame,
                gray,
                K,
                dist,
                p_tip_marker,
                previous_tip,
                previous_branch,
                prior_age,
            )

            tip_pixel = None
            coarse_pixel = None
            tip_mode = "INVALID"
            branch_id = None

            tip_gate_status = "INVALID"
            tip_jump_px = None

            if v4_result is not None:

                candidate_tip_uv = np.asarray(
                    v4_result[
                        "tip_uv"
                    ],
                    dtype=np.float64,
                )

                candidate_coarse_uv = np.asarray(
                    v4_result.get(
                        "coarse_uv",
                        candidate_tip_uv,
                    ),
                    dtype=np.float64,
                )

                candidate_mode = str(
                    v4_result.get(
                        "mode",
                        "V4",
                    )
                )

                candidate_branch_id = int(
                    v4_result.get(
                        "branch_id",
                        -1,
                    )
                )

                candidate_accept = False
                candidate_relock = False
                accepted_tip_uv = None

                # =============================================
                # INITIAL ACQUISITION
                #
                # Even initial acquisition requires 3 stable
                # candidates. Never trust one isolated point.
                # =============================================

                if external_gate_anchor_tip is None:

                    external_relock_candidates.append(
                        candidate_tip_uv.copy()
                    )

                    external_relock_branches.append(
                        candidate_branch_id
                    )

                    if (
                        len(external_relock_candidates)
                        >
                        EXT_TIP_RELOCK_FRAMES
                    ):
                        del external_relock_candidates[
                            :-
                            EXT_TIP_RELOCK_FRAMES
                        ]

                        del external_relock_branches[
                            :-
                            EXT_TIP_RELOCK_FRAMES
                        ]

                    tip_gate_status = (
                        f"INIT_WAIT "
                        f"{len(external_relock_candidates)}/"
                        f"{EXT_TIP_RELOCK_FRAMES}"
                    )

                    if (
                        len(external_relock_candidates)
                        ==
                        EXT_TIP_RELOCK_FRAMES
                    ):

                        relock_array = np.stack(
                            external_relock_candidates,
                            axis=0,
                        )

                        relock_center = np.mean(
                            relock_array,
                            axis=0,
                        )

                        relock_distances = np.linalg.norm(
                            relock_array
                            -
                            relock_center,
                            axis=1,
                        )

                        relock_spread = float(
                            np.max(
                                relock_distances
                            )
                        )

                        same_branch = (
                            len(
                                set(
                                    external_relock_branches
                                )
                            )
                            ==
                            1
                        )

                        if (
                            relock_spread
                            <=
                            EXT_TIP_RELOCK_CLUSTER_RADIUS_PX
                            and
                            same_branch
                        ):

                            candidate_accept = True
                            candidate_relock = True

                            accepted_tip_uv = (
                                relock_center.copy()
                            )

                            tip_gate_status = (
                                f"INIT_LOCK "
                                f"spread="
                                f"{relock_spread:.1f}"
                            )

                else:

                    tip_jump_px = float(
                        np.linalg.norm(
                            candidate_tip_uv
                            -
                            external_gate_anchor_tip
                        )
                    )

                    # =========================================
                    # NORMAL TRACKING
                    # =========================================

                    if (
                        tip_jump_px
                        <=
                        EXT_TIP_NORMAL_GATE_PX
                    ):

                        candidate_accept = True

                        accepted_tip_uv = (
                            candidate_tip_uv.copy()
                        )

                        external_relock_candidates.clear()
                        external_relock_branches.clear()

                        tip_gate_status = (
                            f"TRACK jump="
                            f"{tip_jump_px:.1f}"
                        )

                    # =========================================
                    # MEDIUM JUMP:
                    # require 3 stable consecutive candidates.
                    # =========================================

                    elif (
                        tip_jump_px
                        <=
                        EXT_TIP_RELOCK_MAX_JUMP_PX
                    ):

                        external_relock_candidates.append(
                            candidate_tip_uv.copy()
                        )

                        external_relock_branches.append(
                            candidate_branch_id
                        )

                        if (
                            len(external_relock_candidates)
                            >
                            EXT_TIP_RELOCK_FRAMES
                        ):
                            del external_relock_candidates[
                                :-
                                EXT_TIP_RELOCK_FRAMES
                            ]

                            del external_relock_branches[
                                :-
                                EXT_TIP_RELOCK_FRAMES
                            ]

                        tip_gate_status = (
                            f"RELOCK_WAIT "
                            f"{len(external_relock_candidates)}/"
                            f"{EXT_TIP_RELOCK_FRAMES} "
                            f"jump={tip_jump_px:.1f}"
                        )

                        if (
                            len(external_relock_candidates)
                            ==
                            EXT_TIP_RELOCK_FRAMES
                        ):

                            relock_array = np.stack(
                                external_relock_candidates,
                                axis=0,
                            )

                            relock_center = np.mean(
                                relock_array,
                                axis=0,
                            )

                            relock_distances = np.linalg.norm(
                                relock_array
                                -
                                relock_center,
                                axis=1,
                            )

                            relock_spread = float(
                                np.max(
                                    relock_distances
                                )
                            )

                            same_branch = (
                                len(
                                    set(
                                        external_relock_branches
                                    )
                                )
                                ==
                                1
                            )

                            if (
                                relock_spread
                                <=
                                EXT_TIP_RELOCK_CLUSTER_RADIUS_PX
                                and
                                same_branch
                            ):

                                candidate_accept = True
                                candidate_relock = True

                                accepted_tip_uv = (
                                    relock_center.copy()
                                )

                                tip_gate_status = (
                                    f"RELOCK "
                                    f"spread="
                                    f"{relock_spread:.1f}"
                                )

                    # =========================================
                    # CATASTROPHIC JUMP
                    #
                    # Never allow a one-frame hundreds-of-pixel
                    # error to poison the V4 temporal prior.
                    # =========================================

                    else:

                        external_relock_candidates.clear()
                        external_relock_branches.clear()

                        tip_gate_status = (
                            f"CATASTROPHIC_REJECT "
                            f"jump={tip_jump_px:.1f}"
                        )

                # =============================================
                # ACCEPTED candidate
                # =============================================

                if candidate_accept:

                    now_tip_time = time.time()

                    if external_last_valid_tip is not None and external_last_valid_time is not None:
                        dt = max(now_tip_time - external_last_valid_time, 1e-3)

                        external_tip_velocity = (
                            accepted_tip_uv
                            -
                            external_last_valid_tip
                        ) / dt

                    external_last_valid_tip = accepted_tip_uv.copy()
                    external_last_valid_time = now_tip_time

                    tip_uv = (
                        accepted_tip_uv.copy()
                    )

                    tip_pixel = (
                        int(
                            round(
                                float(
                                    tip_uv[0]
                                )
                            )
                        ),
                        int(
                            round(
                                float(
                                    tip_uv[1]
                                )
                            )
                        ),
                    )

                    coarse_pixel = (
                        int(
                            round(
                                float(
                                    candidate_coarse_uv[0]
                                )
                            )
                        ),
                        int(
                            round(
                                float(
                                    candidate_coarse_uv[1]
                                )
                            )
                        ),
                    )

                    tip_mode = candidate_mode
                    branch_id = candidate_branch_id

                    previous_tip = (
                        tip_uv.copy()
                    )

                    previous_branch = (
                        branch_id
                    )

                    external_gate_anchor_tip = (
                        tip_uv.copy()
                    )

                    external_gate_anchor_branch = (
                        branch_id
                    )

                    prior_age = 0

                    external_relock_candidates.clear()
                    external_relock_branches.clear()

                    if candidate_relock:

                        print(
                            "[EXT TIP RELOCK] "
                            f"T="
                            f"({tip_pixel[0]},"
                            f"{tip_pixel[1]}) "
                            f"branch="
                            f"{branch_id} "
                            f"status="
                            f"{tip_gate_status}",
                            flush=True,
                        )

                # =============================================
                # REJECTED candidate
                #
                # It does NOT enter tip_pixel and therefore
                # cannot produce robot velocity.
                # =============================================

                else:

                    prior_age += 1

                    if (
                        prior_age
                        >
                        v4.MAX_PRIOR_AGE
                    ):

                        # Let V4 stop using the stale internal
                        # temporal prior, but KEEP the outer
                        # trusted safety anchor.
                        previous_tip = None
                        previous_branch = None

            else:

                external_relock_candidates.clear()
                external_relock_branches.clear()

                tip_gate_status = (
                    "NO_CANDIDATE"
                )

                # =============================================
                # EXTERNAL TIP HOLD
                #
                # If edge/V4 temporarily loses the tip,
                # keep a short predicted position instead
                # of immediately disabling IBVS.
                # =============================================

                if (
                    external_last_valid_tip is not None
                    and
                    external_last_valid_time is not None
                ):

                    lost_time = (
                        time.time()
                        -
                        external_last_valid_time
                    )

                    if lost_time < EXT_TIP_HOLD_S:

                        dt = lost_time

                        hold_tip = (
                            external_last_valid_tip
                            +
                            external_tip_velocity * dt
                        )

                        tip_uv = hold_tip.copy()

                        tip_pixel = (
                            int(round(float(tip_uv[0]))),
                            int(round(float(tip_uv[1])))
                        )

                        tip_mode = "HOLD"

                        tip_gate_status = (
                            f"HOLD {lost_time:.3f}s"
                        )


                prior_age += 1

                if (
                    prior_age
                    >
                    v4.MAX_PRIOR_AGE
                ):

                    previous_tip = None
                    previous_branch = None

            # =================================================
            # Target-hole initialization / tracking
            # =================================================

            # =================================================
            # EXTERNAL FIXED TARGET
            #
            # The external camera and target fixture are fixed.
            # Therefore the target hole does NOT need per-frame
            # Hough tracking.
            #
            # One mouse click -> latch one fixed target pixel.
            # Only the tweezer Tip is tracked dynamically.
            # =================================================

            if (
                hole_tracker.select_done
                and
                not hole_initialized
            ):

                if hole_tracker.hole_click is not None:

                    detected_hole = detect_external_hole_ellipse(
                        gray,
                        hole_tracker.hole_click
                    )

                    if detected_hole is not None:

                        hole_tracker.hole_center = (
                            int(detected_hole[0]),
                            int(detected_hole[1]),
                        )

                        hole_tracker.hole_radius = int(
                            detected_hole[2]
                        )

                        print(
                            "[EXT TARGET LOCK ELLIPSE] "
                            f"target={hole_tracker.hole_center} "
                            f"radius={hole_tracker.hole_radius}",
                            flush=True,
                        )

                    else:

                        hole_tracker.hole_center = (
                            int(
                                hole_tracker.hole_click[0]
                            ),
                            int(
                                hole_tracker.hole_click[1]
                            ),
                        )

                        # Fallback display radius only.
                        hole_tracker.hole_radius = 8

                        print(
                            "[EXT TARGET LOCK FALLBACK] "
                            f"target={hole_tracker.hole_center}",
                            flush=True,
                        )

                    hole_initialized = True

            if (
                hole_initialized
                and
                hole_tracker.hole_center
                is not None
            ):

                H = (
                    int(
                        hole_tracker.hole_center[0]
                    ),
                    int(
                        hole_tracker.hole_center[1]
                    ),
                )

            else:

                H = None

            # =================================================
            # Draw marker
            # =================================================

            if corners is not None:

                poly = (
                    np.asarray(
                        corners,
                        dtype=np.int32,
                    )
                    .reshape(
                        -1,
                        1,
                        2,
                    )
                )

                cv2.polylines(
                    img,
                    [poly],
                    True,
                    (0, 255, 255),
                    2,
                )

            # coarse V4 prediction
            if coarse_pixel is not None:

                cv2.circle(
                    img,
                    coarse_pixel,
                    5,
                    (0, 165, 255),
                    -1,
                )

            # FINAL V4 tip
            if tip_pixel is not None:

                draw_cross(
                    img,
                    tip_pixel,
                    (255, 255, 0),
                    28,
                    2,
                )

                cv2.putText(
                    img,
                    "V4 FINAL TIP",
                    (
                        tip_pixel[0] + 15,
                        tip_pixel[1] - 15,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )

            # target
            if H is not None:

                cv2.drawMarker(
                    img,
                    (
                        int(H[0]),
                        int(H[1]),
                    ),
                    (0, 255, 0),
                    cv2.MARKER_CROSS,
                    24,
                    2,
                )

                if (
                    hole_tracker.hole_radius
                    is not None
                ):

                    cv2.circle(
                        img,
                        (
                            int(H[0]),
                            int(H[1]),
                        ),
                        int(
                            hole_tracker.hole_radius
                        ),
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    img,
                    "TARGET",
                    (
                        int(H[0]) + 15,
                        int(H[1]) - 15,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

            # =================================================
            # External proposed velocity
            # =================================================

            now = time.time()

            du = None
            dv = None

            proposed_vx = 0.0
            proposed_vz = 0.0

            command_valid = (
                tip_pixel is not None
                and
                H is not None
            )

            if command_valid:

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

                # image +U -> robot +X
                proposed_vx = limit(
                    KP_X
                    *
                    du_cmd
                )

                # image +V -> robot -Z
                proposed_vz = limit(
                    -KP_Z
                    *
                    dv_cmd
                )

                cv2.line(
                    img,
                    tip_pixel,
                    (
                        int(H[0]),
                        int(H[1]),
                    ),
                    (255, 255, 0),
                    1,
                )

            output_enabled = bool(
                pid_enabled
                and
                command_valid
            )

            if output_enabled:

                output_vx = proposed_vx
                output_vz = proposed_vz

            else:

                output_vx = 0.0
                output_vz = 0.0

            # =================================================
            # JSON
            # =================================================

            state = {

                "timestamp":
                    now,

                "enabled":
                    output_enabled,

                "source":
                    "external_aruco_v4_edge",

                "dictionary":
                    "DICT_5X5_50",

                "marker_id":
                    20,

                "tip_valid":
                    tip_pixel is not None,

                "hole_valid":
                    H is not None,

                "tip_mode":
                    tip_mode,

                "tip_gate_status":
                    tip_gate_status,

                "tip_jump_px":
                    (
                        None
                        if tip_jump_px is None
                        else float(tip_jump_px)
                    ),

                "branch_id":
                    branch_id,

                "hole_pixel":
                    (
                        list(H)
                        if H is not None
                        else None
                    ),

                "tweezer_tip_pixel":
                    (
                        list(tip_pixel)
                        if tip_pixel is not None
                        else None
                    ),

                "tip_state":
                    str(tip_gate_status),

                "hole_state":
                    (
                        "TRACKING"
                        if H is not None
                        else "LOST"
                    ),

                "tip_confidence":
                    (
                        1.0
                        if tip_pixel is not None
                        else 0.0
                    ),

                "hole_confidence":
                    (
                        1.0
                        if H is not None
                        else 0.0
                    ),

                "coarse_tip_pixel":
                    (
                        list(coarse_pixel)
                        if coarse_pixel is not None
                        else None
                    ),

                "du_px":
                    du,

                "dv_px":
                    dv,

                # Values read by fusion:
                "vx_mm_s":
                    float(
                        output_vx
                    ),

                "vy_mm_s":
                    0.0,

                "vz_mm_s":
                    float(
                        output_vz
                    ),

                # Diagnostic values before P gate:
                "proposed_vx_mm_s":
                    float(
                        proposed_vx
                    ),

                "proposed_vz_mm_s":
                    float(
                        proposed_vz
                    ),
            }

            atomic_write_json(
                command_file,
                state,
            )

            # =================================================
            # Screen
            # =================================================

            cv2.putText(
                img,
                (
                    f"V4 {tip_mode} "
                    f"B{branch_id}"
                    if branch_id is not None
                    else
                    "V4 INVALID"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    (0, 255, 0)
                    if tip_pixel is not None
                    else
                    (0, 0, 255)
                ),
                2,
            )

            cv2.putText(
                img,
                (
                    "EXT PROPOSE ON"
                    if pid_enabled
                    else
                    "EXT PROPOSE OFF"
                ),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    (0, 255, 0)
                    if pid_enabled
                    else
                    (0, 0, 255)
                ),
                2,
            )

            if command_valid:

                cv2.putText(
                    img,
                    (
                        f"du={du:+.0f}px "
                        f"dv={dv:+.0f}px"
                    ),
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    img,
                    (
                        f"proposed "
                        f"vx={proposed_vx:+.3f} "
                        f"vz={proposed_vz:+.3f} mm/s"
                    ),
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )

            else:

                cv2.putText(
                    img,
                    (
                        f"TIP="
                        f"{'OK' if tip_pixel is not None else 'LOST'} "
                        f"TARGET="
                        f"{'OK' if H is not None else 'LOST'}"
                    ),
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 0, 255),
                    2,
                )

            # =================================================
            # Log
            # =================================================

            if (
                now
                -
                last_print
                >=
                0.2
            ):

                if command_valid:

                    print(
                        "[EXT V4] "
                        f"mode={tip_mode} "
                        f"B={branch_id} "
                        f"H={H} "
                        f"T={tip_pixel} "
                        f"du={du:+.1f} "
                        f"dv={dv:+.1f} "
                        f"proposed_vx={proposed_vx:+.3f} "
                        f"proposed_vz={proposed_vz:+.3f} "
                        f"enabled={output_enabled}",
                        flush=True,
                    )

                else:

                    print(
                        "[EXT V4] "
                        f"tip="
                        f"{'OK' if tip_pixel is not None else 'LOST'} "
                        f"hole="
                        f"{'OK' if H is not None else 'LOST'} "
                        f"mode={tip_mode} "
                        f"enabled={output_enabled}",
                        flush=True,
                    )

                last_print = now

            cv2.imshow(
                win,
                img,
            )

            key = cv2.waitKey(1) & 0xFF

            # -------------------------------------------------
            # P: enable proposed external JSON velocity
            # -------------------------------------------------

            if key in (
                ord("p"),
                ord("P"),
            ):

                pid_enabled = (
                    not pid_enabled
                )

                print(
                    "[EXT V4] proposed velocity "
                    +
                    (
                        "ENABLED"
                        if pid_enabled
                        else
                        "DISABLED"
                    ),
                    flush=True,
                )

            # -------------------------------------------------
            # N: select target again
            # -------------------------------------------------

            elif key in (
                ord("n"),
                ord("N"),
            ):

                pid_enabled = False

                hole_initialized = False

                reset_hole_tracker()

                print(
                    "[EXT V4] target cleared. "
                    "Click new target hole.",
                    flush=True,
                )

            # -------------------------------------------------
            # R: reset V4 temporal prior
            # -------------------------------------------------

            elif key in (
                ord("r"),
                ord("R"),
            ):

                previous_tip = None
                previous_branch = None
                prior_age = 999

                external_gate_anchor_tip = None
                external_gate_anchor_branch = None
                external_relock_candidates.clear()
                external_relock_branches.clear()

                print(
                    "[EXT V4] temporal prior reset",
                    flush=True,
                )

            elif key in (
                ord("q"),
                ord("Q"),
                27,
            ):

                break

    finally:

        write_disabled(
            command_file,
            "stopped",
        )

        cap.release()

        cv2.destroyAllWindows()

        print(
            "[EXT V4] stopped",
            flush=True,
        )


if __name__ == "__main__":
    main()
