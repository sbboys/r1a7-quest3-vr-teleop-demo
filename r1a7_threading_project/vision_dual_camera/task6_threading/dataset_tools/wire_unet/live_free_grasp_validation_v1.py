#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import csv
import json
import time

import cv2
import numpy as np

import live_wire_unet_three_camera as base
import live_wire_temporal_validation_v1 as temporal

from eval_wire_connected_path_v3 import (
    longest_geodesic_path,
)


# ============================================================
# CONFIG
# ============================================================

WINDOW = "R1-A7 FREE END + GRASP REGION VALIDATION V1"

OUT_ROOT = Path(
    "/home/robot/r1a7_datasets/wire_live_validation"
)

# Only TRACK / RECOVER are new valid visual measurements.
VALID_TRACK_STATES = {
    "TRACK",
    "RECOVER",
    "INIT",
}

# Need enough visible centerline before endpoint reasoning.
MIN_PATH_LENGTH_PX = 80.0

# FREE endpoint = endpoint closer to detected tool.
#
# But if the two endpoint distances are too similar,
# do NOT force a FREE/FIXED decision.
FREE_MARGIN_MIN_PX = 15.0
FREE_MARGIN_RATIO = 0.10

# FREE endpoint region:
# first 10% of path, limited to 15~40 px.
FREE_REGION_RATIO = 0.10
FREE_REGION_MIN_PX = 15.0
FREE_REGION_MAX_PX = 40.0

# FIXED region only for visual reference.
FIXED_REGION_RATIO = 0.10
FIXED_REGION_MIN_PX = 15.0
FIXED_REGION_MAX_PX = 40.0

# Candidate grasp region measured FROM FREE INTO THE WIRE.
GRASP_START_RATIO = 0.05
GRASP_END_RATIO = 0.20

GRASP_START_MIN_PX = 12.0
GRASP_END_MAX_PX = 60.0

# Tangent / local straightness span.
CURVATURE_SPAN = 8

# Consecutive-frame endpoint identity diagnostic.
SWITCH_DISTANCE_MARGIN_PX = 10.0


# ============================================================
# HELPERS
# ============================================================

def arc_length(path_yx):

    p = np.asarray(
        path_yx,
        dtype=np.float32,
    )

    if len(p) < 2:
        return np.zeros(
            len(p),
            dtype=np.float32,
        )

    d = np.diff(
        p,
        axis=0,
    )

    step = np.sqrt(
        np.sum(
            d * d,
            axis=1,
        )
    )

    return np.concatenate(
        [
            np.zeros(
                1,
                dtype=np.float32,
            ),
            np.cumsum(step),
        ]
    )


def endpoint_distance_map(
    tool_mask,
):

    tool = np.asarray(
        tool_mask,
        dtype=bool,
    )

    if not np.any(tool):
        return None

    # Tool pixels = 0, background = 1.
    src = (
        ~tool
    ).astype(
        np.uint8
    )

    return cv2.distanceTransform(
        src,
        cv2.DIST_L2,
        5,
    )


def mask_from_indices(
    shape,
    path_yx,
    indices,
):

    out = np.zeros(
        shape,
        dtype=bool,
    )

    for i in indices:

        y = int(
            path_yx[i][0]
        )

        x = int(
            path_yx[i][1]
        )

        if (
            0 <= y < shape[0]
            and
            0 <= x < shape[1]
        ):
            out[y, x] = True

    return out


def make_region_by_arc(
    shape,
    path_yx,
    cumulative,
    start_s,
    end_s,
):

    idx = np.where(
        (cumulative >= start_s)
        &
        (cumulative <= end_s)
    )[0]

    return (
        mask_from_indices(
            shape,
            path_yx,
            idx,
        ),
        idx,
    )


def point_distance(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    return float(
        np.linalg.norm(
            a - b
        )
    )


def select_straight_grasp_point(
    path_yx,
    candidate_indices,
):

    if len(candidate_indices) == 0:
        return None

    p = np.asarray(
        path_yx,
        dtype=np.float32,
    )

    best = None
    best_angle = None
    best_tangent = None

    n = len(p)

    for idx in candidate_indices:

        i = int(idx)

        i0 = max(
            0,
            i - CURVATURE_SPAN,
        )

        i1 = min(
            n - 1,
            i + CURVATURE_SPAN,
        )

        if (
            i0 == i
            or
            i1 == i
        ):
            continue

        # yx -> xy
        p0 = p[i0][::-1]
        pc = p[i][::-1]
        p1 = p[i1][::-1]

        v0 = (
            pc - p0
        )

        v1 = (
            p1 - pc
        )

        n0 = float(
            np.linalg.norm(v0)
        )

        n1 = float(
            np.linalg.norm(v1)
        )

        if (
            n0 < 1e-6
            or
            n1 < 1e-6
        ):
            continue

        u0 = v0 / n0
        u1 = v1 / n1

        dot = float(
            np.clip(
                np.dot(
                    u0,
                    u1,
                ),
                -1.0,
                1.0,
            )
        )

        angle = float(
            np.degrees(
                np.arccos(dot)
            )
        )

        tangent = (
            p1 - p0
        )

        tn = float(
            np.linalg.norm(
                tangent
            )
        )

        if tn > 1e-6:
            tangent = (
                tangent / tn
            )
        else:
            tangent = np.array(
                [
                    0.0,
                    0.0,
                ],
                dtype=float,
            )

        if (
            best_angle is None
            or
            angle < best_angle
        ):

            best_angle = angle
            best = i
            best_tangent = tangent

    if best is None:

        best = int(
            candidate_indices[
                len(candidate_indices)
                // 2
            ]
        )

        i0 = max(
            0,
            best - CURVATURE_SPAN,
        )

        i1 = min(
            n - 1,
            best + CURVATURE_SPAN,
        )

        p0 = p[i0][::-1]
        p1 = p[i1][::-1]

        tangent = (
            p1 - p0
        )

        tn = float(
            np.linalg.norm(
                tangent
            )
        )

        if tn > 1e-6:
            tangent = (
                tangent / tn
            )
        else:
            tangent = np.array(
                [
                    0.0,
                    0.0,
                ],
                dtype=float,
            )

        best_tangent = tangent
        best_angle = None

    y = int(
        p[best][0]
    )

    x = int(
        p[best][1]
    )

    return {
        "index":
            best,

        "yx":
            (
                y,
                x,
            ),

        # tangent direction from FREE toward FIXED
        "tangent_xy":
            (
                float(
                    best_tangent[0]
                ),
                float(
                    best_tangent[1]
                ),
            ),

        "straightness_angle_deg":
            best_angle,
    }


def empty_result(
    shape,
    state,
    status,
):

    z = np.zeros(
        shape,
        dtype=bool,
    )

    return {
        "state":
            state,

        "status":
            status,

        "measurement_valid":
            False,

        "display_valid":
            False,

        "free_valid":
            False,

        "grasp_valid":
            False,

        "free_endpoint_yx":
            None,

        "fixed_endpoint_yx":
            None,

        "free_region":
            z.copy(),

        "fixed_region":
            z.copy(),

        "grasp_region":
            z.copy(),

        "grasp_point_yx":
            None,

        "grasp_tangent_xy":
            None,

        "straightness_angle_deg":
            None,

        "path_length_px":
            0.0,

        "free_tool_distance_px":
            None,

        "fixed_tool_distance_px":
            None,

        "endpoint_distance_margin_px":
            None,

        "endpoint_distance_margin_ratio":
            None,

        "switch_candidate":
            False,
    }


# ============================================================
# FREE / GRASP ESTIMATOR
# ============================================================

class FreeGraspEstimator:

    def __init__(
        self,
        camera_name,
    ):

        self.camera_name = (
            camera_name
        )

        self.last_valid = None

        self.prev_free_yx = None
        self.prev_fixed_yx = None


    def estimate_fresh(
        self,
        det,
        track,
    ):

        path_mask = np.asarray(
            track["path"],
            dtype=bool,
        )

        shape = path_mask.shape

        if not np.any(
            path_mask
        ):

            return empty_result(
                shape,
                track["state"],
                "NO_PATH",
            )

        try:

            (
                ordered_path,
                _,
                _,
                path_length,
            ) = longest_geodesic_path(
                path_mask
            )

        except Exception:

            return empty_result(
                shape,
                track["state"],
                "ORDER_PATH_FAIL",
            )

        if (
            ordered_path is None
            or
            len(ordered_path) < 2
        ):

            return empty_result(
                shape,
                track["state"],
                "ORDER_PATH_FAIL",
            )

        path_yx = [
            (
                int(p[0]),
                int(p[1]),
            )
            for p in ordered_path
        ]

        cumulative = arc_length(
            path_yx
        )

        if len(cumulative) == 0:

            return empty_result(
                shape,
                track["state"],
                "NO_ARC",
            )

        total_length = float(
            cumulative[-1]
        )

        if (
            total_length
            < MIN_PATH_LENGTH_PX
        ):

            out = empty_result(
                shape,
                track["state"],
                "PATH_TOO_SHORT",
            )

            out[
                "path_length_px"
            ] = total_length

            return out

        tool_dist = (
            endpoint_distance_map(
                det["tool_mask"]
            )
        )

        if tool_dist is None:

            out = empty_result(
                shape,
                track["state"],
                "NO_TOOL",
            )

            out[
                "path_length_px"
            ] = total_length

            return out

        e0 = path_yx[0]
        e1 = path_yx[-1]

        d0 = float(
            tool_dist[
                e0[0],
                e0[1],
            ]
        )

        d1 = float(
            tool_dist[
                e1[0],
                e1[1],
            ]
        )

        margin = abs(
            d0 - d1
        )

        max_d = max(
            d0,
            d1,
            1.0,
        )

        margin_ratio = (
            margin / max_d
        )

        required_margin = max(
            FREE_MARGIN_MIN_PX,
            FREE_MARGIN_RATIO
            * max_d,
        )

        # Do not force semantic identity
        # if both endpoints are similarly distant.
        if margin < required_margin:

            out = empty_result(
                shape,
                track["state"],
                "FREE_AMBIGUOUS",
            )

            out[
                "path_length_px"
            ] = total_length

            out[
                "free_tool_distance_px"
            ] = min(
                d0,
                d1,
            )

            out[
                "fixed_tool_distance_px"
            ] = max(
                d0,
                d1,
            )

            out[
                "endpoint_distance_margin_px"
            ] = margin

            out[
                "endpoint_distance_margin_ratio"
            ] = margin_ratio

            return out

        # ----------------------------------------------------
        # IMPORTANT PROJECT SEMANTICS:
        #
        # FREE = endpoint closer to tweezer/tool.
        # FIXED = farther endpoint.
        # ----------------------------------------------------

        if d0 <= d1:

            oriented = (
                path_yx
            )

            free_distance = d0
            fixed_distance = d1

        else:

            oriented = list(
                reversed(
                    path_yx
                )
            )

            free_distance = d1
            fixed_distance = d0

        cumulative = arc_length(
            oriented
        )

        total_length = float(
            cumulative[-1]
        )

        free_yx = (
            oriented[0]
        )

        fixed_yx = (
            oriented[-1]
        )

        # ----------------------------------------------------
        # FREE END REGION
        # ----------------------------------------------------

        free_region_len = np.clip(
            total_length
            * FREE_REGION_RATIO,
            FREE_REGION_MIN_PX,
            FREE_REGION_MAX_PX,
        )

        (
            free_region,
            _,
        ) = make_region_by_arc(
            shape,
            oriented,
            cumulative,
            0.0,
            float(
                free_region_len
            ),
        )

        # ----------------------------------------------------
        # FIXED END REGION
        # ----------------------------------------------------

        fixed_region_len = np.clip(
            total_length
            * FIXED_REGION_RATIO,
            FIXED_REGION_MIN_PX,
            FIXED_REGION_MAX_PX,
        )

        fixed_start = max(
            0.0,
            total_length
            - float(
                fixed_region_len
            ),
        )

        (
            fixed_region,
            _,
        ) = make_region_by_arc(
            shape,
            oriented,
            cumulative,
            fixed_start,
            total_length,
        )

        # ----------------------------------------------------
        # GRASP REGION
        #
        # Move from FREE INTO the wire.
        # ----------------------------------------------------

        grasp_start = max(
            GRASP_START_MIN_PX,
            total_length
            * GRASP_START_RATIO,
        )

        grasp_end = min(
            GRASP_END_MAX_PX,
            max(
                grasp_start
                + 10.0,
                total_length
                * GRASP_END_RATIO,
            ),
        )

        grasp_end = min(
            grasp_end,
            total_length
            * 0.40,
        )

        (
            grasp_region,
            grasp_indices,
        ) = make_region_by_arc(
            shape,
            oriented,
            cumulative,
            grasp_start,
            grasp_end,
        )

        grasp_info = (
            select_straight_grasp_point(
                oriented,
                grasp_indices,
            )
        )

        grasp_valid = (
            grasp_info is not None
            and
            np.any(
                grasp_region
            )
        )

        # ----------------------------------------------------
        # Identity switch diagnostic
        # ----------------------------------------------------

        switch_candidate = False

        if (
            self.prev_free_yx
            is not None
            and
            self.prev_fixed_yx
            is not None
        ):

            free_to_prev_free = (
                point_distance(
                    free_yx,
                    self.prev_free_yx,
                )
            )

            fixed_to_prev_free = (
                point_distance(
                    fixed_yx,
                    self.prev_free_yx,
                )
            )

            if (
                fixed_to_prev_free
                + SWITCH_DISTANCE_MARGIN_PX
                <
                free_to_prev_free
            ):

                switch_candidate = True

        result = {
            "state":
                track["state"],

            "status":
                "VALID",

            "measurement_valid":
                True,

            "display_valid":
                True,

            "free_valid":
                True,

            "grasp_valid":
                bool(
                    grasp_valid
                ),

            "free_endpoint_yx":
                free_yx,

            "fixed_endpoint_yx":
                fixed_yx,

            "free_region":
                free_region,

            "fixed_region":
                fixed_region,

            "grasp_region":
                grasp_region,

            "grasp_point_yx":
                (
                    grasp_info["yx"]
                    if grasp_info
                    else None
                ),

            "grasp_tangent_xy":
                (
                    grasp_info[
                        "tangent_xy"
                    ]
                    if grasp_info
                    else None
                ),

            "straightness_angle_deg":
                (
                    grasp_info[
                        "straightness_angle_deg"
                    ]
                    if grasp_info
                    else None
                ),

            "path_length_px":
                total_length,

            "free_tool_distance_px":
                free_distance,

            "fixed_tool_distance_px":
                fixed_distance,

            "endpoint_distance_margin_px":
                margin,

            "endpoint_distance_margin_ratio":
                margin_ratio,

            "switch_candidate":
                switch_candidate,
        }

        return result


    def update(
        self,
        det,
        track,
    ):

        state = (
            track["state"]
        )

        shape = (
            track["path"].shape
        )

        # ----------------------------------------------------
        # HOLD:
        # freeze last accepted geometry.
        # DO NOT create a new visual measurement.
        # ----------------------------------------------------

        if state == "HOLD":

            if self.last_valid is None:

                return empty_result(
                    shape,
                    state,
                    "HOLD_NO_HISTORY",
                )

            result = {}

            for k, v in (
                self.last_valid.items()
            ):

                if isinstance(
                    v,
                    np.ndarray,
                ):
                    result[k] = (
                        v.copy()
                    )
                else:
                    result[k] = v

            result["state"] = (
                state
            )

            result["status"] = (
                "HOLD_FROZEN"
            )

            result[
                "measurement_valid"
            ] = False

            result[
                "display_valid"
            ] = True

            result[
                "switch_candidate"
            ] = False

            return result

        # RESET / LOST:
        # do not trust FREE / GRASP geometry.
        if state in (
            "RESET",
            "LOST",
        ):

            return empty_result(
                shape,
                state,
                "TRACKER_NOT_RELIABLE",
            )

        if state not in (
            VALID_TRACK_STATES
        ):

            return empty_result(
                shape,
                state,
                "STATE_NOT_ALLOWED",
            )

        result = (
            self.estimate_fresh(
                det,
                track,
            )
        )

        if (
            result[
                "measurement_valid"
            ]
            and
            result[
                "free_valid"
            ]
        ):

            self.last_valid = {}

            for k, v in (
                result.items()
            ):

                if isinstance(
                    v,
                    np.ndarray,
                ):
                    self.last_valid[k] = (
                        v.copy()
                    )
                else:
                    self.last_valid[k] = v

            self.prev_free_yx = (
                result[
                    "free_endpoint_yx"
                ]
            )

            self.prev_fixed_yx = (
                result[
                    "fixed_endpoint_yx"
                ]
            )

        return result


# ============================================================
# DISPLAY
# ============================================================

def dilate(
    mask,
    radius,
):

    if (
        radius <= 0
        or
        not np.any(mask)
    ):
        return mask

    k = (
        2 * radius
        + 1
    )

    return cv2.dilate(
        mask.astype(
            np.uint8
        ),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                k,
                k,
            ),
        ),
    ).astype(bool)


def draw_result(
    image,
    det,
    track,
    geom,
    name,
    fps,
):

    vis = image.copy()

    overlay = image.copy()

    tool = np.asarray(
        det["tool_mask"],
        dtype=bool,
    )

    tracked_path = np.asarray(
        track["path"],
        dtype=bool,
    )

    # Tool = blue.
    tool_vis = dilate(
        tool,
        1,
    )

    overlay[
        tool_vis
    ] = (
        255,
        0,
        0,
    )

    # Tracked centerline = red.
    path_vis = dilate(
        tracked_path,
        2,
    )

    overlay[
        path_vis
    ] = (
        0,
        0,
        255,
    )

    if geom[
        "display_valid"
    ]:

        # FREE region = green.
        free_vis = dilate(
            geom[
                "free_region"
            ],
            4,
        )

        overlay[
            free_vis
        ] = (
            0,
            255,
            0,
        )

        # FIXED region = cyan.
        fixed_vis = dilate(
            geom[
                "fixed_region"
            ],
            3,
        )

        overlay[
            fixed_vis
        ] = (
            255,
            255,
            0,
        )

        # GRASP region = yellow.
        grasp_vis = dilate(
            geom[
                "grasp_region"
            ],
            5,
        )

        overlay[
            grasp_vis
        ] = (
            0,
            255,
            255,
        )

    vis = cv2.addWeighted(
        image,
        0.62,
        overlay,
        0.38,
        0,
    )

    # --------------------------------------------------------
    # Endpoints
    # --------------------------------------------------------

    if geom[
        "display_valid"
    ]:

        free_yx = geom[
            "free_endpoint_yx"
        ]

        fixed_yx = geom[
            "fixed_endpoint_yx"
        ]

        if free_yx is not None:

            cv2.circle(
                vis,
                (
                    int(
                        free_yx[1]
                    ),
                    int(
                        free_yx[0]
                    ),
                ),
                8,
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

            cv2.putText(
                vis,
                "FREE",
                (
                    int(
                        free_yx[1]
                    )
                    + 8,
                    int(
                        free_yx[0]
                    )
                    - 8,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    0,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

        if fixed_yx is not None:

            cv2.circle(
                vis,
                (
                    int(
                        fixed_yx[1]
                    ),
                    int(
                        fixed_yx[0]
                    ),
                ),
                7,
                (
                    255,
                    255,
                    0,
                ),
                2,
            )

            cv2.putText(
                vis,
                "FIXED",
                (
                    int(
                        fixed_yx[1]
                    )
                    + 8,
                    int(
                        fixed_yx[0]
                    )
                    - 8,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (
                    255,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

        grasp_yx = geom[
            "grasp_point_yx"
        ]

        tangent = geom[
            "grasp_tangent_xy"
        ]

        if (
            geom["grasp_valid"]
            and
            grasp_yx is not None
        ):

            gx = int(
                grasp_yx[1]
            )

            gy = int(
                grasp_yx[0]
            )

            # Representative grasp point = white.
            cv2.circle(
                vis,
                (
                    gx,
                    gy,
                ),
                7,
                (
                    255,
                    255,
                    255,
                ),
                -1,
            )

            cv2.circle(
                vis,
                (
                    gx,
                    gy,
                ),
                9,
                (
                    0,
                    255,
                    255,
                ),
                2,
            )

            cv2.putText(
                vis,
                "GRASP",
                (
                    gx + 9,
                    gy - 9,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (
                    0,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            # Tangent FROM FREE INTO WIRE.
            if tangent is not None:

                tx = float(
                    tangent[0]
                )

                ty = float(
                    tangent[1]
                )

                scale = 35

                p2 = (
                    int(
                        gx
                        + tx
                        * scale
                    ),
                    int(
                        gy
                        + ty
                        * scale
                    ),
                )

                cv2.arrowedLine(
                    vis,
                    (
                        gx,
                        gy,
                    ),
                    p2,
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.25,
                )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    cv2.rectangle(
        vis,
        (
            0,
            0,
        ),
        (
            vis.shape[1],
            112,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    free_text = (
        "YES"
        if geom["free_valid"]
        else "NO"
    )

    grasp_text = (
        "YES"
        if geom["grasp_valid"]
        else "NO"
    )

    line1 = (
        f"{name}  "
        f"STATE={track['state']}  "
        f"STATUS={geom['status']}"
    )

    line2 = (
        f"FREE={free_text}  "
        f"GRASP={grasp_text}  "
        f"L={geom['path_length_px']:.0f}px"
    )

    if (
        geom[
            "free_tool_distance_px"
        ]
        is not None
    ):

        line3 = (
            f"dFREE="
            f"{geom['free_tool_distance_px']:.1f}  "
            f"dFIXED="
            f"{geom['fixed_tool_distance_px']:.1f}  "
            f"margin="
            f"{geom['endpoint_distance_margin_px']:.1f}"
        )

    else:

        line3 = (
            "endpoint-tool distance unavailable"
        )

    switch_text = (
        "  SWITCH?"
        if geom[
            "switch_candidate"
        ]
        else ""
    )

    line4 = (
        f"FPS={fps:.1f}"
        f"{switch_text}"
    )

    color = (
        (0, 255, 0)
        if geom[
            "measurement_valid"
        ]
        else
        (
            0,
            165,
            255,
        )
    )

    if geom[
        "switch_candidate"
    ]:
        color = (
            0,
            0,
            255,
        )

    cv2.putText(
        vis,
        line1,
        (
            12,
            25,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        line2,
        (
            12,
            52,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        line3,
        (
            12,
            78,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (
            220,
            220,
            220,
        ),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        line4,
        (
            12,
            103,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (
            220,
            220,
            220,
        ),
        1,
        cv2.LINE_AA,
    )

    return vis


# ============================================================
# RECORDER
# ============================================================

class Recorder:

    def __init__(
        self,
    ):

        self.active = False
        self.session = None
        self.fp = None
        self.writer = None
        self.start_time = None
        self.frame_index = 0
        self.rows = []


    def start(
        self,
    ):

        if self.active:
            return

        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.session = (
            OUT_ROOT
            /
            f"free_grasp_v1_{stamp}"
        )

        self.session.mkdir(
            parents=True,
            exist_ok=False,
        )

        (
            self.session
            /
            "events"
        ).mkdir()

        fields = [
            "frame_index",
            "elapsed_s",
            "camera",

            "track_state",
            "status",

            "measurement_valid",
            "free_valid",
            "grasp_valid",

            "path_length_px",

            "free_x",
            "free_y",

            "fixed_x",
            "fixed_y",

            "free_tool_distance_px",
            "fixed_tool_distance_px",

            "endpoint_margin_px",
            "endpoint_margin_ratio",

            "grasp_x",
            "grasp_y",

            "grasp_tangent_x",
            "grasp_tangent_y",

            "straightness_angle_deg",

            "switch_candidate",

            "infer_fps",
        ]

        self.fp = open(
            self.session
            /
            "free_grasp_metrics.csv",
            "w",
            newline="",
        )

        self.writer = csv.DictWriter(
            self.fp,
            fieldnames=fields,
        )

        self.writer.writeheader()

        self.start_time = (
            time.perf_counter()
        )

        self.frame_index = 0
        self.rows = []

        self.active = True

        print()
        print("=" * 72)
        print("FREE / GRASP V1 RECORDING STARTED")
        print("SESSION =", self.session)
        print("=" * 72)
        print()


    def add(
        self,
        camera,
        vis,
        geom,
        fps,
    ):

        if not self.active:
            return

        free_yx = (
            geom[
                "free_endpoint_yx"
            ]
        )

        fixed_yx = (
            geom[
                "fixed_endpoint_yx"
            ]
        )

        grasp_yx = (
            geom[
                "grasp_point_yx"
            ]
        )

        tangent = (
            geom[
                "grasp_tangent_xy"
            ]
        )

        row = {
            "frame_index":
                self.frame_index,

            "elapsed_s":
                time.perf_counter()
                - self.start_time,

            "camera":
                camera,

            "track_state":
                geom["state"],

            "status":
                geom["status"],

            "measurement_valid":
                int(
                    geom[
                        "measurement_valid"
                    ]
                ),

            "free_valid":
                int(
                    geom[
                        "free_valid"
                    ]
                ),

            "grasp_valid":
                int(
                    geom[
                        "grasp_valid"
                    ]
                ),

            "path_length_px":
                geom[
                    "path_length_px"
                ],

            "free_x":
                (
                    free_yx[1]
                    if free_yx
                    is not None
                    else ""
                ),

            "free_y":
                (
                    free_yx[0]
                    if free_yx
                    is not None
                    else ""
                ),

            "fixed_x":
                (
                    fixed_yx[1]
                    if fixed_yx
                    is not None
                    else ""
                ),

            "fixed_y":
                (
                    fixed_yx[0]
                    if fixed_yx
                    is not None
                    else ""
                ),

            "free_tool_distance_px":
                (
                    geom[
                        "free_tool_distance_px"
                    ]
                    if geom[
                        "free_tool_distance_px"
                    ]
                    is not None
                    else ""
                ),

            "fixed_tool_distance_px":
                (
                    geom[
                        "fixed_tool_distance_px"
                    ]
                    if geom[
                        "fixed_tool_distance_px"
                    ]
                    is not None
                    else ""
                ),

            "endpoint_margin_px":
                (
                    geom[
                        "endpoint_distance_margin_px"
                    ]
                    if geom[
                        "endpoint_distance_margin_px"
                    ]
                    is not None
                    else ""
                ),

            "endpoint_margin_ratio":
                (
                    geom[
                        "endpoint_distance_margin_ratio"
                    ]
                    if geom[
                        "endpoint_distance_margin_ratio"
                    ]
                    is not None
                    else ""
                ),

            "grasp_x":
                (
                    grasp_yx[1]
                    if grasp_yx
                    is not None
                    else ""
                ),

            "grasp_y":
                (
                    grasp_yx[0]
                    if grasp_yx
                    is not None
                    else ""
                ),

            "grasp_tangent_x":
                (
                    tangent[0]
                    if tangent
                    is not None
                    else ""
                ),

            "grasp_tangent_y":
                (
                    tangent[1]
                    if tangent
                    is not None
                    else ""
                ),

            "straightness_angle_deg":
                (
                    geom[
                        "straightness_angle_deg"
                    ]
                    if geom[
                        "straightness_angle_deg"
                    ]
                    is not None
                    else ""
                ),

            "switch_candidate":
                int(
                    geom[
                        "switch_candidate"
                    ]
                ),

            "infer_fps":
                float(fps),
        }

        self.writer.writerow(
            row
        )

        self.rows.append(
            row
        )

        if (
            geom[
                "switch_candidate"
            ]
            or
            geom[
                "status"
            ]
            not in (
                "VALID",
                "HOLD_FROZEN",
            )
        ):

            stem = (
                f"{self.frame_index:06d}_"
                f"{camera}_"
                f"{geom['status']}"
            )

            cv2.imwrite(
                str(
                    self.session
                    /
                    "events"
                    /
                    f"{stem}.png"
                ),
                vis,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    1,
                ],
            )


    def next_frame(
        self,
    ):

        if not self.active:
            return

        self.frame_index += 1

        if (
            self.frame_index
            % 10
            == 0
        ):
            self.fp.flush()


    def stop(
        self,
    ):

        if not self.active:
            return

        self.active = False

        self.fp.flush()
        self.fp.close()

        summary = {
            "parameters": {
                "free_margin_min_px":
                    FREE_MARGIN_MIN_PX,

                "free_margin_ratio":
                    FREE_MARGIN_RATIO,

                "free_region_ratio":
                    FREE_REGION_RATIO,

                "grasp_start_ratio":
                    GRASP_START_RATIO,

                "grasp_end_ratio":
                    GRASP_END_RATIO,
            },

            "cameras": {},
        }

        for camera in (
            "external",
            "right_wrist",
            "left_wrist",
        ):

            rows = [
                r
                for r in self.rows
                if r["camera"]
                == camera
            ]

            if not rows:
                continue

            n = len(rows)

            free_valid = np.asarray(
                [
                    r[
                        "free_valid"
                    ]
                    for r in rows
                ],
                dtype=float,
            )

            grasp_valid = np.asarray(
                [
                    r[
                        "grasp_valid"
                    ]
                    for r in rows
                ],
                dtype=float,
            )

            measurement_valid = (
                np.asarray(
                    [
                        r[
                            "measurement_valid"
                        ]
                        for r in rows
                    ],
                    dtype=float,
                )
            )

            switch = np.asarray(
                [
                    r[
                        "switch_candidate"
                    ]
                    for r in rows
                ],
                dtype=float,
            )

            states = [
                r[
                    "track_state"
                ]
                for r in rows
            ]

            summary[
                "cameras"
            ][camera] = {
                "frames":
                    n,

                "measurement_valid_rate":
                    float(
                        np.mean(
                            measurement_valid
                        )
                    ),

                "free_valid_rate":
                    float(
                        np.mean(
                            free_valid
                        )
                    ),

                "grasp_valid_rate":
                    float(
                        np.mean(
                            grasp_valid
                        )
                    ),

                "switch_candidate_count":
                    int(
                        np.sum(
                            switch
                        )
                    ),

                "switch_candidate_rate":
                    float(
                        np.mean(
                            switch
                        )
                    ),

                "hold_rate":
                    float(
                        np.mean(
                            [
                                x
                                == "HOLD"
                                for x
                                in states
                            ]
                        )
                    ),

                "reset_rate":
                    float(
                        np.mean(
                            [
                                x
                                == "RESET"
                                for x
                                in states
                            ]
                        )
                    ),

                "lost_rate":
                    float(
                        np.mean(
                            [
                                x
                                == "LOST"
                                for x
                                in states
                            ]
                        )
                    ),
            }

        summary_file = (
            self.session
            /
            "summary.json"
        )

        summary_file.write_text(
            json.dumps(
                summary,
                indent=2,
            )
        )

        print()
        print("=" * 72)
        print("FREE / GRASP V1 SUMMARY")
        print("=" * 72)

        for camera, s in (
            summary[
                "cameras"
            ].items()
        ):

            print()
            print(
                camera.upper()
            )

            print(
                "  frames                 =",
                s["frames"],
            )

            print(
                "  measurement valid rate =",
                f"{s['measurement_valid_rate']:.4f}",
            )

            print(
                "  FREE valid rate        =",
                f"{s['free_valid_rate']:.4f}",
            )

            print(
                "  GRASP valid rate       =",
                f"{s['grasp_valid_rate']:.4f}",
            )

            print(
                "  FREE switch candidates =",
                s[
                    "switch_candidate_count"
                ],
            )

            print(
                "  FREE switch rate       =",
                f"{s['switch_candidate_rate']:.4f}",
            )

            print(
                "  HOLD rate              =",
                f"{s['hold_rate']:.4f}",
            )

            print(
                "  RESET rate             =",
                f"{s['reset_rate']:.4f}",
            )

            print(
                "  LOST rate              =",
                f"{s['lost_rate']:.4f}",
            )

        print()
        print(
            "SESSION =",
            self.session,
        )

        print(
            "SUMMARY =",
            summary_file,
        )

        print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "R1-A7 FREE END + GRASP REGION VALIDATION V1"
    )
    print("=" * 78)

    print()
    print(
        "PROJECT SEMANTICS:"
    )

    print(
        "  FREE  = endpoint closer to tool/tweezer"
    )

    print(
        "  FIXED = farther endpoint"
    )

    print()
    print(
        "DISPLAY:"
    )

    print(
        "  BLUE   = tool/tweezer"
    )

    print(
        "  RED    = temporal tracked centerline"
    )

    print(
        "  GREEN  = FREE endpoint region"
    )

    print(
        "  CYAN   = FIXED endpoint region"
    )

    print(
        "  YELLOW = candidate GRASP region"
    )

    print(
        "  WHITE  = representative grasp point + tangent"
    )

    print()
    print(
        "R       = start / stop recording"
    )

    print(
        "Q / ESC = quit"
    )

    print()
    print(
        "NO DDS / NO LowCmd / NO ROBOT CONTROL"
    )

    print("=" * 78)

    model, device = (
        base.load_model()
    )

    trackers = {
        "external":
            temporal.TemporalTracker(
                "external"
            ),

        "right_wrist":
            temporal.TemporalTracker(
                "right_wrist"
            ),

        "left_wrist":
            temporal.TemporalTracker(
                "left_wrist"
            ),
    }

    estimators = {
        "external":
            FreeGraspEstimator(
                "external"
            ),

        "right_wrist":
            FreeGraspEstimator(
                "right_wrist"
            ),

        "left_wrist":
            FreeGraspEstimator(
                "left_wrist"
            ),
    }

    recorder = Recorder()

    external = None
    right = None
    left = None

    try:

        external = (
            base.open_external()
        )

        ctx = base.Context()

        right = base.open_orbbec(
            ctx,
            base.RIGHT_SERIAL,
        )

        left = base.open_orbbec(
            ctx,
            base.LEFT_SERIAL,
        )

        time.sleep(
            1.0
        )

        cv2.namedWindow(
            WINDOW,
            cv2.WINDOW_NORMAL,
        )

        while True:

            ok_ext, frame_ext = (
                external.read()
            )

            frame_right = (
                base.read_orbbec(
                    right
                )
            )

            frame_left = (
                base.read_orbbec(
                    left
                )
            )

            if (
                not ok_ext
                or
                frame_right is None
                or
                frame_left is None
            ):

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    break

                continue

            det_ext, fps_ext = (
                base.run_one(
                    model,
                    device,
                    frame_ext,
                )
            )

            det_right, fps_right = (
                base.run_one(
                    model,
                    device,
                    frame_right,
                )
            )

            det_left, fps_left = (
                base.run_one(
                    model,
                    device,
                    frame_left,
                )
            )

            tr_ext = (
                trackers[
                    "external"
                ].update(
                    det_ext
                )
            )

            tr_right = (
                trackers[
                    "right_wrist"
                ].update(
                    det_right
                )
            )

            tr_left = (
                trackers[
                    "left_wrist"
                ].update(
                    det_left
                )
            )

            geo_ext = (
                estimators[
                    "external"
                ].update(
                    det_ext,
                    tr_ext,
                )
            )

            geo_right = (
                estimators[
                    "right_wrist"
                ].update(
                    det_right,
                    tr_right,
                )
            )

            geo_left = (
                estimators[
                    "left_wrist"
                ].update(
                    det_left,
                    tr_left,
                )
            )

            vis_ext = draw_result(
                frame_ext,
                det_ext,
                tr_ext,
                geo_ext,
                "EXTERNAL",
                fps_ext,
            )

            vis_right = draw_result(
                frame_right,
                det_right,
                tr_right,
                geo_right,
                "RIGHT WRIST",
                fps_right,
            )

            vis_left = draw_result(
                frame_left,
                det_left,
                tr_left,
                geo_left,
                "LEFT WRIST",
                fps_left,
            )

            if recorder.active:

                recorder.add(
                    "external",
                    vis_ext,
                    geo_ext,
                    fps_ext,
                )

                recorder.add(
                    "right_wrist",
                    vis_right,
                    geo_right,
                    fps_right,
                )

                recorder.add(
                    "left_wrist",
                    vis_left,
                    geo_left,
                    fps_left,
                )

                recorder.next_frame()

            panel = np.hstack(
                [
                    base.resize_panel(
                        vis_ext
                    ),

                    base.resize_panel(
                        vis_right
                    ),

                    base.resize_panel(
                        vis_left
                    ),
                ]
            )

            if recorder.active:

                cv2.putText(
                    panel,
                    "REC",
                    (
                        panel.shape[1]
                        - 70,
                        28,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (
                        0,
                        0,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(
                WINDOW,
                panel,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("r"),
                ord("R"),
            ):

                if recorder.active:
                    recorder.stop()
                else:
                    recorder.start()

            elif key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

    except KeyboardInterrupt:

        print(
            "\n[INFO] Ctrl+C"
        )

    finally:

        if recorder.active:
            recorder.stop()

        if external is not None:

            try:
                external.release()
            except Exception:
                pass

        if right is not None:

            try:
                right.stop()
            except Exception:
                pass

        if left is not None:

            try:
                left.stop()
            except Exception:
                pass

        cv2.destroyAllWindows()

        print(
            "[INFO] FREE/GRASP V1 stopped"
        )


if __name__ == "__main__":
    main()
