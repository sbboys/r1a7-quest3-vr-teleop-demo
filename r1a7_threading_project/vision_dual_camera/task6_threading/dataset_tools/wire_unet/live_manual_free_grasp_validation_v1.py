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

WINDOW = "R1-A7 MANUAL FREE + GRASP TRACKING V1"

OUT_ROOT = Path(
    "/home/robot/r1a7_datasets/wire_live_validation"
)

PANEL_W = base.DISPLAY_W
PANEL_H = base.DISPLAY_H

CAMERA_ORDER = (
    "external",
    "right_wrist",
    "left_wrist",
)

VALID_TRACK_STATES = {
    "INIT",
    "TRACK",
    "RECOVER",
}

# Endpoint identity association.
ASSIGNMENT_MARGIN_MIN_PX = 20.0

# Minimum usable path.
MIN_PATH_LENGTH_PX = 90.0

# FREE visual region:
# first 20 px of path from manually calibrated FREE side.
FREE_REGION_LENGTH_PX = 20.0

# FIXED visual reference.
FIXED_REGION_LENGTH_PX = 20.0

# Grasp region is defined by ARC LENGTH FROM FREE.
#
# It is NOT a fixed image coordinate.
GRASP_START_PX = 20.0
GRASP_END_PX = 60.0

# Representative grasp point.
GRASP_POINT_PX = 40.0

# Tangent span around grasp point.
TANGENT_SPAN_PX = 15.0


# ============================================================
# PATH HELPERS
# ============================================================

def euclidean(a, b):

    a = np.asarray(
        a,
        dtype=np.float32,
    )

    b = np.asarray(
        b,
        dtype=np.float32,
    )

    return float(
        np.linalg.norm(
            a - b
        )
    )


def ordered_path_from_mask(
    path_mask,
):

    path_mask = np.asarray(
        path_mask,
        dtype=bool,
    )

    if not np.any(path_mask):
        return None

    try:

        (
            path,
            _,
            _,
            _,
        ) = longest_geodesic_path(
            path_mask
        )

    except Exception:
        return None

    if (
        path is None
        or
        len(path) < 2
    ):
        return None

    return [
        (
            int(p[0]),
            int(p[1]),
        )
        for p in path
    ]


def cumulative_arc(
    path_yx,
):

    p = np.asarray(
        path_yx,
        dtype=np.float32,
    )

    if len(p) == 0:

        return np.zeros(
            0,
            dtype=np.float32,
        )

    if len(p) == 1:

        return np.zeros(
            1,
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


def nearest_arc_index(
    cumulative,
    target,
):

    if len(cumulative) == 0:
        return None

    return int(
        np.argmin(
            np.abs(
                cumulative
                - float(target)
            )
        )
    )


def path_region(
    shape,
    path_yx,
    cumulative,
    start_s,
    end_s,
):

    out = np.zeros(
        shape,
        dtype=bool,
    )

    idx = np.where(
        (cumulative >= start_s)
        &
        (cumulative <= end_s)
    )[0]

    for i in idx:

        y, x = (
            path_yx[int(i)]
        )

        if (
            0 <= y < shape[0]
            and
            0 <= x < shape[1]
        ):

            out[
                y,
                x,
            ] = True

    return out


def tangent_at_arc(
    path_yx,
    cumulative,
    target_s,
):

    center_idx = nearest_arc_index(
        cumulative,
        target_s,
    )

    if center_idx is None:
        return None

    s0 = max(
        0.0,
        target_s
        - TANGENT_SPAN_PX,
    )

    s1 = min(
        float(
            cumulative[-1]
        ),
        target_s
        + TANGENT_SPAN_PX,
    )

    i0 = nearest_arc_index(
        cumulative,
        s0,
    )

    i1 = nearest_arc_index(
        cumulative,
        s1,
    )

    if (
        i0 is None
        or
        i1 is None
        or
        i0 == i1
    ):
        return None

    y0, x0 = (
        path_yx[i0]
    )

    y1, x1 = (
        path_yx[i1]
    )

    v = np.asarray(
        [
            float(
                x1 - x0
            ),
            float(
                y1 - y0
            ),
        ],
        dtype=np.float32,
    )

    n = float(
        np.linalg.norm(v)
    )

    if n < 1e-6:
        return None

    v /= n

    return (
        float(v[0]),
        float(v[1]),
    )


def empty_geometry(
    shape,
    state,
    status,
    calibrated,
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

        "calibrated":
            calibrated,

        "measurement_valid":
            False,

        "display_valid":
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

        "path_length_px":
            0.0,

        "assignment_cost":
            None,

        "assignment_alt_cost":
            None,

        "assignment_margin":
            None,

        "identity_ambiguous":
            False,
    }


# ============================================================
# MANUAL FREE IDENTITY TRACKER
# ============================================================

class ManualFreeTracker:

    def __init__(
        self,
        camera_name,
    ):

        self.camera_name = (
            camera_name
        )

        self.calibrated = False

        self.prev_free_yx = None
        self.prev_fixed_yx = None

        self.last_geometry = None

        self.manual_click_yx = None


    def clear(
        self,
    ):

        self.calibrated = False

        self.prev_free_yx = None
        self.prev_fixed_yx = None

        self.last_geometry = None

        self.manual_click_yx = None

        print(
            f"[{self.camera_name}] "
            f"manual FREE calibration cleared"
        )


    def orient_from_manual_click(
        self,
        path_yx,
        click_yx,
    ):

        e0 = path_yx[0]
        e1 = path_yx[-1]

        d0 = euclidean(
            click_yx,
            e0,
        )

        d1 = euclidean(
            click_yx,
            e1,
        )

        if d0 <= d1:

            oriented = list(
                path_yx
            )

            snap_distance = d0

        else:

            oriented = list(
                reversed(
                    path_yx
                )
            )

            snap_distance = d1

        return (
            oriented,
            snap_distance,
        )


    def calibrate(
        self,
        path_mask,
        click_yx,
    ):

        path_yx = (
            ordered_path_from_mask(
                path_mask
            )
        )

        if path_yx is None:

            print(
                f"[{self.camera_name}] "
                f"CALIBRATION FAILED: no tracked path"
            )

            return False

        cumulative = (
            cumulative_arc(
                path_yx
            )
        )

        if (
            len(cumulative) == 0
            or
            cumulative[-1]
            < MIN_PATH_LENGTH_PX
        ):

            print(
                f"[{self.camera_name}] "
                f"CALIBRATION FAILED: path too short"
            )

            return False

        (
            oriented,
            snap_distance,
        ) = self.orient_from_manual_click(
            path_yx,
            click_yx,
        )

        self.calibrated = True

        self.manual_click_yx = (
            int(click_yx[0]),
            int(click_yx[1]),
        )

        self.prev_free_yx = (
            oriented[0]
        )

        self.prev_fixed_yx = (
            oriented[-1]
        )

        self.last_geometry = (
            self.build_geometry(
                oriented,
                state="MANUAL",
                status="MANUAL_CALIBRATED",
                measurement_valid=True,
                assignment_cost=0.0,
                assignment_alt_cost=None,
                assignment_margin=None,
                identity_ambiguous=False,
            )
        )

        print()
        print("=" * 70)

        print(
            f"[{self.camera_name}] "
            f"MANUAL FREE CALIBRATED"
        )

        print(
            "click yx       =",
            self.manual_click_yx,
        )

        print(
            "FREE snapped   =",
            self.prev_free_yx,
        )

        print(
            "FIXED endpoint =",
            self.prev_fixed_yx,
        )

        print(
            "click->FREE    =",
            f"{snap_distance:.2f}px",
        )

        print("=" * 70)
        print()

        return True


    def orient_by_identity(
        self,
        path_yx,
    ):

        e0 = path_yx[0]
        e1 = path_yx[-1]

        # Assignment A:
        # e0 = FREE, e1 = FIXED
        cost_a = (
            euclidean(
                e0,
                self.prev_free_yx,
            )
            +
            euclidean(
                e1,
                self.prev_fixed_yx,
            )
        )

        # Assignment B:
        # e1 = FREE, e0 = FIXED
        cost_b = (
            euclidean(
                e1,
                self.prev_free_yx,
            )
            +
            euclidean(
                e0,
                self.prev_fixed_yx,
            )
        )

        if cost_a <= cost_b:

            oriented = list(
                path_yx
            )

            best = cost_a
            alt = cost_b

        else:

            oriented = list(
                reversed(
                    path_yx
                )
            )

            best = cost_b
            alt = cost_a

        margin = float(
            alt - best
        )

        ambiguous = (
            margin
            <
            ASSIGNMENT_MARGIN_MIN_PX
        )

        return (
            oriented,
            float(best),
            float(alt),
            margin,
            ambiguous,
        )


    def build_geometry(
        self,
        oriented,
        state,
        status,
        measurement_valid,
        assignment_cost,
        assignment_alt_cost,
        assignment_margin,
        identity_ambiguous,
    ):

        shape = (
            self.current_shape
        )

        cumulative = (
            cumulative_arc(
                oriented
            )
        )

        total_length = float(
            cumulative[-1]
        )

        if (
            total_length
            < MIN_PATH_LENGTH_PX
        ):

            return empty_geometry(
                shape,
                state,
                "PATH_TOO_SHORT",
                self.calibrated,
            )

        free_yx = (
            oriented[0]
        )

        fixed_yx = (
            oriented[-1]
        )

        free_end = min(
            FREE_REGION_LENGTH_PX,
            total_length,
        )

        free_region = path_region(
            shape,
            oriented,
            cumulative,
            0.0,
            free_end,
        )

        fixed_start = max(
            0.0,
            total_length
            - FIXED_REGION_LENGTH_PX,
        )

        fixed_region = path_region(
            shape,
            oriented,
            cumulative,
            fixed_start,
            total_length,
        )

        grasp_start = min(
            GRASP_START_PX,
            total_length,
        )

        grasp_end = min(
            GRASP_END_PX,
            total_length,
        )

        grasp_region = path_region(
            shape,
            oriented,
            cumulative,
            grasp_start,
            grasp_end,
        )

        grasp_target = min(
            GRASP_POINT_PX,
            total_length,
        )

        grasp_idx = nearest_arc_index(
            cumulative,
            grasp_target,
        )

        grasp_yx = None

        if grasp_idx is not None:

            grasp_yx = (
                oriented[
                    grasp_idx
                ]
            )

        tangent = tangent_at_arc(
            oriented,
            cumulative,
            grasp_target,
        )

        return {
            "state":
                state,

            "status":
                status,

            "calibrated":
                self.calibrated,

            "measurement_valid":
                bool(
                    measurement_valid
                ),

            "display_valid":
                True,

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
                grasp_yx,

            "grasp_tangent_xy":
                tangent,

            "path_length_px":
                total_length,

            "assignment_cost":
                assignment_cost,

            "assignment_alt_cost":
                assignment_alt_cost,

            "assignment_margin":
                assignment_margin,

            "identity_ambiguous":
                bool(
                    identity_ambiguous
                ),
        }


    def frozen_geometry(
        self,
        state,
        status,
    ):

        if self.last_geometry is None:

            return empty_geometry(
                self.current_shape,
                state,
                status,
                self.calibrated,
            )

        result = {}

        for k, v in (
            self.last_geometry.items()
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

        result[
            "state"
        ] = state

        result[
            "status"
        ] = status

        result[
            "measurement_valid"
        ] = False

        result[
            "display_valid"
        ] = True

        return result


    def update(
        self,
        track,
    ):

        path_mask = np.asarray(
            track["path"],
            dtype=bool,
        )

        self.current_shape = (
            path_mask.shape
        )

        state = (
            track["state"]
        )

        if not self.calibrated:

            return empty_geometry(
                self.current_shape,
                state,
                "CLICK_FREE_TO_CALIBRATE",
                False,
            )

        # Short visual interruption:
        # keep FREE identity and last geometry.
        if state == "HOLD":

            return self.frozen_geometry(
                state,
                "HOLD_FROZEN",
            )

        # RESET / LOST:
        # DO NOT throw away manual semantic identity.
        # Just pause updates until current path becomes reliable again.
        if state in (
            "RESET",
            "LOST",
        ):

            return self.frozen_geometry(
                state,
                "IDENTITY_PAUSED",
            )

        if state not in (
            VALID_TRACK_STATES
        ):

            return self.frozen_geometry(
                state,
                "STATE_PAUSED",
            )

        path_yx = (
            ordered_path_from_mask(
                path_mask
            )
        )

        if path_yx is None:

            return self.frozen_geometry(
                state,
                "NO_CURRENT_PATH",
            )

        cumulative = (
            cumulative_arc(
                path_yx
            )
        )

        if (
            len(cumulative) == 0
            or
            cumulative[-1]
            < MIN_PATH_LENGTH_PX
        ):

            return self.frozen_geometry(
                state,
                "CURRENT_PATH_TOO_SHORT",
            )

        (
            oriented,
            best_cost,
            alt_cost,
            margin,
            ambiguous,
        ) = self.orient_by_identity(
            path_yx
        )

        if ambiguous:

            # Important:
            # do not update semantic identity on an ambiguous frame.
            #
            # Keep last geometry rather than allowing FREE/FIXED swap.
            result = self.frozen_geometry(
                state,
                "IDENTITY_AMBIGUOUS_HOLD",
            )

            result[
                "assignment_cost"
            ] = best_cost

            result[
                "assignment_alt_cost"
            ] = alt_cost

            result[
                "assignment_margin"
            ] = margin

            result[
                "identity_ambiguous"
            ] = True

            return result

        result = self.build_geometry(
            oriented,
            state=state,
            status="TRACKED",
            measurement_valid=True,
            assignment_cost=best_cost,
            assignment_alt_cost=alt_cost,
            assignment_margin=margin,
            identity_ambiguous=False,
        )

        # Only a reliable frame updates semantic identity.
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

        self.last_geometry = {}

        for k, v in (
            result.items()
        ):

            if isinstance(
                v,
                np.ndarray,
            ):

                self.last_geometry[k] = (
                    v.copy()
                )

            else:

                self.last_geometry[k] = v

        return result


# ============================================================
# DISPLAY
# ============================================================

def dilate(
    mask,
    radius,
):

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if (
        radius <= 0
        or
        not np.any(mask)
    ):

        return mask

    k = (
        radius
        * 2
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


def draw_scene(
    frame,
    det,
    track,
    geom,
    name,
    fps,
):

    overlay = (
        frame.copy()
    )

    tool = np.asarray(
        det["tool_mask"],
        dtype=bool,
    )

    path = np.asarray(
        track["path"],
        dtype=bool,
    )

    # Tool = BLUE
    overlay[
        dilate(
            tool,
            1,
        )
    ] = (
        255,
        0,
        0,
    )

    # Temporal tracked wire = RED
    overlay[
        dilate(
            path,
            2,
        )
    ] = (
        0,
        0,
        255,
    )

    if geom[
        "display_valid"
    ]:

        # FREE region = GREEN
        overlay[
            dilate(
                geom[
                    "free_region"
                ],
                4,
            )
        ] = (
            0,
            255,
            0,
        )

        # FIXED region = CYAN
        overlay[
            dilate(
                geom[
                    "fixed_region"
                ],
                3,
            )
        ] = (
            255,
            255,
            0,
        )

        # GRASP region = YELLOW
        overlay[
            dilate(
                geom[
                    "grasp_region"
                ],
                5,
            )
        ] = (
            0,
            255,
            255,
        )

    vis = cv2.addWeighted(
        frame,
        0.62,
        overlay,
        0.38,
        0,
    )

    if geom[
        "display_valid"
    ]:

        free_yx = geom[
            "free_endpoint_yx"
        ]

        fixed_yx = geom[
            "fixed_endpoint_yx"
        ]

        grasp_yx = geom[
            "grasp_point_yx"
        ]

        tangent = geom[
            "grasp_tangent_xy"
        ]

        if free_yx is not None:

            fy, fx = (
                free_yx
            )

            cv2.circle(
                vis,
                (
                    int(fx),
                    int(fy),
                ),
                9,
                (
                    0,
                    255,
                    0,
                ),
                3,
            )

            cv2.putText(
                vis,
                "FREE",
                (
                    int(fx) + 10,
                    int(fy) - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (
                    0,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

        if fixed_yx is not None:

            fy, fx = (
                fixed_yx
            )

            cv2.circle(
                vis,
                (
                    int(fx),
                    int(fy),
                ),
                7,
                (
                    255,
                    255,
                    0,
                ),
                2,
            )

        if grasp_yx is not None:

            gy, gx = (
                grasp_yx
            )

            cv2.circle(
                vis,
                (
                    int(gx),
                    int(gy),
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
                    int(gx),
                    int(gy),
                ),
                10,
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
                    int(gx) + 10,
                    int(gy) - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (
                    0,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            if tangent is not None:

                tx, ty = tangent

                scale = 35

                cv2.arrowedLine(
                    vis,
                    (
                        int(gx),
                        int(gy),
                    ),
                    (
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
                    ),
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.25,
                )

    # Header
    cv2.rectangle(
        vis,
        (
            0,
            0,
        ),
        (
            vis.shape[1],
            103,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    calibration_text = (
        "CAL=YES"
        if geom[
            "calibrated"
        ]
        else
        "CAL=NO  CLICK REAL FREE END"
    )

    line1 = (
        f"{name}  "
        f"{calibration_text}"
    )

    line2 = (
        f"STATE={track['state']}  "
        f"STATUS={geom['status']}"
    )

    margin = geom[
        "assignment_margin"
    ]

    line3 = (
        f"L={geom['path_length_px']:.0f}px  "
        f"FPS={fps:.1f}  "
        f"ID_MARGIN="
        f"{margin:.1f}"
        if margin is not None
        else
        f"L={geom['path_length_px']:.0f}px  "
        f"FPS={fps:.1f}"
    )

    color = (
        (
            0,
            255,
            0,
        )
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

    cv2.putText(
        vis,
        line1,
        (
            12,
            26,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
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
        line2,
        (
            12,
            57,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        line3,
        (
            12,
            87,
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
# RECORDING
# ============================================================

class Recorder:

    def __init__(
        self,
    ):

        self.active = False

        self.session = None

        self.fp = None
        self.writer = None

        self.frame_index = 0
        self.start_time = None

        self.rows = []


    def start(
        self,
    ):

        if self.active:
            return

        stamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        self.session = (
            OUT_ROOT
            /
            f"manual_free_grasp_v1_{stamp}"
        )

        self.session.mkdir(
            parents=True,
            exist_ok=False,
        )

        self.fp = open(
            self.session
            /
            "metrics.csv",
            "w",
            newline="",
        )

        fields = [
            "frame",
            "elapsed_s",
            "camera",
            "state",
            "status",

            "calibrated",
            "measurement_valid",

            "free_x",
            "free_y",

            "fixed_x",
            "fixed_y",

            "grasp_x",
            "grasp_y",

            "path_length_px",

            "assignment_cost",
            "assignment_alt_cost",
            "assignment_margin",

            "identity_ambiguous",

            "fps",
        ]

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
        print("MANUAL FREE/GRASP RECORDING STARTED")
        print("SESSION =", self.session)
        print("=" * 72)
        print()


    def add(
        self,
        camera,
        geom,
        fps,
    ):

        if not self.active:
            return

        free = geom[
            "free_endpoint_yx"
        ]

        fixed = geom[
            "fixed_endpoint_yx"
        ]

        grasp = geom[
            "grasp_point_yx"
        ]

        row = {
            "frame":
                self.frame_index,

            "elapsed_s":
                time.perf_counter()
                - self.start_time,

            "camera":
                camera,

            "state":
                geom["state"],

            "status":
                geom["status"],

            "calibrated":
                int(
                    geom[
                        "calibrated"
                    ]
                ),

            "measurement_valid":
                int(
                    geom[
                        "measurement_valid"
                    ]
                ),

            "free_x":
                (
                    free[1]
                    if free
                    is not None
                    else ""
                ),

            "free_y":
                (
                    free[0]
                    if free
                    is not None
                    else ""
                ),

            "fixed_x":
                (
                    fixed[1]
                    if fixed
                    is not None
                    else ""
                ),

            "fixed_y":
                (
                    fixed[0]
                    if fixed
                    is not None
                    else ""
                ),

            "grasp_x":
                (
                    grasp[1]
                    if grasp
                    is not None
                    else ""
                ),

            "grasp_y":
                (
                    grasp[0]
                    if grasp
                    is not None
                    else ""
                ),

            "path_length_px":
                geom[
                    "path_length_px"
                ],

            "assignment_cost":
                (
                    geom[
                        "assignment_cost"
                    ]
                    if geom[
                        "assignment_cost"
                    ]
                    is not None
                    else ""
                ),

            "assignment_alt_cost":
                (
                    geom[
                        "assignment_alt_cost"
                    ]
                    if geom[
                        "assignment_alt_cost"
                    ]
                    is not None
                    else ""
                ),

            "assignment_margin":
                (
                    geom[
                        "assignment_margin"
                    ]
                    if geom[
                        "assignment_margin"
                    ]
                    is not None
                    else ""
                ),

            "identity_ambiguous":
                int(
                    geom[
                        "identity_ambiguous"
                    ]
                ),

            "fps":
                float(fps),
        }

        self.writer.writerow(
            row
        )

        self.rows.append(
            row
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
                "grasp_start_px":
                    GRASP_START_PX,

                "grasp_point_px":
                    GRASP_POINT_PX,

                "grasp_end_px":
                    GRASP_END_PX,

                "assignment_margin_min_px":
                    ASSIGNMENT_MARGIN_MIN_PX,
            },

            "cameras": {},
        }

        for camera in CAMERA_ORDER:

            rows = [
                r
                for r in self.rows
                if r["camera"]
                == camera
            ]

            if not rows:
                continue

            n = len(rows)

            valid = [
                r[
                    "measurement_valid"
                ]
                for r in rows
            ]

            ambiguous = [
                r[
                    "identity_ambiguous"
                ]
                for r in rows
            ]

            statuses = [
                r[
                    "status"
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
                            valid
                        )
                    ),

                "identity_ambiguous_rate":
                    float(
                        np.mean(
                            ambiguous
                        )
                    ),

                "hold_or_pause_rate":
                    float(
                        np.mean(
                            [
                                (
                                    "HOLD"
                                    in x
                                    or
                                    "PAUSED"
                                    in x
                                )
                                for x
                                in statuses
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
        print("MANUAL FREE / GRASP V1 SUMMARY")
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
                "  frames                  =",
                s["frames"],
            )

            print(
                "  measurement valid rate  =",
                f"{s['measurement_valid_rate']:.4f}",
            )

            print(
                "  identity ambiguous rate =",
                f"{s['identity_ambiguous_rate']:.4f}",
            )

            print(
                "  hold/pause rate         =",
                f"{s['hold_or_pause_rate']:.4f}",
            )

        print()
        print(
            "SESSION =",
            self.session,
        )

        print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)

    print(
        "R1-A7 MANUAL FREE + GRASP TRACKING V1"
    )

    print("=" * 78)

    print()
    print(
        "MANUAL CALIBRATION:"
    )

    print(
        "  LEFT CLICK the TRUE FREE endpoint"
    )

    print(
        "  Click independently in each camera panel"
    )

    print(
        "  The click defines IDENTITY only."
    )

    print(
        "  FREE does NOT remain at a fixed image pixel."
    )

    print()
    print(
        "DISPLAY:"
    )

    print(
        "  BLUE   = tool"
    )

    print(
        "  RED    = temporal tracked wire"
    )

    print(
        "  GREEN  = manually initialized FREE side"
    )

    print(
        "  CYAN   = FIXED side"
    )

    print(
        "  YELLOW = grasp region, 20~60 px from FREE"
    )

    print(
        "  WHITE  = grasp representative point (~40 px)"
    )

    print()
    print(
        "C = clear all manual calibration"
    )

    print(
        "R = start / stop recording"
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
        name:
            temporal.TemporalTracker(
                name
            )
        for name in CAMERA_ORDER
    }

    manual_trackers = {
        name:
            ManualFreeTracker(
                name
            )
        for name in CAMERA_ORDER
    }

    recorder = Recorder()

    latest_tracks = {
        name:
            None
        for name in CAMERA_ORDER
    }

    latest_shapes = {
        name:
            None
        for name in CAMERA_ORDER
    }

    # --------------------------------------------------------
    # Mouse callback
    # --------------------------------------------------------

    def on_mouse(
        event,
        x,
        y,
        flags,
        param,
    ):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if (
            y < 0
            or
            y >= PANEL_H
        ):
            return

        panel_index = int(
            x // PANEL_W
        )

        if (
            panel_index < 0
            or
            panel_index
            >= len(
                CAMERA_ORDER
            )
        ):
            return

        camera = (
            CAMERA_ORDER[
                panel_index
            ]
        )

        local_x = (
            x
            -
            panel_index
            * PANEL_W
        )

        local_y = y

        track = (
            latest_tracks[
                camera
            ]
        )

        shape = (
            latest_shapes[
                camera
            ]
        )

        if (
            track is None
            or
            shape is None
        ):

            print(
                f"[{camera}] "
                f"no live path available yet"
            )

            return

        h, w = shape[:2]

        original_x = int(
            np.clip(
                local_x
                * w
                / PANEL_W,
                0,
                w - 1,
            )
        )

        original_y = int(
            np.clip(
                local_y
                * h
                / PANEL_H,
                0,
                h - 1,
            )
        )

        print(
            f"[{camera}] click display=({local_x},{local_y}) "
            f"original=({original_x},{original_y})"
        )

        manual_trackers[
            camera
        ].current_shape = (
            track[
                "path"
            ].shape
        )

        manual_trackers[
            camera
        ].calibrate(
            track["path"],
            (
                original_y,
                original_x,
            ),
        )

    # --------------------------------------------------------
    # Cameras
    # --------------------------------------------------------

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

        # AUTOSIZE is intentional so mouse coordinates
        # correspond directly to the displayed panel image.
        cv2.namedWindow(
            WINDOW,
            cv2.WINDOW_AUTOSIZE,
        )

        cv2.setMouseCallback(
            WINDOW,
            on_mouse,
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

            frames = {
                "external":
                    frame_ext,

                "right_wrist":
                    frame_right,

                "left_wrist":
                    frame_left,
            }

            detections = {}
            fps_values = {}

            for camera in CAMERA_ORDER:

                (
                    detections[
                        camera
                    ],
                    fps_values[
                        camera
                    ],
                ) = base.run_one(
                    model,
                    device,
                    frames[
                        camera
                    ],
                )

            temporal_tracks = {}

            for camera in CAMERA_ORDER:

                temporal_tracks[
                    camera
                ] = trackers[
                    camera
                ].update(
                    detections[
                        camera
                    ]
                )

                latest_tracks[
                    camera
                ] = temporal_tracks[
                    camera
                ]

                latest_shapes[
                    camera
                ] = frames[
                    camera
                ].shape

            geometries = {}

            for camera in CAMERA_ORDER:

                geometries[
                    camera
                ] = manual_trackers[
                    camera
                ].update(
                    temporal_tracks[
                        camera
                    ]
                )

            visuals = {}

            for camera in CAMERA_ORDER:

                visuals[
                    camera
                ] = draw_scene(
                    frames[
                        camera
                    ],
                    detections[
                        camera
                    ],
                    temporal_tracks[
                        camera
                    ],
                    geometries[
                        camera
                    ],
                    camera.upper(),
                    fps_values[
                        camera
                    ],
                )

            if recorder.active:

                for camera in CAMERA_ORDER:

                    recorder.add(
                        camera,
                        geometries[
                            camera
                        ],
                        fps_values[
                            camera
                        ],
                    )

                recorder.next_frame()

            panel = np.hstack(
                [
                    base.resize_panel(
                        visuals[
                            camera
                        ]
                    )
                    for camera
                    in CAMERA_ORDER
                ]
            )

            if recorder.active:

                cv2.putText(
                    panel,
                    "REC",
                    (
                        panel.shape[1]
                        - 65,
                        26,
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
                ord("c"),
                ord("C"),
            ):

                for tracker in (
                    manual_trackers.values()
                ):
                    tracker.clear()

            elif key in (
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
            "[INFO] manual FREE/GRASP V1 stopped"
        )


if __name__ == "__main__":
    main()
