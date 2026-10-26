#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from stable_wire_detector import WireLocalDetectorV65
from marker_stability_monitor import monitor_frame, draw_marker_overlay


HERE = Path(__file__).resolve().parent

TASK6_DIR = None

for p in [HERE] + list(HERE.parents):
    candidate = p / "task6_external_ibvs_v4_aruco_edge.py"
    if candidate.exists():
        TASK6_DIR = p
        break

if TASK6_DIR is None:
    raise RuntimeError(
        "Cannot locate task6_external_ibvs_v4_aruco_edge.py"
    )

sys.path.insert(
    0,
    str(TASK6_DIR)
)

import task6_external_ibvs_v4_aruco_edge as ext_v4


DEVICE = "/dev/video16"

OUTPUT = Path(
    "/tmp/r1a7_external_wire_grasp_v1.json"
)


# External camera mapping already verified by previous Task6:
#
# image +U -> robot +X
# image +V -> robot -Z

KP_X = 0.08
KP_Z = 0.08

MAX_GUIDANCE_MM_S = 10.0

DEADBAND_PX = 4.0


def limit(v):

    return float(
        np.clip(
            float(v),
            -MAX_GUIDANCE_MM_S,
            MAX_GUIDANCE_MM_S
        )
    )


def atomic_json(
    path,
    data
):

    tmp = Path(
        str(path) + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            data,
            indent=2,
            default=lambda x:
                x.item()
                if hasattr(x, "item")
                else
                x.tolist()
                if hasattr(x, "tolist")
                else
                str(x)
        )
    )

    os.replace(
        tmp,
        path
    )


def clip_roi(
    roi,
    shape
):

    H, W = shape[:2]

    x, y, w, h = [
        int(round(v))
        for v in roi
    ]

    x = max(
        0,
        min(
            x,
            W - 1
        )
    )

    y = max(
        0,
        min(
            y,
            H - 1
        )
    )

    x2 = max(
        x + 1,
        min(
            x + w,
            W
        )
    )

    y2 = max(
        y + 1,
        min(
            y + h,
            H
        )
    )

    return (
        x,
        y,
        x2 - x,
        y2 - y
    )


def roi_from_tip(
    tip,
    shape,
    half_w=180,
    half_h=180,
):

    u, v = tip

    return clip_roi(
        (
            u - half_w,
            v - half_h,
            2 * half_w,
            2 * half_h
        ),
        shape
    )


def roi_from_path(
    path,
    shape,
    margin=80,
    min_w=300,
    min_h=300,
):

    pts = np.asarray(
        path,
        dtype=np.float64
    )

    xmin = float(
        np.min(
            pts[:, 0]
        )
    )

    xmax = float(
        np.max(
            pts[:, 0]
        )
    )

    ymin = float(
        np.min(
            pts[:, 1]
        )
    )

    ymax = float(
        np.max(
            pts[:, 1]
        )
    )

    cx = 0.5 * (
        xmin + xmax
    )

    cy = 0.5 * (
        ymin + ymax
    )

    w = max(
        min_w,
        xmax - xmin + 2 * margin
    )

    h = max(
        min_h,
        ymax - ymin + 2 * margin
    )

    return clip_roi(
        (
            cx - w / 2,
            cy - h / 2,
            w,
            h
        ),
        shape
    )


def expand_roi(
    roi,
    shape,
    extra
):

    x, y, w, h = roi

    cx = x + w / 2
    cy = y + h / 2

    return clip_roi(
        (
            cx - w / 2 - extra,
            cy - h / 2 - extra,
            w + 2 * extra,
            h + 2 * extra
        ),
        shape
    )



class TipKalmanFilter:
    """Constant-velocity image-space Kalman filter; diagnostic tracking only."""

    def __init__(self):
        self.x = None
        self.P = None
        self.last_time = None
        self.last_measurement_time = None
        self.max_hold_s = 0.25
        self.R = np.diag([16.0, 16.0])
        self.accel_sigma = 600.0

    def reset(self):
        self.x = None
        self.P = None
        self.last_time = None
        self.last_measurement_time = None

    def _predict_to(self, now):
        dt = float(np.clip(now - self.last_time, 1.0 / 120.0, 0.20))

        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        G = np.array([
            [0.5 * dt * dt, 0],
            [0, 0.5 * dt * dt],
            [dt, 0],
            [0, dt],
        ], dtype=np.float64)

        Q = (self.accel_sigma ** 2) * (G @ G.T)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_time = now

    def update(self, point, now=None):
        if now is None:
            now = time.monotonic()

        z = np.asarray(point, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(z)):
            return None, "INVALID_MEASUREMENT"

        # Never blend a new observation into a stale track.
        if (
            self.x is None
            or self.last_measurement_time is None
            or now - self.last_measurement_time > self.max_hold_s
        ):
            self.reset()
            self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
            self.P = np.diag([16.0, 16.0, 400.0, 400.0])
            self.last_time = now
            self.last_measurement_time = now
            return z.copy(), "KALMAN_INIT"

        self._predict_to(now)

        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        innovation = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        I = np.eye(4, dtype=np.float64)
        self.x = self.x + K @ innovation
        self.P = (
            (I - K @ H) @ self.P @ (I - K @ H).T
            + K @ self.R @ K.T
        )

        self.last_measurement_time = now
        return self.x[:2].copy(), (
            f"KALMAN_TRACK innovation={np.linalg.norm(innovation):.1f}px"
        )

    def predict_only(self, now=None):
        if now is None:
            now = time.monotonic()

        if self.x is None or self.last_measurement_time is None:
            return None, "NO_TRACK"

        lost_s = now - self.last_measurement_time
        if lost_s > self.max_hold_s:
            return None, f"LOST_TIMEOUT {lost_s:.2f}s"

        self._predict_to(now)
        return self.x[:2].copy(), f"PREDICT {lost_s:.2f}s"


class ExternalTipGate:

    def __init__(self):

        self.anchor = None
        self.branch = None

        self.relock = []
        self.relock_branch = []

        self.last_valid = None
        self.last_time = None

        self.velocity = np.zeros(
            2,
            dtype=np.float64
        )

        self.kalman = TipKalmanFilter()
        self.prior_age = 999
        self.init_miss_count = 0


    def update(self, v4_result):
        """Directly accept each valid V4 tip candidate without temporal gating."""
        now = time.time()

        # No candidate: report loss instead of fabricating a measured tip.
        if not isinstance(v4_result, dict) or v4_result.get("tip_uv") is None:
            self.prior_age += 1
            self.relock.clear()
            self.relock_branch.clear()

            return {
                "valid": False,
                "measurement_valid": False,
                "predicted_valid": False,
                "tip": None,
                "mode": "LOST",
                "status": "NO_CANDIDATE",
            }

        # Reject malformed coordinates only; do not reject spatial jumps.
        try:
            cand = np.asarray(
                v4_result["tip_uv"],
                dtype=np.float64,
            ).reshape(-1)

            if cand.size != 2 or not np.all(np.isfinite(cand)):
                raise ValueError("tip_uv must contain two finite values")

            cand = cand.reshape(2)
            branch = int(v4_result.get("branch_id", -1))

        except (TypeError, ValueError, KeyError):
            self.prior_age += 1
            return {
                "valid": False,
                "measurement_valid": False,
                "predicted_valid": False,
                "tip": None,
                "mode": "INVALID_CANDIDATE",
                "status": "INVALID_TIP_UV",
            }

        # Keep velocity information for diagnostics, but never use it to
        # reject a candidate or suppress a coordinate jump.
        previous = self.last_valid

        if previous is not None and self.last_time is not None:
            dt = max(now - self.last_time, 1e-3)
            self.velocity = (cand - previous) / dt
        else:
            self.velocity = np.zeros(2, dtype=np.float64)

        # Immediately accept the current V4 measurement.
        self.anchor = cand.copy()
        self.branch = branch
        self.last_valid = cand.copy()
        self.last_time = now
        self.prior_age = 0

        self.relock.clear()
        self.relock_branch.clear()

        return {
            "valid": True,
            "measurement_valid": True,
            "predicted_valid": False,
            "tip": cand.copy(),
            "mode": "DIRECT",
            "status": f"DIRECT_ACCEPT branch={branch}",
        }


def main():

    K, dist = (
        ext_v4.v2.load_intrinsics()
    )

    candidate_tip_calib = (
        Path(__file__).resolve().parents[6]
        / "calibration"
        / "tweezer_marker_tip_calibration_xminus85_zminus25_branchreproj_candidate.json"
    )

    if not candidate_tip_calib.is_file():
        raise RuntimeError(
            "Required candidate tip calibration missing: "
            + str(candidate_tip_calib)
        )

    ext_v4.v2.TIP_CALIB_JSON = candidate_tip_calib
    print("[EXT TIP] candidate calibration =", candidate_tip_calib, flush=True)

    p_tip_marker = (
        ext_v4.v2.load_coarse_tip_marker()
    )


    cap = cv2.VideoCapture(
        DEVICE,
        cv2.CAP_V4L2
    )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open {DEVICE}"
        )


    tip_gate = (
        ExternalTipGate()
    )


    wire = (
        WireLocalDetectorV65(
            local_length_px=200.0,
            grasp_offset_px=60.0,
            grasp_band_start_px=40.0,
            grasp_band_end_px=110.0,
        )
    )


    wire_ready = False
    active_roi = None
    anchor_roi = None
    miss_count = 0

    clicked = {
        "tip": None
    }


    win = (
        "EXTERNAL V4 + WIRE GRASP GUIDANCE"
    )

    cv2.namedWindow(
        win,
        cv2.WINDOW_NORMAL
    )


    def mouse_cb(
        event,
        x,
        y,
        flags,
        param
    ):

        if (
            event
            ==
            cv2.EVENT_LBUTTONDOWN
        ):

            clicked[
                "tip"
            ] = (
                int(x),
                int(y)
            )

            print(
                "[EXT WIRE CLICK]",
                clicked["tip"],
                flush=True
            )


    cv2.setMouseCallback(
        win,
        mouse_cb
    )


    print()
    print("=" * 72)
    print("EXTERNAL CAMERA V4 + WIRE GRASP GUIDANCE")
    print("SHADOW ONLY - NO LOWCMD")
    print("=" * 72)
    print("LEFT CLICK = initialize real wire free tip")
    print("R          = reset wire")
    print("Q/ESC      = quit")
    print()


    last_print = 0.0


    try:

        while True:

            ok, frame = cap.read()

            # Independent tabletop ArUco stability diagnostic.
            # Never used for localization or robot control.
            marker_status = None
            if ok and frame is not None:
                marker_status = monitor_frame(frame)
            if not ok:
                continue


            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            img = frame.copy()


            # =================================================
            # Existing External V4 tweezer detection
            # =================================================

            (
                v4_result,
                corners,
            ) = ext_v4.get_v4_tip(
                frame,
                gray,
                K,
                dist,
                p_tip_marker,
                tip_gate.anchor,
                tip_gate.branch,
                tip_gate.prior_age,
            )


            tip_state = (
                tip_gate.update(
                    v4_result
                )
            )


            tip_pixel = None

            if (
                tip_state["valid"]
                and
                tip_state["tip"]
                is not None
            ):

                tip_pixel = tuple(
                    np.rint(
                        tip_state[
                            "tip"
                        ]
                    )
                    .astype(int)
                )


            # =================================================
            # Wire initialization
            # =================================================

            if (
                clicked["tip"]
                is not None
            ):

                seed = (
                    clicked["tip"]
                )

                clicked["tip"] = None

                wire = (
                    WireLocalDetectorV65(
                        local_length_px=200.0,
                        grasp_offset_px=60.0,
                        grasp_band_start_px=40.0,
                        grasp_band_end_px=110.0,
                    )
                )

                wire.set_seed_tip(
                    seed
                )

                active_roi = (
                    roi_from_tip(
                        seed,
                        frame.shape
                    )
                )

                anchor_roi = (
                    active_roi
                )

                miss_count = 0
                wire_ready = True

                print(
                    "[EXT WIRE INIT]",
                    seed,
                    flush=True
                )


            # =================================================
            # Current-frame external wire detection
            # =================================================

            if (
                wire_ready
                and
                active_roi is not None
            ):

                wire_result = (
                    wire.detect(
                        frame,
                        active_roi,
                        tweezer_exclusion_mask=None,
                        tweezer_tip=tip_pixel,
                    )
                )

            else:

                wire_result = {
                    "valid": False,
                    "tip_valid": False,
                    "grasp_guidance_valid": False,
                    "state": "WAIT_WIRE_CLICK",
                    "centerline": [],
                    "grasp_segment_uv": [],
                    "wire_tip_uv": None,
                    "grasp_target_uv": None,
                    "score": 0.0,
                }


            wire_geometry_valid = bool(
                wire_result.get(
                    "valid",
                    False
                )
                or
                wire_result.get(
                    "grasp_guidance_valid",
                    False
                )
            )


            if (
                wire_ready
                and
                wire_geometry_valid
                and
                len(
                    wire_result.get(
                        "centerline",
                        []
                    )
                )
                >=
                2
            ):

                miss_count = 0

                active_roi = (
                    roi_from_path(
                        wire_result[
                            "centerline"
                        ],
                        frame.shape
                    )
                )

                if wire_result.get(
                    "tip_valid",
                    False
                ):

                    anchor_roi = (
                        active_roi
                    )

            elif (
                wire_ready
                and
                anchor_roi is not None
            ):

                miss_count = min(
                    miss_count + 1,
                    20
                )

                active_roi = (
                    expand_roi(
                        anchor_roi,
                        frame.shape,
                        min(
                            10 * miss_count,
                            60
                        )
                    )
                )


            target = (
                wire_result.get(
                    "grasp_target_uv"
                )
            )


            # =================================================
            # External image visual guidance
            # =================================================

            du = None
            dv = None

            proposed_vx = 0.0
            proposed_vz = 0.0


            guidance_valid = bool(
                tip_pixel is not None
                and
                target is not None
                and
                wire_result.get(
                    "grasp_guidance_valid",
                    False
                )
            )


            if guidance_valid:

                du = float(
                    target[0]
                    -
                    tip_pixel[0]
                )

                dv = float(
                    target[1]
                    -
                    tip_pixel[1]
                )

                du_cmd = (
                    0.0
                    if
                    abs(du)
                    <=
                    DEADBAND_PX
                    else
                    du
                )

                dv_cmd = (
                    0.0
                    if
                    abs(dv)
                    <=
                    DEADBAND_PX
                    else
                    dv
                )

                proposed_vx = (
                    limit(
                        KP_X
                        *
                        du_cmd
                    )
                )

                proposed_vz = (
                    limit(
                        -KP_Z
                        *
                        dv_cmd
                    )
                )


            # =================================================
            # Draw
            # =================================================

            if corners is not None:

                poly = (
                    np.asarray(
                        corners,
                        dtype=np.int32
                    )
                    .reshape(
                        -1,
                        1,
                        2
                    )
                )

                cv2.polylines(
                    img,
                    [poly],
                    True,
                    (0, 255, 255),
                    2
                )


            if tip_pixel is not None:

                cv2.drawMarker(
                    img,
                    tip_pixel,
                    (255, 255, 0),
                    cv2.MARKER_CROSS,
                    28,
                    2
                )

                cv2.putText(
                    img,
                    "EXT V4 TWEEZER",
                    (
                        tip_pixel[0] + 12,
                        tip_pixel[1] - 12
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 0),
                    2
                )


            if (
                active_roi is not None
            ):

                x, y, w, h = (
                    active_roi
                )

                cv2.rectangle(
                    img,
                    (x, y),
                    (
                        x + w,
                        y + h
                    ),
                    (255, 255, 0),
                    2
                )


            centerline = np.asarray(
                wire_result.get(
                    "centerline",
                    []
                ),
                dtype=np.int32
            )

            if len(centerline) >= 2:

                cv2.polylines(
                    img,
                    [
                        centerline.reshape(
                            -1,
                            1,
                            2
                        )
                    ],
                    False,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )


            band = np.asarray(
                wire_result.get(
                    "grasp_segment_uv",
                    []
                ),
                dtype=np.int32
            )

            if len(band) >= 2:

                cv2.polylines(
                    img,
                    [
                        band.reshape(
                            -1,
                            1,
                            2
                        )
                    ],
                    False,
                    (0, 255, 255),
                    6,
                    cv2.LINE_AA
                )


            wire_tip = (
                wire_result.get(
                    "wire_tip_uv"
                )
            )

            if wire_tip is not None:

                cv2.circle(
                    img,
                    tuple(wire_tip),
                    7,
                    (0, 0, 255),
                    -1
                )


            if target is not None:

                cv2.circle(
                    img,
                    tuple(target),
                    8,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    img,
                    "WIRE GRASP",
                    (
                        target[0] + 10,
                        target[1] + 18
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    2
                )


            if guidance_valid:

                cv2.line(
                    img,
                    tip_pixel,
                    tuple(target),
                    (255, 0, 255),
                    2
                )


            cv2.putText(
                img,
                (
                    f"WIRE={wire_result.get('state')} "
                    f"EXTTIP={tip_state['status']}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (
                    (0, 255, 0)
                    if guidance_valid
                    else
                    (0, 165, 255)
                ),
                2
            )


            if guidance_valid:

                cv2.putText(
                    img,
                    (
                        f"du={du:+.1f}px "
                        f"dv={dv:+.1f}px "
                        f"vx={proposed_vx:+.2f} "
                        f"vz={proposed_vz:+.2f}"
                    ),
                    (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )


            # =================================================
            # Shadow JSON
            # =================================================

            out = {
                "timestamp":
                    time.time(),

                "enabled":
                    False,

                "shadow_mode":
                    True,

                "source":
                    "external_v4_plus_wire",

                "guidance_valid":
                    guidance_valid,

                "tweezer_tip_uv":
                    (
                        list(tip_pixel)
                        if tip_pixel is not None
                        else
                        None
                    ),

                "wire_tip_uv":
                    wire_tip,

                "grasp_target_uv":
                    target,

                "wire_state":
                    wire_result.get(
                        "state"
                    ),

                "wire_tip_valid":
                    bool(
                        wire_result.get(
                            "tip_valid",
                            False
                        )
                    ),

                # Wire filter diagnostics: raw vs filtered.
                "raw_wire_tip_uv":
                    wire_result.get("raw_wire_tip_uv"),

                "raw_grasp_target_uv":
                    wire_result.get("raw_grasp_target_uv"),

                "wire_tip_filter_state":
                    wire_result.get("wire_tip_filter_state"),

                "grasp_filter_state":
                    wire_result.get("grasp_filter_state"),

                "wire_tip_filter_jump_px":
                    wire_result.get("wire_tip_filter_jump_px"),

                "grasp_filter_jump_px":
                    wire_result.get("grasp_filter_jump_px"),

                "grasp_guidance_valid":
                    bool(
                        wire_result.get(
                            "grasp_guidance_valid",
                            False
                        )
                    ),

                "du_px":
                    du,

                "dv_px":
                    dv,

                "proposed_vx_mm_s":
                    float(
                        proposed_vx
                    ),

                "proposed_vy_mm_s":
                    0.0,

                "proposed_vz_mm_s":
                    float(
                        proposed_vz
                    ),
            }


            atomic_json(
                OUTPUT,
                out
            )


            now = time.time()

            if (
                now -
                last_print
                >=
                0.5
            ):

                print(
                    "[EXT WIRE DIAG] "
                    f"guidance={guidance_valid} "
                    f"tweezer_tip={tip_pixel} "
                    f"tip_gate_valid={tip_state.get('valid')} "
                    f"tip_gate_mode={tip_state.get('mode')} "
                    f"tip_gate_status={tip_state.get('status')} "
                    f"v4_tip={v4_result.get('tip_uv') if isinstance(v4_result, dict) else None} "
                    f"v4_branch={v4_result.get('branch_id') if isinstance(v4_result, dict) else None} "
                    f"v4_quality={v4_result.get('quality') if isinstance(v4_result, dict) else None} "
                    f"wire_state={wire_result.get('state')} "
                    f"wire_valid={wire_result.get('valid', False)} "
                    f"tip_valid={wire_result.get('tip_valid', False)} "
                    f"wire_tip={wire_result.get('wire_tip_uv')} "
                    f"raw_wire_tip={wire_result.get('raw_wire_tip_uv')} "
                    f"candidate_count={wire_result.get('candidate_count')} "
                    f"rank_score={wire_result.get('rank_score')} "
                    f"wire_score={wire_result.get('score')} "
                    f"grasp_valid={wire_result.get('grasp_guidance_valid', False)} "
                    f"grasp_target={target} "
                    f"tip_filter={wire_result.get('wire_tip_filter_state')} "
                    f"grasp_filter={wire_result.get('grasp_filter_state')} "
                    f"du={du} dv={dv} "
                    f"vx={proposed_vx:+.2f} vz={proposed_vz:+.2f}",
                    flush=True
                )

                last_print = now


            # Draw ID 1 on the existing External display only.
            draw_marker_overlay(img, marker_status)

            cv2.imshow(
                win,
                img
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )


            if key in (
                ord("r"),
                ord("R")
            ):

                wire_ready = False
                active_roi = None
                anchor_roi = None

                print(
                    "[EXT WIRE] reset - click free tip again",
                    flush=True
                )


            elif key in (
                ord("q"),
                ord("Q"),
                27
            ):

                break


    finally:

        cap.release()

        cv2.destroyAllWindows()

        atomic_json(
            OUTPUT,
            {
                "timestamp":
                    time.time(),
                "enabled":
                    False,
                "shadow_mode":
                    True,
                "guidance_valid":
                    False,
                "state":
                    "STOPPED",
            }
        )


if __name__ == "__main__":
    main()
