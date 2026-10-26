#!/usr/bin/env python3

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Project paths
# ============================================================

HERE = Path(__file__).resolve().parent

ROOT = None
for p in [HERE] + list(HERE.parents):
    if (p / "vision_dual_camera").exists():
        ROOT = p
        break

if ROOT is None:
    raise RuntimeError("Cannot locate r1a7_threading_project root")

TASK6_DIR = (
    ROOT /
    "vision_dual_camera" /
    "task6_threading"
)

sys.path.insert(0, str(TASK6_DIR))
sys.path.insert(0, str(HERE))


# Reuse the already-tested October wrist-camera / tweezer stack.
import task6_wrist_ibvs_controller_v2_tip as wrist

# Reuse our wire temporal tracker.
from wire_realtime_tracker import RealtimeWireTracker


# ============================================================
# Fixed hardware identity from config/cameras.yaml
# ============================================================

RIGHT_WRIST_SERIAL = "CPCBC53000C5"

OUTPUT_FILE = Path(
    "/tmp/r1a7_right_wrist_wire_grasp_shadow.json"
)

WIRE_INIT_FILE = Path(
    "/tmp/r1a7_right_wrist_wire_init.json"
)


# ============================================================
# First-stage grasp geometry
#
# Important:
# grasp target is deliberately INSIDE the free wire end,
# rather than exactly at wire_tip.
#
# This leaves a free wire tail for the later threading step.
# ============================================================

GRASP_ARC_LENGTH_PX = 30.0

# 2-D debug threshold only.
# This is NOT yet permission to close the tweezers.
ALIGN_READY_DISTANCE_PX = 12.0
ALIGN_READY_FRAMES = 5

TIP_EMA_ALPHA = 0.40
TIP_MAX_JUMP_PX = 18.0
TIP_HOLD_S = 0.30
TIP_DETECT_EVERY_N = 2


# ============================================================
# Helpers
# ============================================================

def as_int_point(p):
    if p is None:
        return None

    return (
        int(round(float(p[0]))),
        int(round(float(p[1]))),
    )


def select_wire_seed_points(image):
    """
    Select two points ON THE WIRE:

      W1 = free wire endpoint
      W2 = a point farther inward along the same wire

    W2 is NOT the tweezer tip and is NOT a fixed grasp point.
    It is only an initialization direction/segment seed.
    """

    points = []

    win = "WIRE INIT: W1 FREE TIP -> W2 INWARD"

    cv2.namedWindow(
        win,
        cv2.WINDOW_NORMAL
    )

    def callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            if len(points) < 2:
                points.append(
                    (
                        int(x),
                        int(y)
                    )
                )

        elif event == cv2.EVENT_RBUTTONDOWN:

            if points:
                points.pop()

    cv2.setMouseCallback(
        win,
        callback
    )

    while True:

        vis = image.copy()

        for i, pt in enumerate(points):

            color = (
                (0, 0, 255)
                if i == 0
                else
                (0, 255, 255)
            )

            cv2.circle(
                vis,
                pt,
                8,
                color,
                -1
            )

            label = (
                "W1 FREE TIP"
                if i == 0
                else
                "W2 INWARD"
            )

            cv2.putText(
                vis,
                label,
                (
                    pt[0] + 12,
                    pt[1] - 12
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA
            )

        cv2.putText(
            vis,
            "LEFT: W1 then W2 | RIGHT: undo",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            vis,
            "ENTER: confirm | ESC: cancel",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            win,
            vis
        )

        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10) and len(points) == 2:

            cv2.destroyWindow(win)

            return points

        if key == 27:

            cv2.destroyWindow(win)

            return None


def make_wire_tracker(points):

    tip, inward = points

    data = {
        "tip": [
            int(tip[0]),
            int(tip[1])
        ],
        "grasp": [
            int(inward[0]),
            int(inward[1])
        ]
    }

    WIRE_INIT_FILE.write_text(
        json.dumps(
            data,
            indent=2
        )
    )

    return RealtimeWireTracker(
        init_json=str(WIRE_INIT_FILE)
    )


def grasp_target_from_centerline(
    centerline,
    arc_length_px=GRASP_ARC_LENGTH_PX
):

    if centerline is None:
        return None, None

    pts = np.asarray(
        centerline,
        dtype=np.float64
    ).reshape(-1, 2)

    if len(pts) < 5:
        return None, None

    segment = np.linalg.norm(
        pts[1:] - pts[:-1],
        axis=1
    )

    cumulative = np.concatenate(
        [
            np.zeros(1),
            np.cumsum(segment)
        ]
    )

    if cumulative[-1] < max(
        10.0,
        arc_length_px
    ):
        return None, None

    idx = int(
        np.searchsorted(
            cumulative,
            arc_length_px
        )
    )

    idx = max(
        1,
        min(
            len(pts) - 2,
            idx
        )
    )

    target = pts[idx]

    i0 = max(
        0,
        idx - 3
    )

    i1 = min(
        len(pts) - 1,
        idx + 3
    )

    tangent = (
        pts[i1] -
        pts[i0]
    )

    n = float(
        np.linalg.norm(
            tangent
        )
    )

    if n < 1e-6:
        return target, None

    tangent /= n

    return target, tangent


# ============================================================
# Live tweezer-tip tracker
# ============================================================

class LiveTweezerTip:

    def __init__(
        self,
        init_frame
    ):

        print(
            "[TIP] Loading SAM2 detector...",
            flush=True
        )

        self.detector = (
            wrist.Sam2PcaTweezerDetector(
                checkpoint_path=
                wrist.SAM2_CHECKPOINT
            )
        )

        self.refiner = (
            wrist.TweezerTipRefinerV2()
        )

        print(
            "[TIP] Detector ready.",
            flush=True
        )

        init = wrist.initialize_live_tip(
            init_frame
        )

        if init is None:
            raise RuntimeError(
                "Tweezer tip initialization cancelled"
            )

        (
            self.tip_box,
            self.positive_points,
            self.tip_hint
        ) = init

        self.tip_ema = None
        self.last_valid_tip = None
        self.last_valid_time = None

        self.frame_counter = 0


    def update(
        self,
        frame
    ):

        self.frame_counter += 1

        now = time.time()

        should_detect = (
            self.tip_ema is None
            or
            (
                self.frame_counter
                %
                TIP_DETECT_EVERY_N
                ==
                0
            )
        )

        candidate = None
        state = "LOST"

        if should_detect:

            try:

                coarse = self.detector.detect(
                    frame,
                    box_xyxy=
                    self.tip_box,
                    tip_hint=
                    self.tip_hint,
                    positive_points=
                    self.positive_points,
                )

                if coarse.get(
                    "valid",
                    False
                ):

                    refined = self.refiner.refine(
                        frame,
                        coarse["mask"],
                        coarse["tip_uv"],
                        coarse["direction"],
                    )

                    if refined.get(
                        "valid",
                        False
                    ):

                        candidate = np.asarray(
                            refined[
                                "refined_tip_uv"
                            ],
                            dtype=np.float64
                        )

                        state = "VISUAL"

            except Exception as exc:

                state = (
                    "ERROR:"
                    +
                    type(exc).__name__
                )

        if candidate is not None:

            accept = True

            if self.tip_ema is not None:

                jump = float(
                    np.linalg.norm(
                        candidate -
                        self.tip_ema
                    )
                )

                if jump > TIP_MAX_JUMP_PX:
                    accept = False
                    state = (
                        f"REJECT_JUMP:{jump:.1f}"
                    )

            if accept:

                if self.tip_ema is None:

                    self.tip_ema = (
                        candidate.copy()
                    )

                else:

                    self.tip_ema = (
                        TIP_EMA_ALPHA
                        *
                        candidate
                        +
                        (
                            1.0
                            -
                            TIP_EMA_ALPHA
                        )
                        *
                        self.tip_ema
                    )

                self.last_valid_tip = (
                    self.tip_ema.copy()
                )

                self.last_valid_time = now

                self.tip_hint = (
                    float(
                        self.tip_ema[0]
                    ),
                    float(
                        self.tip_ema[1]
                    )
                )

                return (
                    self.tip_ema.copy(),
                    state
                )

        if (
            self.last_valid_tip is not None
            and
            self.last_valid_time is not None
        ):

            age = (
                now -
                self.last_valid_time
            )

            if age <= TIP_HOLD_S:

                return (
                    self.last_valid_tip.copy(),
                    f"HOLD:{age:.2f}s"
                )

        return None, state


# ============================================================
# JSON output
# ============================================================

def write_output(
    *,
    wire_result,
    wire_target,
    wire_direction,
    tweezer_tip,
    tip_state,
    stable_frames
):

    wire_valid = bool(
        wire_result.get(
            "valid",
            False
        )
    )

    tweezer_valid = (
        tweezer_tip is not None
    )

    target_valid = (
        wire_target is not None
    )

    du = None
    dv = None
    distance = None

    if (
        target_valid
        and
        tweezer_valid
    ):

        delta = (
            np.asarray(
                wire_target,
                dtype=np.float64
            )
            -
            np.asarray(
                tweezer_tip,
                dtype=np.float64
            )
        )

        du = float(
            delta[0]
        )

        dv = float(
            delta[1]
        )

        distance = float(
            np.linalg.norm(
                delta
            )
        )

    output = {

        # Critical safety field:
        # this program NEVER enables robot motion.
        "enabled": False,

        "mode":
            "RIGHT_WRIST_WIRE_GRASP_SHADOW",

        "camera_role":
            "right_wrist",

        "camera_serial":
            RIGHT_WRIST_SERIAL,

        "timestamp":
            float(
                time.time()
            ),

        "wire_valid":
            wire_valid,

        "wire_tip_uv":
            (
                [
                    int(v)
                    for v in
                    wire_result["tip"]
                ]
                if
                wire_result.get("tip")
                is not None
                else
                None
            ),

        "wire_grasp_target_uv":
            (
                [
                    int(round(float(
                        wire_target[0]
                    ))),
                    int(round(float(
                        wire_target[1]
                    )))
                ]
                if target_valid
                else None
            ),

        "wire_direction_uv":
            (
                [
                    float(
                        wire_direction[0]
                    ),
                    float(
                        wire_direction[1]
                    )
                ]
                if
                wire_direction is not None
                else
                None
            ),

        "wire_confidence":
            float(
                wire_result.get(
                    "confidence",
                    0.0
                )
            ),

        "tweezer_valid":
            tweezer_valid,

        "tweezer_tip_uv":
            (
                list(
                    as_int_point(
                        tweezer_tip
                    )
                )
                if
                tweezer_valid
                else
                None
            ),

        "tweezer_state":
            str(
                tip_state
            ),

        "du_px":
            du,

        "dv_px":
            dv,

        "distance_px":
            distance,

        # ONLY image-plane alignment status.
        # NOT permission to close the tweezers.
        "align_ready_2d":
            bool(
                stable_frames
                >=
                ALIGN_READY_FRAMES
            ),

        "align_stable_frames":
            int(
                stable_frames
            ),

        "gripper_close_command":
            False
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 72)
    print("R1-A7 RIGHT WRIST WIRE GRASP - SHADOW")
    print("=" * 72)
    print(
        "Camera serial :",
        RIGHT_WRIST_SERIAL
    )
    print(
        "Robot motion  : DISABLED"
    )
    print(
        "Gripper close : DISABLED"
    )
    print("=" * 72)
    print()

    # Reuse October project helper:
    # open exact right-wrist Orbbec by serial.
    context, device = (
        wrist.hc.find_orbbec_device(
            RIGHT_WRIST_SERIAL
        )
    )

    pipeline = wrist.Pipeline(
        device
    )

    config = wrist.Config()

    profiles = (
        pipeline
        .get_stream_profile_list(
            wrist.OBSensorType.COLOR_SENSOR
        )
    )

    profile = (
        profiles
        .get_video_stream_profile(
            wrist.hc.COLOR_WIDTH,
            wrist.hc.COLOR_HEIGHT,
            wrist.OBFormat.RGB,
            wrist.hc.COLOR_FPS
        )
    )

    config.enable_stream(
        profile
    )

    pipeline.start(
        config
    )

    try:

        init_frame = None

        print(
            "[CAMERA] Waiting for RIGHT wrist image...",
            flush=True
        )

        while init_frame is None:

            frames = (
                pipeline
                .wait_for_frames(
                    1000
                )
            )

            if frames is None:
                continue

            color = (
                frames
                .get_color_frame()
            )

            if color is None:
                continue

            init_frame = (
                wrist.hc
                .color_frame_to_bgr(
                    color
                )
            )

        print(
            "[CAMERA] RIGHT wrist ready:",
            init_frame.shape,
            flush=True
        )

        # ----------------------------------------------------
        # Initialize existing live tweezer-tip detector.
        # ----------------------------------------------------

        tip_tracker = LiveTweezerTip(
            init_frame
        )

        # ----------------------------------------------------
        # Initialize wire only once:
        # W1 = free end
        # W2 = inward point on same wire.
        # ----------------------------------------------------

        wire_points = (
            select_wire_seed_points(
                init_frame
            )
        )

        if wire_points is None:
            print(
                "[WIRE] Initialization cancelled."
            )
            return

        print(
            "[WIRE INIT] free tip =",
            wire_points[0]
        )

        print(
            "[WIRE INIT] inward   =",
            wire_points[1]
        )

        wire_tracker = (
            make_wire_tracker(
                wire_points
            )
        )

        stable_frames = 0

        last_log = 0.0

        window = (
            "R1-A7 RIGHT WRIST "
            "WIRE GRASP SHADOW"
        )

        cv2.namedWindow(
            window,
            cv2.WINDOW_NORMAL
        )

        print()
        print(
            "[RUNNING] Q/ESC = quit"
        )
        print(
            "[RUNNING] No robot motion is generated."
        )
        print()

        while True:

            frames = (
                pipeline
                .wait_for_frames(
                    100
                )
            )

            if frames is None:
                continue

            color = (
                frames
                .get_color_frame()
            )

            if color is None:
                continue

            frame = (
                wrist.hc
                .color_frame_to_bgr(
                    color
                )
            )

            if frame is None:
                continue

            # ----------------------------------------------
            # Wire update
            # ----------------------------------------------

            wire_result = (
                wire_tracker.update(
                    frame
                )
            )

            wire_target = None
            wire_direction = None

            if wire_result.get(
                "valid",
                False
            ):

                (
                    wire_target,
                    wire_direction
                ) = (
                    grasp_target_from_centerline(
                        wire_result.get(
                            "centerline",
                            []
                        )
                    )
                )

            # ----------------------------------------------
            # Tweezer tip update
            # ----------------------------------------------

            (
                tweezer_tip,
                tip_state
            ) = (
                tip_tracker.update(
                    frame
                )
            )

            # ----------------------------------------------
            # Relative image-plane error
            # ----------------------------------------------

            distance = None

            if (
                wire_target is not None
                and
                tweezer_tip is not None
            ):

                distance = float(
                    np.linalg.norm(
                        np.asarray(
                            wire_target
                        )
                        -
                        np.asarray(
                            tweezer_tip
                        )
                    )
                )

            if (
                distance is not None
                and
                distance
                <=
                ALIGN_READY_DISTANCE_PX
                and
                wire_result.get(
                    "valid",
                    False
                )
            ):

                stable_frames += 1

            else:

                stable_frames = 0

            # ----------------------------------------------
            # Draw overlay
            # ----------------------------------------------

            vis = frame.copy()

            centerline = np.asarray(
                wire_result.get(
                    "centerline",
                    []
                ),
                dtype=np.int32
            )

            if len(centerline) >= 2:

                cv2.polylines(
                    vis,
                    [
                        centerline.reshape(
                            -1,
                            1,
                            2
                        )
                    ],
                    False,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            wire_tip = (
                wire_result.get(
                    "tip"
                )
            )

            if wire_tip is not None:

                cv2.circle(
                    vis,
                    tuple(
                        int(v)
                        for v in wire_tip
                    ),
                    7,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    vis,
                    "WIRE FREE TIP",
                    (
                        int(wire_tip[0]) + 10,
                        int(wire_tip[1]) - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

            if wire_target is not None:

                gt = as_int_point(
                    wire_target
                )

                cv2.circle(
                    vis,
                    gt,
                    8,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    vis,
                    "GRASP TARGET",
                    (
                        gt[0] + 10,
                        gt[1] + 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                if wire_direction is not None:

                    arrow_end = (
                        int(round(
                            gt[0]
                            +
                            45
                            *
                            wire_direction[0]
                        )),
                        int(round(
                            gt[1]
                            +
                            45
                            *
                            wire_direction[1]
                        ))
                    )

                    cv2.arrowedLine(
                        vis,
                        gt,
                        arrow_end,
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA,
                        tipLength=0.25
                    )

            if tweezer_tip is not None:

                tt = as_int_point(
                    tweezer_tip
                )

                cv2.circle(
                    vis,
                    tt,
                    8,
                    (255, 255, 0),
                    -1
                )

                cv2.putText(
                    vis,
                    "TWEEZER TIP",
                    (
                        tt[0] + 10,
                        tt[1] - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2
                )

                if wire_target is not None:

                    gt = as_int_point(
                        wire_target
                    )

                    cv2.line(
                        vis,
                        tt,
                        gt,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

            ready = (
                stable_frames
                >=
                ALIGN_READY_FRAMES
            )

            status = (
                "ALIGN READY 2D"
                if ready
                else
                "SEARCH / ALIGN"
            )

            dist_text = (
                "--"
                if distance is None
                else
                f"{distance:.1f}px"
            )

            cv2.putText(
                vis,
                (
                    f"{status}  "
                    f"dist={dist_text}  "
                    f"stable={stable_frames}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (
                    (0, 255, 0)
                    if ready
                    else
                    (0, 165, 255)
                ),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                vis,
                (
                    f"wire={wire_result.get('valid', False)} "
                    f"conf={wire_result.get('confidence', 0.0):.2f} "
                    f"tip={tip_state}"
                ),
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                vis,
                (
                    "SHADOW ONLY - NO ROBOT / "
                    "NO GRIPPER COMMAND"
                ),
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            # ----------------------------------------------
            # Publish shadow JSON
            # ----------------------------------------------

            write_output(
                wire_result=
                    wire_result,
                wire_target=
                    wire_target,
                wire_direction=
                    wire_direction,
                tweezer_tip=
                    tweezer_tip,
                tip_state=
                    tip_state,
                stable_frames=
                    stable_frames
            )

            now = time.time()

            if (
                now -
                last_log
                >
                1.0
            ):

                print(
                    "[RIGHT WRIST]"
                    f" wire="
                    f"{wire_result.get('valid', False)}"
                    f" conf="
                    f"{wire_result.get('confidence', 0.0):.2f}"
                    f" tweezer="
                    f"{as_int_point(tweezer_tip)}"
                    f" target="
                    f"{as_int_point(wire_target)}"
                    f" dist="
                    f"{dist_text}"
                    f" stable="
                    f"{stable_frames}",
                    flush=True
                )

                last_log = now

            cv2.imshow(
                window,
                vis
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key in (
                ord("q"),
                27
            ):
                break

    finally:

        try:
            pipeline.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()

        OUTPUT_FILE.write_text(
            json.dumps(
                {
                    "enabled": False,
                    "mode":
                        "RIGHT_WRIST_WIRE_GRASP_SHADOW",
                    "camera_serial":
                        RIGHT_WRIST_SERIAL,
                    "timestamp":
                        time.time(),
                    "stopped":
                        True,
                    "gripper_close_command":
                        False
                },
                indent=2
            )
        )

        print(
            "[STOP] Right wrist shadow stopped.",
            flush=True
        )


if __name__ == "__main__":
    main()
