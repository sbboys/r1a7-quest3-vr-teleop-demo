#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import time
import signal
import zlib
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path("/home/robot/unitree_sim_isaaclab_threading")

TOOLS_DIR = (
    PROJECT_ROOT
    / "r1a7_threading_project"
    / "transfer_control"
    / "tools"
)

XR_ROOT = Path(
    "/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/"
    "robot_dev/xr_teleoperate"
)

SDK_ROOT = Path(
    "/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/"
    "robot_dev/unitree_sdk2_python"
)

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SDK_ROOT))
sys.path.insert(0, str(XR_ROOT))

import r1a7_threading_vr_lowcmd as base
from r1a7_lowcmd_guard import acquire_lowcmd_guard

os.chdir(XR_ROOT / "teleop")

from teleop.robot_control.robot_arm_ik import R1A7_ArmIK

import pinocchio as pin

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
)

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    LowCmd_,
    LowState_,
)

from unitree_sdk2py.utils.crc import CRC

from unitree_sdk2py.go2.video.video_client import VideoClient


# ============================================================
# Configuration
# ============================================================

INTERFACE = "enp6s0"
DOMAIN_ID = 0

STATE_TOPIC = "rt/lowstate"
COMMAND_TOPIC = "rt/lowcmd"

PUBLISH_FREQUENCY = 250.0

# Conservative joint speed for first visual experiment.
MAX_JOINT_SPEED = 0.6

LOWSTATE_TIMEOUT = 0.25

# IK
IK_MAX_JOINT_STEP = 0.010
IK_ROTATION_WEIGHT = 1.0

# Right tweezer calibration already verified on the real robot.
RIGHT_OPEN_Q = +0.1585
RIGHT_CLOSE_Q = -0.0500

# Gripper target interpolation speed (rad/s equivalent q/s).
GRIPPER_SPEED = 0.6

# ------------------------------------------------------------
# Jacobian calibration
# ------------------------------------------------------------

# 2 mm is deliberately small but more measurable than 1 mm
# in the wide-angle head camera.
CALIB_STEP_M = 0.002

CALIB_SETTLE_POS_M = 0.0006
CALIB_SETTLE_TIME_S = 0.35
CALIB_PHASE_TIMEOUT_S = 10.0

# ------------------------------------------------------------
# Visual servo
# ------------------------------------------------------------

VISUAL_GAIN = 0.25

# Maximum Cartesian correction per NEW head-camera frame.
MAX_VISUAL_STEP_M = 0.00030       # 0.30 mm

PIXEL_DEADBAND = 4.0

# Do not allow local visual servo to wander arbitrarily far
# from the pose at program startup.
MAX_XY_FROM_HOME_M = 0.030        # 30 mm

MAX_JACOBIAN_CONDITION = 50.0

WINDOW = "R1-A7 Head Visual Servo"


# ============================================================
# Runtime variables used by OpenCV callback
# ============================================================

click_mode = None

tip_pt = None
hole_pt = None
marker_pt = None

latest_frame = None
latest_gray = None
prev_gray = None

tracking_ok_tip = False
tracking_ok_marker = False


# ============================================================
# Helpers
# ============================================================

def get_ee_poses(ik: R1A7_ArmIK, arm_q: np.ndarray):
    q = np.asarray(arm_q, dtype=float).reshape(14)

    pin.framesForwardKinematics(
        ik.reduced_robot.model,
        ik.reduced_robot.data,
        q,
    )

    left = np.asarray(
        ik.reduced_robot.data.oMf[
            ik.L_hand_id
        ].homogeneous,
        dtype=float,
    ).copy()

    right = np.asarray(
        ik.reduced_robot.data.oMf[
            ik.R_hand_id
        ].homogeneous,
        dtype=float,
    ).copy()

    return left, right


def set_right_gripper_goal(output, q: float):
    if not output.enable_gripper:
        return

    q = float(
        np.clip(
            q,
            RIGHT_CLOSE_Q,
            RIGHT_OPEN_Q,
        )
    )

    with output.lock:
        output.gripper_goal[1] = q

        # Clear previous right-gripper contact state.
        output.gripper_hold[1] = False
        output.gripper_stall_since[1] = np.nan
        output.gripper_contact_q[1] = np.nan


def solve_and_send(
    ik,
    output,
    left_target,
    right_target,
    arm_q,
    arm_dq,
    mode_machine,
):
    solution, _ = ik.solve_ik(
        left_target,
        right_target,
        arm_q,
        arm_dq,
        position_only=False,
        max_joint_step=IK_MAX_JOINT_STEP,
        rotation_weight=IK_ROTATION_WEIGHT,
    )

    solution = np.asarray(
        solution,
        dtype=float,
    ).reshape(14)

    if not np.all(np.isfinite(solution)):
        raise RuntimeError(
            "IK returned non-finite solution"
        )

    output.set_arm_goal(
        solution,
        mode_machine,
    )

    return solution


def point_array(pt):
    return np.array(
        [[[float(pt[0]), float(pt[1])]]],
        dtype=np.float32,
    )


def track_one(old_gray, new_gray, pt):
    if (
        old_gray is None
        or new_gray is None
        or pt is None
    ):
        return pt, False

    p0 = point_array(pt)

    p1, status, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        new_gray,
        p0,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if (
        p1 is None
        or status is None
        or int(status[0, 0]) != 1
    ):
        return pt, False

    x = float(p1[0, 0, 0])
    y = float(p1[0, 0, 1])

    if not np.isfinite(x) or not np.isfinite(y):
        return pt, False

    return (x, y), True


def clamp_xy_target(
    target,
    home,
):
    dx = float(
        target[0, 3] - home[0, 3]
    )
    dy = float(
        target[1, 3] - home[1, 3]
    )

    radius = float(
        np.hypot(dx, dy)
    )

    if radius > MAX_XY_FROM_HOME_M:
        scale = (
            MAX_XY_FROM_HOME_M
            / radius
        )

        target[0, 3] = (
            home[0, 3]
            + dx * scale
        )

        target[1, 3] = (
            home[1, 3]
            + dy * scale
        )

        return True

    return False


def mouse_callback(
    event,
    x,
    y,
    flags,
    userdata,
):
    global click_mode
    global tip_pt
    global hole_pt
    global marker_pt
    global tracking_ok_tip
    global tracking_ok_marker

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if click_mode == "tip":
        tip_pt = (float(x), float(y))
        tracking_ok_tip = True
        print(
            "[VISION] P_tip =",
            tip_pt,
            flush=True,
        )

    elif click_mode == "hole":
        hole_pt = (float(x), float(y))
        print(
            "[VISION] P_hole =",
            hole_pt,
            flush=True,
        )

    elif click_mode == "marker":
        marker_pt = (
            float(x),
            float(y),
        )
        tracking_ok_marker = True

        print(
            "[VISION] rigid marker =",
            marker_pt,
            flush=True,
        )

    click_mode = None


def draw_overlay(
    frame,
    visual_enabled,
    jacobian,
    calibration_phase,
    actual_right,
    right_target,
):
    out = frame.copy()

    if hole_pt is not None:
        hp = (
            int(round(hole_pt[0])),
            int(round(hole_pt[1])),
        )

        cv2.circle(
            out,
            hp,
            9,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            out,
            "HOLE",
            (hp[0] + 10, hp[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    if tip_pt is not None:
        tp = (
            int(round(tip_pt[0])),
            int(round(tip_pt[1])),
        )

        cv2.circle(
            out,
            tp,
            8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            out,
            "TIP",
            (tp[0] + 10, tp[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    if marker_pt is not None:
        mp = (
            int(round(marker_pt[0])),
            int(round(marker_pt[1])),
        )

        cv2.drawMarker(
            out,
            mp,
            (255, 0, 255),
            cv2.MARKER_CROSS,
            18,
            2,
        )

        cv2.putText(
            out,
            "MARKER",
            (mp[0] + 10, mp[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )

    y0 = 28

    lines = [
        f"servo={'ON' if visual_enabled else 'OFF'}",
        f"calib={calibration_phase}",
        f"J={'READY' if jacobian is not None else 'NONE'}",
    ]

    if (
        tip_pt is not None
        and hole_pt is not None
    ):
        eu = hole_pt[0] - tip_pt[0]
        ev = hole_pt[1] - tip_pt[1]

        lines.append(
            f"eu={eu:+.1f}px ev={ev:+.1f}px "
            f"|e|={np.hypot(eu, ev):.1f}px"
        )

    if actual_right is not None:
        p = actual_right[:3, 3]
        lines.append(
            "FK xyz="
            f"{p[0]:+.4f},"
            f"{p[1]:+.4f},"
            f"{p[2]:+.4f} m"
        )

    if right_target is not None:
        p = right_target[:3, 3]
        lines.append(
            "TARGET="
            f"{p[0]:+.4f},"
            f"{p[1]:+.4f},"
            f"{p[2]:+.4f} m"
        )

    for line in lines:
        cv2.putText(
            out,
            line,
            (15, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 255, 0),
            2,
        )
        y0 += 25

    help1 = (
        "M marker | J calibrate | "
        "T tip | H hole | V servo"
    )

    help2 = (
        "C close | O open | R home | "
        "Q/ESC stop"
    )

    cv2.putText(
        out,
        help1,
        (15, out.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        out,
        help2,
        (15, out.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
    )

    return out


# ============================================================
# Main
# ============================================================

def main():
    global click_mode
    global tip_pt
    global hole_pt
    global marker_pt
    global latest_frame
    global latest_gray
    global prev_gray
    global tracking_ok_tip
    global tracking_ok_marker

    base.validate_robot_interface(
        INTERFACE
    )

    print(
        f"Robot DDS interface: {INTERFACE}",
        flush=True,
    )

    # Prevent another program from publishing rt/lowcmd.
    guard = acquire_lowcmd_guard(
        Path(__file__).name,
        topic=COMMAND_TOPIC,
    )

    stop = False

    def request_stop(
        _signum=None,
        _frame=None,
    ):
        nonlocal stop
        stop = True

    signal.signal(
        signal.SIGINT,
        request_stop,
    )
    signal.signal(
        signal.SIGTERM,
        request_stop,
    )

    output = None

    try:
        # ----------------------------------------------------
        # DDS
        # ----------------------------------------------------

        ChannelFactoryInitialize(
            DOMAIN_ID,
            INTERFACE,
        )

        crc = CRC()

        state_buffer = base.StateBuffer(
            crc
        )

        subscriber = ChannelSubscriber(
            STATE_TOPIC,
            LowState_,
        )

        subscriber.Init(
            state_buffer.callback,
            10,
        )

        deadline = (
            time.monotonic()
            + 10.0
        )

        state = (
            state_buffer.snapshot()
        )

        while (
            state is None
            and time.monotonic()
            < deadline
        ):
            time.sleep(0.05)
            state = state_buffer.snapshot()

        if state is None:
            raise RuntimeError(
                "No valid rt/lowstate received. "
                "NO command was sent."
            )

        (
            upper_q,
            arm_q,
            arm_dq,
            gripper_q,
            gripper_dq,
            mode_machine,
            received_at,
        ) = state

        print(
            "Measured arm q:",
            np.round(
                arm_q,
                4,
            ).tolist(),
            flush=True,
        )

        # ----------------------------------------------------
        # Confirm debug mode
        # ----------------------------------------------------

        base.check_debug_mode(
            MotionSwitcherClient
        )

        # ----------------------------------------------------
        # IK and current FK
        # ----------------------------------------------------

        ik = R1A7_ArmIK(
            Unit_Test=False,
            Visualization=False,
        )

        ik.reset_target_calibration(
            arm_q
        )

        (
            left_target,
            right_target,
        ) = get_ee_poses(
            ik,
            arm_q,
        )

        left_target = (
            left_target.copy()
        )

        right_target = (
            right_target.copy()
        )

        home_right_target = (
            right_target.copy()
        )

        print(
            "Current right EE xyz:",
            np.round(
                right_target[:3, 3],
                5,
            ).tolist(),
            flush=True,
        )

        # Verify IK of current measured pose.
        first_solution, _ = (
            ik.solve_ik(
                left_target,
                right_target,
                arm_q,
                arm_dq,
                position_only=False,
                max_joint_step=(
                    IK_MAX_JOINT_STEP
                ),
                rotation_weight=(
                    IK_ROTATION_WEIGHT
                ),
            )
        )

        first_solution = np.asarray(
            first_solution,
            dtype=float,
        ).reshape(14)

        start_error = float(
            np.max(
                np.abs(
                    first_solution
                    - arm_q
                )
            )
        )

        print(
            f"Startup IK difference: "
            f"{start_error:.5f} rad",
            flush=True,
        )

        # Current-pose FK target should not require
        # any meaningful jump.
        if start_error > 0.08:
            raise RuntimeError(
                "Startup IK difference too large. "
                "Refusing real-robot enable."
            )

        # ----------------------------------------------------
        # LowCmd publisher
        # ----------------------------------------------------

        publisher = ChannelPublisher(
            COMMAND_TOPIC,
            LowCmd_,
        )

        publisher.Init()

        output = base.R1A7LowCmdOutput(
            publisher,
            unitree_hg_msg_dds__LowCmd_,
            crc,
            PUBLISH_FREQUENCY,
            MAX_JOINT_SPEED,
            enable_gripper=True,
            gripper_kp=8.0,
            gripper_kd=0.4,
            gripper_speed=GRIPPER_SPEED,

            # Disable old generic contact-hold heuristic
            # for the custom tweezer.
            gripper_contact_hold=False,
        )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        video = VideoClient()
        video.SetTimeout(3.0)
        video.Init()

        print()
        print(
            "WARNING: REAL R1-A7 rt/lowcmd.",
            flush=True,
        )
        print(
            "No Quest / VR input is used.",
            flush=True,
        )
        print(
            "Head and waist will be held at "
            "their measured startup posture.",
            flush=True,
        )
        print(
            "Keep emergency stop ready.",
            flush=True,
        )
        print()

        answer = input(
            "Type ENABLE to take over the "
            "current robot posture: "
        ).strip()

        if answer.casefold() != "enable":
            print(
                "Aborted. No LowCmd was sent.",
                flush=True,
            )
            return 2

        # Current measured q first:
        # avoids any takeover jump.
        output.enable(
            upper_q,
            mode_machine,
            gripper_q=gripper_q,
            gripper_initial_mode="current",
            right_gripper_open_cap_current=True,
        )

        time.sleep(0.10)

        output.set_arm_goal(
            first_solution,
            mode_machine,
        )

        print(
            "[ACTIVE] LowCmd enabled.",
            flush=True,
        )

        print(
            "Recommended first sequence:",
            flush=True,
        )
        print(
            "  1) M + click a rigid tweezer point",
            flush=True,
        )
        print(
            "  2) J = automatic local Jacobian calibration",
            flush=True,
        )
        print(
            "  3) C = close tweezer on the wire",
            flush=True,
        )
        print(
            "  4) T + click wire tip",
            flush=True,
        )
        print(
            "  5) H + click target hole",
            flush=True,
        )
        print(
            "  6) V = visual XY servo",
            flush=True,
        )

        # ----------------------------------------------------
        # OpenCV
        # ----------------------------------------------------

        cv2.namedWindow(
            WINDOW,
            cv2.WINDOW_NORMAL,
        )

        cv2.setMouseCallback(
            WINDOW,
            mouse_callback,
        )

        last_crc = None

        jacobian = None

        visual_enabled = False

        # Calibration state machine.
        calibration_phase = "IDLE"
        phase_started = 0.0
        settle_since = None

        calib_home = None

        ref_x_marker = None
        ref_x_pos = None

        ref_y_marker = None
        ref_y_pos = None

        d_img_x = None
        d_pos_x = None

        d_img_y = None
        d_pos_y = None

        last_print = 0.0

        while not stop:

            # ------------------------------------------------
            # Robot state safety
            # ------------------------------------------------

            now = time.monotonic()

            state = (
                state_buffer.snapshot()
            )

            if state is None:
                raise RuntimeError(
                    "robot lowstate disappeared"
                )

            (
                _upper_q,
                arm_q,
                arm_dq,
                gripper_q,
                gripper_dq,
                mode_machine,
                received_at,
            ) = state

            state_age = (
                now - received_at
            )

            if (
                state_age
                > LOWSTATE_TIMEOUT
            ):
                raise RuntimeError(
                    "robot lowstate stale for "
                    f"{state_age:.3f}s"
                )

            if output.error:
                raise RuntimeError(
                    "LowCmd publisher failed: "
                    + output.error
                )

            actual_left, actual_right = (
                get_ee_poses(
                    ik,
                    arm_q,
                )
            )

            # ------------------------------------------------
            # Head RGB
            # ------------------------------------------------

            ret, data = (
                video.GetImageSample()
            )

            if (
                ret != 0
                or data is None
                or len(data) == 0
            ):
                continue

            raw = bytes(data)

            crc_img = zlib.crc32(
                raw
            )

            # Only process genuinely new RGB frames.
            if crc_img == last_crc:
                continue

            last_crc = crc_img

            frame = cv2.imdecode(
                np.frombuffer(
                    raw,
                    dtype=np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                continue

            latest_frame = frame

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            latest_gray = gray

            # ------------------------------------------------
            # Optical-flow tracking
            # ------------------------------------------------

            if prev_gray is not None:

                if tip_pt is not None:
                    tip_pt_new, ok = (
                        track_one(
                            prev_gray,
                            gray,
                            tip_pt,
                        )
                    )

                    if ok:
                        tip_pt = tip_pt_new
                        tracking_ok_tip = True
                    else:
                        tracking_ok_tip = False

                        if visual_enabled:
                            visual_enabled = False
                            print(
                                "[SAFETY] P_tip tracking lost; "
                                "visual servo OFF.",
                                flush=True,
                            )

                if marker_pt is not None:
                    marker_new, ok = (
                        track_one(
                            prev_gray,
                            gray,
                            marker_pt,
                        )
                    )

                    if ok:
                        marker_pt = marker_new
                        tracking_ok_marker = True
                    else:
                        tracking_ok_marker = False

                        if calibration_phase != "IDLE":
                            calibration_phase = "ABORT_RETURN"
                            right_target = (
                                home_right_target.copy()
                            )

                            print(
                                "[CALIB] marker tracking lost; "
                                "returning home.",
                                flush=True,
                            )

            prev_gray = gray.copy()

            # ------------------------------------------------
            # Calibration state machine
            # ------------------------------------------------

            if calibration_phase != "IDLE":

                pos_error = float(
                    np.linalg.norm(
                        actual_right[:3, 3]
                        - right_target[:3, 3]
                    )
                )

                if (
                    pos_error
                    < CALIB_SETTLE_POS_M
                ):
                    if settle_since is None:
                        settle_since = now
                else:
                    settle_since = None

                settled = (
                    settle_since is not None
                    and (
                        now - settle_since
                        >= CALIB_SETTLE_TIME_S
                    )
                )

                timed_out = (
                    now - phase_started
                    >= CALIB_PHASE_TIMEOUT_S
                )

                if calibration_phase == "X_OUT":
                    solve_and_send(
                        ik,
                        output,
                        left_target,
                        right_target,
                        arm_q,
                        arm_dq,
                        mode_machine,
                    )

                    if settled or timed_out:
                        if (
                            marker_pt is None
                            or not tracking_ok_marker
                        ):
                            calibration_phase = "ABORT_RETURN"
                        else:
                            d_img_x = (
                                np.asarray(
                                    marker_pt,
                                    dtype=float,
                                )
                                - ref_x_marker
                            )

                            d_pos_x = (
                                actual_right[:2, 3]
                                - ref_x_pos[:2]
                            )

                            print(
                                "[CALIB X] image delta px =",
                                np.round(
                                    d_img_x,
                                    3,
                                ).tolist(),
                                " actual XY m =",
                                np.round(
                                    d_pos_x,
                                    6,
                                ).tolist(),
                                flush=True,
                            )

                            right_target = (
                                calib_home.copy()
                            )

                            calibration_phase = (
                                "X_RETURN"
                            )

                            phase_started = now
                            settle_since = None

                elif calibration_phase == "X_RETURN":
                    solve_and_send(
                        ik,
                        output,
                        left_target,
                        right_target,
                        arm_q,
                        arm_dq,
                        mode_machine,
                    )

                    if settled or timed_out:
                        if (
                            marker_pt is None
                            or not tracking_ok_marker
                        ):
                            calibration_phase = (
                                "ABORT_RETURN"
                            )
                        else:
                            ref_y_marker = (
                                np.asarray(
                                    marker_pt,
                                    dtype=float,
                                ).copy()
                            )

                            ref_y_pos = (
                                actual_right[
                                    :3, 3
                                ].copy()
                            )

                            right_target = (
                                calib_home.copy()
                            )

                            right_target[
                                1, 3
                            ] += CALIB_STEP_M

                            calibration_phase = (
                                "Y_OUT"
                            )

                            phase_started = now
                            settle_since = None

                            print(
                                "[CALIB] testing model +Y",
                                flush=True,
                            )

                elif calibration_phase == "Y_OUT":
                    solve_and_send(
                        ik,
                        output,
                        left_target,
                        right_target,
                        arm_q,
                        arm_dq,
                        mode_machine,
                    )

                    if settled or timed_out:
                        if (
                            marker_pt is None
                            or not tracking_ok_marker
                        ):
                            calibration_phase = (
                                "ABORT_RETURN"
                            )
                        else:
                            d_img_y = (
                                np.asarray(
                                    marker_pt,
                                    dtype=float,
                                )
                                - ref_y_marker
                            )

                            d_pos_y = (
                                actual_right[:2, 3]
                                - ref_y_pos[:2]
                            )

                            print(
                                "[CALIB Y] image delta px =",
                                np.round(
                                    d_img_y,
                                    3,
                                ).tolist(),
                                " actual XY m =",
                                np.round(
                                    d_pos_y,
                                    6,
                                ).tolist(),
                                flush=True,
                            )

                            right_target = (
                                calib_home.copy()
                            )

                            calibration_phase = (
                                "Y_RETURN"
                            )

                            phase_started = now
                            settle_since = None

                elif calibration_phase == "Y_RETURN":
                    solve_and_send(
                        ik,
                        output,
                        left_target,
                        right_target,
                        arm_q,
                        arm_dq,
                        mode_machine,
                    )

                    if settled or timed_out:

                        D_img = np.column_stack(
                            (
                                d_img_x,
                                d_img_y,
                            )
                        )

                        D_robot = np.column_stack(
                            (
                                d_pos_x,
                                d_pos_y,
                            )
                        )

                        try:
                            if (
                                abs(
                                    np.linalg.det(
                                        D_robot
                                    )
                                )
                                < 1e-9
                            ):
                                raise RuntimeError(
                                    "actual calibration XY "
                                    "displacements are singular"
                                )

                            jacobian = (
                                D_img
                                @ np.linalg.inv(
                                    D_robot
                                )
                            )

                            cond = float(
                                np.linalg.cond(
                                    jacobian
                                )
                            )

                            print(
                                "[CALIB] J_img "
                                "(pixel / meter):",
                                flush=True,
                            )

                            print(
                                jacobian,
                                flush=True,
                            )

                            print(
                                f"[CALIB] condition="
                                f"{cond:.2f}",
                                flush=True,
                            )

                            image_motion_x = float(
                                np.linalg.norm(
                                    d_img_x
                                )
                            )

                            image_motion_y = float(
                                np.linalg.norm(
                                    d_img_y
                                )
                            )

                            if (
                                image_motion_x
                                < 1.0
                                or image_motion_y
                                < 1.0
                            ):
                                raise RuntimeError(
                                    "calibration image motion "
                                    "is below 1 pixel; "
                                    "head RGB resolution is "
                                    "insufficient at this distance"
                                )

                            if (
                                not np.isfinite(cond)
                                or cond
                                > MAX_JACOBIAN_CONDITION
                            ):
                                raise RuntimeError(
                                    "image Jacobian is "
                                    "ill-conditioned"
                                )

                            print(
                                "[CALIB] SUCCESS. "
                                "V is now available.",
                                flush=True,
                            )

                        except Exception as exc:
                            jacobian = None

                            print(
                                "[CALIB] FAILED:",
                                exc,
                                flush=True,
                            )

                        calibration_phase = (
                            "IDLE"
                        )

                        right_target = (
                            calib_home.copy()
                        )

                elif calibration_phase == "ABORT_RETURN":
                    right_target = (
                        home_right_target.copy()
                    )

                    solve_and_send(
                        ik,
                        output,
                        left_target,
                        right_target,
                        arm_q,
                        arm_dq,
                        mode_machine,
                    )

                    if settled or timed_out:
                        calibration_phase = (
                            "IDLE"
                        )

                        jacobian = None

                        print(
                            "[CALIB] aborted.",
                            flush=True,
                        )

            # ------------------------------------------------
            # Visual servo
            # ------------------------------------------------

            elif visual_enabled:

                if (
                    jacobian is None
                    or tip_pt is None
                    or hole_pt is None
                    or not tracking_ok_tip
                ):
                    visual_enabled = False

                    print(
                        "[SAFETY] Servo prerequisites lost. "
                        "Servo OFF.",
                        flush=True,
                    )

                else:
                    error_px = np.asarray(
                        [
                            hole_pt[0]
                            - tip_pt[0],

                            hole_pt[1]
                            - tip_pt[1],
                        ],
                        dtype=float,
                    )

                    error_norm = float(
                        np.linalg.norm(
                            error_px
                        )
                    )

                    if (
                        error_norm
                        > PIXEL_DEADBAND
                    ):
                        delta_xy = (
                            VISUAL_GAIN
                            * np.linalg.pinv(
                                jacobian
                            )
                            @ error_px
                        )

                        step_norm = float(
                            np.linalg.norm(
                                delta_xy
                            )
                        )

                        if (
                            step_norm
                            > MAX_VISUAL_STEP_M
                        ):
                            delta_xy *= (
                                MAX_VISUAL_STEP_M
                                / step_norm
                            )

                        candidate = (
                            right_target.copy()
                        )

                        candidate[
                            0, 3
                        ] += float(
                            delta_xy[0]
                        )

                        candidate[
                            1, 3
                        ] += float(
                            delta_xy[1]
                        )

                        hit_limit = (
                            clamp_xy_target(
                                candidate,
                                home_right_target,
                            )
                        )

                        if hit_limit:
                            visual_enabled = False

                            print(
                                "[SAFETY] 30 mm local "
                                "workspace limit reached. "
                                "Servo OFF.",
                                flush=True,
                            )

                        else:
                            right_target = (
                                candidate
                            )

                            solve_and_send(
                                ik,
                                output,
                                left_target,
                                right_target,
                                arm_q,
                                arm_dq,
                                mode_machine,
                            )

                    else:
                        # Inside image deadband:
                        # simply keep the current target.
                        pass

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            vis = draw_overlay(
                frame,
                visual_enabled,
                jacobian,
                calibration_phase,
                actual_right,
                right_target,
            )

            cv2.imshow(
                WINDOW,
                vis,
            )

            key = cv2.waitKey(1) & 0xFF

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            if key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                print(
                    "[STOP] User requested stop.",
                    flush=True,
                )
                break

            elif key in (
                ord("m"),
                ord("M"),
            ):
                click_mode = "marker"

                print(
                    "[VISION] Click a rigid point "
                    "on the right tweezer.",
                    flush=True,
                )

            elif key in (
                ord("t"),
                ord("T"),
            ):
                click_mode = "tip"

                print(
                    "[VISION] Click insertion wire tip.",
                    flush=True,
                )

            elif key in (
                ord("h"),
                ord("H"),
            ):
                click_mode = "hole"

                print(
                    "[VISION] Click target hole center.",
                    flush=True,
                )

            elif key in (
                ord("c"),
                ord("C"),
            ):
                visual_enabled = False

                set_right_gripper_goal(
                    output,
                    RIGHT_CLOSE_Q,
                )

                print(
                    "[GRIPPER] RIGHT CLOSE -> "
                    f"{RIGHT_CLOSE_Q:+.4f}",
                    flush=True,
                )

            elif key in (
                ord("o"),
                ord("O"),
            ):
                visual_enabled = False

                set_right_gripper_goal(
                    output,
                    RIGHT_OPEN_Q,
                )

                print(
                    "[GRIPPER] RIGHT OPEN -> "
                    f"{RIGHT_OPEN_Q:+.4f}",
                    flush=True,
                )

            elif key in (
                ord("r"),
                ord("R"),
            ):
                visual_enabled = False

                right_target = (
                    home_right_target.copy()
                )

                solve_and_send(
                    ik,
                    output,
                    left_target,
                    right_target,
                    arm_q,
                    arm_dq,
                    mode_machine,
                )

                print(
                    "[ARM] returning to startup "
                    "right EE target.",
                    flush=True,
                )

            elif key in (
                ord("j"),
                ord("J"),
            ):
                if calibration_phase != "IDLE":
                    print(
                        "[CALIB] already running.",
                        flush=True,
                    )

                elif (
                    marker_pt is None
                    or not tracking_ok_marker
                ):
                    print(
                        "[CALIB] Press M and click "
                        "a rigid tweezer point first.",
                        flush=True,
                    )

                else:
                    visual_enabled = False
                    jacobian = None

                    # Calibrate from the CURRENT pose.
                    _, actual_now = (
                        get_ee_poses(
                            ik,
                            arm_q,
                        )
                    )

                    calib_home = (
                        actual_now.copy()
                    )

                    right_target = (
                        calib_home.copy()
                    )

                    # Keep startup reference updated to
                    # calibration position for local servo.
                    home_right_target = (
                        calib_home.copy()
                    )

                    ref_x_marker = np.asarray(
                        marker_pt,
                        dtype=float,
                    ).copy()

                    ref_x_pos = (
                        actual_now[
                            :3, 3
                        ].copy()
                    )

                    right_target[
                        0, 3
                    ] += CALIB_STEP_M

                    calibration_phase = (
                        "X_OUT"
                    )

                    phase_started = now
                    settle_since = None

                    print(
                        "[CALIB] START.",
                        flush=True,
                    )

                    print(
                        f"[CALIB] model +X "
                        f"{CALIB_STEP_M * 1000:.1f} mm",
                        flush=True,
                    )

            elif key in (
                ord("v"),
                ord("V"),
            ):
                if visual_enabled:
                    visual_enabled = False

                    print(
                        "[SERVO] OFF",
                        flush=True,
                    )

                elif calibration_phase != "IDLE":
                    print(
                        "[SERVO] calibration still running.",
                        flush=True,
                    )

                elif jacobian is None:
                    print(
                        "[SERVO] no valid Jacobian. "
                        "Run M -> J first.",
                        flush=True,
                    )

                elif (
                    tip_pt is None
                    or hole_pt is None
                ):
                    print(
                        "[SERVO] Set T and H first.",
                        flush=True,
                    )

                elif not tracking_ok_tip:
                    print(
                        "[SERVO] P_tip tracking invalid. "
                        "Set T again.",
                        flush=True,
                    )

                else:
                    visual_enabled = True

                    print(
                        "[SERVO] ON",
                        flush=True,
                    )

            # ------------------------------------------------
            # Console status
            # ------------------------------------------------

            if now - last_print >= 1.0:
                last_print = now

                msg = (
                    f"[STATE] "
                    f"gripperR={gripper_q[1]:+.4f} "
                    f"servo={int(visual_enabled)} "
                    f"calib={calibration_phase}"
                )

                if (
                    tip_pt is not None
                    and hole_pt is not None
                ):
                    eu = (
                        hole_pt[0]
                        - tip_pt[0]
                    )

                    ev = (
                        hole_pt[1]
                        - tip_pt[1]
                    )

                    msg += (
                        f" eu={eu:+.1f} "
                        f"ev={ev:+.1f}"
                    )

                print(
                    msg,
                    flush=True,
                )

        return 0

    finally:
        if output is not None:
            output.close()

            print(
                "R1-A7 LowCmd publisher stopped.",
                flush=True,
            )

        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
