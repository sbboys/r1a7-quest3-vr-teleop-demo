#!/usr/bin/env python3
"""R1-A7 arm smoothness baseline: no vision, no continuous IK.

Two isolated tests are provided:
  1) joint: one small joint excursion using a quintic minimum-jerk profile.
  2) cartesian: solve IK ONCE for a small +X/+Z displacement, then execute the
     resulting joint target with the same minimum-jerk profile.

The LowCmd path follows the current Unitree R1-A7 low-level example baseline:
  - 500 Hz command period by default
  - arm Kp/Kd = [100,100,100,100,50,50,50] / [2,...,2] per arm
  - dq_target = 0
  - tau_ff = 0

This script intentionally does NOT use visual servoing or continuous Cartesian IK.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np


ARM_INDICES = tuple(range(15, 29))
ARM_NAMES = (
    "L_SHOULDER_PITCH", "L_SHOULDER_ROLL", "L_SHOULDER_YAW",
    "L_ELBOW", "L_WRIST_ROLL", "L_WRIST_PITCH", "L_WRIST_YAW",
    "R_SHOULDER_PITCH", "R_SHOULDER_ROLL", "R_SHOULDER_YAW",
    "R_ELBOW", "R_WRIST_ROLL", "R_WRIST_PITCH", "R_WRIST_YAW",
)
NAME_TO_INDEX = {name: i for i, name in enumerate(ARM_NAMES)}

# Same R1-A7 limits already used by the project controller.
ARM_JOINT_LOWER_Q = np.asarray(
    [-3.1416, -0.22689, -1.9199, -0.97564, -1.9199, -1.6144, -1.6144,
     -3.1416, -2.47849, -1.9199, -0.97564, -1.9199, -1.6144, -1.6144],
    dtype=float,
)
ARM_JOINT_UPPER_Q = np.asarray(
    [2.0944, 2.4784, 1.9199, 2.1852, 1.9199, 1.6144, 1.6144,
     2.0944, 0.2268, 1.9199, 2.1852, 1.9199, 1.6144, 1.6144],
    dtype=float,
)
LIMIT_MARGIN_RAD = np.deg2rad(2.0)

# Unitree R1-A7 official low-level example gains (arms only).
OFFICIAL_ARM_KP = np.asarray([100, 100, 100, 100, 50, 50, 50] * 2, dtype=float)
OFFICIAL_ARM_KD = np.asarray([2, 2, 2, 2, 2, 2, 2] * 2, dtype=float)

DEFAULT_SDK_PYTHON = Path(
    "/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/unitree_sdk2_python"
)
DEFAULT_XR_ROOT = Path(
    "/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/xr_teleoperate"
)


def minimum_jerk(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def minimum_jerk_derivative(u: float) -> float:
    """d/du of the quintic minimum-jerk time scaling."""
    u = float(np.clip(u, 0.0, 1.0))
    return 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4


def validate_interface(interface: str) -> str:
    path = Path("/sys/class/net") / interface
    if not path.exists():
        raise RuntimeError(f"DDS interface {interface!r} does not exist")
    address = "unknown"
    try:
        import socket
        import fcntl
        import struct
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        request = struct.pack("256s", interface[:15].encode("ascii"))
        address = socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, request)[20:24])
    except Exception:
        pass
    return address


class StateBuffer:
    def __init__(self, crc):
        self.crc = crc
        self.lock = threading.Lock()
        self.message = None
        self.received_at = 0.0
        self.received_count = 0
        self.crc_errors = 0

    def callback(self, message) -> None:
        self.received_count += 1
        try:
            if int(message.crc) != int(self.crc.Crc(message)):
                self.crc_errors += 1
                return
        except Exception:
            self.crc_errors += 1
            return
        with self.lock:
            self.message = message
            self.received_at = time.monotonic()

    def snapshot(self):
        with self.lock:
            if self.message is None:
                return None
            msg = self.message
            q = np.asarray([msg.motor_state[i].q for i in ARM_INDICES], dtype=float)
            dq = np.asarray([msg.motor_state[i].dq for i in ARM_INDICES], dtype=float)
            tau_est = np.asarray(
                [getattr(msg.motor_state[i], "tau_est", 0.0) for i in ARM_INDICES],
                dtype=float,
            )
            motorstate = np.asarray(
                [getattr(msg.motor_state[i], "motorstate", 0) for i in ARM_INDICES],
                dtype=int,
            )
            return q, dq, tau_est, motorstate, int(msg.mode_machine), self.received_at


class ArmLowCmdPublisher:
    def __init__(self, publisher, low_cmd_factory, crc, mode_machine: int):
        self.publisher = publisher
        self.message = low_cmd_factory()
        self.crc = crc
        self.mode_machine = int(mode_machine)

    def write(self, q_cmd: np.ndarray, tau_ff_cmd: Optional[np.ndarray] = None) -> None:
        q_cmd = np.asarray(q_cmd, dtype=float).reshape(14)
        if tau_ff_cmd is None:
            tau_ff_cmd = np.zeros(14, dtype=float)
        else:
            tau_ff_cmd = np.asarray(tau_ff_cmd, dtype=float).reshape(14)
        if not np.all(np.isfinite(tau_ff_cmd)):
            raise ValueError("refusing non-finite tau_ff command")
        self.message.mode_pr = 0
        self.message.mode_machine = self.mode_machine
        for local_i, motor_i in enumerate(ARM_INDICES):
            motor = self.message.motor_cmd[motor_i]
            motor.mode = 1
            motor.tau = float(tau_ff_cmd[local_i])
            motor.q = float(q_cmd[local_i])
            motor.dq = 0.0
            motor.kp = float(OFFICIAL_ARM_KP[local_i])
            motor.kd = float(OFFICIAL_ARM_KD[local_i])
        self.message.crc = self.crc.Crc(self.message)
        self.publisher.Write(self.message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R1-A7 no-vision arm smoothness baseline test"
    )
    parser.add_argument("--interface", required=True)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--state-topic", default="rt/lowstate")
    parser.add_argument("--command-topic", default="rt/lowcmd")
    parser.add_argument("--sdk-python-root", type=Path, default=DEFAULT_SDK_PYTHON)
    parser.add_argument("--xr-root", type=Path, default=DEFAULT_XR_ROOT)
    parser.add_argument("--test", choices=("joint", "cartesian"), default="joint")
    parser.add_argument("--joint", choices=ARM_NAMES, default="R_WRIST_PITCH")
    parser.add_argument("--amplitude-deg", type=float, default=1.0)
    parser.add_argument("--axis", choices=("x", "z"), default="z")
    parser.add_argument("--step-mm", type=float, default=2.0)
    parser.add_argument("--duration-s", type=float, default=2.5)
    parser.add_argument("--hold-before-s", type=float, default=0.5)
    parser.add_argument("--hold-at-goal-s", type=float, default=0.5)
    parser.add_argument("--hold-after-s", type=float, default=1.0)
    parser.add_argument("--publish-frequency", type=float, default=500.0)
    parser.add_argument("--state-timeout", type=float, default=0.20)
    parser.add_argument("--max-goal-joint-delta-deg", type=float, default=8.0)
    parser.add_argument("--max-tracking-error-deg", type=float, default=12.0)
    parser.add_argument("--max-actual-speed-deg-s", type=float, default=90.0)
    parser.add_argument(
        "--friction-ff-nm",
        type=float,
        default=0.0,
        help=(
            "Symmetric low-speed friction feedforward magnitude for the selected "
            "joint test. Applied smoothly from qdot_cmd with tanh; holds use zero."
        ),
    )
    parser.add_argument(
        "--friction-velocity-scale-deg-s",
        type=float,
        default=0.15,
        help="Velocity scale for tanh friction transition (deg/s).",
    )
    parser.add_argument(
        "--friction-ff-positive-nm",
        type=float,
        default=None,
        help=(
            "Optional positive-direction friction feedforward magnitude. "
            "Defaults to --friction-ff-nm."
        ),
    )
    parser.add_argument(
        "--friction-ff-negative-nm",
        type=float,
        default=None,
        help=(
            "Optional negative-direction friction feedforward magnitude. "
            "Defaults to --friction-ff-nm."
        ),
    )
    parser.add_argument(
        "--stiction-settle-assist",
        action="store_true",
        help=(
            "During hold_goal/hold_end, retain a smooth direction-latched "
            "assist while the selected joint is still behind the target."
        ),
    )
    parser.add_argument(
        "--stiction-error-on-deg",
        type=float,
        default=0.10,
        help="Full settle assist at/above this selected-joint error (deg).",
    )
    parser.add_argument(
        "--stiction-error-off-deg",
        type=float,
        default=0.05,
        help="Settle assist is zero at/below this selected-joint error (deg).",
    )
    parser.add_argument(
        "--stiction-assist-ramp-s",
        type=float,
        default=0.30,
        help=(
            "Minimum-jerk ramp time for settle assist at the beginning "
            "of hold_goal/hold_end (s)."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp"))
    args = parser.parse_args()

    if not (250.0 <= args.publish_frequency <= 1000.0):
        parser.error("--publish-frequency must be within [250, 1000] Hz")
    if args.duration_s < 1.0:
        parser.error("--duration-s must be >= 1.0 s")
    if not (0.05 <= abs(args.amplitude_deg) <= 5.0):
        parser.error("absolute --amplitude-deg must be within [0.05, 5.0]")
    if not (0.2 <= abs(args.step_mm) <= 5.0):
        parser.error("absolute --step-mm must be within [0.2, 5.0]")
    if not (0.0 <= args.friction_ff_nm <= 0.20):
        parser.error("--friction-ff-nm must be within [0.0, 0.20] Nm for this diagnostic")

    if args.friction_ff_positive_nm is None:
        args.friction_ff_positive_nm = args.friction_ff_nm

    if args.friction_ff_negative_nm is None:
        args.friction_ff_negative_nm = args.friction_ff_nm

    if not (0.0 <= args.friction_ff_positive_nm <= 0.20):
        parser.error("--friction-ff-positive-nm must be within [0.0, 0.20] Nm")

    if not (0.0 <= args.friction_ff_negative_nm <= 0.20):
        parser.error("--friction-ff-negative-nm must be within [0.0, 0.20] Nm")

    if not (0.02 <= args.friction_velocity_scale_deg_s <= 1.0):
        parser.error("--friction-velocity-scale-deg-s must be within [0.02, 1.0] deg/s")

    if not (
        0.0
        <= args.stiction_error_off_deg
        < args.stiction_error_on_deg
        <= 1.0
    ):
        parser.error(
            "require 0 <= --stiction-error-off-deg "
            "< --stiction-error-on-deg <= 1.0"
        )

    any_ff = max(
        args.friction_ff_positive_nm,
        args.friction_ff_negative_nm,
    ) > 0.0

    if args.test != "joint" and (
        any_ff or args.stiction_settle_assist
    ):
        parser.error(
            "friction/stiction compensation is enabled only for --test joint"
        )

    if args.stiction_settle_assist and not any_ff:
        parser.error(
            "--stiction-settle-assist requires a nonzero friction FF magnitude"
        )

    if not (0.05 <= args.stiction_assist_ramp_s <= 2.0):
        parser.error(
            "--stiction-assist-ramp-s must be within [0.05, 2.0] s"
        )
    return args


def wait_for_state(buffer: StateBuffer, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = buffer.snapshot()
        if state is not None:
            return state
        time.sleep(0.02)
    raise RuntimeError(
        f"no valid LowState received (raw={buffer.received_count}, crc_errors={buffer.crc_errors})"
    )


def compute_goal(args, q_start, dq_start):
    if args.test == "joint":
        q_goal = q_start.copy()
        idx = NAME_TO_INDEX[args.joint]
        q_goal[idx] += math.radians(args.amplitude_deg)
        soft_low = ARM_JOINT_LOWER_Q + LIMIT_MARGIN_RAD
        soft_high = ARM_JOINT_UPPER_Q - LIMIT_MARGIN_RAD
        if not (soft_low[idx] <= q_goal[idx] <= soft_high[idx]):
            raise RuntimeError(
                f"requested {args.joint} goal {q_goal[idx]:+.4f} rad violates 2-deg soft limit"
            )
        return q_goal, {"kind": "joint", "joint": args.joint}

    xr_root = args.xr_root.expanduser().resolve()
    if not (xr_root / "teleop").is_dir():
        raise RuntimeError(f"xr_teleoperate checkout not found: {xr_root}")
    if str(xr_root) not in sys.path:
        sys.path.insert(0, str(xr_root))
    import pinocchio as pin
    from teleop.robot_control.robot_arm_ik import R1A7_ArmIK

    ik = R1A7_ArmIK(Unit_Test=False, Visualization=False)
    ik.reset_target_calibration(q_start)
    pin.framesForwardKinematics(ik.reduced_robot.model, ik.reduced_robot.data, q_start)
    left_start = np.asarray(
        ik.reduced_robot.data.oMf[ik.L_hand_id].homogeneous, dtype=float
    ).copy()
    right_start = np.asarray(
        ik.reduced_robot.data.oMf[ik.R_hand_id].homogeneous, dtype=float
    ).copy()
    right_target = right_start.copy()
    axis_i = {"x": 0, "z": 2}[args.axis]
    right_target[axis_i, 3] += args.step_mm / 1000.0

    q_goal, _tau = ik.solve_ik(
        left_start,
        right_target,
        q_start,
        dq_start,
        position_only=False,
        max_joint_step=0.080,
        rotation_weight=0.05,
    )
    q_goal = np.asarray(q_goal, dtype=float).reshape(14)
    pin.framesForwardKinematics(ik.reduced_robot.model, ik.reduced_robot.data, q_goal)
    right_pred = np.asarray(
        ik.reduced_robot.data.oMf[ik.R_hand_id].homogeneous, dtype=float
    ).copy()
    predicted_delta_mm = (right_pred[:3, 3] - right_start[:3, 3]) * 1000.0
    return q_goal, {
        "kind": "cartesian",
        "axis": args.axis,
        "step_mm": args.step_mm,
        "predicted_delta_mm": predicted_delta_mm.tolist(),
    }


def make_log_path(args) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    label = args.joint.lower() if args.test == "joint" else f"{args.axis}_{args.step_mm:g}mm"
    return args.log_dir.expanduser() / f"r1a7_arm_smoothness_{args.test}_{label}_{stamp}.csv"


def main() -> int:
    args = parse_args()
    sdk_root = args.sdk_python_root.expanduser().resolve()
    if not (sdk_root / "unitree_sdk2py").is_dir():
        raise RuntimeError(f"unitree_sdk2_python checkout not found: {sdk_root}")
    sys.path.insert(0, str(sdk_root))

    here = Path(__file__).resolve()
    tools_dir = here.parents[1] / "tools"
    if tools_dir.is_dir():
        sys.path.insert(0, str(tools_dir))

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    from r1a7_lowcmd_guard import acquire_lowcmd_guard

    print("R1-A7 ELBOW CONTINUOUS LOW-SPEED V2 DITHER", flush=True)
    print("NO vision. NO continuous Cartesian IK.", flush=True)
    print(
        f"test={args.test} publish={args.publish_frequency:g} Hz "
        f"friction_ff_pos={args.friction_ff_positive_nm:.3f} Nm "
        f"friction_ff_neg={args.friction_ff_negative_nm:.3f} Nm "
        f"velocity_scale={args.friction_velocity_scale_deg_s:.3f} deg/s "
        f"settle_assist={args.stiction_settle_assist} "
        f"error_band={args.stiction_error_off_deg:.3f}/"
        f"{args.stiction_error_on_deg:.3f} deg "
        f"settle_ramp={args.stiction_assist_ramp_s:.3f}s",
        flush=True,
    )
    print(f"official arm gains: Kp={OFFICIAL_ARM_KP.tolist()} Kd={OFFICIAL_ARM_KD.tolist()}", flush=True)
    print(f"DDS interface: {args.interface} ({validate_interface(args.interface)})", flush=True)

    ChannelFactoryInitialize(args.domain_id, args.interface)
    crc = CRC()
    state_buffer = StateBuffer(crc)
    subscriber = ChannelSubscriber(args.state_topic, LowState_)
    subscriber.Init(state_buffer.callback, 10)
    state = wait_for_state(state_buffer)
    q_start, dq_start, tau_start, motorstate, mode_machine, received_at = state

    if np.any(motorstate != 0):
        bad = [(ARM_NAMES[i], int(code)) for i, code in enumerate(motorstate) if code != 0]
        raise RuntimeError(f"arm motorstate reports errors: {bad}")

    q_goal, goal_meta = compute_goal(args, q_start, dq_start)
    joint_delta_deg = np.rad2deg(q_goal - q_start)
    max_goal_delta_deg = float(np.max(np.abs(joint_delta_deg)))
    print("start q rad:", np.round(q_start, 5).tolist(), flush=True)
    print("goal metadata:", json.dumps(goal_meta, ensure_ascii=False), flush=True)
    print("goal joint delta deg:", np.round(joint_delta_deg, 4).tolist(), flush=True)
    print(f"max |goal joint delta| = {max_goal_delta_deg:.4f} deg", flush=True)

    if max_goal_delta_deg > args.max_goal_joint_delta_deg:
        raise RuntimeError(
            f"one-shot goal requires {max_goal_delta_deg:.2f} deg joint motion; "
            f"limit is {args.max_goal_joint_delta_deg:.2f} deg"
        )
    soft_low = ARM_JOINT_LOWER_Q + LIMIT_MARGIN_RAD
    soft_high = ARM_JOINT_UPPER_Q - LIMIT_MARGIN_RAD
    if np.any(q_goal < soft_low) or np.any(q_goal > soft_high):
        raise RuntimeError("one-shot IK goal violates 2-deg arm soft limits")

    if args.plan_only:
        print("PLAN ONLY: no rt/lowcmd publisher was created.", flush=True)
        return 0

    # Strict diagnostic rule: do not continue when MotionSwitcher status is unknown.
    switcher = MotionSwitcherClient()
    switcher.SetTimeout(1.0)
    switcher.Init()
    status, result = switcher.CheckMode()
    if status != 0:
        raise RuntimeError(
            f"MotionSwitcher CheckMode failed with status={status}; refusing LowCmd smoothness test"
        )
    active_mode = str((result or {}).get("name", ""))
    if active_mode:
        raise RuntimeError(
            f"motion service {active_mode!r} is active; switch robot to debug/low-level mode first"
        )
    print("MotionSwitcher: no active motion service.", flush=True)

    guard = acquire_lowcmd_guard(Path(__file__).name, topic=args.command_topic)
    publisher = ChannelPublisher(args.command_topic, LowCmd_)
    publisher.Init()
    output = ArmLowCmdPublisher(publisher, unitree_hg_msg_dds__LowCmd_, crc, mode_machine)

    confirmation = input(
        "Clear the arm workspace and keep E-stop ready. Type RUNSMOOTH to execute: "
    ).strip()
    if confirmation != "RUNSMOOTH":
        print("Cancelled before publishing LowCmd.", flush=True)
        guard.close()
        return 0

    log_path = make_log_path(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "time_monotonic", "phase", "q_cmd", "qdot_cmd", "tau_ff_cmd",
        "q_actual", "dq_actual", "tau_est",
        "tracking_error_deg_max", "state_age_s",
        "cls_rate_scale",
        "cls_directed_lead_deg",
        "cls_v500_deg_s",
        "cls_v1000_deg_s",
        "cls_adaptive_ff_nm",
        "cls_dither_nm",
        "cls_dither_active",
    ])
    print(f"logging: {log_path}", flush=True)

    stop = False
    def request_stop(_sig=None, _frame=None):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    period = 1.0 / args.publish_frequency
    next_tick = time.monotonic()

    # Robust actual-speed watchdog.
    #
    # raw dq_actual contains isolated spikes on this robot, so the normal
    # soft speed limit is evaluated using q_actual finite differences,
    # synchronized to fresh LowState timestamps and low-pass filtered.
    #
    # A much higher raw-dq hard limit remains as an emergency backstop.
    speed_prev_q_actual = None
    speed_prev_state_time = None
    speed_ema_deg_s = np.zeros(14, dtype=float)
    speed_over_count = np.zeros(14, dtype=int)

    speed_ema_alpha = 0.20
    speed_required_samples = 5
    hard_raw_speed_limit_deg_s = 10.0

    selected_joint_i = (
        NAME_TO_INDEX[args.joint]
        if args.test == "joint"
        else None
    )

    def friction_magnitude_for_direction(direction: float) -> float:
        if direction > 0.0:
            return float(args.friction_ff_positive_nm)

        if direction < 0.0:
            return float(args.friction_ff_negative_nm)

        return 0.0

    def run_segment(
        phase: str,
        q0: np.ndarray,
        q1: np.ndarray,
        duration: float,
        settle_direction: float = 0.0,
    ):
        nonlocal next_tick, stop
        nonlocal speed_prev_q_actual
        nonlocal speed_prev_state_time
        nonlocal speed_ema_deg_s
        nonlocal speed_over_count

        started = time.monotonic()

        # ---------------------------------------------------------
        # Continuous low-speed controller V1.
        #
        # No binary RUN/PAUSE trajectory switching.
        #
        # The virtual trajectory rate is continuously reduced as
        # measured tracking lead grows.
        #
        # A 500 ms / 1 s measured-motion observer drives a small,
        # bounded adaptive friction assist.
        # ---------------------------------------------------------

        progress_s = 0.0
        last_loop_time = started

        # Continuous virtual-time rate scale.
        rate_scale = 1.0

        # The rate governor acts on the PREVIOUS measured lead.
        # One 500 Hz cycle of delay is negligible and avoids
        # rearranging the LowState read path.
        lead_for_rate_deg = 0.0

        # Rate-governor lead band.
        lead_rate_full_deg = 0.060
        lead_rate_zero_deg = 0.120

        # Adaptive low-speed assist.
        adaptive_ff_base_nm = 0.020
        adaptive_ff_max_nm = 0.050
        adaptive_ff_nm = adaptive_ff_base_nm

        adaptive_ff_up_nm_s = 0.005
        adaptive_ff_down_nm_s = 0.003
        adaptive_ff_release_nm_s = 0.050

        # Bounded low-frequency stiction dither.
        #
        # This does NOT increase the existing 0.050 Nm ceiling.
        dither_start_lead_deg = 0.105
        dither_max_rate_scale = 0.100
        dither_start_ff_nm = 0.045
        dither_max_abs_v1000_deg_s = 0.002

        dither_center_nm = 0.0475
        dither_amp_nm = 0.0025
        dither_freq_hz = 1.0

        dither_active = False
        dither_nm = 0.0

        # Motion observer thresholds.
        observer_stick_vel_deg_s = 0.002
        observer_moving_vel_deg_s = 0.006
        observer_release_vel_deg_s = 0.080

        # Only adapt once tracking lead is meaningful.
        adaptive_lead_on_deg = 0.080

        # Assist is smoothly gated in with tracking lead.
        assist_gate_off_deg = 0.020
        assist_gate_full_deg = 0.060

        # Measured-position history:
        # (LowState timestamp, selected-joint position in deg)
        observer_history = []
        observer_last_state_time = None

        observer_v500_deg_s = 0.0
        observer_v1000_deg_s = 0.0

        # Hard protection against sitting forever at the maximum
        # bounded assist with no measured motion.
        continuous_stall_started = None
        continuous_stall_timeout_s = 8.0

        # Console diagnostic rate.
        last_cls_print = started

        segment_direction = 0.0

        if (
            args.test == "joint"
            and selected_joint_i is not None
        ):
            segment_delta = float(
                q1[selected_joint_i]
                - q0[selected_joint_i]
            )

            if abs(segment_delta) > 1.0e-12:
                segment_direction = (
                    1.0
                    if segment_delta > 0.0
                    else -1.0
                )

        while not stop:
            now = time.monotonic()

            loop_dt = max(
                0.0,
                now - last_loop_time,
            )

            last_loop_time = now

            # Continuous virtual-time advancement.
            #
            # rate_scale is computed from the most recently
            # measured tracking lead.
            progress_s += loop_dt * rate_scale

            u = min(
                1.0,
                progress_s / max(duration, 1e-6),
            )

            alpha = minimum_jerk(u)

            q_cmd = q0 + alpha * (q1 - q0)

            # Analytic velocity of the same minimum-jerk trajectory.
            if duration > 1.0e-6 and u < 1.0:
                alpha_dot = (
                    minimum_jerk_derivative(u)
                    / duration
                )

                qdot_cmd = (
                    alpha_dot
                    * (q1 - q0)
                    * rate_scale
                )

            else:
                qdot_cmd = np.zeros(
                    14,
                    dtype=float,
                )

            # Read measured state BEFORE calculating settle assist.
            state_now = state_buffer.snapshot()

            if state_now is None:
                raise RuntimeError(
                    "LowState disappeared"
                )

            (
                q_actual,
                dq_actual,
                tau_est,
                motorstate_now,
                _,
                received_now,
            ) = state_now

            state_age = now - received_now

            if state_age > args.state_timeout:
                raise RuntimeError(
                    f"LowState stale for {state_age:.3f}s"
                )

            if np.any(motorstate_now != 0):
                bad = [
                    (
                        ARM_NAMES[i],
                        int(code),
                    )
                    for i, code
                    in enumerate(motorstate_now)
                    if code != 0
                ]

                raise RuntimeError(
                    f"motorstate error during motion: {bad}"
                )

            # -------------------------------------------------
            # Continuous rate governor + low-speed observer.
            # -------------------------------------------------

            directed_lead_deg = 0.0

            if (
                segment_direction != 0.0
                and selected_joint_i is not None
            ):
                joint_i = selected_joint_i

                directed_lead_deg = (
                    segment_direction
                    * math.degrees(
                        float(
                            q_cmd[joint_i]
                            - q_actual[joint_i]
                        )
                    )
                )

                lead_for_rate_deg = max(
                    0.0,
                    directed_lead_deg,
                )

                # ---------------------------------------------
                # Continuous virtual-time rate governor.
                #
                # <= 0.060 deg lead: full trajectory speed.
                # >= 0.120 deg lead: virtual trajectory stops.
                #
                # Smooth minimum-jerk interpolation avoids
                # RUN -> PAUSE -> RUN discontinuities.
                # ---------------------------------------------

                if (
                    lead_for_rate_deg
                    <= lead_rate_full_deg
                ):
                    rate_scale = 1.0

                elif (
                    lead_for_rate_deg
                    >= lead_rate_zero_deg
                ):
                    rate_scale = 0.0

                else:
                    x = (
                        (
                            lead_for_rate_deg
                            - lead_rate_full_deg
                        )
                        / (
                            lead_rate_zero_deg
                            - lead_rate_full_deg
                        )
                    )

                    x = float(
                        np.clip(
                            x,
                            0.0,
                            1.0,
                        )
                    )

                    rate_scale = (
                        1.0
                        - minimum_jerk(x)
                    )

                # ---------------------------------------------
                # Fresh-state low-speed motion observer.
                # ---------------------------------------------

                observer_dt = 0.0

                fresh_observer_state = (
                    observer_last_state_time is None
                    or received_now
                    > observer_last_state_time + 1.0e-9
                )

                if fresh_observer_state:

                    if observer_last_state_time is not None:
                        observer_dt = (
                            received_now
                            - observer_last_state_time
                        )

                    observer_last_state_time = received_now

                    q_actual_deg_selected = math.degrees(
                        float(
                            q_actual[joint_i]
                        )
                    )

                    observer_history.append(
                        (
                            received_now,
                            q_actual_deg_selected,
                        )
                    )

                    # Keep slightly more than 1 s of history.
                    cutoff = (
                        received_now
                        - 1.20
                    )

                    while (
                        len(observer_history) > 2
                        and observer_history[1][0]
                        < cutoff
                    ):
                        observer_history.pop(0)

                    def measured_velocity(window_s):
                        if len(observer_history) < 2:
                            return 0.0

                        t_now, q_now = (
                            observer_history[-1]
                        )

                        target_t = (
                            t_now
                            - window_s
                        )

                        candidate = None

                        for item in reversed(
                            observer_history[:-1]
                        ):
                            if item[0] <= target_t:
                                candidate = item
                                break

                        if candidate is None:
                            candidate = (
                                observer_history[0]
                            )

                        t_old, q_old = candidate

                        dt_obs = (
                            t_now
                            - t_old
                        )

                        # Do not trust a "1 s" velocity if only a
                        # tiny fraction of the requested window is
                        # available.
                        if (
                            dt_obs
                            < 0.70 * window_s
                        ):
                            return 0.0

                        return (
                            q_now
                            - q_old
                        ) / dt_obs

                    observer_v500_deg_s = (
                        measured_velocity(0.500)
                    )

                    observer_v1000_deg_s = (
                        measured_velocity(1.000)
                    )

                    directed_v500 = (
                        segment_direction
                        * observer_v500_deg_s
                    )

                    directed_v1000 = (
                        segment_direction
                        * observer_v1000_deg_s
                    )

                    # -----------------------------------------
                    # Bounded adaptive friction compensation.
                    #
                    # Never exceeds 0.05 Nm in V1.
                    # -----------------------------------------

                    if (
                        observer_dt > 0.0
                    ):
                        if (
                            directed_v500
                            >= observer_release_vel_deg_s
                        ):
                            # Faster decay if the joint suddenly
                            # begins to release.
                            adaptive_ff_nm -= (
                                adaptive_ff_release_nm_s
                                * observer_dt
                            )

                        elif (
                            directed_lead_deg
                            >= adaptive_lead_on_deg
                            and directed_v1000
                            <= observer_stick_vel_deg_s
                        ):
                            # Slowly increase while tracking lead
                            # exists but real motion has nearly
                            # disappeared.
                            adaptive_ff_nm += (
                                adaptive_ff_up_nm_s
                                * observer_dt
                            )

                        elif (
                            directed_v1000
                            >= observer_moving_vel_deg_s
                        ):
                            # Real low-speed motion is healthy:
                            # slowly return toward maintenance FF.
                            adaptive_ff_nm -= (
                                adaptive_ff_down_nm_s
                                * observer_dt
                            )

                        elif (
                            directed_lead_deg
                            < lead_rate_full_deg
                        ):
                            adaptive_ff_nm -= (
                                adaptive_ff_down_nm_s
                                * observer_dt
                            )

                        adaptive_ff_nm = float(
                            np.clip(
                                adaptive_ff_nm,
                                adaptive_ff_base_nm,
                                adaptive_ff_max_nm,
                            )
                        )

                    # -----------------------------------------
                    # Hard bounded-stall protection.
                    # -----------------------------------------

                    hard_stall = (
                        directed_lead_deg
                        >= 0.115
                        and adaptive_ff_nm
                        >= adaptive_ff_max_nm - 0.001
                        and directed_v1000
                        <= observer_stick_vel_deg_s
                    )

                    if hard_stall:
                        if continuous_stall_started is None:
                            continuous_stall_started = now

                        elif (
                            now
                            - continuous_stall_started
                            >= continuous_stall_timeout_s
                        ):
                            raise RuntimeError(
                                "continuous low-speed stall: "
                                f"lead={directed_lead_deg:+.4f} deg, "
                                f"v500={directed_v500:+.5f} deg/s, "
                                f"v1000={directed_v1000:+.5f} deg/s, "
                                f"rate_scale={rate_scale:.3f}, "
                                f"adaptive_ff={adaptive_ff_nm:.4f} Nm"
                            )

                    else:
                        continuous_stall_started = None

                # ---------------------------------------------
                # Bounded stiction dither.
                #
                # Do not add torque above the V1 ceiling.
                # Instead, once deeply inside the sticking
                # region, modulate the applied assist between
                # 0.045 and 0.050 Nm.
                # ---------------------------------------------

                dither_active = (
                    directed_lead_deg
                    >= dither_start_lead_deg
                    and rate_scale
                    <= dither_max_rate_scale
                    and adaptive_ff_nm
                    >= dither_start_ff_nm
                    and abs(
                        segment_direction
                        * observer_v1000_deg_s
                    )
                    <= dither_max_abs_v1000_deg_s
                )

                if dither_active:
                    dither_level_nm = (
                        dither_center_nm
                        + dither_amp_nm
                        * math.sin(
                            2.0
                            * math.pi
                            * dither_freq_hz
                            * (now - started)
                        )
                    )

                    # dither_nm is an offset from the current
                    # adaptive state. The final applied assist
                    # remains inside 0.045 ... 0.050 Nm.
                    dither_nm = (
                        dither_level_nm
                        - adaptive_ff_nm
                    )

                else:
                    dither_nm = 0.0

                # Periodic diagnostics.
                if (
                    now - last_cls_print
                    >= 2.0
                ):
                    print(
                        "CLS "
                        f"lead={directed_lead_deg:+.4f}deg "
                        f"rate={rate_scale:.3f} "
                        f"v500={segment_direction * observer_v500_deg_s:+.5f}deg/s "
                        f"v1s={segment_direction * observer_v1000_deg_s:+.5f}deg/s "
                        f"ff_state={adaptive_ff_nm:.4f}Nm "
                        f"dither={dither_nm:+.4f}Nm "
                        f"dither_on={int(dither_active)}",
                        flush=True,
                    )

                    last_cls_print = now

            else:
                rate_scale = 1.0
                dither_active = False
                dither_nm = 0.0

            # -------------------------------------------------
            # Direction-latched friction / stiction assist.
            # -------------------------------------------------

            tau_ff_cmd = np.zeros(
                14,
                dtype=float,
            )

            if args.test == "joint":
                joint_i = selected_joint_i

                v_scale = math.radians(
                    args.friction_velocity_scale_deg_s
                )

                joint_qdot = float(
                    qdot_cmd[joint_i]
                )

                # During normal trajectory motion use qdot_cmd.
                if abs(joint_qdot) > 1.0e-12:
                    magnitude = (
                        friction_magnitude_for_direction(
                            joint_qdot
                        )
                    )

                    tau_ff_cmd[joint_i] = (
                        magnitude
                        * math.tanh(
                            joint_qdot
                            / max(
                                v_scale,
                                1.0e-9,
                            )
                        )
                    )

                # During hold_goal / hold_end:
                #
                # Do NOT infer a new direction from noisy error.
                # Keep the motion direction latched from the
                # trajectory which led into this hold.
                elif (
                    args.stiction_settle_assist
                    and settle_direction != 0.0
                ):
                    error_deg = math.degrees(
                        float(
                            q_cmd[joint_i]
                            - q_actual[joint_i]
                        )
                    )

                    directed_error_deg = (
                        float(settle_direction)
                        * error_deg
                    )

                    # Only assist if measured q is still behind
                    # the target in the previously commanded
                    # direction.
                    if (
                        directed_error_deg
                        > args.stiction_error_off_deg
                    ):
                        denom = max(
                            (
                                args.stiction_error_on_deg
                                - args.stiction_error_off_deg
                            ),
                            1.0e-9,
                        )

                        factor = np.clip(
                            (
                                directed_error_deg
                                - args.stiction_error_off_deg
                            )
                            / denom,
                            0.0,
                            1.0,
                        )

                        magnitude = (
                            friction_magnitude_for_direction(
                                settle_direction
                            )
                        )

                        # Smoothly ramp settle assist from zero when
                        # entering hold_goal/hold_end.  Without this
                        # ramp, a large residual error could produce
                        # an immediate torque step.
                        ramp_u = np.clip(
                            (now - started)
                            / max(
                                args.stiction_assist_ramp_s,
                                1.0e-6,
                            ),
                            0.0,
                            1.0,
                        )

                        ramp_alpha = minimum_jerk(
                            float(ramp_u)
                        )

                        # Also smooth the error-band factor so the
                        # assist fades continuously near the target.
                        factor_smooth = minimum_jerk(
                            float(factor)
                        )

                        tau_ff_cmd[joint_i] = (
                            float(settle_direction)
                            * magnitude
                            * float(ramp_alpha)
                            * float(factor_smooth)
                        )

            # -------------------------------------------------
            # Continuous bounded adaptive assist.
            #
            # The existing command-line friction FF should be set
            # to zero for this experiment so this block is the
            # only low-speed compensation source.
            # -------------------------------------------------

            if (
                segment_direction != 0.0
                and selected_joint_i is not None
            ):
                joint_i = selected_joint_i

                # Smoothly introduce assist as tracking lead
                # develops.  This avoids a torque step at segment
                # start while still retaining assist when the
                # trajectory rate has been reduced toward zero.
                if (
                    directed_lead_deg
                    <= assist_gate_off_deg
                ):
                    assist_gate = 0.0

                elif (
                    directed_lead_deg
                    >= assist_gate_full_deg
                ):
                    assist_gate = 1.0

                else:
                    gate_x = (
                        (
                            directed_lead_deg
                            - assist_gate_off_deg
                        )
                        / (
                            assist_gate_full_deg
                            - assist_gate_off_deg
                        )
                    )

                    assist_gate = minimum_jerk(
                        float(
                            np.clip(
                                gate_x,
                                0.0,
                                1.0,
                            )
                        )
                    )

                assist_level_nm = float(
                    np.clip(
                        adaptive_ff_nm
                        + dither_nm,
                        adaptive_ff_base_nm,
                        adaptive_ff_max_nm,
                    )
                )

                assist_nm = (
                    segment_direction
                    * assist_level_nm
                    * assist_gate
                )

                tau_ff_cmd[joint_i] += assist_nm

                # Absolute diagnostic safety bound for V1.
                tau_ff_cmd[joint_i] = float(
                    np.clip(
                        tau_ff_cmd[joint_i],
                        -0.05,
                        +0.05,
                    )
                )

            tracking_error_deg = np.rad2deg(
                q_cmd - q_actual
            )

            max_tracking = float(
                np.max(
                    np.abs(
                        tracking_error_deg
                    )
                )
            )

            # -------------------------------------------------
            # Robust speed watchdog
            # -------------------------------------------------

            raw_dq_deg_s = np.rad2deg(dq_actual)
            raw_abs_deg_s = np.abs(raw_dq_deg_s)

            raw_peak_i = int(np.argmax(raw_abs_deg_s))
            raw_peak = float(raw_abs_deg_s[raw_peak_i])

            # Emergency raw-dq backstop.  This is intentionally much
            # higher than the normal filtered speed limit because raw
            # dq_actual shows isolated sensor spikes.
            if raw_peak > hard_raw_speed_limit_deg_s:
                raise RuntimeError(
                    f"HARD raw dq speed "
                    f"{raw_dq_deg_s[raw_peak_i]:+.2f} deg/s "
                    f"on {ARM_NAMES[raw_peak_i]} exceeds "
                    f"{hard_raw_speed_limit_deg_s:.2f} deg/s"
                )

            fresh_speed_sample = False

            if (
                speed_prev_state_time is None
                or received_now > speed_prev_state_time + 1.0e-9
            ):
                if speed_prev_q_actual is not None:
                    state_dt = (
                        received_now
                        - speed_prev_state_time
                    )

                    if 1.0e-4 <= state_dt <= 0.050:
                        q_speed_deg_s = np.rad2deg(
                            q_actual
                            - speed_prev_q_actual
                        ) / state_dt

                        speed_ema_deg_s = (
                            speed_ema_alpha
                            * q_speed_deg_s
                            + (1.0 - speed_ema_alpha)
                            * speed_ema_deg_s
                        )

                        fresh_speed_sample = True

                speed_prev_q_actual = q_actual.copy()
                speed_prev_state_time = received_now

            if fresh_speed_sample:
                over = (
                    np.abs(speed_ema_deg_s)
                    > args.max_actual_speed_deg_s
                )

                speed_over_count = np.where(
                    over,
                    speed_over_count + 1,
                    0,
                )

                bad = np.where(
                    speed_over_count
                    >= speed_required_samples
                )[0]

                if len(bad):
                    bad_i = int(
                        bad[
                            np.argmax(
                                np.abs(
                                    speed_ema_deg_s[bad]
                                )
                            )
                        ]
                    )

                    raise RuntimeError(
                        f"filtered actual joint speed "
                        f"{speed_ema_deg_s[bad_i]:+.2f} deg/s "
                        f"on {ARM_NAMES[bad_i]} exceeded "
                        f"{args.max_actual_speed_deg_s:.2f} deg/s "
                        f"for {speed_required_samples} fresh states; "
                        f"raw dq={raw_dq_deg_s[bad_i]:+.2f} deg/s"
                    )

            actual_speed_deg_s = float(
                np.max(
                    np.abs(
                        speed_ema_deg_s
                    )
                )
            )

            if (
                max_tracking
                > args.max_tracking_error_deg
            ):
                raise RuntimeError(
                    f"tracking error "
                    f"{max_tracking:.2f} deg "
                    f"exceeds limit "
                    f"{args.max_tracking_error_deg:.2f} deg"
                )

            output.write(
                q_cmd,
                tau_ff_cmd=tau_ff_cmd,
            )

            writer.writerow([
                f"{now:.9f}",
                phase,
                json.dumps(
                    q_cmd.tolist(),
                    separators=(",", ":"),
                ),
                json.dumps(
                    qdot_cmd.tolist(),
                    separators=(",", ":"),
                ),
                json.dumps(
                    tau_ff_cmd.tolist(),
                    separators=(",", ":"),
                ),
                json.dumps(
                    q_actual.tolist(),
                    separators=(",", ":"),
                ),
                json.dumps(
                    dq_actual.tolist(),
                    separators=(",", ":"),
                ),
                json.dumps(
                    tau_est.tolist(),
                    separators=(",", ":"),
                ),
                f"{max_tracking:.6f}",
                f"{state_age:.6f}",
                f"{rate_scale:.6f}",
                f"{directed_lead_deg:.6f}",
                f"{observer_v500_deg_s:.6f}",
                f"{observer_v1000_deg_s:.6f}",
                f"{adaptive_ff_nm:.6f}",
                f"{dither_nm:.6f}",
                int(dither_active),
            ])

            if u >= 1.0:
                break

            next_tick += period

            wait = (
                next_tick
                - time.monotonic()
            )

            if wait > 0.0:
                time.sleep(wait)

            else:
                next_tick = time.monotonic()

    def hold(
        phase: str,
        q_cmd: np.ndarray,
        duration: float,
        settle_direction: float = 0.0,
    ):
        run_segment(
            phase,
            q_cmd,
            q_cmd,
            duration,
            settle_direction=settle_direction,
        )

    try:
        # ==========================================================
        # R_ELBOW +0.5 deg / 30 s rate-matched probe.
        #
        # NO Cartesian motion.
        # NO friction FF.
        # One LowCmd publisher.
        # ==========================================================

        elbow_i = NAME_TO_INDEX["R_ELBOW"]

        q_ref = q_start.copy()
        q_probe = q_ref.copy()

        q_probe[elbow_i] += math.radians(0.5)

        soft_low = (
            ARM_JOINT_LOWER_Q
            + LIMIT_MARGIN_RAD
        )

        soft_high = (
            ARM_JOINT_UPPER_Q
            - LIMIT_MARGIN_RAD
        )

        if (
            np.any(q_probe < soft_low)
            or np.any(q_probe > soft_high)
        ):
            raise RuntimeError(
                "R_ELBOW +0.5 deg probe violates "
                "arm soft limits"
            )

        print(
            "ELBOW CONTINUOUS LOW-SPEED V2 DITHER ONLY",
            flush=True,
        )

        print(
            f"start R_ELBOW = "
            f"{math.degrees(q_ref[elbow_i]):+.4f} deg",
            flush=True,
        )

        print(
            f"target R_ELBOW = "
            f"{math.degrees(q_probe[elbow_i]):+.4f} deg",
            flush=True,
        )

        hold(
            "governor_ref_hold",
            q_ref,
            1.0,
        )

        if not stop:
            run_segment(
                "rate_matched_plus0p5",
                q_ref,
                q_probe,
                30.0,
            )

        if not stop:
            hold(
                "governor_plus2_hold",
                q_probe,
                3.0,
            )

        if not stop:
            st = state_buffer.snapshot()

            if st is None:
                raise RuntimeError(
                    "LowState unavailable after probe"
                )

            qa = st[0]

            e_selected = math.degrees(
                q_probe[elbow_i]
                - qa[elbow_i]
            )

            e_all = float(
                np.max(
                    np.abs(
                        np.rad2deg(
                            q_probe - qa
                        )
                    )
                )
            )

            print(
                f"FINAL PROBE CHECK: "
                f"R_ELBOW error="
                f"{e_selected:+.4f} deg, "
                f"all-arm max="
                f"{e_all:.4f} deg",
                flush=True,
            )

            print(
                "FINAL q_actual rad:",
                np.round(
                    qa,
                    6,
                ).tolist(),
                flush=True,
            )

            if abs(e_selected) > 0.30:
                raise RuntimeError(
                    "R_ELBOW probe residual "
                    "exceeds 0.30 deg"
                )

        print(
            "LEAD-GOVERNOR PROBE FINISHED.",
            flush=True,
        )

        print(
            "No Cartesian motion was executed.",
            flush=True,
        )
    except Exception as exc:
        print(f"ABORT: {type(exc).__name__}: {exc}", flush=True)
        # Best-effort short hold at the latest measured posture.
        state_now = state_buffer.snapshot()
        if state_now is not None:
            q_hold = state_now[0].copy()
            deadline = time.monotonic() + 0.30
            while time.monotonic() < deadline:
                output.write(q_hold, tau_ff_cmd=np.zeros(14, dtype=float))
                time.sleep(period)
        raise
    finally:
        csv_file.flush()
        csv_file.close()
        guard.close()
        print(f"CSV saved: {log_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
