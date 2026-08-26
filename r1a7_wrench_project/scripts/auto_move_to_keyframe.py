#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


# ============================================================
# Project path
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ============================================================
# Unitree SDK
# ============================================================

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


# ============================================================
# R1-A7 joint mapping
# ============================================================

ARM_NAMES = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",

    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


LEFT_ARM_INDICES = [
    15, 16, 17, 18, 19, 20, 21,
]

RIGHT_ARM_INDICES = [
    22, 23, 24, 25, 26, 27, 28,
]

ARM_INDICES = (
    LEFT_ARM_INDICES
    + RIGHT_ARM_INDICES
)


# ============================================================
# Local indices in 14-DoF arm vector
# ============================================================

LEFT_SHOULDER = [0, 1, 2]
LEFT_ELBOW = [3]
LEFT_WRIST = [4, 5, 6]

RIGHT_SHOULDER = [7, 8, 9]
RIGHT_ELBOW = [10]
RIGHT_WRIST = [11, 12, 13]


WRIST_LOCAL_INDICES = (
    LEFT_WRIST
    + RIGHT_WRIST
)

ARM_CORE_LOCAL_INDICES = (
    LEFT_SHOULDER
    + LEFT_ELBOW
    + RIGHT_SHOULDER
    + RIGHT_ELBOW
)


# ============================================================
# Other controlled motors
# ============================================================

WAIST_INDEX = 13

AUX_HOLD_INDICES = [
    29,
    30,
]

GRIPPER_INDICES = [
    31,
    33,
]


# ============================================================
# R1-A7 arm limits
# ============================================================

R1A7_ARM_LIMITS = np.array(
    [
        # Left arm
        [-3.1416, 2.0944],
        [-0.2269, 2.4784],
        [-1.9199, 1.9199],
        [-0.9757, 2.1850],
        [-1.9199, 1.9199],
        [-1.61429558, 1.61429558],
        [-1.61429558, 1.61429558],

        # Right arm
        [-3.1416, 2.0944],
        [-2.4784, 0.2269],
        [-1.9199, 1.9199],
        [-0.9757, 2.1850],
        [-1.9199, 1.9199],
        [-1.61429558, 1.61429558],
        [-1.61429558, 1.61429558],
    ],
    dtype=float,
)


# ============================================================
# Recovery segment
# ============================================================

@dataclass
class RecoverySegment:
    name: str
    target: np.ndarray
    hold_time: float


# ============================================================
# Utility
# ============================================================

def load_yaml(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def load_keyframe(
    path: Path,
    name: str,
) -> np.ndarray:

    data = load_yaml(path)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Invalid YAML: {path}"
        )

    if "keyframes" not in data:
        raise RuntimeError(
            "Missing keyframes section"
        )

    if name not in data["keyframes"]:
        raise RuntimeError(
            f"Keyframe '{name}' does not exist"
        )

    kf = data["keyframes"][name]

    q = np.asarray(
        kf["left_joint_position"]
        + kf["right_joint_position"],
        dtype=float,
    )

    if q.shape != (14,):
        raise RuntimeError(
            "Keyframe must contain 14 arm joints"
        )

    if not np.all(np.isfinite(q)):
        raise RuntimeError(
            "Keyframe contains non-finite values"
        )

    return q


def quintic(u):
    return (
        10.0 * u**3
        - 15.0 * u**4
        + 6.0 * u**5
    )


# ============================================================
# Main controller
# ============================================================

class AutoKeyframeMover:

    def __init__(
        self,
        args: argparse.Namespace,
    ):

        self.args = args

        self.crc = CRC()

        self.low_cmd = (
            unitree_hg_msg_dds__LowCmd_()
        )

        self.low_state: Optional[
            LowState_
        ] = None

        self.last_lowstate_time: Optional[
            float
        ] = None

        self.lowstate_count = 0

        self.subscriber = None
        self.publisher = None

        self.waist_hold_q: Optional[
            float
        ] = None

        self.gripper_hold_q: Optional[
            np.ndarray
        ] = None

        self.commands_published = 0

    # ========================================================
    # DDS
    # ========================================================

    def lowstate_handler(
        self,
        msg: LowState_,
    ) -> None:

        self.low_state = msg

        self.last_lowstate_time = (
            time.monotonic()
        )

        self.lowstate_count += 1

    def connect_state_only(
        self,
    ) -> None:

        ChannelFactoryInitialize(
            self.args.domain_id,
            self.args.interface,
        )

        self.subscriber = (
            ChannelSubscriber(
                self.args.state_topic,
                LowState_,
            )
        )

        self.subscriber.Init(
            self.lowstate_handler,
            10,
        )

    def create_publisher(
        self,
    ) -> None:

        if self.publisher is not None:
            return

        self.publisher = (
            ChannelPublisher(
                self.args.command_topic,
                LowCmd_,
            )
        )

        self.publisher.Init()

    def state_age(
        self,
    ) -> float:

        if self.last_lowstate_time is None:
            return float("inf")

        return (
            time.monotonic()
            - self.last_lowstate_time
        )

    def wait_for_state(
        self,
        timeout: float = 3.0,
    ) -> None:

        deadline = (
            time.monotonic()
            + timeout
        )

        while (
            time.monotonic()
            < deadline
        ):

            if (
                self.low_state is not None
                and
                self.state_age()
                <= self.args.state_timeout
            ):
                return

            time.sleep(
                0.02
            )

        raise RuntimeError(
            "No fresh rt/lowstate"
        )

    def arm_qdq(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.low_state is None:
            raise RuntimeError(
                "No lowstate"
            )

        ms = (
            self.low_state.motor_state
        )

        q = np.asarray(
            [
                float(ms[i].q)
                for i in ARM_INDICES
            ],
            dtype=float,
        )

        dq = np.asarray(
            [
                float(ms[i].dq)
                for i in ARM_INDICES
            ],
            dtype=float,
        )

        return q, dq

    def arm_tau_est(
        self,
    ) -> np.ndarray:

        if self.low_state is None:
            raise RuntimeError(
                "No lowstate"
            )

        ms = (
            self.low_state.motor_state
        )

        values = []

        for idx in ARM_INDICES:

            motor = ms[idx]

            try:
                values.append(
                    float(motor.tau_est)
                )

            except Exception:
                values.append(
                    0.0
                )

        return np.asarray(
            values,
            dtype=float,
        )

    # ========================================================
    # MotionSwitcher
    # ========================================================

    def check_motion_switcher(
        self,
    ):

        try:

            msc = (
                MotionSwitcherClient()
            )

            msc.SetTimeout(
                2.0
            )

            msc.Init()

            status, result = (
                msc.CheckMode()
            )

        except Exception as exc:

            return (
                False,
                -1,
                repr(exc),
            )

        active_name = ""

        if isinstance(
            result,
            dict,
        ):
            active_name = str(
                result.get(
                    "name",
                    "",
                )
            ).strip()

        ok = (
            status == 0
            and active_name == ""
        )

        return (
            ok,
            status,
            result,
        )

    # ========================================================
    # Joint limits
    # ========================================================

    def check_joint_limits(
        self,
        q: np.ndarray,
    ):

        margin = float(
            self.args.joint_limit_margin
        )

        low = (
            R1A7_ARM_LIMITS[:, 0]
            + margin
        )

        high = (
            R1A7_ARM_LIMITS[:, 1]
            - margin
        )

        mask = np.logical_and(
            q >= low,
            q <= high,
        )

        return (
            bool(np.all(mask)),
            low,
            high,
            mask,
        )

    # ========================================================
    # Gain mapping
    # ========================================================

    def arm_gain(
        self,
        joint_i: int,
    ):

        local_i = (
            joint_i % 7
        )

        # Shoulder pitch
        if local_i == 0:

            return (
                self.args.kp_shoulder_pitch,
                self.args.kd_shoulder_pitch,
            )

        # Shoulder roll/yaw + elbow
        if local_i < 4:

            return (
                self.args.kp_low,
                self.args.kd_low,
            )

        # Wrist pitch
        if local_i == 5:

            return (
                self.args.kp_wrist_pitch,
                self.args.kd_wrist_pitch,
            )

        # Wrist roll/yaw
        return (
            self.args.kp_wrist,
            self.args.kd_wrist,
        )

    # ========================================================
    # AUTO duration
    # ========================================================

    def calculate_safe_duration(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        min_duration: Optional[float] = None,
    ):

        delta = np.abs(
            q1 - q0
        )

        max_delta = float(
            np.max(delta)
        )

        t_velocity = (
            1.875
            * max_delta
            / self.args.max_velocity
        )

        t_acceleration = float(
            np.sqrt(
                5.7735026919
                * max_delta
                / self.args.max_acceleration
            )
        )

        theoretical = max(
            t_velocity,
            t_acceleration,
        )

        required = (
            theoretical
            * self.args.duration_margin
        )

        if min_duration is None:
            min_duration = (
                self.args.min_auto_duration
            )

        duration = max(
            float(min_duration),
            required,
        )

        step = float(
            self.args.duration_step
        )

        duration = float(
            np.ceil(
                duration / step
            )
            * step
        )

        if (
            duration
            > self.args.max_auto_duration
        ):

            raise RuntimeError(
                "AUTO duration exceeds "
                f"{self.args.max_auto_duration:.1f}s"
            )

        return duration

    def trajectory_metrics(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        duration: float,
    ):

        delta = np.abs(
            q1 - q0
        )

        T = float(
            duration
        )

        peak_velocity = float(
            np.max(
                1.875
                * delta
                / T
            )
        )

        peak_acceleration = float(
            np.max(
                5.7735026919
                * delta
                / T**2
            )
        )

        n = max(
            2,
            int(
                np.ceil(
                    T
                    * self.args.hz
                )
            )
            + 1,
        )

        t = np.linspace(
            0.0,
            T,
            n,
        )

        h = quintic(
            t / T
        )

        trajectory = (
            q0[None, :]
            + h[:, None]
            * (
                q1 - q0
            )[None, :]
        )

        max_step = float(
            np.max(
                np.abs(
                    np.diff(
                        trajectory,
                        axis=0,
                    )
                )
            )
        )

        return {
            "max_delta": float(
                np.max(delta)
            ),
            "peak_velocity": (
                peak_velocity
            ),
            "peak_acceleration": (
                peak_acceleration
            ),
            "max_step": max_step,
            "samples": n,
        }

    # ========================================================
    # Recovery planner
    # ========================================================

    def interpolate_group_segments(
        self,
        start_q: np.ndarray,
        final_q: np.ndarray,
        indices: list[int],
        name_prefix: str,
    ) -> list[RecoverySegment]:

        delta = (
            final_q[indices]
            - start_q[indices]
        )

        max_delta = float(
            np.max(
                np.abs(delta)
            )
        )

        if (
            max_delta
            <= self.args.recovery_stage_delta
        ):

            steps = 1

        else:

            steps = int(
                np.ceil(
                    max_delta
                    / self.args.recovery_stage_delta
                )
            )

        segments = []

        base = (
            start_q.copy()
        )

        for step_i in range(
            1,
            steps + 1,
        ):

            alpha = (
                step_i
                / steps
            )

            target = (
                base.copy()
            )

            target[indices] = (
                start_q[indices]
                + alpha
                * (
                    final_q[indices]
                    - start_q[indices]
                )
            )

            segments.append(
                RecoverySegment(
                    name=(
                        f"{name_prefix}_"
                        f"{step_i:02d}_of_{steps:02d}"
                    ),
                    target=target,
                    hold_time=(
                        self.args.recovery_hold_time
                    ),
                )
            )

        return segments

    def build_recovery_plan(
        self,
        q_current: np.ndarray,
        q_home: np.ndarray,
    ) -> list[RecoverySegment]:

        direct_delta = float(
            np.max(
                np.abs(
                    q_home
                    - q_current
                )
            )
        )

        # ----------------------------------------------------
        # Direct HOME
        # ----------------------------------------------------

        if (
            direct_delta
            <= self.args.direct_home_delta
        ):

            return [
                RecoverySegment(
                    name="FINAL_HOME",
                    target=q_home.copy(),
                    hold_time=(
                        self.args.hold_time
                    ),
                )
            ]

        plan = []

        working_q = (
            q_current.copy()
        )

        # ----------------------------------------------------
        # Stage A:
        # Wrist recovery first
        # ----------------------------------------------------

        wrist_target = (
            working_q.copy()
        )

        wrist_target[
            WRIST_LOCAL_INDICES
        ] = (
            q_home[
                WRIST_LOCAL_INDICES
            ]
        )

        wrist_segments = (
            self.interpolate_group_segments(
                working_q,
                wrist_target,
                WRIST_LOCAL_INDICES,
                "WRIST_RECOVERY",
            )
        )

        plan.extend(
            wrist_segments
        )

        if wrist_segments:

            working_q = (
                wrist_segments[-1]
                .target.copy()
            )

        # ----------------------------------------------------
        # Stage B:
        # Shoulder + elbow recovery
        # ----------------------------------------------------

        arm_target = (
            working_q.copy()
        )

        arm_target[
            ARM_CORE_LOCAL_INDICES
        ] = (
            q_home[
                ARM_CORE_LOCAL_INDICES
            ]
        )

        arm_segments = (
            self.interpolate_group_segments(
                working_q,
                arm_target,
                ARM_CORE_LOCAL_INDICES,
                "ARM_RECOVERY",
            )
        )

        plan.extend(
            arm_segments
        )

        if arm_segments:

            working_q = (
                arm_segments[-1]
                .target.copy()
            )

        # ----------------------------------------------------
        # Final exact HOME
        # ----------------------------------------------------

        plan.append(
            RecoverySegment(
                name="FINAL_HOME",
                target=q_home.copy(),
                hold_time=(
                    self.args.hold_time
                ),
            )
        )

        return plan

    def print_recovery_plan(
        self,
        q_current: np.ndarray,
        q_home: np.ndarray,
        plan: list[RecoverySegment],
    ):

        direct_delta = float(
            np.max(
                np.abs(
                    q_home - q_current
                )
            )
        )

        print()
        print(
            "=" * 100
        )

        print(
            "R1-A7 RECOVERY PLAN"
        )

        print(
            "=" * 100
        )

        print(
            f"Current -> HOME max delta : "
            f"{direct_delta:.6f} rad"
        )

        print(
            f"Direct HOME threshold     : "
            f"{self.args.direct_home_delta:.6f} rad"
        )

        if (
            direct_delta
            <= self.args.direct_home_delta
        ):

            print(
                "Recovery mode             : "
                "DIRECT HOME"
            )

        else:

            print(
                "Recovery mode             : "
                "SEGMENTED RECOVERY"
            )

        print(
            f"Recovery stage max delta  : "
            f"{self.args.recovery_stage_delta:.6f} rad"
        )

        print(
            f"Total segments            : "
            f"{len(plan)}"
        )

        print()

        previous = (
            q_current.copy()
        )

        for i, segment in enumerate(
            plan,
            start=1,
        ):

            delta = float(
                np.max(
                    np.abs(
                        segment.target
                        - previous
                    )
                )
            )

            duration = (
                self.calculate_safe_duration(
                    previous,
                    segment.target,
                    min_duration=(
                        self.args.recovery_min_duration
                        if segment.name
                        != "FINAL_HOME"
                        else
                        self.args.min_auto_duration
                    ),
                )
            )

            print(
                f"{i:02d}. "
                f"{segment.name:28s} "
                f"max_delta={delta:.6f} rad "
                f"duration={duration:.2f}s"
            )

            previous = (
                segment.target.copy()
            )

        print(
            "=" * 100
        )

    # ========================================================
    # General preflight
    # ========================================================

    def preflight(
        self,
        q_home: np.ndarray,
        verbose: bool = True,
    ) -> tuple[
        bool,
        list[RecoverySegment],
    ]:

        self.wait_for_state(
            3.0
        )

        q_current, dq_current = (
            self.arm_qdq()
        )

        state_age = (
            self.state_age()
        )

        state_ok = (
            state_age
            <= self.args.state_timeout
        )

        max_dq = float(
            np.max(
                np.abs(
                    dq_current
                )
            )
        )

        velocity_ok = (
            max_dq
            <= self.args.max_start_dq
        )

        (
            current_limit_ok,
            _,
            _,
            _,
        ) = self.check_joint_limits(
            q_current
        )

        (
            target_limit_ok,
            _,
            _,
            _,
        ) = self.check_joint_limits(
            q_home
        )

        (
            motion_ok,
            motion_status,
            motion_result,
        ) = self.check_motion_switcher()

        plan = (
            self.build_recovery_plan(
                q_current,
                q_home,
            )
        )

        plan_ok = True

        previous = (
            q_current.copy()
        )

        for segment in plan:

            (
                limit_ok,
                _,
                _,
                _,
            ) = self.check_joint_limits(
                segment.target
            )

            if not limit_ok:

                plan_ok = False
                break

            duration = (
                self.calculate_safe_duration(
                    previous,
                    segment.target,
                    min_duration=(
                        self.args.recovery_min_duration
                        if segment.name
                        != "FINAL_HOME"
                        else
                        self.args.min_auto_duration
                    ),
                )
            )

            metrics = (
                self.trajectory_metrics(
                    previous,
                    segment.target,
                    duration,
                )
            )

            if (
                metrics["peak_velocity"]
                > self.args.max_velocity
            ):
                plan_ok = False

            if (
                metrics["peak_acceleration"]
                > self.args.max_acceleration
            ):
                plan_ok = False

            if (
                metrics["max_step"]
                > self.args.max_step
            ):
                plan_ok = False

            previous = (
                segment.target.copy()
            )

        passed = all(
            [
                state_ok,
                velocity_ok,
                current_limit_ok,
                target_limit_ok,
                motion_ok,
                plan_ok,
            ]
        )

        if verbose:

            print()
            print(
                "=" * 100
            )

            print(
                "R1-A7 SEGMENTED AUTO-HOME PRE-FLIGHT"
            )

            print(
                "=" * 100
            )

            print(
                f"Lowstate age        : "
                f"{state_age:.6f}s "
                f"[{'PASS' if state_ok else 'FAIL'}]"
            )

            print(
                f"Max initial |dq|    : "
                f"{max_dq:.6f} rad/s "
                f"[{'PASS' if velocity_ok else 'FAIL'}]"
            )

            print(
                f"Current limits      : "
                f"{'PASS' if current_limit_ok else 'FAIL'}"
            )

            print(
                f"HOME limits         : "
                f"{'PASS' if target_limit_ok else 'FAIL'}"
            )

            print(
                f"MotionSwitcher      : "
                f"{'PASS' if motion_ok else 'FAIL'}"
            )

            print(
                f"  status            : "
                f"{motion_status}"
            )

            print(
                f"  result            : "
                f"{motion_result}"
            )

            print(
                f"Recovery plan       : "
                f"{'PASS' if plan_ok else 'FAIL'}"
            )

            print()

            print(
                "Arm gains:"
            )

            print(
                "  shoulder pitch    : "
                f"{self.args.kp_shoulder_pitch}/"
                f"{self.args.kd_shoulder_pitch}"
            )

            print(
                "  shoulder/elbow    : "
                f"{self.args.kp_low}/"
                f"{self.args.kd_low}"
            )

            print(
                "  wrist pitch       : "
                f"{self.args.kp_wrist_pitch}/"
                f"{self.args.kd_wrist_pitch}"
            )

            print(
                "  wrist roll/yaw    : "
                f"{self.args.kp_wrist}/"
                f"{self.args.kd_wrist}"
            )

            self.print_recovery_plan(
                q_current,
                q_home,
                plan,
            )

            print()

            print(
                "PRE_FLIGHT = "
                + (
                    "PASS"
                    if passed
                    else "FAIL"
                )
            )

            print(
                "ROBOT_COMMAND_PUBLISHED = FALSE"
            )

            print(
                "=" * 100
            )

        return (
            passed,
            plan,
        )

    # ========================================================
    # LowCmd
    # ========================================================

    def init_lowcmd_zero(
        self,
    ):

        for motor in (
            self.low_cmd.motor_cmd
        ):

            motor.tau = 0.0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    def capture_hold_positions(
        self,
    ):

        self.wait_for_state(
            2.0
        )

        ms = (
            self.low_state.motor_state
        )

        self.waist_hold_q = float(
            ms[WAIST_INDEX].q
        )

        self.gripper_hold_q = (
            np.asarray(
                [
                    float(
                        ms[
                            GRIPPER_INDICES[0]
                        ].q
                    ),
                    float(
                        ms[
                            GRIPPER_INDICES[1]
                        ].q
                    ),
                ],
                dtype=float,
            )
        )

    def publish_arm(
        self,
        q_desired: np.ndarray,
    ):

        if self.publisher is None:
            raise RuntimeError(
                "Publisher not initialized"
            )

        if (
            self.state_age()
            > self.args.state_timeout
        ):

            raise RuntimeError(
                "LOWSTATE WATCHDOG TIMEOUT"
            )

        (
            limits_ok,
            _,
            _,
            _,
        ) = self.check_joint_limits(
            q_desired
        )

        if not limits_ok:

            raise RuntimeError(
                "Command outside arm limits"
            )

        self.init_lowcmd_zero()

        if hasattr(
            self.low_cmd,
            "mode_pr",
        ):
            self.low_cmd.mode_pr = 0

        if (
            hasattr(
                self.low_cmd,
                "mode_machine",
            )
            and
            hasattr(
                self.low_state,
                "mode_machine",
            )
        ):

            self.low_cmd.mode_machine = (
                self.low_state.mode_machine
            )

        ms = (
            self.low_state.motor_state
        )

        # ----------------------------------------------------
        # Waist hold
        # ----------------------------------------------------

        motor = (
            self.low_cmd.motor_cmd[
                WAIST_INDEX
            ]
        )

        motor.mode = 1
        motor.tau = 0.0
        motor.q = float(
            self.waist_hold_q
        )
        motor.dq = 0.0
        motor.kp = float(
            self.args.hold_kp
        )
        motor.kd = float(
            self.args.hold_kd
        )

        # ----------------------------------------------------
        # Auxiliary motors 29 / 30
        # ----------------------------------------------------

        for idx in AUX_HOLD_INDICES:

            if (
                idx >= len(ms)
                or
                idx >= len(
                    self.low_cmd.motor_cmd
                )
            ):
                continue

            motor = (
                self.low_cmd.motor_cmd[
                    idx
                ]
            )

            motor.mode = 1
            motor.tau = 0.0
            motor.q = float(
                ms[idx].q
            )
            motor.dq = 0.0
            motor.kp = float(
                self.args.hold_kp
            )
            motor.kd = float(
                self.args.hold_kd
            )

        # ----------------------------------------------------
        # 14 arm joints
        # ----------------------------------------------------

        for joint_i, (
            idx,
            q,
        ) in enumerate(
            zip(
                ARM_INDICES,
                q_desired,
            )
        ):

            motor = (
                self.low_cmd.motor_cmd[
                    idx
                ]
            )

            kp, kd = (
                self.arm_gain(
                    joint_i
                )
            )

            motor.mode = 1
            motor.tau = 0.0
            motor.q = float(q)
            motor.dq = 0.0
            motor.kp = float(kp)
            motor.kd = float(kd)

        # ----------------------------------------------------
        # Grippers
        # ----------------------------------------------------

        for i, idx in enumerate(
            GRIPPER_INDICES
        ):

            motor = (
                self.low_cmd.motor_cmd[
                    idx
                ]
            )

            motor.mode = 1
            motor.tau = 0.0
            motor.q = float(
                self.gripper_hold_q[i]
            )
            motor.dq = 0.0
            motor.kp = float(
                self.args.gripper_kp
            )
            motor.kd = float(
                self.args.gripper_kd
            )

        self.low_cmd.crc = (
            self.crc.Crc(
                self.low_cmd
            )
        )

        self.publisher.Write(
            self.low_cmd
        )

        self.commands_published += 1

    def release(
        self,
    ):

        if self.publisher is None:
            return

        try:

            self.init_lowcmd_zero()

            if hasattr(
                self.low_cmd,
                "mode_pr",
            ):
                self.low_cmd.mode_pr = 0

            if (
                self.low_state is not None
                and
                hasattr(
                    self.low_cmd,
                    "mode_machine",
                )
            ):
                self.low_cmd.mode_machine = (
                    self.low_state.mode_machine
                )

            self.low_cmd.crc = (
                self.crc.Crc(
                    self.low_cmd
                )
            )

            self.publisher.Write(
                self.low_cmd
            )

        except Exception as exc:

            print(
                "Release warning:",
                repr(exc),
            )

    # ========================================================
    # Takeover test
    # ========================================================

    def takeover_test(
        self,
    ):

        q_takeover, _ = (
            self.arm_qdq()
        )

        start = (
            time.monotonic()
        )

        next_tick = start

        max_drift = 0.0

        print()
        print(
            "=" * 100
        )

        print(
            "LOWCMD ZERO-DISPLACEMENT TAKEOVER TEST"
        )

        print(
            "=" * 100
        )

        while (
            time.monotonic()
            - start
            < self.args.takeover_time
        ):

            q, dq = (
                self.arm_qdq()
            )

            drift = float(
                np.max(
                    np.abs(
                        q
                        - q_takeover
                    )
                )
            )

            max_drift = max(
                max_drift,
                drift,
            )

            if (
                max_drift
                > self.args.max_takeover_drift
            ):

                raise RuntimeError(
                    "Takeover drift too large"
                )

            self.publish_arm(
                q_takeover
            )

            next_tick += (
                1.0 / self.args.hz
            )

            dt = (
                next_tick
                - time.monotonic()
            )

            if dt > 0:
                time.sleep(dt)

        q_after, dq_after = (
            self.arm_qdq()
        )

        print(
            f"Takeover max drift : "
            f"{max_drift:.6f} rad"
        )

        print(
            f"Final max |dq|     : "
            f"{np.max(np.abs(dq_after)):.6f}"
        )

        print(
            "TAKEOVER_TEST = PASS"
        )

        print(
            "=" * 100
        )

        return (
            q_after.copy()
        )

    # ========================================================
    # Diagnostic helpers
    # ========================================================

    def calculate_segment_diagnostics(
        self,
        q_target: np.ndarray,
        q_measured: np.ndarray,
    ):

        error_vector = (
            q_measured
            - q_target
        )

        abs_error = (
            np.abs(
                error_vector
            )
        )

        # ----------------------------------------------------
        # Global worst joint
        # ----------------------------------------------------

        worst_i = int(
            np.argmax(
                abs_error
            )
        )

        global_max_error = float(
            abs_error[
                worst_i
            ]
        )

        global_signed_error = float(
            error_vector[
                worst_i
            ]
        )

        # ----------------------------------------------------
        # Wrist group
        # ----------------------------------------------------

        wrist_abs = (
            abs_error[
                WRIST_LOCAL_INDICES
            ]
        )

        wrist_local_argmax = int(
            np.argmax(
                wrist_abs
            )
        )

        wrist_worst_i = (
            WRIST_LOCAL_INDICES[
                wrist_local_argmax
            ]
        )

        wrist_max_error = float(
            abs_error[
                wrist_worst_i
            ]
        )

        wrist_signed_error = float(
            error_vector[
                wrist_worst_i
            ]
        )

        # ----------------------------------------------------
        # Shoulder + elbow group
        # ----------------------------------------------------

        arm_abs = (
            abs_error[
                ARM_CORE_LOCAL_INDICES
            ]
        )

        arm_local_argmax = int(
            np.argmax(
                arm_abs
            )
        )

        arm_worst_i = (
            ARM_CORE_LOCAL_INDICES[
                arm_local_argmax
            ]
        )

        arm_max_error = float(
            abs_error[
                arm_worst_i
            ]
        )

        arm_signed_error = float(
            error_vector[
                arm_worst_i
            ]
        )

        return {
            "error_vector": error_vector,
            "abs_error": abs_error,

            "global_worst_i": worst_i,
            "global_max_error": (
                global_max_error
            ),
            "global_signed_error": (
                global_signed_error
            ),

            "wrist_worst_i": wrist_worst_i,
            "wrist_max_error": (
                wrist_max_error
            ),
            "wrist_signed_error": (
                wrist_signed_error
            ),

            "arm_worst_i": arm_worst_i,
            "arm_max_error": (
                arm_max_error
            ),
            "arm_signed_error": (
                arm_signed_error
            ),
        }

    def print_segment_diagnostics(
        self,
        diagnostics,
    ):

        global_i = (
            diagnostics[
                "global_worst_i"
            ]
        )

        wrist_i = (
            diagnostics[
                "wrist_worst_i"
            ]
        )

        arm_i = (
            diagnostics[
                "arm_worst_i"
            ]
        )

        print()
        print(
            "Segment diagnostic:"
        )

        print(
            f"  Worst joint      : "
            f"{ARM_NAMES[global_i]}"
        )

        print(
            f"  Worst |error|    : "
            f"{diagnostics['global_max_error']:.6f} rad"
        )

        print(
            f"  Worst signed err : "
            f"{diagnostics['global_signed_error']:+.6f} rad"
        )

        print(
            f"  Wrist max error  : "
            f"{diagnostics['wrist_max_error']:.6f} rad "
            f"({ARM_NAMES[wrist_i]}, "
            f"{diagnostics['wrist_signed_error']:+.6f})"
        )

        print(
            f"  Arm max error    : "
            f"{diagnostics['arm_max_error']:.6f} rad "
            f"({ARM_NAMES[arm_i]}, "
            f"{diagnostics['arm_signed_error']:+.6f})"
        )

    def print_all_joint_errors(
        self,
        q_target: np.ndarray,
        q_measured: np.ndarray,
    ):

        error = (
            q_measured
            - q_target
        )

        print()
        print(
            "Per-joint final error:"
        )

        print(
            "-" * 92
        )

        print(
            f"{'joint':28s} "
            f"{'target':>11s} "
            f"{'measured':>11s} "
            f"{'error':>11s} "
            f"{'deg':>10s}"
        )

        print(
            "-" * 92
        )

        for i, name in enumerate(
            ARM_NAMES
        ):

            print(
                f"{name:28s} "
                f"{q_target[i]:+11.6f} "
                f"{q_measured[i]:+11.6f} "
                f"{error[i]:+11.6f} "
                f"{np.degrees(error[i]):+10.3f}"
            )

        print(
            "-" * 92
        )

    # ========================================================
    # Execute one segment
    # ========================================================

    def execute_segment(
        self,
        name: str,
        q_target: np.ndarray,
        final_home: bool,
    ) -> bool:

        q_start, _ = (
            self.arm_qdq()
        )

        min_duration = (
            self.args.min_auto_duration
            if final_home
            else
            self.args.recovery_min_duration
        )

        duration = (
            self.calculate_safe_duration(
                q_start,
                q_target,
                min_duration=min_duration,
            )
        )

        metrics = (
            self.trajectory_metrics(
                q_start,
                q_target,
                duration,
            )
        )

        if (
            metrics["peak_velocity"]
            > self.args.max_velocity
        ):
            raise RuntimeError(
                f"{name}: velocity check failed"
            )

        if (
            metrics["peak_acceleration"]
            > self.args.max_acceleration
        ):
            raise RuntimeError(
                f"{name}: acceleration check failed"
            )

        if (
            metrics["max_step"]
            > self.args.max_step
        ):
            raise RuntimeError(
                f"{name}: trajectory step check failed"
            )

        print()
        print(
            "=" * 100
        )

        print(
            f"SEGMENT START: {name}"
        )

        print(
            "=" * 100
        )

        print(
            f"Duration       : "
            f"{duration:.3f}s"
        )

        print(
            f"Max joint delta: "
            f"{metrics['max_delta']:.6f} rad"
        )

        print(
            f"Peak velocity  : "
            f"{metrics['peak_velocity']:.6f} rad/s"
        )

        print(
            f"Peak accel     : "
            f"{metrics['peak_acceleration']:.6f} rad/s^2"
        )

        # ----------------------------------------------------
        # Quintic trajectory
        # ----------------------------------------------------

        start = (
            time.monotonic()
        )

        next_tick = start
        next_print = start

        while True:

            now = (
                time.monotonic()
            )

            elapsed = (
                now - start
            )

            if elapsed >= duration:
                break

            q_measured, dq_measured = (
                self.arm_qdq()
            )

            u = float(
                np.clip(
                    elapsed
                    / duration,
                    0.0,
                    1.0,
                )
            )

            h = quintic(
                u
            )

            q_plan = (
                q_start
                + h
                * (
                    q_target
                    - q_start
                )
            )

            lead = np.clip(
                q_plan
                - q_measured,
                -self.args.max_command_lead,
                self.args.max_command_lead,
            )

            q_cmd = (
                q_measured
                + lead
            )

            self.publish_arm(
                q_cmd
            )

            if (
                now
                >= next_print
            ):

                error = float(
                    np.max(
                        np.abs(
                            q_target
                            - q_measured
                        )
                    )
                )

                print(
                    f"[{name}] "
                    f"t={elapsed:.2f}/"
                    f"{duration:.2f}s "
                    f"target_err="
                    f"{error:.6f} "
                    f"max|dq|="
                    f"{np.max(np.abs(dq_measured)):.6f}"
                )

                next_print = (
                    now
                    + self.args.print_period
                )

            next_tick += (
                1.0
                / self.args.hz
            )

            dt = (
                next_tick
                - time.monotonic()
            )

            if dt > 0:
                time.sleep(dt)

        # ====================================================
        # Segment hold and reach verification
        # ====================================================

        hold_duration = (
            self.args.hold_time
            if final_home
            else
            self.args.recovery_hold_time
        )

        threshold = (
            self.args.home_threshold
            if final_home
            else
            self.args.recovery_threshold
        )

        stable_required = (
            self.args.home_stable_time
            if final_home
            else
            self.args.recovery_stable_time
        )

        hold_start = (
            time.monotonic()
        )

        next_tick = (
            hold_start
        )

        next_print = (
            hold_start
        )

        stable_start = None

        stable_reached = False

        best_error = float(
            "inf"
        )

        while (
            time.monotonic()
            - hold_start
            < hold_duration
        ):

            now = (
                time.monotonic()
            )

            q, dq = (
                self.arm_qdq()
            )

            error_vector = (
                q_target
                - q
            )

            max_error = float(
                np.max(
                    np.abs(
                        error_vector
                    )
                )
            )

            best_error = min(
                best_error,
                max_error,
            )

            # ------------------------------------------------
            # IMPORTANT:
            # PASS/FAIL logic unchanged.
            # Still uses GLOBAL 14-DoF max error.
            # ------------------------------------------------

            if (
                max_error
                < threshold
            ):

                if stable_start is None:
                    stable_start = now

                if (
                    now
                    - stable_start
                    >= stable_required
                ):

                    stable_reached = True

            else:

                stable_start = None

            lead = np.clip(
                error_vector,
                -self.args.max_command_lead,
                self.args.max_command_lead,
            )

            self.publish_arm(
                q + lead
            )

            if (
                now
                >= next_print
            ):

                stable_time = (
                    0.0
                    if stable_start is None
                    else
                    now - stable_start
                )

                print(
                    f"[{name}_hold] "
                    f"max_err="
                    f"{max_error:.6f} "
                    f"best="
                    f"{best_error:.6f} "
                    f"stable="
                    f"{stable_time:.2f}/"
                    f"{stable_required:.2f}"
                )

                next_print = (
                    now
                    + self.args.print_period
                )

            next_tick += (
                1.0
                / self.args.hz
            )

            dt = (
                next_tick
                - time.monotonic()
            )

            if dt > 0:
                time.sleep(dt)

        # ====================================================
        # Final measurement
        # ====================================================

        q_final, dq_final = (
            self.arm_qdq()
        )

        diagnostics = (
            self.calculate_segment_diagnostics(
                q_target,
                q_final,
            )
        )

        final_error = float(
            diagnostics[
                "global_max_error"
            ]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Original acceptance logic remains unchanged.
        # ----------------------------------------------------

        passed = (
            final_error
            < threshold
            and
            stable_reached
        )

        print()
        print(
            f"{name} RESULT"
        )

        print(
            f"Final error   : "
            f"{final_error:.6f} rad"
        )

        print(
            f"Best error    : "
            f"{best_error:.6f} rad"
        )

        print(
            f"Final max |dq|: "
            f"{np.max(np.abs(dq_final)):.6f} rad/s"
        )

        print(
            f"Stable reached: "
            f"{stable_reached}"
        )

        # ====================================================
        # NEW DIAGNOSTIC OUTPUT
        # ====================================================

        self.print_segment_diagnostics(
            diagnostics
        )

        # If a segment fails, print all 14 joints.
        if not passed:

            print()
            print(
                "DIAGNOSTIC DETAIL:"
            )

            print(
                "Segment failed using the existing "
                "global 14-DoF acceptance criterion."
            )

            self.print_all_joint_errors(
                q_target,
                q_final,
            )

        print()

        print(
            f"{name} = "
            f"{'PASS' if passed else 'FAIL'}"
        )

        print(
            "=" * 100
        )

        return passed

    # ========================================================
    # Execute entire recovery
    # ========================================================

    def execute(
        self,
        q_home: np.ndarray,
    ) -> int:

        passed, _ = (
            self.preflight(
                q_home,
                verbose=False,
            )
        )

        if not passed:

            raise RuntimeError(
                "Pre-flight failed"
            )

        self.capture_hold_positions()

        self.create_publisher()

        q_current, _ = (
            self.arm_qdq()
        )

        plan = (
            self.build_recovery_plan(
                q_current,
                q_home,
            )
        )

        self.print_recovery_plan(
            q_current,
            q_home,
            plan,
        )

        print()
        print(
            "=" * 100
        )

        print(
            "REAL ROBOT SEGMENTED AUTO-HOME ARMED"
        )

        print(
            "=" * 100
        )

        print(
            f"Segments : {len(plan)}"
        )

        print(
            "LowCmd will remain active across "
            "all recovery segments."
        )

        print()

        print(
            "Type RECOVER HOME exactly to continue."
        )

        phrase = input(
            "> "
        )

        if (
            phrase
            != "RECOVER HOME"
        ):

            print(
                "Confirmation rejected."
            )

            return 2

        print()
        print(
            "Starting in 3 seconds..."
        )

        for n in [
            3,
            2,
            1,
        ]:

            print(
                n
            )

            time.sleep(
                1.0
            )

        try:

            # =================================================
            # Single LowCmd takeover
            # =================================================

            q_after = (
                self.takeover_test()
            )

            # -------------------------------------------------
            # Rebuild plan using actual post-takeover q
            # -------------------------------------------------

            plan = (
                self.build_recovery_plan(
                    q_after,
                    q_home,
                )
            )

            print()
            print(
                "POST-TAKEOVER RECOVERY PLAN"
            )

            self.print_recovery_plan(
                q_after,
                q_home,
                plan,
            )

            # =================================================
            # Execute all segments
            # =================================================

            for i, segment in enumerate(
                plan,
                start=1,
            ):

                print()

                print(
                    f"Executing segment "
                    f"{i}/{len(plan)}: "
                    f"{segment.name}"
                )

                final_home = (
                    segment.name
                    == "FINAL_HOME"
                )

                ok = (
                    self.execute_segment(
                        segment.name,
                        segment.target,
                        final_home=final_home,
                    )
                )

                if not ok:

                    print()
                    print(
                        "=" * 100
                    )

                    print(
                        "RECOVERY ABORTED"
                    )

                    print(
                        f"Failed segment: "
                        f"{segment.name}"
                    )

                    print(
                        "The controller behavior and "
                        "thresholds were NOT automatically changed."
                    )

                    print(
                        "=" * 100
                    )

                    return 5

            # =================================================
            # Final HOME summary
            # =================================================

            print()
            print(
                "=" * 100
            )

            print(
                "SEGMENTED AUTO HOME RESULT"
            )

            print(
                "=" * 100
            )

            q_final, dq_final = (
                self.arm_qdq()
            )

            error = (
                q_final
                - q_home
            )

            abs_error = (
                np.abs(
                    error
                )
            )

            worst = int(
                np.argmax(
                    abs_error
                )
            )

            max_error = float(
                abs_error[
                    worst
                ]
            )

            print(
                f"MAX error : "
                f"{max_error:.6f} rad "
                f"("
                f"{np.degrees(max_error):.3f} deg"
                f")"
            )

            print(
                f"Worst     : "
                f"{ARM_NAMES[worst]}"
            )

            print(
                f"Final |dq|: "
                f"{np.max(np.abs(dq_final)):.6f} rad/s"
            )

            print()

            for i, name in enumerate(
                ARM_NAMES
            ):

                print(
                    f"{name:28s} "
                    f"target="
                    f"{q_home[i]:+.6f} "
                    f"final="
                    f"{q_final[i]:+.6f} "
                    f"error="
                    f"{error[i]:+.6f}"
                )

            ready = (
                max_error
                < self.args.home_threshold
            )

            print()

            print(
                "AUTO_HOME_READY = "
                + (
                    "PASS"
                    if ready
                    else "FAIL"
                )
            )

            print(
                "=" * 100
            )

            return (
                0
                if ready
                else 3
            )

        finally:

            self.release()

            print(
                "LowCmd gains released."
            )


# ============================================================
# Argument parser
# ============================================================

def build_parser():

    p = argparse.ArgumentParser(
        description=(
            "R1-A7 segmented safe AUTO HOME "
            "with recovery diagnostics"
        )
    )

    p.add_argument(
        "keyframe",
        nargs="?",
        default="HOME",
    )

    p.add_argument(
        "--interface",
        default="enp6s0",
    )

    p.add_argument(
        "--domain-id",
        type=int,
        default=0,
    )

    p.add_argument(
        "--state-topic",
        default="rt/lowstate",
    )

    p.add_argument(
        "--command-topic",
        default="rt/lowcmd",
    )

    p.add_argument(
        "--keyframes",
        type=Path,
        default=(
            ROOT
            / "config"
            / "keyframes.yaml"
        ),
    )

    # ========================================================
    # Motion
    # ========================================================

    p.add_argument(
        "--hz",
        type=float,
        default=100.0,
    )

    p.add_argument(
        "--state-timeout",
        type=float,
        default=0.20,
    )

    p.add_argument(
        "--max-velocity",
        type=float,
        default=0.06,
    )

    p.add_argument(
        "--max-acceleration",
        type=float,
        default=0.50,
    )

    p.add_argument(
        "--max-step",
        type=float,
        default=0.010,
    )

    p.add_argument(
        "--max-command-lead",
        type=float,
        default=0.060,
    )

    p.add_argument(
        "--max-start-dq",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--joint-limit-margin",
        type=float,
        default=0.04,
    )

    # ========================================================
    # AUTO duration
    # ========================================================

    p.add_argument(
        "--min-auto-duration",
        type=float,
        default=8.0,
    )

    p.add_argument(
        "--recovery-min-duration",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--duration-margin",
        type=float,
        default=1.10,
    )

    p.add_argument(
        "--duration-step",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--max-auto-duration",
        type=float,
        default=20.0,
    )

    # ========================================================
    # Segmented recovery
    # ========================================================

    p.add_argument(
        "--direct-home-delta",
        type=float,
        default=0.35,
        help=(
            "Maximum joint displacement for "
            "one direct HOME move."
        ),
    )

    p.add_argument(
        "--recovery-stage-delta",
        type=float,
        default=0.20,
        help=(
            "Maximum nominal joint displacement "
            "per recovery stage."
        ),
    )

    p.add_argument(
        "--recovery-threshold",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--recovery-stable-time",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--recovery-hold-time",
        type=float,
        default=2.0,
    )

    # ========================================================
    # Gains
    # ========================================================

    p.add_argument(
        "--kp-shoulder-pitch",
        type=float,
        default=100.0,
    )

    p.add_argument(
        "--kd-shoulder-pitch",
        type=float,
        default=3.5,
    )

    p.add_argument(
        "--kp-low",
        type=float,
        default=80.0,
    )

    p.add_argument(
        "--kd-low",
        type=float,
        default=3.0,
    )

    p.add_argument(
        "--kp-wrist",
        type=float,
        default=40.0,
    )

    p.add_argument(
        "--kd-wrist",
        type=float,
        default=1.5,
    )

    p.add_argument(
        "--kp-wrist-pitch",
        type=float,
        default=60.0,
    )

    p.add_argument(
        "--kd-wrist-pitch",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--hold-kp",
        type=float,
        default=10.0,
    )

    p.add_argument(
        "--hold-kd",
        type=float,
        default=0.8,
    )

    p.add_argument(
        "--gripper-kp",
        type=float,
        default=8.0,
    )

    p.add_argument(
        "--gripper-kd",
        type=float,
        default=1.5,
    )

    # ========================================================
    # Takeover
    # ========================================================

    p.add_argument(
        "--takeover-time",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--max-takeover-drift",
        type=float,
        default=0.05,
    )

    # ========================================================
    # Final HOME
    # ========================================================

    p.add_argument(
        "--hold-time",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--home-threshold",
        type=float,
        default=0.03,
    )

    p.add_argument(
        "--home-stable-time",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--print-period",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--execute",
        action="store_true",
    )

    return p


# ============================================================
# Main
# ============================================================

def main():

    args = (
        build_parser()
        .parse_args()
    )

    args.keyframe = (
        args.keyframe.upper()
    )

    # --------------------------------------------------------
    # Argument validation
    # --------------------------------------------------------

    if args.hz <= 0:
        raise RuntimeError(
            "--hz must be > 0"
        )

    if args.state_timeout <= 0:
        raise RuntimeError(
            "--state-timeout must be > 0"
        )

    if args.max_velocity <= 0:
        raise RuntimeError(
            "--max-velocity must be > 0"
        )

    if args.max_acceleration <= 0:
        raise RuntimeError(
            "--max-acceleration must be > 0"
        )

    if args.max_step <= 0:
        raise RuntimeError(
            "--max-step must be > 0"
        )

    if args.max_command_lead <= 0:
        raise RuntimeError(
            "--max-command-lead must be > 0"
        )

    if args.max_start_dq <= 0:
        raise RuntimeError(
            "--max-start-dq must be > 0"
        )

    if (
        args.recovery_stage_delta
        <= 0
    ):
        raise RuntimeError(
            "--recovery-stage-delta must be > 0"
        )

    if (
        args.direct_home_delta
        <= 0
    ):
        raise RuntimeError(
            "--direct-home-delta must be > 0"
        )

    if (
        args.recovery_threshold
        <= 0
    ):
        raise RuntimeError(
            "--recovery-threshold must be > 0"
        )

    if (
        args.home_threshold
        <= 0
    ):
        raise RuntimeError(
            "--home-threshold must be > 0"
        )

    if (
        args.duration_margin
        < 1.0
    ):
        raise RuntimeError(
            "--duration-margin must be >= 1.0"
        )

    if (
        args.duration_step
        <= 0
    ):
        raise RuntimeError(
            "--duration-step must be > 0"
        )

    if (
        args.max_auto_duration
        <= 0
    ):
        raise RuntimeError(
            "--max-auto-duration must be > 0"
        )

    # --------------------------------------------------------
    # Load HOME keyframe
    # --------------------------------------------------------

    q_home = (
        load_keyframe(
            args.keyframes,
            args.keyframe,
        )
    )

    # --------------------------------------------------------
    # Controller
    # --------------------------------------------------------

    mover = (
        AutoKeyframeMover(
            args
        )
    )

    mover.connect_state_only()

    # --------------------------------------------------------
    # Initial read-only PRE-FLIGHT
    # --------------------------------------------------------

    (
        passed,
        _,
    ) = mover.preflight(
        q_home,
        verbose=True,
    )

    if not passed:

        print()
        print(
            "ABORT: pre-flight failed."
        )

        return 2

    # --------------------------------------------------------
    # PRE-FLIGHT only
    # --------------------------------------------------------

    if not args.execute:

        print()
        print(
            "PRE-FLIGHT ONLY."
        )

        print(
            "Robot was NOT commanded."
        )

        return 0

    # --------------------------------------------------------
    # Real robot execution
    # --------------------------------------------------------

    return (
        mover.execute(
            q_home
        )
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )