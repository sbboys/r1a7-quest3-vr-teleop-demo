#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
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


# Motor 13:
# Hold at the measured position captured before LowCmd takeover.
WAIST_INDEX = 13


# Auxiliary motors used by the verified R1-A7 LowCmd path.
AUX_HOLD_INDICES = [
    29,
    30,
]


# Two-finger gripper motor indices.
GRIPPER_INDICES = [
    31,
    33,
]


# ============================================================
# R1-A7 arm limits
#
# Order exactly matches ARM_NAMES.
# ============================================================

R1A7_ARM_LIMITS = np.array(
    [
        # Left
        [-3.1416, 2.0944],
        [-0.2269, 2.4784],
        [-1.9199, 1.9199],
        [-0.9757, 2.1850],
        [-1.9199, 1.9199],
        [-1.61429558, 1.61429558],
        [-1.61429558, 1.61429558],

        # Right
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
# Utility functions
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
            f"Invalid YAML structure: {path}"
        )

    if "keyframes" not in data:
        raise RuntimeError(
            "keyframes.yaml has no 'keyframes' section"
        )

    if name not in data["keyframes"]:
        raise RuntimeError(
            f"Keyframe '{name}' does not exist"
        )

    kf = data["keyframes"][name]

    if (
        "left_joint_position" not in kf
        or
        "right_joint_position" not in kf
    ):
        raise RuntimeError(
            f"Keyframe '{name}' does not contain "
            "left/right joint positions"
        )

    q = np.asarray(
        kf["left_joint_position"]
        + kf["right_joint_position"],
        dtype=float,
    )

    if q.shape != (14,):
        raise RuntimeError(
            f"{name} must contain exactly 14 arm joints"
        )

    if not np.all(
        np.isfinite(q)
    ):
        raise RuntimeError(
            f"{name} contains non-finite joint values"
        )

    return q


def quintic(u):
    """
    Quintic time scaling:

        h(u) = 10u^3 - 15u^4 + 6u^5

    Properties:

        h(0) = 0
        h(1) = 1

        dh/dt = 0 at both endpoints
        d2h/dt2 = 0 at both endpoints
    """

    return (
        10.0 * u**3
        - 15.0 * u**4
        + 6.0 * u**5
    )


# ============================================================
# R1-A7 AUTO keyframe controller
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

        # Actual duration used by the current motion.
        #
        # If --duration is omitted:
        #     calculated automatically.
        #
        # If --duration is specified:
        #     uses that manual value.
        self.active_duration: Optional[
            float
        ] = None

    # ========================================================
    # DDS / LowState
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
                self.last_lowstate_time is not None
            ):

                if (
                    self.state_age()
                    <= self.args.state_timeout
                ):
                    return

            time.sleep(
                0.02
            )

        raise RuntimeError(
            "No fresh rt/lowstate received"
        )

    def state_age(
        self,
    ) -> float:

        if (
            self.last_lowstate_time
            is None
        ):
            return float(
                "inf"
            )

        return (
            time.monotonic()
            - self.last_lowstate_time
        )

    def arm_qdq(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        if self.low_state is None:
            raise RuntimeError(
                "No lowstate"
            )

        states = (
            self.low_state.motor_state
        )

        max_idx = max(
            ARM_INDICES
        )

        if (
            len(states)
            <= max_idx
        ):
            raise RuntimeError(
                f"lowstate has only {len(states)} motors; "
                f"need index {max_idx}"
            )

        q = np.asarray(
            [
                float(
                    states[i].q
                )
                for i in ARM_INDICES
            ],
            dtype=float,
        )

        dq = np.asarray(
            [
                float(
                    states[i].dq
                )
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

        states = (
            self.low_state.motor_state
        )

        values = []

        for idx in ARM_INDICES:

            motor = (
                states[idx]
            )

            value = 0.0

            if hasattr(
                motor,
                "tau_est",
            ):

                try:
                    value = float(
                        motor.tau_est
                    )
                except Exception:
                    value = 0.0

            elif hasattr(
                motor,
                "tau",
            ):

                try:
                    value = float(
                        motor.tau
                    )
                except Exception:
                    value = 0.0

            elif hasattr(
                motor,
                "current",
            ):

                try:
                    value = float(
                        motor.current
                    )
                except Exception:
                    value = 0.0

            values.append(
                value
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
    ) -> tuple[
        bool,
        int,
        object,
    ]:

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

            print(
                "MotionSwitcher check error:",
                repr(exc),
            )

            return (
                False,
                -1,
                None,
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
            bool(
                np.all(mask)
            ),
            low,
            high,
            mask,
        )

    # ========================================================
    # AUTO duration calculation
    # ========================================================

    def calculate_safe_duration(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
    ) -> tuple[
        float,
        dict,
    ]:

        delta = np.abs(
            q1 - q0
        )

        max_delta = float(
            np.max(delta)
        )

        if (
            self.args.max_velocity
            <= 0.0
        ):
            raise RuntimeError(
                "--max-velocity must be > 0"
            )

        if (
            self.args.max_acceleration
            <= 0.0
        ):
            raise RuntimeError(
                "--max-acceleration must be > 0"
            )

        # ----------------------------------------------------
        # Quintic peak velocity:
        #
        # v_peak =
        #
        #     1.875 * Delta_q / T
        #
        # therefore:
        #
        # T >= 1.875 * Delta_q / v_limit
        # ----------------------------------------------------

        t_velocity = (
            1.875
            * max_delta
            / self.args.max_velocity
        )

        # ----------------------------------------------------
        # Quintic peak acceleration:
        #
        # a_peak ≈
        #
        #     5.7735026919 * Delta_q / T^2
        #
        # therefore:
        #
        # T >= sqrt(
        #     5.7735 * Delta_q / a_limit
        # )
        # ----------------------------------------------------

        t_acceleration = float(
            np.sqrt(
                5.7735026919
                * max_delta
                / self.args.max_acceleration
            )
        )

        theoretical_min = max(
            t_velocity,
            t_acceleration,
        )

        # Add safety margin only to the dynamically
        # calculated minimum.
        required_dynamic_time = (
            theoretical_min
            * self.args.duration_margin
        )

        # Never make normal HOME movement excessively fast
        # just because the current pose is already close.
        duration = max(
            self.args.min_auto_duration,
            required_dynamic_time,
        )

        step = float(
            self.args.duration_step
        )

        if (
            step
            <= 0.0
        ):
            raise RuntimeError(
                "--duration-step must be > 0"
            )

        # Round UP to the configured step.
        #
        # Examples with step=0.5:
        #
        # 8.01 -> 8.5
        # 9.20 -> 9.5
        # 10.01 -> 10.5
        duration = (
            np.ceil(
                duration
                / step
            )
            * step
        )

        duration = float(
            duration
        )

        if (
            duration
            > self.args.max_auto_duration
        ):

            raise RuntimeError(
                "Automatically calculated trajectory "
                "duration is too long: "
                f"{duration:.3f} s > "
                f"{self.args.max_auto_duration:.3f} s. "
                "Current pose may be too far from target."
            )

        info = {
            "max_delta": max_delta,

            "velocity_min_duration": float(
                t_velocity
            ),

            "acceleration_min_duration": float(
                t_acceleration
            ),

            "theoretical_min_duration": float(
                theoretical_min
            ),

            "duration_margin": float(
                self.args.duration_margin
            ),

            "selected_duration": duration,
        }

        return (
            duration,
            info,
        )

    # ========================================================
    # Trajectory metrics
    # ========================================================

    def trajectory_metrics(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        duration: float,
    ) -> dict:

        delta = np.abs(
            q1 - q0
        )

        max_delta = float(
            np.max(delta)
        )

        T = float(
            duration
        )

        if (
            T
            <= 0.0
        ):
            raise RuntimeError(
                "Trajectory duration must be > 0"
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
                / (T**2)
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

        u = (
            t / T
        )

        h = quintic(
            u
        )

        trajectory = (
            q0[None, :]
            + h[:, None]
            * (
                q1
                - q0
            )[None, :]
        )

        step = np.diff(
            trajectory,
            axis=0,
        )

        max_step = float(
            np.max(
                np.abs(step)
            )
        )

        return {
            "max_delta": max_delta,
            "peak_velocity": peak_velocity,
            "peak_acceleration": peak_acceleration,
            "max_step": max_step,
            "samples": n,
        }

    # ========================================================
    # Joint gain mapping
    # ========================================================

    def arm_gain(
        self,
        joint_i: int,
    ) -> tuple[
        float,
        float,
    ]:

        local_i = (
            joint_i % 7
        )

        # ----------------------------------------------------
        # local_i:
        #
        # 0 = shoulder pitch
        # 1 = shoulder roll
        # 2 = shoulder yaw
        # 3 = elbow
        # 4 = wrist roll
        # 5 = wrist pitch
        # 6 = wrist yaw
        # ----------------------------------------------------

        # Shoulder pitch:
        #
        # Tuned higher after HOME steady-state tests.
        if (
            local_i == 0
        ):

            return (
                self.args.kp_shoulder_pitch,
                self.args.kd_shoulder_pitch,
            )

        # Shoulder roll / shoulder yaw / elbow.
        if (
            local_i < 4
        ):

            return (
                self.args.kp_low,
                self.args.kd_low,
            )

        # Wrist pitch:
        #
        # Tuned higher after HOME steady-state tests.
        if (
            local_i == 5
        ):

            return (
                self.args.kp_wrist_pitch,
                self.args.kd_wrist_pitch,
            )

        # Wrist roll / wrist yaw.
        return (
            self.args.kp_wrist,
            self.args.kd_wrist,
        )

    # ========================================================
    # Resolve duration
    # ========================================================

    def resolve_duration(
        self,
        q_current: np.ndarray,
        q_target: np.ndarray,
    ) -> tuple[
        float,
        str,
        Optional[dict],
    ]:

        # AUTO mode.
        if (
            self.args.duration
            is None
        ):

            duration, info = (
                self.calculate_safe_duration(
                    q_current,
                    q_target,
                )
            )

            return (
                duration,
                "AUTO",
                info,
            )

        # MANUAL mode.
        duration = float(
            self.args.duration
        )

        return (
            duration,
            "MANUAL",
            None,
        )

    # ========================================================
    # PRE-FLIGHT
    # ========================================================

    def preflight(
        self,
        q_target: np.ndarray,
        verbose: bool = True,
    ) -> bool:

        self.wait_for_state(
            timeout=3.0
        )

        q_current, dq_current = (
            self.arm_qdq()
        )

        # ----------------------------------------------------
        # Resolve trajectory duration.
        # ----------------------------------------------------

        try:

            (
                duration,
                duration_mode,
                duration_info,
            ) = self.resolve_duration(
                q_current,
                q_target,
            )

        except RuntimeError as exc:

            if verbose:

                print()
                print(
                    "=" * 108
                )

                print(
                    "R1-A7 AUTO KEYFRAME PRE-FLIGHT"
                )

                print(
                    "=" * 108
                )

                print(
                    "AUTO DURATION CALCULATION FAILED"
                )

                print(
                    str(exc)
                )

                print()

                print(
                    "PRE_FLIGHT = FAIL"
                )

                print(
                    "ROBOT_COMMAND_PUBLISHED = FALSE"
                )

                print(
                    "=" * 108
                )

            return False

        self.active_duration = (
            duration
        )

        metrics = (
            self.trajectory_metrics(
                q_current,
                q_target,
                duration,
            )
        )

        (
            current_limit_ok,
            current_low,
            current_high,
            current_mask,
        ) = self.check_joint_limits(
            q_current
        )

        (
            target_limit_ok,
            target_low,
            target_high,
            target_mask,
        ) = self.check_joint_limits(
            q_target
        )

        state_age = (
            self.state_age()
        )

        state_ok = (
            state_age
            <= self.args.state_timeout
        )

        initial_velocity = float(
            np.max(
                np.abs(
                    dq_current
                )
            )
        )

        initial_velocity_ok = (
            initial_velocity
            <= self.args.max_start_dq
        )

        velocity_ok = (
            metrics[
                "peak_velocity"
            ]
            <= self.args.max_velocity
        )

        acceleration_ok = (
            metrics[
                "peak_acceleration"
            ]
            <= self.args.max_acceleration
        )

        step_ok = (
            metrics[
                "max_step"
            ]
            <= self.args.max_step
        )

        delta_ok = (
            metrics[
                "max_delta"
            ]
            <= self.args.max_initial_delta
        )

        (
            motion_mode_ok,
            motion_status,
            motion_result,
        ) = self.check_motion_switcher()

        passed = all(
            [
                state_ok,
                current_limit_ok,
                target_limit_ok,
                initial_velocity_ok,
                velocity_ok,
                acceleration_ok,
                step_ok,
                delta_ok,
                motion_mode_ok,
            ]
        )

        if verbose:

            print()
            print(
                "=" * 108
            )

            print(
                "R1-A7 AUTO KEYFRAME PRE-FLIGHT"
            )

            print(
                "=" * 108
            )

            print(
                f"Target keyframe       : "
                f"{self.args.keyframe}"
            )

            print(
                f"Interface             : "
                f"{self.args.interface}"
            )

            print(
                f"State topic           : "
                f"{self.args.state_topic}"
            )

            print(
                f"Command topic         : "
                f"{self.args.command_topic}"
            )

            print(
                f"Lowstate count        : "
                f"{self.lowstate_count}"
            )

            print(
                f"Lowstate age          : "
                f"{state_age:.6f} s "
                f"["
                f"{'PASS' if state_ok else 'FAIL'}"
                f"]"
            )

            print()

            print(
                "MotionSwitcher:"
            )

            print(
                f"  status              : "
                f"{motion_status}"
            )

            print(
                f"  result              : "
                f"{motion_result}"
            )

            print(
                f"  no active mode      : "
                f"{'PASS' if motion_mode_ok else 'FAIL'}"
            )

            print()

            print(
                "Trajectory:"
            )

            print(
                f"  Duration mode       : "
                f"{duration_mode}"
            )

            print(
                f"  Duration            : "
                f"{duration:.3f} s"
            )

            if (
                duration_info
                is not None
            ):

                print(
                    f"  Max joint delta     : "
                    f"{duration_info['max_delta']:.6f} rad"
                )

                print(
                    f"  Velocity min time   : "
                    f"{duration_info['velocity_min_duration']:.3f} s"
                )

                print(
                    f"  Accel min time      : "
                    f"{duration_info['acceleration_min_duration']:.3f} s"
                )

                print(
                    f"  Theoretical min     : "
                    f"{duration_info['theoretical_min_duration']:.3f} s"
                )

                print(
                    f"  Safety margin       : "
                    f"{duration_info['duration_margin']:.3f}"
                )

                print(
                    f"  AUTO selected time  : "
                    f"{duration_info['selected_duration']:.3f} s"
                )

            print(
                f"  Control frequency   : "
                f"{self.args.hz:.1f} Hz"
            )

            print(
                f"  Samples             : "
                f"{metrics['samples']}"
            )

            print()

            print(
                "Arm gains:"
            )

            print(
                f"  shoulder pitch      : "
                f"kp="
                f"{self.args.kp_shoulder_pitch:.1f}, "
                f"kd="
                f"{self.args.kd_shoulder_pitch:.1f}"
            )

            print(
                f"  shoulder roll/yaw   : "
                f"kp="
                f"{self.args.kp_low:.1f}, "
                f"kd="
                f"{self.args.kd_low:.1f}"
            )

            print(
                f"  elbow               : "
                f"kp="
                f"{self.args.kp_low:.1f}, "
                f"kd="
                f"{self.args.kd_low:.1f}"
            )

            print(
                f"  wrist pitch         : "
                f"kp="
                f"{self.args.kp_wrist_pitch:.1f}, "
                f"kd="
                f"{self.args.kd_wrist_pitch:.1f}"
            )

            print(
                f"  wrist roll/yaw      : "
                f"kp="
                f"{self.args.kp_wrist:.1f}, "
                f"kd="
                f"{self.args.kd_wrist:.1f}"
            )

            print()

            print(
                f"Max initial |dq|      : "
                f"{initial_velocity:.6f} rad/s "
                f"["
                f"{'PASS' if initial_velocity_ok else 'FAIL'}"
                f"]"
            )

            print(
                f"Largest joint motion  : "
                f"{metrics['max_delta']:.6f} rad "
                f"["
                f"{'PASS' if delta_ok else 'FAIL'}"
                f"]"
            )

            print(
                f"Peak planned velocity : "
                f"{metrics['peak_velocity']:.6f} rad/s "
                f"["
                f"{'PASS' if velocity_ok else 'FAIL'}"
                f"]"
            )

            print(
                f"Peak planned accel    : "
                f"{metrics['peak_acceleration']:.6f} rad/s^2 "
                f"["
                f"{'PASS' if acceleration_ok else 'FAIL'}"
                f"]"
            )

            print(
                f"Max trajectory step   : "
                f"{metrics['max_step']:.6f} rad "
                f"["
                f"{'PASS' if step_ok else 'FAIL'}"
                f"]"
            )

            print()

            print(
                f"Current joint limits  : "
                f"{'PASS' if current_limit_ok else 'FAIL'}"
            )

            print(
                f"Target joint limits   : "
                f"{'PASS' if target_limit_ok else 'FAIL'}"
            )

            print()

            print(
                f"{'joint':28s}"
                f"{'current':>12s}"
                f"{'target':>12s}"
                f"{'delta':>12s}"
                f"{'deg':>10s}"
            )

            print(
                "-" * 76
            )

            for i, name in enumerate(
                ARM_NAMES
            ):

                delta_i = (
                    q_target[i]
                    - q_current[i]
                )

                print(
                    f"{name:28s}"
                    f"{q_current[i]:+12.6f}"
                    f"{q_target[i]:+12.6f}"
                    f"{delta_i:+12.6f}"
                    f"{np.degrees(delta_i):+10.3f}"
                )

            print(
                "-" * 76
            )

            if (
                not current_limit_ok
            ):

                print()
                print(
                    "Current joint-limit failures:"
                )

                for i in np.where(
                    ~current_mask
                )[0]:

                    print(
                        f"  {ARM_NAMES[i]} "
                        f"q={q_current[i]:+.6f}, "
                        f"range=["
                        f"{current_low[i]:+.6f}, "
                        f"{current_high[i]:+.6f}]"
                    )

            if (
                not target_limit_ok
            ):

                print()
                print(
                    "Target joint-limit failures:"
                )

                for i in np.where(
                    ~target_mask
                )[0]:

                    print(
                        f"  {ARM_NAMES[i]} "
                        f"q={q_target[i]:+.6f}, "
                        f"range=["
                        f"{target_low[i]:+.6f}, "
                        f"{target_high[i]:+.6f}]"
                    )

            if (
                self.low_state
                is not None
            ):

                ms = (
                    self.low_state.motor_state
                )

                print()

                if (
                    len(ms)
                    > WAIST_INDEX
                ):

                    print(
                        f"Motor 13 current q    : "
                        f"{float(ms[WAIST_INDEX].q):+.6f}"
                    )

                if (
                    len(ms)
                    > max(GRIPPER_INDICES)
                ):

                    print(
                        f"Left gripper q        : "
                        f"{float(ms[31].q):+.6f}"
                    )

                    print(
                        f"Right gripper q       : "
                        f"{float(ms[33].q):+.6f}"
                    )

            print()

            print(
                "=" * 108
            )

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
                "=" * 108
            )

            print()

        return passed

    # ========================================================
    # LowCmd helper
    # ========================================================

    def init_lowcmd_zero(
        self,
    ) -> None:

        for motor in (
            self.low_cmd.motor_cmd
        ):

            motor.tau = 0.0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    # ========================================================
    # LowCmd publish
    # ========================================================

    def publish_arm(
        self,
        q_desired: np.ndarray,
    ) -> None:

        if (
            self.publisher
            is None
        ):
            raise RuntimeError(
                "Publisher not initialized"
            )

        if (
            self.low_state
            is None
        ):
            raise RuntimeError(
                "No lowstate"
            )

        if (
            self.state_age()
            > self.args.state_timeout
        ):

            raise RuntimeError(
                "LOWSTATE WATCHDOG TIMEOUT"
            )

        if (
            self.waist_hold_q
            is None
        ):

            raise RuntimeError(
                "waist_hold_q not initialized"
            )

        if (
            self.gripper_hold_q
            is None
        ):

            raise RuntimeError(
                "gripper_hold_q not initialized"
            )

        q_desired = np.asarray(
            q_desired,
            dtype=float,
        )

        if (
            q_desired.shape
            != (14,)
        ):

            raise RuntimeError(
                "q_desired must contain 14 joints"
            )

        if (
            not np.all(
                np.isfinite(
                    q_desired
                )
            )
        ):

            raise RuntimeError(
                "q_desired contains non-finite values"
            )

        (
            limit_ok,
            _,
            _,
            _,
        ) = self.check_joint_limits(
            q_desired
        )

        if (
            not limit_ok
        ):

            raise RuntimeError(
                "Refusing to publish command outside "
                "safe joint limits"
            )

        self.init_lowcmd_zero()

        # ----------------------------------------------------
        # LowCmd mode fields
        # ----------------------------------------------------

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
        # Motor 13 fixed hold
        # ----------------------------------------------------

        if (
            WAIST_INDEX
            < len(
                self.low_cmd.motor_cmd
            )
        ):

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
        #
        # Hold near current measured q.
        # ----------------------------------------------------

        for idx in (
            AUX_HOLD_INDICES
        ):

            if (
                idx >= len(ms)
                or
                idx
                >= len(
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
        # Dual-arm command
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

            motor.mode = 1
            motor.tau = 0.0

            motor.q = float(
                q
            )

            motor.dq = 0.0

            kp, kd = (
                self.arm_gain(
                    joint_i
                )
            )

            motor.kp = float(
                kp
            )

            motor.kd = float(
                kd
            )

        # ----------------------------------------------------
        # Hold grippers at takeover position
        # ----------------------------------------------------

        for i, idx in enumerate(
            GRIPPER_INDICES
        ):

            if (
                idx >= len(ms)
                or
                idx
                >= len(
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
                self.gripper_hold_q[i]
            )

            motor.dq = 0.0

            motor.kp = float(
                self.args.gripper_kp
            )

            motor.kd = float(
                self.args.gripper_kd
            )

        # ----------------------------------------------------
        # CRC + publish
        # ----------------------------------------------------

        self.low_cmd.crc = (
            self.crc.Crc(
                self.low_cmd
            )
        )

        self.publisher.Write(
            self.low_cmd
        )

        self.commands_published += 1

    # ========================================================
    # Release LowCmd gains
    # ========================================================

    def release(
        self,
    ) -> None:

        if (
            self.publisher
            is None
        ):
            return

        try:

            self.init_lowcmd_zero()

            if hasattr(
                self.low_cmd,
                "mode_pr",
            ):

                self.low_cmd.mode_pr = 0

            if (
                self.low_state
                is not None
                and
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
                "Warning: LowCmd release failed:",
                repr(exc),
            )

    # ========================================================
    # Capture waist / gripper hold positions
    # ========================================================

    def capture_hold_positions(
        self,
    ) -> None:

        self.wait_for_state(
            timeout=2.0
        )

        if (
            self.low_state
            is None
        ):

            raise RuntimeError(
                "No lowstate"
            )

        ms = (
            self.low_state.motor_state
        )

        required_max_idx = max(
            [WAIST_INDEX]
            + GRIPPER_INDICES
        )

        if (
            len(ms)
            <= required_max_idx
        ):

            raise RuntimeError(
                "LowState does not contain "
                "required motors"
            )

        self.waist_hold_q = float(
            ms[
                WAIST_INDEX
            ].q
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

    # ========================================================
    # Zero-displacement LowCmd takeover
    # ========================================================

    def takeover_test(
        self,
    ) -> tuple[
        bool,
        np.ndarray,
    ]:

        self.wait_for_state(
            timeout=2.0
        )

        q_takeover, _ = (
            self.arm_qdq()
        )

        tau_before = (
            self.arm_tau_est()
        )

        start = (
            time.monotonic()
        )

        next_tick = start
        next_print = start

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

        print(
            f"Duration : "
            f"{self.args.takeover_time:.3f} s"
        )

        print(
            "Target   : current measured arm pose"
        )

        print(
            "Expected : no significant arm motion"
        )

        print()

        while (
            time.monotonic()
            - start
            < self.args.takeover_time
        ):

            if (
                self.state_age()
                > self.args.state_timeout
            ):

                raise RuntimeError(
                    "LOWSTATE WATCHDOG TIMEOUT "
                    "during takeover"
                )

            q_measured, dq_measured = (
                self.arm_qdq()
            )

            drift = float(
                np.max(
                    np.abs(
                        q_measured
                        - q_takeover
                    )
                )
            )

            max_drift = max(
                max_drift,
                drift,
            )

            if (
                drift
                > self.args.max_takeover_drift
            ):

                raise RuntimeError(
                    "TAKEOVER DRIFT TOO LARGE: "
                    f"{drift:.6f} rad > "
                    f"{self.args.max_takeover_drift:.6f} rad"
                )

            self.publish_arm(
                q_takeover
            )

            now = (
                time.monotonic()
            )

            if (
                now
                >= next_print
            ):

                next_print = (
                    now
                    + self.args.print_period
                )

                print(
                    f"[takeover] "
                    f"drift={drift:.6f} rad "
                    f"max|dq|="
                    f"{np.max(np.abs(dq_measured)):.6f} rad/s "
                    f"published="
                    f"{self.commands_published}"
                )

            next_tick += (
                1.0
                / self.args.hz
            )

            sleep_time = (
                next_tick
                - time.monotonic()
            )

            if (
                sleep_time
                > 0.0
            ):

                time.sleep(
                    sleep_time
                )

        q_after, dq_after = (
            self.arm_qdq()
        )

        tau_after = (
            self.arm_tau_est()
        )

        final_drift = float(
            np.max(
                np.abs(
                    q_after
                    - q_takeover
                )
            )
        )

        tau_change = float(
            np.max(
                np.abs(
                    tau_after
                    - tau_before
                )
            )
        )

        passed = (
            max_drift
            <= self.args.max_takeover_drift
        )

        print()

        print(
            f"Takeover final drift : "
            f"{final_drift:.6f} rad "
            f"({np.degrees(final_drift):.3f} deg)"
        )

        print(
            f"Takeover max drift   : "
            f"{max_drift:.6f} rad "
            f"({np.degrees(max_drift):.3f} deg)"
        )

        print(
            f"Final max |dq|       : "
            f"{np.max(np.abs(dq_after)):.6f} rad/s"
        )

        print(
            f"Max tau_est change   : "
            f"{tau_change:.6f}"
        )

        print(
            f"LowCmd published     : "
            f"{self.commands_published}"
        )

        print()

        print(
            "TAKEOVER_TEST = "
            + (
                "PASS"
                if passed
                else "FAIL"
            )
        )

        print(
            "=" * 100
        )

        print()

        return (
            passed,
            q_after.copy(),
        )

    # ========================================================
    # Execute trajectory + HOME hold
    # ========================================================

    def execute_home_motion(
        self,
        q_start: np.ndarray,
        q_target: np.ndarray,
    ) -> int:

        if (
            self.active_duration
            is None
        ):

            raise RuntimeError(
                "active_duration has not been resolved"
            )

        T = float(
            self.active_duration
        )

        start = (
            time.monotonic()
        )

        next_tick = start
        next_print = start

        print()
        print(
            "=" * 100
        )

        print(
            "AUTO HOME MOTION START"
        )

        print(
            "=" * 100
        )

        print(
            f"Duration : "
            f"{T:.3f} s"
        )

        print(
            f"Rate     : "
            f"{self.args.hz:.1f} Hz"
        )

        print()

        # ====================================================
        # Quintic trajectory
        # ====================================================

        while True:

            now = (
                time.monotonic()
            )

            elapsed = (
                now - start
            )

            if (
                elapsed
                >= T
            ):
                break

            if (
                self.state_age()
                > self.args.state_timeout
            ):

                raise RuntimeError(
                    "LOWSTATE WATCHDOG TIMEOUT "
                    "during trajectory"
                )

            q_measured, dq_measured = (
                self.arm_qdq()
            )

            u = float(
                np.clip(
                    elapsed / T,
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

            # ------------------------------------------------
            # Command-lead limiter
            #
            # The command is never allowed to be farther
            # than max_command_lead from the measured q.
            # ------------------------------------------------

            lead = np.clip(
                q_plan
                - q_measured,
                -self.args.max_command_lead,
                self.args.max_command_lead,
            )

            q_command = (
                q_measured
                + lead
            )

            self.publish_arm(
                q_command
            )

            if (
                now
                >= next_print
            ):

                next_print = (
                    now
                    + self.args.print_period
                )

                target_error = float(
                    np.max(
                        np.abs(
                            q_target
                            - q_measured
                        )
                    )
                )

                cmd_lead = float(
                    np.max(
                        np.abs(
                            q_command
                            - q_measured
                        )
                    )
                )

                tau = (
                    self.arm_tau_est()
                )

                print(
                    f"[motion] "
                    f"t="
                    f"{elapsed:5.2f}/"
                    f"{T:.2f} s "
                    f"target_err="
                    f"{target_error:.6f} "
                    f"cmd_lead="
                    f"{cmd_lead:.6f} "
                    f"max|dq|="
                    f"{np.max(np.abs(dq_measured)):.6f} "
                    f"max|tau_est|="
                    f"{np.max(np.abs(tau)):.6f}"
                )

                # Main diagnostic joints.
                for idx in [
                    0,   # left shoulder pitch
                    2,   # left shoulder yaw
                    5,   # left wrist pitch
                    7,   # right shoulder pitch
                    9,   # right shoulder yaw
                    10,  # right elbow
                    12,  # right wrist pitch
                ]:

                    print(
                        f"    "
                        f"{ARM_NAMES[idx]:24s} "
                        f"q="
                        f"{q_measured[idx]:+.5f} "
                        f"plan="
                        f"{q_plan[idx]:+.5f} "
                        f"cmd="
                        f"{q_command[idx]:+.5f} "
                        f"err="
                        f"{q_target[idx] - q_measured[idx]:+.5f}"
                    )

            next_tick += (
                1.0
                / self.args.hz
            )

            sleep_time = (
                next_tick
                - time.monotonic()
            )

            if (
                sleep_time
                > 0.0
            ):

                time.sleep(
                    sleep_time
                )

        # ====================================================
        # HOME hold
        # ====================================================

        print()
        print(
            "Trajectory complete; holding HOME..."
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

        best_error = float(
            "inf"
        )

        stable_start: Optional[
            float
        ] = None

        stable_reached = False

        while (
            time.monotonic()
            - hold_start
            < self.args.hold_time
        ):

            if (
                self.state_age()
                > self.args.state_timeout
            ):

                raise RuntimeError(
                    "LOWSTATE WATCHDOG TIMEOUT "
                    "during HOME hold"
                )

            now = (
                time.monotonic()
            )

            q_measured, dq_measured = (
                self.arm_qdq()
            )

            error = (
                q_target
                - q_measured
            )

            max_error = float(
                np.max(
                    np.abs(
                        error
                    )
                )
            )

            best_error = min(
                best_error,
                max_error,
            )

            # ------------------------------------------------
            # HOME stable-time detector
            # ------------------------------------------------

            if (
                max_error
                < self.args.home_threshold
            ):

                if (
                    stable_start
                    is None
                ):

                    stable_start = (
                        now
                    )

                if (
                    now
                    - stable_start
                    >= self.args.home_stable_time
                ):

                    stable_reached = (
                        True
                    )

            else:

                stable_start = (
                    None
                )

            lead = np.clip(
                error,
                -self.args.max_command_lead,
                self.args.max_command_lead,
            )

            q_command = (
                q_measured
                + lead
            )

            self.publish_arm(
                q_command
            )

            if (
                now
                >= next_print
            ):

                next_print = (
                    now
                    + self.args.print_period
                )

                stable_elapsed = (
                    0.0
                    if stable_start is None
                    else (
                        now
                        - stable_start
                    )
                )

                print(
                    f"[home_hold] "
                    f"t="
                    f"{now - hold_start:5.2f}/"
                    f"{self.args.hold_time:.2f} s "
                    f"max_err="
                    f"{max_error:.6f} rad "
                    f"best="
                    f"{best_error:.6f} rad "
                    f"stable="
                    f"{stable_elapsed:.3f}/"
                    f"{self.args.home_stable_time:.3f} s "
                    f"max|dq|="
                    f"{np.max(np.abs(dq_measured)):.6f}"
                )

            next_tick += (
                1.0
                / self.args.hz
            )

            sleep_time = (
                next_tick
                - time.monotonic()
            )

            if (
                sleep_time
                > 0.0
            ):

                time.sleep(
                    sleep_time
                )

        # ====================================================
        # Final HOME result
        # ====================================================

        q_final, dq_final = (
            self.arm_qdq()
        )

        error = (
            q_final
            - q_target
        )

        abs_error = (
            np.abs(
                error
            )
        )

        worst_idx = int(
            np.argmax(
                abs_error
            )
        )

        max_error = float(
            abs_error[
                worst_idx
            ]
        )

        mean_abs = float(
            np.mean(
                abs_error
            )
        )

        rms = float(
            np.sqrt(
                np.mean(
                    error**2
                )
            )
        )

        final_threshold_pass = (
            max_error
            < self.args.home_threshold
        )

        # Formal HOME acceptance:
        #
        # 1. Final pose must still be inside tolerance.
        # 2. It must have continuously remained inside
        #    tolerance for home_stable_time.
        home_ready = (
            final_threshold_pass
            and
            stable_reached
        )

        print()
        print(
            "=" * 100
        )

        print(
            "AUTO HOME RESULT"
        )

        print(
            "=" * 100
        )

        print(
            f"MAX error : "
            f"{max_error:.6f} rad "
            f"({np.degrees(max_error):.3f} deg)"
        )

        print(
            f"Best hold : "
            f"{best_error:.6f} rad "
            f"({np.degrees(best_error):.3f} deg)"
        )

        print(
            f"Mean abs  : "
            f"{mean_abs:.6f} rad "
            f"({np.degrees(mean_abs):.3f} deg)"
        )

        print(
            f"RMS error : "
            f"{rms:.6f} rad "
            f"({np.degrees(rms):.3f} deg)"
        )

        print(
            f"Worst     : "
            f"{ARM_NAMES[worst_idx]}"
        )

        print(
            f"Final max |dq| : "
            f"{np.max(np.abs(dq_final)):.6f} rad/s"
        )

        print(
            f"Final threshold pass : "
            f"{final_threshold_pass}"
        )

        print(
            f"Stable HOME reached  : "
            f"{stable_reached}"
        )

        print()

        print(
            f"{'joint':28s}"
            f"{'target':>12s}"
            f"{'final':>12s}"
            f"{'error':>12s}"
            f"{'deg':>10s}"
        )

        print(
            "-" * 76
        )

        for i, name in enumerate(
            ARM_NAMES
        ):

            print(
                f"{name:28s}"
                f"{q_target[i]:+12.6f}"
                f"{q_final[i]:+12.6f}"
                f"{error[i]:+12.6f}"
                f"{np.degrees(error[i]):+10.3f}"
            )

        print(
            "-" * 76
        )

        if (
            home_ready
        ):

            print(
                "AUTO_HOME_READY = PASS"
            )

            result = 0

        else:

            print(
                "AUTO_HOME_READY = FAIL"
            )

            result = 3

        print(
            "=" * 100
        )

        print()

        return (
            result
        )

    # ========================================================
    # Real execution sequence
    # ========================================================

    def execute(
        self,
        q_target: np.ndarray,
    ) -> int:

        # ----------------------------------------------------
        # Recheck state before publisher is created.
        # ----------------------------------------------------

        self.wait_for_state(
            timeout=2.0
        )

        if (
            not self.preflight(
                q_target,
                verbose=False,
            )
        ):

            raise RuntimeError(
                "Pre-flight failed immediately "
                "before execution"
            )

        self.capture_hold_positions()

        self.create_publisher()

        if (
            self.active_duration
            is None
        ):

            raise RuntimeError(
                "active_duration has not been resolved"
            )

        print()
        print(
            "=" * 100
        )

        print(
            "REAL ROBOT LOWCMD CONTROL IS ARMED"
        )

        print(
            "=" * 100
        )

        print(
            f"Keyframe       : "
            f"{self.args.keyframe}"
        )

        print(
            f"Duration mode  : "
            f"{'AUTO' if self.args.duration is None else 'MANUAL'}"
        )

        print(
            f"Duration       : "
            f"{self.active_duration:.3f} s"
        )

        print(
            f"Waist hold q   : "
            f"{self.waist_hold_q:+.6f}"
        )

        print(
            f"Left grip hold : "
            f"{self.gripper_hold_q[0]:+.6f}"
        )

        print(
            f"Right grip hold: "
            f"{self.gripper_hold_q[1]:+.6f}"
        )

        print()

        if (
            self.args.takeover_only
        ):

            confirmation_text = (
                "TAKEOVER"
            )

            print(
                "TAKEOVER-ONLY MODE:"
            )

            print(
                "The robot will only hold its CURRENT "
                "measured arm pose."
            )

            print(
                "It will NOT move toward HOME."
            )

        else:

            confirmation_text = (
                "MOVE HOME"
            )

            print(
                "HOME MOTION MODE:"
            )

            print(
                "The robot will first perform a "
                "zero-displacement takeover test,"
            )

            print(
                "then move to HOME if takeover succeeds."
            )

        print()

        print(
            f"Type {confirmation_text} exactly "
            f"to continue."
        )

        phrase = input(
            "> "
        )

        if (
            phrase
            != confirmation_text
        ):

            print(
                "Confirmation rejected."
            )

            print(
                "No robot motion command published."
            )

            return 2

        # ----------------------------------------------------
        # Recheck again after operator confirmation.
        # ----------------------------------------------------

        self.wait_for_state(
            timeout=2.0
        )

        if (
            not self.preflight(
                q_target,
                verbose=False,
            )
        ):

            raise RuntimeError(
                "Pre-flight failed after confirmation"
            )

        self.capture_hold_positions()

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

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Every return after non-zero LowCmd gains begin is
        # inside this try/finally.
        #
        # Therefore gains are always released.
        # ----------------------------------------------------

        try:

            # =================================================
            # Stage 1
            # Zero-displacement LowCmd takeover
            # =================================================

            (
                takeover_ok,
                q_after_takeover,
            ) = self.takeover_test()

            if (
                not takeover_ok
            ):

                print(
                    "ABORT: takeover test failed."
                )

                return 4

            if (
                self.args.takeover_only
            ):

                print(
                    "TAKEOVER_ONLY = PASS"
                )

                print(
                    "HOME trajectory was NOT executed."
                )

                return 0

            # =================================================
            # Stage 2
            # AUTO duration recheck after REAL takeover
            # =================================================

            if (
                self.args.duration
                is None
            ):

                (
                    duration,
                    duration_info,
                ) = self.calculate_safe_duration(
                    q_after_takeover,
                    q_target,
                )

                self.active_duration = (
                    duration
                )

                print()
                print(
                    "=" * 100
                )

                print(
                    "AUTO DURATION RECHECK AFTER TAKEOVER"
                )

                print(
                    "=" * 100
                )

                print(
                    f"Max joint delta       : "
                    f"{duration_info['max_delta']:.6f} rad"
                )

                print(
                    f"Velocity min duration : "
                    f"{duration_info['velocity_min_duration']:.3f} s"
                )

                print(
                    f"Accel min duration    : "
                    f"{duration_info['acceleration_min_duration']:.3f} s"
                )

                print(
                    f"Theoretical min       : "
                    f"{duration_info['theoretical_min_duration']:.3f} s"
                )

                print(
                    f"Safety margin         : "
                    f"{duration_info['duration_margin']:.3f}"
                )

                print(
                    f"Selected duration     : "
                    f"{self.active_duration:.3f} s"
                )

                print(
                    "=" * 100
                )

                print()

            if (
                self.active_duration
                is None
            ):

                raise RuntimeError(
                    "active_duration has not been resolved"
                )

            # =================================================
            # Stage 3
            # Post-takeover trajectory safety check
            # =================================================

            metrics = (
                self.trajectory_metrics(
                    q_after_takeover,
                    q_target,
                    self.active_duration,
                )
            )

            if (
                metrics[
                    "max_delta"
                ]
                > self.args.max_initial_delta
            ):

                raise RuntimeError(
                    "Post-takeover trajectory exceeds "
                    "maximum permitted joint displacement"
                )

            if (
                metrics[
                    "peak_velocity"
                ]
                > self.args.max_velocity
            ):

                raise RuntimeError(
                    "Post-takeover trajectory exceeds "
                    "velocity limit"
                )

            if (
                metrics[
                    "peak_acceleration"
                ]
                > self.args.max_acceleration
            ):

                raise RuntimeError(
                    "Post-takeover trajectory exceeds "
                    "acceleration limit"
                )

            if (
                metrics[
                    "max_step"
                ]
                > self.args.max_step
            ):

                raise RuntimeError(
                    "Post-takeover trajectory exceeds "
                    "step limit"
                )

            print(
                "POST_TAKEOVER_TRAJECTORY_CHECK = PASS"
            )

            print(
                f"Final motion duration : "
                f"{self.active_duration:.3f} s"
            )

            # =================================================
            # Stage 4
            # Real HOME motion
            # =================================================

            return (
                self.execute_home_motion(
                    q_after_takeover,
                    q_target,
                )
            )

        finally:

            self.release()

            print(
                "LowCmd gains released."
            )


# ============================================================
# Arguments
# ============================================================

def build_parser():

    p = argparse.ArgumentParser(
        description=(
            "R1-A7 conservative automatic "
            "keyframe mover with automatic "
            "safe-duration calculation"
        )
    )

    p.add_argument(
        "keyframe",
        nargs="?",
        default="HOME",
    )

    # ========================================================
    # DDS
    # ========================================================

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

    # Use path relative to this project,
    # independent of current working directory.
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
    # AUTO / MANUAL trajectory duration
    # ========================================================

    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Manual trajectory duration in seconds. "
            "If omitted, duration is calculated "
            "automatically."
        ),
    )

    p.add_argument(
        "--min-auto-duration",
        type=float,
        default=8.0,
        help=(
            "Minimum automatically selected "
            "trajectory duration."
        ),
    )

    p.add_argument(
        "--duration-margin",
        type=float,
        default=1.10,
        help=(
            "Safety multiplier applied to the "
            "theoretical minimum duration."
        ),
    )

    p.add_argument(
        "--duration-step",
        type=float,
        default=0.5,
        help=(
            "AUTO duration is rounded upward "
            "to this time step."
        ),
    )

    p.add_argument(
        "--max-auto-duration",
        type=float,
        default=20.0,
        help=(
            "Abort if automatically calculated "
            "duration exceeds this value."
        ),
    )

    # ========================================================
    # Control frequency / safety
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

    # Keep the already verified conservative speed limit.
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
        "--max-initial-delta",
        type=float,
        default=0.35,
    )

    p.add_argument(
        "--joint-limit-margin",
        type=float,
        default=0.04,
    )

    # ========================================================
    # Arm gains
    # ========================================================

    # Tuned shoulder pitch.
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

    # Shoulder roll / yaw / elbow.
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

    # Wrist roll / yaw.
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

    # Tuned wrist pitch.
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

    # ========================================================
    # Waist / auxiliary hold
    # ========================================================

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

    # ========================================================
    # Gripper hold
    # ========================================================

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
    # LowCmd takeover safety
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

    p.add_argument(
        "--takeover-only",
        action="store_true",
        help=(
            "Acquire LowCmd and hold the current "
            "measured pose only. "
            "Do not execute HOME motion."
        ),
    )

    # ========================================================
    # HOME validation
    # ========================================================

    p.add_argument(
        "--hold-time",
        type=float,
        default=10.0,
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

    # ========================================================
    # Real motion gate
    # ========================================================

    p.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually publish rt/lowcmd. "
            "Without this flag the program runs "
            "PRE-FLIGHT ONLY."
        ),
    )

    return p


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        build_parser()
    )

    args = (
        parser.parse_args()
    )

    args.keyframe = (
        args.keyframe.upper()
    )

    # ========================================================
    # Argument validation
    # ========================================================

    if (
        args.hz
        <= 0.0
    ):

        raise RuntimeError(
            "--hz must be > 0"
        )

    if (
        args.duration
        is not None
        and
        args.duration
        <= 0.0
    ):

        raise RuntimeError(
            "--duration must be > 0"
        )

    if (
        args.min_auto_duration
        <= 0.0
    ):

        raise RuntimeError(
            "--min-auto-duration must be > 0"
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
        <= 0.0
    ):

        raise RuntimeError(
            "--duration-step must be > 0"
        )

    if (
        args.max_auto_duration
        < args.min_auto_duration
    ):

        raise RuntimeError(
            "--max-auto-duration must be >= "
            "--min-auto-duration"
        )

    if (
        args.state_timeout
        <= 0.0
    ):

        raise RuntimeError(
            "--state-timeout must be > 0"
        )

    if (
        args.max_velocity
        <= 0.0
    ):

        raise RuntimeError(
            "--max-velocity must be > 0"
        )

    if (
        args.max_acceleration
        <= 0.0
    ):

        raise RuntimeError(
            "--max-acceleration must be > 0"
        )

    if (
        args.max_step
        <= 0.0
    ):

        raise RuntimeError(
            "--max-step must be > 0"
        )

    if (
        args.max_command_lead
        <= 0.0
    ):

        raise RuntimeError(
            "--max-command-lead must be > 0"
        )

    if (
        args.max_start_dq
        <= 0.0
    ):

        raise RuntimeError(
            "--max-start-dq must be > 0"
        )

    if (
        args.max_initial_delta
        <= 0.0
    ):

        raise RuntimeError(
            "--max-initial-delta must be > 0"
        )

    if (
        args.takeover_time
        <= 0.0
    ):

        raise RuntimeError(
            "--takeover-time must be > 0"
        )

    if (
        args.max_takeover_drift
        <= 0.0
    ):

        raise RuntimeError(
            "--max-takeover-drift must be > 0"
        )

    if (
        args.hold_time
        <= 0.0
    ):

        raise RuntimeError(
            "--hold-time must be > 0"
        )

    if (
        args.home_threshold
        <= 0.0
    ):

        raise RuntimeError(
            "--home-threshold must be > 0"
        )

    if (
        args.home_stable_time
        <= 0.0
    ):

        raise RuntimeError(
            "--home-stable-time must be > 0"
        )

    if (
        args.print_period
        <= 0.0
    ):

        raise RuntimeError(
            "--print-period must be > 0"
        )

    # ========================================================
    # Load target keyframe
    # ========================================================

    q_target = (
        load_keyframe(
            args.keyframes,
            args.keyframe,
        )
    )

    # ========================================================
    # Controller
    # ========================================================

    mover = (
        AutoKeyframeMover(
            args
        )
    )

    mover.connect_state_only()

    # ========================================================
    # Initial PRE-FLIGHT
    # ========================================================

    passed = (
        mover.preflight(
            q_target,
            verbose=True,
        )
    )

    if (
        not passed
    ):

        print(
            "ABORT: pre-flight failed."
        )

        return 2

    # ========================================================
    # PRE-FLIGHT only
    # ========================================================

    if (
        not args.execute
    ):

        print(
            "PRE-FLIGHT ONLY."
        )

        print(
            "Robot was NOT commanded."
        )

        return 0

    # ========================================================
    # Real robot execution
    # ========================================================

    return (
        mover.execute(
            q_target
        )
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )