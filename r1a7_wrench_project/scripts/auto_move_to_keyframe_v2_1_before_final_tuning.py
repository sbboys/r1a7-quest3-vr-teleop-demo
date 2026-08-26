#!/usr/bin/env python3
from __future__ import annotations

# AUTO_HOME V2.1 candidate
# Derived from the frozen/verified V2 without changing HOME, gains,
# trajectory limits, recovery thresholds, ARM recovery, or FINAL_HOME.
# V2.1 changes:
#   1) default max WRIST iterations: 8 -> 12
#   2) WRIST-only consecutive low-progress watchdog:
#      progress < 0.005 rad for 2 consecutive iterations -> FAIL
#
# IMPORTANT: keep the original auto_move_to_keyframe_v2_verified.py frozen.


import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
# Joint mapping
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

    data = load_yaml(
        path
    )

    if not isinstance(
        data,
        dict,
    ):
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

    kf = data[
        "keyframes"
    ][name]

    q = np.asarray(
        kf["left_joint_position"]
        + kf["right_joint_position"],
        dtype=float,
    )

    if q.shape != (14,):
        raise RuntimeError(
            "Keyframe must contain 14 arm joints"
        )

    if not np.all(
        np.isfinite(q)
    ):
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
# Controller
# ============================================================

class AutoKeyframeMoverV21:

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
            return float(
                "inf"
            )

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
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        if self.low_state is None:
            raise RuntimeError(
                "No lowstate"
            )

        ms = (
            self.low_state.motor_state
        )

        q = np.asarray(
            [
                float(
                    ms[i].q
                )
                for i in ARM_INDICES
            ],
            dtype=float,
        )

        dq = np.asarray(
            [
                float(
                    ms[i].dq
                )
                for i in ARM_INDICES
            ],
            dtype=float,
        )

        return q, dq

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
            and
            active_name == ""
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
    # Gains
    # ========================================================

    def arm_gain(
        self,
        joint_i: int,
    ):

        local_i = (
            joint_i % 7
        )

        if local_i == 0:

            return (
                self.args.kp_shoulder_pitch,
                self.args.kd_shoulder_pitch,
            )

        if local_i < 4:

            return (
                self.args.kp_low,
                self.args.kd_low,
            )

        if local_i == 5:

            return (
                self.args.kp_wrist_pitch,
                self.args.kd_wrist_pitch,
            )

        return (
            self.args.kp_wrist,
            self.args.kd_wrist,
        )

    # ========================================================
    # Duration / trajectory
    # ========================================================

    def calculate_safe_duration(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        min_duration: Optional[
            float
        ] = None,
    ) -> float:

        delta = np.abs(
            q1 - q0
        )

        max_delta = float(
            np.max(delta)
        )

        if max_delta <= 1e-12:

            if min_duration is not None:
                return float(
                    min_duration
                )

            return float(
                self.args.min_auto_duration
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
            float(
                min_duration
            ),
            required,
        )

        step = float(
            self.args.duration_step
        )

        duration = float(
            np.ceil(
                duration
                / step
            )
            * step
        )

        if (
            duration
            > self.args.max_auto_duration
        ):

            raise RuntimeError(
                f"AUTO duration "
                f"{duration:.2f}s exceeds "
                f"{self.args.max_auto_duration:.2f}s"
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
                q1
                - q0
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
            "max_step": (
                max_step
            ),
            "samples": (
                n
            ),
        }

    # ========================================================
    # Group error
    # ========================================================

    @staticmethod
    def group_error(
        q: np.ndarray,
        q_ref: np.ndarray,
        indices: list[int],
    ):

        err = (
            q[indices]
            - q_ref[indices]
        )

        abs_err = np.abs(
            err
        )

        local_argmax = int(
            np.argmax(
                abs_err
            )
        )

        worst_i = (
            indices[
                local_argmax
            ]
        )

        return (
            float(
                abs_err[
                    local_argmax
                ]
            ),
            worst_i,
            float(
                err[
                    local_argmax
                ]
            ),
        )

    # ========================================================
    # V2 adaptive target
    # ========================================================

    def make_adaptive_target(
        self,
        q_current: np.ndarray,
        q_home: np.ndarray,
        indices: list[int],
        stage_delta: float,
    ) -> np.ndarray:

        target = (
            q_current.copy()
        )

        remaining = (
            q_home[indices]
            - q_current[indices]
        )

        step = np.clip(
            remaining,
            -stage_delta,
            stage_delta,
        )

        target[indices] = (
            q_current[indices]
            + step
        )

        return target

    def estimate_nominal_iterations(
        self,
        q_current: np.ndarray,
        q_home: np.ndarray,
        indices: list[int],
        stage_delta: float,
    ) -> int:

        max_error, _, _ = (
            self.group_error(
                q_current,
                q_home,
                indices,
            )
        )

        if (
            max_error
            <= self.args.recovery_threshold
        ):
            return 0

        return int(
            np.ceil(
                max_error
                / stage_delta
            )
        )

    # ========================================================
    # PRE-FLIGHT
    # ========================================================

    def preflight(
        self,
        q_home: np.ndarray,
        verbose: bool = True,
    ) -> bool:

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

        global_error = float(
            np.max(
                np.abs(
                    q_home
                    - q_current
                )
            )
        )

        (
            wrist_error,
            wrist_i,
            _,
        ) = self.group_error(
            q_current,
            q_home,
            WRIST_LOCAL_INDICES,
        )

        (
            arm_error,
            arm_i,
            _,
        ) = self.group_error(
            q_current,
            q_home,
            ARM_CORE_LOCAL_INDICES,
        )

        direct_mode = (
            global_error
            <= self.args.direct_home_delta
        )

        wrist_nominal = (
            self.estimate_nominal_iterations(
                q_current,
                q_home,
                WRIST_LOCAL_INDICES,
                self.args.wrist_stage_delta,
            )
        )

        arm_nominal = (
            self.estimate_nominal_iterations(
                q_current,
                q_home,
                ARM_CORE_LOCAL_INDICES,
                self.args.arm_stage_delta,
            )
        )

        iteration_budget_ok = True

        if not direct_mode:

            if (
                wrist_nominal
                > self.args.max_wrist_iterations
            ):
                iteration_budget_ok = False

            if (
                arm_nominal
                > self.args.max_arm_iterations
            ):
                iteration_budget_ok = False

        first_target_ok = True

        preview_rows = []

        if direct_mode:

            duration = (
                self.calculate_safe_duration(
                    q_current,
                    q_home,
                    min_duration=(
                        self.args.min_auto_duration
                    ),
                )
            )

            metrics = (
                self.trajectory_metrics(
                    q_current,
                    q_home,
                    duration,
                )
            )

            preview_rows.append(
                (
                    "FINAL_HOME",
                    metrics,
                    duration,
                )
            )

            first_target_ok = (
                metrics["peak_velocity"]
                <= self.args.max_velocity
                and
                metrics["peak_acceleration"]
                <= self.args.max_acceleration
                and
                metrics["max_step"]
                <= self.args.max_step
            )

        else:

            q_preview = (
                q_current.copy()
            )

            if wrist_nominal > 0:

                wrist_target = (
                    self.make_adaptive_target(
                        q_preview,
                        q_home,
                        WRIST_LOCAL_INDICES,
                        self.args.wrist_stage_delta,
                    )
                )

                (
                    limit_ok,
                    _,
                    _,
                    _,
                ) = self.check_joint_limits(
                    wrist_target
                )

                duration = (
                    self.calculate_safe_duration(
                        q_preview,
                        wrist_target,
                        min_duration=(
                            self.args.recovery_min_duration
                        ),
                    )
                )

                metrics = (
                    self.trajectory_metrics(
                        q_preview,
                        wrist_target,
                        duration,
                    )
                )

                preview_rows.append(
                    (
                        "WRIST_ITER_01",
                        metrics,
                        duration,
                    )
                )

                first_target_ok = (
                    first_target_ok
                    and limit_ok
                    and metrics["peak_velocity"]
                    <= self.args.max_velocity
                    and metrics["peak_acceleration"]
                    <= self.args.max_acceleration
                    and metrics["max_step"]
                    <= self.args.max_step
                )

            if arm_nominal > 0:

                arm_target = (
                    self.make_adaptive_target(
                        q_preview,
                        q_home,
                        ARM_CORE_LOCAL_INDICES,
                        self.args.arm_stage_delta,
                    )
                )

                (
                    limit_ok,
                    _,
                    _,
                    _,
                ) = self.check_joint_limits(
                    arm_target
                )

                duration = (
                    self.calculate_safe_duration(
                        q_preview,
                        arm_target,
                        min_duration=(
                            self.args.recovery_min_duration
                        ),
                    )
                )

                metrics = (
                    self.trajectory_metrics(
                        q_preview,
                        arm_target,
                        duration,
                    )
                )

                preview_rows.append(
                    (
                        "ARM_ITER_01",
                        metrics,
                        duration,
                    )
                )

                first_target_ok = (
                    first_target_ok
                    and limit_ok
                    and metrics["peak_velocity"]
                    <= self.args.max_velocity
                    and metrics["peak_acceleration"]
                    <= self.args.max_acceleration
                    and metrics["max_step"]
                    <= self.args.max_step
                )

        passed = all(
            [
                state_ok,
                velocity_ok,
                current_limit_ok,
                target_limit_ok,
                motion_ok,
                iteration_budget_ok,
                first_target_ok,
            ]
        )

        if verbose:

            print()
            print(
                "=" * 104
            )

            print(
                "R1-A7 AUTO-HOME V2.1 ADAPTIVE RECOVERY PRE-FLIGHT"
            )

            print(
                "=" * 104
            )

            print(
                f"Lowstate age             : "
                f"{state_age:.6f}s "
                f"[{'PASS' if state_ok else 'FAIL'}]"
            )

            print(
                f"Max initial |dq|         : "
                f"{max_dq:.6f} rad/s "
                f"[{'PASS' if velocity_ok else 'FAIL'}]"
            )

            print(
                f"Current joint limits     : "
                f"{'PASS' if current_limit_ok else 'FAIL'}"
            )

            print(
                f"HOME joint limits        : "
                f"{'PASS' if target_limit_ok else 'FAIL'}"
            )

            print(
                f"MotionSwitcher           : "
                f"{'PASS' if motion_ok else 'FAIL'}"
            )

            print(
                f"  status                 : "
                f"{motion_status}"
            )

            print(
                f"  result                 : "
                f"{motion_result}"
            )

            print()

            print(
                f"Global HOME error        : "
                f"{global_error:.6f} rad"
            )

            print(
                f"Wrist HOME error         : "
                f"{wrist_error:.6f} rad "
                f"({ARM_NAMES[wrist_i]})"
            )

            print(
                f"Arm HOME error           : "
                f"{arm_error:.6f} rad "
                f"({ARM_NAMES[arm_i]})"
            )

            print(
                f"Direct HOME threshold    : "
                f"{self.args.direct_home_delta:.6f} rad"
            )

            print(
                f"Recovery mode            : "
                f"{'DIRECT HOME' if direct_mode else 'ADAPTIVE RECOVERY'}"
            )

            print()

            if not direct_mode:

                print(
                    f"Wrist stage delta        : "
                    f"{self.args.wrist_stage_delta:.6f} rad"
                )

                print(
                    f"Arm stage delta          : "
                    f"{self.args.arm_stage_delta:.6f} rad"
                )

                print(
                    f"Recovery threshold       : "
                    f"{self.args.recovery_threshold:.6f} rad"
                )

                print(
                    f"Nominal wrist iterations : "
                    f"{wrist_nominal} / "
                    f"max {self.args.max_wrist_iterations}"
                )

                print(
                    f"Nominal arm iterations   : "
                    f"{arm_nominal} / "
                    f"max {self.args.max_arm_iterations}"
                )

                print(
                    f"Iteration budget         : "
                    f"{'PASS' if iteration_budget_ok else 'FAIL'}"
                )

            print()

            print(
                "First-step trajectory preview:"
            )

            for (
                name,
                metrics,
                duration,
            ) in preview_rows:

                print(
                    f"  {name:18s} "
                    f"duration={duration:5.2f}s "
                    f"max_delta={metrics['max_delta']:.6f} "
                    f"peak_v={metrics['peak_velocity']:.6f} "
                    f"peak_a={metrics['peak_acceleration']:.6f} "
                    f"max_step={metrics['max_step']:.6f}"
                )

            print(
                f"First-step checks        : "
                f"{'PASS' if first_target_ok else 'FAIL'}"
            )

            print()

            print(
                "Arm gains:"
            )

            print(
                f"  shoulder pitch         : "
                f"kp={self.args.kp_shoulder_pitch}, "
                f"kd={self.args.kd_shoulder_pitch}"
            )

            print(
                f"  shoulder roll/yaw/elbow: "
                f"kp={self.args.kp_low}, "
                f"kd={self.args.kd_low}"
            )

            print(
                f"  wrist pitch            : "
                f"kp={self.args.kp_wrist_pitch}, "
                f"kd={self.args.kd_wrist_pitch}"
            )

            print(
                f"  wrist roll/yaw         : "
                f"kp={self.args.kp_wrist}, "
                f"kd={self.args.kd_wrist}"
            )

            print()

            print(
                f"PRE_FLIGHT = "
                f"{'PASS' if passed else 'FAIL'}"
            )

            print(
                "ROBOT_COMMAND_PUBLISHED = FALSE"
            )

            print(
                "=" * 104
            )

        return passed

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
            self.low_state is not None
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

        # Waist
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

        # Auxiliary motors
        for idx in (
            AUX_HOLD_INDICES
        ):

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

        # Arms
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

        # Grippers
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
                "Release warning:",
                repr(exc),
            )

    # ========================================================
    # Takeover
    # ========================================================

    def takeover_test(
        self,
    ) -> np.ndarray:

        q_takeover, _ = (
            self.arm_qdq()
        )

        start = (
            time.monotonic()
        )

        next_tick = (
            start
        )

        max_drift = 0.0

        print()
        print(
            "=" * 104
        )

        print(
            "LOWCMD ZERO-DISPLACEMENT TAKEOVER TEST"
        )

        print(
            "=" * 104
        )

        while (
            time.monotonic()
            - start
            < self.args.takeover_time
        ):

            q, _ = (
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
                1.0
                / self.args.hz
            )

            dt = (
                next_tick
                - time.monotonic()
            )

            if dt > 0:

                time.sleep(
                    dt
                )

        q_after, dq_after = (
            self.arm_qdq()
        )

        print(
            f"Takeover max drift : "
            f"{max_drift:.6f} rad"
        )

        print(
            f"Final max |dq|     : "
            f"{np.max(np.abs(dq_after)):.6f} rad/s"
        )

        print(
            "TAKEOVER_TEST = PASS"
        )

        print(
            "=" * 104
        )

        return (
            q_after.copy()
        )

    # ========================================================
    # Generic trajectory
    # ========================================================

    def run_trajectory(
        self,
        name: str,
        q_target: np.ndarray,
        min_duration: float,
    ):

        q_start, _ = (
            self.arm_qdq()
        )

        duration = (
            self.calculate_safe_duration(
                q_start,
                q_target,
                min_duration=(
                    min_duration
                ),
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
            "=" * 104
        )

        print(
            f"TRAJECTORY START: {name}"
        )

        print(
            "=" * 104
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

        start = (
            time.monotonic()
        )

        next_tick = (
            start
        )

        next_print = (
            start
        )

        while True:

            now = (
                time.monotonic()
            )

            elapsed = (
                now
                - start
            )

            if (
                elapsed
                >= duration
            ):
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

                target_error = float(
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
                    f"{target_error:.6f} "
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

                time.sleep(
                    dt
                )

        return (
            duration,
            metrics,
        )

    # ========================================================
    # Hold and group-to-HOME verification
    # ========================================================

    def hold_and_check_group_home(
        self,
        name: str,
        q_target: np.ndarray,
        q_home: np.ndarray,
        active_indices: list[int],
        inactive_indices: list[int],
    ):

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

        best_home_error = float(
            "inf"
        )

        q_reference_inactive = (
            q_target.copy()
        )

        while (
            time.monotonic()
            - hold_start
            < self.args.recovery_hold_time
        ):

            now = (
                time.monotonic()
            )

            q, dq = (
                self.arm_qdq()
            )

            (
                active_home_error,
                active_i,
                active_signed,
            ) = self.group_error(
                q,
                q_home,
                active_indices,
            )

            (
                inactive_hold_error,
                inactive_i,
                inactive_signed,
            ) = self.group_error(
                q,
                q_reference_inactive,
                inactive_indices,
            )

            (
                target_error,
                target_i,
                target_signed,
            ) = self.group_error(
                q,
                q_target,
                active_indices,
            )

            best_home_error = min(
                best_home_error,
                active_home_error,
            )

            if (
                active_home_error
                < self.args.recovery_threshold
            ):

                if stable_start is None:

                    stable_start = (
                        now
                    )

                if (
                    now
                    - stable_start
                    >= self.args.recovery_stable_time
                ):

                    stable_reached = True

            else:

                stable_start = None

            # Hold current micro-target.
            lead = np.clip(
                q_target
                - q,
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
                    now
                    - stable_start
                )

                print(
                    f"[{name}_hold] "
                    f"active_home="
                    f"{active_home_error:.6f} "
                    f"active_target="
                    f"{target_error:.6f} "
                    f"inactive_hold="
                    f"{inactive_hold_error:.6f} "
                    f"stable="
                    f"{stable_time:.2f}/"
                    f"{self.args.recovery_stable_time:.2f}"
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

                time.sleep(
                    dt
                )

        q_final, dq_final = (
            self.arm_qdq()
        )

        (
            active_home_error,
            active_i,
            active_signed,
        ) = self.group_error(
            q_final,
            q_home,
            active_indices,
        )

        (
            active_target_error,
            target_i,
            target_signed,
        ) = self.group_error(
            q_final,
            q_target,
            active_indices,
        )

        (
            inactive_hold_error,
            inactive_i,
            inactive_signed,
        ) = self.group_error(
            q_final,
            q_reference_inactive,
            inactive_indices,
        )

        return {
            "q_final": (
                q_final
            ),
            "dq_final": (
                dq_final
            ),

            "active_home_error": (
                active_home_error
            ),
            "active_home_worst_i": (
                active_i
            ),
            "active_home_signed": (
                active_signed
            ),

            "active_target_error": (
                active_target_error
            ),
            "active_target_worst_i": (
                target_i
            ),
            "active_target_signed": (
                target_signed
            ),

            "inactive_hold_error": (
                inactive_hold_error
            ),
            "inactive_hold_worst_i": (
                inactive_i
            ),
            "inactive_hold_signed": (
                inactive_signed
            ),

            "best_home_error": (
                best_home_error
            ),
            "stable_reached": (
                stable_reached
            ),
        }

    # ========================================================
    # Adaptive Wrist / Arm Recovery
    # ========================================================

    def adaptive_group_recovery(
        self,
        group_name: str,
        active_indices: list[int],
        inactive_indices: list[int],
        q_home: np.ndarray,
        stage_delta: float,
        max_iterations: int,
    ) -> bool:

        print()
        print(
            "=" * 104
        )

        print(
            f"ADAPTIVE {group_name} RECOVERY"
        )

        print(
            "=" * 104
        )

        print(
            f"Stage delta       : "
            f"{stage_delta:.6f} rad"
        )

        print(
            f"HOME threshold    : "
            f"{self.args.recovery_threshold:.6f} rad"
        )

        print(
            f"Max iterations    : "
            f"{max_iterations}"
        )

        wrist_stall_count = 0

        if group_name == "WRIST":
            print(
                f"Min progress      : "
                f"{self.args.min_wrist_progress:.6f} rad"
            )
            print(
                f"Max stall count   : "
                f"{self.args.max_wrist_stall_iterations}"
            )

        for iteration in range(
            1,
            max_iterations + 1,
        ):

            q_current, _ = (
                self.arm_qdq()
            )

            (
                home_error_before,
                home_worst_i,
                _,
            ) = self.group_error(
                q_current,
                q_home,
                active_indices,
            )

            print()

            print(
                f"{group_name}_ITER_"
                f"{iteration:02d} BEFORE: "
                f"HOME error="
                f"{home_error_before:.6f} rad "
                f"({ARM_NAMES[home_worst_i]})"
            )

            # ------------------------------------------------
            # If already close enough, verify stability.
            # ------------------------------------------------

            if (
                home_error_before
                < self.args.recovery_threshold
            ):

                q_target = (
                    q_current.copy()
                )

                q_target[
                    active_indices
                ] = (
                    q_home[
                        active_indices
                    ]
                )

                result = (
                    self.hold_and_check_group_home(
                        f"{group_name}_ITER_"
                        f"{iteration:02d}_VERIFY",
                        q_target,
                        q_home,
                        active_indices,
                        inactive_indices,
                    )
                )

                if result[
                    "stable_reached"
                ]:

                    print(
                        f"{group_name}_READY = PASS"
                    )

                    print(
                        "=" * 104
                    )

                    return True

                # Re-read actual state if verification did
                # not achieve the stable criterion.
                q_current, _ = (
                    self.arm_qdq()
                )

                (
                    home_error_before,
                    home_worst_i,
                    _,
                ) = self.group_error(
                    q_current,
                    q_home,
                    active_indices,
                )

            # ------------------------------------------------
            # Create next target FROM THE REAL MEASURED q.
            # ------------------------------------------------

            q_target = (
                self.make_adaptive_target(
                    q_current,
                    q_home,
                    active_indices,
                    stage_delta,
                )
            )

            (
                limit_ok,
                _,
                _,
                _,
            ) = self.check_joint_limits(
                q_target
            )

            if not limit_ok:

                print(
                    f"{group_name}_ITER_"
                    f"{iteration:02d} = "
                    f"FAIL (target joint limit)"
                )

                return False

            (
                step_error,
                step_worst_i,
                _,
            ) = self.group_error(
                q_target,
                q_current,
                active_indices,
            )

            (
                remaining_after_target,
                rem_worst_i,
                _,
            ) = self.group_error(
                q_target,
                q_home,
                active_indices,
            )

            print(
                f"{group_name}_ITER_"
                f"{iteration:02d} TARGET: "
                f"step={step_error:.6f} rad "
                f"({ARM_NAMES[step_worst_i]}), "
                f"nominal remaining="
                f"{remaining_after_target:.6f} rad "
                f"({ARM_NAMES[rem_worst_i]})"
            )

            self.run_trajectory(
                f"{group_name}_ITER_"
                f"{iteration:02d}",
                q_target,
                min_duration=(
                    self.args.recovery_min_duration
                ),
            )

            result = (
                self.hold_and_check_group_home(
                    f"{group_name}_ITER_"
                    f"{iteration:02d}",
                    q_target,
                    q_home,
                    active_indices,
                    inactive_indices,
                )
            )

            progress = (
                home_error_before
                - result[
                    "active_home_error"
                ]
            )

            print()

            print(
                f"{group_name}_ITER_"
                f"{iteration:02d} RESULT"
            )

            print(
                f"  HOME error before : "
                f"{home_error_before:.6f} rad"
            )

            print(
                f"  HOME error after  : "
                f"{result['active_home_error']:.6f} rad "
                f"("
                f"{ARM_NAMES[result['active_home_worst_i']]}, "
                f"{result['active_home_signed']:+.6f}"
                f")"
            )

            print(
                f"  Progress          : "
                f"{progress:+.6f} rad"
            )

            print(
                f"  Micro-target err  : "
                f"{result['active_target_error']:.6f} rad "
                f"("
                f"{ARM_NAMES[result['active_target_worst_i']]}"
                f")"
            )

            print(
                f"  Inactive hold err : "
                f"{result['inactive_hold_error']:.6f} rad "
                f"("
                f"{ARM_NAMES[result['inactive_hold_worst_i']]}"
                f")"
            )

            print(
                f"  Stable at HOME    : "
                f"{result['stable_reached']}"
            )

            print(
                f"  Final max |dq|    : "
                f"{np.max(np.abs(result['dq_final'])):.6f} rad/s"
            )

            if result[
                "stable_reached"
            ]:

                print(
                    f"{group_name}_READY = PASS"
                )

                print(
                    "=" * 104
                )

                return True

            # ------------------------------------------------
            # V2.1 WRIST progress watchdog.
            # ------------------------------------------------
            if group_name == "WRIST":

                if (
                    progress
                    < self.args.min_wrist_progress
                ):
                    wrist_stall_count += 1
                else:
                    wrist_stall_count = 0

                print(
                    f"  Wrist stall count : "
                    f"{wrist_stall_count}/"
                    f"{self.args.max_wrist_stall_iterations} "
                    f"(min progress "
                    f"{self.args.min_wrist_progress:.6f} rad)"
                )

                if (
                    wrist_stall_count
                    >= self.args.max_wrist_stall_iterations
                ):

                    print(
                        f"{group_name}_READY = FAIL"
                    )

                    print(
                        "Reason: consecutive WRIST iterations "
                        "made insufficient HOME-error progress."
                    )

                    print(
                        "=" * 104
                    )

                    return False

            # Key V2.1 behavior:
            # DO NOT fail only because this micro-target
            # has residual error.
            # Next iteration will be rebuilt from real q.
            print(
                f"{group_name}_ITER_"
                f"{iteration:02d} = CONTINUE"
            )

        # ----------------------------------------------------
        # Max iteration failure
        # ----------------------------------------------------

        q_final, _ = (
            self.arm_qdq()
        )

        (
            final_error,
            final_i,
            final_signed,
        ) = self.group_error(
            q_final,
            q_home,
            active_indices,
        )

        print()

        print(
            f"{group_name}_READY = FAIL"
        )

        print(
            f"Reason: max iterations reached "
            f"with HOME error "
            f"{final_error:.6f} rad "
            f"("
            f"{ARM_NAMES[final_i]}, "
            f"{final_signed:+.6f}"
            f")"
        )

        print(
            "=" * 104
        )

        return False

    # ========================================================
    # FINAL HOME
    # ========================================================

    def final_home(
        self,
        q_home: np.ndarray,
    ) -> bool:

        self.run_trajectory(
            "FINAL_HOME",
            q_home,
            min_duration=(
                self.args.min_auto_duration
            ),
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
            < self.args.hold_time
        ):

            now = (
                time.monotonic()
            )

            q, dq = (
                self.arm_qdq()
            )

            error_vector = (
                q_home
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

            if (
                max_error
                < self.args.home_threshold
            ):

                if stable_start is None:

                    stable_start = (
                        now
                    )

                if (
                    now
                    - stable_start
                    >= self.args.home_stable_time
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
                    now
                    - stable_start
                )

                print(
                    f"[FINAL_HOME_hold] "
                    f"max_err="
                    f"{max_error:.6f} "
                    f"best="
                    f"{best_error:.6f} "
                    f"stable="
                    f"{stable_time:.2f}/"
                    f"{self.args.home_stable_time:.2f}"
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

                time.sleep(
                    dt
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

        ready = (
            max_error
            < self.args.home_threshold
            and
            stable_reached
        )

        print()
        print(
            "=" * 104
        )

        print(
            "FINAL_HOME RESULT"
        )

        print(
            "=" * 104
        )

        print(
            f"MAX error      : "
            f"{max_error:.6f} rad "
            f"("
            f"{np.degrees(max_error):.3f} deg"
            f")"
        )

        print(
            f"Worst          : "
            f"{ARM_NAMES[worst]}"
        )

        print(
            f"Best hold error: "
            f"{best_error:.6f} rad"
        )

        print(
            f"Stable reached : "
            f"{stable_reached}"
        )

        print(
            f"Final max |dq| : "
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

        print()

        print(
            f"FINAL_HOME = "
            f"{'PASS' if ready else 'FAIL'}"
        )

        print(
            f"AUTO_HOME_READY = "
            f"{'PASS' if ready else 'FAIL'}"
        )

        print(
            "=" * 104
        )

        return ready

    # ========================================================
    # Execute V2
    # ========================================================

    def execute(
        self,
        q_home: np.ndarray,
    ) -> int:

        if not self.preflight(
            q_home,
            verbose=False,
        ):

            raise RuntimeError(
                "Pre-flight failed"
            )

        self.capture_hold_positions()

        self.create_publisher()

        q_current, _ = (
            self.arm_qdq()
        )

        global_error = float(
            np.max(
                np.abs(
                    q_home
                    - q_current
                )
            )
        )

        direct_mode = (
            global_error
            <= self.args.direct_home_delta
        )

        print()
        print(
            "=" * 104
        )

        print(
            "REAL ROBOT AUTO-HOME V2.1 ARMED"
        )

        print(
            "=" * 104
        )

        print(
            f"Initial global HOME error : "
            f"{global_error:.6f} rad"
        )

        print(
            f"Mode                      : "
            f"{'DIRECT HOME' if direct_mode else 'ADAPTIVE RECOVERY'}"
        )

        print(
            "LowCmd will remain active "
            "until completion or abort."
        )

        print()

        print(
            "Type RECOVER HOME V2.1 exactly to continue."
        )

        phrase = input(
            "> "
        )

        if (
            phrase
            != "RECOVER HOME V2.1"
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

            q_after = (
                self.takeover_test()
            )

            global_error = float(
                np.max(
                    np.abs(
                        q_home
                        - q_after
                    )
                )
            )

            direct_mode = (
                global_error
                <= self.args.direct_home_delta
            )

            print()
            print(
                "POST-TAKEOVER V2.1 DECISION"
            )

            print(
                f"Global HOME error : "
                f"{global_error:.6f} rad"
            )

            print(
                f"Mode              : "
                f"{'DIRECT HOME' if direct_mode else 'ADAPTIVE RECOVERY'}"
            )

            if not direct_mode:

                wrist_ok = (
                    self.adaptive_group_recovery(
                        "WRIST",
                        WRIST_LOCAL_INDICES,
                        ARM_CORE_LOCAL_INDICES,
                        q_home,
                        self.args.wrist_stage_delta,
                        self.args.max_wrist_iterations,
                    )
                )

                if not wrist_ok:

                    print(
                        "RECOVERY ABORTED: "
                        "WRIST adaptive recovery failed."
                    )

                    return 5

                arm_ok = (
                    self.adaptive_group_recovery(
                        "ARM",
                        ARM_CORE_LOCAL_INDICES,
                        WRIST_LOCAL_INDICES,
                        q_home,
                        self.args.arm_stage_delta,
                        self.args.max_arm_iterations,
                    )
                )

                if not arm_ok:

                    print(
                        "RECOVERY ABORTED: "
                        "ARM adaptive recovery failed."
                    )

                    return 6

            home_ok = (
                self.final_home(
                    q_home
                )
            )

            return (
                0
                if home_ok
                else 3
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
            "R1-A7 AUTO HOME V2.1 candidate with "
            "measured-state adaptive recovery and wrist progress watchdog"
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

    # Motion
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

    # AUTO duration
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

    # V2 recovery
    p.add_argument(
        "--direct-home-delta",
        type=float,
        default=0.35,
    )

    p.add_argument(
        "--wrist-stage-delta",
        type=float,
        default=0.15,
    )

    p.add_argument(
        "--arm-stage-delta",
        type=float,
        default=0.20,
    )

    p.add_argument(
        "--max-wrist-iterations",
        type=int,
        default=12,
        help=(
            "Maximum WRIST adaptive-recovery iterations. "
            "V2.1 candidate default: 12 (frozen V2 default was 8)."
        ),
    )

    p.add_argument(
        "--max-arm-iterations",
        type=int,
        default=6,
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

    # V2.1 WRIST progress watchdog
    p.add_argument(
        "--min-wrist-progress",
        type=float,
        default=0.005,
        help=(
            "Minimum meaningful decrease in WRIST HOME error per iteration "
            "before the iteration is counted as stalled."
        ),
    )

    p.add_argument(
        "--max-wrist-stall-iterations",
        type=int,
        default=2,
        help=(
            "Abort WRIST recovery after this many consecutive iterations "
            "with progress below --min-wrist-progress."
        ),
    )

    # Gains
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

    # Takeover
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

    # HOME
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


def validate_args(
    args: argparse.Namespace,
):

    positive_fields = [
        "hz",
        "state_timeout",
        "max_velocity",
        "max_acceleration",
        "max_step",
        "max_command_lead",
        "max_start_dq",
        "joint_limit_margin",
        "min_auto_duration",
        "recovery_min_duration",
        "duration_step",
        "max_auto_duration",
        "direct_home_delta",
        "wrist_stage_delta",
        "arm_stage_delta",
        "recovery_threshold",
        "recovery_stable_time",
        "recovery_hold_time",
        "min_wrist_progress",
        "takeover_time",
        "max_takeover_drift",
        "hold_time",
        "home_threshold",
        "home_stable_time",
        "print_period",
    ]

    for field in (
        positive_fields
    ):

        if (
            getattr(
                args,
                field,
            )
            <= 0
        ):

            raise RuntimeError(
                f"--"
                f"{field.replace('_', '-')}"
                f" must be > 0"
            )

    if (
        args.duration_margin
        < 1.0
    ):

        raise RuntimeError(
            "--duration-margin must be >= 1.0"
        )

    if (
        args.max_wrist_iterations
        < 1
    ):

        raise RuntimeError(
            "--max-wrist-iterations must be >= 1"
        )

    if (
        args.max_wrist_stall_iterations
        < 1
    ):

        raise RuntimeError(
            "--max-wrist-stall-iterations must be >= 1"
        )

    if (
        args.max_arm_iterations
        < 1
    ):

        raise RuntimeError(
            "--max-arm-iterations must be >= 1"
        )


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

    validate_args(
        args
    )

    q_home = (
        load_keyframe(
            args.keyframes,
            args.keyframe,
        )
    )

    mover = (
        AutoKeyframeMoverV21(
            args
        )
    )

    mover.connect_state_only()

    passed = (
        mover.preflight(
            q_home,
            verbose=True,
        )
    )

    if not passed:

        print()
        print(
            "ABORT: pre-flight failed."
        )

        return 2

    if not args.execute:

        print()
        print(
            "PRE-FLIGHT ONLY."
        )

        print(
            "Robot was NOT commanded."
        )

        return 0

    return (
        mover.execute(
            q_home
        )
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
