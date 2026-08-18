from __future__ import annotations

import time
from typing import List, Optional, Sequence

from .robot_interface import RobotInterface
from .robot_state import RobotState


DEFAULT_LEFT_ARM_INDICES = [15, 16, 17, 18, 19, 20, 21]
DEFAULT_RIGHT_ARM_INDICES = [22, 23, 24, 25, 26, 27, 28]
DEFAULT_GRIPPER_INDICES = [31, 33]


class R1A7Interface(RobotInterface):
    """Read-only R1-A7 adapter plus blocked real-motion stubs.

    The current safe next step is recording the robot while the existing handset
    teleop script drives it. This adapter therefore subscribes to lowstate but
    refuses all command publishing methods.
    """

    def __init__(
        self,
        dry_run: bool = True,
        enable_robot_motion: bool = False,
        interface: str = "enx9c69d37d0967",
        domain_id: int = 0,
        state_topic: str = "rt/lowstate",
        left_arm_indices: Sequence[int] = DEFAULT_LEFT_ARM_INDICES,
        right_arm_indices: Sequence[int] = DEFAULT_RIGHT_ARM_INDICES,
        gripper_indices: Sequence[int] = DEFAULT_GRIPPER_INDICES,
    ):
        self.dry_run = dry_run
        self.enable_robot_motion = enable_robot_motion
        self.interface = interface
        self.domain_id = domain_id
        self.state_topic = state_topic
        self.arm_indices = list(left_arm_indices) + list(right_arm_indices)
        self.gripper_indices = list(gripper_indices)
        self.low_state = None
        self.last_state_time: Optional[float] = None
        self.subscriber = None
        self.connected = False

    def _motion_blocked(self) -> None:
        raise RuntimeError(
            "R1A7Interface is read-only in this phase. Use the existing handset "
            "teleop script for real robot movement; this adapter only records rt/lowstate."
        )

    def connect(self) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        except Exception as exc:
            raise RuntimeError("unitree_sdk2py is required for real read-only state recording") from exc

        ChannelFactoryInitialize(self.domain_id, self.interface)
        self.subscriber = ChannelSubscriber(self.state_topic, LowState_)
        self.subscriber.Init(self._lowstate_handler, 10)
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        return None

    def acquire_arm_control(self) -> None:
        return None

    def release_arm_control(self) -> None:
        return None

    def get_robot_state(self) -> RobotState:
        if self.low_state is None:
            return RobotState(timestamp=time.monotonic(), communication_ok=False, robot_mode="waiting_lowstate")
        motor_state = self.low_state.motor_state
        max_idx = max(self.arm_indices + self.gripper_indices)
        if len(motor_state) <= max_idx:
            raise RuntimeError(f"lowstate has {len(motor_state)} motors, requested index {max_idx}")
        joint_position = [float(motor_state[i].q) for i in self.arm_indices]
        joint_velocity = [float(motor_state[i].dq) for i in self.arm_indices]
        joint_current = [_read_motor_current_or_tau(motor_state[i]) for i in self.arm_indices]
        left_gripper, right_gripper = [float(motor_state[i].q) for i in self.gripper_indices]
        left_current, right_current = [_read_motor_current_or_tau(motor_state[i]) for i in self.gripper_indices]
        fault_code = _read_fault_code(self.low_state)
        return RobotState(
            timestamp=self.last_state_time or time.monotonic(),
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            joint_torque_or_current=joint_current,
            gripper_position={"left": left_gripper, "right": right_gripper},
            gripper_current={"left": left_current, "right": right_current},
            robot_mode=_read_robot_mode(self.low_state),
            fault_code=fault_code,
            communication_ok=True,
        )

    def send_joint_position_command(self, positions: List[float]) -> None:
        self._motion_blocked()

    def send_joint_trajectory_point(self, positions: List[float], duration_s: Optional[float] = None) -> None:
        self._motion_blocked()

    def send_gripper_command(self, left: Optional[float] = None, right: Optional[float] = None) -> None:
        self._motion_blocked()

    def hold_position(self) -> None:
        self._motion_blocked()

    def safe_stop(self) -> None:
        return None

    def emergency_stop(self) -> None:
        return None

    def _lowstate_handler(self, msg) -> None:
        self.low_state = msg
        self.last_state_time = time.monotonic()


def _read_motor_current_or_tau(motor) -> float:
    for name in ("tau_est", "tau", "current"):
        if hasattr(motor, name):
            try:
                return float(getattr(motor, name))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _read_fault_code(low_state) -> Optional[int]:
    for name in ("fault_code", "error_code"):
        if hasattr(low_state, name):
            try:
                return int(getattr(low_state, name))
            except (TypeError, ValueError):
                return None
    return None


def _read_robot_mode(low_state) -> str:
    for name in ("mode_machine", "mode_pr"):
        if hasattr(low_state, name):
            return f"{name}={getattr(low_state, name)}"
    return "lowstate"
