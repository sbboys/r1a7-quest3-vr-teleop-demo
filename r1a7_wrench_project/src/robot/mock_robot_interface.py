from __future__ import annotations

import time
from typing import List, Optional

from .robot_interface import RobotInterface
from .robot_state import RobotState


class MockRobotInterface(RobotInterface):
    def __init__(self, joint_count: int = 14):
        self.connected = False
        self.control_acquired = False
        self.joint_position = [0.0] * joint_count
        self.joint_velocity = [0.0] * joint_count
        self.joint_current = [0.0] * joint_count
        self.grippers = {"left": 4.86, "right": 4.80}
        self.events: list[str] = []

    def connect(self) -> None:
        self.connected = True
        self.events.append("connect")

    def disconnect(self) -> None:
        self.connected = False
        self.control_acquired = False
        self.events.append("disconnect")

    def acquire_arm_control(self) -> None:
        if not self.connected:
            raise RuntimeError("mock robot is not connected")
        self.control_acquired = True
        self.events.append("acquire_arm_control")

    def release_arm_control(self) -> None:
        self.control_acquired = False
        self.events.append("release_arm_control")

    def get_robot_state(self) -> RobotState:
        return RobotState(
            timestamp=time.monotonic(),
            joint_position=list(self.joint_position),
            joint_velocity=list(self.joint_velocity),
            joint_torque_or_current=list(self.joint_current),
            gripper_position=dict(self.grippers),
            gripper_current={"left": None, "right": None},
            robot_mode="mock_control" if self.control_acquired else "mock_idle",
            fault_code=None,
            communication_ok=self.connected,
        )

    def send_joint_position_command(self, positions: List[float]) -> None:
        self._set_joint_position(positions)
        self.events.append("send_joint_position_command")

    def send_joint_trajectory_point(self, positions: List[float], duration_s: Optional[float] = None) -> None:
        self._set_joint_position(positions)
        self.events.append(f"send_joint_trajectory_point:{duration_s}")

    def send_gripper_command(self, left: Optional[float] = None, right: Optional[float] = None) -> None:
        if left is not None:
            self.grippers["left"] = float(left)
        if right is not None:
            self.grippers["right"] = float(right)
        self.events.append("send_gripper_command")

    def hold_position(self) -> None:
        self.events.append("hold_position")

    def safe_stop(self) -> None:
        self.events.append("safe_stop")
        self.release_arm_control()

    def emergency_stop(self) -> None:
        self.events.append("emergency_stop")
        self.release_arm_control()

    def _set_joint_position(self, positions: List[float]) -> None:
        if not self.connected:
            raise RuntimeError("mock robot is not connected")
        if not self.control_acquired:
            raise RuntimeError("mock arm control is not acquired")
        if len(positions) != len(self.joint_position):
            raise ValueError(f"expected {len(self.joint_position)} joints, got {len(positions)}")
        now = list(map(float, positions))
        self.joint_velocity = [new - old for old, new in zip(self.joint_position, now)]
        self.joint_position = now
