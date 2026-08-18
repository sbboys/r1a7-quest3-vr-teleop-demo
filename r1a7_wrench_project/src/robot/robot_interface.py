from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .robot_state import RobotState


class RobotInterface(ABC):
    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def acquire_arm_control(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def release_arm_control(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_robot_state(self) -> RobotState:
        raise NotImplementedError

    def get_joint_positions(self) -> List[float]:
        return self.get_robot_state().joint_position

    def get_joint_velocities(self) -> List[float]:
        return self.get_robot_state().joint_velocity

    def get_joint_torques_or_currents(self) -> List[float]:
        return self.get_robot_state().joint_torque_or_current

    def get_gripper_state(self) -> dict:
        return self.get_robot_state().gripper_position

    @abstractmethod
    def send_joint_position_command(self, positions: List[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_joint_trajectory_point(self, positions: List[float], duration_s: Optional[float] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_gripper_command(self, left: Optional[float] = None, right: Optional[float] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def hold_position(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def safe_stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def emergency_stop(self) -> None:
        raise NotImplementedError
