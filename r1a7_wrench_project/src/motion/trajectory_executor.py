from __future__ import annotations

from typing import Sequence

from .trajectory_generator import quintic_interpolate
from ..robot.robot_interface import RobotInterface
from ..robot.safety_monitor import SafetyMonitor


class TrajectoryExecutor:
    def __init__(self, robot: RobotInterface, safety: SafetyMonitor, control_frequency_hz: float = 100.0):
        self.robot = robot
        self.safety = safety
        self.dt_s = 1.0 / max(1.0, control_frequency_hz)

    def move_joint_positions(self, target: Sequence[float], duration_s: float) -> list[list[float]]:
        state = self.robot.get_robot_state()
        state_result = self.safety.validate_state(state)
        if not state_result.ok:
            raise RuntimeError("; ".join(state_result.reasons))
        path = quintic_interpolate(state.joint_position, target, duration_s, self.dt_s)
        motion = self.safety.motion_allowed()
        for point in path:
            result = self.safety.validate_joint_command(point, state.joint_position)
            if not result.ok:
                raise RuntimeError("; ".join(result.reasons))
            if motion.ok:
                self.robot.send_joint_trajectory_point(point, self.dt_s)
        return path
