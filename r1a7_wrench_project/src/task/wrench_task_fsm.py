from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .task_state import TaskState
from ..robot.robot_interface import RobotInterface
from ..robot.safety_monitor import SafetyMonitor


DEFAULT_DRY_RUN_SEQUENCE = [
    TaskState.CHECK_SYSTEM,
    TaskState.MOVE_HOME,
    TaskState.MOVE_TO_PRE_GRASP,
    TaskState.MOVE_TO_GRASP,
    TaskState.CLOSE_GRIPPER,
    TaskState.VERIFY_GRASP,
    TaskState.LIFT_WRENCH,
    TaskState.MOVE_TO_PRE_NUT,
    TaskState.APPROACH_NUT,
    TaskState.SEAT_WRENCH,
    TaskState.TIGHTEN,
    TaskState.LOOSEN,
    TaskState.RETRACT,
    TaskState.PLACE_WRENCH,
    TaskState.RETURN_HOME,
    TaskState.COMPLETE,
]


@dataclass
class WrenchTaskFSM:
    robot: RobotInterface
    safety: SafetyMonitor
    sequence: List[TaskState] = field(default_factory=lambda: list(DEFAULT_DRY_RUN_SEQUENCE))
    state: TaskState = TaskState.IDLE
    history: List[TaskState] = field(default_factory=list)

    def step(self) -> TaskState:
        if self.state in (TaskState.COMPLETE, TaskState.ERROR, TaskState.EMERGENCY_STOP):
            return self.state
        next_state = self.sequence[len(self.history)] if len(self.history) < len(self.sequence) else TaskState.COMPLETE
        self.state = next_state
        self.history.append(next_state)
        if next_state == TaskState.CHECK_SYSTEM:
            result = self.safety.validate_state(self.robot.get_robot_state())
            if not result.ok:
                self.state = TaskState.ERROR
        return self.state

    def run_dry(self) -> List[TaskState]:
        while self.state not in (TaskState.COMPLETE, TaskState.ERROR, TaskState.EMERGENCY_STOP):
            self.step()
        return list(self.history)
