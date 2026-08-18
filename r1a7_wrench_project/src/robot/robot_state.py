from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RobotState:
    timestamp: float
    joint_position: List[float] = field(default_factory=list)
    joint_velocity: List[float] = field(default_factory=list)
    joint_torque_or_current: List[float] = field(default_factory=list)
    gripper_position: Dict[str, float] = field(default_factory=dict)
    gripper_current: Dict[str, Optional[float]] = field(default_factory=dict)
    robot_mode: str = "unknown"
    fault_code: Optional[int] = None
    communication_ok: bool = False
