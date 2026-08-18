from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from .robot_state import RobotState


@dataclass
class SafetyConfig:
    dry_run: bool = True
    enable_robot_motion: bool = False
    state_timeout_ms: int = 200
    max_joint_step_rad: float = 0.01
    max_joint_velocity_rad_s: float = 0.3
    max_joint_current: Optional[float] = None
    stop_on_communication_loss: bool = True
    stop_on_fault: bool = True
    stop_on_current_limit: bool = True


@dataclass
class SafetyResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)


class SafetyMonitor:
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.last_command: Optional[List[float]] = None

    def motion_allowed(self) -> SafetyResult:
        reasons = []
        if self.config.dry_run:
            reasons.append("dry_run is true")
        if not self.config.enable_robot_motion:
            reasons.append("enable_robot_motion is false")
        return SafetyResult(ok=not reasons, reasons=reasons)

    def validate_state(self, state: RobotState, now: Optional[float] = None) -> SafetyResult:
        now = time.monotonic() if now is None else now
        reasons = []
        if self.config.stop_on_communication_loss and not state.communication_ok:
            reasons.append("communication is not ok")
        age_ms = (now - state.timestamp) * 1000.0
        if age_ms > self.config.state_timeout_ms:
            reasons.append(f"state timeout {age_ms:.1f} ms")
        if self.config.stop_on_fault and state.fault_code not in (None, 0):
            reasons.append(f"fault_code={state.fault_code}")
        if self.config.stop_on_current_limit and self.config.max_joint_current is not None:
            if any(abs(v) > self.config.max_joint_current for v in state.joint_torque_or_current):
                reasons.append("joint current limit exceeded")
        return SafetyResult(ok=not reasons, reasons=reasons)

    def validate_joint_command(self, command: Sequence[float], current: Optional[Sequence[float]] = None) -> SafetyResult:
        reasons = []
        previous: Iterable[float]
        if current is not None:
            previous = current
        elif self.last_command is not None:
            previous = self.last_command
        else:
            previous = command
        max_step = max(abs(float(a) - float(b)) for a, b in zip(command, previous)) if command else 0.0
        if max_step > self.config.max_joint_step_rad:
            reasons.append(f"joint step {max_step:.4f} exceeds {self.config.max_joint_step_rad:.4f}")
        if not reasons:
            self.last_command = list(map(float, command))
        return SafetyResult(ok=not reasons, reasons=reasons)
