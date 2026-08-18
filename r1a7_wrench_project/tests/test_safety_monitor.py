from src.robot.robot_state import RobotState
from src.robot.safety_monitor import SafetyConfig, SafetyMonitor


def test_motion_blocked_by_default():
    monitor = SafetyMonitor(SafetyConfig())
    result = monitor.motion_allowed()
    assert not result.ok
    assert "dry_run is true" in result.reasons
    assert "enable_robot_motion is false" in result.reasons


def test_validate_state_requires_communication():
    monitor = SafetyMonitor(SafetyConfig())
    state = RobotState(timestamp=0.0, communication_ok=False)
    result = monitor.validate_state(state, now=0.0)
    assert not result.ok
    assert "communication is not ok" in result.reasons
