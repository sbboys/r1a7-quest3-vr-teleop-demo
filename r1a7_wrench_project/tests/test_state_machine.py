from src.robot.mock_robot_interface import MockRobotInterface
from src.robot.safety_monitor import SafetyConfig, SafetyMonitor
from src.task.task_state import TaskState
from src.task.wrench_task_fsm import WrenchTaskFSM


def test_mock_fsm_reaches_complete():
    robot = MockRobotInterface()
    robot.connect()
    monitor = SafetyMonitor(SafetyConfig())
    fsm = WrenchTaskFSM(robot=robot, safety=monitor)
    history = fsm.run_dry()
    assert history[-1] == TaskState.COMPLETE
    assert TaskState.TIGHTEN in history
