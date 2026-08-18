import pytest

from src.robot.r1a7_interface import R1A7Interface


def test_real_interface_refuses_motion_methods():
    robot = R1A7Interface()
    with pytest.raises(RuntimeError, match="read-only"):
        robot.send_joint_position_command([0.0] * 14)
