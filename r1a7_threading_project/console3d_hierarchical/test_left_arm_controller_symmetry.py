#!/usr/bin/env python3
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = (
    PROJECT_ROOT
    / "transfer_control"
    / "experimental"
    / "r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py"
)
ACTIVE_IK_PATH = Path(
    "/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/"
    "xr_teleoperate/teleop/robot_control/robot_arm_ik.py"
)
IK_PATCH_PATH = (
    PROJECT_ROOT
    / "transfer_control"
    / "patches"
    / "r1a7_arm_ik_posture_symmetry_20261016.patch"
)


def load_controller_module():
    spec = importlib.util.spec_from_file_location(
        "r1a7_symmetric_controller_test_module", CONTROLLER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MotorCommand:
    def __init__(self) -> None:
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0


class LowCommand:
    def __init__(self) -> None:
        self.mode_pr = 0
        self.mode_machine = 0
        self.motor_cmd = [MotorCommand() for _ in range(34)]
        self.crc = 0


class FakeCRC:
    @staticmethod
    def Crc(_message) -> int:
        return 123


class LeftArmControllerSymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.controller = load_controller_module()

    def test_left_gravity_scale_defaults_to_right_setting(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                str(CONTROLLER_PATH),
                "--host-ip",
                "192.168.123.223",
                "--right-arm-gravity-feedforward-scale",
                "0.7",
            ],
        ):
            args = self.controller.parse_args()
        self.assertEqual(args.left_arm_gravity_feedforward_scale, 0.7)
        self.assertEqual(args.right_arm_gravity_feedforward_scale, 0.7)

    def test_lowcmd_kd_dq_and_torque_are_symmetric(self) -> None:
        output = self.controller.R1A7LowCmdOutput(
            publisher=None,
            low_cmd_factory=LowCommand,
            crc=FakeCRC(),
            publish_frequency=250.0,
            max_joint_speed=2.0,
            left_arm_gravity_feedforward_scale=1.0,
            right_arm_gravity_feedforward_scale=1.0,
            gravity_feedforward_max_torque=10.0,
            gravity_feedforward_slew=1.0e6,
        )
        arm_q = self.controller.np.zeros(14, dtype=float)
        arm_dq_ref = self.controller.np.zeros(14, dtype=float)
        arm_dq_ref[0] = 0.05
        arm_dq_ref[7] = -0.05
        arm_tau_ff = self.controller.np.zeros(14, dtype=float)
        arm_tau_ff[0] = 0.4
        arm_tau_ff[7] = -0.5
        output.set_arm_goal(
            arm_q,
            mode_machine=0,
            arm_tau_ff=arm_tau_ff,
            arm_dq_ref=arm_dq_ref,
        )

        message = output._build_message()

        self.assertEqual(message.motor_cmd[15].kd, 30.0)
        self.assertEqual(message.motor_cmd[22].kd, 30.0)
        self.assertAlmostEqual(message.motor_cmd[15].dq, 0.05)
        self.assertAlmostEqual(message.motor_cmd[22].dq, -0.05)
        self.assertAlmostEqual(message.motor_cmd[15].tau, 0.4)
        self.assertAlmostEqual(message.motor_cmd[22].tau, -0.5)

    def test_active_ik_accepts_symmetric_left_posture_parameters(self) -> None:
        required = {
            "left_wrist_pitch_weight",
            "left_wrist_posture_weight",
            "left_wrist_posture_ref",
            "left_elbow_posture_weight",
            "left_elbow_posture_ref",
        }
        if not ACTIVE_IK_PATH.exists():
            patch_text = IK_PATCH_PATH.read_text(encoding="utf-8")
            for name in required:
                self.assertIn(name, patch_text)
            return

        tree = ast.parse(ACTIVE_IK_PATH.read_text(encoding="utf-8"))
        solve = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.name == "R1A7_ArmIK"
            for node in node.body
            if isinstance(node, ast.FunctionDef) and node.name == "solve_ik"
        )
        argument_names = {argument.arg for argument in solve.args.args}
        self.assertTrue(required.issubset(argument_names))


if __name__ == "__main__":
    unittest.main()
