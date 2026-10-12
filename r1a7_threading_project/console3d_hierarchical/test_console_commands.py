#!/usr/bin/env python3
import unittest

from r1a7_threading_project.console3d.server import command_allowed


class ConsoleCommandTests(unittest.TestCase):
    def test_both_cartesian_arm_selections_are_allowed(self) -> None:
        self.assertTrue(command_allowed("cartesian-arm:left"))
        self.assertTrue(command_allowed("cartesian-arm:right"))

    def test_unknown_cartesian_arm_is_rejected(self) -> None:
        self.assertFalse(command_allowed("cartesian-arm:both"))

    def test_gripper_and_joint_commands_remain_scoped(self) -> None:
        self.assertTrue(command_allowed("gripper:left:open"))
        self.assertTrue(command_allowed("joint:13:+"))
        self.assertFalse(command_allowed("joint:14:+"))
        self.assertFalse(command_allowed("gripper:left:calibrate"))


if __name__ == "__main__":
    unittest.main()
