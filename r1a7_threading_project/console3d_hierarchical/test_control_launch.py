#!/usr/bin/env python3
import unittest

from server import build_control_arguments


class ControlLaunchTests(unittest.TestCase):
    def test_defaults_match_verified_terminal_baseline(self) -> None:
        arguments = build_control_arguments({})
        expected_values = {
            "--interface": "enp6s0",
            "--domain-id": "0",
            "--host-ip": "192.168.123.223",
            "--jog-step-mm": "30.0",
            "--jog-vertical-step-mm": "10.0",
            "--joint-jog-step-deg": "1.0",
            "--gripper-speed": "1.5",
            "--cartesian-jog-speed-mm-s": "10.0",
            "--cartesian-jog-accel-mm-s2": "20.0",
        }
        for option, expected in expected_values.items():
            self.assertEqual(arguments[arguments.index(option) + 1], expected)

        for terminal_default in (
            "--ik-frequency",
            "--publish-frequency",
            "--max-joint-speed",
            "--right-arm-gravity-feedforward-scale",
        ):
            self.assertNotIn(terminal_default, arguments)

    def test_web_values_are_bounded(self) -> None:
        arguments = build_control_arguments(
            {
                "xy_step_mm": 100,
                "z_step_mm": 100,
                "joint_step_deg": 100,
                "gripper_speed": 100,
                "cartesian_speed_mm_s": 100,
                "cartesian_accel_mm_s2": 100,
            }
        )
        expected_maxima = {
            "--jog-step-mm": "30.0",
            "--jog-vertical-step-mm": "10.0",
            "--joint-jog-step-deg": "2.0",
            "--gripper-speed": "2.0",
            "--cartesian-jog-speed-mm-s": "12.0",
            "--cartesian-jog-accel-mm-s2": "40.0",
        }
        for option, expected in expected_maxima.items():
            self.assertEqual(arguments[arguments.index(option) + 1], expected)


if __name__ == "__main__":
    unittest.main()
