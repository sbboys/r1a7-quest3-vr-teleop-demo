#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import server
from server import VisualServoGate, build_control_arguments


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
            "--right-gripper-close-q": "-0.2",
            "--cartesian-jog-speed-mm-s": "10.0",
            "--cartesian-jog-accel-mm-s2": "20.0",
        }
        for option, expected in expected_values.items():
            self.assertEqual(arguments[arguments.index(option) + 1], expected)

        for terminal_default in (
            "--ik-frequency",
            "--publish-frequency",
            "--max-joint-speed",
            "--left-arm-gravity-feedforward-scale",
            "--right-arm-gravity-feedforward-scale",
        ):
            self.assertNotIn(terminal_default, arguments)

        safety_values = {
            "--ibvs-command-path": "/tmp/r1a7_console_ibvs_velocity.json",
            "--ibvs-max-age-s": "0.2",
            "--ibvs-max-speed-mm-s": "10.0",
            "--ibvs-min-speed-mm-s": "0.0",
        }
        for option, expected in safety_values.items():
            self.assertEqual(arguments[arguments.index(option) + 1], expected)
        self.assertIn("--no-udp-visual-servo", arguments)

    def test_web_values_are_bounded(self) -> None:
        arguments = build_control_arguments(
            {
                "xy_step_mm": 100,
                "z_step_mm": 100,
                "joint_step_deg": 100,
                "gripper_speed": 100,
                "right_gripper_close_q": -100,
                "cartesian_speed_mm_s": 100,
                "cartesian_accel_mm_s2": 100,
            }
        )
        expected_maxima = {
            "--jog-step-mm": "30.0",
            "--jog-vertical-step-mm": "10.0",
            "--joint-jog-step-deg": "2.0",
            "--gripper-speed": "2.0",
            "--right-gripper-close-q": "-0.2",
            "--cartesian-jog-speed-mm-s": "12.0",
            "--cartesian-jog-accel-mm-s2": "40.0",
        }
        for option, expected in expected_maxima.items():
            self.assertEqual(arguments[arguments.index(option) + 1], expected)

        arguments = build_control_arguments({"right_gripper_close_q": 100})
        self.assertEqual(
            arguments[arguments.index("--right-gripper-close-q") + 1],
            "-0.05",
        )

    def test_visual_servo_gate_uses_dual_limit_and_toggle_lease(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fusion.json"
            output = Path(directory) / "gate.json"
            source.write_text(
                json.dumps(
                    {
                        "timestamp": time.time(),
                        "shadow_mode": True,
                        "ready_for_motion": True,
                        "external_valid": True,
                        "wrist_valid": True,
                        "fusion_mode": "DUAL",
                        "proposed_vx_mm_s": 3.0,
                        "proposed_vy_mm_s": 4.0,
                        "proposed_vz_mm_s": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            gate = VisualServoGate(source, output)
            gate.bind_control_snapshot(
                lambda: {
                    "phase": "enabled",
                    "running": True,
                    "telemetry": {
                        "timestamp": time.time(),
                        "control_mode": "cartesian",
                        "cartesian_arm": "right",
                        "motion_state": "holding",
                        "lag_mm": 1000.0,
                    },
                }
            )
            with patch.object(
                server.CAMERAS,
                "threading_vision_snapshot",
                return_value={"ready": True},
            ):
                gate.publish_once()
                initial_payload = json.loads(output.read_text())
                self.assertFalse(initial_payload["enabled"])
                self.assertEqual(initial_payload["reason"], "toggle_off")
                gate.keepalive()
                payload = json.loads(output.read_text())
                self.assertTrue(payload["enabled"])
                self.assertEqual(payload["reason"], "toggle_active")
                speed = sum(payload[key] ** 2 for key in ("vx_mm_s", "vy_mm_s", "vz_mm_s")) ** 0.5
                self.assertAlmostEqual(speed, 5.0)
                self.assertEqual(payload["fusion_mode"], "DUAL")
                self.assertEqual(payload["max_speed_mm_s"], 10.0)
                gate.stop(inhibit_s=0.5)
                self.assertFalse(json.loads(output.read_text())["enabled"])
                with self.assertRaisesRegex(RuntimeError, "toggle_stop_settling"):
                    gate.keepalive()

    def test_visual_servo_gate_restores_single_and_blind_speed_layers(self) -> None:
        cases = (
            ("EXTERNAL_ONLY", True, False, (3.0, 4.0, 0.0), 2.0),
            ("WRIST_ONLY", False, True, (0.0, 3.0, 4.0), 2.0),
            ("BLIND_GRACE", False, False, (0.3, 0.0, 0.4), 0.5),
        )
        for mode, external_valid, wrist_valid, velocity, expected_speed in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "fusion.json"
                output = Path(directory) / "gate.json"
                source.write_text(
                    json.dumps(
                        {
                            "timestamp": time.time(),
                            "shadow_mode": True,
                            "ready_for_motion": True,
                            "external_valid": external_valid,
                            "wrist_valid": wrist_valid,
                            "fusion_mode": mode,
                            "proposed_vx_mm_s": velocity[0],
                            "proposed_vy_mm_s": velocity[1],
                            "proposed_vz_mm_s": velocity[2],
                        }
                    ),
                    encoding="utf-8",
                )
                gate = VisualServoGate(source, output)
                gate.bind_control_snapshot(
                    lambda: {
                        "phase": "enabled",
                        "running": True,
                        "telemetry": {
                            "timestamp": time.time(),
                            "control_mode": "cartesian",
                            "cartesian_arm": "right",
                            "motion_state": "holding",
                            "lag_mm": 1.0,
                        },
                    }
                )
                with patch.object(
                    server.CAMERAS,
                    "threading_vision_snapshot",
                    return_value={"ready": True},
                ):
                    gate.keepalive()
                    payload = json.loads(output.read_text())
                speed = sum(
                    payload[key] ** 2
                    for key in ("vx_mm_s", "vy_mm_s", "vz_mm_s")
                ) ** 0.5
                self.assertTrue(payload["enabled"])
                self.assertAlmostEqual(speed, expected_speed)
                self.assertEqual(payload["fusion_mode"], mode)
                self.assertEqual(payload["max_speed_mm_s"], expected_speed)


if __name__ == "__main__":
    unittest.main()
