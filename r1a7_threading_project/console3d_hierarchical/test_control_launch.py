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

        safety_values = {
            "--ibvs-command-path": "/tmp/r1a7_console_ibvs_velocity.json",
            "--ibvs-max-age-s": "0.2",
            "--ibvs-max-speed-mm-s": "2.0",
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

    def test_visual_servo_gate_requires_dual_fresh_guidance_and_deadman(self) -> None:
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
                        "lag_mm": 1.0,
                    },
                }
            )
            with patch.object(
                server.CAMERAS,
                "threading_vision_snapshot",
                return_value={"ready": True},
            ):
                gate.publish_once()
                self.assertFalse(json.loads(output.read_text())["enabled"])
                gate.keepalive()
                payload = json.loads(output.read_text())
                self.assertTrue(payload["enabled"])
                speed = sum(payload[key] ** 2 for key in ("vx_mm_s", "vy_mm_s", "vz_mm_s")) ** 0.5
                self.assertAlmostEqual(speed, 2.0)
                gate.stop(inhibit_s=0.5)
                self.assertFalse(json.loads(output.read_text())["enabled"])
                with self.assertRaisesRegex(RuntimeError, "deadman_stop_settling"):
                    gate.keepalive()


if __name__ == "__main__":
    unittest.main()
