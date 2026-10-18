#!/usr/bin/env python3
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from camera_manager import (
    CAMERAS,
    CameraManager,
    VideoDevice,
    choose_device,
    preview_command,
    threading_vision_commands,
)


class CameraManagerTests(unittest.TestCase):
    def test_orbbec_selection_rejects_depth_only_node(self) -> None:
        spec = next(item for item in CAMERAS if item.camera_id == "orbbec_left")
        devices = [
            VideoDevice("/dev/video0", "Orbbec", spec.serial, ("Z16 ",)),
            VideoDevice("/dev/video6", "Orbbec", spec.serial, ("YUYV", "MJPG")),
        ]
        self.assertEqual(choose_device(spec, devices).node, "/dev/video6")

    def test_serials_keep_wrist_cameras_distinct(self) -> None:
        left = next(item for item in CAMERAS if item.camera_id == "wrist_left")
        devices = [
            VideoDevice("/dev/video20", "JR0002", "JR0002", ("MJPG",)),
            VideoDevice("/dev/video18", "JR0001", "JR0001", ("MJPG",)),
        ]
        self.assertEqual(choose_device(left, devices).node, "/dev/video18")

    def test_mjpeg_preview_is_preferred(self) -> None:
        command, pixel_format = preview_command(
            VideoDevice("/dev/video18", "JR0001", "JR0001", ("MJPG", "YUYV"))
        )
        self.assertEqual(pixel_format, "MJPG")
        self.assertIn("image/jpeg,width=640,height=480,framerate=30/1", command)
        self.assertIn("device=/dev/video18", command)

    def test_rescan_clears_only_recovered_missing_device_error(self) -> None:
        with patch("camera_manager.discover_video_devices", return_value=[]):
            manager = CameraManager(Path("/project"))
        spec = next(item for item in CAMERAS if item.camera_id == "orbbec_right")
        runtime = manager.runtime[spec.camera_id]
        runtime.last_error = f"未检测到 {spec.label} 的彩色采集节点"
        manager.devices = [
            VideoDevice("/dev/video14", "Orbbec", spec.serial, ("YUYV", "MJPG"))
        ]

        camera = next(
            item
            for item in manager.snapshot()["cameras"]
            if item["id"] == spec.camera_id
        )

        self.assertTrue(camera["detected"])
        self.assertEqual(camera["node"], "/dev/video14")
        self.assertEqual(camera["last_error"], "")

        runtime.last_error = "预览启动失败，返回码 1"
        camera = next(
            item
            for item in manager.snapshot()["cameras"]
            if item["id"] == spec.camera_id
        )
        self.assertEqual(camera["last_error"], "预览启动失败，返回码 1")

    def test_head_timeout_reports_missing_official_udp_frame(self) -> None:
        with patch("camera_manager.discover_video_devices", return_value=[]):
            manager = CameraManager(Path("/project"))
        runtime = manager.runtime["head"]
        runtime.process = unittest.mock.Mock(returncode=None, pid=123)
        runtime.started_at = time.monotonic() - 13.0

        camera = next(
            item for item in manager.snapshot()["cameras"] if item["id"] == "head"
        )

        self.assertIn("UDP 5001", camera["last_error"])
        self.assertIn("补丁版本", camera["last_error"])
        self.assertEqual(camera["backend"], "unitree_udp_h264")

    def test_threading_vision_uses_dynamic_external_node_and_shadow_fusion(self) -> None:
        commands = threading_vision_commands(Path("/project"), "/dev/video42")
        self.assertEqual(commands["external"][-1], "/dev/video42")
        self.assertIn("task6_wrist_ibvs_controller_v2_tip.py", commands["wrist"][2])
        self.assertEqual(commands["fusion"][-1], "--shadow-mode")


if __name__ == "__main__":
    unittest.main()
