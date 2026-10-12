#!/usr/bin/env python3
from pathlib import Path
import unittest

from camera_manager import CAMERAS, VideoDevice, choose_device, preview_command


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


if __name__ == "__main__":
    unittest.main()
