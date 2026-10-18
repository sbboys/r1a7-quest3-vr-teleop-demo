#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEAD_SERVICE_PATH = PROJECT_ROOT / "vision" / "r1a7_head_service_prepare.py"


def load_head_service_module():
    spec = importlib.util.spec_from_file_location(
        "r1a7_head_service_state_test_module", HEAD_SERVICE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HeadServiceStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.head_service = load_head_service_module()

    def test_restart_does_not_overwrite_original_service_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "head-service-state.json"
            with patch.object(self.head_service, "STATE_PATH", state_path):
                self.head_service.save_original_state(
                    {"video_hub": 0, "stereo_patch_pc1": 1}, "enp6s0"
                )
                first = json.loads(state_path.read_text(encoding="utf-8"))

                self.head_service.save_original_state(
                    {"video_hub": 1, "stereo_patch_pc1": 0}, "enp6s0"
                )
                retained = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(retained, first)
        self.assertEqual(retained["video_hub"], 0)
        self.assertEqual(retained["stereo_patch_pc1"], 1)


if __name__ == "__main__":
    unittest.main()
