#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_logger import EpisodeRecorder
from src.robot.mock_robot_interface import MockRobotInterface


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a mock teleoperation episode without changing teleop behavior")
    parser.add_argument("--episode-id", default=f"episode_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()

    robot = MockRobotInterface()
    robot.connect()
    recorder = EpisodeRecorder(Path("data/episodes"))
    recorder.start(
        args.episode_id,
        {
            "robot": "Unitree R1-A7",
            "task": "wrench_nut_operation",
            "operator": "unknown",
            "date": time.strftime("%Y-%m-%d"),
            "camera_available": False,
            "success": None,
            "failure_reason": None,
        },
    )
    end = time.monotonic() + args.duration
    while time.monotonic() < end:
        recorder.record_state(robot.get_robot_state(), "mock_record")
        time.sleep(0.02)
    recorder.record_event("episode_end", {"mode": "mock"})
    print(recorder.episode_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
