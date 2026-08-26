#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_logger import EpisodeRecorder
from src.robot.r1a7_interface import R1A7Interface


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record R1-A7 rt/lowstate while the existing handset teleop script controls the robot"
    )
    parser.add_argument("--interface", default="enp6s0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--state-topic", default="rt/lowstate")
    parser.add_argument("--episode-id", default=f"episode_real_lowstate_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--operator", default="unknown")
    args = parser.parse_args()

    robot = R1A7Interface(
        dry_run=True,
        enable_robot_motion=False,
        interface=args.interface,
        domain_id=args.domain_id,
        state_topic=args.state_topic,
    )
    robot.connect()

    recorder = EpisodeRecorder(Path("data/episodes"))
    recorder.start(
        args.episode_id,
        {
            "robot": "Unitree R1-A7",
            "task": "wrench_nut_operation",
            "operator": args.operator,
            "date": time.strftime("%Y-%m-%d"),
            "sdk_version": "unknown",
            "firmware_version": "unknown",
            "control_frequency": args.rate_hz,
            "success": None,
            "failure_reason": None,
            "wrench_type": None,
            "nut_size": None,
            "fixture_version": None,
            "camera_available": False,
            "recording_mode": "read_only_lowstate_during_handset_teleop",
        },
    )
    recorder.record_event(
        "recording_started",
        {
            "interface": args.interface,
            "domain_id": args.domain_id,
            "state_topic": args.state_topic,
            "publishes_commands": False,
        },
    )

    deadline = time.monotonic() + max(0.0, args.duration)
    dt = 1.0 / max(1.0, args.rate_hz)
    samples = 0
    while time.monotonic() < deadline:
        state = robot.get_robot_state()
        recorder.record_state(state, "handset_teleop")
        samples += 1
        time.sleep(dt)

    recorder.record_event("recording_finished", {"samples": samples})
    print(recorder.episode_dir)
    print(f"samples={samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
