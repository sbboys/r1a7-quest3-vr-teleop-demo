#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robot.mock_robot_interface import MockRobotInterface
from src.robot.r1a7_interface import R1A7Interface
from src.robot.safety_monitor import SafetyConfig, SafetyMonitor
from src.task.wrench_task_fsm import WrenchTaskFSM


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the R1-A7 wrench FSM skeleton")
    parser.add_argument("--backend", choices=["mock", "real"], default="mock")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--enable-robot-motion", action="store_true", default=False)
    args = parser.parse_args()

    safety = SafetyMonitor(SafetyConfig(dry_run=args.dry_run, enable_robot_motion=args.enable_robot_motion))
    robot = MockRobotInterface() if args.backend == "mock" else R1A7Interface(args.dry_run, args.enable_robot_motion)
    robot.connect()
    robot.acquire_arm_control()
    fsm = WrenchTaskFSM(robot=robot, safety=safety)
    history = fsm.run_dry()
    for state in history:
        print(state.value)
    robot.safe_stop()
    return 0 if fsm.state.value == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
