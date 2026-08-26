#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robot.r1a7_interface import R1A7Interface


ARM_NAMES = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


def load_keyframe(path: Path, name: str):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    kf = data["keyframes"][name]

    q = np.asarray(
        kf["left_joint_position"]
        + kf["right_joint_position"],
        dtype=float,
    )

    if len(q) != 14:
        raise RuntimeError(
            f"{name} must contain 14 arm joints"
        )

    return q


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "keyframe",
        nargs="?",
        default="HOME",
    )

    parser.add_argument(
        "--interface",
        default="enp6s0",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--keyframes",
        type=Path,
        default=Path(
            "r1a7_wrench_project/config/keyframes.yaml"
        ),
    )

    args = parser.parse_args()

    q_target = load_keyframe(
        args.keyframes,
        args.keyframe,
    )

    robot = R1A7Interface(
        dry_run=True,
        enable_robot_motion=False,
        interface=args.interface,
    )

    robot.connect()

    deadline = time.monotonic() + args.timeout
    state = None

    while time.monotonic() < deadline:
        state = robot.get_robot_state()

        if (
            state.communication_ok
            and len(state.joint_position) == 14
        ):
            break

        time.sleep(0.02)

    if (
        state is None
        or not state.communication_ok
        or len(state.joint_position) != 14
    ):
        print("ERROR: no valid lowstate received")
        return 2

    q_current = np.asarray(
        state.joint_position,
        dtype=float,
    )

    error = q_current - q_target
    abs_error = np.abs(error)

    max_idx = int(np.argmax(abs_error))
    max_error = float(abs_error[max_idx])

    print()
    print("=" * 94)
    print(
        f"R1-A7 LIVE KEYFRAME CHECK: {args.keyframe}"
    )
    print("=" * 94)

    print(
        f"{'joint':28s}"
        f"{'target':>12s}"
        f"{'current':>12s}"
        f"{'error':>12s}"
        f"{'deg':>10s}"
        f"{'result':>10s}"
    )

    print("-" * 94)

    for i, name in enumerate(ARM_NAMES):
        passed = abs_error[i] < args.threshold

        print(
            f"{name:28s}"
            f"{q_target[i]:+12.6f}"
            f"{q_current[i]:+12.6f}"
            f"{error[i]:+12.6f}"
            f"{np.degrees(error[i]):+10.3f}"
            f"{('PASS' if passed else 'FAIL'):>10s}"
        )

    print("-" * 94)

    print(
        f"MAX error : {max_error:.6f} rad "
        f"({np.degrees(max_error):.3f} deg)"
    )

    print(
        f"Worst     : {ARM_NAMES[max_idx]}"
    )

    print()

    if max_error < args.threshold:
        print(
            f"{args.keyframe}_READY = PASS"
        )
        result = 0
    else:
        print(
            f"{args.keyframe}_READY = FAIL"
        )
        result = 2

    print("=" * 94)
    print()

    return result


if __name__ == "__main__":
    raise SystemExit(main())
