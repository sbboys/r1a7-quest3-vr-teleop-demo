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


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_keyframe(path: Path, name: str):
    data = load_yaml(path)

    if "keyframes" not in data:
        raise RuntimeError("Missing 'keyframes' section")

    if name not in data["keyframes"]:
        raise RuntimeError(
            f"Keyframe '{name}' not found"
        )

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


def wait_for_state(robot, timeout):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        state = robot.get_robot_state()

        if (
            state.communication_ok
            and len(state.joint_position) == 14
        ):
            return state

        time.sleep(0.02)

    raise RuntimeError(
        "No valid rt/lowstate received"
    )


def quintic_profile(u):
    return (
        10.0 * u**3
        - 15.0 * u**4
        + 6.0 * u**5
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "DRY-RUN keyframe motion planner. "
            "This script never publishes robot commands."
        )
    )

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
        "--keyframes",
        type=Path,
        default=Path(
            "r1a7_wrench_project/config/keyframes.yaml"
        ),
    )

    parser.add_argument(
        "--safety",
        type=Path,
        default=Path(
            "r1a7_wrench_project/config/safety.yaml"
        ),
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Requested trajectory duration. "
            "If omitted, choose automatically."
        ),
    )

    parser.add_argument(
        "--minimum-duration",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load target and safety configuration
    # --------------------------------------------------

    q_target = load_keyframe(
        args.keyframes,
        args.keyframe,
    )

    safety = load_yaml(
        args.safety
    )

    frequency = float(
        safety["control_frequency_hz"]
    )

    max_step = float(
        safety["max_joint_step_rad"]
    )

    max_velocity = float(
        safety["max_joint_velocity_rad_s"]
    )

    max_acceleration = float(
        safety["max_joint_acceleration_rad_s2"]
    )

    # --------------------------------------------------
    # Read real robot state
    # READ ONLY
    # --------------------------------------------------

    robot = R1A7Interface(
        dry_run=True,
        enable_robot_motion=False,
        interface=args.interface,
    )

    robot.connect()

    state = wait_for_state(
        robot,
        args.timeout,
    )

    q_current = np.asarray(
        state.joint_position,
        dtype=float,
    )

    delta = q_target - q_current
    abs_delta = np.abs(delta)

    # --------------------------------------------------
    # Quintic trajectory:
    #
    # h(u) = 10u^3 - 15u^4 + 6u^5
    #
    # max h'(u)  = 1.875
    # max |h''|  = 5.77350269
    # --------------------------------------------------

    max_delta = float(
        np.max(abs_delta)
    )

    velocity_duration = (
        1.875
        * max_delta
        / max_velocity
    )

    acceleration_duration = np.sqrt(
        5.7735026919
        * max_delta
        / max_acceleration
    )

    theoretical_min_duration = max(
        velocity_duration,
        acceleration_duration,
    )

    if args.duration is None:
        duration = max(
            args.minimum_duration,
            theoretical_min_duration * 1.20,
        )
    else:
        duration = float(
            args.duration
        )

    if duration <= 0.0:
        raise RuntimeError(
            "Duration must be > 0"
        )

    # --------------------------------------------------
    # Generate complete dry-run trajectory
    # --------------------------------------------------

    sample_count = max(
        2,
        int(np.ceil(duration * frequency)) + 1,
    )

    t = np.linspace(
        0.0,
        duration,
        sample_count,
    )

    u = t / duration

    h = quintic_profile(u)

    trajectory = (
        q_current[None, :]
        + h[:, None] * delta[None, :]
    )

    step_delta = np.diff(
        trajectory,
        axis=0,
    )

    max_discrete_step = float(
        np.max(np.abs(step_delta))
    )

    # Analytical peak velocity / acceleration
    peak_velocity_each = (
        1.875
        * abs_delta
        / duration
    )

    peak_acceleration_each = (
        5.7735026919
        * abs_delta
        / (duration**2)
    )

    peak_velocity = float(
        np.max(peak_velocity_each)
    )

    peak_acceleration = float(
        np.max(peak_acceleration_each)
    )

    worst_idx = int(
        np.argmax(abs_delta)
    )

    # --------------------------------------------------
    # Safety checks
    # --------------------------------------------------

    velocity_ok = (
        peak_velocity <= max_velocity
    )

    acceleration_ok = (
        peak_acceleration <= max_acceleration
    )

    step_ok = (
        max_discrete_step <= max_step
    )

    duration_ok = (
        duration >= theoretical_min_duration
    )

    safety_pass = all(
        [
            velocity_ok,
            acceleration_ok,
            step_ok,
            duration_ok,
        ]
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print()
    print("=" * 110)
    print(
        f"R1-A7 KEYFRAME MOTION PLAN - DRY RUN: "
        f"{args.keyframe}"
    )
    print("=" * 110)

    print()
    print("IMPORTANT:")
    print(
        "  This script is READ ONLY."
    )
    print(
        "  It does NOT publish rt/lowcmd."
    )
    print(
        "  It does NOT move the robot."
    )

    print()
    print(
        f"Control frequency       : "
        f"{frequency:.1f} Hz"
    )

    print(
        f"Max joint velocity      : "
        f"{max_velocity:.6f} rad/s"
    )

    print(
        f"Max joint acceleration  : "
        f"{max_acceleration:.6f} rad/s^2"
    )

    print(
        f"Max joint step          : "
        f"{max_step:.6f} rad"
    )

    print()
    print(
        f"Theoretical min duration: "
        f"{theoretical_min_duration:.3f} s"
    )

    print(
        f"Planned duration        : "
        f"{duration:.3f} s"
    )

    print(
        f"Trajectory samples      : "
        f"{sample_count}"
    )

    print()

    print(
        f"{'joint':28s}"
        f"{'current':>12s}"
        f"{'target':>12s}"
        f"{'delta':>12s}"
        f"{'deg':>10s}"
    )

    print("-" * 80)

    for i, name in enumerate(
        ARM_NAMES
    ):
        print(
            f"{name:28s}"
            f"{q_current[i]:+12.6f}"
            f"{q_target[i]:+12.6f}"
            f"{delta[i]:+12.6f}"
            f"{np.degrees(delta[i]):+10.3f}"
        )

    print("-" * 80)

    print()
    print(
        f"Largest motion          : "
        f"{max_delta:.6f} rad "
        f"({np.degrees(max_delta):.3f} deg)"
    )

    print(
        f"Worst joint             : "
        f"{ARM_NAMES[worst_idx]}"
    )

    print()
    print(
        f"Predicted peak velocity : "
        f"{peak_velocity:.6f} rad/s"
    )

    print(
        f"Velocity limit          : "
        f"{max_velocity:.6f} rad/s "
        f"[{'PASS' if velocity_ok else 'FAIL'}]"
    )

    print()
    print(
        f"Predicted peak accel    : "
        f"{peak_acceleration:.6f} rad/s^2"
    )

    print(
        f"Acceleration limit      : "
        f"{max_acceleration:.6f} rad/s^2 "
        f"[{'PASS' if acceleration_ok else 'FAIL'}]"
    )

    print()
    print(
        f"Max discrete step       : "
        f"{max_discrete_step:.6f} rad"
    )

    print(
        f"Step limit              : "
        f"{max_step:.6f} rad "
        f"[{'PASS' if step_ok else 'FAIL'}]"
    )

    print()
    print(
        f"Duration check          : "
        f"{'PASS' if duration_ok else 'FAIL'}"
    )

    print()
    print("=" * 110)

    if safety_pass:
        print(
            "TRAJECTORY_SAFETY = PASS"
        )
    else:
        print(
            "TRAJECTORY_SAFETY = FAIL"
        )

    print(
        "ROBOT_COMMAND_PUBLISHED = FALSE"
    )

    print("=" * 110)
    print()

    return (
        0
        if safety_pass
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())

