#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

SEQUENCE = [
    "TEACH_START_R",
    "ELBOW_UP_R",
    "ARM_UP_R",
    "PRE_GRASP_HIGH_R",
    "GRASP_NEAR_R",
    "CLOSE_GRIPPER_R",
    "POST_GRASP_LIFT_R",
    "PLACE_DOWN_R",
    "OPEN_GRIPPER_R",
]


def default_python_bin() -> str:
    tv_python = Path.home() / "miniconda3" / "envs" / "tv" / "bin" / "python"
    if tv_python.exists():
        return str(tv_python)
    return sys.executable


def load_keyframes(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "keyframes" not in data:
        raise RuntimeError(f"invalid keyframe file: {path}")
    return data["keyframes"]


def print_plan(keyframes: dict, keyframe_path: Path) -> None:
    print("=" * 88)
    print("R1-A7 WRENCH GRASP REFERENCE V2 HIGH-APPROACH")
    print("=" * 88)
    print(f"keyframes : {keyframe_path}")
    print("sequence  : " + " -> ".join(SEQUENCE))
    print()
    for name in SEQUENCE:
        if name not in keyframes:
            raise RuntimeError(f"missing keyframe: {name}")
        kf = keyframes[name]
        rq = kf.get("right_joint_position")
        if not isinstance(rq, list) or len(rq) != 7:
            raise RuntimeError(f"{name}: right_joint_position must contain 7 values")
        print(
            f"{name:18s} right_q=["
            + ", ".join(f"{float(v):+.4f}" for v in rq)
            + f"] right_gripper={kf.get('right_gripper_position')}"
        )
    print()


def run_stage(args: argparse.Namespace, keyframe: str) -> int:
    cmd = [
        args.python_bin,
        str(ROOT / "scripts" / "auto_move_to_keyframe_v2_1.py"),
        keyframe,
        "--interface",
        args.interface,
        "--keyframes",
        str(args.keyframes),
        "--max-auto-duration",
        str(args.max_auto_duration),
        "--hold-time",
        str(args.hold_time),
        "--home-threshold",
        str(args.stage_threshold),
        "--state-timeout",
        str(args.state_timeout),
        "--hold-left-current",
        "--use-keyframe-gripper",
        "--force-direct",
        "--max-wrist-iterations",
        str(args.max_wrist_iterations),
        "--max-arm-iterations",
        str(args.max_arm_iterations),
    ]
    if args.execute:
        cmd.extend(["--execute", "--assume-yes"])

    print("=" * 88)
    print(f"STAGE {keyframe}")
    print("=" * 88)
    print(" ".join(cmd))
    print()
    if args.dry_print:
        return 0
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def run_continuous_sequence(args: argparse.Namespace) -> int:
    cmd = [
        args.python_bin,
        str(ROOT / "scripts" / "auto_move_to_keyframe_v2_1.py"),
        "--sequence",
        ",".join(SEQUENCE[SEQUENCE.index(args.start_at) :]),
        "--interface",
        args.interface,
        "--keyframes",
        str(args.keyframes),
        "--max-auto-duration",
        str(args.max_auto_duration),
        "--hold-time",
        str(args.hold_time),
        "--home-threshold",
        str(args.stage_threshold),
        "--state-timeout",
        str(args.state_timeout),
        "--hold-left-current",
        "--use-keyframe-gripper",
        "--force-direct",
        "--max-wrist-iterations",
        str(args.max_wrist_iterations),
        "--max-arm-iterations",
        str(args.max_arm_iterations),
    ]
    if args.execute:
        cmd.extend(["--execute", "--assume-yes"])

    print("=" * 88)
    print("CONTINUOUS LOWCMD SEQUENCE")
    print("=" * 88)
    print(" ".join(cmd))
    print()
    if args.dry_print:
        return 0
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run V2 high-approach right-arm wrench grasp replay."
    )
    parser.add_argument(
        "--keyframes",
        type=Path,
        default=ROOT / "config" / "wrench_grasp_reference_v2_high.yaml",
    )
    parser.add_argument("--interface", default="enp6s0")
    parser.add_argument("--python-bin", default=default_python_bin())
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--assume-yes", action="store_true")
    parser.add_argument("--dry-print", action="store_true")
    parser.add_argument("--max-auto-duration", type=float, default=90.0)
    parser.add_argument("--start-at", choices=SEQUENCE, default=SEQUENCE[0])
    parser.add_argument("--state-timeout", type=float, default=0.50)
    parser.add_argument("--hold-time", type=float, default=3.0)
    parser.add_argument("--stage-threshold", type=float, default=0.12)
    parser.add_argument("--max-wrist-iterations", type=int, default=16)
    parser.add_argument("--max-arm-iterations", type=int, default=8)
    parser.add_argument("--inter-stage-wait-s", type=float, default=4.0)
    parser.add_argument(
        "--legacy-subprocess-stages",
        action="store_true",
        help=(
            "Run each keyframe as a separate process. This is only for diagnostics; "
            "normal grasp replay should keep one continuous LowCmd session."
        ),
    )
    args = parser.parse_args()

    keyframes = load_keyframes(args.keyframes)
    print_plan(keyframes, args.keyframes)

    if args.execute and not args.assume_yes:
        print("REAL ROBOT EXECUTION REQUESTED.")
        phrase = input("Type EXECUTE WRENCH GRASP V2 exactly to continue: ")
        if phrase != "EXECUTE WRENCH GRASP V2":
            print("Confirmation rejected.")
            return 2
    elif args.execute:
        print("REAL ROBOT EXECUTION REQUESTED; wrapper confirmation skipped by --assume-yes.")
    else:
        print("DRY-RUN MODE: no robot command will be published by stage preflights.")
        print()

    if not args.legacy_subprocess_stages:
        rc = run_continuous_sequence(args)
        if rc != 0:
            print(f"ABORT: continuous sequence exited with code {rc}")
            return rc
        print("=" * 88)
        print("WRENCH_GRASP_REFERENCE_V2_HIGH = PASS")
        print("=" * 88)
        return 0

    active_sequence = SEQUENCE[SEQUENCE.index(args.start_at) :]
    for index, keyframe in enumerate(active_sequence):
        rc = run_stage(args, keyframe)
        if rc != 0:
            print(f"ABORT: stage {keyframe} exited with code {rc}")
            return rc
        if index + 1 < len(active_sequence) and args.inter_stage_wait_s > 0:
            print()
            print(f"Waiting {args.inter_stage_wait_s:.1f}s for joint velocity to settle...")
            time.sleep(args.inter_stage_wait_s)

    print("=" * 88)
    print("WRENCH_GRASP_REFERENCE_V2_HIGH = PASS")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
