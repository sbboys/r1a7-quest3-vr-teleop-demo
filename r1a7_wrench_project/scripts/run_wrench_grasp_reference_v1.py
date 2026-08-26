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
    "RIGHT_SAFE",
    "PRE_GRASP_R",
    "GRASP_R",
    "LIFT_SAFE_R",
]


def default_python_bin() -> str:
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "python"
        if candidate.exists():
            return str(candidate)

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
    print("R1-A7 WRENCH GRASP REFERENCE V1")
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
            f"{name:12s} right_q=["
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
        "--use-keyframe-gripper",
        "--force-direct",
        "--max-wrist-iterations",
        str(args.max_wrist_iterations),
        "--max-arm-iterations",
        str(args.max_arm_iterations),
    ]

    if args.execute:
        cmd.extend(
            [
                "--execute",
                "--assume-yes",
            ]
        )

    print("=" * 88)
    print(f"STAGE {keyframe}")
    print("=" * 88)
    print(" ".join(cmd))
    print()

    if args.dry_print:
        return 0

    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first fixed-scene right-arm wrench grasp baseline from "
            "demonstration-derived keyframes."
        )
    )
    parser.add_argument(
        "--keyframes",
        type=Path,
        default=ROOT / "config" / "wrench_grasp_reference_v1.yaml",
    )
    parser.add_argument(
        "--interface",
        default="enp6s0",
    )
    parser.add_argument(
        "--python-bin",
        default=default_python_bin(),
        help="Python interpreter used to run the DDS motion stage script.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Command the real robot. Without this flag each stage is dry-run preflight only.",
    )
    parser.add_argument(
        "--assume-yes",
        action="store_true",
        help="Skip this wrapper's real-robot confirmation prompt.",
    )
    parser.add_argument(
        "--dry-print",
        action="store_true",
        help="Only print generated stage commands; do not run even dry-run preflight.",
    )
    parser.add_argument(
        "--max-auto-duration",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--start-at",
        choices=SEQUENCE,
        default=SEQUENCE[0],
        help="Start execution from this keyframe in the reference sequence.",
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--stage-threshold",
        type=float,
        default=0.12,
        help="Per-stage final max joint error threshold in radians.",
    )
    parser.add_argument(
        "--max-wrist-iterations",
        type=int,
        default=16,
        help="Adaptive wrist recovery iteration budget for each grasp stage.",
    )
    parser.add_argument(
        "--max-arm-iterations",
        type=int,
        default=8,
        help="Adaptive arm recovery iteration budget for each grasp stage.",
    )
    parser.add_argument(
        "--inter-stage-wait-s",
        type=float,
        default=4.0,
        help="Settle time after each successful stage before starting the next preflight.",
    )
    args = parser.parse_args()

    keyframes = load_keyframes(args.keyframes)
    print_plan(keyframes, args.keyframes)

    if args.execute and not args.assume_yes:
        print("REAL ROBOT EXECUTION REQUESTED.")
        print("Before continuing: wrench is in the same fixed pose as demonstrations,")
        print("right arm path is clear, and no other LowCmd publisher is running.")
        phrase = input("Type EXECUTE WRENCH GRASP V1 exactly to continue: ")
        if phrase != "EXECUTE WRENCH GRASP V1":
            print("Confirmation rejected.")
            return 2
    elif args.execute:
        print("REAL ROBOT EXECUTION REQUESTED; wrapper confirmation skipped by --assume-yes.")
    else:
        print("DRY-RUN MODE: no robot command will be published by stage preflights.")
        print()

    start_index = SEQUENCE.index(args.start_at)
    active_sequence = SEQUENCE[start_index:]

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
    print("WRENCH_GRASP_REFERENCE_V1 = PASS")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
