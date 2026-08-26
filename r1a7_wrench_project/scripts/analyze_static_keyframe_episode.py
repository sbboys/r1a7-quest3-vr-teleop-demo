#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


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


def _load_keyframe_q(path: Path, name: str) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    keyframe = data["keyframes"][name]
    q = np.asarray(
        keyframe["left_joint_position"] + keyframe["right_joint_position"],
        dtype=float,
    )
    if q.shape != (14,):
        raise RuntimeError(f"{name} should contain 14 arm joints, got {q.shape}")
    return q


def _json_cell(row: dict[str, str], key: str, default: Any) -> Any:
    value = row.get(key, "")
    if not value:
        return default
    return json.loads(value)


def _load_episode(
    episode_dir: Path,
    trim_seconds: float,
    static_dq_threshold: float,
) -> dict[str, Any]:
    states_csv = episode_dir / "states.csv"
    if not states_csv.exists():
        raise FileNotFoundError(states_csv)

    timestamps = []
    q_rows = []
    dq_rows = []
    left_gripper = []
    right_gripper = []
    total_rows = 0
    invalid_rows = 0

    with states_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            total_rows += 1
            if row.get("communication_ok") != "True":
                invalid_rows += 1
                continue
            try:
                q = _json_cell(row, "joint_position", [])
                dq = _json_cell(row, "joint_velocity", [])
                gripper = _json_cell(row, "gripper_position", {})
            except Exception:
                invalid_rows += 1
                continue
            if len(q) != 14 or len(dq) != 14:
                invalid_rows += 1
                continue
            timestamps.append(float(row["timestamp_monotonic"]))
            q_rows.append(q)
            dq_rows.append(dq)
            left_gripper.append(float(gripper.get("left", np.nan)))
            right_gripper.append(float(gripper.get("right", np.nan)))

    if len(q_rows) < 10:
        raise RuntimeError("Too few valid 14-joint samples")

    t = np.asarray(timestamps, dtype=float)
    q = np.asarray(q_rows, dtype=float)
    dq = np.asarray(dq_rows, dtype=float)

    start_t = t[0] + trim_seconds
    end_t = t[-1] - trim_seconds
    central_mask = (t >= start_t) & (t <= end_t)
    if np.count_nonzero(central_mask) < 10:
        raise RuntimeError("Not enough central-window samples")

    q_c = q[central_mask]
    dq_c = dq[central_mask]
    left_g_c = np.asarray(left_gripper, dtype=float)[central_mask]
    right_g_c = np.asarray(right_gripper, dtype=float)[central_mask]
    max_abs_dq_per_sample = np.max(np.abs(dq_c), axis=1)
    static_mask = max_abs_dq_per_sample < static_dq_threshold

    if np.count_nonzero(static_mask) >= 20:
        q_static = q_c[static_mask]
        left_g_static = left_g_c[static_mask]
        right_g_static = right_g_c[static_mask]
    else:
        q_static = q_c
        left_g_static = left_g_c
        right_g_static = right_g_c

    return {
        "total_rows": total_rows,
        "invalid_rows": invalid_rows,
        "valid_rows": len(q),
        "central_rows": len(q_c),
        "duration_s": float(t[-1] - t[0]),
        "static_rows": int(np.count_nonzero(static_mask)),
        "static_fraction": float(np.count_nonzero(static_mask) / len(static_mask)),
        "max_abs_dq_overall": float(np.max(max_abs_dq_per_sample)),
        "p95_max_abs_dq": float(np.percentile(max_abs_dq_per_sample, 95)),
        "q_median": np.median(q_static, axis=0),
        "q_span90": np.percentile(q_static, 95, axis=0)
        - np.percentile(q_static, 5, axis=0),
        "dq_abs_p95": np.percentile(np.abs(dq_c), 95, axis=0),
        "left_gripper_median": float(np.nanmedian(left_g_static)),
        "right_gripper_median": float(np.nanmedian(right_g_static)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a static keyframe lowstate episode and compare it with HOME."
    )
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--name", default="RIGHT_SAFE")
    parser.add_argument(
        "--keyframes",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "keyframes.yaml",
    )
    parser.add_argument("--reference", default="HOME")
    parser.add_argument("--trim-seconds", type=float, default=1.0)
    parser.add_argument("--static-dq-threshold", type=float, default=0.05)
    args = parser.parse_args()

    stats = _load_episode(
        args.episode_dir,
        trim_seconds=args.trim_seconds,
        static_dq_threshold=args.static_dq_threshold,
    )
    q_reference = _load_keyframe_q(args.keyframes, args.reference)
    q_median = stats["q_median"]
    delta = q_median - q_reference
    worst_idx = int(np.argmax(np.abs(delta)))

    print("=" * 72)
    print(f"R1-A7 {args.name} STATIC QUALITY")
    print("=" * 72)
    print(f"episode           : {args.episode_dir}")
    print(f"reference         : {args.reference} from {args.keyframes}")
    print(f"total CSV rows    : {stats['total_rows']}")
    print(f"invalid rows      : {stats['invalid_rows']}")
    print(f"valid rows        : {stats['valid_rows']}")
    print(f"central samples   : {stats['central_rows']}")
    print(f"duration valid    : {stats['duration_s']:.3f} s")
    print(f"static samples    : {stats['static_rows']} / {stats['central_rows']}")
    print(f"static fraction   : {100.0 * stats['static_fraction']:.2f}%")
    print(f"max |dq| overall  : {stats['max_abs_dq_overall']:.6f} rad/s")
    print(f"P95 max |dq|      : {stats['p95_max_abs_dq']:.6f} rad/s")
    print(f"max P95-P05 q span: {np.max(stats['q_span90']):.6f} rad")
    print(f"max |{args.name}-{args.reference}|: {np.max(np.abs(delta)):.6f} rad")
    print(f"worst delta joint : {ARM_NAMES[worst_idx]} ({delta[worst_idx]:+.6f} rad)")
    print()
    print("Joint statistics:")
    print(
        f"{'joint':28s} {'q_median':>11s} {'q_span90':>11s} "
        f"{'dq_p95':>11s} {'delta_ref':>11s}"
    )
    print("-" * 78)
    for i, joint in enumerate(ARM_NAMES):
        print(
            f"{joint:28s} {q_median[i]:+11.6f} {stats['q_span90'][i]:11.6f} "
            f"{stats['dq_abs_p95'][i]:11.6f} {delta[i]:+11.6f}"
        )
    print()
    print(f"{args.name} candidate:")
    print("left_joint_position:")
    print("  [" + ", ".join(f"{v:.9f}" for v in q_median[:7]) + "]")
    print("right_joint_position:")
    print("  [" + ", ".join(f"{v:.9f}" for v in q_median[7:]) + "]")
    print(f"left_gripper_position: {stats['left_gripper_median']:.9f}")
    print(f"right_gripper_position: {stats['right_gripper_median']:.9f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
