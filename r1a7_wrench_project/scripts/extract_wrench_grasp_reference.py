#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

RIGHT_ARM_SLICE = slice(7, 14)

DEFAULT_WINDOWS = {
    "RIGHT_SAFE": (1.0, 3.0),
    "PRE_GRASP_R": (6.0, 8.0),
    "GRASP_R": (15.0, 17.0),
    "LIFT_R": (25.0, 29.0),
}


def parse_vector(value: str) -> list[float]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"expected list, got {type(parsed).__name__}")
    return [float(v) for v in parsed]


def load_rows(episode_dir: Path) -> list[dict[str, Any]]:
    path = episode_dir / "states.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("arm_control_enabled") != "True":
                continue
            try:
                q = parse_vector(row["joint_position"])
                dq = parse_vector(row["joint_velocity"])
                cmd = parse_vector(row["command_position"])
                gripper = parse_vector(row["gripper_command"])
            except Exception:
                continue
            if len(q) != 14 or len(dq) != 14 or len(cmd) != 14 or len(gripper) < 2:
                continue
            rows.append(
                {
                    "t": float(row["timestamp_monotonic"]),
                    "q": q,
                    "dq": dq,
                    "cmd": cmd,
                    "right_trigger": float(row["right_trigger"]),
                    "gripper": gripper,
                }
            )

    if not rows:
        raise RuntimeError(f"no valid active rows in {path}")

    t0 = rows[0]["t"]
    for row in rows:
        row["elapsed"] = row["t"] - t0
    return rows


def select_window(rows: list[dict[str, Any]], start_s: float, end_s: float) -> list[dict[str, Any]]:
    selected = [r for r in rows if start_s <= r["elapsed"] <= end_s]
    if selected:
        return selected
    mid = 0.5 * (start_s + end_s)
    return [min(rows, key=lambda r: abs(r["elapsed"] - mid))]


def median_vector(vectors: list[list[float]]) -> list[float]:
    return [float(median(col)) for col in zip(*vectors)]


def episode_keyframes(rows: list[dict[str, Any]], windows: dict[str, tuple[float, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (start_s, end_s) in windows.items():
        selected = select_window(rows, start_s, end_s)
        out[name] = {
            "right_joint_position": median_vector([r["q"][RIGHT_ARM_SLICE] for r in selected]),
            "right_command_position": median_vector([r["cmd"][RIGHT_ARM_SLICE] for r in selected]),
            "right_gripper_position": float(median([r["gripper"][1] for r in selected])),
            "right_trigger": float(median([r["right_trigger"] for r in selected])),
            "sample_count": len(selected),
            "window_s": [start_s, end_s],
        }
    return out


def aggregate_keyframes(per_episode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = list(DEFAULT_WINDOWS)
    aggregated: dict[str, Any] = {}
    for name in names:
        items = [ep[name] for ep in per_episode.values()]
        aggregated[name] = {
            "source_episodes": list(per_episode),
            "aggregation": "median_of_episode_window_medians",
            "right_joint_position": median_vector([i["right_joint_position"] for i in items]),
            "right_gripper_position": float(median([i["right_gripper_position"] for i in items])),
            "right_trigger": float(median([i["right_trigger"] for i in items])),
            "window_s": list(DEFAULT_WINDOWS[name]),
        }
    return aggregated


def load_home_left(keyframes_path: Path) -> list[float]:
    data = yaml.safe_load(keyframes_path.read_text(encoding="utf-8"))
    return [float(v) for v in data["keyframes"]["HOME"]["left_joint_position"]]


def build_keyframe_yaml(aggregated: dict[str, Any], left_home: list[float]) -> dict[str, Any]:
    keyframes: dict[str, Any] = {}
    for name, item in aggregated.items():
        keyframes[name] = {
            "source_episodes": item["source_episodes"],
            "aggregation": item["aggregation"],
            "left_joint_position": left_home,
            "right_joint_position": item["right_joint_position"],
            "left_gripper_position": None,
            "right_gripper_position": item["right_gripper_position"],
            "right_trigger": item["right_trigger"],
            "description": f"V1 right-arm wrench grasp reference keyframe: {name}",
        }
    return {
        "keyframes": keyframes,
        "notes": [
            "Generated from wrench_grasp_001-008 by extract_wrench_grasp_reference.py.",
            "Left arm is fixed to HOME for the first right-arm autonomous baseline.",
            "Review with dry-run before any real robot execution.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract V1 right-arm wrench grasp reference keyframes from VR episodes."
    )
    parser.add_argument(
        "--episodes-root",
        type=Path,
        default=ROOT / "data" / "episodes",
    )
    parser.add_argument(
        "--episode-glob",
        default="wrench_grasp_00[1-8]",
    )
    parser.add_argument(
        "--home-keyframes",
        type=Path,
        default=ROOT / "config" / "keyframes.yaml",
    )
    parser.add_argument(
        "--output-keyframes",
        type=Path,
        default=ROOT / "config" / "wrench_grasp_reference_v1.yaml",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=ROOT / "results" / "wrench_grasp_reference_v1_report.json",
    )
    args = parser.parse_args()

    episodes = sorted(args.episodes_root.glob(args.episode_glob))
    if not episodes:
        raise RuntimeError(f"no episodes matched {args.episodes_root / args.episode_glob}")

    per_episode: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        rows = load_rows(episode)
        per_episode[episode.name] = episode_keyframes(rows, DEFAULT_WINDOWS)

    aggregated = aggregate_keyframes(per_episode)
    left_home = load_home_left(args.home_keyframes)
    yaml_data = build_keyframe_yaml(aggregated, left_home)

    args.output_keyframes.parent.mkdir(parents=True, exist_ok=True)
    args.output_keyframes.write_text(
        yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )

    report = {
        "episode_count": len(episodes),
        "episodes": [p.name for p in episodes],
        "windows": {k: list(v) for k, v in DEFAULT_WINDOWS.items()},
        "per_episode": per_episode,
        "aggregated": aggregated,
        "output_keyframes": str(args.output_keyframes),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"episodes       : {len(episodes)}")
    print(f"keyframes      : {args.output_keyframes}")
    print(f"report         : {args.output_report}")
    for name, item in aggregated.items():
        q = item["right_joint_position"]
        print(
            f"{name:12s} right_q="
            + "[" + ", ".join(f"{v:+.4f}" for v in q) + "] "
            + f"right_gripper={item['right_gripper_position']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
