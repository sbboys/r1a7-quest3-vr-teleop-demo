#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robot.mock_robot_interface import MockRobotInterface


def _load_json_cell(row: dict[str, str], key: str, default: Any) -> Any:
    value = row.get(key, "")
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {key}: {value!r}") from exc


def _read_valid_rows(states_csv: Path) -> list[dict[str, str]]:
    with states_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    valid = []
    for row in rows:
        if row.get("communication_ok") != "True":
            continue
        joints = _load_json_cell(row, "joint_position", [])
        if isinstance(joints, list) and len(joints) == 14:
            valid.append(row)
    if not valid:
        raise RuntimeError(f"no valid 14-joint communication_ok=True rows found in {states_csv}")
    return valid


def _select_row(rows: list[dict[str, str]], sample_index: int | None, nearest_time: float | None) -> dict[str, str]:
    if sample_index is not None and nearest_time is not None:
        raise ValueError("use only one of --sample-index or --nearest-time")
    if sample_index is not None:
        if sample_index < 0:
            sample_index = len(rows) + sample_index
        if sample_index < 0 or sample_index >= len(rows):
            raise IndexError(f"sample index out of range for {len(rows)} valid rows")
        return rows[sample_index]
    if nearest_time is not None:
        return min(rows, key=lambda row: abs(float(row["timestamp_system"]) - nearest_time))
    return rows[-1]


def _block_from_state_row(name: str, row: dict[str, str], description: str) -> dict[str, Any]:
    joints = _load_json_cell(row, "joint_position", [])
    gripper_position = _load_json_cell(row, "gripper_position", {})
    return {
        name: {
            "timestamp_system": float(row["timestamp_system"]),
            "source_fsm_state": row.get("fsm_state", "unknown"),
            "left_joint_position": joints[:7],
            "right_joint_position": joints[7:14],
            "left_gripper_position": gripper_position.get("left"),
            "right_gripper_position": gripper_position.get("right"),
            "description": description,
        }
    }


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9g}"
    return json.dumps(value, ensure_ascii=False)


def _format_keyframe_yaml(block: dict[str, Any]) -> str:
    lines = ["keyframes:"]
    for name, data in block.items():
        lines.append(f"  {name}:")
        for key, value in data.items():
            if isinstance(value, list):
                payload = ", ".join(_format_scalar(item) for item in value)
                lines.append(f"    {key}: [{payload}]")
            else:
                lines.append(f"    {key}: {_format_scalar(value)}")
    lines.extend(
        [
            "notes:",
            "  - Generated from a read-only lowstate episode.",
            "  - Verify each keyframe on the real robot before using it for scripted motion.",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_keyframe_entry(name: str, data: dict[str, Any]) -> str:
    lines = [f"  {name}:"]
    for key, value in data.items():
        if isinstance(value, list):
            payload = ", ".join(_format_scalar(item) for item in value)
            lines.append(f"    {key}: [{payload}]")
        else:
            lines.append(f"    {key}: {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _write_keyframe(output: Path, block: dict[str, Any]) -> None:
    name, data = next(iter(block.items()))
    if output.exists():
        text = output.read_text(encoding="utf-8")
        if f"  {name}:" in text:
            raise RuntimeError(f"{output} already contains keyframe {name!r}")
        entry = _format_keyframe_entry(name, data)
        if "keyframes: {}" in text:
            text = text.replace("keyframes: {}", "keyframes:\n" + entry.rstrip(), 1)
        elif "\nnotes:" in text:
            text = text.replace("\nnotes:", "\n" + entry + "notes:", 1)
        else:
            text = text.rstrip() + "\n" + entry
        output.write_text(text, encoding="utf-8")
        return
    output.write_text(_format_keyframe_yaml(block), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a keyframe from the current robot state")
    parser.add_argument("name")
    parser.add_argument("--output", default=str(ROOT / "config" / "keyframes.yaml"))
    parser.add_argument("--backend", choices=["mock"], default="mock")
    parser.add_argument("--from-episode", type=Path, help="Episode directory containing states.csv")
    parser.add_argument("--sample-index", type=int, help="0-based valid-row index; negative indexes count from the end")
    parser.add_argument("--nearest-time", type=float, help="Select the valid row closest to this timestamp_system value")
    parser.add_argument("--description", default="")
    parser.add_argument("--write", action="store_true", help="Write to --output when it has no existing keyframes")
    args = parser.parse_args()

    if args.from_episode:
        states_csv = args.from_episode / "states.csv"
        rows = _read_valid_rows(states_csv)
        row = _select_row(rows, args.sample_index, args.nearest_time)
        description = args.description or f"captured from {args.from_episode}"
        block = _block_from_state_row(args.name, row, description)
    else:
        robot = MockRobotInterface()
        robot.connect()
        state = robot.get_robot_state()
        block = {
            args.name: {
                "timestamp_system": time.time(),
                "left_joint_position": state.joint_position[:7],
                "right_joint_position": state.joint_position[7:14],
                "left_gripper_position": state.gripper_position.get("left"),
                "right_gripper_position": state.gripper_position.get("right"),
                "description": args.description or "mock capture; use --from-episode for real read-only logs",
            }
        }

    if args.write:
        _write_keyframe(Path(args.output), block)
        print(f"Wrote {args.name} to {args.output}")
        return 0

    print(json.dumps(block, ensure_ascii=False, indent=2))
    print(f"Not writing automatically. Add --write to replace an empty keyframes file at {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
