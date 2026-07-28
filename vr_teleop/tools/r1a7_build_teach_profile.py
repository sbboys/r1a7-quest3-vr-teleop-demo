#!/usr/bin/env python3
"""Build a camera-control profile from R1-A7 right-arm teach-in data."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


JOINT_NAMES = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


def latest_session(root: Path) -> Path:
    sessions = sorted([p for p in root.iterdir() if p.is_dir()])
    if not sessions:
        raise FileNotFoundError(f"no teach-in sessions under {root}")
    return sessions[-1]


def q_list(item: dict) -> list[float]:
    q = item["q"]
    return [float(q[name]) for name in JOINT_NAMES]


def vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [float(x - y) for x, y in zip(a, b)]


def build_profile(session: Path) -> dict:
    samples_path = session / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    grouped = defaultdict(list)
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("recording"):
                continue
            motion = item.get("motion") or item.get("label") or "unlabeled"
            grouped[motion].append(item)

    motions = {}
    for motion, items in sorted(grouped.items()):
        if len(items) < 5:
            continue
        selected_joint = items[0].get("selected_joint", "")
        if selected_joint not in JOINT_NAMES:
            continue
        selected_index = JOINT_NAMES.index(selected_joint)
        selected_values = [float(item["q"][selected_joint]) for item in items]
        min_i = min(range(len(items)), key=lambda i: selected_values[i])
        max_i = max(range(len(items)), key=lambda i: selected_values[i])

        base_q = q_list(items[0])
        min_q = q_list(items[min_i])
        max_q = q_list(items[max_i])
        end_q = q_list(items[-1])
        motions[motion] = {
            "label": items[0].get("label", motion),
            "selected_joint": selected_joint,
            "selected_joint_index": selected_index,
            "samples": len(items),
            "duration_s": float(items[-1]["t"]) - float(items[0]["t"]),
            "base_q": base_q,
            "min_q": min_q,
            "max_q": max_q,
            "end_q": end_q,
            "min_delta": vec_sub(min_q, base_q),
            "max_delta": vec_sub(max_q, base_q),
            "end_delta": vec_sub(end_q, base_q),
            "selected_min": min(selected_values),
            "selected_max": max(selected_values),
            "selected_base": selected_values[0],
            "selected_range": max(selected_values) - min(selected_values),
        }

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_session": str(session),
        "joint_names": JOINT_NAMES,
        "motions": motions,
        "camera_mapping": {
            "reach": {
                "motion": "shoulder_pitch_forward_back",
                "positive_branch": "min",
                "input_scale": 0.45,
            },
            "horizontal_roll": {
                "motion": "shoulder_roll_out_in",
                "positive_branch": "min",
                "input_scale": 0.28,
                "weight": 0.30,
            },
            "horizontal_yaw": {
                "motion": "shoulder_yaw_twist",
                "positive_branch": "max",
                "input_scale": 0.20,
                "weight": 0.85,
            },
            "elbow_bend": {
                "motion": "elbow_bend_extend",
                "positive_branch": "max",
                "input_scale": 1.20,
            },
            "wrist_roll": {
                "motion": "wrist_roll_rotate",
                "positive_branch": "max",
                "input_scale": 1.00,
            },
            "wrist_pitch": {
                "motion": "wrist_pitch_up_down",
                "positive_branch": "max",
                "input_scale": 0.80,
            },
            "wrist_yaw": {
                "motion": "wrist_yaw_left_right",
                "positive_branch": "max",
                "input_scale": 0.80,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build R1-A7 camera teach profile")
    parser.add_argument("session", nargs="?", default="", help="teach session dir; default uses latest")
    parser.add_argument("--root", default="data/r1a7_teach")
    parser.add_argument("--output", default="", help="output JSON; default writes session/teach_profile.json")
    parser.add_argument("--latest-copy", default="data/r1a7_teach/latest_profile.json")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    session = Path(args.session).expanduser().resolve() if args.session else latest_session(root)
    profile = build_profile(session)
    output = Path(args.output).expanduser().resolve() if args.output else session / "teach_profile.json"
    output.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    latest = Path(args.latest_copy).expanduser().resolve()
    latest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, latest)

    print(f"[R1-A7 TEACH PROFILE] session: {session}")
    print(f"[R1-A7 TEACH PROFILE] wrote: {output}")
    print(f"[R1-A7 TEACH PROFILE] latest copy: {latest}")
    for name, motion in profile["motions"].items():
        print(
            f"- {name}: joint={motion['selected_joint']} samples={motion['samples']} "
            f"range={motion['selected_range']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
