#!/usr/bin/env python3
"""Summarize R1-A7 teach-in samples by motion label."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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


def summarize(session: Path) -> list[dict]:
    samples_path = session / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)
    grouped = defaultdict(list)
    grouped_active = defaultdict(list)
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            label = item.get("label", "unlabeled")
            grouped[label].append(item)
            if item.get("recording"):
                grouped_active[label].append(item)

    rows = []
    for label, items in sorted(grouped.items()):
        active_items = grouped_active.get(label)
        if active_items:
            items = active_items
        duration = float(items[-1]["t"]) - float(items[0]["t"]) if len(items) > 1 else 0.0
        row = {
            "label": label,
            "samples": len(items),
            "duration_s": duration,
            "recording_only": bool(active_items),
        }
        for name in JOINT_NAMES:
            values = [float(item["q"][name]) for item in items]
            row[f"{name}_min"] = min(values)
            row[f"{name}_max"] = max(values)
            row[f"{name}_mean"] = sum(values) / len(values)
            row[f"{name}_range"] = max(values) - min(values)
        row["selected_joint"] = items[0].get("selected_joint", "")
        row["motion"] = items[0].get("motion", "")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize R1-A7 teach-in trajectories")
    parser.add_argument("session", nargs="?", default="", help="session dir; default uses latest under --root")
    parser.add_argument("--root", default="data/r1a7_teach")
    args = parser.parse_args()

    session = Path(args.session).expanduser().resolve() if args.session else latest_session(Path(args.root).expanduser().resolve())
    rows = summarize(session)
    out = session / "label_summary.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[R1-A7 TEACH SUMMARY] session: {session}")
    print(f"[R1-A7 TEACH SUMMARY] labels: {len(rows)}")
    print(f"[R1-A7 TEACH SUMMARY] wrote: {out}")
    for row in rows:
        print(f"- {row['label']}: samples={row['samples']} duration={row['duration_s']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
