#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick Gemini 336L pose-source smoke test.

Run this before Isaac Sim to verify that Orbbec SDK, MediaPipe and the camera
produce wrist-relative 3D targets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from action_provider.gemini_pose_source import GeminiPoseSource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["left", "right"], default="right")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--min_visibility", type=float, default=0.35)
    parser.add_argument("--min_wrist_shoulder_m", type=float, default=0.035)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    source = GeminiPoseSource(
        hand=args.hand,
        show=args.show,
        debug=args.debug,
        min_visibility=args.min_visibility,
        min_wrist_shoulder_m=args.min_wrist_shoulder_m,
    )
    source.start()
    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            target = source.get_latest(max_age_s=1.0)
            if target is not None and target.valid:
                rel = target.wrist_rel_m
                print(
                    f"wrist_rel_m=({rel[0]:+.3f}, {rel[1]:+.3f}, {rel[2]:+.3f}) "
                    f"grip={target.grip:.2f}"
                )
            else:
                if args.debug:
                    reason, counts = source.get_debug_snapshot()
                    print(f"no valid target reason={reason} counts={counts}")
                else:
                    print("no valid target")
            time.sleep(0.5)
    finally:
        source.stop()


if __name__ == "__main__":
    main()
