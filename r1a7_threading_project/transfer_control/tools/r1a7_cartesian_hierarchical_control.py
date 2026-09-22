#!/usr/bin/env python3
"""Run the existing Cartesian controller with an experimental IK class.

The validated controller and official XR IK source are not edited. This entry
point patches only the current Python process before delegating to the existing
controller's main program.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_XR_ROOT = Path(
    "/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/xr_teleoperate"
)
BASE_CONTROLLER = (
    PROJECT
    / "transfer_control"
    / "experimental"
    / "r1a7_cartesian_stream_control_experimental.py"
)


def argument_value(name: str, default: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def main() -> None:
    xr_root = Path(
        argument_value(
            "--xr-root",
            os.environ.get("XR_TELEOP_ROOT", str(DEFAULT_XR_ROOT)),
        )
    ).expanduser().resolve()
    if not (xr_root / "teleop" / "robot_control" / "robot_arm_ik.py").is_file():
        raise RuntimeError(f"official XR IK checkout not found: {xr_root}")

    sys.path.insert(0, str(PROJECT.parent))
    sys.path.insert(0, str(xr_root))
    previous_cwd = Path.cwd()
    try:
        os.chdir(xr_root / "teleop")
        import teleop.robot_control.robot_arm_ik as official_ik
        from r1a7_threading_project.transfer_control.experimental.r1a7_hierarchical_ik import (
            R1A7HierarchicalArmIK,
        )

        official_ik.R1A7_ArmIK = R1A7HierarchicalArmIK
        print(
            "[EXPERIMENTAL HIERARCHICAL IK] "
            "XYZ primary, orientation secondary, elbow/wrist posture tertiary.",
            flush=True,
        )
        runpy.run_path(str(BASE_CONTROLLER), run_name="__main__")
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
