#!/usr/bin/env python3
"""Preview Vive/OpenVR controller motion through R1-A7 dual-arm IK.

This script is read-only: it does not publish DDS commands to the robot.
It calibrates the current controller positions as zero, maps relative
controller translation into the A7 base frame, solves dual-arm IK, and prints
the resulting left/right 7-DoF joint targets.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.r1a7_dual_arm_ik import LEFT_JOINTS, RIGHT_JOINTS, R1A7DualArmIK


STEAMVR_ROOT = Path.home() / ".local/share/Steam/steamapps/common/SteamVR"


def _controller_position(vr, openvr, role_name: str) -> Optional[np.ndarray]:
    poses = vr.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding,
        0.0,
        openvr.k_unMaxTrackedDeviceCount,
    )
    target_role = {
        "left": openvr.TrackedControllerRole_LeftHand,
        "right": openvr.TrackedControllerRole_RightHand,
    }[role_name]
    for idx, pose in enumerate(poses):
        if vr.getTrackedDeviceClass(idx) != openvr.TrackedDeviceClass_Controller:
            continue
        if vr.getControllerRoleForTrackedDeviceIndex(idx) != target_role:
            continue
        if not vr.isTrackedDeviceConnected(idx) or not pose.bPoseIsValid:
            return None
        mat = pose.mDeviceToAbsoluteTracking
        return np.array([mat[0][3], mat[1][3], mat[2][3]], dtype=float)
    return None


def _vr_delta_to_robot(delta_vr: np.ndarray, scale: float) -> np.ndarray:
    # OpenVR standing frame: +X right, +Y up, -Z forward.
    # A7 base frame: +X forward, +Y left, +Z up.
    return scale * np.array([-delta_vr[2], -delta_vr[0], delta_vr[1]], dtype=float)


def _apply_axis_options(delta: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    out = delta.copy()
    out[0] *= float(getattr(args, "x_sign", 1.0))
    out[1] *= float(getattr(args, "y_sign", 1.0))
    out[2] *= float(getattr(args, "z_sign", 1.0))
    mode = getattr(args, "axis_mode", "full")
    if mode == "vertical":
        out[0] = 0.0
        out[1] = 0.0
    elif mode == "lateral":
        out[0] = 0.0
        out[2] = 0.0
    elif mode == "depth":
        out[1] = 0.0
        out[2] = 0.0
    return out


def _active_joint_values(ik: R1A7DualArmIK, q: np.ndarray, names: list[str]) -> list[float]:
    values = []
    for name in names:
        jid = ik.model.getJointId(name)
        values.append(float(q[ik.model.joints[jid].idx_q]))
    return values


def _format_arm(prefix: str, names: list[str], values: list[float]) -> str:
    parts = []
    for name, value in zip(names, values):
        short = name.replace("_joint", "").replace(prefix + "_", "")
        parts.append(f"{short}={value:+.3f}")
    return " ".join(parts)


def _limit_warnings(ik: R1A7DualArmIK, names: list[str], q: np.ndarray) -> list[str]:
    warnings = []
    margin = 0.03
    for name in names:
        jid = ik.model.getJointId(name)
        idx = ik.model.joints[jid].idx_q
        value = float(q[idx])
        lower = float(ik.lower[idx])
        upper = float(ik.upper[idx])
        if value <= lower + margin:
            warnings.append(f"{name} near lower {value:+.3f}/{lower:+.3f}")
        elif value >= upper - margin:
            warnings.append(f"{name} near upper {value:+.3f}/{upper:+.3f}")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="R1-A7 VR controller IK preview")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means forever")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--print_period", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=0.45, help="robot meters per controller meter")
    parser.add_argument("--max_delta_m", type=float, default=0.18)
    parser.add_argument(
        "--axis_mode",
        choices=["full", "vertical", "lateral", "depth"],
        default="full",
        help="limit mapped robot target motion for calibration",
    )
    parser.add_argument("--x_sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--y_sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--z_sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--tcp_x", type=float, default=0.0)
    parser.add_argument("--ik_max_iters", type=int, default=40)
    parser.add_argument(
        "--swap_hands",
        action="store_true",
        help="map the OpenVR left controller to the A7 right arm and vice versa",
    )
    parser.add_argument(
        "--right_only",
        action="store_true",
        help="ignore left-controller motion and keep the left A7 wrist at its calibrated target",
    )
    parser.add_argument(
        "--left_only",
        action="store_true",
        help="ignore right-controller motion and keep the right A7 wrist at its calibrated target",
    )
    args = parser.parse_args()
    if args.left_only and args.right_only:
        print("[R1-A7 VR IK] choose only one of --left_only / --right_only")
        return 2

    os.environ.setdefault("VR_OVERRIDE", str(STEAMVR_ROOT))
    try:
        import openvr  # type: ignore
    except Exception as exc:
        print("[R1-A7 VR IK] missing openvr module:", exc)
        return 4

    vr = None
    try:
        vr = openvr.init(openvr.VRApplication_Other)
        ik = R1A7DualArmIK(tcp_x=args.tcp_x)
        q = pin.neutral(ik.model)
        left_zero = None
        right_zero = None
        deadline = time.monotonic() + max(0.0, args.duration)
        next_print = 0.0
        print("[R1-A7 VR IK] waiting for valid left/right controller poses ...")
        while True:
            now = time.monotonic()
            if args.duration > 0 and now >= deadline:
                break
            left_pos = _controller_position(vr, openvr, "left")
            right_pos = _controller_position(vr, openvr, "right")
            if left_pos is None or right_pos is None:
                if now >= next_print:
                    next_print = now + max(0.25, args.print_period)
                    print("[R1-A7 VR IK] controller pose lost; hold still in base-station view")
                time.sleep(1.0 / max(1.0, args.hz))
                continue
            if args.swap_hands:
                left_pos, right_pos = right_pos, left_pos
            if left_zero is None:
                left_zero = left_pos.copy()
                right_zero = right_pos.copy()
                print("[R1-A7 VR IK] calibrated left zero:", left_zero.tolist())
                print("[R1-A7 VR IK] calibrated right zero:", right_zero.tolist())

            left_delta = _apply_axis_options(_vr_delta_to_robot(left_pos - left_zero, args.scale), args)
            right_delta = _apply_axis_options(_vr_delta_to_robot(right_pos - right_zero, args.scale), args)
            if args.right_only:
                left_delta[:] = 0.0
            if args.left_only:
                right_delta[:] = 0.0
            left_delta = np.clip(left_delta, -args.max_delta_m, args.max_delta_m)
            right_delta = np.clip(right_delta, -args.max_delta_m, args.max_delta_m)

            left_target = ik.left.target.copy()
            right_target = ik.right.target.copy()
            left_target.translation += left_delta
            right_target.translation += right_delta
            q, err, iters = ik.solve(
                left_target,
                right_target,
                q0=q,
                max_iters=args.ik_max_iters,
            )

            if now >= next_print:
                next_print = now + max(0.05, args.print_period)
                left_q = _active_joint_values(ik, q, LEFT_JOINTS)
                right_q = _active_joint_values(ik, q, RIGHT_JOINTS)
                print(
                    "[R1-A7 VR IK]"
                    f" err={err:.4f} iters={iters}"
                    f" left_delta=({left_delta[0]:+.3f},{left_delta[1]:+.3f},{left_delta[2]:+.3f})"
                    f" right_delta=({right_delta[0]:+.3f},{right_delta[1]:+.3f},{right_delta[2]:+.3f})"
                )
                print("  left :", _format_arm("left", LEFT_JOINTS, left_q))
                print("  right:", _format_arm("right", RIGHT_JOINTS, right_q))
                warnings = _limit_warnings(ik, LEFT_JOINTS + RIGHT_JOINTS, q)
                if warnings:
                    print("  limit:", "; ".join(warnings))

            time.sleep(1.0 / max(1.0, args.hz))
        return 0
    finally:
        if vr is not None:
            try:
                openvr.shutdown()  # type: ignore[name-defined]
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
