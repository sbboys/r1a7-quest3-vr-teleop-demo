#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="Convert manually selected fixed-camera hole pixel to R1-A7 Base point and a front/back approach point."
    )
    ap.add_argument(
        "--hole-json",
        default="/tmp/r1a7_task6_hole.json",
        help="JSON written by mark_hole_fixed.py",
    )
    ap.add_argument(
        "--intrinsics",
        default="calibration/fixed_camera_intrinsics/results/usb_zoom_camera_intrinsics.json",
    )
    ap.add_argument(
        "--extrinsic",
        default="calibration/fixed_camera_extrinsic/usb_zoom_camera_base_extrinsic_frozen_20260920.json",
    )
    ap.add_argument(
        "--workplane",
        default="calibration/r1a7_workplane_base.json",
    )
    ap.add_argument(
        "--approach-mm",
        type=float,
        default=50.0,
        help="Distance from the hole along the board normal toward the robot/base side.",
    )
    ap.add_argument(
        "--output-json",
        default="/tmp/r1a7_task6_hole_base.json",
    )
    args = ap.parse_args()

    hole = load_json(args.hole_json)
    if not hole.get("detected", False):
        raise SystemExit("ERROR: hole has not been selected yet.")

    u = float(hole["u_px"])
    v = float(hole["v_px"])

    intr = load_json(args.intrinsics)
    ext = load_json(args.extrinsic)
    wp = load_json(args.workplane)

    cal = intr["calibration"]
    K = np.asarray(cal["camera_matrix"], dtype=np.float64)
    dist = np.asarray(cal["dist_coeffs"], dtype=np.float64).reshape(-1)

    T_base_camera = np.asarray(ext["T_base_camera"], dtype=np.float64)
    T_base_board = np.asarray(wp["T_base_board"], dtype=np.float64)

    # Undistort pixel and obtain normalized camera ray [x, y, 1].
    pix = np.array([[[u, v]]], dtype=np.float64)
    und = cv2.undistortPoints(pix, K, dist)
    ray_c = np.array([und[0, 0, 0], und[0, 0, 1], 1.0], dtype=np.float64)
    ray_c /= np.linalg.norm(ray_c)

    R_base_camera = T_base_camera[:3, :3]
    O_base = T_base_camera[:3, 3]
    ray_base = R_base_camera @ ray_c
    ray_base /= np.linalg.norm(ray_base)

    # Board plane: point P0 and board +Z axis as plane normal.
    P0_base = T_base_board[:3, 3]
    n_base = T_base_board[:3, 2]
    n_base /= np.linalg.norm(n_base)

    denom = float(np.dot(n_base, ray_base))
    if abs(denom) < 1e-8:
        raise SystemExit("ERROR: camera ray is nearly parallel to the workplane.")

    lam = float(np.dot(n_base, P0_base - O_base) / denom)
    if lam <= 0:
        raise SystemExit(f"ERROR: plane intersection is behind camera (lambda={lam:.6f}).")

    P_hole = O_base + lam * ray_base

    # Choose the board-normal direction that points toward Base origin (robot side).
    to_base_origin = -P_hole
    sign = 1.0 if float(np.dot(n_base, to_base_origin)) >= 0.0 else -1.0
    approach_dir = sign * n_base

    approach_m = args.approach_mm / 1000.0
    P_approach = P_hole + approach_m * approach_dir

    result = {
        "hole_pixel": [u, v],
        "hole_base_m": P_hole.tolist(),
        "hole_base_mm": (P_hole * 1000.0).tolist(),
        "board_normal_base": n_base.tolist(),
        "approach_direction_toward_robot_base": approach_dir.tolist(),
        "approach_distance_mm": args.approach_mm,
        "approach_base_m": P_approach.tolist(),
        "approach_base_mm": (P_approach * 1000.0).tolist(),
        "ray_plane_lambda_m": lam,
    }

    print("===== TASK6 HOLE BASE =====")
    print(f"pixel (u,v)       = ({u:.2f}, {v:.2f})")
    print(
        "hole Base [m]    = "
        f"[{P_hole[0]:.6f}, {P_hole[1]:.6f}, {P_hole[2]:.6f}]"
    )
    print(
        "board normal     = "
        f"[{n_base[0]:.6f}, {n_base[1]:.6f}, {n_base[2]:.6f}]"
    )
    print(
        "approach dir     = "
        f"[{approach_dir[0]:.6f}, {approach_dir[1]:.6f}, {approach_dir[2]:.6f}]"
    )
    print(
        f"approach {args.approach_mm:.1f}mm  = "
        f"[{P_approach[0]:.6f}, {P_approach[1]:.6f}, {P_approach[2]:.6f}]"
    )
    print("NOTE: calculation only; this script sends NO robot command.")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {out}")


if __name__ == "__main__":
    main()
