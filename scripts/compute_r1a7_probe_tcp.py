#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin


RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

RIGHT_ARM_KEYS = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


def load_waypoints(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if len(items) < 4:
        raise RuntimeError(f"need at least 4 waypoints, got {len(items)}")
    return items


def frame_pose(model: pin.Model, data: pin.Data, q: np.ndarray, frame_id: int) -> pin.SE3:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id]


def solve_probe_tcp(poses: list[pin.SE3]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Unknowns are probe point p in wrist frame and the fixed contact point c in base frame:
    #   R_i p + t_i = c
    # -> [R_i, -I] [p, c]^T = -t_i
    rows = []
    rhs = []
    for pose in poses:
        rows.append(np.hstack([pose.rotation, -np.eye(3)]))
        rhs.append(-pose.translation)
    a = np.vstack(rows)
    b = np.concatenate(rhs)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    p_wrist = x[:3]
    c_base = x[3:]
    residuals = np.array([pose.rotation @ p_wrist + pose.translation - c_base for pose in poses])
    return p_wrist, c_base, residuals


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute R1-A7 right probe TCP from fixed-point waypoints")
    parser.add_argument("--waypoints", required=True)
    parser.add_argument("--urdf", default="/home/robot/IsaacLab/bolt_nut_assembly/R1_A7_official/A7.urdf")
    parser.add_argument("--frame", default="right_wrist_yaw_link")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    waypoints_path = Path(args.waypoints)
    waypoints = load_waypoints(waypoints_path)
    model = pin.buildModelFromUrdf(args.urdf)
    data = model.createData()
    frame_id = model.getFrameId(args.frame)
    if frame_id >= len(model.frames):
        raise RuntimeError(f"frame not found: {args.frame}")

    q_model = pin.neutral(model)
    joint_q_indices = [model.joints[model.getJointId(name)].idx_q for name in RIGHT_ARM_JOINTS]

    poses = []
    used = []
    for item in waypoints:
        q = item["q"]
        for idx, key in zip(joint_q_indices, RIGHT_ARM_KEYS):
            q_model[idx] = float(q[key])
        pose = frame_pose(model, data, q_model.copy(), frame_id)
        poses.append(pin.SE3(pose.rotation.copy(), pose.translation.copy()))
        used.append({key: float(q[key]) for key in RIGHT_ARM_KEYS})

    p_wrist, c_base, residuals = solve_probe_tcp(poses)
    residual_norm = np.linalg.norm(residuals, axis=1)

    result = {
        "source_waypoints": str(waypoints_path.resolve()),
        "urdf": args.urdf,
        "frame": args.frame,
        "sample_count": len(waypoints),
        "probe_tcp_in_frame_m": p_wrist.tolist(),
        "fixed_contact_point_in_model_base_m": c_base.tolist(),
        "residuals_m": residuals.tolist(),
        "residual_norms_m": residual_norm.tolist(),
        "residual_mean_m": float(residual_norm.mean()),
        "residual_max_m": float(residual_norm.max()),
        "right_arm_joint_samples": used,
    }

    output = Path(args.output) if args.output else waypoints_path.with_name("probe_tcp_result.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Probe TCP result")
    print(f"  samples: {len(waypoints)}")
    print(f"  frame: {args.frame}")
    print("  probe_tcp_in_frame_m:", np.array2string(p_wrist, precision=6, suppress_small=False))
    print("  fixed_contact_point_in_model_base_m:", np.array2string(c_base, precision=6, suppress_small=False))
    print(f"  residual_mean_m: {residual_norm.mean():.6f}")
    print(f"  residual_max_m:  {residual_norm.max():.6f}")
    print("  residual_norms_m:", np.array2string(residual_norm, precision=6, suppress_small=False))
    print(f"  saved: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
