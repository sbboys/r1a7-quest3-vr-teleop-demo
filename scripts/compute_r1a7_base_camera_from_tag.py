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

TAG_CORNER_ORDER = {
    "tag_left_top": np.array([-0.5, 0.5, 0.0], dtype=float),
    "tag_right_top": np.array([0.5, 0.5, 0.0], dtype=float),
    "tag_right_bottom": np.array([0.5, -0.5, 0.0], dtype=float),
    "tag_left_bottom": np.array([-0.5, -0.5, 0.0], dtype=float),
}


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = dst_centroid - r @ src_centroid
    residuals = (src @ r.T + t) - dst
    return r, t, residuals


def to_homogeneous(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute T_base_tag and T_base_camera for R1-A7 fixed AprilTag")
    parser.add_argument("--waypoints", required=True)
    parser.add_argument("--probe-tcp", required=True)
    parser.add_argument("--camera-extrinsic", required=True)
    parser.add_argument("--tag-size-m", type=float, default=0.092)
    parser.add_argument("--urdf", default="/home/robot/IsaacLab/bolt_nut_assembly/R1_A7_official/A7.urdf")
    parser.add_argument("--frame", default="right_wrist_yaw_link")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    waypoints = load_jsonl(Path(args.waypoints))
    latest_by_label = {item["label"]: item for item in waypoints if item.get("label") in TAG_CORNER_ORDER}
    missing = [label for label in TAG_CORNER_ORDER if label not in latest_by_label]
    if missing:
        raise RuntimeError(f"missing tag corner waypoints: {missing}")

    probe = json.loads(Path(args.probe_tcp).read_text(encoding="utf-8"))
    p_probe = np.array(probe["probe_tcp_in_frame_m"], dtype=float)

    model = pin.buildModelFromUrdf(args.urdf)
    data = model.createData()
    frame_id = model.getFrameId(args.frame)
    q_model = pin.neutral(model)
    joint_q_indices = [model.joints[model.getJointId(name)].idx_q for name in RIGHT_ARM_JOINTS]

    tag_points = []
    base_points = []
    used_waypoints = {}
    for label, unit_pt in TAG_CORNER_ORDER.items():
        item = latest_by_label[label]
        for idx, key in zip(joint_q_indices, RIGHT_ARM_KEYS):
            q_model[idx] = float(item["q"][key])
        pin.forwardKinematics(model, data, q_model)
        pin.updateFramePlacements(model, data)
        pose = data.oMf[frame_id]
        p_base = pose.rotation @ p_probe + pose.translation
        tag_points.append(unit_pt * args.tag_size_m)
        base_points.append(p_base)
        used_waypoints[label] = {
            "t": item.get("t"),
            "waypoint_q": {key: float(item["q"][key]) for key in RIGHT_ARM_KEYS},
            "base_point_m": p_base.tolist(),
            "tag_point_m": (unit_pt * args.tag_size_m).tolist(),
        }

    tag_points_np = np.vstack(tag_points)
    base_points_np = np.vstack(base_points)
    r_base_tag, t_base_tag, residuals = rigid_transform(tag_points_np, base_points_np)
    residual_norms = np.linalg.norm(residuals, axis=1)
    t_base_tag = to_homogeneous(r_base_tag, t_base_tag)

    camera = json.loads(Path(args.camera_extrinsic).read_text(encoding="utf-8"))
    t_camera_to_tag = np.array(camera["T_camera_to_tag"], dtype=float)
    t_tag_to_camera = np.array(camera["T_tag_to_camera"], dtype=float)
    # Mapping convention: P_base = T_base_tag * T_camera_to_tag * P_camera.
    t_base_camera = t_base_tag @ t_camera_to_tag
    t_camera_base = np.linalg.inv(t_base_camera)

    result = {
        "waypoints": str(Path(args.waypoints).resolve()),
        "probe_tcp": str(Path(args.probe_tcp).resolve()),
        "camera_extrinsic": str(Path(args.camera_extrinsic).resolve()),
        "tag_size_m": args.tag_size_m,
        "frame": args.frame,
        "T_base_tag": t_base_tag.tolist(),
        "T_tag_base": np.linalg.inv(t_base_tag).tolist(),
        "T_tag_to_camera": t_tag_to_camera.tolist(),
        "T_camera_to_tag": t_camera_to_tag.tolist(),
        "T_base_camera": t_base_camera.tolist(),
        "T_camera_base": t_camera_base.tolist(),
        "corner_residuals_m": {
            label: residual.tolist()
            for label, residual in zip(TAG_CORNER_ORDER.keys(), residuals)
        },
        "corner_residual_norms_m": {
            label: float(norm)
            for label, norm in zip(TAG_CORNER_ORDER.keys(), residual_norms)
        },
        "corner_residual_mean_m": float(residual_norms.mean()),
        "corner_residual_max_m": float(residual_norms.max()),
        "used_waypoints": used_waypoints,
    }

    output = Path(args.output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Base/camera calibration result")
    print(f"  tag_size_m: {args.tag_size_m}")
    print(f"  residual_mean_m: {residual_norms.mean():.6f}")
    print(f"  residual_max_m:  {residual_norms.max():.6f}")
    print("  T_base_tag:")
    print(np.array2string(t_base_tag, precision=6, suppress_small=True))
    print("  T_base_camera:")
    print(np.array2string(t_base_camera, precision=6, suppress_small=True))
    print(f"  saved: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
