#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight MuJoCo VR-controller test for R1-A7 dual arms.

This is a simulation-only bring-up tool.  It uses Quest/WebXR controller poses
through Unitree's TeleVuerWrapper and updates the MuJoCo A7 arm joints with a
small damped-least-squares IK step.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


XR_TELEOP = Path(os.getenv("XR_TELEOP_ROOT", "/home/robot/xr_teleoperate"))
XR_TELEOP_TELEOP = XR_TELEOP / "teleop"
XR_TELEOP_TV_SRC = XR_TELEOP_TELEOP / "televuer" / "src"
for path in (XR_TELEOP_TV_SRC, XR_TELEOP_TELEOP):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from televuer import TeleVuerWrapper  # noqa: E402


LEFT_ARM = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def jid(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if idx < 0:
        raise RuntimeError(f"missing joint {name}")
    return idx


def bid(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if idx < 0:
        raise RuntimeError(f"missing body {name}")
    return idx


def joint_qpos_ids(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    return np.array([model.jnt_qposadr[jid(model, name)] for name in names], dtype=np.int32)


def joint_dof_ids(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    return np.array([model.jnt_dofadr[jid(model, name)] for name in names], dtype=np.int32)


def limit_delta(delta: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    if norm > max_norm and norm > 1e-9:
        return delta * (max_norm / norm)
    return delta


def solve_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    dof_ids: np.ndarray,
    target: np.ndarray,
    damping: float,
    cartesian_step: float,
) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    j = jacp[:, dof_ids]
    err = limit_delta(target - data.xpos[body_id], cartesian_step)
    lhs = j @ j.T + (damping * damping) * np.eye(3)
    dq = j.T @ np.linalg.solve(lhs, err)
    return np.nan_to_num(dq)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/robot/IsaacLab/bolt_nut_assembly/R1_A7_official/A7.urdf")
    parser.add_argument("--host-ip", default=os.getenv("HOST_IP", "192.168.1.127"))
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--max-delta", type=float, default=0.18)
    parser.add_argument("--cartesian-step", type=float, default=0.012)
    parser.add_argument("--joint-step", type=float, default=0.025)
    parser.add_argument("--damping", type=float, default=0.06)
    parser.add_argument("--print-period", type=float, default=0.5)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.005

    left_q = joint_qpos_ids(model, LEFT_ARM)
    right_q = joint_qpos_ids(model, RIGHT_ARM)
    left_dof = joint_dof_ids(model, LEFT_ARM)
    right_dof = joint_dof_ids(model, RIGHT_ARM)
    left_body = bid(model, "left_wrist_yaw_link")
    right_body = bid(model, "right_wrist_yaw_link")

    # Match the relaxed IsaacLab test posture enough for visibility.
    init = {
        "left_shoulder_roll_joint": 0.25,
        "left_elbow_joint": 0.30,
        "right_shoulder_roll_joint": -0.25,
        "right_elbow_joint": 0.30,
    }
    for name, value in init.items():
        data.qpos[model.jnt_qposadr[jid(model, name)]] = value
    mujoco.mj_forward(model, data)

    tv = TeleVuerWrapper(
        use_hand_tracking=False,
        binocular=False,
        img_shape=(480, 640),
        display_mode="pass-through",
        zmq=False,
        webrtc=False,
        arm_reference_mode="head_yaw",
    )

    left_home = data.xpos[left_body].copy()
    right_home = data.xpos[right_body].copy()
    left_zero = None
    right_zero = None
    last_log = 0.0
    print("[R1-A7 MuJoCo VR] model:", args.model)
    print("[R1-A7 MuJoCo VR] open Quest URL:")
    print(f"https://{args.host_ip}:8012/?ws=wss://{args.host_ip}:8012")
    print("[R1-A7 MuJoCo VR] waiting for controller poses ...")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                tele = tv.get_tele_data()
                if tele.motion_data_ready:
                    left_vr = np.asarray(tele.left_wrist_pose[:3, 3], dtype=np.float64)
                    right_vr = np.asarray(tele.right_wrist_pose[:3, 3], dtype=np.float64)
                    if left_zero is None:
                        left_zero = left_vr.copy()
                        right_zero = right_vr.copy()
                        print("[R1-A7 MuJoCo VR] calibrated controller zero")

                    left_delta = np.clip((left_vr - left_zero) * args.scale, -args.max_delta, args.max_delta)
                    right_delta = np.clip((right_vr - right_zero) * args.scale, -args.max_delta, args.max_delta)
                    left_target = left_home + left_delta
                    right_target = right_home + right_delta

                    dq_l = solve_step(model, data, left_body, left_dof, left_target, args.damping, args.cartesian_step)
                    dq_r = solve_step(model, data, right_body, right_dof, right_target, args.damping, args.cartesian_step)
                    data.qpos[left_q] += np.clip(dq_l, -args.joint_step, args.joint_step)
                    data.qpos[right_q] += np.clip(dq_r, -args.joint_step, args.joint_step)

                    # Respect joint limits.
                    for qadr, name in zip(np.r_[left_q, right_q], LEFT_ARM + RIGHT_ARM):
                        j = jid(model, name)
                        if model.jnt_limited[j]:
                            lo, hi = model.jnt_range[j]
                            data.qpos[qadr] = np.clip(data.qpos[qadr], lo, hi)

                mujoco.mj_forward(model, data)
                viewer.sync()

                now = time.monotonic()
                if now - last_log > args.print_period:
                    last_log = now
                    state = "ready" if tele.motion_data_ready else "waiting"
                    lp = np.round(data.xpos[left_body], 3).tolist()
                    rp = np.round(data.xpos[right_body], 3).tolist()
                    print(f"[R1-A7 MuJoCo VR] {state} left_wrist={lp} right_wrist={rp}")
                time.sleep(model.opt.timestep)
    finally:
        tv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
