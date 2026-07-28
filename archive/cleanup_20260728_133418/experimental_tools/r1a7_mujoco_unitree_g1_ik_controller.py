#!/usr/bin/env python3
"""MuJoCo R1-A7 dual-arm test driven by Unitree's official G1_29 IK."""

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
for path in (XR_TELEOP_TV_SRC, XR_TELEOP_TELEOP, XR_TELEOP):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from televuer import TeleVuerWrapper  # noqa: E402
from robot_control.robot_arm_ik import G1_29_ArmIK  # noqa: E402


ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def joint_id(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if idx < 0:
        raise RuntimeError(f"missing joint: {name}")
    return idx


def joint_qpos_ids(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    return np.array([model.jnt_qposadr[joint_id(model, name)] for name in names], dtype=np.int32)


def joint_dof_ids(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    return np.array([model.jnt_dofadr[joint_id(model, name)] for name in names], dtype=np.int32)


def clip_to_limits(model: mujoco.MjModel, qpos: np.ndarray, qpos_ids: np.ndarray) -> np.ndarray:
    out = qpos.copy()
    for i, name in enumerate(ARM_JOINTS):
        jid = joint_id(model, name)
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            out[i] = np.clip(out[i], lo, hi)
    return out


def body_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.round(data.xpos[body_id], 3).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/robot/IsaacLab/bolt_nut_assembly/R1_A7_official/A7.urdf")
    parser.add_argument("--host-ip", default=os.getenv("HOST_IP", "192.168.1.127"))
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--joint-step", type=float, default=0.035)
    parser.add_argument("--print-period", type=float, default=0.5)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    if args.frequency <= 0:
        raise ValueError("--frequency must be positive")

    old_cwd = Path.cwd()
    os.chdir(XR_TELEOP_TELEOP)
    try:
        arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
    finally:
        os.chdir(old_cwd)

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    model.opt.timestep = min(1.0 / args.frequency, 0.01)
    arm_qpos_ids = joint_qpos_ids(model, ARM_JOINTS)
    arm_dof_ids = joint_dof_ids(model, ARM_JOINTS)

    q_home = np.zeros(14, dtype=np.float64)
    q_home[1] = 0.25
    q_home[3] = 0.30
    q_home[8] = -0.25
    q_home[10] = 0.30
    data.qpos[arm_qpos_ids] = clip_to_limits(model, q_home, arm_qpos_ids)
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

    print("[R1-A7 MuJoCo G1_29 IK] model:", args.model)
    print("[R1-A7 MuJoCo G1_29 IK] IK: Unitree G1_29_ArmIK.solve_ik")
    print("[R1-A7 MuJoCo G1_29 IK] open Quest URL:")
    print(f"https://{args.host_ip}:8012/?ws=wss://{args.host_ip}:8012")
    print("[R1-A7 MuJoCo G1_29 IK] waiting for Quest controller poses ...")

    dt = 1.0 / args.frequency
    last_log = 0.0

    def step_once() -> None:
        nonlocal last_log
        tele = tv.get_tele_data()
        if tele.motion_data_ready:
            current_q = data.qpos[arm_qpos_ids].copy()
            current_dq = data.qvel[arm_dof_ids].copy()
            sol_q, _sol_tau = arm_ik.solve_ik(
                tele.left_wrist_pose,
                tele.right_wrist_pose,
                current_q,
                current_dq,
            )
            sol_q = np.asarray(sol_q, dtype=np.float64).reshape(14)
            sol_q = clip_to_limits(model, sol_q, arm_qpos_ids)
            delta = np.clip(sol_q - current_q, -args.joint_step, args.joint_step)
            data.qpos[arm_qpos_ids] = current_q + delta

        mujoco.mj_forward(model, data)
        now = time.monotonic()
        if now - last_log >= args.print_period:
            last_log = now
            state = "ready" if tele.motion_data_ready else "waiting"
            left = body_position(model, data, "left_wrist_yaw_link")
            right = body_position(model, data, "right_wrist_yaw_link")
            q = np.round(data.qpos[arm_qpos_ids], 3).tolist()
            print(f"[R1-A7 MuJoCo G1_29 IK] {state} left={left} right={right} q={q}")

    try:
        if args.no_viewer:
            while True:
                step_once()
                time.sleep(dt)
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    step_once()
                    viewer.sync()
                    time.sleep(dt)
    finally:
        tv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
