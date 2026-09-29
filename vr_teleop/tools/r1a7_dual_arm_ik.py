#!/usr/bin/env python3
"""Pinocchio damped-least-squares IK for the official Unitree R1-A7 URDF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pinocchio as pin


DEFAULT_URDF = "/home/robot/IsaacLab/bolt_nut_assembly/R1_A7_official/A7.urdf"
LEFT_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@dataclass
class ArmTarget:
    frame: str
    joint_names: list[str]
    offset: np.ndarray
    target: pin.SE3


class R1A7DualArmIK:
    def __init__(
        self,
        urdf: str = DEFAULT_URDF,
        left_frame: str = "left_wrist_yaw_link",
        right_frame: str = "right_wrist_yaw_link",
        tcp_x: float = 0.0,
    ):
        self.model = pin.buildModelFromUrdf(urdf)
        self.data = self.model.createData()
        self.left_ids = [self.model.getJointId(name) for name in LEFT_JOINTS]
        self.right_ids = [self.model.getJointId(name) for name in RIGHT_JOINTS]
        self.left_v = [self.model.joints[jid].idx_v for jid in self.left_ids]
        self.right_v = [self.model.joints[jid].idx_v for jid in self.right_ids]
        self.active_v = self.left_v + self.right_v
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        offset = np.array([tcp_x, 0.0, 0.0], dtype=float)
        q0 = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q0)
        pin.updateFramePlacements(self.model, self.data)
        self.left = ArmTarget(
            left_frame,
            LEFT_JOINTS,
            offset,
            self._frame_pose(q0, left_frame, offset),
        )
        self.right = ArmTarget(
            right_frame,
            RIGHT_JOINTS,
            offset,
            self._frame_pose(q0, right_frame, offset),
        )

    def _frame_pose(self, q: np.ndarray, frame: str, offset: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        fid = self.model.getFrameId(frame)
        return self.data.oMf[fid] * pin.SE3(np.eye(3), offset)

    def _frame_jacobian(self, q: np.ndarray, frame: str, offset: np.ndarray) -> np.ndarray:
        fid = self.model.getFrameId(frame)
        jac = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        if np.linalg.norm(offset) <= 1e-12:
            return jac
        # Linear velocity at TCP = linear velocity at frame + angular x offset.
        adj = jac.copy()
        adj[:3, :] += -pin.skew(offset) @ jac[3:, :]
        return adj

    @staticmethod
    def _pose_error(current: pin.SE3, target: pin.SE3) -> np.ndarray:
        pos_err = target.translation - current.translation
        rot_err = pin.log3(current.rotation.T @ target.rotation)
        return np.concatenate([pos_err, rot_err])

    def solve(
        self,
        left_target: pin.SE3,
        right_target: pin.SE3,
        q0: np.ndarray | None = None,
        max_iters: int = 80,
        damping: float = 0.04,
        step_scale: float = 0.55,
        tol: float = 1e-3,
    ) -> tuple[np.ndarray, float, int]:
        q = pin.neutral(self.model) if q0 is None else q0.copy()
        q = np.clip(q, self.lower, self.upper)
        active = np.array(self.active_v, dtype=int)
        for it in range(max_iters):
            left_pose = self._frame_pose(q, self.left.frame, self.left.offset)
            right_pose = self._frame_pose(q, self.right.frame, self.right.offset)
            err = np.concatenate(
                [
                    self._pose_error(left_pose, left_target),
                    self._pose_error(right_pose, right_target),
                ]
            )
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                return q, err_norm, it
            j_full = np.vstack(
                [
                    self._frame_jacobian(q, self.left.frame, self.left.offset),
                    self._frame_jacobian(q, self.right.frame, self.right.offset),
                ]
            )
            j = j_full[:, active]
            lhs = j @ j.T + (damping**2) * np.eye(j.shape[0])
            dq_active = j.T @ np.linalg.solve(lhs, err)
            dq = np.zeros(self.model.nv)
            dq[active] = np.clip(step_scale * dq_active, -0.08, 0.08)
            q = pin.integrate(self.model, q, dq)
            q = np.clip(q, self.lower, self.upper)
        return q, err_norm, max_iters

    def summary(self, q: np.ndarray) -> str:
        values = []
        for name in LEFT_JOINTS + RIGHT_JOINTS:
            jid = self.model.getJointId(name)
            idx = self.model.joints[jid].idx_q
            values.append(f"{name}={q[idx]:+.4f}")
        return "\n".join(values)


def _parse_delta(text: str) -> np.ndarray:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return np.array(values, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description="R1-A7 dual-arm Pinocchio IK self-test")
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--left_frame", default="left_wrist_yaw_link")
    parser.add_argument("--right_frame", default="right_wrist_yaw_link")
    parser.add_argument("--tcp_x", type=float, default=0.0)
    parser.add_argument("--left_delta", type=_parse_delta, default=np.array([0.03, 0.02, 0.02]))
    parser.add_argument("--right_delta", type=_parse_delta, default=np.array([0.03, -0.02, 0.02]))
    parser.add_argument("--max_iters", type=int, default=100)
    args = parser.parse_args()

    ik = R1A7DualArmIK(args.urdf, args.left_frame, args.right_frame, args.tcp_x)
    left_target = ik.left.target.copy()
    right_target = ik.right.target.copy()
    left_target.translation += args.left_delta
    right_target.translation += args.right_delta
    q, err, iters = ik.solve(left_target, right_target, max_iters=args.max_iters)
    print("[R1-A7 IK] model nq/nv:", ik.model.nq, ik.model.nv)
    print("[R1-A7 IK] left frame:", args.left_frame, "delta:", args.left_delta.tolist())
    print("[R1-A7 IK] right frame:", args.right_frame, "delta:", args.right_delta.tolist())
    print(f"[R1-A7 IK] result err={err:.6f} iters={iters}")
    print(ik.summary(q))
    return 0 if err < 0.02 else 2


if __name__ == "__main__":
    raise SystemExit(main())
