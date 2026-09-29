# -*- coding: utf-8 -*-
"""R1-A7 dual-arm VR IK validation provider for IsaacLab simulation.

This provider follows Unitree xr_teleoperate's data path:
Quest/WebXR -> TeleVuerWrapper -> left/right wrist poses.  Unlike the official
G1_29 runtime controller, it only writes IsaacLab simulation actions.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from action_provider.action_base import ActionProvider
from isaaclab.utils.math import quat_apply, subtract_frame_transforms
from tasks.common_observations.camera_state import get_camera_image
from tools.shared_memory_utils import MultiImageReader


XR_TELEOP = Path(os.getenv("XR_TELEOP_ROOT", "/home/robot/xr_teleoperate"))
XR_TELEOP_TELEOP = XR_TELEOP / "teleop"
if XR_TELEOP_TELEOP.is_dir() and str(XR_TELEOP_TELEOP) not in sys.path:
    sys.path.insert(0, str(XR_TELEOP_TELEOP))

from televuer import TeleVuerWrapper  # noqa: E402


class R1A7VRDualArmIKProvider(ActionProvider):
    LEFT_ARM_JOINT_NAMES = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]
    RIGHT_ARM_JOINT_NAMES = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    LEFT_BODY_CANDIDATES = (
        ("left_dex1_base_link", (0.0, 0.115, 0.0)),
        ("left_wrist_yaw_link", (0.0, 0.0, 0.0)),
    )
    RIGHT_BODY_CANDIDATES = (
        ("right_dex1_base_link", (0.0, -0.115, 0.0)),
        ("right_wrist_yaw_link", (0.0, 0.0, 0.0)),
    )

    def __init__(self, env, args_cli):
        super().__init__("R1A7VRDualArmIK")
        self.env = env
        self.args_cli = args_cli
        self.robot = env.scene["robot"]
        self.device = env.device
        self.num_envs = env.num_envs
        if self.num_envs != 1:
            raise RuntimeError(f"vr_ik requires num_envs=1, got {self.num_envs}")

        self.joint_names = list(self.robot.joint_names)
        self.body_names = list(self.robot.body_names)
        self.head_body_id, self.head_body_name = self._select_head_body()
        self.left_arm_ids = self._joint_ids(self.LEFT_ARM_JOINT_NAMES)
        self.right_arm_ids = self._joint_ids(self.RIGHT_ARM_JOINT_NAMES)
        self.arm_ids = self.left_arm_ids + self.right_arm_ids

        self.left_body_id, self.left_body_name, self.left_offset_local = self._select_body(self.LEFT_BODY_CANDIDATES)
        self.right_body_id, self.right_body_name, self.right_offset_local = self._select_body(self.RIGHT_BODY_CANDIDATES)
        self.left_jacobian_id = self.left_body_id - 1 if self.robot.is_fixed_base else self.left_body_id
        self.right_jacobian_id = self.right_body_id - 1 if self.robot.is_fixed_base else self.right_body_id

        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.action_joint_ids = self._resolve_action_joint_ids()
        self.control_dt = 1.0 / max(1.0, float(getattr(args_cli, "step_hz", 50)))

        self.scale = float(getattr(args_cli, "vr_ik_scale", 0.18))
        self.max_delta = float(getattr(args_cli, "vr_ik_max_delta_m", 0.12))
        self.cartesian_step = float(getattr(args_cli, "vr_ik_cartesian_step", 0.010))
        self.joint_step = float(getattr(args_cli, "vr_ik_joint_step", 0.018))
        self.joint_speed = float(getattr(args_cli, "vr_ik_joint_speed", 0.55))
        self.command_lead = float(getattr(args_cli, "vr_ik_command_lead", 0.08))
        self.damping = float(getattr(args_cli, "vr_ik_damping", 0.08))
        self.filter_alpha = float(getattr(args_cli, "vr_ik_filter_alpha", 0.55))
        self.preview_only = bool(getattr(args_cli, "vr_ik_preview_only", False))
        self.use_orientation = bool(getattr(args_cli, "vr_ik_orientation", False))
        self.orientation_gain = float(getattr(args_cli, "vr_ik_orientation_gain", 0.15))

        self.command_q = self.robot.data.joint_pos[:, self.arm_ids].clone()
        self.xr_left_zero = None
        self.xr_right_zero = None
        self.left_home_pos = None
        self.right_home_pos = None
        self.left_home_quat = None
        self.right_home_quat = None
        self.left_target_pos = None
        self.right_target_pos = None
        self.last_motion_ready = False
        self.last_log = 0.0
        self.last_image_log = 0.0
        self.frame_step = 0
        self.xr_image_rotate = int(os.getenv("A7_XR_IMAGE_ROTATE", "0"))
        self.xr_image_flip = os.getenv("A7_XR_IMAGE_FLIP", "none").lower()

        input_mode = getattr(args_cli, "vr_ik_input_mode", "hand")
        display_mode = getattr(args_cli, "vr_ik_display_mode", "pass-through")
        self.display_mode = display_mode
        self.image_reader = None
        self.xr_need_local_img = display_mode != "pass-through"
        img_shape = (480, 640)
        if self.xr_need_local_img:
            self.image_reader = MultiImageReader()
        self.tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=input_mode == "hand",
            binocular=False,
            img_shape=img_shape,
            display_mode=display_mode,
            zmq=self.xr_need_local_img,
            webrtc=False,
            arm_reference_mode="head_yaw",
        )

        print("\n" + "=" * 96)
        print("[R1-A7 VR IK] provider constructed")
        print("[R1-A7 VR IK] mode:", f"input={input_mode}", f"display={display_mode}", f"preview_only={self.preview_only}")
        print("[R1-A7 VR IK] XR image:", f"enabled={self.xr_need_local_img}", f"shape={img_shape}")
        print("[R1-A7 VR IK] XR image transform:", f"rotate={self.xr_image_rotate}", f"flip={self.xr_image_flip}")
        print("[R1-A7 VR IK] open Quest URL: https://<host-ip>:8012/?ws=wss://<host-ip>:8012")
        print("[R1-A7 VR IK] head body:", self.head_body_name, "body_id=", self.head_body_id)
        print("[R1-A7 VR IK] head-like bodies:", [name for name in self.body_names if "head" in name.lower()])
        print("[R1-A7 VR IK] left body:", self.left_body_name, "jacobian_id=", self.left_jacobian_id, "arm_ids=", self.left_arm_ids)
        print("[R1-A7 VR IK] right body:", self.right_body_name, "jacobian_id=", self.right_jacobian_id, "arm_ids=", self.right_arm_ids)
        print("[R1-A7 VR IK] scale/max_delta:", self.scale, self.max_delta)
        print("=" * 96 + "\n")

    def _joint_ids(self, names: list[str]) -> list[int]:
        missing = [name for name in names if name not in self.joint_names]
        if missing:
            raise RuntimeError(f"Missing joints {missing}; available={self.joint_names}")
        return [self.joint_names.index(name) for name in names]

    def _select_body(self, candidates):
        for name, offset in candidates:
            if name in self.body_names:
                return (
                    self.body_names.index(name),
                    name,
                    torch.tensor(offset, dtype=torch.float32, device=self.device).view(1, 3),
                )
        raise RuntimeError(f"No candidate body found from {candidates}; available={self.body_names}")

    def _select_head_body(self) -> tuple[Optional[int], Optional[str]]:
        preferred = ("head_yaw_link", "head_pitch_link", "head_link")
        for name in preferred:
            if name in self.body_names:
                return self.body_names.index(name), name
        for idx, name in enumerate(self.body_names):
            if "head" in name.lower():
                return idx, name
        return None, None

    @staticmethod
    def _ids_to_list(ids, count: int):
        if ids is None:
            return None
        if isinstance(ids, slice):
            return list(range(count))[ids]
        if isinstance(ids, torch.Tensor):
            return [int(v) for v in ids.detach().cpu().flatten().tolist()]
        if isinstance(ids, range):
            return list(ids)
        if isinstance(ids, (list, tuple)):
            return [int(v) for v in ids]
        return None

    def _resolve_action_joint_ids(self):
        manager = getattr(self.env, "action_manager", None)
        term = None
        if manager is not None:
            terms = getattr(manager, "_terms", None)
            if isinstance(terms, dict):
                term = terms.get("joint_pos") or (next(iter(terms.values())) if len(terms) == 1 else None)
            elif isinstance(terms, (list, tuple)) and len(terms) == 1:
                term = terms[0]
        if term is not None:
            for attr in ("_joint_ids", "joint_ids"):
                ids = self._ids_to_list(getattr(term, attr, None), len(self.joint_names))
                if ids is not None:
                    return ids
        return list(range(len(self.joint_names)))

    def _root_pose_w(self):
        data = self.robot.data
        if hasattr(data, "root_link_pose_w"):
            return data.root_link_pose_w
        if hasattr(data, "root_pose_w"):
            return data.root_pose_w
        return data.root_state_w[:, 0:7]

    def _body_pose_w(self):
        data = self.robot.data
        if hasattr(data, "body_link_pose_w"):
            return data.body_link_pose_w
        if hasattr(data, "body_pose_w"):
            return data.body_pose_w
        return data.body_state_w[:, :, 0:7]

    def _body_pose_b(self, body_id: int):
        root = self._root_pose_w()
        body = self._body_pose_w()[:, body_id, :]
        return subtract_frame_transforms(root[:, 0:3], root[:, 3:7], body[:, 0:3], body[:, 3:7])

    def _head_pose_w(self):
        if self.head_body_id is None:
            return None
        return self._body_pose_w()[:, self.head_body_id, :]

    def _tool_pose_b(self, side: str):
        if side == "left":
            pos, quat = self._body_pose_b(self.left_body_id)
            offset = quat_apply(quat, self.left_offset_local)
        else:
            pos, quat = self._body_pose_b(self.right_body_id)
            offset = quat_apply(quat, self.right_offset_local)
        return pos + offset, quat, offset

    def _body_jacobian(self, side: str):
        view = self.robot.root_view if hasattr(self.robot, "root_view") else self.robot.root_physx_view
        all_j = view.get_jacobians()
        if side == "left":
            return all_j[:, self.left_jacobian_id, :, self.left_arm_ids]
        return all_j[:, self.right_jacobian_id, :, self.right_arm_ids]

    @staticmethod
    def _skew(v: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(v[:, 0])
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        return torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1).view(-1, 3, 3)

    def _tool_jacobian(self, side: str, offset_b: torch.Tensor):
        body_j = self._body_jacobian(side)
        jv = body_j[:, 0:3, :]
        jw = body_j[:, 3:6, :]
        jv_tool = jv - torch.bmm(self._skew(offset_b), jw)
        return torch.cat((jv_tool, jw), dim=1)

    @staticmethod
    def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        return torch.cat((q[:, 0:1], -q[:, 1:4]), dim=-1)

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dim=-1,
        )

    @staticmethod
    def _limit_vector(v: torch.Tensor, max_norm: float) -> torch.Tensor:
        norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)
        return v * torch.clamp(max_norm / norm, max=1.0)

    def _vr_pos(self, pose: np.ndarray) -> torch.Tensor:
        return torch.tensor(pose[:3, 3], dtype=torch.float32, device=self.device).view(1, 3)

    def _update_targets_from_vr(self, tele_data, left_pos, right_pos, left_quat, right_quat):
        if self.left_home_pos is None:
            self.left_home_pos = left_pos.clone()
            self.right_home_pos = right_pos.clone()
            self.left_home_quat = left_quat.clone()
            self.right_home_quat = right_quat.clone()
            self.left_target_pos = left_pos.clone()
            self.right_target_pos = right_pos.clone()
            self.command_q = self.robot.data.joint_pos[:, self.arm_ids].clone()

        if not tele_data.motion_data_ready:
            return

        left_vr = self._vr_pos(tele_data.left_wrist_pose)
        right_vr = self._vr_pos(tele_data.right_wrist_pose)
        if self.xr_left_zero is None:
            self.xr_left_zero = left_vr.clone()
            self.xr_right_zero = right_vr.clone()
            print("[R1-A7 VR IK] calibrated XR zero")
            print("[R1-A7 VR IK] left zero:", self.xr_left_zero[0].tolist())
            print("[R1-A7 VR IK] right zero:", self.xr_right_zero[0].tolist())

        left_delta = torch.clamp((left_vr - self.xr_left_zero) * self.scale, -self.max_delta, self.max_delta)
        right_delta = torch.clamp((right_vr - self.xr_right_zero) * self.scale, -self.max_delta, self.max_delta)
        self.left_target_pos = self.left_home_pos + left_delta
        self.right_target_pos = self.right_home_pos + right_delta

    def _solve_arm(self, side: str, tool_pos, tool_quat, offset_b, target_pos, current_arm):
        pos_error = target_pos - tool_pos
        pos_step = self._limit_vector(pos_error, self.cartesian_step)
        j = self._tool_jacobian(side, offset_b)
        task = pos_step
        j_task = j[:, 0:3, :]

        if self.use_orientation:
            target_quat = self.left_home_quat if side == "left" else self.right_home_quat
            q_err = self._quat_multiply(target_quat, self._quat_conjugate(tool_quat))
            sign = torch.where(q_err[:, 0:1] < 0.0, -torch.ones_like(q_err[:, 0:1]), torch.ones_like(q_err[:, 0:1]))
            rot_step = self._limit_vector(2.0 * sign * q_err[:, 1:4], 0.04)
            task = torch.cat((task, self.orientation_gain * rot_step), dim=-1)
            j_task = j

        eye = torch.eye(j_task.shape[1], device=self.device, dtype=j_task.dtype).unsqueeze(0)
        lhs = torch.bmm(j_task, j_task.transpose(1, 2)) + (self.damping ** 2) * eye
        solved = torch.linalg.solve(lhs, task.unsqueeze(-1))
        dq = torch.bmm(j_task.transpose(1, 2), solved).squeeze(-1)
        max_step = min(self.joint_step, self.joint_speed * self.control_dt)
        dq = torch.clamp(torch.nan_to_num(dq), -max_step, max_step)
        return current_arm + dq, float(torch.linalg.norm(pos_error[0]).item())

    def _to_action(self, desired: torch.Tensor) -> torch.Tensor:
        return (desired - self.default_joint_pos)[:, self.action_joint_ids]

    def _transform_xr_image(self, frame: np.ndarray) -> np.ndarray:
        rotate = self.xr_image_rotate % 360
        if rotate == 90:
            frame = np.rot90(frame, k=3)
        elif rotate == 180:
            frame = np.rot90(frame, k=2)
        elif rotate == 270:
            frame = np.rot90(frame, k=1)

        if self.xr_image_flip in ("h", "horizontal", "x"):
            frame = np.flip(frame, axis=1)
        elif self.xr_image_flip in ("v", "vertical", "y"):
            frame = np.flip(frame, axis=0)
        elif self.xr_image_flip in ("both", "hv", "xy"):
            frame = np.flip(np.flip(frame, axis=1), axis=0)
        return np.ascontiguousarray(frame)

    def get_action(self, env) -> Optional[torch.Tensor]:
        tele_data = self.tv_wrapper.get_tele_data()
        self.frame_step += 1
        if self.xr_need_local_img and (self.frame_step % 2 == 0):
            try:
                get_camera_image(env)
            except Exception as exc:
                now = time.monotonic()
                if now - self.last_log > 1.0:
                    print(f"[R1-A7 VR IK] failed to update camera observation: {exc}")
        if self.xr_need_local_img and self.image_reader is not None:
            try:
                frame = self.image_reader.read_single_image("head")
                if frame is not None:
                    frame = self._transform_xr_image(frame)
                    self.tv_wrapper.render_to_xr(frame)
                    now = time.monotonic()
                    if now - self.last_image_log > 2.0:
                        print(f"[R1-A7 VR IK] rendered XR frame shape={tuple(frame.shape)} mean={float(frame.mean()):.2f}")
                        self.last_image_log = now
            except Exception as exc:
                now = time.monotonic()
                if now - self.last_log > 1.0:
                    print(f"[R1-A7 VR IK] failed to render XR image: {exc}")

        left_pos, left_quat, left_offset = self._tool_pose_b("left")
        right_pos, right_quat, right_offset = self._tool_pose_b("right")
        self._update_targets_from_vr(tele_data, left_pos, right_pos, left_quat, right_quat)

        current = self.robot.data.joint_pos
        left_current = current[:, self.left_arm_ids]
        right_current = current[:, self.right_arm_ids]
        left_next, left_err = self._solve_arm("left", left_pos, left_quat, left_offset, self.left_target_pos, left_current)
        right_next, right_err = self._solve_arm("right", right_pos, right_quat, right_offset, self.right_target_pos, right_current)
        next_arm = torch.cat((left_next, right_next), dim=-1)

        lead = torch.clamp(next_arm - current[:, self.arm_ids], -self.command_lead, self.command_lead)
        next_arm = current[:, self.arm_ids] + lead
        self.command_q = self.filter_alpha * next_arm + (1.0 - self.filter_alpha) * self.command_q

        desired = self.default_joint_pos.clone()
        if not self.preview_only and tele_data.motion_data_ready:
            desired[:, self.arm_ids] = self.command_q

        now = time.monotonic()
        if now - self.last_log > float(getattr(self.args_cli, "vr_ik_print_period", 0.4)):
            self.last_log = now
            ready = "ready" if tele_data.motion_data_ready else "waiting"
            ltar = self.left_target_pos[0].tolist() if self.left_target_pos is not None else None
            rtar = self.right_target_pos[0].tolist() if self.right_target_pos is not None else None
            head_pose = self._head_pose_w()
            head_text = ""
            if head_pose is not None:
                head_pos = [round(v, 4) for v in head_pose[0, 0:3].tolist()]
                head_quat = [round(v, 4) for v in head_pose[0, 3:7].tolist()]
                head_text = f" head_pos_w={head_pos} head_quat_w={head_quat}"
            print(
                f"[R1-A7 VR IK] {ready} preview={self.preview_only} "
                f"left_err={left_err:.4f} right_err={right_err:.4f} "
                f"left_target={ltar} right_target={rtar}"
                f"{head_text}"
            )

        return self._to_action(desired)

    def cleanup(self):
        try:
            self.tv_wrapper.close()
        except Exception:
            pass
        try:
            if self.image_reader is not None:
                self.image_reader.close()
        except Exception:
            pass
        print("[R1-A7 VR IK] cleanup")
