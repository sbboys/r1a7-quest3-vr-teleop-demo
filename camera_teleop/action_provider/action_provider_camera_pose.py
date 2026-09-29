# -*- coding: utf-8 -*-
"""R1-A7 camera-pose teleoperation ActionProvider.

Right/left human wrist motion from GeminiPoseSource is mapped to an R1-A7 arm
tool target in the robot base frame. The arm is solved with bounded DLS IK and
the Dex1 gripper is controlled by hand-open/closed estimation.
"""

from __future__ import annotations

from typing import Optional

import torch

from action_provider.action_base import ActionProvider
from action_provider.gemini_pose_source import GeminiPoseSource
from isaaclab.utils.math import quat_apply, subtract_frame_transforms


class CameraPoseActionProvider(ActionProvider):
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

    IK_BODY_CANDIDATES = {
        "left": (
            ("left_dex1_base_link", (0.0, 0.115, 0.0)),
            ("left_wrist_yaw_link", (0.1565, 0.0, 0.0)),
        ),
        "right": (
            ("right_dex1_base_link", (0.0, -0.115, 0.0)),
            ("right_wrist_yaw_link", (0.1565, 0.0, 0.0)),
        ),
    }

    LEFT_MASTER = "left_dex1_Joint1_1"
    LEFT_SLAVE = "left_dex1_Joint2_1"
    RIGHT_MASTER = "right_dex1_Joint1_1"
    RIGHT_SLAVE = "right_dex1_Joint2_1"

    GRIPPER_OPEN = -0.018
    GRIPPER_CLOSE = 0.018

    DLS_LAMBDA = 0.10
    ORIENTATION_GAIN = 0.12
    MAX_ORIENTATION_STEP = 0.045
    MAX_CARTESIAN_STEP = 0.018
    MAX_JOINT_STEP = 0.035
    MAX_JOINT_SPEED = 1.20
    MAX_COMMAND_LEAD = 0.140
    MAX_HUMAN_DELTA_M = 0.75
    COMMAND_FILTER_ALPHA = 0.65
    NULLSPACE_GAIN = 0.04

    def __init__(self, env, args_cli):
        super().__init__("R1A7CameraPoseTeleop")
        self.env = env
        self.args_cli = args_cli
        self.robot = env.scene["robot"]
        self.device = env.device
        self.num_envs = env.num_envs
        if self.num_envs != 1:
            raise RuntimeError(f"camera_pose requires num_envs=1, got {self.num_envs}")

        self.robot_arm = getattr(args_cli, "camera_pose_robot_arm", "left").lower()
        self.human_hand = getattr(args_cli, "camera_pose_human_hand", "right").lower()
        if self.robot_arm not in ("left", "right"):
            raise RuntimeError("--camera_pose_robot_arm must be left or right")

        self.joint_names = list(self.robot.joint_names)
        self.body_names = list(self.robot.body_names)
        missing = [
            name
            for name in self.LEFT_ARM_JOINT_NAMES + self.RIGHT_ARM_JOINT_NAMES
            if name not in self.joint_names
        ]
        if missing:
            raise RuntimeError(f"Missing R1-A7 arm joints: {missing}")
        self.left_arm_ids = [self.joint_names.index(name) for name in self.LEFT_ARM_JOINT_NAMES]
        self.right_arm_ids = [self.joint_names.index(name) for name in self.RIGHT_ARM_JOINT_NAMES]
        self.all_arm_ids = self.left_arm_ids + self.right_arm_ids
        self.arm_ids = self.left_arm_ids if self.robot_arm == "left" else self.right_arm_ids

        selected = next(
            (
                (name, offset)
                for name, offset in self.IK_BODY_CANDIDATES[self.robot_arm]
                if name in self.body_names
            ),
            None,
        )
        if selected is None:
            raise RuntimeError(f"No {self.robot_arm} IK body found. Bodies: {self.body_names}")
        self.ik_body_name, tool_offset = selected
        self.ik_body_id = self.body_names.index(self.ik_body_name)
        self.ik_jacobian_id = self.ik_body_id - 1 if self.robot.is_fixed_base else self.ik_body_id
        self.tool_offset_local = torch.tensor(tool_offset, dtype=torch.float32, device=self.device).view(1, 3)

        self.left_master_id = self._joint_id(self.LEFT_MASTER)
        self.left_slave_id = self._joint_id(self.LEFT_SLAVE)
        self.right_master_id = self._joint_id(self.RIGHT_MASTER)
        self.right_slave_id = self._joint_id(self.RIGHT_SLAVE)

        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.action_joint_ids = self._resolve_action_joint_ids()

        step_hz = float(getattr(args_cli, "step_hz", 30.0))
        self.control_dt = 1.0 / max(step_hz, 1.0)
        self.scale = float(getattr(args_cli, "camera_pose_scale", 0.45))
        self.max_age_s = float(getattr(args_cli, "camera_pose_max_age", 0.35))
        self.target_alpha = float(getattr(args_cli, "camera_pose_target_alpha", 0.65))
        self.max_cartesian_step = float(getattr(args_cli, "camera_pose_cartesian_step", self.MAX_CARTESIAN_STEP))
        self.max_joint_step = float(getattr(args_cli, "camera_pose_joint_step", self.MAX_JOINT_STEP))
        self.max_joint_speed = float(getattr(args_cli, "camera_pose_joint_speed", self.MAX_JOINT_SPEED))
        self.max_command_lead = float(getattr(args_cli, "camera_pose_command_lead", self.MAX_COMMAND_LEAD))
        self.joint_limit_margin = float(getattr(args_cli, "camera_pose_joint_limit_margin", 0.03))
        self.singularity_threshold = float(getattr(args_cli, "camera_pose_singularity_threshold", 0.035))
        self.joint_correction_enabled = bool(getattr(args_cli, "camera_pose_joint_correction", False))
        self.joint_correction_gain = float(getattr(args_cli, "camera_pose_joint_correction_gain", 0.35))
        self.joint_correction_max_step = float(getattr(args_cli, "camera_pose_joint_correction_max_step", 0.012))
        self.wrist_orientation_enabled = bool(getattr(args_cli, "camera_pose_wrist_orientation", False))
        self.wrist_orientation_gain = float(getattr(args_cli, "camera_pose_wrist_orientation_gain", 0.55))
        self.wrist_orientation_max_step = float(getattr(args_cli, "camera_pose_wrist_orientation_max_step", 0.018))
        self.planar_only = bool(getattr(args_cli, "camera_pose_planar_only", False))
        self.lock_wrist = bool(getattr(args_cli, "camera_pose_lock_wrist", False))
        self.elbow_assist_enabled = bool(getattr(args_cli, "camera_pose_elbow_assist", False))
        self.elbow_assist_gain = float(getattr(args_cli, "camera_pose_elbow_assist_gain", 0.8))
        self.elbow_assist_max_step = float(getattr(args_cli, "camera_pose_elbow_assist_max_step", 0.012))
        self.elbow_assist_sign = float(getattr(args_cli, "camera_pose_elbow_assist_sign", -1.0))
        self.torso_safe = bool(getattr(args_cli, "camera_pose_torso_safe", False))
        self.direct_planar = bool(getattr(args_cli, "camera_pose_direct_planar", False))
        self.direct_roll_gain = float(getattr(args_cli, "camera_pose_direct_roll_gain", 2.2))
        self.direct_pitch_gain = float(getattr(args_cli, "camera_pose_direct_pitch_gain", 1.6))
        self.direct_yaw_gain = float(getattr(args_cli, "camera_pose_direct_yaw_gain", 1.4))
        self.direct_elbow_gain = float(getattr(args_cli, "camera_pose_direct_elbow_gain", 1.0))
        self.direct_depth_gain = float(getattr(args_cli, "camera_pose_direct_depth_gain", 1.4))
        self.direct_depth_sign = float(getattr(args_cli, "camera_pose_direct_depth_sign", -1.0))
        self.direct_pitch_depth_ratio = float(getattr(args_cli, "camera_pose_direct_pitch_depth_ratio", 0.35))
        self.direct_elbow_depth_ratio = float(getattr(args_cli, "camera_pose_direct_elbow_depth_ratio", -0.85))
        self.direct_vertical_sign = float(getattr(args_cli, "camera_pose_direct_vertical_sign", 1.0))
        self.direct_elbow_vertical_ratio = float(getattr(args_cli, "camera_pose_direct_elbow_vertical_ratio", 0.25))
        self.direct_skeleton_lift_gain = float(getattr(args_cli, "camera_pose_direct_skeleton_lift_gain", 1.10))
        self.direct_skeleton_side_roll_gain = float(getattr(args_cli, "camera_pose_direct_skeleton_side_roll_gain", 0.95))
        self.direct_skeleton_side_yaw_gain = float(getattr(args_cli, "camera_pose_direct_skeleton_side_yaw_gain", 0.45))
        self.direct_skeleton_reach_pitch_gain = float(getattr(args_cli, "camera_pose_direct_skeleton_reach_pitch_gain", 0.25))
        self.direct_skeleton_reach_yaw_gain = float(getattr(args_cli, "camera_pose_direct_skeleton_reach_yaw_gain", 0.35))
        self.direct_skeleton_elbow_gain = float(getattr(args_cli, "camera_pose_direct_skeleton_elbow_gain", 0.85))
        self.direct_skeleton_lift_elbow_gain = float(getattr(args_cli, "camera_pose_direct_skeleton_lift_elbow_gain", 0.20))
        self.direct_max_step = float(getattr(args_cli, "camera_pose_direct_max_step", 0.030))
        self.direct_view_vertical = bool(getattr(args_cli, "camera_pose_direct_view_vertical", False))
        self.direct_view_horizontal = bool(getattr(args_cli, "camera_pose_direct_view_horizontal", False))
        self.mirror_input = bool(getattr(args_cli, "camera_pose_mirror_input", False))
        self.direct_pitch_min = float(getattr(args_cli, "camera_pose_direct_pitch_min", -0.8))
        self.direct_pitch_max = float(getattr(args_cli, "camera_pose_direct_pitch_max", 1.2))
        self.direct_elbow_min = float(getattr(args_cli, "camera_pose_direct_elbow_min", 0.05))
        self.direct_elbow_max = float(getattr(args_cli, "camera_pose_direct_elbow_max", 1.5))
        self.lost_hold_s = float(getattr(args_cli, "camera_pose_lost_hold_s", 0.55))
        self.lost_return_s = float(getattr(args_cli, "camera_pose_lost_return_s", 1.40))
        self.binary_grip = bool(getattr(args_cli, "camera_pose_binary_grip", False))
        self.grip_threshold = float(getattr(args_cli, "camera_pose_grip_threshold", 0.55))
        self.debug_pose = bool(getattr(args_cli, "camera_pose_debug", False))
        self.last_min_singular = float("nan")

        workspace = getattr(args_cli, "camera_pose_workspace", "0.20,0.55,0.05,0.35,0.02,0.35")
        values = [float(v.strip()) for v in workspace.split(",")]
        if len(values) != 6:
            raise RuntimeError("--camera_pose_workspace must be xmin,xmax,ymin,ymax,zmin,zmax")
        self.workspace_min = torch.tensor([values[0], values[2], values[4]], device=self.device).view(1, 3)
        self.workspace_max = torch.tensor([values[1], values[3], values[5]], device=self.device).view(1, 3)

        self.arm_command = self.robot.data.joint_pos[:, self.arm_ids].clone()
        self.arm_home = self.default_joint_pos[:, self.arm_ids].clone()
        self.locked_wrist_command = self.arm_command[:, 4:7].clone()
        self.last_dq_command = torch.zeros_like(self.arm_command)
        self.initialized = False
        self.initial_tool_pos = None
        self.human_rel_zero = None
        self.human_joint_zero = None
        self.human_palm_zero = None
        self.human_view_zero = None
        self.latest_human_delta = None
        self.latest_human_view_delta = None
        self.latest_target = None
        self.last_direct_input = None
        self.last_direct_target = None
        self.last_camera_target_step = 0
        self.last_lost_age_s = 0.0
        self.target_tool_pos = None
        self.target_tool_quat = None
        self.filtered_target = None
        self.gripper_target = self.GRIPPER_OPEN
        self.step_count = 0
        self.last_log_step = 0

        self.source = GeminiPoseSource(
            hand=self.human_hand,
            show=bool(getattr(args_cli, "camera_pose_show", False)),
            mirror_view=bool(getattr(args_cli, "camera_pose_mirror_view", False)),
            filter_alpha=float(getattr(args_cli, "camera_pose_filter_alpha", 0.25)),
            debug=self.debug_pose,
            min_visibility=float(getattr(args_cli, "camera_pose_min_visibility", 0.35)),
            min_wrist_shoulder_m=float(getattr(args_cli, "camera_pose_min_wrist_shoulder_m", 0.035)),
        )

        print("\n" + "=" * 96)
        print("[R1-A7 CAMERA POSE] controller constructed")
        print("[R1-A7 CAMERA POSE] robot arm:", self.robot_arm)
        print("[R1-A7 CAMERA POSE] human hand:", self.human_hand)
        print("[R1-A7 CAMERA POSE] IK body/Jacobian id:", self.ik_body_name, self.ik_jacobian_id)
        print("[R1-A7 CAMERA POSE] controlled arm ids:", self.arm_ids)
        print("[R1-A7 CAMERA POSE] dual-arm limit ids:", self.all_arm_ids)
        print("[R1-A7 CAMERA POSE] workspace:", values)
        print("[R1-A7 CAMERA POSE] scale:", self.scale)
        print(
            "[R1-A7 CAMERA POSE] responsiveness:",
            {
                "target_alpha": self.target_alpha,
                "cartesian_step": self.max_cartesian_step,
                "joint_step": self.max_joint_step,
                "joint_speed": self.max_joint_speed,
                "command_lead": self.max_command_lead,
                "joint_limit_margin": self.joint_limit_margin,
                "singularity_threshold": self.singularity_threshold,
                "joint_correction": self.joint_correction_enabled,
                "joint_correction_gain": self.joint_correction_gain,
                "joint_correction_max_step": self.joint_correction_max_step,
                "wrist_orientation": self.wrist_orientation_enabled,
                "wrist_orientation_gain": self.wrist_orientation_gain,
                "wrist_orientation_max_step": self.wrist_orientation_max_step,
                "planar_only": self.planar_only,
                "lock_wrist": self.lock_wrist,
                "elbow_assist": self.elbow_assist_enabled,
                "elbow_assist_gain": self.elbow_assist_gain,
                "elbow_assist_max_step": self.elbow_assist_max_step,
                "elbow_assist_sign": self.elbow_assist_sign,
                "torso_safe": self.torso_safe,
                "direct_planar": self.direct_planar,
                "direct_roll_gain": self.direct_roll_gain,
                "direct_pitch_gain": self.direct_pitch_gain,
                "direct_yaw_gain": self.direct_yaw_gain,
                "direct_elbow_gain": self.direct_elbow_gain,
                "direct_depth_gain": self.direct_depth_gain,
                "direct_depth_sign": self.direct_depth_sign,
                "direct_pitch_depth_ratio": self.direct_pitch_depth_ratio,
                "direct_elbow_depth_ratio": self.direct_elbow_depth_ratio,
                "direct_vertical_sign": self.direct_vertical_sign,
                "direct_elbow_vertical_ratio": self.direct_elbow_vertical_ratio,
                "direct_skeleton_lift_gain": self.direct_skeleton_lift_gain,
                "direct_skeleton_side_roll_gain": self.direct_skeleton_side_roll_gain,
                "direct_skeleton_side_yaw_gain": self.direct_skeleton_side_yaw_gain,
                "direct_skeleton_reach_pitch_gain": self.direct_skeleton_reach_pitch_gain,
                "direct_skeleton_reach_yaw_gain": self.direct_skeleton_reach_yaw_gain,
                "direct_skeleton_elbow_gain": self.direct_skeleton_elbow_gain,
                "direct_skeleton_lift_elbow_gain": self.direct_skeleton_lift_elbow_gain,
                "direct_max_step": self.direct_max_step,
                "direct_view_vertical": self.direct_view_vertical,
                "direct_view_horizontal": self.direct_view_horizontal,
                "mirror_input": self.mirror_input,
                "direct_pitch_range": (self.direct_pitch_min, self.direct_pitch_max),
                "direct_elbow_range": (self.direct_elbow_min, self.direct_elbow_max),
                "lost_hold_s": self.lost_hold_s,
                "lost_return_s": self.lost_return_s,
                "binary_grip": self.binary_grip,
            },
        )
        print("[R1-A7 CAMERA POSE] joint-limit source:", self._joint_limit_source_name())
        print("=" * 96 + "\n")

    def start(self):
        self.source.start()
        super().start()

    def stop(self):
        self.source.stop()
        super().stop()

    def cleanup(self):
        self.source.stop()
        print("[R1-A7 CAMERA POSE] provider cleanup")

    def _joint_id(self, name: str) -> int:
        if name not in self.joint_names:
            raise RuntimeError(f"Missing required joint: {name}")
        return self.joint_names.index(name)

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
                term = terms.get("joint_pos")
                if term is None and len(terms) == 1:
                    term = next(iter(terms.values()))
            elif isinstance(terms, (list, tuple)) and len(terms) == 1:
                term = terms[0]

        if term is not None:
            for attr in ("_joint_ids", "joint_ids"):
                ids = self._ids_to_list(getattr(term, attr, None), len(self.joint_names))
                if ids is not None:
                    if len(ids) != len(self.joint_names):
                        raise RuntimeError(
                            f"camera_pose expects all {len(self.joint_names)} joints, got {len(ids)}"
                        )
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

    def _ik_body_pose_b(self):
        root = self._root_pose_w()
        body = self._body_pose_w()[:, self.ik_body_id, :]
        return subtract_frame_transforms(root[:, 0:3], root[:, 3:7], body[:, 0:3], body[:, 3:7])

    def _tool_pose_b(self):
        body_pos_b, body_quat_b = self._ik_body_pose_b()
        offset_b = quat_apply(body_quat_b, self.tool_offset_local)
        return body_pos_b + offset_b, body_quat_b, offset_b

    def _body_jacobian(self):
        view = self.robot.root_view if hasattr(self.robot, "root_view") else self.robot.root_physx_view
        return view.get_jacobians()[:, self.ik_jacobian_id, :, self.arm_ids]

    @staticmethod
    def _skew(v: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(v[:, 0])
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        return torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1).view(-1, 3, 3)

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

    def _orientation_error(self, current: torch.Tensor) -> torch.Tensor:
        q_err = self._quat_multiply(self.target_tool_quat, self._quat_conjugate(current))
        sign = torch.where(q_err[:, 0:1] < 0.0, -torch.ones_like(q_err[:, 0:1]), torch.ones_like(q_err[:, 0:1]))
        return 2.0 * sign * q_err[:, 1:4]

    def _tool_jacobian(self, offset_b: torch.Tensor) -> torch.Tensor:
        body_j = self._body_jacobian()
        jv = body_j[:, 0:3, :]
        jw = body_j[:, 3:6, :]
        jv_tool = jv - torch.bmm(self._skew(offset_b), jw)
        return torch.cat((jv_tool, jw), dim=1)

    def _write_grippers(self, desired: torch.Tensor, value: float):
        desired[:, self.left_master_id] = value if self.robot_arm == "left" else self.GRIPPER_OPEN
        desired[:, self.left_slave_id] = value if self.robot_arm == "left" else self.GRIPPER_OPEN
        desired[:, self.right_master_id] = value if self.robot_arm == "right" else self.GRIPPER_OPEN
        desired[:, self.right_slave_id] = value if self.robot_arm == "right" else self.GRIPPER_OPEN

    def _joint_limits(self):
        data = self.robot.data
        for name in ("soft_joint_pos_limits", "joint_pos_limits", "default_joint_pos_limits", "default_joint_limits"):
            limits = getattr(data, name, None)
            if limits is not None:
                return limits
        return None

    def _joint_limit_source_name(self) -> str:
        data = self.robot.data
        for name in ("soft_joint_pos_limits", "joint_pos_limits", "default_joint_pos_limits", "default_joint_limits"):
            if getattr(data, name, None) is not None:
                return name
        return "none"

    def _apply_joint_limits(self, desired: torch.Tensor, joint_ids=None) -> torch.Tensor:
        limits = self._joint_limits()
        if limits is None:
            return desired

        if joint_ids is None:
            joint_ids = self.all_arm_ids + [
                self.left_master_id,
                self.left_slave_id,
                self.right_master_id,
                self.right_slave_id,
            ]
        ids_t = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        margin = max(0.0, self.joint_limit_margin)
        lower = limits[:, ids_t, 0] + margin
        upper = limits[:, ids_t, 1] - margin
        inverted = lower > upper
        lower = torch.where(inverted, limits[:, ids_t, 0], lower)
        upper = torch.where(inverted, limits[:, ids_t, 1], upper)
        desired[:, ids_t] = torch.maximum(torch.minimum(desired[:, ids_t], upper), lower)

        gripper_ids = torch.tensor(
            [self.left_master_id, self.left_slave_id, self.right_master_id, self.right_slave_id],
            dtype=torch.long,
            device=self.device,
        )
        desired[:, gripper_ids] = torch.clamp(desired[:, gripper_ids], self.GRIPPER_OPEN, self.GRIPPER_CLOSE)
        return desired

    def _to_action(self, desired: torch.Tensor) -> torch.Tensor:
        return (desired - self.default_joint_pos)[:, self.action_joint_ids]

    @staticmethod
    def _limit_vector(v: torch.Tensor, max_norm: float) -> torch.Tensor:
        norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)
        return v * torch.clamp(max_norm / norm, max=1.0)

    def _camera_rel_to_base_delta(self, rel) -> torch.Tensor:
        rel_t = torch.tensor(rel, dtype=torch.float32, device=self.device).view(1, 3)
        # Orbbec depth returns +Z away from the camera. The operator-facing
        # convention used here is: camera +X right, +Y down, -Z forward;
        # robot -X forward, +Y down, +Z left.
        if self.planar_only:
            # Isaac robot base frame: X forward, Y left, Z up. In planar mode
            # keep depth fixed and use only camera-view left/right and up/down.
            delta = torch.stack((torch.zeros_like(rel_t[:, 2]), -rel_t[:, 0], -rel_t[:, 1]), dim=-1).view(1, 3)
        else:
            delta = torch.stack((-rel_t[:, 2], rel_t[:, 1], -rel_t[:, 0]), dim=-1).view(1, 3)
        return delta * self.scale

    def _apply_wrist_lock(self, arm_command: torch.Tensor) -> torch.Tensor:
        if not self.lock_wrist:
            return arm_command
        locked = arm_command.clone()
        locked[:, 4:7] = self.locked_wrist_command
        return locked

    def _human_joint_features(self, target) -> Optional[torch.Tensor]:
        if target.shoulder_m is None or target.elbow_m is None or target.wrist_m is None:
            return None

        shoulder = torch.tensor(target.shoulder_m, dtype=torch.float32, device=self.device)
        elbow = torch.tensor(target.elbow_m, dtype=torch.float32, device=self.device)
        wrist = torch.tensor(target.wrist_m, dtype=torch.float32, device=self.device)
        upper_raw = elbow - shoulder
        forearm_raw = wrist - elbow
        upper = torch.stack((upper_raw[0], upper_raw[1], -upper_raw[2]))
        forearm = torch.stack((forearm_raw[0], forearm_raw[1], -forearm_raw[2]))
        upper_norm = torch.linalg.norm(upper).clamp_min(1e-6)
        forearm_norm = torch.linalg.norm(forearm).clamp_min(1e-6)
        upper_dir = upper / upper_norm
        elbow_cos = torch.clamp(torch.dot(upper, forearm) / (upper_norm * forearm_norm), -1.0, 1.0)
        elbow_bend = torch.pi - torch.acos(elbow_cos)

        # Display camera frame: +X right, +Y down, -Z forward.
        lift = -upper_dir[1]
        side = upper_dir[0]
        reach = -upper_dir[2]
        return torch.stack((lift, side, reach, elbow_bend)).view(1, 4)

    def _apply_human_joint_correction(
        self,
        target,
        ik_arm_command: torch.Tensor,
        current_arm: torch.Tensor,
    ) -> torch.Tensor:
        if not self.joint_correction_enabled:
            return ik_arm_command

        features = self._human_joint_features(target)
        if features is None:
            return ik_arm_command
        if self.human_joint_zero is None:
            self.human_joint_zero = features.clone()
            print("[R1-A7 CAMERA POSE] calibrated human joint zero:", features[0].detach().cpu().tolist())
            return ik_arm_command

        delta = torch.clamp(features - self.human_joint_zero, -1.2, 1.2)
        lift = delta[:, 0]
        side = delta[:, 1]
        reach = delta[:, 2]
        elbow_bend = delta[:, 3]

        mapped = torch.zeros_like(ik_arm_command)
        side_sign = -1.0 if self.robot_arm == "left" else 1.0
        mapped[:, 0] = -0.45 * lift + 0.20 * reach
        mapped[:, 1] = side_sign * 0.35 * side
        mapped[:, 2] = side_sign * 0.28 * reach
        mapped[:, 3] = 0.60 * elbow_bend
        mapped[:, 4] = 0.00
        mapped[:, 5] = -0.18 * lift
        mapped[:, 6] = side_sign * 0.18 * side

        gain = max(0.0, min(1.0, self.joint_correction_gain))
        human_arm_command = self.arm_home + mapped
        blended = (1.0 - gain) * ik_arm_command + gain * human_arm_command
        max_step = max(0.0, self.joint_correction_max_step)
        correction_step = torch.clamp(blended - ik_arm_command, -max_step, max_step)
        corrected = ik_arm_command + correction_step
        lead = torch.clamp(corrected - current_arm, -self.max_command_lead, self.max_command_lead)
        return current_arm + lead

    @staticmethod
    def _wrap_torch_angles(angles: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angles + torch.pi, 2.0 * torch.pi) - torch.pi

    def _apply_palm_wrist_orientation(
        self,
        target,
        arm_command: torch.Tensor,
        current_arm: torch.Tensor,
    ) -> torch.Tensor:
        if not self.wrist_orientation_enabled or target is None or target.palm_angles_rad is None:
            return arm_command

        palm = torch.tensor(target.palm_angles_rad, dtype=torch.float32, device=self.device).view(1, 3)
        if self.human_palm_zero is None:
            self.human_palm_zero = palm.clone()
            print("[R1-A7 CAMERA POSE] calibrated human palm zero:", palm[0].detach().cpu().tolist())
            return arm_command

        delta = torch.clamp(self._wrap_torch_angles(palm - self.human_palm_zero), -1.25, 1.25)
        palm_roll = delta[:, 0]
        palm_pitch = delta[:, 1]
        palm_yaw = delta[:, 2]

        wrist_delta = torch.zeros_like(arm_command)
        wrist_delta[:, 4] = palm_roll
        wrist_delta[:, 5] = palm_pitch
        wrist_delta[:, 6] = -palm_yaw

        gain = max(0.0, min(1.0, self.wrist_orientation_gain))
        max_step = max(0.0, self.wrist_orientation_max_step)
        target_command = arm_command + gain * wrist_delta
        wrist_step = torch.clamp(target_command - arm_command, -max_step, max_step)
        corrected = arm_command + wrist_step
        lead = torch.clamp(corrected - current_arm, -self.max_command_lead, self.max_command_lead)
        return current_arm + lead

    def _update_target_from_camera(self):
        target = self.source.get_latest(self.max_age_s)
        if target is None or not target.valid:
            self.latest_target = None
            self.latest_human_delta = None
            self.latest_human_view_delta = None
            return False
        self.latest_target = target

        if self.human_rel_zero is None:
            self.human_rel_zero = target.wrist_rel_m.copy()
            print("[R1-A7 CAMERA POSE] calibrated human zero:", self.human_rel_zero.tolist())
        target_view = getattr(target, "wrist_rel_view", None)
        if target_view is not None and self.human_view_zero is None:
            self.human_view_zero = target_view.copy()
            print("[R1-A7 CAMERA POSE] calibrated view zero:", self.human_view_zero.tolist())

        human_delta = target.wrist_rel_m - self.human_rel_zero
        if float(torch.linalg.norm(torch.tensor(human_delta, dtype=torch.float32)).item()) > self.MAX_HUMAN_DELTA_M:
            self.latest_human_delta = None
            self.latest_human_view_delta = None
            return False
        self.latest_human_delta = human_delta.copy()
        if target_view is not None and self.human_view_zero is not None:
            self.latest_human_view_delta = target_view - self.human_view_zero
        else:
            self.latest_human_view_delta = None
        raw_target = self.initial_tool_pos + self._camera_rel_to_base_delta(human_delta)
        raw_target = torch.maximum(torch.minimum(raw_target, self.workspace_max), self.workspace_min)

        if self.filtered_target is None:
            self.filtered_target = raw_target.clone()
        else:
            alpha = max(0.01, min(1.0, self.target_alpha))
            self.filtered_target = (1.0 - alpha) * self.filtered_target + alpha * raw_target
        self.target_tool_pos = self.filtered_target.clone()

        grip = max(0.0, min(1.0, float(target.grip)))
        if self.binary_grip:
            grip = 1.0 if grip >= self.grip_threshold else 0.0
        self.gripper_target = self.GRIPPER_OPEN + grip * (self.GRIPPER_CLOSE - self.GRIPPER_OPEN)
        return True

    def _apply_elbow_assist(self, arm_command, current_arm):
        if not self.elbow_assist_enabled or self.latest_human_delta is None:
            return arm_command

        vertical = float(self.latest_human_delta[1])
        desired_elbow = self.arm_home[:, 3] + self.elbow_assist_sign * self.elbow_assist_gain * vertical
        max_step = max(0.0, self.elbow_assist_max_step)
        step = torch.clamp(desired_elbow - arm_command[:, 3], -max_step, max_step)

        corrected = arm_command.clone()
        corrected[:, 3] = arm_command[:, 3] + step
        lead = torch.clamp(corrected - current_arm, -self.max_command_lead, self.max_command_lead)
        return current_arm + lead

    def _apply_torso_safety_limits(self, arm_command):
        if not self.torso_safe:
            return arm_command

        corrected = arm_command.clone()
        if self.robot_arm == "left":
            lower = torch.tensor([-1.10, -0.35, -1.15, 0.05], device=self.device, dtype=arm_command.dtype).view(1, 4)
            upper = torch.tensor([1.15, 1.10, 1.15, 1.65], device=self.device, dtype=arm_command.dtype).view(1, 4)
        else:
            lower = torch.tensor([-1.10, -1.10, -1.15, 0.18], device=self.device, dtype=arm_command.dtype).view(1, 4)
            upper = torch.tensor([1.15, 0.35, 1.15, 1.65], device=self.device, dtype=arm_command.dtype).view(1, 4)
        corrected[:, :4] = torch.maximum(torch.minimum(corrected[:, :4], upper), lower)
        return corrected

    def _direct_camera_inputs(self):
        horizontal_metric = float(self.latest_human_delta[0])
        horizontal = horizontal_metric
        vertical_metric = float(self.latest_human_delta[1])
        vertical = vertical_metric
        depth_metric = float(self.latest_human_delta[2])
        direct_reach = self.direct_depth_sign * depth_metric
        if self.latest_human_view_delta is not None:
            if self.direct_view_horizontal:
                horizontal = float(self.latest_human_view_delta[0])
            if self.direct_view_vertical:
                vertical = float(self.latest_human_view_delta[1])
        raw_vertical = vertical
        vertical *= self.direct_vertical_sign
        if self.mirror_input:
            horizontal = -horizontal
        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "vertical_metric": vertical_metric,
            "horizontal_metric": horizontal_metric,
            "raw_vertical": raw_vertical,
            "depth_metric": depth_metric,
            "direct_reach": direct_reach,
        }

    def _retarget_human_arm_to_a7(self, inputs, current_arm):
        side = 1.0 if self.robot_arm == "left" else -1.0
        horizontal = inputs["horizontal"]
        vertical = inputs["vertical"]
        direct_reach = inputs["direct_reach"]
        target = self.arm_home.clone()

        # Wrist-relative camera motion provides robust coarse control even
        # when full shoulder-elbow-wrist depth is not available.
        target[:, 0] = self.arm_home[:, 0] + self.direct_pitch_gain * vertical
        target[:, 1] = self.arm_home[:, 1] - side * 0.45 * self.direct_roll_gain * horizontal
        target[:, 2] = self.arm_home[:, 2] + side * self.direct_yaw_gain * horizontal
        elbow_vertical_ratio = max(0.0, min(1.0, self.direct_elbow_vertical_ratio))
        target[:, 3] = self.arm_home[:, 3] + elbow_vertical_ratio * self.direct_elbow_gain * vertical
        target[:, 0] += self.direct_pitch_depth_ratio * self.direct_depth_gain * direct_reach
        target[:, 3] += self.direct_elbow_depth_ratio * self.direct_depth_gain * direct_reach

        features = self._human_joint_features(self.latest_target)
        skeleton_delta_cpu = None
        if features is not None:
            if self.human_joint_zero is None:
                self.human_joint_zero = features.clone()
                print("[R1-A7 CAMERA POSE] calibrated direct skeleton zero:", features[0].detach().cpu().tolist())
            skeleton_delta = torch.clamp(features - self.human_joint_zero, -1.4, 1.4)
            skeleton_delta_cpu = skeleton_delta[0].detach().cpu().tolist()
            lift = skeleton_delta[:, 0]
            side_feature = skeleton_delta[:, 1]
            skeleton_reach = skeleton_delta[:, 2]
            elbow_bend = skeleton_delta[:, 3]

            # A7 7-DoF upper-arm retargeting:
            # lift -> shoulder pitch, side -> shoulder roll/yaw,
            # reach -> shoulder pitch/yaw, bend -> elbow.
            target[:, 0] += (
                self.direct_skeleton_lift_gain * self.direct_vertical_sign * lift
                + self.direct_skeleton_reach_pitch_gain * skeleton_reach
            )
            target[:, 1] += -side * self.direct_skeleton_side_roll_gain * side_feature
            target[:, 2] += side * (
                self.direct_skeleton_reach_yaw_gain * skeleton_reach
                + self.direct_skeleton_side_yaw_gain * side_feature
            )
            target[:, 3] += (
                self.direct_skeleton_elbow_gain * elbow_bend
                + self.direct_skeleton_lift_elbow_gain * self.direct_vertical_sign * lift
            )

        target[:, 0] = torch.clamp(target[:, 0], self.direct_pitch_min, self.direct_pitch_max)
        target[:, 3] = torch.clamp(target[:, 3], self.direct_elbow_min, self.direct_elbow_max)
        if self.lock_wrist:
            target[:, 4:7] = self.locked_wrist_command
        else:
            target[:, 4:7] = current_arm[:, 4:7]
        return target, skeleton_delta_cpu

    def _step_arm_command_toward(self, target, current_arm, max_step):
        max_step = max(0.0, float(max_step))
        command_base = self.arm_command if self.arm_command.shape == current_arm.shape else current_arm
        step = torch.clamp(target - command_base, -max_step, max_step)
        command = command_base + step
        lead = torch.clamp(command - current_arm, -self.max_command_lead, self.max_command_lead)
        self.last_dq_command = step.clone()
        return current_arm + lead

    def _hold_or_return_after_target_loss(self, current_arm):
        lost_steps = max(0, self.step_count - self.last_camera_target_step)
        self.last_lost_age_s = lost_steps * self.control_dt
        if self.last_lost_age_s <= max(0.0, self.lost_hold_s):
            hold_command = current_arm + torch.clamp(
                self.arm_command - current_arm,
                -self.max_command_lead,
                self.max_command_lead,
            )
            self.last_dq_command = torch.zeros_like(current_arm)
            return hold_command

        return_s = max(self.lost_return_s, self.control_dt)
        max_step = max(0.002, self.max_joint_speed * self.control_dt / return_s)
        return self._step_arm_command_toward(self.arm_home, current_arm, max_step)

    def _solve_direct_planar(self, current_arm):
        if self.latest_human_delta is None:
            self.last_direct_input = None
            self.last_direct_target = None
            return current_arm

        inputs = self._direct_camera_inputs()
        target, skeleton_delta = self._retarget_human_arm_to_a7(inputs, current_arm)
        max_step = min(max(0.0, self.direct_max_step), self.max_joint_speed * self.control_dt)
        command = self._step_arm_command_toward(target, current_arm, max_step)
        self.last_direct_input = (
            inputs["horizontal"],
            inputs["vertical"],
            inputs["vertical_metric"],
            inputs["horizontal_metric"],
            inputs["raw_vertical"],
            inputs["depth_metric"],
            inputs["direct_reach"],
            skeleton_delta,
        )
        self.last_direct_target = target.clone()
        return command

    def _solve_tool_pose_ik(self, tool_pos, tool_quat, offset_b, current_arm):
        pos_error = self.target_tool_pos - tool_pos
        pos_step = self._limit_vector(pos_error, self.max_cartesian_step)
        rot_error = self._orientation_error(tool_quat)
        rot_step = self._limit_vector(rot_error, self.MAX_ORIENTATION_STEP)

        j = self._tool_jacobian(offset_b)
        if self.lock_wrist:
            solve_dofs = 4
            j_solve = j[:, :3, :solve_dofs]
            task_vec = pos_step
            eye_task = torch.eye(3, device=self.device, dtype=j.dtype).unsqueeze(0)
        else:
            solve_dofs = len(self.arm_ids)
            j_solve = j[:, :, :solve_dofs]
            task_vec = torch.cat((pos_step, self.ORIENTATION_GAIN * rot_step), dim=-1)
            eye_task = torch.eye(6, device=self.device, dtype=j.dtype).unsqueeze(0)
        singular_values = torch.linalg.svdvals(j_solve)
        min_singular = singular_values[:, -1].clamp_min(0.0)
        self.last_min_singular = float(min_singular[0].detach().cpu().item())
        threshold = max(self.singularity_threshold, 1e-6)
        slowdown = torch.clamp(min_singular / threshold, min=0.15, max=1.0).unsqueeze(-1)
        task_vec = task_vec * slowdown
        extra_damping = torch.clamp(threshold - min_singular, min=0.0) * 4.0
        dls_lambda = (self.DLS_LAMBDA + extra_damping).view(-1, 1, 1)
        lhs = torch.bmm(j_solve, j_solve.transpose(1, 2)) + (dls_lambda ** 2) * eye_task
        solved = torch.linalg.solve(lhs, task_vec.unsqueeze(-1))
        j_t = j_solve.transpose(1, 2)
        dq_task_solve = torch.bmm(j_t, solved).squeeze(-1)

        pinv = torch.bmm(j_t, torch.linalg.solve(lhs, eye_task))
        eye_solve = torch.eye(solve_dofs, device=self.device, dtype=j.dtype).unsqueeze(0)
        null_projector = eye_solve - torch.bmm(pinv, j_solve)
        home_error = self.arm_home[:, :solve_dofs] - current_arm[:, :solve_dofs]
        dq_null = torch.bmm(null_projector, home_error.unsqueeze(-1)).squeeze(-1)
        dq_solve = dq_task_solve + self.NULLSPACE_GAIN * dq_null
        dq = torch.zeros_like(current_arm)
        dq[:, :solve_dofs] = dq_solve

        max_step = min(self.max_joint_step, self.max_joint_speed * self.control_dt)
        dq = torch.clamp(torch.nan_to_num(dq), -max_step, max_step)
        candidate = self.arm_command + dq
        filtered = self.COMMAND_FILTER_ALPHA * candidate + (1.0 - self.COMMAND_FILTER_ALPHA) * self.arm_command
        lead = torch.clamp(filtered - current_arm, -self.max_command_lead, self.max_command_lead)
        self.arm_command = current_arm + lead

        limits = self._joint_limits()
        if limits is not None:
            ids_t = torch.tensor(self.arm_ids, dtype=torch.long, device=self.device)
            margin = max(0.0, self.joint_limit_margin)
            arm_limits = limits[:, ids_t, :]
            lower = arm_limits[..., 0] + margin
            upper = arm_limits[..., 1] - margin
            inverted = lower > upper
            lower = torch.where(inverted, arm_limits[..., 0], lower)
            upper = torch.where(inverted, arm_limits[..., 1], upper)
            self.arm_command = torch.maximum(torch.minimum(self.arm_command, upper), lower)

        self.last_dq_command = dq.clone()
        return torch.nan_to_num(self.arm_command)

    def _initialize(self):
        tool_pos, tool_quat, _ = self._tool_pose_b()
        self.initial_tool_pos = tool_pos.clone()
        self.target_tool_pos = tool_pos.clone()
        self.filtered_target = tool_pos.clone()
        self.target_tool_quat = tool_quat.clone()
        self.human_rel_zero = None
        self.human_joint_zero = None
        self.human_palm_zero = None
        self.human_view_zero = None
        self.latest_human_delta = None
        self.latest_human_view_delta = None
        self.latest_target = None
        self.last_direct_input = None
        self.last_direct_target = None
        self.last_camera_target_step = self.step_count
        self.last_lost_age_s = 0.0
        self.arm_command = self.robot.data.joint_pos[:, self.arm_ids].clone()
        self.locked_wrist_command = self.arm_command[:, 4:7].clone()
        self.initialized = True
        print("[R1-A7 CAMERA POSE] initialized at tool pose:", tool_pos[0].tolist())
        if self.lock_wrist:
            print("[R1-A7 CAMERA POSE] locked wrist joints:", self.locked_wrist_command[0].tolist())

    def get_action(self, env) -> Optional[torch.Tensor]:
        del env
        self.step_count += 1
        current = self.robot.data.joint_pos.clone()
        desired = self.default_joint_pos.clone()

        if not self.initialized:
            desired[:, self.arm_ids] = current[:, self.arm_ids]
            self._write_grippers(desired, self.GRIPPER_OPEN)
            desired = self._apply_joint_limits(desired)
            if self.step_count > max(8, int(0.35 / self.control_dt)):
                self._initialize()
            return self._to_action(desired)

        has_camera_target = self._update_target_from_camera()
        tool_pos, tool_quat, offset_b = self._tool_pose_b()
        current_arm = current[:, self.arm_ids]

        if has_camera_target:
            self.last_camera_target_step = self.step_count
            self.last_lost_age_s = 0.0
            if self.direct_planar:
                arm_command = self._solve_direct_planar(current_arm)
            else:
                arm_command = self._solve_tool_pose_ik(tool_pos, tool_quat, offset_b, current_arm)
                arm_command = self._apply_human_joint_correction(self.latest_target, arm_command, current_arm)
                arm_command = self._apply_palm_wrist_orientation(self.latest_target, arm_command, current_arm)
                arm_command = self._apply_elbow_assist(arm_command, current_arm)
            arm_command = self._apply_torso_safety_limits(arm_command)
            arm_command = self._apply_wrist_lock(arm_command)
            self.arm_command = arm_command.clone()
            desired[:, self.arm_ids] = arm_command
        else:
            desired[:, self.arm_ids] = self._hold_or_return_after_target_loss(current_arm)

        self._write_grippers(desired, self.gripper_target)
        desired = self._apply_joint_limits(desired)
        if self.initialized:
            self.arm_command = desired[:, self.arm_ids].clone()
        action = self._to_action(desired)

        if self.step_count - self.last_log_step >= max(1, int(0.5 / self.control_dt)):
            self.last_log_step = self.step_count
            err = float(torch.linalg.norm(self.target_tool_pos - tool_pos, dim=-1)[0].item())
            direct_debug = ""
            if self.direct_planar and self.last_direct_input is not None and self.last_direct_target is not None:
                direct_target = self.last_direct_target[0].detach().cpu().tolist()
                current_arm_list = current_arm[0].detach().cpu().tolist()
                command_arm_list = self.arm_command[0].detach().cpu().tolist()
                skeleton_debug = ""
                if self.last_direct_input[7] is not None:
                    sk = self.last_direct_input[7]
                    skeleton_debug = f" skel=({sk[0]:+.3f},{sk[1]:+.3f},{sk[2]:+.3f},{sk[3]:+.3f})"
                direct_debug = (
                    f" direct_in=({self.last_direct_input[0]:+.3f},{self.last_direct_input[1]:+.3f}"
                    f",rawY={self.last_direct_input[4]:+.3f},mY={self.last_direct_input[2]:+.3f},mX={self.last_direct_input[3]:+.3f}"
                    f",mZ={self.last_direct_input[5]:+.3f},reach={self.last_direct_input[6]:+.3f})"
                    f"{skeleton_debug}"
                    f" pitch cur/cmd/tgt={current_arm_list[0]:+.3f}/{command_arm_list[0]:+.3f}/{direct_target[0]:+.3f}"
                    f" roll cur/cmd/tgt={current_arm_list[1]:+.3f}/{command_arm_list[1]:+.3f}/{direct_target[1]:+.3f}"
                    f" yaw cur/cmd/tgt={current_arm_list[2]:+.3f}/{command_arm_list[2]:+.3f}/{direct_target[2]:+.3f}"
                    f" elbow cur/cmd/tgt={current_arm_list[3]:+.3f}/{command_arm_list[3]:+.3f}/{direct_target[3]:+.3f}"
                )
            print(
                f"[R1-A7 CAMERA POSE] target={'yes' if has_camera_target else 'no '} "
                f"err={err:.4f} tool={tool_pos[0].tolist()} target={self.target_tool_pos[0].tolist()} "
                f"grip={self.gripper_target:+.4f} dq_max={float(self.last_dq_command.abs().max()):.4f} "
                f"sigma_min={self.last_min_singular:.4f} joint_corr={self.joint_correction_enabled} "
                f"wrist_ori={self.wrist_orientation_enabled} planar={self.planar_only} lock_wrist={self.lock_wrist} "
                f"elbow_assist={self.elbow_assist_enabled} torso_safe={self.torso_safe} direct={self.direct_planar}"
                f" lost_age={self.last_lost_age_s:.2f}s"
                f"{direct_debug}"
            )
            if self.debug_pose and not has_camera_target:
                reason, counts = self.source.get_debug_snapshot()
                print(f"[R1-A7 CAMERA POSE] Gemini no-target reason={reason} counts={counts}")

        return action
