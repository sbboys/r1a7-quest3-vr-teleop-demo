#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G1-29DoF 右臂差分 IK 验证 Action Provider。

阶段：
WAIT -> 右手掌沿机器人基座 Z 轴上移 5 cm -> HOLD -> 返回初始位姿 -> DONE

本文件只验证：
1. 右臂关节索引；
2. right_hand_palm_link 末端索引；
3. PhysX Jacobian；
4. DifferentialIKController；
5. 全 43 维关节动作映射。

验证通过后，再扩展为：
PREGRASP -> APPROACH -> CLOSE -> LIFT。
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from action_provider.action_base import ActionProvider
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.utils.math import subtract_frame_transforms


class PoseGraspActionProvider(ActionProvider):
    """内部真值位姿 + 差分 IK 的第一阶段验证控制器。"""

    RIGHT_ARM_JOINT_NAMES = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    EE_BODY_NAME = "right_hand_palm_link"

    def __init__(self, env, args_cli):
        super().__init__(name="PoseGraspIKTest")

        self.env = env
        self.args_cli = args_cli
        self.robot = env.scene["robot"]
        self.object = env.scene["object"]
        self.device = env.device
        self.num_envs = env.num_envs

        if self.num_envs != 1:
            raise RuntimeError(
                f"当前 IK 验证仅支持 num_envs=1，实际为 {self.num_envs}"
            )

        self.joint_names = list(self.robot.joint_names)
        self.body_names = list(self.robot.body_names)

        self.right_arm_joint_ids = [
            self.joint_names.index(name)
            for name in self.RIGHT_ARM_JOINT_NAMES
        ]
        self.ee_body_id = self.body_names.index(self.EE_BODY_NAME)

        if self.robot.is_fixed_base:
            self.ee_jacobian_id = self.ee_body_id - 1
        else:
            self.ee_jacobian_id = self.ee_body_id

        self.action_dim = int(self.env.action_manager.total_action_dim)
        self.num_joints = int(self.robot.num_joints)

        if self.action_dim != self.num_joints:
            raise RuntimeError(
                "当前任务不是全关节位置动作："
                f"action_dim={self.action_dim}, num_joints={self.num_joints}"
            )

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        )
        self.ik_controller = DifferentialIKController(
            ik_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        self.initial_joint_pos = self.robot.data.joint_pos.clone()
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()

        ee_pos_b, ee_quat_b = self._get_ee_pose_b()
        self.initial_pose_b = torch.cat(
            (ee_pos_b.clone(), ee_quat_b.clone()),
            dim=-1,
        )
        self.up_pose_b = self.initial_pose_b.clone()
        self.up_pose_b[:, 2] += 0.05

        self.target_pose_b = self.initial_pose_b.clone()
        self.ik_controller.reset()
        self.ik_controller.set_command(self.target_pose_b)

        self.state = "WAIT"
        self.state_start_time = time.monotonic()
        self.stable_steps = 0
        self.last_print_time = 0.0
        self.done_reported = False

        print("\n" + "=" * 78)
        print("[POSE_GRASP] IK test initialized")
        print(f"[POSE_GRASP] right arm joint ids: {self.right_arm_joint_ids}")
        print(
            f"[POSE_GRASP] EE body: {self.EE_BODY_NAME}, "
            f"body_id={self.ee_body_id}, "
            f"jacobian_id={self.ee_jacobian_id}"
        )
        print(f"[POSE_GRASP] action_dim={self.action_dim}")
        print(
            "[POSE_GRASP] initial EE pose in robot base:",
            self.initial_pose_b[0].detach().cpu().numpy(),
        )
        print(
            "[POSE_GRASP] target: base-frame Z +0.05 m, "
            "orientation unchanged"
        )
        print("=" * 78 + "\n")

    def _get_root_pose_w(self) -> torch.Tensor:
        data = self.robot.data
        if hasattr(data, "root_link_pose_w"):
            return data.root_link_pose_w
        if hasattr(data, "root_pose_w"):
            return data.root_pose_w
        return data.root_state_w[:, 0:7]

    def _get_body_pose_w(self) -> torch.Tensor:
        data = self.robot.data
        if hasattr(data, "body_link_pose_w"):
            return data.body_link_pose_w
        if hasattr(data, "body_pose_w"):
            return data.body_pose_w
        return data.body_state_w[:, :, 0:7]

    def _get_ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pose_w = self._get_root_pose_w()
        body_pose_w = self._get_body_pose_w()
        ee_pose_w = body_pose_w[:, self.ee_body_id, :]

        return subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

    def _get_jacobian(self) -> torch.Tensor:
        if hasattr(self.robot, "root_view"):
            root_view = self.robot.root_view
        elif hasattr(self.robot, "root_physx_view"):
            root_view = self.robot.root_physx_view
        else:
            raise AttributeError(
                "robot 没有 root_view 或 root_physx_view，无法读取 Jacobian"
            )

        jacobians = root_view.get_jacobians()
        return jacobians[
            :,
            self.ee_jacobian_id,
            :,
            self.right_arm_joint_ids,
        ]

    def _set_state(
        self,
        state: str,
        target_pose_b: torch.Tensor,
    ) -> None:
        self.state = state
        self.state_start_time = time.monotonic()
        self.stable_steps = 0
        self.target_pose_b = target_pose_b.clone()
        self.ik_controller.reset()
        self.ik_controller.set_command(self.target_pose_b)
        print(f"\n[POSE_GRASP] >>> state = {self.state}")

    def _clamp_arm_target(
        self,
        current_arm_q: torch.Tensor,
        desired_arm_q: torch.Tensor,
    ) -> torch.Tensor:
        # 限制每一仿真步的最大关节变化，防止 IK 第一次计算时跳变。
        max_step = 0.025
        delta = torch.clamp(
            desired_arm_q - current_arm_q,
            min=-max_step,
            max=max_step,
        )
        desired_arm_q = current_arm_q + delta

        limits = getattr(
            self.robot.data,
            "soft_joint_pos_limits",
            None,
        )
        if limits is not None:
            arm_limits = limits[:, self.right_arm_joint_ids, :]
            margin = 0.02
            desired_arm_q = torch.maximum(
                desired_arm_q,
                arm_limits[..., 0] + margin,
            )
            desired_arm_q = torch.minimum(
                desired_arm_q,
                arm_limits[..., 1] - margin,
            )

        return desired_arm_q

    def _update_state_machine(self, pos_error: float) -> None:
        elapsed = time.monotonic() - self.state_start_time

        if pos_error < 0.010:
            self.stable_steps += 1
        else:
            self.stable_steps = 0

        if self.state == "WAIT" and elapsed >= 2.0:
            self._set_state("UP", self.up_pose_b)

        elif self.state == "UP":
            if self.stable_steps >= 12:
                self._set_state("HOLD", self.up_pose_b)
            elif elapsed >= 8.0:
                print(
                    "[POSE_GRASP][WARN] UP 超时，仍进入 HOLD；"
                    f"最后位置误差={pos_error:.4f} m"
                )
                self._set_state("HOLD", self.up_pose_b)

        elif self.state == "HOLD" and elapsed >= 2.0:
            self._set_state("RETURN", self.initial_pose_b)

        elif self.state == "RETURN":
            if self.stable_steps >= 12:
                self._set_state("DONE", self.initial_pose_b)
            elif elapsed >= 8.0:
                print(
                    "[POSE_GRASP][WARN] RETURN 超时；"
                    f"最后位置误差={pos_error:.4f} m"
                )
                self._set_state("DONE", self.initial_pose_b)

    def get_action(self, env) -> Optional[torch.Tensor]:
        ee_pos_b, ee_quat_b = self._get_ee_pose_b()
        jacobian = self._get_jacobian()

        current_joint_pos = self.robot.data.joint_pos
        current_arm_q = current_joint_pos[
            :,
            self.right_arm_joint_ids,
        ]

        desired_arm_q = self.ik_controller.compute(
            ee_pos_b,
            ee_quat_b,
            jacobian,
            current_arm_q,
        )
        desired_arm_q = self._clamp_arm_target(
            current_arm_q,
            desired_arm_q,
        )

        if not torch.isfinite(desired_arm_q).all():
            raise RuntimeError("IK 输出包含 NaN 或 Inf")

        # 其余身体、左臂和双手保持脚本启动时的位置。
        desired_joint_pos = self.initial_joint_pos.clone()
        desired_joint_pos[
            :,
            self.right_arm_joint_ids,
        ] = desired_arm_q

        # 当前任务 JointPositionActionCfg(use_default_offset=True)，
        # 所以动作值是“绝对目标关节角 - 默认关节角”。
        action = desired_joint_pos - self.default_joint_pos

        pos_error_tensor = torch.linalg.norm(
            self.target_pose_b[:, 0:3] - ee_pos_b,
            dim=-1,
        )
        pos_error = float(pos_error_tensor[0].item())

        now = time.monotonic()
        if now - self.last_print_time >= 0.5:
            self.last_print_time = now
            object_pose = (
                self.object.data.root_link_pose_w
                if hasattr(self.object.data, "root_link_pose_w")
                else self.object.data.root_pose_w
            )
            print(
                f"[POSE_GRASP] state={self.state:<6} "
                f"pos_err={pos_error:.4f} m "
                f"ee_b=({ee_pos_b[0,0]:+.3f},"
                f"{ee_pos_b[0,1]:+.3f},"
                f"{ee_pos_b[0,2]:+.3f}) "
                f"obj_w=({object_pose[0,0]:+.3f},"
                f"{object_pose[0,1]:+.3f},"
                f"{object_pose[0,2]:+.3f})"
            )

        self._update_state_machine(pos_error)

        if self.state == "DONE" and not self.done_reported:
            self.done_reported = True
            print(
                "\n[POSE_GRASP] IK_TEST_DONE："
                "右手掌上移与返回流程已结束。"
            )

        return action

    def cleanup(self):
        print("[POSE_GRASP] cleanup")
