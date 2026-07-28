#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replace the robot in the current A7 cylinder task with R1-A7 + Dex1.

Target model:
    /home/robot/IsaacLab/bolt_nut_assembly/g1_dex1_r1.usd

Target repository:
    /home/robot/unitree_sim_isaaclab

The script:
1. checks the USD and its referenced configuration layer;
2. writes robots/a7_dex1.py;
3. patches the existing cylinder task to use A7_DEX1_CFG;
4. updates the pose_grasp provider to prefer the new grasp-center body;
5. creates timestamped backups;
6. performs Python syntax checks.

It does not modify the USD itself.
"""

from __future__ import annotations

import datetime
import py_compile
import re
import shutil
from pathlib import Path


REPO = Path("/home/robot/unitree_sim_isaaclab")
MODEL_USD = Path(
    "/home/robot/IsaacLab/bolt_nut_assembly/g1_dex1_r1.usd"
)
MODEL_DEP = MODEL_USD.parent / "configuration" / "g1_dex1_r1_sensor.usd"

ROBOT_CFG = REPO / "robots" / "a7_dex1.py"
TASK_CFG = (
    REPO
    / "tasks"
    / "a7_tasks"
    / "pick_place_cylinder_a7"
    / "pickplace_cylinder_a7_joint_env_cfg.py"
)
TASK_REGISTER = (
    REPO
    / "tasks"
    / "a7_tasks"
    / "pick_place_cylinder_a7"
    / "__init__.py"
)
PROVIDER = REPO / "action_provider" / "action_provider_pose_grasp.py"
SIM_MAIN = REPO / "sim_main.py"


ROBOT_CFG_TEXT = r'''
# -*- coding: utf-8 -*-
"""Isaac Lab configuration for the final R1-A7 + dual Dex1 USD."""

from copy import deepcopy

from isaaclab.actuators import ImplicitActuatorCfg

from robots.a7 import A7_CFG


R1_A7_DEX1_USD_PATH = (
    "/home/robot/IsaacLab/bolt_nut_assembly/g1_dex1_r1.usd"
)

# Preserve robots/a7.py as the reusable 17-DoF base configuration.
A7_DEX1_CFG = deepcopy(A7_CFG)
A7_DEX1_CFG.spawn.usd_path = R1_A7_DEX1_USD_PATH

# The supplied gripper URDF uses q=-0.020 m as the open state.
# Increasing q moves the fingers toward the closing direction.
A7_DEX1_CFG.init_state.joint_pos.update(
    {
        "left_dex1_Joint1_1": -0.020,
        "right_dex1_Joint1_1": -0.020,
    }
)

# Only the master joint on each gripper is driven. The second slider must
# remain a mimic/dependent joint in the imported USD.
A7_DEX1_CFG.actuators["dex1_grippers"] = ImplicitActuatorCfg(
    joint_names_expr=[
        "left_dex1_Joint1_1",
        "right_dex1_Joint1_1",
    ],
    effort_limit=20.0,
    velocity_limit=0.2,
    stiffness=220.0,
    damping=12.0,
    armature=0.001,
    friction=0.05,
)
'''


PROVIDER_TEXT = r'''
# -*- coding: utf-8 -*-
"""R1-A7 + Dex1 left-arm differential-IK validation provider.

This provider is intentionally a safe first-stage test:
    WAIT -> move the left grasp frame up 3 cm -> HOLD -> RETURN -> DONE

The grippers remain open. A later controller can extend the state machine to
PREGRASP -> APPROACH -> CLOSE -> LIFT after the model replacement is verified.
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
    ARM_JOINT_NAMES = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]

    # Fixed links may be merged depending on how the USD was imported.
    # Prefer the grasp center and fall back without preventing startup.
    EE_BODY_CANDIDATES = (
        "left_grasp_center",
        "left_dex1_base_link",
        "left_wrist_yaw_link",
    )

    LEFT_GRIPPER_MASTER = "left_dex1_Joint1_1"
    RIGHT_GRIPPER_MASTER = "right_dex1_Joint1_1"
    GRIPPER_OPEN = -0.020

    def __init__(self, env, args_cli):
        super().__init__("R1A7Dex1PoseGraspIKTest")

        self.env = env
        self.args_cli = args_cli
        self.robot = env.scene["robot"]
        self.object = env.scene["object"]
        self.device = env.device
        self.num_envs = env.num_envs

        if self.num_envs != 1:
            raise RuntimeError(
                "Current pose_grasp validation requires num_envs=1; "
                f"received {self.num_envs}."
            )

        self.joint_names = list(self.robot.joint_names)
        self.body_names = list(self.robot.body_names)

        missing_arm = [
            name for name in self.ARM_JOINT_NAMES
            if name not in self.joint_names
        ]
        if missing_arm:
            raise RuntimeError(
                "R1-A7 arm joints are missing from the loaded USD: "
                f"{missing_arm}\nLoaded joints: {self.joint_names}"
            )

        self.arm_ids = [
            self.joint_names.index(name)
            for name in self.ARM_JOINT_NAMES
        ]

        self.ee_body_name = next(
            (
                name for name in self.EE_BODY_CANDIDATES
                if name in self.body_names
            ),
            None,
        )
        if self.ee_body_name is None:
            raise RuntimeError(
                "No usable left end-effector body was found. Tried: "
                f"{self.EE_BODY_CANDIDATES}\n"
                f"Loaded bodies: {self.body_names}"
            )

        self.ee_body_id = self.body_names.index(self.ee_body_name)
        self.ee_jacobian_id = (
            self.ee_body_id - 1
            if self.robot.is_fixed_base
            else self.ee_body_id
        )

        self.left_gripper_id = self._optional_joint_id(
            self.LEFT_GRIPPER_MASTER
        )
        self.right_gripper_id = self._optional_joint_id(
            self.RIGHT_GRIPPER_MASTER
        )

        self.initial_joint_pos = self.robot.data.joint_pos.clone()
        self.default_joint_pos = (
            self.robot.data.default_joint_pos.clone()
        )

        # Keep the known master joints open during the initial IK test.
        if self.left_gripper_id is not None:
            self.initial_joint_pos[:, self.left_gripper_id] = (
                self.GRIPPER_OPEN
            )
        if self.right_gripper_id is not None:
            self.initial_joint_pos[:, self.right_gripper_id] = (
                self.GRIPPER_OPEN
            )

        cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        )
        self.ik = DifferentialIKController(
            cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        pos_b, quat_b = self._ee_pose_b()
        self.initial_pose = torch.cat(
            (pos_b.clone(), quat_b.clone()),
            dim=-1,
        )
        self.target_pose = self.initial_pose.clone()
        self.up_pose = self.initial_pose.clone()
        self.up_pose[:, 2] += 0.03

        self.ik.set_command(self.target_pose)
        self.state = "WAIT"
        self.state_time = time.monotonic()
        self.stable = 0
        self.last_print = 0.0

        print("\n" + "=" * 76)
        print("[R1-A7 DEX1] final USD loaded")
        print("[R1-A7 DEX1] joints:", self.joint_names)
        print("[R1-A7 DEX1] bodies:", self.body_names)
        print("[R1-A7 DEX1] left arm ids:", self.arm_ids)
        print(
            "[R1-A7 DEX1] EE:",
            self.ee_body_name,
            "body_id=",
            self.ee_body_id,
            "jacobian_id=",
            self.ee_jacobian_id,
        )
        print(
            "[R1-A7 DEX1] gripper master ids:",
            self.left_gripper_id,
            self.right_gripper_id,
        )
        print("[R1-A7 DEX1] action width:", self.initial_joint_pos.shape[1])
        print("=" * 76 + "\n")

    def _optional_joint_id(self, name: str) -> int | None:
        try:
            return self.joint_names.index(name)
        except ValueError:
            return None

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

    def _ee_pose_b(self):
        root = self._root_pose_w()
        ee = self._body_pose_w()[:, self.ee_body_id, :]
        return subtract_frame_transforms(
            root[:, 0:3],
            root[:, 3:7],
            ee[:, 0:3],
            ee[:, 3:7],
        )

    def _jacobian(self):
        view = (
            self.robot.root_view
            if hasattr(self.robot, "root_view")
            else self.robot.root_physx_view
        )
        all_jacobians = view.get_jacobians()
        return all_jacobians[
            :,
            self.ee_jacobian_id,
            :,
            self.arm_ids,
        ]

    def _set_state(self, state, pose):
        self.state = state
        self.state_time = time.monotonic()
        self.stable = 0
        self.target_pose = pose.clone()
        self.ik.set_command(self.target_pose)
        print(f"[R1-A7 DEX1] >>> {state}")

    def get_action(self, env) -> Optional[torch.Tensor]:
        del env

        ee_pos, ee_quat = self._ee_pose_b()
        jacobian = self._jacobian()

        current = self.robot.data.joint_pos
        current_arm = current[:, self.arm_ids]

        desired_arm = self.ik.compute(
            ee_pos,
            ee_quat,
            jacobian,
            current_arm,
        )

        delta = torch.clamp(
            desired_arm - current_arm,
            min=-0.008,
            max=0.008,
        )
        desired_arm = current_arm + delta

        limits = getattr(
            self.robot.data,
            "soft_joint_pos_limits",
            None,
        )
        if limits is not None:
            arm_limits = limits[:, self.arm_ids, :]
            desired_arm = torch.maximum(
                desired_arm,
                arm_limits[..., 0] + 0.02,
            )
            desired_arm = torch.minimum(
                desired_arm,
                arm_limits[..., 1] - 0.02,
            )

        desired = self.initial_joint_pos.clone()
        desired[:, self.arm_ids] = desired_arm

        if self.left_gripper_id is not None:
            desired[:, self.left_gripper_id] = self.GRIPPER_OPEN
        if self.right_gripper_id is not None:
            desired[:, self.right_gripper_id] = self.GRIPPER_OPEN

        # The task action term uses default joint positions as offsets.
        action = desired - self.default_joint_pos

        error = float(
            torch.linalg.norm(
                self.target_pose[:, 0:3] - ee_pos,
                dim=-1,
            )[0].item()
        )
        elapsed = time.monotonic() - self.state_time
        self.stable = self.stable + 1 if error < 0.008 else 0

        if self.state == "WAIT" and elapsed > 2.0:
            self._set_state("UP", self.up_pose)
        elif self.state == "UP":
            if self.stable > 12 or elapsed > 8.0:
                self._set_state("HOLD", self.up_pose)
        elif self.state == "HOLD" and elapsed > 2.0:
            self._set_state("RETURN", self.initial_pose)
        elif self.state == "RETURN":
            if self.stable > 12 or elapsed > 8.0:
                self._set_state("DONE", self.initial_pose)

        now = time.monotonic()
        if now - self.last_print > 0.5:
            self.last_print = now
            print(
                f"[R1-A7 DEX1] state={self.state:<6} "
                f"pos_err={error:.4f} m"
            )

        return action

    def cleanup(self):
        print("[R1-A7 DEX1] pose provider cleanup")
'''


def backup(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = path.with_name(path.name + f".dex1_{stamp}.bak")
    shutil.copy2(path, destination)
    print(f"[backup] {destination}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
    print(f"[write] {path}")


def verify_model() -> None:
    if not MODEL_USD.is_file():
        raise FileNotFoundError(
            f"Final robot USD was not found: {MODEL_USD}"
        )

    # The top-level USD provided by the user is a wrapper that references this
    # configuration layer. Without it the model will not load completely.
    data = MODEL_USD.read_bytes()
    dependency_token = b"configuration/g1_dex1_r1_sensor.usd"
    if dependency_token in data and not MODEL_DEP.is_file():
        raise FileNotFoundError(
            "The final USD references a missing layer:\n"
            f"  {MODEL_DEP}\n"
            "Keep the complete configuration directory beside the USD."
        )

    print(f"[ok] model: {MODEL_USD}")
    if MODEL_DEP.is_file():
        print(f"[ok] dependency: {MODEL_DEP}")


def patch_task_cfg() -> None:
    if not TASK_CFG.is_file():
        raise FileNotFoundError(
            "The current A7 cylinder task configuration was not found: "
            f"{TASK_CFG}"
        )

    text = TASK_CFG.read_text(encoding="utf-8")

    # Replace either the old base-model import or a previous Dex1 import.
    text = re.sub(
        r"from robots\.a7(?:_dex1)? import (?:A7_CFG|A7_DEX1_CFG)",
        "from robots.a7_dex1 import A7_DEX1_CFG",
        text,
    )

    text = re.sub(
        r"robot:\s*ArticulationCfg\s*=\s*A7_CFG\.replace\(",
        "robot: ArticulationCfg = A7_DEX1_CFG.replace(",
        text,
    )
    text = re.sub(
        r"robot:\s*ArticulationCfg\s*=\s*A7_CFG\b",
        "robot: ArticulationCfg = A7_DEX1_CFG",
        text,
    )

    if "from robots.a7_dex1 import A7_DEX1_CFG" not in text:
        raise RuntimeError(
            "Could not patch the robot import in the cylinder task."
        )
    if "robot: ArticulationCfg = A7_DEX1_CFG" not in text:
        raise RuntimeError(
            "Could not patch the robot scene configuration."
        )

    TASK_CFG.write_text(text, encoding="utf-8")
    print(f"[patch] {TASK_CFG}")


def verify_task_registration() -> None:
    if not TASK_REGISTER.is_file():
        raise FileNotFoundError(TASK_REGISTER)
    text = TASK_REGISTER.read_text(encoding="utf-8")
    if "Isaac-PickPlace-Cylinder-A7-Joint" not in text:
        raise RuntimeError(
            "Task registration does not contain "
            "Isaac-PickPlace-Cylinder-A7-Joint."
        )
    print("[ok] task registration")


def syntax_check() -> None:
    for path in (ROBOT_CFG, TASK_CFG, PROVIDER, SIM_MAIN):
        if not path.is_file():
            raise FileNotFoundError(path)
        py_compile.compile(str(path), doraise=True)
        print(f"[compile ok] {path}")


def main() -> None:
    if not REPO.is_dir():
        raise FileNotFoundError(REPO)

    verify_model()
    verify_task_registration()

    for path in (ROBOT_CFG, TASK_CFG, PROVIDER):
        backup(path)

    write_text(ROBOT_CFG, ROBOT_CFG_TEXT)
    patch_task_cfg()
    write_text(PROVIDER, PROVIDER_TEXT)
    syntax_check()

    print("\n" + "=" * 76)
    print("R1-A7 + Dex1 has replaced the robot in the cylinder task.")
    print(f"USD:  {MODEL_USD}")
    print("Task: Isaac-PickPlace-Cylinder-A7-Joint")
    print("Robot type: a7")
    print("Action source: pose_grasp")
    print("Do not enable Dex3 DDS for this model.")
    print("=" * 76)


if __name__ == "__main__":
    main()
