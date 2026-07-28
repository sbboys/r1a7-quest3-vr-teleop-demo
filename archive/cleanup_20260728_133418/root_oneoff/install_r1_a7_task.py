
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install R1-A7 task into unitree_sim_isaaclab."""

from __future__ import annotations

import datetime
import py_compile
import re
import shutil
from pathlib import Path

REPO = Path("/home/robot/unitree_sim_isaaclab")
MODEL_DIR = Path(
    "/home/robot/IsaacLab/bolt_nut_assembly/R1_A7_official"
)
A7_USD = MODEL_DIR / "A7.usd"
A7_SENSOR_USD = MODEL_DIR / "configuration" / "A7_sensor.usd"

ROBOT_CFG = REPO / "robots" / "a7.py"
TASKS_INIT = REPO / "tasks" / "__init__.py"
A7_TASK_ROOT = REPO / "tasks" / "a7_tasks"
TASK_DIR = A7_TASK_ROOT / "pick_place_cylinder_a7"
TASK_CFG = TASK_DIR / "pickplace_cylinder_a7_joint_env_cfg.py"
TASK_REGISTER = TASK_DIR / "__init__.py"
A7_TASKS_INIT = A7_TASK_ROOT / "__init__.py"
SIM_MAIN = REPO / "sim_main.py"
PROVIDER_FACTORY = (
    REPO / "action_provider" / "create_action_provider.py"
)
PROVIDER = (
    REPO / "action_provider" / "action_provider_pose_grasp.py"
)

ROBOT_CFG_TEXT = '\n# -*- coding: utf-8 -*-\n"""R1-A7 articulation configuration for Isaac Lab."""\n\nimport isaaclab.sim as sim_utils\nfrom isaaclab.actuators import ImplicitActuatorCfg\nfrom isaaclab.assets import ArticulationCfg\n\nA7_USD_PATH = (\n    "/home/robot/IsaacLab/bolt_nut_assembly/"\n    "R1_A7_official/A7.usd"\n)\n\nA7_CFG = ArticulationCfg(\n    prim_path="/World/envs/env_.*/Robot",\n    spawn=sim_utils.UsdFileCfg(\n        usd_path=A7_USD_PATH,\n        activate_contact_sensors=True,\n        rigid_props=sim_utils.RigidBodyPropertiesCfg(\n            disable_gravity=False,\n            retain_accelerations=True,\n            linear_damping=0.0,\n            angular_damping=0.0,\n            max_linear_velocity=100.0,\n            max_angular_velocity=100.0,\n            max_depenetration_velocity=1.0,\n        ),\n        articulation_props=sim_utils.ArticulationRootPropertiesCfg(\n            enabled_self_collisions=False,\n            solver_position_iteration_count=8,\n            solver_velocity_iteration_count=4,\n            fix_root_link=True,\n        ),\n    ),\n    init_state=ArticulationCfg.InitialStateCfg(\n        pos=(-0.15, 0.0, 0.76),\n        rot=(0.7071, 0.0, 0.0, 0.7071),\n        joint_pos={\n            "waist_yaw_joint": 0.0,\n            "left_shoulder_pitch_joint": 0.0,\n            "left_shoulder_roll_joint": 0.25,\n            "left_shoulder_yaw_joint": 0.0,\n            "left_elbow_joint": 0.30,\n            "left_wrist_roll_joint": 0.0,\n            "left_wrist_pitch_joint": 0.0,\n            "left_wrist_yaw_joint": 0.0,\n            "right_shoulder_pitch_joint": 0.0,\n            "right_shoulder_roll_joint": -0.25,\n            "right_shoulder_yaw_joint": 0.0,\n            "right_elbow_joint": 0.30,\n            "right_wrist_roll_joint": 0.0,\n            "right_wrist_pitch_joint": 0.0,\n            "right_wrist_yaw_joint": 0.0,\n            "head_pitch_joint": 0.0,\n            "head_yaw_joint": 0.0,\n        },\n        joint_vel={".*": 0.0},\n    ),\n    soft_joint_pos_limit_factor=0.95,\n    actuators={\n        "waist": ImplicitActuatorCfg(\n            joint_names_expr=["waist_yaw_joint"],\n            effort_limit=60.0,\n            velocity_limit=18.7,\n            stiffness=100.0,\n            damping=5.0,\n            armature=0.01,\n        ),\n        "shoulders": ImplicitActuatorCfg(\n            joint_names_expr=[".*_shoulder_.*_joint"],\n            effort_limit=60.0,\n            velocity_limit=18.7,\n            stiffness=90.0,\n            damping=5.0,\n            armature=0.01,\n        ),\n        "elbows": ImplicitActuatorCfg(\n            joint_names_expr=[".*_elbow_joint"],\n            effort_limit=33.0,\n            velocity_limit=52.4,\n            stiffness=70.0,\n            damping=4.0,\n            armature=0.01,\n        ),\n        "wrist_roll": ImplicitActuatorCfg(\n            joint_names_expr=[".*_wrist_roll_joint"],\n            effort_limit=33.0,\n            velocity_limit=52.4,\n            stiffness=45.0,\n            damping=2.5,\n            armature=0.005,\n        ),\n        "wrist_pitch_yaw": ImplicitActuatorCfg(\n            joint_names_expr=[\n                ".*_wrist_pitch_joint",\n                ".*_wrist_yaw_joint",\n            ],\n            effort_limit=10.0,\n            velocity_limit=37.7,\n            stiffness=35.0,\n            damping=2.0,\n            armature=0.005,\n        ),\n        "head_pitch": ImplicitActuatorCfg(\n            joint_names_expr=["head_pitch_joint"],\n            effort_limit=33.0,\n            velocity_limit=52.4,\n            stiffness=30.0,\n            damping=2.0,\n            armature=0.005,\n        ),\n        "head_yaw": ImplicitActuatorCfg(\n            joint_names_expr=["head_yaw_joint"],\n            effort_limit=0.86,\n            velocity_limit=11.0,\n            stiffness=10.0,\n            damping=1.0,\n            armature=0.002,\n        ),\n    },\n)\n'
A7_TASKS_INIT_TEXT = '\n"""R1-A7 tasks."""\nfrom . import pick_place_cylinder_a7\n\n__all__ = ["pick_place_cylinder_a7"]\n'
TASK_REGISTER_TEXT = '\nimport gymnasium as gym\n\nfrom . import pickplace_cylinder_a7_joint_env_cfg\n\ngym.register(\n    id="Isaac-PickPlace-Cylinder-A7-Joint",\n    entry_point="isaaclab.envs:ManagerBasedRLEnv",\n    kwargs={\n        "env_cfg_entry_point":\n            pickplace_cylinder_a7_joint_env_cfg.\n            PickPlaceCylinderA7JointEnvCfg,\n    },\n    disable_env_checker=True,\n)\n'
TASK_CFG_TEXT = '\n# -*- coding: utf-8 -*-\n"""R1-A7 cylinder scene configuration."""\n\nimport torch\nimport isaaclab.envs.mdp as base_mdp\n\nfrom isaaclab.assets import ArticulationCfg\nfrom isaaclab.envs import ManagerBasedRLEnvCfg\nfrom isaaclab.managers import EventTermCfg\nfrom isaaclab.managers import ObservationGroupCfg as ObsGroup\nfrom isaaclab.managers import ObservationTermCfg as ObsTerm\nfrom isaaclab.managers import RewardTermCfg as RewTerm\nfrom isaaclab.managers import SceneEntityCfg\nfrom isaaclab.managers import TerminationTermCfg as DoneTerm\nfrom isaaclab.utils import configclass\n\nfrom robots.a7 import A7_CFG\nfrom tasks.common_config.camera_configs import CameraBaseCfg\nfrom tasks.common_event.event_manager import (\n    SimpleEvent,\n    SimpleEventManager,\n)\nfrom tasks.common_observations.camera_state import get_camera_image\nfrom tasks.common_rewards.base_reward_pickplace_cylindercfg import (\n    compute_reward,\n)\nfrom tasks.common_scene.base_scene_pickplace_cylindercfg import (\n    TableCylinderSceneCfg,\n)\nfrom tasks.common_termination.base_termination_pick_place_cylinder import (\n    reset_object_estimate,\n)\n\n\n@configclass\nclass A7TableCylinderSceneCfg(TableCylinderSceneCfg):\n    robot: ArticulationCfg = A7_CFG\n\n    front_camera = CameraBaseCfg.get_camera_config(\n        prim_path=(\n            "/World/envs/env_.*/Robot/"\n            "head_yaw_link/front_camera"\n        ),\n        update_period=0.04,\n        height=480,\n        width=640,\n        focal_length=12.0,\n        focus_distance=400.0,\n        horizontal_aperture=20.0,\n        clipping_range=(0.05, 100.0),\n        pos_offset=(0.08, 0.0, 0.02),\n        rot_offset=(0.5, -0.5, 0.5, -0.5),\n    )\n\n\n@configclass\nclass ActionsCfg:\n    joint_pos = base_mdp.JointPositionActionCfg(\n        asset_name="robot",\n        joint_names=[".*"],\n        scale=1.0,\n        use_default_offset=True,\n    )\n\n\n@configclass\nclass ObservationsCfg:\n    @configclass\n    class PolicyCfg(ObsGroup):\n        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)\n        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)\n        camera_image = ObsTerm(func=get_camera_image)\n\n        def __post_init__(self):\n            self.enable_corruption = False\n            self.concatenate_terms = False\n\n    policy: PolicyCfg = PolicyCfg()\n\n\n@configclass\nclass TerminationsCfg:\n    success = DoneTerm(func=reset_object_estimate)\n\n\n@configclass\nclass RewardsCfg:\n    reward = RewTerm(func=compute_reward, weight=1.0)\n\n\n@configclass\nclass EventCfg:\n    reset_object = EventTermCfg(\n        func=base_mdp.reset_root_state_uniform,\n        mode="reset",\n        params={\n            "pose_range": {\n                "x": [-0.05, 0.05],\n                "y": [-0.05, 0.05],\n            },\n            "velocity_range": {},\n            "asset_cfg": SceneEntityCfg("object"),\n        },\n    )\n\n\n@configclass\nclass PickPlaceCylinderA7JointEnvCfg(ManagerBasedRLEnvCfg):\n    scene: A7TableCylinderSceneCfg = A7TableCylinderSceneCfg(\n        num_envs=1,\n        env_spacing=2.5,\n        replicate_physics=True,\n    )\n    observations: ObservationsCfg = ObservationsCfg()\n    actions: ActionsCfg = ActionsCfg()\n    terminations: TerminationsCfg = TerminationsCfg()\n    events: EventCfg = EventCfg()\n    commands = None\n    rewards: RewardsCfg = RewardsCfg()\n    curriculum = None\n\n    def __post_init__(self):\n        self.decimation = 2\n        self.episode_length_s = 20.0\n        self.sim.dt = 0.005\n        self.sim.render_interval = self.decimation\n        self.sim.physx.bounce_threshold_velocity = 0.01\n        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = (\n            1024 * 1024 * 4\n        )\n        self.sim.physx.gpu_total_aggregate_pairs_capacity = (\n            16 * 1024\n        )\n        self.sim.physx.friction_correlation_distance = 0.00625\n\n        self.event_manager = SimpleEventManager()\n        self.event_manager.register(\n            "reset_object_self",\n            SimpleEvent(\n                func=lambda env: base_mdp.reset_root_state_uniform(\n                    env,\n                    torch.arange(\n                        env.num_envs,\n                        device=env.device,\n                    ),\n                    pose_range={\n                        "x": [-0.05, 0.05],\n                        "y": [0.0, 0.05],\n                    },\n                    velocity_range={},\n                    asset_cfg=SceneEntityCfg("object"),\n                )\n            ),\n        )\n        self.event_manager.register(\n            "reset_all_self",\n            SimpleEvent(\n                func=lambda env: base_mdp.reset_scene_to_default(\n                    env,\n                    torch.arange(\n                        env.num_envs,\n                        device=env.device,\n                    ),\n                )\n            ),\n        )\n'
PROVIDER_TEXT = '\n# -*- coding: utf-8 -*-\n"""R1-A7 left-arm differential IK validation provider."""\n\nfrom __future__ import annotations\n\nimport time\nfrom typing import Optional\n\nimport torch\n\nfrom action_provider.action_base import ActionProvider\nfrom isaaclab.controllers import (\n    DifferentialIKController,\n    DifferentialIKControllerCfg,\n)\nfrom isaaclab.utils.math import subtract_frame_transforms\n\n\nclass PoseGraspActionProvider(ActionProvider):\n    ARM_JOINT_NAMES = [\n        "left_shoulder_pitch_joint",\n        "left_shoulder_roll_joint",\n        "left_shoulder_yaw_joint",\n        "left_elbow_joint",\n        "left_wrist_roll_joint",\n        "left_wrist_pitch_joint",\n        "left_wrist_yaw_joint",\n    ]\n    EE_BODY_NAME = "left_wrist_yaw_link"\n\n    def __init__(self, env, args_cli):\n        super().__init__("A7PoseGraspIKTest")\n        self.env = env\n        self.args_cli = args_cli\n        self.robot = env.scene["robot"]\n        self.object = env.scene["object"]\n\n        names = list(self.robot.joint_names)\n        bodies = list(self.robot.body_names)\n        self.arm_ids = [\n            names.index(name)\n            for name in self.ARM_JOINT_NAMES\n        ]\n        self.ee_body_id = bodies.index(self.EE_BODY_NAME)\n        self.ee_jacobian_id = (\n            self.ee_body_id - 1\n            if self.robot.is_fixed_base\n            else self.ee_body_id\n        )\n\n        self.initial_joint_pos = self.robot.data.joint_pos.clone()\n        self.default_joint_pos = (\n            self.robot.data.default_joint_pos.clone()\n        )\n\n        cfg = DifferentialIKControllerCfg(\n            command_type="pose",\n            use_relative_mode=False,\n            ik_method="dls",\n            ik_params={"lambda_val": 0.05},\n        )\n        self.ik = DifferentialIKController(\n            cfg,\n            num_envs=env.num_envs,\n            device=env.device,\n        )\n\n        pos_b, quat_b = self._ee_pose_b()\n        self.initial_pose = torch.cat(\n            (pos_b.clone(), quat_b.clone()),\n            dim=-1,\n        )\n        self.target_pose = self.initial_pose.clone()\n        self.up_pose = self.initial_pose.clone()\n        self.up_pose[:, 2] += 0.03\n\n        self.ik.set_command(self.target_pose)\n        self.state = "WAIT"\n        self.state_time = time.monotonic()\n        self.stable = 0\n        self.last_print = 0.0\n\n        print("\\n[A7] model loaded")\n        print("[A7] joints:", names)\n        print("[A7] bodies:", bodies)\n        print("[A7] arm ids:", self.arm_ids)\n        print(\n            "[A7] EE:",\n            self.EE_BODY_NAME,\n            "body_id=",\n            self.ee_body_id,\n            "jacobian_id=",\n            self.ee_jacobian_id,\n        )\n\n    def _root_pose_w(self):\n        data = self.robot.data\n        if hasattr(data, "root_link_pose_w"):\n            return data.root_link_pose_w\n        if hasattr(data, "root_pose_w"):\n            return data.root_pose_w\n        return data.root_state_w[:, 0:7]\n\n    def _body_pose_w(self):\n        data = self.robot.data\n        if hasattr(data, "body_link_pose_w"):\n            return data.body_link_pose_w\n        if hasattr(data, "body_pose_w"):\n            return data.body_pose_w\n        return data.body_state_w[:, :, 0:7]\n\n    def _ee_pose_b(self):\n        root = self._root_pose_w()\n        ee = self._body_pose_w()[:, self.ee_body_id, :]\n        return subtract_frame_transforms(\n            root[:, 0:3],\n            root[:, 3:7],\n            ee[:, 0:3],\n            ee[:, 3:7],\n        )\n\n    def _jacobian(self):\n        view = (\n            self.robot.root_view\n            if hasattr(self.robot, "root_view")\n            else self.robot.root_physx_view\n        )\n        return view.get_jacobians()[\n            :,\n            self.ee_jacobian_id,\n            :,\n            self.arm_ids,\n        ]\n\n    def _set_state(self, state, pose):\n        self.state = state\n        self.state_time = time.monotonic()\n        self.stable = 0\n        self.target_pose = pose.clone()\n        self.ik.set_command(self.target_pose)\n        print(f"[A7] >>> {state}")\n\n    def get_action(self, env) -> Optional[torch.Tensor]:\n        ee_pos, ee_quat = self._ee_pose_b()\n        jacobian = self._jacobian()\n\n        current = self.robot.data.joint_pos\n        current_arm = current[:, self.arm_ids]\n\n        desired_arm = self.ik.compute(\n            ee_pos,\n            ee_quat,\n            jacobian,\n            current_arm,\n        )\n\n        delta = torch.clamp(\n            desired_arm - current_arm,\n            min=-0.008,\n            max=0.008,\n        )\n        desired_arm = current_arm + delta\n\n        limits = getattr(\n            self.robot.data,\n            "soft_joint_pos_limits",\n            None,\n        )\n        if limits is not None:\n            arm_limits = limits[:, self.arm_ids, :]\n            desired_arm = torch.maximum(\n                desired_arm,\n                arm_limits[..., 0] + 0.02,\n            )\n            desired_arm = torch.minimum(\n                desired_arm,\n                arm_limits[..., 1] - 0.02,\n            )\n\n        desired = self.initial_joint_pos.clone()\n        desired[:, self.arm_ids] = desired_arm\n        action = desired - self.default_joint_pos\n\n        error = float(\n            torch.linalg.norm(\n                self.target_pose[:, 0:3] - ee_pos,\n                dim=-1,\n            )[0].item()\n        )\n        elapsed = time.monotonic() - self.state_time\n        self.stable = self.stable + 1 if error < 0.008 else 0\n\n        if self.state == "WAIT" and elapsed > 2.0:\n            self._set_state("UP", self.up_pose)\n        elif self.state == "UP":\n            if self.stable > 12 or elapsed > 8.0:\n                self._set_state("HOLD", self.up_pose)\n        elif self.state == "HOLD" and elapsed > 2.0:\n            self._set_state("RETURN", self.initial_pose)\n        elif self.state == "RETURN":\n            if self.stable > 12 or elapsed > 8.0:\n                self._set_state("DONE", self.initial_pose)\n\n        now = time.monotonic()\n        if now - self.last_print > 0.5:\n            self.last_print = now\n            print(\n                f"[A7] state={self.state:<6} "\n                f"pos_err={error:.4f} m"\n            )\n\n        return action\n\n    def cleanup(self):\n        print("[A7] pose provider cleanup")\n'


def backup(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(path.name + f".a7_{stamp}.bak")
    shutil.copy2(path, dst)
    print(f"[backup] {dst}")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
    print(f"[write] {path}")


def patch_tasks_init() -> None:
    text = TASKS_INIT.read_text(encoding="utf-8")
    line = "from . import a7_tasks"
    if line not in text:
        TASKS_INIT.write_text(
            text.rstrip() + "\n" + line + "\n",
            encoding="utf-8",
        )
        print("[patch] tasks/__init__.py")


def patch_sim_main() -> None:
    text = SIM_MAIN.read_text(encoding="utf-8")
    if '"pose_grasp"' in text:
        return
    pattern = re.compile(
        r'(parser\.add_argument\(\s*"--action_source".*?'
        r'choices\s*=\s*\[)(.*?)(\])',
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(
            "Cannot locate --action_source choices in sim_main.py"
        )
    body = match.group(2).rstrip()
    comma = "" if body.endswith(",") else ","
    replacement = (
        match.group(1)
        + body
        + comma
        + ' "pose_grasp"'
        + match.group(3)
    )
    SIM_MAIN.write_text(
        text[:match.start()] + replacement + text[match.end():],
        encoding="utf-8",
    )
    print("[patch] sim_main.py")


def patch_factory() -> None:
    text = PROVIDER_FACTORY.read_text(encoding="utf-8")
    import_line = (
        "from action_provider.action_provider_pose_grasp "
        "import PoseGraspActionProvider\n"
    )
    if import_line not in text:
        text = import_line + text

    if 'args.action_source == "pose_grasp"' not in text:
        match = re.search(
            r"(?P<indent>[ \t]*)else:\s*\n"
            r"(?P=indent)[ \t]+print\(f?[\"']unknown action source:",
            text,
        )
        if not match:
            raise RuntimeError(
                "Cannot locate final else in create_action_provider.py"
            )
        indent = match.group("indent")
        branch = (
            f'{indent}elif args.action_source == "pose_grasp":\n'
            f"{indent}    return PoseGraspActionProvider(\n"
            f"{indent}        env=env,\n"
            f"{indent}        args_cli=args,\n"
            f"{indent}    )\n"
        )
        text = text[:match.start()] + branch + text[match.start():]

    PROVIDER_FACTORY.write_text(text, encoding="utf-8")
    print("[patch] create_action_provider.py")


def main() -> None:
    if not REPO.is_dir():
        raise FileNotFoundError(REPO)
    if not A7_USD.is_file():
        raise FileNotFoundError(A7_USD)
    if not A7_SENSOR_USD.is_file():
        raise FileNotFoundError(
            "A7.usd references a missing dependency: "
            f"{A7_SENSOR_USD}. Keep the complete model folder."
        )

    for path in (
        TASKS_INIT,
        SIM_MAIN,
        PROVIDER_FACTORY,
        PROVIDER,
        ROBOT_CFG,
        TASK_CFG,
        TASK_REGISTER,
        A7_TASKS_INIT,
    ):
        backup(path)

    write(ROBOT_CFG, ROBOT_CFG_TEXT)
    write(A7_TASKS_INIT, A7_TASKS_INIT_TEXT)
    write(TASK_REGISTER, TASK_REGISTER_TEXT)
    write(TASK_CFG, TASK_CFG_TEXT)
    write(PROVIDER, PROVIDER_TEXT)

    patch_tasks_init()
    patch_sim_main()
    patch_factory()

    for path in (
        ROBOT_CFG,
        TASKS_INIT,
        A7_TASKS_INIT,
        TASK_REGISTER,
        TASK_CFG,
        SIM_MAIN,
        PROVIDER_FACTORY,
        PROVIDER,
    ):
        py_compile.compile(str(path), doraise=True)

    print("\nR1-A7 task installation complete.")
    print("Task: Isaac-PickPlace-Cylinder-A7-Joint")
    print("Use --robot_type a7")
    print("Do not use --enable_dex3_dds")


if __name__ == "__main__":
    main()
