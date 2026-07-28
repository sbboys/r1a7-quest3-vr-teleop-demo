# -*- coding: utf-8 -*-
"""R1-A7 articulation configuration for Isaac Lab."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

A7_USD_PATH = (
    "/home/robot/IsaacLab/bolt_nut_assembly/"
    "R1_A7_official/A7.usd"
)

A7_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=A7_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.15, 0.0, 0.76),
        rot=(0.7071, 0.0, 0.0, 0.7071),
        joint_pos={
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.25,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.30,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.25,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.30,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
            "head_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit=60.0,
            velocity_limit=18.7,
            stiffness=100.0,
            damping=5.0,
            armature=0.01,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint"],
            effort_limit=60.0,
            velocity_limit=18.7,
            stiffness=90.0,
            damping=5.0,
            armature=0.01,
        ),
        "elbows": ImplicitActuatorCfg(
            joint_names_expr=[".*_elbow_joint"],
            effort_limit=33.0,
            velocity_limit=52.4,
            stiffness=70.0,
            damping=4.0,
            armature=0.01,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_roll_joint"],
            effort_limit=33.0,
            velocity_limit=52.4,
            stiffness=45.0,
            damping=2.5,
            armature=0.005,
        ),
        "wrist_pitch_yaw": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit=10.0,
            velocity_limit=37.7,
            stiffness=35.0,
            damping=2.0,
            armature=0.005,
        ),
        "head_pitch": ImplicitActuatorCfg(
            joint_names_expr=["head_pitch_joint"],
            effort_limit=33.0,
            velocity_limit=52.4,
            stiffness=30.0,
            damping=2.0,
            armature=0.005,
        ),
        "head_yaw": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint"],
            effort_limit=0.86,
            velocity_limit=11.0,
            stiffness=10.0,
            damping=1.0,
            armature=0.002,
        ),
    },
)
