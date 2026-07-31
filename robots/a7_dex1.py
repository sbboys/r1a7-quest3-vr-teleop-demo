# -*- coding: utf-8 -*-
"""R1-A7 + dual Dex1 V4 configuration.

Use a freshly imported USD from R1_A7_Dex1_isaaclab_sync_v4.urdf.
The USD has four independent physical finger sliders. IsaacLab commands both
sliders of each gripper with one logical scalar.
"""

from copy import deepcopy

from isaaclab.actuators import ImplicitActuatorCfg

from robots.a7 import A7_CFG


R1_A7_DEX1_USD_PATH = (
    "/home/robot/IsaacLab/bolt_nut_assembly/g1_dex1_r1_v4.usd"
)

A7_DEX1_CFG = deepcopy(A7_CFG)
A7_DEX1_CFG.spawn.usd_path = R1_A7_DEX1_USD_PATH

# Internal gripper links overlap geometrically during import; disable robot
# self-collision for the first stable grasp test. Object/finger collisions remain.
if A7_DEX1_CFG.spawn.articulation_props is not None:
    A7_DEX1_CFG.spawn.articulation_props.enabled_self_collisions = False

A7_DEX1_CFG.init_state.joint_pos.update(
    {
        "left_dex1_Joint1_1": -0.018,
        "left_dex1_Joint2_1": -0.018,
        "right_dex1_Joint1_1": -0.018,
        "right_dex1_Joint2_1": -0.018,
    }
)

A7_DEX1_CFG.actuators["dex1_grippers"] = ImplicitActuatorCfg(
    joint_names_expr=[
        "left_dex1_Joint1_1",
        "left_dex1_Joint2_1",
        "right_dex1_Joint1_1",
        "right_dex1_Joint2_1",
    ],
    effort_limit=40.0,
    velocity_limit=0.25,
    stiffness=650.0,
    damping=32.0,
    armature=0.002,
    friction=0.0,
)
