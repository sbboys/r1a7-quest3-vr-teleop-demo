import gymnasium as gym

from . import pickplace_cylinder_a7_joint_env_cfg

gym.register(
    id="Isaac-PickPlace-Cylinder-A7-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
            pickplace_cylinder_a7_joint_env_cfg.
            PickPlaceCylinderA7JointEnvCfg,
    },
    disable_env_checker=True,
)
