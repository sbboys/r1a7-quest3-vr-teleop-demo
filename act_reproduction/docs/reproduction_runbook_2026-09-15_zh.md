# R1 A7 ACT 复现与迁移运行说明

## 复现边界

官方 ACT Transfer Cube 基线使用代码提交 `742c753c0d4a5d87076c8f69e5628c79a8cc5488`。本分支保存可运行源码和 MuJoCo 资产，不保存训练数据与 checkpoint。

## 已验证结果

- 生成并确认 50 条 `sim_transfer_cube_scripted` 示教数据，目录为 `/data/ACT/datasets/transfer_cube_50`。
- 完成 5000 epochs 训练，checkpoint 位于 `/data/ACT/checkpoints/transfer_cube_act_5000ep/policy_best.ckpt`。
- 普通 ACT 闭环仿真：43/50，成功率 86%。
- Temporal Aggregation 闭环仿真：41/50，成功率 82%。
- R1-A7 迁移使用 canonical 16D 顺序：`[left_arm_7, left_gripper_1, right_arm_7, right_gripper_1]`。
- R1-A7 V1：5 条示教，500 epochs；V2：6 条示教，500 epochs，best validation loss 1.916447 @ epoch 411。

## 迁移代码

`r1a7_eval/r1a7_act_inference_server.py` 负责加载 checkpoint、三路 BGR 转 RGB、qpos 归一化和 ACT 推理；`r1a7_eval/r1a7_unitree_official_vuer_real_lowcmd.py` 负责与 R1-A7 控制链连接；三相机采集和健康监控代码位于同目录。

## 运行原则

- ACT GPU 推理和机器人 `tv` 控制环境通过 Unix socket 分离。
- 真机控制前确认 `enp6s0`、机器人地址 `192.168.123.161`、DDS domain `0`、`rt/lowstate` 和 `rt/lowcmd`。
- 首次运行使用单步、低速、限位和急停可达的测试方式；不要直接使用未重新核对的历史命令。

