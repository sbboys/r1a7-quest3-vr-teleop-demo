# R1-A7 ACT 仿真复现与迁移

## 目标

复现官方 ACT 仿真流程，并评估向 R1-A7、三相机、Dex1 和真实低层控制迁移的可行性。

## 代码位置

- 官方复现：`act_reproduction/official_act/`
- R1-A7 评估和迁移：`act_reproduction/r1a7_eval/`
- 复现说明：`act_reproduction/docs/`

## 数据规则

训练数据和 checkpoint 不直接提交。使用 `datasets/` 和 `checkpoints/` 中的 manifest 记录路径、哈希和下载方式。

