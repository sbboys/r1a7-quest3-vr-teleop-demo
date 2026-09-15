# ACT 数据与模型清单

本文件只记录本机路径、用途和完整性校验方法，不上传数据集和模型权重。

## 数据集

- 官方 Transfer Cube 训练集：`/data/ACT/datasets/transfer_cube_50`
- 官方 Transfer Cube 测试集：`/data/ACT/datasets/transfer_cube_test`
- R1-A7 ACT V1：`/data/R1A7/wrench_act_pilot_v1`
- R1-A7 ACT V2：`/data/R1A7/wrench_act_pilot_v2`

## Checkpoint

- 官方 5000 epochs 最优模型：`/data/ACT/checkpoints/transfer_cube_act_5000ep/policy_best.ckpt`
- 对应统计文件：该目录下的 `dataset_stats.pkl`
- R1-A7 V1/V2 checkpoint：分别位于对应实验目录的 checkpoint 子目录，上传前需重新确认实际文件名。

## 校验命令

```bash
sha256sum /data/ACT/checkpoints/transfer_cube_act_5000ep/policy_best.ckpt
sha256sum /data/ACT/checkpoints/transfer_cube_act_5000ep/dataset_stats.pkl
find /data/ACT/datasets/transfer_cube_50 -type f -name '*.hdf5' -print0 | sort -z | xargs -0 sha256sum
```

校验值可能因重新生成数据或重新训练而变化；GitHub 中只保留路径、用途和生成/校验方法。

