# R1 A7 ACT Reproduction

This branch packages the completed ACT Transfer Cube simulation reproduction and the R1-A7 migration and evaluation code. It intentionally excludes datasets, checkpoints, runtime logs, Python caches, and local machine credentials.

## Layout

- `official_act/`: ACT Transfer Cube simulation, training, evaluation, DETR-VAE model, and MuJoCo assets.
- `r1a7_eval/`: R1-A7 16D migration, three-camera recording, inference server, real-robot control client, and analysis tools.
- `docs/`: reproduction commands, verified results, and data/checkpoint manifests.

## Quick start

Use the existing `tv` environment for R1-A7 control and the existing `aloha` environment for ACT training or offline evaluation. The original ACT environment files are retained as references only.

```bash
cd /home/robot/unitree_sim_isaaclab_threading/act_reproduction/official_act
conda activate aloha
python record_sim_episodes.py --task_name sim_transfer_cube --dataset_dir /data/ACT/datasets/transfer_cube_50 --num_episodes 50
python imitate_episodes.py --task_name transfer_cube --ckpt_dir /data/ACT/checkpoints/transfer_cube_act_5000ep --policy_class ACT --batch_size 8 --seed 0 --num_epochs 5000 --pretrain_ckpt none --eval
```

The exact verified commands and results are in `docs/reproduction_runbook_2026-09-15_zh.md`. Check the local robot IP, NIC, DDS domain, camera serial numbers, and safety limits before any real-robot command.

