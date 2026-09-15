# 环境复用记录

更新日期：2026-09-10

## 当前决策

第一阶段先复用已有 conda 环境：

- `tv`：用于 R1-A7 机器人控制、VR/TeleVuer、相机和 LowCmd 相关运行。
- `aloha`：用于 ACT 或后续学习模型训练、推理相关运行。

当前不新建环境，不升级核心依赖。

## 需要新建环境的触发条件

出现以下情况时再克隆环境：

1. 需要安装与 `tv` 或 `aloha` 现有依赖冲突的新库。
2. 需要升级 PyTorch、CUDA、numpy、opencv、pyrealsense2、unitree_sdk2py 等核心依赖。
3. 细线项目改动导致扳手项目或已有 VR 控制链不可复现。

建议命名：

```bash
conda create -n tv_threading --clone tv
conda create -n aloha_threading --clone aloha
```

## 建议保存环境快照

在开始安装任何新依赖前执行：

```bash
conda env export -n tv > r1a7_threading_project/docs/env_tv_before_threading_2026-09-10.yml
conda env export -n aloha > r1a7_threading_project/docs/env_aloha_before_threading_2026-09-10.yml
```

