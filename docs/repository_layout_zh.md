# 仓库管理划分说明

本仓库是 R1-A7 Quest 3 遥操作演示项目的独立 private demo 仓库。

## 目录划分

```text
camera_teleop/
```

相机识别、Gemini/Orbbec 相机位姿输入、相机控制机械臂相关代码。

```text
vr_teleop/
```

Quest 3 手柄、TeleVuer、G1_29 IK、R1-A7 双臂、Dex1 双夹爪联合遥操作代码。

```text
simulation/
```

IsaacLab / R1-A7 仿真任务、机器人配置、DDS 辅助模块。

```text
scripts/
```

现场启动、停止、检查、Dex1 服务、相机控制和示教相关脚本。

```text
tools/ action_provider/ robots/ tasks/ dds/
```

兼容原始工作目录的可运行入口。现场 SOP 和 Demo 指南中的命令默认使用这些根目录路径。

```text
docs/
```

Demo 指南、完整项目汇总、SOP、故障排查和历史整理说明。

```text
archive/
```

历史备份、旧实验脚本、旧日志和下载目录重复脚本。保留用于追溯，不作为正式入口。

```text
third_party_notes/
```

大型第三方依赖说明和下载脚本。Orbbec SDK/Viewer 二进制包未直接提交到 Git。

## 正式入口

真机双臂与夹爪联合控制：

```text
tools/r1a7_vr_dual_arm_g1ik_real.py
```

Dex1 服务：

```text
scripts/run_dex1_1_service.sh
```

Demo 操作指南：

```text
docs/r1a7_quest3_dual_arm_dex1_demo_guide_zh.md
```

完整项目汇总：

```text
docs/r1a7_quest3_dual_arm_dex1_project_summary_zh.md
```
