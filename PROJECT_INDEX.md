# R1-A7 项目总索引

本仓库包含 R1-A7 机器人仿真、VR 遥操作、手臂控制台、细线穿孔、视觉伺服、ACT 复现和扳手任务等多个阶段性项目。

## 使用规则

- `main` 只保留可公开说明和基础演示入口，不作为日常实验分支。
- 每个项目使用独立目录；分支用于开发线、实验线或冻结快照，不用分支代替目录分类。
- 真机控制代码、仿真代码、数据记录和研究文档分别管理。
- 数据集、视频、模型和 checkpoint 不直接作为日常提交内容；使用清单记录本地路径、大小、SHA256 和下载方式。
- 删除旧分支前必须先在 `BRANCH_CATALOG.md` 登记用途并创建 tag。

## 项目入口

| 项目 | 目录或入口 | 当前定位 | 主要入口文档 |
|---|---|---|---|
| R1-A7 基础仿真与官方示例 | `simulation/`、`tasks/`、`robots/`、`dds/` | 共享基础层 | `README_zh-CN.md` |
| Quest 3 VR 遥操作 | `vr_teleop/`、`tools/r1a7_vr_dual_arm_g1ik_real.py` | 已验证运行线 | `docs/r1a7_quest3_dual_arm_dex1_demo_guide_zh.md` |
| 相机遥操作与相机调试 | `camera_teleop/`、`tools/r1a7_camera_real_teleop.py` | 设备与视觉输入 | `docs/r1a7_camera_debug_runbook_zh.md` |
| 扳手任务 | `r1a7_wrench_project/` | 基线与数据采集 | `docs/r1a7_wrench_work_summary_2026-08-18_zh.md` |
| 细线镊子穿孔 | `r1a7_threading_project/` | 当前主研究项目 | `r1a7_threading_project/README.md` |
| 3D 控制台 | `r1a7_threading_project/console3d/` | R1-A7 手臂/夹爪可视化控制 | `r1a7_threading_project/console3d/README_zh.md` |
| ACT 复现与迁移 | `act_reproduction/` | 独立复现与迁移实验 | `act_reproduction/README.md`（若存在） |
| 标定与视觉伺服 | `calibration/`、`r1a7_threading_project/calibration/` | 标定、双相机和视觉引导 | `docs/` 与项目内 calibration 说明 |
| 第三方依赖 | `.gitmodules`、`third_party_notes/` | 外部依赖说明 | `third_party_notes/` |

## 文档分类

- `docs/architecture/`：软件架构、模块关系、代码解析报告。
- `docs/operations/`：启动、停止、通信、相机、VR 和真机安全操作。
- `docs/experiments/`：实验目的、参数、现象、日志、结论和下一步。
- `docs/source_records/`：由 Word、PPT 和聊天记录整理出的工作过程。
- `docs/archive/`：已结束项目或历史快照的说明。

当前历史文档仍保留在原路径，以保证旧命令和链接不失效。后续新增文档按上述分类放置，物理迁移在单独重构分支中进行。

## 开始查找

1. 先阅读本文件，确认项目归属。
2. 再阅读 `BRANCH_CATALOG.md`，确认代码所在分支和状态。
3. 再进入项目 README，确认环境、硬件、启动命令和安全要求。
4. 最后查看实验记录，不把实验记录中的临时命令当作稳定入口。

