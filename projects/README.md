# R1-A7 项目分类入口

这里是统一工作区的项目导航层。项目代码仍保留原有运行路径，避免破坏已经验证的启动命令；每个入口文件夹说明项目目标、代码位置、历史来源和推荐阅读顺序。

## 项目列表

| 项目 | 入口说明 | 代码位置 |
|---|---|---|
| [细线穿孔](r1a7_threading/README.md) | 当前主研究项目，镊子夹线、穿孔、视觉引导和实验记录 | `r1a7_threading_project/` |
| [扳手螺母](r1a7_wrench/README.md) | 扳手示教、VR 迁移、抓取和数据记录 | `r1a7_wrench_project/` |
| [ACT 复现](r1a7_act_reproduction/README.md) | 官方 ACT 仿真复现、R1-A7 迁移和离线评估 | `act_reproduction/` |
| [手臂控制台](r1a7_arm_console/README.md) | 3D 模型、笛卡尔移动、关节控制和夹爪控制 | `r1a7_threading_project/console3d/` |
| [视觉伺服](r1a7_visual_servo/README.md) | 双相机、标定、目标检测和视觉引导 | `r1a7_threading_project/vision/`、`vision_dual_camera/` |
| [VR 遥操作](r1a7_vr_teleop/README.md) | Quest 3、TeleVuer、双臂 IK 和 Dex1 | `tools/`、`action_provider/`、`vr_teleop/` |
| [相机识别控制](r1a7_camera_control/README.md) | Gemini、腕部/固定相机输入和相机位姿控制 | `camera_teleop/`、`tools/` |
| [共享运行时](r1a7_shared_runtime/README.md) | DDS、机器人模型、仿真任务和公共脚本 | `dds/`、`robots/`、`tasks/`、`scripts/` |

## 推荐阅读顺序

1. 先阅读目标项目入口。
2. 再阅读 `PROJECT_INDEX.md` 和 `BRANCH_CATALOG.md`。
3. 再查看 `docs/` 中的操作手册和实验记录。
4. 需要运行时才进入原始代码路径，不要从历史备份开始复制代码。

