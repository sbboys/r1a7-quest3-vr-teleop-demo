# R1-A7 项目总索引

本仓库是 R1-A7 多项目研发工作区，不是单一演示程序。当前统一工作线为
`codex/r1a7-console-camera-suite-2026-10-11`，它以 2026-09-29 统一工作区为底座，已纳入
2026-10-10 视觉伺服/低速控制成果和 2026-09-30 手臂控制台方向切换修正。

> 本索引是导航，不改变已经验证的运行路径。历史文档中的临时命令不自动等同于当前稳定入口。

## 00—09 逻辑目录

| 编号 | 领域 | 当前物理入口 | 状态 | 首选说明 |
|---:|---|---|---|---|
| 00 | 仓库治理 | `PROJECT_INDEX.md`、`BRANCH_CATALOG.md`、`docs/` | 当前 | [整理报告](docs/r1a7_github_reorganization_2026-10-11_zh.md) |
| 01 | 共享运行时 | `simulation/`、`tasks/`、`robots/`、`dds/` | 共享基础 | [共享运行时入口](projects/r1a7_shared_runtime/README.md) |
| 02 | Quest 3 / VR 遥操作 | `vr_teleop/`、`tools/`、`action_provider/` | 已验证工作线 | [VR 遥操作入口](projects/r1a7_vr_teleop/README.md) |
| 03 | 相机与识别控制 | `camera_teleop/`、`tools/` | 设备输入层 | [相机控制入口](projects/r1a7_camera_control/README.md) |
| 04 | 扳手任务 | `r1a7_wrench_project/` | 冻结基线/采集线 | [扳手项目入口](projects/r1a7_wrench/README.md) |
| 05 | 细线镊子穿孔 | `r1a7_threading_project/` | 夹线/左腕已衔接，指标 v1 已建立，真机抓线待验 | [细线项目入口](projects/r1a7_threading/README.md) |
| 06 | 双相机视觉伺服 | `r1a7_threading_project/vision_dual_camera/` | 穿孔控制与夹线 Shadow 两条研究线 | [视觉伺服入口](projects/r1a7_visual_servo/README.md) |
| 07 | 低速平滑控制 | `r1a7_threading_project/transfer_control/` | 10 mm/s 基线已记录 | [控制交接](r1a7_threading_project/docs/r1a7_visual_servo_control_handoff_2026-10-10_zh.md) |
| 08 | 手臂 3D 控制台 | `r1a7_threading_project/console3d*/` | 30/250 Hz 低速控制、双臂对称参数与六路相机已衔接 | [控制台入口](projects/r1a7_arm_console/README.md) |
| 09 | ACT 复现与迁移 | `act_reproduction/` | 独立复现线 | [ACT 入口](projects/r1a7_act_reproduction/README.md) |

## 稳定入口和实验入口

- 稳定/推荐入口：项目 README 明确标记并已有验证记录的脚本。
- 实验入口：`experimental/`、`diagnostics/`、日期化探针和单次条件测试脚本。
- 历史入口：旧分支、`archive/`、备份文件和原始工作记录，仅用于追溯。
- 真机入口：凡发布 `rt/lowcmd` 或可能驱动手臂/夹爪的程序，运行前必须核对通信、控制权、停止方法和工作空间。

## 数据和大文件边界

- 源码、配置、小型标定结果、清单和结论文档可提交。
- 视频、rosbag、原始图像集、训练数据、权重和 checkpoint 默认不继续直接提交。
- 已进入历史的较大文件暂不执行历史改写；新增大文件使用 `datasets/`、`checkpoints/` 清单或 Git LFS/外部存储。
- 清单至少记录文件名、大小、SHA256、采集日期、设备、环境和获取方式。

## 推荐阅读顺序

1. 本索引：确认工作属于哪个项目。
2. [分支目录表](BRANCH_CATALOG.md)：确认应使用当前整合线还是历史快照。
3. 对应的 `projects/<project>/README.md`：确认代码位置和安全边界。
4. 项目内操作手册、交接文档和实验记录。

当前阶段总览：[2026-10-18 实验工作与控制台更新报告](r1a7_threading_project/docs/r1a7_experiment_console_update_2026-10-18_zh.md)。

最新研究衔接：[2026-10-26 双相机夹线与左腕 007Z 衔接记录](r1a7_threading_project/docs/r1a7_wire_grasp_left_wrist_sync_2026-10-26_zh.md)。

研究评价入口：[2026-10-27 细线夹取与穿孔研究指标协议](r1a7_threading_project/docs/r1a7_wire_threading_metrics_protocol_2026-10-27_zh.md)。

## 仓库规则

- `main` 保持早期 Quest 3 演示基线，不在本次整理中改写历史或强行合并。
- 新工作从统一工作线分出短期功能分支；验证后回到统一工作线。
- 旧快照先打 tag、登记用途，再决定是否在 GitHub 上隐藏或归档；本次不删除任何分支。
- 路径重构必须单独进行，并同步修正导入、脚本和文档链接。
