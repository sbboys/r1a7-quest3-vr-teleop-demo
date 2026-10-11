# 统一仓库来源矩阵

| 工作内容 | 当前统一入口 | 历史分支来源 | 2026-10-11 处理结果 |
|---|---|---|---|
| 细线镊子穿孔 | `r1a7_threading_project/` | `r1a7-threading-project`、`codex/r1a7-threading-project-complete-2026-09-22` | 采用完整快照并保留时间线 |
| 3D 手臂控制台 | `r1a7_threading_project/console3d/`、`console3d_hierarchical/` | `codex/r1a7-arm-console` | 已纳入方向连续切换修正 `b1608303` |
| 双相机视觉伺服 | `r1a7_threading_project/vision_dual_camera/` | `codex/r1a7-dual-camera-visual-servo-2026-09-29`、`codex/r1a7-visual-servo-control-2026-10-10` | 已纳入 10 月视觉状态机和启动栈修正 |
| 低速平滑控制 | `r1a7_threading_project/transfer_control/` | `codex/r1a7-visual-servo-control-2026-10-10` | 已纳入诊断、实验控制器和 10 mm/s 交接记录 |
| 扳手螺母 | `r1a7_wrench_project/` | `r1a7-wrench-baseline` 及后续快照 | 保留独立项目和历史采集边界 |
| ACT 复现 | `act_reproduction/` | `r1a7-act-reproduction` | 纳入统一工作区，保留独立入口 |
| VR 遥操作 | `tools/`、`action_provider/`、`vr_teleop/` | `main`、后期研发快照 | 当前实现与早期独立演示并列保留 |
| 相机识别控制 | `camera_teleop/`、`tools/` | `main`、`r1a7-camera-debug-runbook` | 旧实现用于追溯，当前入口以项目 README 为准 |
| 仿真基础 | `tasks/`、`robots/`、`dds/`、`simulation/` | 上游基础与后期快照 | 兼容树并存，暂不重命名 |
| 交接快照 | GitHub 原分支/tag | `codex-handoff-20260731` | 保留历史，不重复复制 |

## 说明

`main`、相机手册和 2026-07-31 交接快照与后期研发工作区没有共同祖先，所以本矩阵描述的是内容来源，不表示它们已经通过 Git 合并连接。

统一工作线负责后续使用，历史分支和 `snapshot/*` tag 负责追溯。路径搬迁不属于本次整理范围。
