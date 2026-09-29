# 统一仓库来源矩阵

| 工作内容 | 统一入口 | 原始分支来源 | 处理方式 |
|---|---|---|---|
| 细线镊子穿孔 | `r1a7_threading_project/` | `r1a7-threading-project`、`codex/r1a7-threading-project-complete-2026-09-22` | 采用最新完整快照 |
| 3D 控制台 | `r1a7_threading_project/console3d/` | `codex/r1a7-arm-console` | 合并到细线项目入口 |
| 双相机视觉伺服 | `r1a7_threading_project/vision_dual_camera/` | `codex/r1a7-dual-camera-visual-servo-2026-09-29` | 采用最新视觉伺服快照 |
| 扳手螺母 | `r1a7_wrench_project/` | `r1a7-wrench-baseline`及后续快照 | 保留为独立项目 |
| ACT 复现 | `act_reproduction/` | `r1a7-act-reproduction` | 纳入统一分支，独立入口 |
| VR 遥操作 | `tools/`、`action_provider/`、`vr_teleop/` | `main`、`r1a7-camera-debug-runbook`及后续快照 | 当前实现与早期实现并列保留 |
| 相机识别控制 | `camera_teleop/`、`tools/` | `main`、`r1a7-camera-debug-runbook` | 旧实现用于追溯，当前入口以项目 README 为准 |
| 仿真基础 | `tasks/`、`robots/`、`dds/`、`simulation/` | `main`及后续快照 | 当前运行树和早期兼容树并存，暂不强行重命名 |
| 交接快照 | GitHub 原分支 | `codex-handoff-20260731` | 保留历史，不重复复制 |

## 为什么保留旧分支

旧分支是实验时间线和提交证据，删除后会失去历史上下文。统一分支负责后续使用，旧分支负责追溯。确认新入口稳定后，再给旧分支打 tag 并设置为 archived。

