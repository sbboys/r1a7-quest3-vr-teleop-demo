# GitHub 分支目录表

仓库：`sbboys/r1a7-quest3-vr-teleop-demo`

本表基于 2026-09-29 对远端 `demo` 的实际检查结果。当前分支大部分是完整快照，不能按普通功能分支直接合并。

## 远端分支

| 分支 | 最新提交 | 文件数 | 定位 | 状态 |
|---|---:|---:|---|---|
| `main` | `e365fa8` | 236 | Quest 3 VR 遥操作基础演示 | 基础入口 |
| `r1a7-camera-debug-runbook` | `397dac0` | 237 | 基于 main 的相机调试运行手册 | 可归并为 main 文档更新 |
| `codex-handoff-20260731` | `e903e91` | 84 | 2026-07-31 交接快照 | 历史归档 |
| `r1a7-wrench-baseline` | `1c288c2` | 537 | 扳手任务基线 | 历史实验基线 |
| `r1a7-threading-project` | `95aee60` | 619 | 细线穿孔项目早期快照 | 历史快照 |
| `r1a7-act-reproduction` | `2a49fab` | 689 | ACT 复现与 R1-A7 迁移 | 独立复现项目 |
| `codex/r1a7-threading-project-complete-2026-09-22` | `d156f3c` | 863 | 细线项目完整快照 | 冻结归档 |
| `codex/r1a7-arm-console` | `4e5ec2d` | 677 | 手臂控制台快照 | 功能快照 |
| `codex/r1a7-dual-camera-visual-servo-2026-09-29` | `621e53d` | 886 | 双相机视觉伺服和代码解析报告 | 当前资料最完整 |
| `codex/repository-management-2026-09-29` | 本分支 | 以最新视觉伺服快照为基础 | 仓库管理与分类 | 整理中 |
| `codex/r1a7-unified-workspace-2026-09-29` | `494e0ee` | 统一项目入口和跨分支内容 | ACT、细线、扳手、VR、相机、视觉伺服、控制台 | 当前整合分支 |

## 重要结构结论

### 1. 分支不是一条连续开发线

`main` 和 `r1a7-camera-debug-runbook` 使用早期快照根提交；大多数 2026-09 项目分支使用另一根提交。它们之间没有共同祖先，不能使用普通的 `main..feature` 方式判断增量，也不应直接强行合并。

### 2. 后期分支是累计快照

`r1a7-threading-project`、ACT、控制台、完整细线项目和双相机分支逐步叠加了多个历史项目。因此后期分支同时包含 `r1a7_wrench_project/`、`r1a7_threading_project/`、基础仿真文件和大量历史文档。

### 3. 主分支是轻量基础入口

`main` 主要包含：

- `camera_teleop/`：相机识别和相机位姿遥操作；
- `vr_teleop/`：Quest 3、TeleVuer、G1_29 IK、R1-A7 双臂和 Dex1 遥操作；
- `simulation/`、`tasks/`、`robots/`、`dds/`：仿真和 DDS 基础；
- `scripts/`、`tools/`：运行和检查脚本；
- `docs/`：基础运行手册；
- `archive/`：历史脚本和备份；
- `third_party_notes/`：第三方依赖安装和下载说明。

## 分支使用规则

- 稳定版本：`main` 或经过验证的 tag。
- 日常集成：`develop/<project>`。
- 单项开发：`feature/<project>-<topic>`。
- 单次实验：`exp/<project>-<date>-<topic>`。
- 冻结快照：`archive/<project>-<date>`。

现有快照分支暂不删除。完成目录索引和 tag 后，再将不再使用的分支改为 archived，并在 GitHub 分支保护设置中禁止直接向 `main` 推送实验代码。
