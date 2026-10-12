# GitHub 分支目录表

仓库：[`sbboys/r1a7-quest3-vr-teleop-demo`](https://github.com/sbboys/r1a7-quest3-vr-teleop-demo)

本表基于 2026-10-11 对远端 `demo` 的实时核验。整理前共有 12 条远端分支、0 个 tag；
本次新增统一工作分支 `codex/r1a7-workspace-2026-10-11`，并为 12 个历史节点和当前统一节点建立 `snapshot/*` 标签；不删除、不改写任何历史分支。

2026-10-12 补充：从统一工作线建立
`codex/r1a7-console-camera-suite-2026-10-11`，用于 8098 控制台的低速控制器、
双夹爪和六路相机完善。该分支不改写历史，验收记录见
`r1a7_threading_project/docs/r1a7_console_control_camera_suite_2026-10-12_zh.md`。

## 当前推荐分支

| 分支 | 基线 | 纳入内容 | 用途 |
|---|---|---|---|
| `codex/r1a7-workspace-2026-10-11` | `3cced5f2` | `e703b9f0` 视觉伺服/低速控制；`b1608303` 控制台修正；本次治理文档和检查流程 | 后续研发统一入口 |
| `codex/r1a7-console-camera-suite-2026-10-11` | 上述统一工作线 | 100/500 Hz 控制、双臂/夹爪、双腕/外部/双奥比/头部相机 | 当前控制台功能分支 |

## 远端分支清单

| 分支 | 2026-10-11 提交 | 文件数 | 定位 | 后续策略 |
|---|---:|---:|---|---|
| `main` | `e365fa81` | 236 | Quest 3 VR 遥操作早期公开演示 | 保持只读基线，不强行合并 |
| `r1a7-camera-debug-runbook` | `397dac04` | 237 | 相机调试手册 | 历史参考，打快照 tag |
| `codex-handoff-20260731` | `e903e919` | 84 | 2026-07-31 交接快照 | 历史归档，打快照 tag |
| `r1a7-wrench-baseline` | `1c288c22` | 537 | 扳手任务基线 | 项目基线，打快照 tag |
| `r1a7-threading-project` | `95aee60f` | 619 | 细线穿孔早期快照 | 已被完整快照覆盖 |
| `r1a7-act-reproduction` | `2a49fab3` | 689 | ACT 复现与迁移 | 已纳入统一工作区，保留独立快照 |
| `codex/r1a7-threading-project-complete-2026-09-22` | `d156f3cd` | 863 | 细线项目完整快照 | 冻结归档 |
| `codex/r1a7-arm-console` | `b1608303` | 678 | 手臂控制台及方向切换修正 | 已合并到统一工作线 |
| `codex/r1a7-dual-camera-visual-servo-2026-09-29` | `621e53d5` | 886 | 双相机视觉伺服与架构报告 | 已被 10 月工作线覆盖 |
| `codex/repository-management-2026-09-29` | `fccc0aa1` | 895 | 首轮仓库管理索引 | 已被统一工作区覆盖 |
| `codex/r1a7-unified-workspace-2026-09-29` | `3cced5f2` | 1105 | 首轮跨项目统一工作区 | 本次整合基线 |
| `codex/r1a7-visual-servo-control-2026-10-10` | `e703b9f0` | 931 | 最新视觉伺服与低速控制成果 | 已合并到统一工作线 |

## 快照标签

| 标签 | 指向提交 |
|---|---:|
| `snapshot/quest3-demo-2026-07-28` | `e365fa81` |
| `snapshot/quest3-handoff-2026-07-31` | `e903e919` |
| `snapshot/camera-debug-runbook-2026-08-14` | `397dac04` |
| `snapshot/wrench-baseline-2026-09-03` | `1c288c22` |
| `snapshot/threading-project-2026-09-15` | `95aee60f` |
| `snapshot/act-reproduction-2026-09-15` | `2a49fab3` |
| `snapshot/threading-complete-2026-09-22` | `d156f3cd` |
| `snapshot/dual-camera-visual-servo-2026-09-29` | `621e53d5` |
| `snapshot/repository-management-2026-09-29` | `fccc0aa1` |
| `snapshot/unified-workspace-2026-09-29` | `3cced5f2` |
| `snapshot/arm-console-fix-2026-09-30` | `b1608303` |
| `snapshot/visual-servo-control-2026-10-10` | `e703b9f0` |
| `snapshot/r1a7-workspace-2026-10-11` | 当前统一工作节点 |

## 谱系结论

### 两组历史不能直接合并

`main`、`r1a7-camera-debug-runbook`、`codex-handoff-20260731` 与后期研发工作区没有共同祖先。
它们是独立导入的快照，不应使用强制合并来制造一条虚假的线性历史。

### 后期工作线已经收敛

- 统一工作区与 10 月视觉伺服线的共同基点是 `621e53d5`。
- 控制台修正与统一工作区共享后期研发历史。
- 当前统一工作线包含上述三条线的提交；冲突仅发生在项目 README，已合并保留两侧入口。

## 分支命名与生命周期

- 当前统一工作：`codex/r1a7-workspace-YYYY-MM-DD`
- 功能开发：`codex/<project>-<topic>`
- 实验：`exp/<project>-<date>-<topic>`
- 冻结标签：`snapshot/<project>-YYYY-MM-DD`

历史分支暂不删除。待统一工作线完成真机回归和 GitHub 默认分支/保护策略确认后，再决定是否在界面中隐藏旧分支。
