# R1-A7 当前工作状态

最后整理：2026-10-27（Asia/Shanghai）

本文件是新对话和长对话恢复工作的第一入口。它记录稳定工作线和验证边界，不代替
实时设备检查。

## 唯一当前代码入口

| 项目 | 当前值 |
|---|---|
| 主工作区 | `/home/robot/unitree_sim_isaaclab_console` |
| 推荐分支 | `codex/r1a7-console-camera-suite-2026-10-11` |
| 指标设定前基线 | `3ab3bfc` — 2026-10-26 夹线/左腕同步提交 |
| GitHub | `git@github.com:sbboys/r1a7-quest3-vr-teleop-demo.git` |
| 控制台 | `r1a7_threading_project/console3d_hierarchical/start_console_8098.sh` |
| 页面版本 | `20261018-console-work-summary` |

分支头会继续变化。新对话必须用 `git rev-parse HEAD` 和 `git status --short` 获取
实时值，不得把上表提交号当作永远不变的 HEAD。

## 其他目录的定位

- `/home/robot/unitree_sim_isaaclab_threading`：实验研究参考区，当前存在大量未提交
  和未跟踪文件。它保存原始实验过程，但不是当前 GitHub 控制台代码事实源。
- `/home/robot/R1A7_VR_dual_arm_transfer_20260831_001`：外部 R1A7 IK/运行依赖。
- `/home/robot/桌面/jiqiren/renwu`：原始 Word 工作记录来源。
- `/home/robot/桌面/R1-A7_控制台_连续控制_视觉引导项目文件_2026-09-22`：历史桌面
  快照，只用于追溯。

## 已确认完成

- GitHub 工作区和分支索引已整理，主控制台功能线已推送远端。
- 控制台恢复终端同基线参数：30 Hz IK、250 Hz LowCmd、10 mm/s、20 mm/s²。
- 左右肩 Kd30、DQIK/POSVEL、重力前馈接口和腕肘软姿态约束已做对称化。
- 右夹爪闭合端点可调，默认 `-0.20`，未提高原有增益强制闭合。
- 视觉伺服改为单击锁存，执行上限按双目/单目/盲行分为 10/2/0.5 mm/s。
- 腕部识别脚本已与研究逻辑衔接，并保留部署环境变量覆盖。
- 头部相机已接入官方 Stereo patch PC1、UDP 5001、RTP/H.264 路线，并保存/恢复
  手机 App 相关服务原状态。
- 2026-10-27 使用 `tv` 环境和 Unitree SDK 路径复测，18 项控制台单元测试通过。
- 2026-10-20 至 10-22 的双相机夹线源码、左腕 007Z 独立识别程序、候选标定和五份
  原始记录已于 2026-10-26 衔接到当前 GitHub 工作线。
- 2026-10-27 已建立细线识别、夹持、对准、穿孔和安全五层指标协议，并补充逐次试验
  CSV 模板、机器可读候选阈值和离线统计工具。该更新只完成研究设计和工具验证，未执行真机。
- 2026-10-27 已复核本地 19 份关节/右臂正式 CSV、14 组镊尖稳定性结果和三档穿孔
  历史数据，形成现有指标基线；仅 10 mm/s Test02 有单次成功标签，三档成功率仍不可计算。

## 2026-10-26 新增研究线边界

- 双相机夹线栈默认 Shadow，控制门支持 `DUAL`、`WRIST_ONLY`、`EXTERNAL_ONLY`，
  速度文件三维合速度上限为 `10 mm/s`，不启用盲行或自动闭爪。
- 左腕 007Z 使用 `pyorbbecsdk` RGB 取流，序列号 `CPCBC530007Z`，主程序仍固定
  `motion_enabled=false`，且正式引导有效位保持关闭。
- 左腕和外部镊尖标定仍为 candidate；左腕内参来源需要重新确认。
- 夹爪自动控制文档是方案，不是已经完成的功能。

## 尚未通过的硬件项目

- 左臂真实低速方向点动；
- 右夹爪 `-0.20` 的真实最大闭合和夹持力；
- 控制台视觉伺服驱动真实机械臂完成穿孔；
- 头部相机 UDP 5001 的真实 RTP/H.264 首帧；
- 15/25 mm/s 穿孔实验的成功/失败标签和重复资格验证。
- 双相机夹线栈驱动真实手臂抓线；
- 左腕 007Z 补丁后的相机实测、独立内参/镊尖标定验证；
- 自动识别 `wire_inside_jaws` 后闭合夹爪。

## 下一条机器人工作线

在用户明确要求真机测试后：

1. 生成只读运行状态快照；
2. 确认唯一 LowCmd 发布者、急停、起始姿态和相机有效性；
3. 左臂先做 1 mm、10 mm/s 以内的单轴短行程；
4. 再分步验证右夹爪闭合端点；
5. 视觉链先 Shadow，再做 10 mm/s 双目短时执行；
6. 头部相机单独检查 UDP 5001 数据包和机器人补丁版本。

详细证据见：
`r1a7_threading_project/docs/r1a7_experiment_console_update_2026-10-18_zh.md`。

双相机夹线和左腕 007Z 衔接见：
`r1a7_threading_project/docs/r1a7_wire_grasp_left_wrist_sync_2026-10-26_zh.md`。

研究指标、阶段门控和试次统计口径见：
`r1a7_threading_project/docs/r1a7_wire_threading_metrics_protocol_2026-10-27_zh.md`。

历史数据能够支持的当前基线及证据缺口见：
`r1a7_threading_project/docs/r1a7_existing_motion_perception_task_metrics_2026-10-27_zh.md`。

桌面和新对话恢复结构见：
`docs/r1a7_desktop_continuity_setup_2026-10-18_zh.md`。

## 2026-11-02 文档与研究快照补充

最后补充核对：2026-11-02。以上10/27控制台集成状态作为历史基线保留；新增研究整理分支为 `codex/r1a7-vision-doc-sync-2026-11-02`（基于8d01fda）。

- 两份原件与远端字节相同；10/30和11/02原件、四份全文、25个U-Net实验源码和小型结果已整理。
- console仍是原集成入口；11/02文档指定的实际新视觉/控制研究发生于threading，两者不是同一版本。
- 文档记载新视觉发生过抬臂和冲突停止，不能当作通过闭环验收；本次未运行硬件。
- 当前threading源码已含Z冲突补丁与O/V/C，均只归档，未覆盖原控制台入口；运行/真机待验。
- 下一步：按差异快照审查集成及不触发输出的接口测试，再进行独立授权的硬件回归。

入口：[完整核对报告](../r1a7_threading_project/docs/r1a7_four_document_github_audit_2026-11-02_zh.md)。

## 2026-11-03 合并交接

用户授权将四文档整理分支继续合并至 `codex/r1a7-console-camera-suite-2026-10-11`。采用fast-forward，纳入归档提交 `36445a6` 及本条交接记录；不通过重写历史或替换控制器完成合并。GitHub插件缺少创建PR权限，因此使用已验证可用的Git SSH通道直接更新研发分支。

原位Fusion、Gate和控制台控制器保持原版本；研究补丁仍仅为快照。合并没有启动任何硬件，未升级硬件验收等级。下一步仍为控制差异审查与无真实输出的接口测试。

## 2026-11-04 右腕张开镊子研究补充

[11/03来源与当前文件核对报告](../r1a7_threading_project/docs/r1a7_right_wrist_open_autograsp_archive_2026-11-04_zh.md)承接c1f466a。已归档64图数据的元信息、V3 mixed replay、9图历史测试、双模型和midpoint源码；权重/原始图像外置。

当前研究文件已含midpoint接入及Right Wrist ev CLOSE条件删除，文档当时“待确认”已据此补充为“源码存在、运行未验收”。直接自动抓取状态机在研究源码中已实现，因此旧“自动闭爪仅方案”只保留为原控制台历史状态，不能覆盖本次研究线。控制器只保存快照，不部署现用控制台。

本次只做文件归档和静态检查，没有新运动或完整CLOSE→DONE成功证据。下一步仍需单独核验TIP语义、CLOSE判据、HOME打断与停止/新鲜度行为。
