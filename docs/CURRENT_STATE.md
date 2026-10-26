# R1-A7 当前工作状态

最后整理：2026-10-26（Asia/Shanghai）

本文件是新对话和长对话恢复工作的第一入口。它记录稳定工作线和验证边界，不代替
实时设备检查。

## 唯一当前代码入口

| 项目 | 当前值 |
|---|---|
| 主工作区 | `/home/robot/unitree_sim_isaaclab_console` |
| 推荐分支 | `codex/r1a7-console-camera-suite-2026-10-11` |
| 本次同步前基线 | `23bb4e0` — 当前分支与远端一致的基线 |
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
- 2026-10-18 上传前 18 项控制台单元测试通过。
- 2026-10-20 至 10-22 的双相机夹线源码、左腕 007Z 独立识别程序、候选标定和五份
  原始记录已于 2026-10-26 衔接到当前 GitHub 工作线。

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

桌面和新对话恢复结构见：
`docs/r1a7_desktop_continuity_setup_2026-10-18_zh.md`。
