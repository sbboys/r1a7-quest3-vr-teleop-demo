# R1-A7 长周期项目协作规则

本仓库是 R1-A7 当前代码与文档事实源。新对话、长对话压缩后恢复工作或更换模型时，
必须先按下列顺序读取，不得只依赖聊天记忆：

1. `docs/CURRENT_STATE.md`
2. `docs/DECISIONS.md`
3. `docs/VALIDATION_MATRIX.md`
4. `docs/NEW_CHAT_START.md`
5. 当前任务所属项目的 README 和最近日期化报告

## 工作区边界

- 当前主工作区：`/home/robot/unitree_sim_isaaclab_console`
- 当前 GitHub 远端：`demo` → `sbboys/r1a7-quest3-vr-teleop-demo`
- 研究/实验参考区：`/home/robot/unitree_sim_isaaclab_threading`
- 外部 IK 运行依赖：`/home/robot/R1A7_VR_dual_arm_transfer_20260831_001`
- 桌面 `R1-A7_控制台_连续控制_视觉引导项目文件_2026-09-22` 是历史快照，不是当前代码入口。

研究区当前包含大量未提交实验文件。除非用户明确要求，不得清理、重置、移动或把它
整体当作当前 GitHub 状态。需要采用研究成果时，先对比文件、运行命令、日志和验证边界，
再把经过选择的内容同步到主工作区。

## 开始工作

1. 运行 `git status --short`、`git branch --show-current`、`git log -1 --oneline`。
2. 读取上述状态文档，列出“已完成”“需要实时复核”“尚未验证”三类信息。
3. 涉及机器人、相机或控制台运行时，执行只读脚本：
   `bash scripts/r1a7_context_snapshot.sh`。
4. 文档与实际文件不一致时先报告差异，不要自行切换到旧脚本或另一工作区。

## 真机安全边界

- 用户没有明确要求真实运动时，只允许源码、文档、日志、网络和进程的只读检查。
- 真实运动前必须重新确认网卡/路由、机器人起始姿态、急停、相机首帧、控制器进程、
  锁文件和唯一 `rt/lowcmd` 发布者；历史记录不能替代这些实时检查。
- 代码编译、单元测试、网页加载、零运动启动都不等于真实机械臂或夹爪验收通过。
- 运行结果必须写入持久目录，不能只保存在 `/tmp`。

## 当前控制线

- 8098 控制台入口：
  `r1a7_threading_project/console3d_hierarchical/start_console_8098.sh`
- 控制器入口：
  `r1a7_threading_project/transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py`
- 当前默认基线：30 Hz IK、250 Hz LowCmd、10 mm/s、20 mm/s²。
- 视觉执行速度分层：双目 10、单目 2、短时盲行 0.5 mm/s。
- 头部相机采用官方 Stereo patch PC1 → UDP 5001 → RTP/H.264 路线；真实首帧仍未通过。

## 结束和交接

完成一个可复用阶段后，更新 `docs/CURRENT_STATE.md` 与
`docs/VALIDATION_MATRIX.md`，记录提交、运行命令、验证级别、未解决问题和下一步。
禁止把“代码已更新”写成“硬件已验证”。
