# R1-A7 手臂控制台 GitHub 上传清单

上传日期：2026-09-22
目标远程：`git@github.com:sbboys/r1a7-quest3-vr-teleop-demo.git`
目标分支：`codex/r1a7-arm-console`

## 上传内容

- `r1a7_threading_project/console3d/`
  - 8097 经典控制台服务端和前端；
  - Three.js、URDFLoader 本地依赖；
  - R1-A7、Dex1-1 URDF 和 STL 模型；
  - 中文运行说明。
- `r1a7_threading_project/console3d_hierarchical/`
  - 8098 稳定基线控制台；
  - 终端基线参数封装；
  - 明确停止运动、失焦停止和断开流程；
  - 中文运行说明。
- `r1a7_threading_project/transfer_control/experimental/`
  - 网页整行命令兼容控制副本；
  - 历史分层 IK 实验实现；
  - 不上传 `__pycache__`。
- `r1a7_threading_project/transfer_control/tools/r1a7_cartesian_stream_control.py`
  - 当前终端控制入口，用于对照和 8097 运行。
- `r1a7_threading_project/transfer_control/tools/r1a7_cartesian_hierarchical_control.py`
  - 历史分层 IK 注入入口，当前 8098 不使用。
- `r1a7_threading_project/transfer_control/tools/r1a7_lowcmd_guard.py`
  - 已在基础分支中跟踪，用于防止多个 LowCmd 发布者并存。
- `r1a7_threading_project/transfer_control/tools/visual_servo_receiver.py`
  - 当前终端控制入口的必要 UDP 接收器依赖；不包含上层视觉任务实现。
- `r1a7_threading_project/docs/r1a7_arm_console_work_summary_2026-09-22_zh.md`
  - 从创建到完成的工作总结、问题处理和当前边界。
- 本上传清单。
- `r1a7_threading_project/README.md`
  - 增加控制台文档入口。

## 不上传内容

- `/tmp/r1a7_cartesian_trace_*.csv`、LowCmd 日志和现场运行日志；
- `__pycache__`、`.pyc` 和浏览器缓存；
- 带 `before_*`、`backup`、`.bak` 等名称的历史备份脚本；
- Task6/Task7 孔位视觉、双目视觉伺服、相机调试等上层任务代码（仅保留终端入口必须导入的 `visual_servo_receiver.py`）；
- 训练数据、视频、checkpoint、conda 环境和本机绝对路径下的外部 SDK；
- 工作树中与控制台无关的 `.gitignore` 修改。

## 运行依赖

- Ubuntu 工作站；
- `/home/robot/miniconda3/envs/tv`；
- `numpy`、`aiohttp`、宇树 SDK2 Python；
- 官方/现有 `R1A7_ArmIK` 运行目录；
- 机器人 DDS 网卡，当前示例为 `enp6s0` / `192.168.123.223`；
- R1-A7 已切换到允许 `rt/lowcmd` 的 debug mode。

外部 SDK 和 conda 环境不复制到 GitHub，仓库仅保存调用方、控制台资源和复现说明。

## 上传前检查

```bash
git status --short
git diff --cached --stat
git diff --cached --check
```

确认暂存区只包含本清单列出的控制台相关文件后再提交和推送。
