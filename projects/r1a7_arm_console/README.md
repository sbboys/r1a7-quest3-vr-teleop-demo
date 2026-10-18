# R1-A7 手臂控制台

## 功能

提供 R1-A7 机器人三维模型、视角控制、笛卡尔方向控制、单关节控制、夹爪控制、连接参数和运行日志。

## 代码位置

- 主要控制台：`r1a7_threading_project/console3d/`
- 分层控制台实验：`r1a7_threading_project/console3d_hierarchical/`
- 3D 模型：`r1a7_threading_project/console3d/model/`
- 前端：`r1a7_threading_project/console3d/static/`
- 服务端：`r1a7_threading_project/console3d/server.py`
- 方向切换修正：`r1a7_threading_project/docs/r1a7_arm_console_direction_switch_correction_2026-09-30_zh.md`
- 2026-10-12 控制与六路相机完善记录：`r1a7_threading_project/docs/r1a7_console_control_camera_suite_2026-10-12_zh.md`
- 2026-10-18 实验与控制台连续更新：`r1a7_threading_project/docs/r1a7_experiment_console_update_2026-10-18_zh.md`

## 当前状态

当前 8098 控制台使用与独立终端基线一致的 30 Hz IK / 250 Hz LowCmd 控制器，
支持左右臂末端、双夹爪，以及双腕、外部、双奥比中光和头部相机控制。左右臂
肩俯仰 Kd、DQIK/POSVEL、重力前馈接口和腕肘软姿态约束已经做对称化；真实左臂
低速运动仍需在唯一 LowCmd 发布者和急停可用条件下验收。
USB 相机节点按序列号动态识别，不使用固定 `/dev/videoN`。

UVC 五路预览已有实际启停记录；头部相机已按官方 Stereo patch PC1 路径接入
`192.168.123.164:5001` 的 RTP/H.264 流，并保留原服务状态恢复，但最近一次在线
检查仍未收到首帧。真机左臂、右夹爪最大闭合端点和视觉伺服执行层尚未在本次
网页更新中完成硬件验收。

控制台是操作入口，不等于底层控制算法。修改底层手臂控制脚本前，应先确认项目文档的保护边界。

