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

## 当前状态

当前 8098 控制台已衔接 2026-10-10 的 100 Hz IK / 500 Hz LowCmd 低速控制器，
支持左右臂末端、双夹爪，以及双腕、外部、双奥比中光和头部相机控制。
USB 相机节点按序列号动态识别，不使用固定 `/dev/videoN`。

UVC 五路预览已经完成实际启停验证；头部 VideoClient 启停已接入，但本次
在线检查未收到头部图像，需在机器人视频服务在线时复测。真机手臂与夹爪
尚未在本次网页更新中授权执行。

控制台是操作入口，不等于底层控制算法。修改底层手臂控制脚本前，应先确认项目文档的保护边界。

