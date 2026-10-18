# R1-A7 手臂与视觉控制台完善记录（2026-10-12）

> 本文保留 2026-10-12 当时状态用于追溯。后续频率、视觉授权、分层速度、左右臂
> 对称控制、夹爪端点和头部相机路线已经更新；当前说明以
> [2026-10-18 实验工作与控制台更新报告](r1a7_experiment_console_update_2026-10-18_zh.md)
> 为准。

## 1. 目标与入口

本次在原 8098 网页控制台上完成三类衔接：

1. 把机器人基本控制更新到 2026-10-10 的低速控制状态；
2. 恢复双夹爪和左右臂末端选择；
3. 统一双腕、外部、双奥比中光和头部相机的检测与预览控制。

当前代码入口：

- 服务：`r1a7_threading_project/console3d_hierarchical/server.py`
- 前端：`r1a7_threading_project/console3d_hierarchical/index.html`
- 相机管理：`r1a7_threading_project/console3d_hierarchical/camera_manager.py`
- 桌面启动脚本：`r1a7_threading_project/console3d_hierarchical/start_console_8098.sh`
- 地址：`http://127.0.0.1:8098/`

## 2. 控制链更新

控制台不再调用旧的 `r1a7_cartesian_stream_control_experimental.py`，改为：

`transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py`

固定参数如下：

| 项目 | 当前值 | 说明 |
|---|---:|---|
| IK 频率 | 30 Hz | 与操作员验证通过的终端命令默认值一致 |
| LowCmd 频率 | 250 Hz | 与操作员验证通过的终端命令默认值一致 |
| 末端默认速度 | 10 mm/s | 已有记录的验证速度 |
| 末端默认加速度 | 20 mm/s² | 与终端脚本默认值一致 |
| 最大关节速度 | 2.0 rad/s | 与终端脚本默认值一致 |
| 右臂重力前馈 | 0.0 | 未完成全局资格验证，保持关闭 |
| D435i 自动记录 | 关闭 | 避免控制台连接自动产生数据 |

新增/恢复的命令：

- `cartesian-arm:left/right`：左右臂末端控制选择；
- `gripper:left/right:open/hold/close`：双夹爪张开、保持和闭合；
- 网页“连续切换”发送的 `jog:start:<方向>` 在运动中改向时只更新方向，保留
  当前笛卡尔目标、关节流、工具姿态和速度状态，不再重新锚定到实测位姿；
- `jog:speed:<1..12>` 与 `jog:accel:<5..40>`：控制器运行中实时更新末端
  速度和加速度上限，修改网页数值后无需断开重连；
- 夹爪从实测当前位置初始化，右侧保留启动位置开度上限和接触保持；
- 窗口失焦、页面隐藏、空格和 `Esc` 继续发送 `jog:stop`。

## 3. 相机动态识别

相机节点不写死。每次启动预览前使用 udev 序列号和 V4L2 格式重新枚举，
并排除无彩色格式的深度/元数据节点。

本次工作站检测结果：

| 视角 | 稳定身份 | 本次节点 | 能力/状态 |
|---|---|---|---|
| 左腕 | `JR0001` | `/dev/video18` | MJPEG/YUYV，640×480@30 |
| 右腕 | `JR0002` | `/dev/video20` | MJPEG/YUYV，640×480@30 |
| 外部 | `20685204b12d5283` | `/dev/video16` | MJPEG/H264/HEVC/YUYV |
| 奥比中光 007Z | `CPCBC530007Z` | `/dev/video6` | Gemini 336L 彩色节点 |
| 奥比中光 00C5 | `CPCBC53000C5` | `/dev/video14` | Gemini 336L 彩色节点 |
| 头部 | Unitree VideoClient | DDS `enp6s0` | 独立视觉测量窗口 |

USB 相机优先采用 MJPEG 640×480@30，支持单路启停、双腕成组启停、双奥比
成组启停和全部停止。停止逻辑只向本控制台创建的进程组发送信号，不清理
其他程序占用的相机。

### 3.1 细线穿孔识别与只读引导（2026-10-13）

控制台相机区域新增“启动穿孔识别”，接入当前 Task 6 链路：

- External V4：外部相机识别镊尖、目标孔并计算 X/Z 建议量；
- Wrist SAM2：右侧 Gemini 336L（`CPCBC53000C5`）识别镊尖和目标孔；
- Dual Fusion：融合两路状态，显示 `DUAL`、`EXTERNAL_ONLY`、
  `WRIST_ONLY`、`BLIND_GRACE` 或 `STOP`，并在网页显示建议 XYZ 速度。

启动时按外部相机序列号动态解析 `/dev/video*`，并只停止本控制台创建的外部
相机、右侧 Gemini 普通预览，避免同一设备被两个进程重复占用。识别窗口中的
操作仍遵循原研究流程：完成镊尖初始化后选择目标孔。

默认使用 `tv_sam2` Python 环境和现有研究工作区中的 SAM2 模型文件；部署到
其他机器时可分别用 `R1A7_SAM2_PYTHON`、`R1A7_SAM2_ROOT` 和
`R1A7_SAM2_CHECKPOINT` 指定运行时、源码目录与模型权重，不需要修改控制台代码。

控制台强制以 `--shadow-mode` 启动融合，网页只显示建议速度，输出 JSON 中
`enabled=false`，不会因为“开启相机”直接驱动机械臂。

### 3.2 右臂视觉伺服安全门（2026-10-13）

在操作员确认镊尖和目标孔能够完成初始化后，控制台增加独立的“按住执行视觉
伺服”死手授权。视觉融合原文件仍保持影子模式，控制台安全门读取其中的建议
速度，验收全部条件后写入独立文件
`/tmp/r1a7_console_ibvs_velocity.json`，机器人控制器只读取该文件。

安全门同时要求：

- 机器人控制器已连接并完成 LowCmd 授权；
- 控制器为右臂笛卡尔模式；
- 控制遥测不超过 `1.0 s`、手动点动已经停止，末端跟踪误差不超过
  `15 mm`；
- External、Wrist 和 Fusion 三个视觉进程全部运行；
- External 与 Wrist 两路都有效，融合处于 `DUAL` 或双路优先模式；
- 融合时间戳不超过 `0.20 s`；
- 操作员持续按住网页按钮，浏览器每 `0.25 s` 刷新授权，授权有效期仅
  `0.75 s`；
- 建议速度为有限数值，并按三维合速度限制在 `2.0 mm/s`。

松开按钮、页面失焦/隐藏、切换手臂、点击停止、任一路相机失效、融合转入
单相机或盲动模式、控制器断开，都会把安全门输出改为 `enabled=false`。控制器
内部还会再次检查命令时间戳（`0.20 s`）和 `2.0 mm/s` 限速；即使网页服务
异常退出，旧命令也不能持续驱动。旧 UDP 视觉增量入口在控制台启动路径中关闭，
避免两套视觉命令叠加。控制台路径同时关闭原 `1.0 mm/s` 最小速度抬升，保留
融合算法接近孔位时输出的小速度，减少近目标振荡风险。

本次代码和失效保护已完成离线/接口验证。真实机械臂运动仍应先在急停可用、
工作区清空的条件下做短时按住测试，再根据实测方向和振荡情况决定是否进入完整
穿孔过程。

## 4. 验收结果

已完成：

- Python 编译检查；
- 相机序列号选择和彩色节点选择单元测试；
- 双臂选择、夹爪和关节命令白名单测试；
- 末端实时速度/加速度命令范围测试；
- 页面加载、六张相机卡片渲染和浏览器脚本错误检查；
- 左腕单路实际启动/停止；
- 双腕两路同时启动/停止；
- 两台 Gemini 彩色流同时启动/停止；
- 外部 USB ZOOM 相机启动/停止；
- 停止后使用 `fuser` 确认上述 `/dev/video*` 节点已释放。

头部相机当前结果：控制进程和 DDS VideoClient 可以启动、停止，但本次在线
检查未收到图像，日志为 `[ClientStub] send request error`。界面会明确显示
“等待图像”和错误提示，不把“进程仍在运行”误报为“图像正常”。后续需要
在机器人头部视频服务在线时复测。

2026-10-12 补充：宇树官方 SDK 仍提供
`unitree_sdk2py.go2.video.VideoClient.GetImageSample()`，项目在 2026-09-30
的 GitHub 工作记录中也已使用该路径收到 1280×720 RGB。官方新版
`xr_teleoperate` 另推荐由 PC2（默认 `192.168.123.164`）运行 TeleImager，
但本机当前无可达 PC2 图像服务，因此控制台仍保留已验证的
VideoClient 主路径，不将未验证的 TeleImager 标记为可用。

启动头部相机时，控制台只确保 `video_hub` 运行，不再关闭
`videohub_rtsp_server` / WebRTC App 链路。启动后 12 秒仍没有收到首帧时，
界面显示“未收到画面”，避免把 DDS 子进程存活误判为相机就绪。

核对来源：

- [Unitree SDK2 VideoClient 官方接口](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/go2/video/video_client.hpp)
- [Unitree SDK2 官方 GetImageSample 示例](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_video_client.cpp)
- [xr_teleoperate 官方实机图像服务流程](https://github.com/unitreerobotics/xr_teleoperate#31-图像服务)
- [本项目已验证的头部相机控制台提交](https://github.com/sbboys/r1a7-quest3-vr-teleop-demo/commit/e627515e0c54c863562c8bb20f5a00da177c797f)

补充网络核验：`enp6s0` 为 `192.168.123.223/24`，机器人
`192.168.123.161` 连续 ping 成功（约 0.34 ms），邻居表也存在机器人 MAC。
因此本次头部图像失败不是网卡未启用或三层路由中断，现象与既有相机手册中
“机器人端未向电脑提供可访问视频帧”的结论一致。控制台只通过
官方 `RobotStateClient` 确保已知的 `video_hub` 运行，不操作其他机器人服务。

## 5. 真机验证边界

切换桌面入口时检测到页面已存在一次操作员授权的真实连接。遥测显示末端速度
为 `0.0 mm/s`、运动状态为位置保持。迁移前先发送
`jog:stop`，再由正常退出命令释放 LowCmd，控制器返回码为 0。

该连接验证了控制器能接入真机并零速保持，但没有执行方向点动或夹爪开合，
因此左右臂运动和双夹爪仍需在急停可用、工作区清空、确认唯一 LowCmd 控制源
后逐项低速验收。15/25 mm/s 不属于当前合格范围。

2026-10-12 晚间根据操作员复测反馈修正了网页连续切向仍重新锚定、运行中修改
速度未下发的问题。代码、命令解析和浏览器加载已离线验证，服务已更新；修正后
的真实切向平滑度和 1–12 mm/s 各档实际速度仍以操作员下一轮低速真机验收为准。

2026-10-13 再次对照操作员验证正常的终端启动命令后发现，旧网页服务曾额外把
IK/LowCmd 频率改为 `100/500 Hz`、最大关节速度改为 `0.8 rad/s`、默认末端
加速度改为 `30 mm/s²`，因此并非真正同基线。现已撤销这些覆盖，恢复脚本默认的
`30/250 Hz`、`2.0 rad/s`、`20 mm/s²`；仅保留网页命令适配、可调参数和
`--no-record-d435i`（防止控制台操作意外启动数据录制）。
