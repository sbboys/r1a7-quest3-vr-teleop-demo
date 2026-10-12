# R1-A7 手臂与视觉控制台完善记录（2026-10-12）

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
| IK 频率 | 100 Hz | 2026-10-10 低速基线 |
| LowCmd 频率 | 500 Hz | 2026-10-10 低速基线 |
| 末端默认速度 | 10 mm/s | 已有记录的验证速度 |
| 末端默认加速度 | 30 mm/s² | 网页默认值 |
| 最大关节速度 | 0.8 rad/s | 网页手动控制保护值 |
| 右臂重力前馈 | 0.0 | 未完成全局资格验证，保持关闭 |
| D435i 自动记录 | 关闭 | 避免控制台连接自动产生数据 |

新增/恢复的命令：

- `cartesian-arm:left/right`：左右臂末端控制选择；
- `gripper:left/right:open/hold/close`：双夹爪张开、保持和闭合；
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

## 4. 验收结果

已完成：

- Python 编译检查；
- 相机序列号选择和彩色节点选择单元测试；
- 双臂选择、夹爪和关节命令白名单测试；
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

## 5. 真机验证边界

切换桌面入口时检测到页面已存在一次操作员授权的真实连接。遥测显示末端速度
为 `0.0 mm/s`、运动状态为位置保持；运行日志确认实际参数为 IK `100 Hz`、
LowCmd `500 Hz`、最大关节速度 `0.8 rad/s`、重力前馈 `0.0`。迁移前先发送
`jog:stop`，再由正常退出命令释放 LowCmd，控制器返回码为 0。

该连接验证了控制器能接入真机并零速保持，但没有执行方向点动或夹爪开合，
因此左右臂运动和双夹爪仍需在急停可用、工作区清空、确认唯一 LowCmd 控制源
后逐项低速验收。15/25 mm/s 不属于当前合格范围。
