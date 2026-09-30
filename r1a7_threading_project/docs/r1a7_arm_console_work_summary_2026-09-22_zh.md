# R1-A7 机器人手臂控制台工作总结

更新时间：2026-09-30（补充方向切换纠错）
项目目录：`r1a7_threading_project`
GitHub 分支：`codex/r1a7-arm-console`

## 1. 工作目标

为宇树 R1-A7 双臂人形机器人制作一个工业机械臂风格的软件控制台，使操作者可以在同一页面完成：

- 连接并授权真机 LowCmd 控制；
- 控制末端沿 Base 坐标系 `±X / ±Y / ±Z` 点动；
- 控制左右手臂各个关节正向或反向点动；
- 控制左右 Dex1-1 夹爪，并在数字孪生中显示开合；
- 查看 R1-A7、双臂、夹爪、末端目标、实际状态和运行日志；
- 在长按、松手、页面失焦和异常情况下可靠停止运动并保持当前位置。

该工作以现有 Python、DDS、IK 和 `rt/lowcmd` 控制链为基础，没有把核心控制循环迁移到 C++。

## 2. 最终软件组成

### 2.1 经典控制台（8097）

目录：`r1a7_threading_project/console3d/`

- Three.js + URDFLoader 显示 R1-A7 和左右 Dex1-1。
- 支持末端点动、14 个手臂关节点动和双夹爪控制。
- 使用 `transfer_control/tools/r1a7_cartesian_stream_control.py`。
- 保留完整控制功能，但该终端脚本在 2026-09-21 后继续加入视觉伺服和孔位接近功能，使用前必须重新做真机回归。

### 2.2 稳定基线控制台（8098，当前推荐）

目录：`r1a7_threading_project/console3d_hierarchical/`

目录名称来自早期分层 IK 实验，但最终运行路径已经改回稳定的原加权 IK。页面标题为“R1-A7 手臂控制台”。

当前后台固定参数：

- 控制入口：`transfer_control/experimental/r1a7_cartesian_stream_control_experimental.py`；
- IK：原加权 IK，不注入分层 IK；
- 右臂重力补偿：`0.25`；
- 默认末端速度：`10 mm/s`；
- 默认末端加速度：`30 mm/s²`；
- 不传 `--max-joint-speed`，使用控制器默认 `2.0 rad/s`；
- 夹爪暂时禁用，优先验证手臂轨迹；
- 网页 stdin 使用整行命令，保留 `jog:stop`、`joint:*` 等结构化命令。

## 3. 最终控制链

```text
浏览器按钮/键盘
    │
    ├─ 短按：单步命令
    ├─ 按住点动：jog:start + 150 ms keepalive，松手 jog:stop
    ├─ 连续切换：新的 jog:start 直接替换活动方向，不插入 jog:stop
    └─ 明确停止：停止按钮/空格/Esc/失焦发送 jog:stop
    │
aiohttp HTTP + WebSocket 服务
    │ stdin 整行命令
网页兼容控制进程
    │ 30 Hz 实测状态 + IK
250 Hz LowCmd 输出层
    │
rt/lowcmd → R1-A7 真机

rt/lowstate / 控制遥测
    └─ WebSocket → 数字孪生、关节角、TCP、误差和日志
```

同一时刻只允许一个 `rt/lowcmd` 发布者。网页控制台和直接终端控制不能同时 ENABLE。

## 4. 工作过程

### 4.1 六方向控制与初版网页

1. 根据 2026-09-15 的六方向运动记录，将终端键盘控制封装为本地 HTTP/WebSocket 服务。
2. 增加 `±X / ±Y / ±Z` 方向按钮、连接参数、授权和停止操作。
3. 将 R1-A7 URDF 导入 Three.js，数字孪生由真机 LowState 驱动。

### 4.2 单关节、夹爪和数字孪生

1. 增加左右臂切换和 14 个关节的 `+ / -` 点动。
2. 增加左右 Dex1-1 的张开、保持和闭合。
3. 将左右夹爪实际电机量程映射为 Dex1-1 滑动指关节开度。
4. 修正夹爪与左右腕部的安装偏置，使夹爪基座与手臂末端对齐。

### 4.3 界面整理

1. 将三维模型区改为方形，并调整到整个桌面界面约四分之一。
2. 将连接、末端点动、单关节点动、夹爪和日志安排在同一屏。
3. 优化中文状态名称，使界面接近封装软件，而不是调试网页。

### 4.4 连续运动与停止逻辑

初版连续点击会形成待执行命令积压，表现为机器人一点一点移动、延迟和抖动。最终改为：

- 短按执行有限步长；
- 按住超过约 `180 ms` 进入连续速度模式；
- 浏览器每 `150 ms` 发送 keepalive；
- “按住点动”模式松手发送 `jog:stop`；
- “连续切换”模式点击另一方向时直接发送新的 `jog:start:<direction>`，保持目标、姿态参考、IK 和速度状态连续；
- 页面失焦、页面隐藏、空格键、`Esc` 或“停止运动”按钮都发送 `jog:stop`；
- “断开连接”先停止运动，再退出控制进程并释放 LowCmd 锁。

这说明主要问题在命令协议、轨迹生成和 IK 连续性，而不是 Python 语言本身；仅改成 C++ 不能自动解决轨迹跳变和奇异位形。

## 5. 主要问题与处理结果

| 问题 | 原因 | 处理结果 |
|---|---|---|
| 页面没有显示 | 服务未运行、端口或浏览器缓存问题 | 检查 8097/8098 监听，使用带版本参数的 URL 强制刷新 |
| 点击 ENABLE 没有作用 | 控制器未进入等待授权、已有 LowCmd 发布者或页面连接断开 | 增加状态显示，启动前检查进程、锁和 DDS 网卡 |
| 三维夹爪不随真机开合 | 只加载夹爪外形或量程映射不一致 | 加载 Dex1-1 可动关节并按左右实际量程归一化 |
| 夹爪与手臂没有固定好 | 腕部模型存在结构偏置 | 增加左右腕部独立安装补偿，仅影响显示 |
| 连续控制积压、延迟 | 每次点击都追加离散位置目标 | 改为 start/keepalive/stop 连续速度协议 |
| `+Z` 切换 `+X` 时瞬间抖动 | 原网页松开 `+Z` 先发送 `jog:stop`，随后 `+X` 重新 START；这与终端测试中不停止直接 `U→W` 的语义不一致 | 新增“连续切换”模式，方向变化只发送新的 `jog:start`，不在方向之间插入 `jog:stop` |
| 长按运动仍抖动 | IK 冗余解变化、接近奇异区、跟踪层叠加 | 保留实测状态驱动的 IK 和单一 LowCmd 输出链；不盲目提高增益 |
| `+X` 时手臂先下降，`+Z` 时先后缩 | 固定末端姿态、冗余 IK 路径和工作空间约束相互作用 | 通过 `+X+Z` 对比实验定位为 IK 路径问题；多次不理想修改已撤销 |
| 到一定位置只抖动、不再前进 | 肘部接近伸直或雅可比条件变差 | 显示构型条件和进度告警，使用 `-X` 退出边界，不把速度直接放大 |
| 8098 与终端腕部运动不同 | 8098 曾使用分层 IK、`0.8 rad/s` 和重力补偿 `0` | 8098 改回原加权 IK、默认 `2.0 rad/s` 和重力补偿 `0.25` |
| 松手后意外出现 `-X` | raw-key 解析把字符串 `jog:stop` 中的 `s` 当成方向键 | 网页控制副本检测 stdin 是否为 TTY；管道模式读取完整行命令 |
| 没有明确的软件停止键 | 原界面主要依赖松手和断开连接 | 新增“停止运动”、空格、`Esc`、窗口失焦和页面隐藏保护 |
| 终端按 `W` 没反应 | 最近一次日志显示命令已收到，但 `command_velocity_mm_s=0`，构型条件约 164，并出现 `TCP made no progress` | 先按 `S` 短距离退出前伸边界，再测试其他方向；每次用空格停止 |

## 6. 当前运行方法

### 6.1 启动推荐控制台

```bash
cd /home/robot/unitree_sim_isaaclab_threading

PYTHONNOUSERSITE=1 \
/home/robot/miniconda3/envs/tv/bin/python -u \
r1a7_threading_project/console3d_hierarchical/server.py \
--host 127.0.0.1 \
--port 8098
```

打开：`http://127.0.0.1:8098/?v=20260920-motion-stop`

### 6.2 操作顺序

1. 保持急停可用，人员和线缆离开双臂工作空间。
2. 点击“连接机器人”，等待“等待授权”。
3. 点击“授权控制”，确认变为“控制已连接”。
4. 首次只使用“按住点动”执行一个方向的短距离测试。
5. 验证方向切换时选择“连续切换”，点击“上 +Z”后不要停止，直接点击“前 +X”。
6. 确认日志出现 `SWITCH UP +Z -> FORWARD +X`，且中间没有 `STOP UP +Z`。
7. 需要保持当前位置时点击“停止运动”，或按空格/`Esc`。
8. 测试结束点击“断开连接”，确认控制进程退出。

“停止运动”只把速度归零并保持当前位置，控制器仍在线；“断开连接”会退出控制进程并释放 `rt/lowcmd`。

## 7. 文件结构

```text
r1a7_threading_project/
├── console3d/                         # 8097 经典全功能控制台
│   ├── server.py
│   ├── static/                        # UI、Three.js、URDFLoader
│   ├── model/                         # R1-A7 与 Dex1-1 模型
│   └── README_zh.md
├── console3d_hierarchical/            # 8098 稳定基线控制台
│   ├── server.py
│   ├── index.html
│   ├── app.js
│   └── README_zh.md
├── transfer_control/
│   ├── experimental/
│   │   ├── r1a7_cartesian_stream_control_experimental.py
│   │   └── r1a7_hierarchical_ik.py    # 历史实验，当前 8098 不使用
│   └── tools/
│       ├── r1a7_cartesian_stream_control.py
│       ├── r1a7_cartesian_hierarchical_control.py  # 历史入口
│       ├── r1a7_lowcmd_guard.py
│       └── visual_servo_receiver.py       # 当前终端入口的导入依赖
└── docs/
    ├── r1a7_arm_console_work_summary_2026-09-22_zh.md
    └── r1a7_arm_console_github_manifest_2026-09-22_zh.md
```

## 8. 验证状态

已验证：

- Python 服务端和 JavaScript 前端语法检查通过；
- 8098 页面、模型、WebSocket、状态机和日志可正常加载；
- 控制进程可进入 `waiting_enable`；
- 网页结构化命令不会再被 raw-key 模式拆成单字符；
- `jog:stop` 后遥测进入 `holding`，速度为 `0.0 mm/s`；
- 8098 启动命令不再包含分层 IK 和 `--max-joint-speed 0.8`。
- 2026-09-30 真机确认：“连续切换”模式执行 `+Z→+X` 后右臂恢复正常，方向之间不再因网页自动 STOP/START 产生瞬时抖动。

仍需继续验证：

- 在相同初始姿态下录制终端基线与 8098 的 `+X / +Z / +X+Z` 对照轨迹；
- 对比肩、肘、腕目标曲线和 TCP 轨迹，而不是只看肉眼姿态；
- 在远离奇异区的位置验证连续速度和停止距离；
- 夹爪重新启用前单独验证镊子接触保持和开度标定。

## 9. 版本边界与已知限制

1. 8098 使用 2026-09-20 的网页兼容控制副本；2026-09-30 修正了网页方向切换时序。这里的“稳定”只表示控制入口和参数固定，不表示所有真机操作路径已经完成验证。
2. `tools/r1a7_cartesian_stream_control.py` 在 2026-09-21 新增视觉伺服、IBVS 文件速度输入和标记孔位 `T` 命令，已不再与 8098 副本逐行一致；`visual_servo_receiver.py` 仅作为该入口的必要导入依赖随仓库保存。
3. 当前没有把 2026-09-21 的视觉功能合并进网页控制副本，因为这些功能尚未完成控制台真机回归。
4. 8098 暂时关闭夹爪控制；页面中的夹爪区域用于保留数字孪生结构。
5. “按住点动”适合单方向短距离操作；需要连续改变方向时必须选择“连续切换”，否则松开按钮会按设计停止当前运动。
6. 软件停止不能替代实体急停、碰撞保护和现场监护。

## 10. 核心文件 SHA-256

```text
918bd2d57f673a674571db3034b074d3ed143cdd0fc402d257763832ba79ba40  console3d/server.py
585b6f02e6e913fd0f6201d41f6fe83b1f380b69acc8cee636d147a83ff11766  console3d/static/app.js
27e3e40291f7e899428f4b94eaa6e1a7d515f8dfa6d1588fa5b89302871b876b  console3d_hierarchical/server.py
c82345fe56937056e9e5f2b6bb688df84cffcf5730cb680c88c5306e7b0dbb48  console3d_hierarchical/app.js
0ab47242af97eaf3b7ec9407536b067786fd0f6cb25b4ce206563ecdeb93f8ba  transfer_control/tools/r1a7_cartesian_stream_control.py
e20ab931a32f07e77eacbb7be4808a9419ed2cccd85a4a7eab9b47659618c2ab  transfer_control/experimental/r1a7_cartesian_stream_control_experimental.py
```

## 11. 资料来源

- `2026-09-15_R1-A7机器人手臂六方向运动控制工作记录.docx`
- `R1-A7机器人手臂连续控制调试工作汇总_2026-09-19.docx`
- 2026-09-15 至 2026-09-22 的控制台开发与真机调试对话
- 当前仓库源码、运行进程、日志和 `/tmp/r1a7_cartesian_trace_*.csv` 诊断结果
