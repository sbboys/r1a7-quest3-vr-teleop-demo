# R1-A7 细线夹取与穿孔项目当前对话整理

整理时间：2026-09-14 23:10 CST

本文档整理当前 Codex 对话中与 R1-A7 细线夹取、镊子绑定、VR 示教、相机接入和后续 GitHub 上传相关的阶段性内容。后续用户会继续补充 GPT 中这几天的工作内容，届时再统一合并为完整项目工作记录并准备上传 GitHub。

## 1. 项目目标

本项目目标是使用 R1-A7 双臂人形机器人和 Dex1-1 两指夹爪，将镊子固定到夹爪上，使夹爪开合能够带动镊子同步开合；再利用镊子夹取细线，引导细线穿过孔，实现类似人工穿针引线的任务。

当前任务重点：

- 完成右夹爪与镊子的机械绑定。
- 标定夹爪开合量与镊子夹线能力之间的关系。
- 通过 VR 手动操作验证夹取细线和穿孔流程。
- 记录夹爪闭合状态、VR 输入和相机画面，为后续自主识别与学习控制打基础。
- 明确 R1-A7 双臂人形机器人是否适合使用头部相机、腕部相机或外部相机进行视觉伺服。

## 2. 项目隔离与环境决策

为了避免污染之前的 ACT/扳手项目，已经创建独立 worktree：

```text
/home/robot/unitree_sim_isaaclab_threading
```

细线项目目录：

```text
/home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project
```

当前原则：

- 不在原 `r1a7_wrench_project` 中继续混入细线任务逻辑。
- 第一阶段先复用已有 conda 环境，不新建环境。
- `tv` 环境用于 R1-A7 控制、VR/TeleVuer、LowCmd、相机相关运行。
- `aloha` 环境用于 ACT 或后续模仿学习训练、推理。
- 后续只有在依赖冲突或需要升级核心库时，再克隆新环境，例如 `tv_threading`、`aloha_threading`。

已保存环境复用说明：

```text
/home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project/docs/environment_reuse_notes_zh.md
```

## 3. 镊子与夹爪绑定阶段

用户当前操作策略：

- 先将镊子固定到右夹爪。
- 初始状态定义为：夹爪刚与镊子接触，镊子尚未明显闭合。
- 后续夹爪继续合拢时，镊子尖端应同步合拢。
- 因为无法直接从机器人状态获取镊子尖端间距，所以当前采用“夹爪位置 + 人工观察夹线效果”的方式建立标定表。

已有标定文档：

```text
/home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project/docs/gripper_tweezer_binding_notes_zh.md
```

重要记录：

- 第一次接触 baseline：右夹爪 `q ~= 0.5565`。
- 后续用户重新调整接触位置，更新接触 baseline：右夹爪 `q ~= 0.3858`。
- 进入 VR 前另一次记录中右夹爪约为 `q ~= 0.1619`。

当前认识：

- 镊子尖端间距目前不能由机器人直接计算得到。
- 若要精确间距，需要外部测量方式，例如卡尺、显微标尺、相机标定板或固定比例尺。
- 第一阶段更实用的指标是功能性结果：能否夹住细线、是否伤线、镊子是否滑动、是否变形、是否遮挡相机。

## 4. 夹爪开合与 VR 控制记录

细线项目专用 VR 控制脚本：

```text
/home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project/transfer_control/tools/r1a7_threading_vr_lowcmd.py
```

用户现场发现的问题：

- 启动 VR 后，夹爪会立刻恢复到张开状态。
- 这会破坏“夹爪刚接触镊子”的初始位置。

处理思路：

- 启动时使用当前实测夹爪位置作为初始目标。
- 右 trigger 松开时，右夹爪最多只能回到启动时的当前开度，不允许回到完全张开。

相关参数：

```text
--gripper-initial-mode current
--right-gripper-open-cap-current
```

VR 记录字段已经扩展，用于记录操作过程中的夹爪状态：

```text
measured_gripper_q
measured_gripper_dq
goal_gripper_q
sent_gripper_q
gripper_contact_hold
gripper_contact_q
left_trigger
right_trigger
```

这样后续可以复查：

- 操作者 trigger 输入。
- 右夹爪实际位置。
- 右夹爪目标位置。
- 夹爪是否按当前接触位进行保持。
- 镊子是否随夹爪闭合而闭合。

## 5. VR 启动与通信排查

曾使用的 VR 控制启动命令形式：

```bash
cd /home/robot/unitree_sim_isaaclab_threading

env -u PYTHONPATH \
PYTHONNOUSERSITE=1 \
AIOHTTP_NOSENDFILE=1 \
XR_TELEOP_ROOT=/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/xr_teleoperate \
/home/robot/miniconda3/envs/tv/bin/python -u \
r1a7_threading_project/transfer_control/tools/r1a7_threading_vr_lowcmd.py \
  --interface enp6s0 \
  --domain-id 0 \
  --host-ip 192.168.1.111 \
  --ik-frequency 30 \
  --publish-frequency 250 \
  --max-joint-speed 0.5 \
  --gripper-initial-mode current \
  --right-gripper-open-cap-current \
  --record-root /data/R1A7/threading_pilot_v1 \
  --record-episode-id threading_vr_003 \
  --no-record-d435i
```

曾遇到的问题：

```text
TCP port 8012 is already in use
```

原因：

- 已有一个 `r1a7_threading_vr_lowcmd.py` 进程占用了 `8012`。

处理方式：

```bash
ps -eo pid,ppid,stat,comm,args | grep -E 'r1a7_threading_vr_lowcmd|8012' | grep -v grep
kill <PID>
ss -ltnp | grep ':8012' || echo '8012 free'
```

当前网络相关信息：

- DDS 控制网口：`enp6s0`
- 控制网段 PC IP：`192.168.123.223/24`
- R1-A7 控制 IP：`192.168.123.161`
- Quest/浏览器访问曾使用：`192.168.1.111:8012`
- PC 同时可能存在：
  - `192.168.1.111`
  - `192.168.1.114`

## 6. 头部相机与视觉伺服判断

用户确认：R1-A7 是双臂人形机器人，不是 R1 七轴机械臂。因此视觉方案不能简单套用单机械臂固定基座视觉伺服，需要考虑人形机器人双臂、头部相机、腕部相机和外部相机的配合。

当前判断：

- 头部相机适合作为全局观察或任务初始化感知。
- 对细线夹取、镊子尖端、孔位对准这类毫米级任务，仅依赖头部 RGB 相机不够稳。
- 精细穿孔阶段更适合使用腕部 RGB 相机或外部近距离相机。
- 没有深度图时，可以先采用 2D 视觉伺服、平面约束、固定工作台高度、人工标定孔位和镊子尖端位置。
- 后续如果需要更可靠的 3D 定位，再引入双目、深度相机或多视角几何。

## 7. 机器人网络相机排查

曾根据官方 TeleImager 说明检查过如下端口：

- WebRTC：
  - 头部：`60001`
  - 左腕：`60002`
  - 右腕：`60003`
- ZMQ：
  - 头部：`55555`
  - 左腕：`55556`
  - 右腕：`55557`

对 `192.168.1.115` 的检查结果：

```text
60000 closed
60001 closed
60002 closed
60003 closed
55555 closed
55556 closed
55557 closed
```

结论：

- 机器人 LAN 地址当前未开放 TeleImager 腕部图传服务。
- 当前可用路径不是通过网络打开本体腕部 RGB 相机，而是通过 USB 直连上位机识别为 UVC 摄像头。

## 8. GitHub 旧记录查找结果

当前仓库远端：

```text
demo git@github.com:sbboys/r1a7-quest3-vr-teleop-demo.git
origin https://github.com/unitreerobotics/unitree_sim_isaaclab.git
```

GitHub 远端 `demo` 上存在相关分支：

```text
r1a7-camera-debug-runbook
r1a7-wrench-baseline
codex-handoff-20260731
main
```

之前关于腕部相机打开的关键文档：

```text
/home/robot/unitree_sim_isaaclab/docs/r1a7_camera_debug_runbook_zh.md
```

旧记录结论：

- 之前成功打开的腕部相机不是通过机器人 APP/AP/网络图传打开。
- 当时是通过 USB 直连上位机，作为 UVC 摄像头打开。
- 当时识别为：
  - `JR0001`
  - `JR0002`
  - `QinHeng Electronics USB2.0 HUB`

## 9. 当前已成功重新检测并打开本体 RGB 腕部相机

用户重新连接后，重新检测 USB 和视频节点，已经识别到本体 RGB 腕部相机。

当前 `lsusb` 中出现：

```text
1a86:809f QinHeng Electronics USB2.0 HUB
0001:0001 Fry's Electronics JR0001
0002:0002 Ingram passport00
1a86:5395 QinHeng Electronics USB 10/100 LAN
1a86:55e7 QinHeng Electronics UART+SPI+I2C+JTAG
1a86:80b5 QinHeng Electronics USB Multiple Card Reader_V1.0
```

当前 `v4l2-ctl --list-devices` 中对应节点：

```text
JR0001: /dev/video8 /dev/video9
JR0002: /dev/video16 /dev/video17
```

采集测试结果：

```text
/dev/video8   OK
/dev/video9   不是捕获设备
/dev/video16  OK
/dev/video17  不是捕获设备
```

因此当前两个可用采集节点是：

```text
JR0001: /dev/video8
JR0002: /dev/video16
```

已启动的两路预览命令：

```bash
gst-launch-1.0 v4l2src device=/dev/video8 ! \
  image/jpeg,width=640,height=480,framerate=30/1 ! \
  jpegdec ! videoconvert ! autovideosink sync=false
```

```bash
gst-launch-1.0 v4l2src device=/dev/video16 ! \
  image/jpeg,width=640,height=480,framerate=30/1 ! \
  jpegdec ! videoconvert ! autovideosink sync=false
```

两路均进入 `PLAYING` 状态，说明本体 RGB 腕部相机已经打开。

注意：

- 当前 `/dev/video8` 和 `/dev/video16` 才是本体 RGB 腕部相机采集节点。
- `/dev/video0`、`/dev/video2` 当前属于 Orbbec Gemini，不应按旧节点直接使用。
- 以后每次插拔 USB 后，视频节点编号可能变化，应先用 `v4l2-ctl --list-devices` 确认。

## 10. 当前阶段结论

截至当前对话，已经完成：

- 明确细线项目独立于原 ACT/扳手项目。
- 明确第一阶段复用 `tv` 和 `aloha` 环境。
- 建立镊子绑定、夹爪 contact baseline 和相对闭合测试思路。
- 明确 VR 控制中夹爪初始状态需要保持当前接触位。
- 明确 VR 记录需要保存夹爪实际状态、目标状态和 trigger 输入。
- 完成头部相机、网络图传和 TeleImager 端口排查。
- 查回 GitHub 中之前关于腕部相机 USB 打开的旧记录。
- 当前已经重新识别并打开两个本体 RGB 腕部相机：
  - `/dev/video8`
  - `/dev/video16`

## 11. 后续待用户补充 GPT 内容后继续整理

用户后续会发送 GPT 中这几天的工作内容。收到后需要继续完成：

1. 将 GPT 工作内容与本 Codex 对话记录合并。
2. 按时间线整理完整工作日志。
3. 按模块整理：
   - 项目创建与环境配置。
   - 镊子绑定与夹爪标定。
   - VR 控制与数据记录。
   - 相机接入与视觉方案。
   - 研究方案与 PPT 汇报。
   - 当前阻塞点和下一步实验计划。
4. 检查是否需要补充脚本、README、操作手册或实验记录表。
5. 生成可上传 GitHub 的最终文档集。
6. 确认目标远端和分支后，再执行 GitHub 上传。

## 12. 下一阶段建议

短期建议先完成以下实验闭环：

1. 固定镊子到右夹爪，确认不会滑动、旋转、干涉相机或碰撞线缆。
2. 保持 `/dev/video8` 和 `/dev/video16` 两路腕部 RGB 相机开启。
3. 使用 VR 控制右夹爪，从当前 contact baseline 缓慢闭合。
4. 记录镊子是否能稳定夹住细线、是否伤线、是否会把线推走。
5. 保存一次完整 VR episode，包括低层状态、夹爪状态、trigger 输入和相机画面。
6. 根据成功片段抽取关键帧，作为后续自主夹线和穿孔策略的第一版参考。
