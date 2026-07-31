# R1-A7 Quest 3 手柄遥控双臂与 Dex1 双夹爪 Demo 操作指南

版本：2026-07-31

本指南用于现场演示：

```text
Quest 3 左右手柄移动 -> R1-A7 左右双臂跟随运动
Quest 3 左右食指扳机 -> Dex1 左右夹爪开合
```

演示使用正式脚本：

```text
/home/robot/unitree_sim_isaaclab/tools/r1a7_vr_dual_arm_g1ik_real.py
```

当前已验证参数：

```text
arm_velocity_limit=5.0
max_joint_offset_rad=2.80
ik_delta_scale=1.35
ik_joint_scales=0.80,0.65,0.45,0.90,0.55,1,0.75,0.80,0.65,0.45,0.90,0.55,1,0.55
joint_limit_margin_rad=0.04
shoulder_pitch_low_margin_rad=0.18
stale_pose_timeout=0.5
frozen_pose_hold_timeout=0.12
rearm_hold_time=1.0
```

2026-07-31 最新交接状态、测试结论和未解决问题见：

```text
docs/r1a7_quest3_vr_handoff_2026-07-31_zh.md
```

## 1. 演示前准备

### 1.1 进入项目目录

所有命令默认在控制主机 `robot` 用户下执行：

```bash
cd /home/robot/unitree_sim_isaaclab
```

### 1.2 检测机器人控制网卡

当前已验证的机器人控制网卡为：

```text
enx9c69d37d0967
```

先确认该网卡存在并处于 UP 状态：

```bash
ip -br link show enx9c69d37d0967
ip -br addr show enx9c69d37d0967
```

期望能看到类似结果：

```text
enx9c69d37d0967 UP
enx9c69d37d0967 UP 192.168.123.xxx/24
```

如果网卡不存在，先查看当前所有网卡：

```bash
ip -br link
ip -br addr
```

如果网卡存在但没有 `192.168.123.xxx/24` 地址，可临时配置一个控制网段地址：

```bash
sudo ip addr add 192.168.123.127/24 dev enx9c69d37d0967
sudo ip link set enx9c69d37d0967 up
```

再次确认：

```bash
ip -br addr show enx9c69d37d0967
```

### 1.3 检测机器人网络连通性

当前机器人控制侧常用地址：

```text
192.168.123.223
```

检测连通性：

```bash
ping -c 3 192.168.123.223
```

能收到回复再继续。如果 ping 不通，先检查：

```text
1. 机器人是否已开机。
2. 控制网线是否接好。
3. enx9c69d37d0967 是否 UP。
4. 控制主机是否在 192.168.123.0/24 网段。
5. 是否接错到了 Wi-Fi 或其他网口。
```

查看到机器人网段的路由：

```bash
ip route | grep -E '192.168.123|enx9c69d37d0967' || true
```

期望至少看到 `192.168.123.0/24` 走 `enx9c69d37d0967`。

如果 `ping 192.168.123.223` 不通，不要继续启动 Demo。此时即使 Quest 页面能打开，双臂控制也无法稳定下发到机器人。

### 1.4 检测 Quest 3 访问地址

Quest 3 浏览器访问的是控制主机的 Wi-Fi 或局域网地址，当前已验证为：

```text
192.168.1.127
```

确认本机确实有该地址：

```bash
ip -br addr | grep -E '192.168.1.127|wlan|wl|enp|eth' || true
```

如果本机地址发生变化，启动命令里的：

```text
--host_ip 192.168.1.127
```

以及 Quest 浏览器地址都要同步改成新的控制主机 IP。

Quest 3 浏览器入口格式为：

```text
https://控制主机IP:8012/?ws=wss://控制主机IP:8012
```

当前 Demo 使用：

```text
https://192.168.1.127:8012/?ws=wss://192.168.1.127:8012
```

### 1.5 检查 DDS 与旧进程残留

Demo 启动前必须确认没有旧控制服务占用端口或继续发布命令：

```bash
ss -lntp | grep -E '8012|60000|60001' || true
ps -ef | grep -E 'r1a7_vr|quest_dex1|dex1_1_gripper_server|vuer|teleop' | grep -v grep || true
```

正常情况下以上命令无输出。

如果有残留进程，优先回到原终端按 `Ctrl+C` 停止。找不到原终端时，再根据 `ps` 输出的 PID 结束对应进程：

```bash
kill <PID>
```

结束后重新检查：

```bash
ss -lntp | grep -E '8012|60000|60001' || true
ps -ef | grep -E 'r1a7_vr|quest_dex1|dex1_1_gripper_server|vuer|teleop' | grep -v grep || true
```

### 1.6 检查 Dex1 夹爪串口

Dex1 服务需要能访问左右夹爪对应的 USB 串口。启动前先查看串口：

```bash
ls -l /dev/ttyUSB*
```

当前现场曾验证到：

```text
right Dex1: /dev/ttyUSB0
left Dex1:  /dev/ttyUSB3
```

实际串口号可能因重新插拔而变化，以 Dex1 服务启动日志为准。启动服务后应看到：

```text
Motor ID 0 Side right
Motor ID 1 Side left
Dex1-1 Gripper Server started.
```

如果 Dex1 服务已经运行，但后续 `/dev/ttyUSB*` 设备发生重新枚举，旧服务可能继续存在，却无法再正常提供 `rt/dex1/left/state` 和 `rt/dex1/right/state`。这时联合控制脚本会反复显示 `Waiting to subscribe dds...`。

判断方法：

```bash
ps -ef | grep -E 'dex1_1_gripper_server|r1a7_vr|vuer|teleop' | grep -v grep || true
ls -l --time-style=long-iso /dev/ttyUSB*
```

如果 `/dev/ttyUSB*` 的更新时间晚于 `dex1_1_gripper_server` 的启动时间，说明夹爪串口可能在服务启动后重新枚举。处理方法是先停止联合控制脚本，再重启 Dex1 服务。

如果 `/dev/ttyUSB*` 不存在，先检查 USB 线、供电和系统识别：

```bash
dmesg | tail -n 50
```

### 1.7 安全与姿态检查

现场安全：

```text
1. 清空机器人双臂和夹爪工作空间。
2. 不允许人员站在双臂运动范围内。
3. 准备好急停。
4. 桌面、线缆、夹具和演示物体不要靠近肩部、肘部和腕部极限区域。
5. 第一次进入 VR 后只做小幅慢速动作。
```

机器人姿态：

```text
1. R1-A7 双臂保持自然向下或自然中立姿态。
2. Quest 左右手柄也保持自然中立姿态。
3. 不要在手柄姿态很歪、很高或很远的位置进入 VR。
```

## 2. 启动 Demo

### 2.1 内走线 Dex1 夹爪说明

当前这台 R1-A7 的 Dex1 夹爪是内走线版本，夹爪作为机器人手臂末端的低层电机直接走 `rt/lowcmd` 控制：

```text
左夹爪：31
右夹爪：33
```

因此本 Demo 不需要单独启动 `dex1_1_gripper_server`，也不需要单独 Dex1 SDK 服务。夹爪会随联合控制脚本一起启动，Quest 左右食指扳机分别控制左右夹爪。

### 2.2 启动双臂与夹爪联合控制

打开终端 2：

```bash
cd /home/robot/unitree_sim_isaaclab
PYTHONNOUSERSITE=1 /home/robot/miniconda3/bin/conda run --no-capture-output -n tv \
python -u tools/r1a7_vr_dual_arm_g1ik_real.py \
  --interface enx9c69d37d0967 \
  --domain_id 0 \
  --state_topic rt/lowstate \
  --command_topic rt/lowcmd \
  --enter_debug_mode \
  --enable_gripper \
  --gripper_mode lowcmd \
  --lowcmd_gripper_indices 31,33 \
  --no-lowcmd_gripper_relative \
  --lowcmd_gripper_left_open_q 4.86 \
  --lowcmd_gripper_left_close_q -0.08 \
  --lowcmd_gripper_right_open_q 4.80 \
  --lowcmd_gripper_right_close_q -0.20 \
  --lowcmd_gripper_velocity_limit 8.0 \
  --lowcmd_gripper_kp 25.0 \
  --lowcmd_gripper_kd 0.8 \
  --host_ip 192.168.1.127 \
  --duration 0 \
  --g1_style_gains \
  --g1_style_velocity_clip \
  --arm_velocity_limit 5.0 \
  --hold_kp 10.0 \
  --hold_kd 0.8 \
  --max_joint_offset_rad 2.80 \
  --ik_delta_scale 1.35 \
  --ik_joint_scales 0.80,0.65,0.45,0.90,0.55,1,0.75,0.80,0.65,0.45,0.90,0.55,1,0.55 \
  --joint_limit_margin_rad 0.04 \
  --shoulder_pitch_low_margin_rad 0.18 \
  --stale_pose_timeout 0.5 \
  --frozen_pose_hold_timeout 0.12 \
  --rearm_hold_time 1.0 \
  --pose_change_eps 1e-4 \
  --limit_diag_eps 1e-3 \
  --print_period 0.5
```

终端出现安全确认后输入：

```text
ENABLE
```

成功启动后应看到：

```text
DDS initialized
IK: Unitree G1_29_ArmIK.solve_ik
lowcmd gripper trigger control: ENABLED
calibrated robot arm q
calibrated lowcmd gripper q
waiting_quest_pose
```

## 3. Quest 3 进入 VR

在 Quest 3 浏览器打开或刷新：

```text
https://192.168.1.127:8012/?ws=wss://192.168.1.127:8012
```

进入 VR 后，终端 2 日志应从：

```text
waiting_quest_pose
```

变为：

```text
calibrated_ik_zero
relative_tracking
```

出现 `relative_tracking` 后，Demo 正式进入手柄控制状态。

## 4. 演示操作方式

双臂控制：

```text
左手柄移动 -> 机器人左臂跟随
右手柄移动 -> 机器人右臂跟随
```

夹爪控制：

```text
左食指扳机松开 -> 左夹爪张开
左食指扳机按下 -> 左夹爪闭合
右食指扳机松开 -> 右夹爪张开
右食指扳机按下 -> 右夹爪闭合
```

推荐演示顺序：

```text
1. 左右手柄保持中立，确认机器人双臂不突然运动。
2. 小幅前后移动左手柄，展示左臂跟随。
3. 小幅前后移动右手柄，展示右臂跟随。
4. 左右手柄同时做慢速对称移动，展示双臂同步。
5. 左右食指扳机分别按下和松开，展示左右夹爪独立开合。
6. 双臂移动到安全抓取区域后，再演示夹爪闭合。
7. 退出 VR，再重新进入 VR，观察机器人保持当前位置并重新标定后继续控制。
```

演示时不要做：

```text
1. 快速甩动手柄。
2. 手柄突然大幅上举或横扫。
3. 让双臂靠近身体、桌沿、线缆或夹具。
4. 在机器人接近关节限位时继续强推手柄。
```

## 5. Demo 现场观察指标

终端 2 中重点观察：

```text
relative_tracking
cmd_err
tgt_err
dq
gripper connected=True
trigger_L / trigger_R
```

判断正常：

```text
1. 日志持续显示 relative_tracking。
2. 手柄移动时 left_cmd/right_cmd 连续变化。
3. 机器人双臂方向与左右手柄对应正确。
4. 夹爪 connected=True。
5. 按左右扳机时对应夹爪开合。
6. 退出 VR 后进入 stale_hold，不继续追旧手柄目标。
```

`dq=2.000` 表示动作达到当前相对关节限幅。演示中如果频繁顶到该值，应减小动作范围，不要继续大幅推动手柄。

## 6. 退出 VR 与重新进入

退出 VR 或关闭 Quest 浏览器后，程序应进入：

```text
stale_hold
tele_fresh=False
```

这时机器人双臂应保持当前真实关节位置，不再追退出前的旧手柄目标。

重新进入 VR 后，程序会先重新标定：

```text
fresh Quest poses
calibrated_ik_zero
relative_tracking
```

看到 `relative_tracking` 后再继续演示。不要在刚进入 VR 的瞬间快速挥动手柄。

## 7. 标准停止流程

终端 2 按 `Ctrl+C` 停止联合控制脚本。

必须看到：

```text
[R1-A7 VR G1IK REAL] released lowcmd gains
```

终端 1 按 `Ctrl+C` 停止 Dex1 服务。

最后检查残留：

```bash
ss -lntp | grep -E '8012|60000|60001' || true
ps -ef | grep -E 'r1a7_vr|quest_dex1|dex1_1_gripper_server|vuer|teleop' | grep -v grep || true
```

正常情况下无输出。

## 8. 异常处理

Quest 页面打不开：

```text
1. 确认终端 2 已启动并输入 ENABLE。
2. 确认 URL 为 https://192.168.1.127:8012/?ws=wss://192.168.1.127:8012。
3. 检查 8012 是否被旧进程占用。
4. 清理残留后重新启动 Demo。
```

机器人双臂没有反应：

```text
1. 看终端是否还停在 waiting_quest_pose。
2. 确认 Quest 已进入 VR，不只是打开页面。
3. 检查是否出现 relative_tracking。
4. 检查是否有旧 rt/lowcmd 发布进程。
5. 停止并重新启动联合控制脚本。
```

夹爪没有反应：

```text
1. 确认 Dex1 服务终端正在运行。
2. 确认联合脚本带 --enable_gripper。
3. 确认日志中 gripper connected=True。
4. 检查左右 USB 串口是否被其他进程占用。
```

反复出现 `Waiting to subscribe dds...`：

```text
含义：
Dex1_1_Gripper_Controller 正在等待 Dex1 服务端 DDS 订阅/状态链路建立。

常见原因：
1. Dex1 服务没有启动，或启动后没有成功打开串口。
2. Dex1 服务和联合控制脚本使用的网卡或 DDS Domain 不一致。
3. 旧的联合控制脚本没有停止，又重复启动了一个新的联合控制脚本。
4. 多个进程同时初始化 Dex1 控制器或发布 rt/lowcmd，导致现场状态混乱。
5. Dex1 服务启动后，/dev/ttyUSB* 发生重新枚举，旧服务没有重新打开当前串口。

处理：
1. 立即检查进程：
   ps -ef | grep -E 'r1a7_vr|quest_dex1|dex1_1_gripper_server|vuer|teleop' | grep -v grep || true
2. 确认只能有一个 tools/r1a7_vr_dual_arm_g1ik_real.py 进程。
3. 如果有多个联合控制脚本，全部停止后重新启动，只保留一个。
4. 确认 Dex1 服务仍在运行：
   ps -ef | grep -E 'dex1_1_gripper_server' | grep -v grep || true
5. 确认没有 8012 旧端口残留：
   ss -lntp | grep -E '8012|60000|60001' || true
6. 查看 Dex1 串口：
   ls -l --time-style=long-iso /dev/ttyUSB*
7. 如果串口更新时间晚于 Dex1 服务启动时间，按以下顺序恢复：
   先停止联合控制脚本。
   再停止 dex1_1_gripper_server。
   重新执行 ./scripts/run_dex1_1_service.sh。
   看到 Motor ID 0 Side right、Motor ID 1 Side left、Dex1-1 Gripper Server started 后，再启动联合控制脚本。
```

退出 VR 后机器人异常运动：

```text
1. 立即停止联合控制脚本。
2. 确认本次启动命令包含 --stale_pose_timeout 0.25 和 --rearm_hold_time 1.0。
3. 确认运行的是 tools/r1a7_vr_dual_arm_g1ik_real.py。
4. 清理残留进程后重新启动。
```

双臂运动过大或接近限位：

```text
1. 立即减小手柄动作。
2. 必要时停止 Demo。
3. 后续可降低 max_joint_offset_rad 或 ik_delta_scale。
```

## 9. Demo 快速清单

演示前：

```text
[ ] 已进入 /home/robot/unitree_sim_isaaclab。
[ ] enx9c69d37d0967 存在并处于 UP。
[ ] enx9c69d37d0967 具有 192.168.123.xxx/24 地址。
[ ] ping 192.168.123.223 成功。
[ ] 控制主机当前 Quest 访问 IP 已确认。
[ ] Quest URL 与 --host_ip 使用同一个控制主机 IP。
[ ] /dev/ttyUSB* 能看到 Dex1 相关串口。
[ ] 工作空间清空。
[ ] 急停可用。
[ ] 双臂自然中立。
[ ] 手柄自然中立。
[ ] 无端口和进程残留。
[ ] Dex1 服务已启动。
[ ] 联合控制脚本已启动并输入 ENABLE。
[ ] Quest 成功进入 VR。
[ ] 日志出现 relative_tracking。
```

演示中：

```text
[ ] 左手柄控制左臂。
[ ] 右手柄控制右臂。
[ ] 左扳机控制左夹爪。
[ ] 右扳机控制右夹爪。
[ ] 动作小幅慢速。
[ ] 没有靠近障碍物或关节极限。
```

演示后：

```text
[ ] 终端 2 Ctrl+C。
[ ] 看到 released lowcmd gains。
[ ] 终端 1 Ctrl+C。
[ ] ss 检查无 8012/60000/60001。
[ ] ps 检查无遥操作相关残留进程。
```
