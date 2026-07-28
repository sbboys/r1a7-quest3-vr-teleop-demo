# R1-A7 使用 Quest 3 手柄控制双臂记录

本文档记录当前已验证的 R1-A7 真实机器人双臂 VR 手柄控制流程。当前方案参考宇树 G1_29 官方双臂遥操作方式：Quest 3 手柄提供左右腕部位姿，使用宇树 `G1_29_ArmIK.solve_ik()` 求解 14 个双臂关节角，再映射到 R1-A7 双臂 14 个关节并通过 `rt/lowcmd` 下发。

## 当前结论

当前可用链路为：

```text
Quest 3 手柄
  -> Vuer / TeleVuerWrapper
  -> Unitree G1_29_ArmIK.solve_ik()
  -> R1-A7 双臂关节映射
  -> rt/lowcmd
  -> 真实机器人双臂
```

当前采用的真机脚本：

```text
/home/robot/unitree_sim_isaaclab/tools/r1a7_vr_dual_arm_g1ik_real.py
```

当前测试结果显示，机器人双臂运动范围已经基本接近手柄运动范围。最后一组有效参数为：

```text
max_joint_offset_rad=2.00
ik_delta_scale=1.00
arm_velocity_limit=5.0
```

这组参数下日志曾出现：

```text
relative_tracking dq=2.000
cmd_err=0.14 左右
tgt_err=0.25~0.38
```

其中 `dq=2.000` 表示动作已经达到当前设置的相对关节活动上限。继续扩大范围时，不建议只继续提高 `max_joint_offset_rad`，应优先考虑调节 VR 位姿进入 G1 IK 前的空间映射比例。

## 网络与设备

PC 侧主机 IP：

```text
192.168.1.127
```

机器人控制网口：

```text
enx9c69d37d0967
```

Quest 3 浏览器入口：

```text
https://192.168.1.127:8012/?ws=wss://192.168.1.127:8012
```

DDS 话题：

```text
状态: rt/lowstate
命令: rt/lowcmd
```

## R1-A7 双臂关节顺序

当前脚本使用的 R1-A7 双臂电机索引与官方文档一致：

```text
左臂: 15,16,17,18,19,20,21
右臂: 22,23,24,25,26,27,28
```

对应关节：

```text
15 L_SHOULDER_PITCH
16 L_SHOULDER_ROLL
17 L_SHOULDER_YAW
18 L_ELBOW
19 L_WRIST_ROLL
20 L_WRIST_PITCH
21 L_WRIST_YAW

22 R_SHOULDER_PITCH
23 R_SHOULDER_ROLL
24 R_SHOULDER_YAW
25 R_ELBOW
26 R_WRIST_ROLL
27 R_WRIST_PITCH
28 R_WRIST_YAW
```

脚本中已加入 R1-A7 官方关节限位保护。控制目标同时受两类限制：

```text
1. R1-A7 官方每关节机械限位
2. home_q ± max_joint_offset_rad 相对启动姿态限幅
```

## 启动前检查

确认没有旧的 VR 或 Vuer 进程残留：

```bash
ss -lntp | rg '8012|60000|60001' || true
ps -ef | rg 'r1a7_vr|teleop_hand|vuer|teleop' | rg -v 'rg|grep' || true
```

如果有旧进程，先停止旧进程，避免端口或 `lowcmd` 发布冲突。

启动前建议：

```text
1. 机器人双臂摆到自然中立姿态。
2. 双臂周围不要有人或障碍物。
3. 操作者准备好急停。
4. Quest 3 和 PC 处于可互通网络。
5. 手柄也放在自然中立位置，再进入 VR。
```

## 启动命令

进入仓库：

```bash
cd /home/robot/unitree_sim_isaaclab
```

如果本次需要同时控制 Dex1 左右夹爪，先单独启动 Dex1 服务，并保持该终端运行：

```bash
./scripts/run_dex1_1_service.sh
```

看到以下内容表示夹爪服务已启动：

```text
Dex1-1 Gripper Server started.
```

再打开新终端，启动真机双臂和夹爪联合手柄控制：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/bin/conda run --no-capture-output -n tv \
python -u tools/r1a7_vr_dual_arm_g1ik_real.py \
  --interface enx9c69d37d0967 \
  --domain_id 0 \
  --state_topic rt/lowstate \
  --command_topic rt/lowcmd \
  --enter_debug_mode \
  --enable_gripper \
  --host_ip 192.168.1.127 \
  --duration 0 \
  --g1_style_gains \
  --g1_style_velocity_clip \
  --arm_velocity_limit 5.0 \
  --hold_kp 10.0 \
  --hold_kd 0.8 \
  --max_joint_offset_rad 2.00 \
  --ik_delta_scale 1.00 \
  --stale_pose_timeout 0.25 \
  --rearm_hold_time 1.0 \
  --pose_change_eps 1e-4 \
  --print_period 0.25
```

终端出现以下提示后输入：

```text
ENABLE
```

然后在 Quest 3 浏览器打开或刷新：

```text
https://192.168.1.127:8012/?ws=wss://192.168.1.127:8012
```

进入 VR 后，日志会从：

```text
waiting
```

变为：

```text
calibrated_ik_zero
relative_tracking
```

出现 `relative_tracking` 后，说明 Quest 手柄数据已经进入控制链路，机器人双臂开始跟随。

## 停止流程

在运行终端按 `Ctrl+C`，或者由控制端发送停止后，应看到：

```text
[R1-A7 VR G1IK REAL] released lowcmd gains
```

如果同时启动了 Dex1 服务，也在 Dex1 服务终端按 `Ctrl+C` 停止。

停止后检查：

```bash
ss -lntp | rg '8012|60000|60001' || true
ps -ef | rg 'r1a7_vr|quest_dex1|dex1_1_gripper_server|vuer|teleop' | rg -v 'rg|grep' || true
```

正常情况下无输出，表示 Vuer 端口和控制进程已经退出。

## 参数含义

`ik_delta_scale`：

```text
G1_29 IK 输出关节变化量到 R1-A7 真机关节变化量的比例。
数值越大，机器人动作越大。
```

`max_joint_offset_rad`：

```text
每个关节相对启动姿态 home_q 的最大允许偏移。
如果日志中的 dq 顶到该值，说明动作范围被该参数卡住。
```

`arm_velocity_limit`：

```text
关节速度限制，单位 rad/s。
如果机器人方向和范围正确，但跟随慢半拍，可以适当提高。
```

`cmd_err`：

```text
当前命令与机器人实际关节位置的最大误差。
数值越大，说明机器人越跟不上当前命令。
```

`tgt_err`：

```text
目标关节位置与机器人实际关节位置的最大误差。
持续偏大时，说明目标动作过大、速度不足或接近控制限制。
```

`dq`：

```text
当前相对初始姿态的最大关节偏移。
如果 dq 等于 max_joint_offset_rad，说明相对活动范围已经被限幅。
```

`stale_pose_timeout`：

```text
Quest 手柄位姿长时间不更新时的超时时间。
超过该时间后，脚本进入 stale_hold，不再继续追退出 VR 前的最后一帧目标。当前建议值为 0.25 秒。
```

`pose_change_eps`：

```text
判断 Quest 左右手柄位姿是否有新变化的阈值。
用于区分仍在更新的手柄位姿和退出 VR 后冻结的最后一帧数据。
```

`rearm_hold_time`：

```text
退出 VR 后重新进入时的重新标定等待时间。
这段时间内机器人保持当前真实关节位置，同时刷新 ik_zero_q，避免一进入 VR 就追旧目标或首帧抖动目标。
当前建议值为 1.0 秒。
```

## 调参记录

初始可用参数：

```text
max_joint_offset_rad=0.45
ik_delta_scale=0.22
arm_velocity_limit=3.5
```

现象：方向正确，机器人能跟随，但动作范围明显小于手柄。

扩大到：

```text
max_joint_offset_rad=0.80
ik_delta_scale=0.35
arm_velocity_limit=3.5
```

现象：动作范围增大，日志中 `dq=0.551`，未顶到 `0.80`。

扩大到：

```text
max_joint_offset_rad=1.10
ik_delta_scale=0.70
arm_velocity_limit=4.0
```

现象：`dq=1.100`，顶到相对限幅，机器人动作仍偏小。

扩大到：

```text
max_joint_offset_rad=1.60
ik_delta_scale=1.00
arm_velocity_limit=4.5
```

现象：`dq=1.600`，再次顶到相对限幅，动作范围进一步增大。

最终测试参数：

```text
max_joint_offset_rad=2.00
ik_delta_scale=1.00
arm_velocity_limit=5.0
```

现象：机器人运动基本接近手柄运动范围，但 `dq=2.000` 说明仍会触发相对关节限幅。

## 后续优化方向

如果后续仍希望更接近严格的手柄 1:1 映射，不建议继续无限增大 `max_joint_offset_rad`。更合理的下一步是在 `tele.left_wrist_pose` / `tele.right_wrist_pose` 进入 `G1_29_ArmIK.solve_ik()` 前加入 VR 位姿空间比例，例如：

```text
vr_position_scale=2.0
```

这样是放大末端目标位姿，再由 IK 自然求解关节，而不是在 IK 求解后继续硬放大关节差值。该方式更适合继续优化手柄空间到机器人动作空间的同步关系。

## 安全注意事项

当前最终参数已经属于大范围真机动作：

```text
max_joint_offset_rad=2.00
arm_velocity_limit=5.0
```

测试时必须遵守：

```text
1. 起步时只做小幅慢速动作。
2. 不要快速甩动手柄。
3. 不要让双臂靠近身体、桌面边缘或其他障碍物。
4. 重点观察肩 roll、肩 yaw、肘关节是否接近极限。
5. 出现不符合预期的方向、抖动、卡顿或碰撞风险时立即停止。
```

## 退出 VR 后的保护

TeleVuer 的 `motion_data_ready` 只表示曾经收到过 Quest 手柄数据，不表示当前 VR 页面仍在持续发送新数据。退出 VR 或关闭浏览器时，最后一帧手柄位姿可能仍保留在共享数据中。如果控制脚本继续使用这帧旧数据求 IK，机器人双臂会继续追最后目标并保持在异常姿态。

当前脚本已加入本地新鲜度检测：

```text
1. 如果 Quest 左右手柄位姿超过 stale_pose_timeout 没有变化，进入 stale_hold。
2. stale_hold 状态下，目标关节改为当前真实关节 state_q，不再追旧手柄目标。
3. 重新进入 VR 并检测到新鲜位姿后，脚本先进入 rearm_hold，保持当前真实关节并持续刷新 ik_zero_q，默认 1 秒后才允许相对跟踪。
```

日志中看到以下内容表示保护生效：

```text
Quest pose stream stale; holding current arm q and waiting for fresh poses
stale_hold
fresh Quest poses; recalibrated robot arm q
```

实际测试中，退出 VR 后日志曾长期保持：

```text
stale_hold
tele_fresh=False
pose_age=190s+
cmd_err=0.000
tgt_err=0.000
```

这表示保护逻辑已经接管：手柄位姿不再更新时，脚本没有继续使用旧位姿求 IK，而是把目标关节保持为当前真实关节。停止程序时看到：

```text
[R1-A7 VR G1IK REAL] released lowcmd gains
```

表示真机双臂控制增益已释放。随后 `ss` 和 `ps` 检查无输出，表示 `8012/60000/60001` 端口、Vuer 进程、双臂控制进程和 Dex1 服务均已清理完成。

如果后续再次出现“退出 VR 后双臂突然运动”或“重新进入 VR 后从异常姿态开始控制”，优先检查以下三项：

```text
1. 启动命令中是否包含 --stale_pose_timeout 0.25。
2. 启动命令中是否包含 --rearm_hold_time 1.0。
3. 日志是否先出现 stale_hold，再在重新进入 VR 后经过 rearm_hold/calibrated_ik_zero，最后才进入 relative_tracking。
```
