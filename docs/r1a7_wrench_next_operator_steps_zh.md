# R1-A7 扳手任务下一步现场操作步骤

## 当前结论

下一步采用现有手柄/Quest 遥控机器人执行动作，新扳手项目只做只读状态记录。不要让新 FSM 或轨迹执行器直接控制真机。

当前 VR 遥控脚本已经调整为：

- 退出 VR、Quest 姿态断流或 WebSocket 异常时，双臂保持当前 `rt/lowstate` 实测姿态。
- 不再回到脚本启动时的初始双臂向下姿态。
- 再次进入 VR 后，脚本先执行 `rearm_hold`，以当前机器人姿态作为新的跟随基准。
- 看到 `relative_tracking` 后再移动手柄。

本机侧只读检查结果：

- 机器人控制网口：`enp6s0`
- 本机控制网段地址：`192.168.123.223/24`
- 机器人可达地址：`192.168.123.161`
- 路由：到 `192.168.123.161` 走 `enp6s0`
- Quest/路由侧网口：`enx9c69d37d0967`
- Quest/路由侧本机地址：`192.168.1.103/24`
- 当前未发现旧 `r1a7_vr` / `teleop` / `record_real_lowstate` 进程
- 当前未发现 `8012` / `60000` / `60001` 端口占用
- 只读 `rt/lowstate` 记录已验证，2 秒记录 40 条样本

## 操作前检查

在机器人旁边确认：

1. 急停可用。
2. 双臂运动范围内无人。
3. 扳手、螺母工装、桌面边缘和线缆不会进入肩、肘、腕危险区域。
4. 第一次只做空载动作，不抓扳手。
5. 只允许运行一个真实运动控制脚本。

在电脑上确认：

```bash
cd /home/robot/unitree_sim_isaaclab
ip -br addr show enp6s0
ip -br addr show enx9c69d37d0967
ip route get 192.168.123.161
ps -ef | grep -E 'r1a7_vr|quest_dex1|dex1_1_gripper_server|vuer|teleop|record_real_lowstate' | grep -v grep || true
ss -lntp | grep -E '8012|60000|60001' || true
```

## 终端 1：启动手柄遥控

使用现有已验证脚本控制机器人。启动后需要手动输入 `ENABLE`。

```bash
cd /home/robot/unitree_sim_isaaclab
# 当前只读检查确认机器人链路走 enp6s0；若现场路由不同，以
# `ip route get 192.168.123.161` 输出的 dev 为准。
PYTHONNOUSERSITE=1 /home/robot/miniconda3/bin/conda run --no-capture-output -n tv \
python -u tools/r1a7_vr_dual_arm_g1ik_real.py \
  --interface enp6s0 \
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
  --lowcmd_gripper_velocity_limit 1.5 \
  --lowcmd_gripper_kp 8.0 \
  --lowcmd_gripper_kd 1.5 \
  --lowcmd_gripper_contact_hold \
  --lowcmd_gripper_contact_trigger_alpha 0.85 \
  --lowcmd_gripper_contact_error 0.08 \
  --lowcmd_gripper_contact_stall_eps 0.004 \
  --lowcmd_gripper_contact_stall_time 0.25 \
  --lowcmd_gripper_contact_hold_bias 0.035 \
  --arm_enable_button right_ctrl_aButton \
  --host_ip 192.168.1.103 \
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

Quest 浏览器打开：

```text
https://192.168.1.103:8012/?ws=wss://192.168.1.103:8012
```

## 终端 2：启动只读记录

先记录 60 秒空载遥控数据：

```bash
cd /home/robot/unitree_sim_isaaclab
PYTHONNOUSERSITE=1 /home/robot/miniconda3/bin/conda run --no-capture-output -n tv \
python -u r1a7_wrench_project/scripts/record_real_lowstate.py \
  --interface enp6s0 \
  --domain-id 0 \
  --state-topic rt/lowstate \
  --duration 60 \
  --rate-hz 50 \
  --episode-id episode_wrench_free_motion_$(date +%Y%m%d_%H%M%S)
```

## 第一条数据采集动作

只做空载，不碰扳手：

1. 双手柄保持中立，确认机器人不突动。
2. 左臂小幅前后移动。
3. 右臂小幅前后移动。
4. 左右夹爪各开合一次。
5. 右臂慢速移动到扳手预抓取区域上方，但不接触扳手。
6. 回到安全姿态。
7. 停止终端 2 记录器。
8. 停止终端 1 遥控脚本，确认出现 `released lowcmd gains`。

## 通过标准

- 终端 1 持续显示 `relative_tracking`。
- 退出 VR 后终端显示 `waiting_quest_pose_hold`、`stale_hold` 或类似 hold 状态，双臂保持退出时姿态。
- 再次进入 VR 后先显示 `rearm_hold` / `calibrated_ik_zero`，然后恢复 `relative_tracking`。
- 夹爪能按左右扳机分别开合。
- 终端 2 输出 episode 路径和样本数。
- `states.csv` 中 `communication_ok` 大部分为 `True`。
- `joint_position` 为 14 个数。
- `gripper_position` 包含 `left` 和 `right`。
- 机器人无碰撞、无过流、无急停。

## 通过后再做

按同样方式分别记录：

1. `PRE_GRASP`：右夹爪到扳手前方。
2. `GRASP`：右夹爪到抓取位置。
3. `LIFT`：夹住扳手后小幅抬升。
4. `TRANSFER_SAFE`：搬运安全中间位。
5. `PRE_NUT`：到螺母前方预接触位。

每个姿态先人工确认安全，再写入 `r1a7_wrench_project/config/keyframes.yaml`。
