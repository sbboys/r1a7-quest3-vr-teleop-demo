# R1-A7 Quest 3 双臂与 Dex1 双夹爪 VR 遥操作项目完整汇总

版本：2026-07-28 最终汇总

适用平台：

```text
Unitree R1-A7 真实机器人
Dex1-1 左右夹爪
Meta Quest 3 左右手柄
Ubuntu 22.04 控制主机 robot
```

当前结论：

```text
Quest 3 左右手柄腕部位姿可以控制 R1-A7 左右双臂。
Quest 3 左右食指扳机可以分别控制 Dex1 左右夹爪。
双臂控制采用 Unitree G1_29 官方 IK 求解，再映射到 R1-A7 双臂 14 个关节。
最终真机参数已完成测试：arm_velocity_limit=5.0, max_joint_offset_rad=2.00, ik_delta_scale=1.00。
退出 VR、刷新页面、重新进入 VR 时，已加入 stale_hold 与 rearm_hold 保护，避免追旧手柄位姿导致双臂跳变。
```

高风险提示：

```text
真机运行前必须清空双臂和夹爪工作空间。
必须准备急停。
必须确认没有其他进程同时发布 rt/lowcmd 或 rt/dex1/*/cmd。
每次重新上电、重新进入 VR 或更换场地后，先做小幅慢速动作。
```

## 1. 项目目标

本项目目标是使用 Quest 3 手柄完成 R1-A7 真实机器人的双臂与双夹爪遥操作：

```text
1. 左右手柄腕部位姿控制左右 7 自由度机械臂。
2. 左右手柄食指扳机分别控制左右 Dex1-1 夹爪开合。
3. 双臂控制参考 Unitree G1_29 官方遥操作方法，使用 G1_29_ArmIK.solve_ik()。
4. 保留 R1-A7 官方关节限位、启动姿态相对限位、速度限制、退出 VR 保护和停止释放。
5. 形成可复现的启动、停止、调参、排障和验收流程。
```

不适用场景：

```text
无人值守运行。
没有急停或专人观察。
双臂工作空间内有人、线缆、桌沿、夹具或障碍物。
存在未知 rt/lowcmd 或 rt/dex1/*/cmd 发布进程。
未验证方向和零位就直接使用大范围高速参数。
```

## 2. 总体控制链路

双臂链路：

```text
Quest 3 左右手柄腕部位姿
  -> Vuer / TeleVuerWrapper
  -> Unitree G1_29_ArmIK.solve_ik()
  -> G1_29 双臂 14 关节 IK 输出
  -> 相对零点重定向 home_q + ik_delta_scale * (sol_q - ik_zero_q)
  -> R1-A7 官方关节限位与相对限幅
  -> DDS rt/lowcmd
  -> R1-A7 左右双臂
  -> DDS rt/lowstate 返回实际关节状态
```

夹爪链路：

```text
Quest 3 左右食指扳机
  -> Vuer / TeleVuerWrapper
  -> 联合控制脚本
  -> Dex1_1_Gripper_Controller
  -> DDS rt/dex1/left/cmd 与 rt/dex1/right/cmd
  -> dex1_1_gripper_server
  -> 本机 USB 串口
  -> Dex1-1 左右夹爪
```

关键区别：

```text
双臂通过 rt/lowcmd 控制。
Dex1 夹爪通过 rt/dex1/left|right/cmd 控制。
夹爪不是作为 R1-A7 lowcmd 里的第 29、30 个关节控制。
```

## 3. 当前硬件、网络与目录

当前网络和 DDS 配置：

```text
控制主机 IP: 192.168.1.127
机器人控制网卡: enx9c69d37d0967
机器人网段地址: 192.168.123.223/24
DDS Domain ID: 0
Quest/Vuer 端口: 8012
双臂状态话题: rt/lowstate
双臂命令话题: rt/lowcmd
Dex1 左夹爪话题: rt/dex1/left/cmd, rt/dex1/left/state
Dex1 右夹爪话题: rt/dex1/right/cmd, rt/dex1/right/state
Quest 3 浏览器入口: https://192.168.1.127:8012/?ws=wss://192.168.1.127:8012
```

主要目录：

```text
/home/robot/unitree_sim_isaaclab
/home/robot/xr_teleoperate
/home/robot/IsaacLab/unitree_robots/dex1_1_service
```

当前 Dex1 服务运行在控制主机本机，直接使用本机识别到的 `/dev/ttyUSB*` 串口，不依赖先登录机器人 PC2 启动夹爪服务。

## 4. 正式文件资产

正式联合控制入口：

```text
tools/r1a7_vr_dual_arm_g1ik_real.py
```

用途：

```text
Quest 3 左右手柄位姿 -> G1_29 IK -> R1-A7 双臂
Quest 左右食指扳机 -> Dex1 左右夹爪
```

该脚本当前包含：

```text
--enable_gripper
--stale_pose_timeout
--rearm_hold_time
--pose_change_eps
Dex1 trigger control
stale_hold
rearm_hold
released lowcmd gains
```

夹爪单独测试入口：

```text
tools/r1a7_quest_dex1_trigger_only_test.py
```

Dex1 DDS 辅助测试：

```text
tools/dex1_1_gripper_dds.py
```

Dex1 服务脚本：

```text
scripts/run_dex1_1_service.sh
scripts/start_dex1_1_service_bg.sh
scripts/status_dex1_1_service_bg.sh
scripts/stop_dex1_1_service_bg.sh
```

相关文档：

```text
docs/r1a7_quest3_g1ik_dual_arm_teleop_zh.md
docs/r1a7_quest3_project_file_cleanup_zh.md
docs/r1a7_quest3_dual_arm_dex1_project_summary_zh.md
```

## 5. R1-A7 双臂关节映射

当前脚本使用 R1-A7 官方双臂关节顺序：

```text
左臂:
15 L_SHOULDER_PITCH
16 L_SHOULDER_ROLL
17 L_SHOULDER_YAW
18 L_ELBOW
19 L_WRIST_ROLL
20 L_WRIST_PITCH
21 L_WRIST_YAW

右臂:
22 R_SHOULDER_PITCH
23 R_SHOULDER_ROLL
24 R_SHOULDER_YAW
25 R_ELBOW
26 R_WRIST_ROLL
27 R_WRIST_PITCH
28 R_WRIST_YAW
```

控制目标同时受两类限制：

```text
1. R1-A7 官方每关节机械限位。
2. home_q ± max_joint_offset_rad 相对启动姿态限幅。
```

相对 IK 重定向逻辑：

```text
home_q = 启动时机器人当前双臂真实关节角
ik_zero_q = 第一帧有效 Quest 位姿对应的 G1_29 IK 输出
sol_q = 当前 Quest 位姿对应的 G1_29 IK 输出

delta_q = ik_delta_scale * (sol_q - ik_zero_q)
delta_q = clip(delta_q, -max_joint_offset_rad, +max_joint_offset_rad)
target_q = home_q + delta_q
target_q = clip(target_q, R1-A7 官方关节限位)
```

该方式避免把 G1_29 的绝对关节姿态直接强制写入 R1-A7，而是只使用 G1_29 IK 的相对变化量。

## 6. 最终真机参数

当前最终测试通过参数：

```text
arm_velocity_limit=5.0
max_joint_offset_rad=2.00
ik_delta_scale=1.00
hold_kp=10.0
hold_kd=0.8
stale_pose_timeout=0.25
rearm_hold_time=1.0
pose_change_eps=1e-4
```

参数含义：

```text
ik_delta_scale:
G1_29 IK 输出关节变化量到 R1-A7 真机关节变化量的比例。越大，机器人动作越大。

max_joint_offset_rad:
每个关节相对启动姿态 home_q 的最大允许偏移。如果 dq 顶到该值，说明动作范围被该参数卡住。

arm_velocity_limit:
关节速度限制，单位 rad/s。方向和范围正确但跟随慢时可适当提高。

stale_pose_timeout:
Quest 手柄位姿超过该时间不更新后进入 stale_hold。当前建议 0.25 秒。

rearm_hold_time:
退出 VR 后重新进入时，先保持当前真实关节并刷新 ik_zero_q 的等待时间。当前建议 1.0 秒。

pose_change_eps:
判断 Quest 位姿是否有新变化的阈值，用于区分在线数据和退出 VR 后冻结的最后一帧。
```

调参记录：

```text
0.45 / 0.22 / 3.5:
方向正确，能跟随，但机器人动作明显小于手柄。

0.80 / 0.35 / 3.5:
动作范围增大，未完全满足。

1.10 / 0.70 / 4.0:
dq 顶到 1.100，仍偏小。

1.60 / 1.00 / 4.5:
dq 顶到 1.600，范围继续增大。

2.00 / 1.00 / 5.0:
机器人运动基本接近手柄运动范围，作为当前最终测试参数。
```

说明：

```text
dq=2.000 表示相对关节限幅已经触发。
后续如果还要更接近严格 1:1，不建议继续无限增大 max_joint_offset_rad。
更合理方向是在手柄位姿进入 G1_29 IK 前增加 VR 位置空间比例，例如 vr_position_scale。
```

## 7. 启动前检查

确认没有旧端口或进程残留：

```bash
ss -lntp | rg '8012|60000|60001' || true
ps -ef | rg 'r1a7_vr|quest_dex1|dex1_1_gripper_server|vuer|teleop' | rg -v 'rg|grep' || true
```

正常情况下无输出。若有输出，先停止旧进程，避免端口占用或多个进程同时发布 `rt/lowcmd`。

启动前现场检查：

```text
1. 机器人双臂放在自然向下或自然中立姿态。
2. 左右手柄也放在自然中立位置，再进入 VR。
3. 双臂和夹爪工作空间清空。
4. 操作者和旁站人员准备好急停。
5. Quest 3 与控制主机网络互通。
6. 确认机器人控制网卡为 enx9c69d37d0967。
7. 确认 DDS Domain ID 为 0。
```

## 8. 标准启动 SOP

终端 1：启动 Dex1 服务。

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/run_dex1_1_service.sh
```

成功日志应包含：

```text
Motor ID 0 Side right
Motor ID 1 Side left
Dex1-1 Gripper Server started.
```

终端 2：启动最终联合控制脚本。

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

出现安全提示后输入：

```text
ENABLE
```

期望日志：

```text
[R1-A7 VR G1IK REAL] DDS initialized
[R1-A7 VR G1IK REAL] IK: Unitree G1_29_ArmIK.solve_ik
[R1-A7 VR G1IK REAL] Dex1 trigger control: ENABLED
[R1-A7 VR G1IK REAL] calibrated robot arm q
[R1-A7 VR G1IK REAL] waiting_quest_pose
```

Quest 3 浏览器打开或刷新：

```text
https://192.168.1.127:8012/?ws=wss://192.168.1.127:8012
```

进入 VR 后期望状态：

```text
calibrated_ik_zero
relative_tracking
```

出现 `relative_tracking` 后，双臂开始跟随 Quest 左右手柄。左右食指扳机分别控制左右 Dex1 夹爪。

## 9. 标准停止 SOP

停止顺序：

```text
1. 先停止真机联合控制脚本。
2. 再停止 Dex1 服务。
3. 最后检查端口和进程残留。
```

终端 2 按 `Ctrl+C`，应看到：

```text
[R1-A7 VR G1IK REAL] released lowcmd gains
```

终端 1 按 `Ctrl+C` 停止 Dex1 服务。

停止后检查：

```bash
ss -lntp | rg '8012|60000|60001' || true
ps -ef | rg 'r1a7_vr|quest_dex1|dex1_1_gripper_server|vuer|teleop' | rg -v 'rg|grep' || true
```

正常情况下无输出，表示：

```text
Vuer 8012 端口已释放。
60000/60001 没有残留服务。
双臂控制进程已退出。
Dex1 服务进程已退出。
没有遥操作脚本继续运行。
```

## 10. 退出 VR 与重新进入 VR 的修复

曾出现的问题：

```text
启动程序后双臂保持自然向下。
手柄退出 VR 后，机器人双臂会突然运动再归位。
再次进入 VR 后，机器人不从当前初始姿态开始，而是先摆到异常位置再运动。
```

根因：

```text
TeleVuer 的 motion_data_ready 只表示曾经收到过 Quest 数据。
它不表示当前 Quest 页面仍在持续发送新手柄位姿。
退出 VR 或关闭浏览器后，最后一帧手柄位姿仍可能保留在共享数据中。
如果脚本继续把旧位姿当作实时目标求 IK，双臂会追旧目标或在重进 VR 首帧发生跳变。
```

当前修复：

```text
1. 使用 pose_change_eps 判断左右手柄位姿是否还在变化。
2. 超过 stale_pose_timeout=0.25 秒无新位姿，进入 stale_hold。
3. stale_hold 中目标关节等于当前真实 state_q，不再追旧 IK 目标。
4. 重新进入 VR 并检测到新鲜位姿后，先进入 rearm_hold。
5. rearm_hold 持续 rearm_hold_time=1.0 秒，保持当前真实关节并刷新 ik_zero_q。
6. 重新标定完成后才进入 relative_tracking。
```

保护生效日志：

```text
Quest pose stream stale; holding current arm q and waiting for fresh poses
stale_hold
fresh Quest poses; recalibrated robot arm q
calibrated_ik_zero
relative_tracking
```

实际停止前已观察到：

```text
stale_hold
tele_fresh=False
pose_age=190s+
cmd_err=0.000
tgt_err=0.000
```

该日志说明退出 VR 后控制目标已经保持为真实当前关节，不再追旧手柄目标。

如果后续仍出现退出或重进 VR 跳变，优先检查：

```text
1. 启动命令是否包含 --stale_pose_timeout 0.25。
2. 启动命令是否包含 --rearm_hold_time 1.0。
3. 启动命令是否包含 --pose_change_eps 1e-4。
4. 日志是否先 stale_hold，再 rearm_hold/calibrated_ik_zero，最后 relative_tracking。
5. 是否存在旧进程仍在发布 rt/lowcmd。
```

## 11. 成功判据

启动成功判据：

```text
DDS initialized
IK: Unitree G1_29_ArmIK.solve_ik
Dex1 trigger control: ENABLED
calibrated robot arm q
waiting_quest_pose
```

Quest 连接成功判据：

```text
motion_ready=True
tele_fresh=True
calibrated_ik_zero
relative_tracking
```

双臂控制成功判据：

```text
左右手柄移动时，左右手臂方向对应正确。
left_q/right_q 连续变化。
left_cmd/right_cmd 连续变化。
cmd_err 和 tgt_err 不持续异常增大。
没有机械限位撞击、异常抖动或无法停止。
```

夹爪控制成功判据：

```text
gripper connected=True
左食指扳机控制左夹爪。
右食指扳机控制右夹爪。
release trigger -> OPEN
pull trigger -> CLOSE
merror=0
```

停止成功判据：

```text
released lowcmd gains
ss 检查无 8012/60000/60001 残留
ps 检查无 r1a7_vr/quest_dex1/dex1_1_gripper_server/vuer/teleop 残留
```

## 12. 常见故障与处理

Quest 页面打不开：

```text
检查 8012 是否监听。
确认控制脚本已启动并通过 ENABLE。
确认 Quest 和 PC 在同一可互通网络。
确认 URL 使用 192.168.1.127。
清理旧 Vuer 进程后重启。
```

机器人双臂没有反应：

```text
检查是否进入 relative_tracking。
检查是否仍停在 waiting_quest_pose。
检查 Quest 是否真的进入 VR 并发送手柄位姿。
检查 rt/lowstate 是否持续更新。
检查是否有旧进程占用或多个 rt/lowcmd 发布者。
停止并清理进程后重新启动。
```

双臂动作范围小于手柄：

```text
检查 dq 是否顶到 max_joint_offset_rad。
当前最终参数为 max_joint_offset_rad=2.00, ik_delta_scale=1.00, arm_velocity_limit=5.0。
如果仍偏小，不建议继续无限增大关节限幅，后续应在 VR 位姿进入 G1 IK 前增加空间比例。
```

双臂方向不对：

```text
不要改 R1-A7 关节顺序。
优先检查 Quest 手柄位姿到 G1_29 IK 目标位姿的坐标变换。
当前已验证左右臂运动方向正确，正式脚本不要回退到旧 MuJoCo 自写 IK 或 Cartesian 脚本。
```

退出 VR 后双臂跳变：

```text
确认使用正式脚本 tools/r1a7_vr_dual_arm_g1ik_real.py。
确认启动命令包含 stale_pose_timeout、rearm_hold_time、pose_change_eps。
查看日志是否进入 stale_hold。
如果没有 stale_hold，说明手柄新鲜度检测没有生效或运行的不是最新脚本。
```

夹爪没有反应：

```text
确认 ./scripts/run_dex1_1_service.sh 已启动。
确认日志出现 Motor ID 0 Side right 与 Motor ID 1 Side left。
确认联合脚本带 --enable_gripper。
确认 gripper connected=True。
确认没有旧 Dex1 服务或串口占用。
```

停止后再次测试异常：

```text
执行 ss 和 ps 检查。
确认没有残留 8012/60000/60001。
确认没有 r1a7_vr、quest_dex1、dex1_1_gripper_server、vuer、teleop 残留进程。
重新启动时让机器人双臂和手柄都保持自然中立姿态。
```

## 13. 项目文件整理状态

已确认正式入口：

```text
tools/r1a7_vr_dual_arm_g1ik_real.py
tools/r1a7_quest_dex1_trigger_only_test.py
tools/dex1_1_gripper_dds.py
scripts/run_dex1_1_service.sh
```

历史文件已统一归档：

```text
archive/cleanup_20260728_133418/
```

归档内容包括：

```text
历史 .bak、backup、pre-Dex1 脚本备份
MuJoCo、preview、Cartesian、pose IK、right-arm-only 等早期实验脚本
根目录一次性安装、修复、替换脚本
旧运行日志
/home/robot/下载 中已被正式 tools 替代的重复脚本
```

暂未清理内容：

```text
相机识别相关脚本
右臂示教相关脚本
IsaacLab 旧任务
Dex1 维护工具
```

这些内容可能仍服务于其他任务，不应在当前双臂与双夹爪项目中直接删除。

## 14. 后续优化方向

后续若继续提升控制效果，建议按以下顺序进行：

```text
1. 增加 VR 位姿进入 G1_29 IK 前的空间比例参数，例如 vr_position_scale。
2. 分左右臂分别设置位置比例和姿态比例。
3. 加入工作空间软限位，避免双臂靠近身体、桌面或自碰撞区域。
4. 增加启动前自动检查旧进程和端口残留。
5. 将 cmd_err、tgt_err、dq、pose_age 写入结构化日志。
6. 进一步记录抓取任务中的夹爪开合阈值和物体接触策略。
```

当前不建议的方向：

```text
不建议继续无限增大 max_joint_offset_rad。
不建议绕过 R1-A7 官方关节限位。
不建议重新使用已归档的 MuJoCo 旧控制器作为真机正式入口。
不建议在真机上使用未经过仿真和小幅动作验证的新坐标变换。
```

## 15. 最终验收检查表

每次完整测试前后按以下清单确认：

```text
[ ] 双臂和夹爪工作空间已清空。
[ ] 急停可用。
[ ] Quest 与 PC 网络互通。
[ ] 没有旧 8012/60000/60001 端口残留。
[ ] 没有旧 r1a7_vr/quest_dex1/dex1_1_gripper_server/vuer/teleop 进程残留。
[ ] Dex1 服务启动成功。
[ ] 联合脚本启动成功并输入 ENABLE。
[ ] 日志出现 DDS initialized。
[ ] 日志出现 calibrated robot arm q。
[ ] Quest 进入 VR 后日志出现 relative_tracking。
[ ] 左右手柄分别控制左右手臂。
[ ] 左右扳机分别控制左右夹爪。
[ ] 退出 VR 后日志进入 stale_hold，不追旧位姿。
[ ] 重新进入 VR 后先重新标定，再进入 relative_tracking。
[ ] 停止时出现 released lowcmd gains。
[ ] 停止后 ss 和 ps 检查无残留。
```
