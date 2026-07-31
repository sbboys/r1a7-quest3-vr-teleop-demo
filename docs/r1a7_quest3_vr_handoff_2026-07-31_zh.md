# R1-A7 Quest 3 VR 遥操作项目交接记录 2026-07-31

本文记录 2026-07-31 当天 R1-A7 真机、Quest 3 手柄、双臂和内走线 Dex1 夹爪遥操作工作的当前状态，便于新的 Codex 或开发者直接接手。

## 1. 当前仓库状态

- 本地目录：`/home/robot/unitree_sim_isaaclab`
- 当前分支：`main`
- 当前 HEAD：`e30c25b update readme`
- 原始 `origin`：`https://github.com/unitreerobotics/unitree_sim_isaaclab.git`
- 用户目标 GitHub 仓库：`git@github.com:sbboys/r1a7-quest3-vr-teleop-demo.git`
- 注意：不要把本项目私有改动推送到宇树官方 `origin`。

本次工作区里存在较多历史改动和未跟踪文件：

- 已跟踪改动：`README_zh-CN.md`、`sim_main.py`、`requirements.txt`、`action_provider/create_action_provider.py`、`tasks/__init__.py`、`tasks/common_observations/camera_state.py`
- 大量 `__pycache__/*.pyc` 显示为删除，属于缓存清理，不是本次功能核心。
- 本次 VR 真机核心脚本目前是未跟踪文件：`tools/r1a7_vr_dual_arm_g1ik_real.py`
- 相关文档目录：`docs/`
- 相关脚本目录：`scripts/`
- 相关工具：`tools/r1a7_*`、`tools/check_vr_controllers.py`、`tools/test_gemini_pose.py`

## 2. 今日核心目标

将 Quest 3 手柄位姿用于控制真实 R1-A7 双臂，并用手柄扳机实时控制内走线 Dex1 双夹爪。

实际采用链路：

```text
Quest 3 浏览器 / Vuer
  -> TeleVuerWrapper 手柄位姿和 trigger
  -> Unitree 官方 G1_29_ArmIK.solve_ik()
  -> R1-A7 双臂 14 关节相对映射
  -> rt/lowcmd 控制 15-28 号手臂关节
  -> rt/lowcmd 控制 31/33 号内走线夹爪电机
```

没有继续使用旧的 `tools/r1a7_mujoco_vr_controller.py`。真机路径直接参考宇树 G1_29 官方 IK。

## 3. 关键文件

- `tools/r1a7_vr_dual_arm_g1ik_real.py`
  - 真机 Quest 3 VR 双臂和夹爪控制主脚本。
  - 使用 `/home/robot/xr_teleoperate/teleop/robot_control/robot_arm_ik.py` 中的 `G1_29_ArmIK`。
  - 使用 `/home/robot/xr_teleoperate/teleop/televuer/src` 的 `TeleVuerWrapper`。
  - 发布真实机器人 `rt/lowcmd`，订阅 `rt/lowstate`。

- `docs/r1a7_quest3_dual_arm_dex1_demo_guide_zh.md`
  - Demo 操作指南，包含网络、DDS、浏览器、启动流程。

- `docs/r1a7_quest3_dual_arm_dex1_project_summary_zh.md`
  - 项目汇总文档。

- `docs/r1a7_quest3_g1ik_dual_arm_teleop_zh.md`
  - G1_29 IK 映射和调参说明。

- `docs/r1a7_camera_arm_gripper_runbook_zh.md`
  - Gemini 相机识别、右臂和 Dex1 夹爪历史工作流。

## 4. 当前运行环境

- 主机：`robot@robot-System-Product-Name`
- 系统 Python：`Python 3.10.12`
- conda 环境：`tv`
- conda `tv` Python：`Python 3.10.20`
- GPU：NVIDIA GeForce RTX 4070 系列，约 12GB 显存
- NVIDIA Driver：`580.173.02`
- CUDA：`13.0`
- 关键 Python 包：
  - `numpy 1.26.4`
  - `scipy 1.15.2`
  - `pinocchio 3.1.0`
  - `matplotlib 3.7.5`
  - `pyparsing 3.3.2`
  - `torch 2.3.0`
  - `vuer 0.0.60`

之前遇到过 `ModuleNotFoundError: No module named 'pyparsing'`，已通过在 `tv` 环境安装/修复 `pyparsing` 解决。

## 5. 网络和 DDS 参数

当前真机测试使用：

- 机器人控制网口：`enx9c69d37d0967`
- 主机 IP：`192.168.1.127`
- Quest 3 访问地址：

```text
https://192.168.1.127:8012/?ws=wss://192.168.1.127:8012
```

- DDS domain：`0`
- 状态 topic：`rt/lowstate`
- 控制 topic：`rt/lowcmd`
- Vuer / HTTPS / websocket 端口：`8012`

启动前检查：

```bash
ps -ef | grep -E 'r1a7_vr|quest_dex1|dex1_1_gripper_server|vuer|teleop' | grep -v -E 'grep|rg' || true
ss -lntp | grep -E '8012|60000|60001' || true
```

## 6. R1-A7 关节和夹爪配置

双臂关节：

- 左臂：`15,16,17,18,19,20,21`
- 右臂：`22,23,24,25,26,27,28`

关节名称顺序：

```text
left_shoulder_pitch
left_shoulder_roll
left_shoulder_yaw
left_elbow
left_wrist_roll
left_wrist_pitch
left_wrist_yaw
right_shoulder_pitch
right_shoulder_roll
right_shoulder_yaw
right_elbow
right_wrist_roll
right_wrist_pitch
right_wrist_yaw
```

内走线 Dex1 夹爪作为低层电机控制：

- 左夹爪：`31`
- 右夹爪：`33`
- 左夹爪开：`4.86`
- 左夹爪关：`-0.08`
- 右夹爪开：`4.80`
- 右夹爪关：`-0.20`
- 夹爪速度限制：`8.0`
- 夹爪增益：`kp=25.0`，`kd=0.8`

TeleVuer trigger 当前表现：

- `10.0`：松开，对应夹爪打开
- `0.0`：完全按下，对应夹爪闭合

## 7. 当前主启动命令

最近一次实测命令如下：

```bash
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

程序启动后会提示：

```text
Type ENABLE to continue:
```

必须手动输入：

```text
ENABLE
```

## 8. 今天已实现的代码逻辑

### 8.1 真机安全启动

- 启动时检查 MotionSwitcher 当前模式。
- 可通过 `--enter_debug_mode` 释放机器人原有运动模式。
- 启动前打印真实低层控制警告。
- 未输入 `ENABLE` 不会继续发布真实低层命令。

### 8.2 Quest 数据新鲜度判断

脚本会区分：

- `waiting_quest_pose`：没有 Quest 手柄位姿。
- `rearm_hold`：刚收到新位姿，保持当前机器人 q，用于重标定。
- `calibrated_ik_zero`：记录当前 IK 零点。
- `relative_tracking`：开始按相对位移控制双臂。
- `stale_hold`：Quest 位姿断流或冻结，保持当前真实关节。

新增逻辑：

- `--frozen_pose_hold_timeout`
- 当 Quest 矩阵长时间不变化，但 `motion_data_ready` 还没有掉线时，提前冻结，避免退出 VR 前最后一帧造成双臂跳变。
- 预期日志：

```text
Quest pose frozen before stale timeout; holding current arm q
```

### 8.3 G1_29 IK 到 R1-A7 的相对映射

核心公式：

```text
delta_q = ik_delta_scale * ik_joint_scales * (sol_q - ik_zero_q)
target_q = home_q + clip(delta_q, +/- max_joint_offset_rad)
```

然后再按 R1-A7 真实关节限位进行裁剪。

新增软限位：

- `--joint_limit_margin_rad`
- `--shoulder_pitch_low_margin_rad`

用途：

- 避免直接撞到真实硬限位。
- 针对高抬手时 `shoulder_pitch` 下限连续触发的问题，给左右肩 Pitch 低限位加额外安全边界。

### 8.4 夹爪控制

夹爪不走独立 SDK，而是作为低层电机 31/33 控制。

当前测试结果：

- 左右夹爪都可以通过 Quest 手柄扳机实时张开/闭合。
- 左右幅值已调一致。
- 当前配置下夹爪响应速度和幅度已满足测试要求。

## 9. 今日测试过程和结果

### 9.1 启动和连接

检查残留进程：

```bash
ps -ef | grep -E 'r1a7_vr|quest_dex1|dex1_1_gripper_server|vuer|teleop' | grep -v -E 'grep|rg' || true
ss -lntp | grep -E '8012|60000|60001' || true
```

结果：

- 没有残留控制进程。
- 没有端口占用。

启动后日志显示：

```text
DDS initialized
IK: Unitree G1_29_ArmIK.solve_ik
calibrated robot arm q: [...]
calibrated lowcmd gripper q: [4.857, 4.8]
```

Quest 刷新后日志显示：

```text
websocket is connected
fresh Quest poses; rearm holding current q
relative_tracking
```

说明 Quest、Vuer、DDS、低层夹爪和双臂控制链路连通。

### 9.2 已尝试过的关键参数

早期较激进参数：

```text
max_joint_offset_rad=2.00
ik_delta_scale=1.00
arm_velocity_limit=5.0
```

效果：

- 双臂范围接近手柄。
- 但高抬手时容易出现肩关节限位。

进一步放大后：

```text
max_joint_offset_rad=2.80
ik_delta_scale=1.35
```

初始均匀比例导致：

```text
left_shoulder_yaw high
right_shoulder_yaw high
right_wrist_roll high
```

随后引入 `--ik_joint_scales`。

较稳定候选：

```text
1,0.65,0.45,0.85,0.55,1,0.75,1,0.65,0.45,0.65,0.55,1,0.55
```

结果：

- `shoulder_yaw` 问题明显缓解。
- 高抬时转为 `shoulder_pitch` 触限。

今日最后测试参数：

```text
0.80,0.65,0.45,0.90,0.55,1,0.75,0.80,0.65,0.45,0.90,0.55,1,0.55
```

配合：

```text
joint_limit_margin_rad=0.04
shoulder_pitch_low_margin_rad=0.18
frozen_pose_hold_timeout=0.12
```

结果：

- `shoulder_pitch` 触限明显减少。
- 大幅动作时出现新的主要问题：右肘低限位。

### 9.3 最新未解决问题

最新测试中出现：

```text
arm_limit : right_elbow low raw=-0.949 clip=-0.936
arm_limit : right_elbow low raw=-0.989 clip=-0.936
arm_limit : right_elbow low raw=-1.175 clip=-0.936
arm_limit : right_elbow low raw=-1.178 clip=-0.936
```

含义：

- 之前主要是肩 Pitch 被 G1_29 IK 推到 R1-A7 下限。
- 经过软限位和肩 Pitch 比例降低后，IK 会把一部分大幅手柄动作转移到肘部。
- 当前右肘在高抬或大幅动作时可能进入低限位附近，导致手臂继续不动或姿态不自然。

建议下一次测试参数：

```text
--ik_joint_scales 0.80,0.65,0.45,0.70,0.55,1,0.75,0.80,0.65,0.45,0.60,0.55,1,0.55
```

相比今日最后一轮：

- 左肘：`0.90 -> 0.70`
- 右肘：`0.90 -> 0.60`

目标：

- 降低 IK 将高抬动作分配到肘关节低限位的概率。
- 代价是肘部跟随幅度会降低，但真机会更稳定。

### 9.4 VR 退出跳变问题状态

已完成代码修复：

- 当手柄位姿不更新但还被 Vuer 标记为 ready 时，提前冻结。
- 当 Quest 断流时进入 `stale_hold`，重置 `ik_zero_q` 和 `home_q`。
- 重新进入 VR 时通过 `rearm_hold` 重新标定。

尚未完全验证：

- 今日最后测试没有完整看到 `Quest pose frozen before stale timeout` 日志。
- 需要下一次专门执行：进入 VR -> 双臂运动 -> 直接退出 VR 页面 -> 观察是否进入 `stale_hold` 或 `Quest pose frozen...`，并确认双臂没有突然跳变。

## 10. 已执行的验证命令

语法检查：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/bin/conda run --no-capture-output -n tv \
python -m py_compile tools/r1a7_vr_dual_arm_g1ik_real.py
```

结果：

- 通过。

真机启动和停止：

- 多次启动 `tools/r1a7_vr_dual_arm_g1ik_real.py`
- 多次输入 `ENABLE`
- 通过 `Ctrl-C` 停止
- 退出时日志显示：

```text
released lowcmd gains
```

## 11. 接手建议

下一位开发者优先做三件事：

1. 使用新的肘部降低参数再跑一轮真机测试：

```text
--ik_joint_scales 0.80,0.65,0.45,0.70,0.55,1,0.75,0.80,0.65,0.45,0.60,0.55,1,0.55
```

2. 专门验证 VR 退出保护：

- 进入 VR。
- 双臂运动到非初始位置。
- 直接退出 VR 或关闭 Quest 浏览器页面。
- 观察日志是否出现 `Quest pose frozen before stale timeout` 或 `stale_hold`。
- 观察真机是否保持当前位姿，不再突然摆到奇怪位置。

3. 如果仍然高抬停住，不要继续简单放大 `ik_delta_scale`。

原因：

- 当前不是链路不通，而是 G1_29 IK 输出和 R1-A7 真实关节可达空间不完全一致。
- 继续放大只会把目标推到 `shoulder_pitch` 或 `elbow` 限位。
- 更合理的方向是：
  - 降低肘部/肩部触限关节比例。
  - 调整 Quest 手柄到 G1 IK 目标位姿的坐标映射。
  - 对高抬动作单独做任务空间压缩或重映射。
  - 必要时为 R1-A7 建立独立 IK，而不是继续完全依赖 G1_29 IK。

## 12. 安全注意

- 每次启动前必须确认急停可用。
- 手柄进入 VR 后先保持自然初始位。
- 看到 `rearm_hold` 后等待进入 `relative_tracking` 再动作。
- 出现连续 `arm_limit` 时应停止测试并重新调比例。
- 退出时必须确认日志包含：

```text
released lowcmd gains
```

