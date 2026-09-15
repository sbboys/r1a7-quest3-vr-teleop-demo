# R1-A7 细线夹取与镊子绑定项目交接记录

更新日期：2026-09-10

## 当前项目位置

本项目已从原扳手 ACT 项目中隔离出来，使用独立 worktree：

```text
/home/robot/unitree_sim_isaaclab_threading
```

细线项目目录：

```text
/home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project
```

当前原则：

- 不修改原 `r1a7_wrench_project`。
- 第一阶段只做镊子绑定、夹爪开合标定、VR 手动验证和记录。
- 先复用已有 `tv` 和 `aloha` conda 环境。
- 后续如果需要安装冲突依赖，再克隆 `tv_threading` 或 `aloha_threading`。

## 已创建的关键文件

```text
r1a7_threading_project/README.md
r1a7_threading_project/config/cameras.yaml
r1a7_threading_project/config/gripper_tweezer_calibration.yaml
r1a7_threading_project/config/task_stages.yaml
r1a7_threading_project/scripts/calibrate_tweezer_gripper.py
r1a7_threading_project/transfer_control/tools/r1a7_threading_vr_lowcmd.py
r1a7_threading_project/transfer_control/tools/r1a7_lowcmd_guard.py
r1a7_threading_project/docs/threading_project_plan_zh.md
r1a7_threading_project/docs/gripper_tweezer_binding_notes_zh.md
r1a7_threading_project/docs/environment_reuse_notes_zh.md
r1a7_threading_project/docs/env_tv_before_threading_2026-09-10.yml
r1a7_threading_project/docs/env_aloha_before_threading_2026-09-10.yml
```

## 夹爪与镊子 baseline 记录

最初把“夹爪刚接触镊子但镊子尚未明显合拢”的位置定义为 contact baseline。

第一次接触 baseline：

```text
right gripper q = 0.5565
```

记录文件：

```text
r1a7_threading_project/docs/tweezer_contact_baseline_2026-09-10.json
```

随后做过小步闭合测试：

```text
baseline + 0.00 -> measured q = 0.5565
baseline - 0.03 -> measured q = 0.5362
baseline - 0.05 -> measured q = 0.5189
```

后来用户重新调整了夹爪接触镊子的位置，并重新实测 baseline：

```text
right gripper q = 0.3858
```

最新 baseline 记录文件：

```text
r1a7_threading_project/docs/tweezer_contact_baseline_latest_2026-09-10.json
```

## 夹爪标定脚本

新增脚本：

```text
r1a7_threading_project/scripts/calibrate_tweezer_gripper.py
```

用途：

- 只控制右侧 DEX1 夹爪 motor index 33。
- 可按绝对归一化开度测试。
- 也可按当前 measured q 作为 baseline 做相对闭合测试。
- 可把 baseline 写入 JSON，便于后续复查。

曾使用的 baseline 读取命令：

```bash
cd /home/robot/unitree_sim_isaaclab_threading

/home/robot/miniconda3/envs/tv/bin/python \
  r1a7_threading_project/scripts/calibrate_tweezer_gripper.py \
  --interface enp6s0 \
  --relative_offsets_q 0 \
  --velocity_limit_q_per_s 0.05 \
  --hold_s 0.5 \
  --baseline_out r1a7_threading_project/docs/tweezer_contact_baseline_latest_2026-09-10.json \
  --assume_yes
```

## VR 控制脚本修改

细线项目专用 VR 控制脚本：

```text
r1a7_threading_project/transfer_control/tools/r1a7_threading_vr_lowcmd.py
```

它来自已验证的 R1-A7 TeleVuer LowCmd 控制路径，但为了细线项目做了局部修改。

新增参数：

```text
--gripper-initial-mode current
--right-gripper-open-cap-current
```

原因：

原始 VR 控制逻辑中，右 trigger 松开会映射为完全打开，右夹爪目标会回到约 `4.80`。这会破坏镊子与夹爪刚接触的初始状态。

已观察到的问题：

```text
tracking 启动后，右夹爪从接触位逐步打开到 4.80。
```

修复方式：

- `--gripper-initial-mode current`：启动时用实测 gripper q 作为夹爪目标。
- `--right-gripper-open-cap-current`：右 trigger 松开时，右夹爪最多只能回到启动时的当前 q，不允许打开到全开位置。

## 记录字段扩展

VR 记录器已扩展 `states.csv` 字段。现在会记录：

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
- 是否触发 contact hold。
- 夹爪是否真的随 trigger 闭合。

## VR 启动地址与网络状态

当前已验证机器人 DDS 网卡：

```text
enp6s0 = 192.168.123.223/24
```

当前 Quest 访问地址曾验证可进入：

```text
https://192.168.1.111:8012/?ws=wss://192.168.1.111:8012
```

注意：PC 同时出现过两个 `192.168.1.x` 地址：

```text
enx9c69d37d0967 = 192.168.1.111/24
wlx90de8048405a = 192.168.1.114/24
```

如果 Quest 无法进入 `192.168.1.111`，可尝试：

```text
https://192.168.1.114:8012/?ws=wss://192.168.1.114:8012
```

## 推荐重新启动 VR 控制命令

先清掉残留进程：

```bash
cd /home/robot/unitree_sim_isaaclab_threading

pkill -f 'r1a7_threading_vr_lowcmd.py' || true
sleep 1
ss -ltnp | grep ':8012' || echo '8012 free'
```

如果仍显示端口占用，查看并手动 kill 对应 PID：

```bash
ps -eo pid,ppid,stat,comm,args | grep -E 'r1a7_threading_vr_lowcmd|8012' | grep -v grep
kill <PID>
sleep 1
ss -ltnp | grep ':8012' || echo '8012 free'
```

启动细线专用 VR 控制：

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

启动后流程：

```text
1. Quest 打开访问地址。
2. 终端输入 ENABLE。
3. 在 Quest 中对齐手柄和机器人手臂。
4. 释放右手 A，再按右手 A 开始 TeleVuer tracking。
5. 左手 A 开始记录。
6. 操作右 trigger，观察镊子是否随夹爪闭合。
7. 左手 A 停止并保存记录。
8. 终端输入 q 退出。
```

## 当前已知问题

1. 如果程序停在 `Type ENABLE...`，Quest 页面能进但机器人不会响应。需要在终端输入 `ENABLE`。
2. 如果还没按右手 A 或终端 R，程序只是在 waiting_start，不会发布跟踪控制。
3. 原始 trigger 逻辑会使右夹爪松开时回到全开；必须使用细线脚本并带上 `--right-gripper-open-cap-current`。
4. 如果 8012 被占用，需要先停止旧 `r1a7_threading_vr_lowcmd.py` 进程。
5. 若 Quest 无法访问 192.168.1.111，可改试 192.168.1.114。

## 下一步建议

1. 重新确认当前夹爪与镊子的接触状态。
2. 用 `calibrate_tweezer_gripper.py` 重新记录最新 baseline。
3. 启动 VR 控制，确保右夹爪不会自动打开到 4.80。
4. 左手 A 开始记录，右 trigger 慢速闭合镊子。
5. 结束后检查 `/data/R1A7/threading_pilot_v1/.../states.csv` 中的夹爪字段。
6. 根据 `measured_gripper_q`、`sent_gripper_q` 和现场观察，确定下一轮安全闭合范围。

