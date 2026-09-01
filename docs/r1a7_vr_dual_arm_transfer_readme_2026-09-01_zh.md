# R1-A7 VR 双臂遥操作迁移说明

这个包用于把当前电脑上已经打通的 R1-A7 + Quest3 + TeleVuer + R1A7_ArmIK + Unitree `rt/lowcmd` 实机双臂遥操作链路迁移到另一台电脑。

## 1. 当前成功链路

数据流是：

```text
Quest3 手柄位姿
-> TeleVuer WebXR 页面
-> R1A7_ArmIK
-> R1-A7 双臂 14 个关节目标角
-> Unitree SDK2 Python DDS
-> rt/lowcmd
-> R1-A7 实机双臂
```

默认实机遥操作入口：

```bash
cd /home/version/下载/unitree_sdk2-main
bash tools/run_r1a7_unitree_official_vuer_real_lowcmd.sh
```

默认带数据记录入口：

```bash
cd /home/version/下载/unitree_sdk2-main
bash tools/run_r1a7_unitree_official_vuer_real_lowcmd_record.sh
```

启动后，在 Quest3 浏览器里打开终端打印的地址，通常形如：

```text
https://电脑Wi-Fi-IP:8012/?ws=wss://电脑Wi-Fi-IP:8012
```

终端输入 `ENABLE` 以后，程序仍然不会马上动机器人。进入 Quest 页面后，按右手柄 `A` 或左手柄 `X` 才开始跟随。

## 2. 目录结构

```text
R1A7_VR_dual_arm_transfer_20260831_001/
├── unitree_sdk2-main/
│   ├── tools/
│   ├── sim/mujoco_r1/models/
│   ├── datasets/dataset_setup_001/notes/
│   └── experiments/r1a7_quest3_official_vuer_lowcmd_rebase_v3_2026-08-02/
└── robot_dev/
    ├── xr_teleoperate/
    └── unitree_sdk2_python/
```

## 3. 核心文件说明

| 文件 | 原始路径 | 作用 |
|---|---|---|
| `run_r1a7_unitree_official_vuer_real_lowcmd.sh` | `unitree_sdk2-main/tools/` | 实机遥操作启动脚本。自动检测 Quest 使用的 Wi-Fi IP，启动 TeleVuer 和实机低层控制程序。 |
| `r1a7_unitree_official_vuer_real_lowcmd.py` | `unitree_sdk2-main/tools/` | 实机遥操作主程序。读取 Quest 手柄，调用 `R1A7_ArmIK`，通过 `rt/lowcmd` 发布双臂关节命令。 |
| `run_r1a7_unitree_official_vuer_real_lowcmd_record.sh` | `unitree_sdk2-main/tools/` | 带数据记录的实机遥操作启动脚本。默认速度 `0.8 rad/s`，用于采集 episode。 |
| `r1a7_unitree_official_vuer_real_lowcmd_record.py` | `unitree_sdk2-main/tools/` | 带记录功能的实机遥操作主程序。记录 VR、IK、lowstate、lowcmd、相机时间戳等数据。 |
| `r1a7_lowcmd_guard.py` | `unitree_sdk2-main/tools/` | 防止多个 `rt/lowcmd` 发布器同时运行，避免机器人命令冲突。 |
| `r1a7_dex1_trigger_controller.py` | `unitree_sdk2-main/tools/` | DEX1-1 夹爪扳机控制模块。只有启动记录程序时加 `--enable-dex1` 才会使用。 |
| `r1a7_unitree_episode_recorder.py` | `unitree_sdk2-main/tools/` | Unitree 风格 episode 记录封装。 |
| `score_r1a7_episode_quality.py` | `unitree_sdk2-main/tools/` | 对采集 episode 做质量评分。 |
| `robot_arm_ik.py` | `robot_dev/xr_teleoperate/teleop/robot_control/` | IK 核心文件。当前使用的是 `R1A7_ArmIK`，基于 G1 官方 IK 风格，用 R1-A7 URDF 适配。 |
| `weighted_moving_filter.py` | `robot_dev/xr_teleoperate/teleop/utils/` | IK 输出平滑滤波。 |
| `episode_writer.py` | `robot_dev/xr_teleoperate/teleop/utils/` | 官方风格 episode 写入器。 |
| `televuer.py` / `tv_wrapper.py` | `robot_dev/xr_teleoperate/teleop/televuer/src/televuer/` | Quest3 WebXR/TeleVuer 连接入口。 |
| `A7.urdf` | `unitree_sdk2-main/sim/mujoco_r1/models/` | R1-A7 机器人 URDF，IK 建模依赖这个文件。 |
| `r1a7_model_cache.pkl` | `robot_dev/xr_teleoperate/teleop/` | R1-A7 IK 模型缓存，加快启动。 |
| `unitree_sdk2py/` | `robot_dev/unitree_sdk2_python/` | Unitree SDK2 Python DDS 通信库，负责 `rt/lowstate` 订阅和 `rt/lowcmd` 发布。 |

## 4. 另一台电脑需要注意的路径

当前脚本里有几个固定路径。最省事的迁移方式是让另一台电脑也使用相同路径：

```text
/home/version/下载/unitree_sdk2-main
/home/version/robot_dev/xr_teleoperate
/home/version/robot_dev/unitree_sdk2_python
```

如果另一台电脑用户名或目录不同，需要改这些地方：

```text
tools/run_r1a7_unitree_official_vuer_real_lowcmd.sh
tools/run_r1a7_unitree_official_vuer_real_lowcmd_record.sh
tools/r1a7_unitree_official_vuer_real_lowcmd.py
tools/r1a7_unitree_official_vuer_real_lowcmd_record.py
robot_dev/xr_teleoperate/teleop/robot_control/robot_arm_ik.py
```

尤其注意 `robot_arm_ik.py` 中的 R1-A7 URDF 路径：

```python
self.urdf_path = "/home/version/下载/unitree_sdk2-main/sim/mujoco_r1/models/A7.urdf"
self.model_dir = "/home/version/下载/unitree_sdk2-main/sim/mujoco_r1/models/"
```

## 5. 环境依赖

这个包只包含项目文件，不包含完整 Conda 环境。另一台电脑需要准备 Python 环境，当前电脑使用的是 `tv` 环境。

主要 Python 依赖包括：

```text
numpy
pinocchio
casadi
meshcat
vuer / televuer 相关依赖
opencv-python 或 opencv-contrib-python
unitree_sdk2py
```

DDS 网络要求：

```text
机器人网口通常是 eno1
机器人 lowstate topic: rt/lowstate
机器人 command topic: rt/lowcmd
domain-id 默认 0
Quest3 和电脑需要在同一 Wi-Fi 或热点网络下
```

## 6. 实机启动顺序

1. 连接机器人网线，确认 `eno1` 是 up。
2. 连接 Quest3 和电脑到同一个 Wi-Fi/热点。
3. 在电脑终端运行：

```bash
cd /home/version/下载/unitree_sdk2-main
bash tools/run_r1a7_unitree_official_vuer_real_lowcmd.sh
```

4. 终端看到 Quest URL 后，在 Quest3 浏览器打开该地址。
5. 终端输入：

```text
ENABLE
```

6. Quest3 进入 `Enter AR/VR teleop`。
7. 手柄对齐机器人当前双臂姿态。
8. 按右手柄 `A` 或左手柄 `X` 开始跟随。
9. 结束时先退出 Quest 网页，再在终端 `Ctrl+C`。如有残留进程，需要人工检查后结束。

## 7. 带记录启动顺序

```bash
cd /home/version/下载/unitree_sdk2-main
R1A7_RECORD_TASK_DIR=datasets/dataset_setup_001/unitree_xr_data/right_fixed_pin_socket_alignment \
bash tools/run_r1a7_unitree_official_vuer_real_lowcmd_record.sh
```

这个版本会同时控制机器人和记录数据。默认逻辑是按右手柄 `A` 或左手柄 `X` 后，遥操作和记录同时开始。退出 Quest 网页后保存 episode。

采集后评分：

```bash
cd /home/version/下载/unitree_sdk2-main
python tools/score_r1a7_episode_quality.py \
  --task-dir datasets/dataset_setup_001/unitree_xr_data/right_fixed_pin_socket_alignment \
  --summary-csv datasets/dataset_setup_001/unitree_xr_data/right_fixed_pin_socket_alignment/quality_summary.csv \
  --print-details
```

## 8. 安全注意事项

这条链路会直接向 R1-A7 发布 `rt/lowcmd`。运行前必须确认：

```text
只有一个遥操作程序在运行
只有一个 8012 TeleVuer 服务在运行
机器人双臂工作空间清空
急停随时可用
不要同时运行 MuJoCo 镜像控制、旧版 real_only、holding 或其他 lowcmd 发布程序
```

`r1a7_lowcmd_guard.py` 会拦截一部分已知冲突进程，但不能替代人工安全检查。

## 9. 当前包没有包含的内容

这个包没有包含：

```text
Conda 环境本体
已经采集的大数据集
Orbbec SDK 二进制库
Qwen/VLM 大模型权重
整个 unitree_sdk2 C++ 编译产物
```

如果另一台电脑也要做双相机采集、VLM、Orbbec 可视化或完整训练，需要额外迁移这些依赖。

## 10. 推荐先做的验证

在另一台电脑上不要一上来直接动机器人，建议按顺序检查：

```text
1. Python 能 import unitree_sdk2py
2. Python 能 import televuer
3. Python 能 import teleop.robot_control.robot_arm_ik.R1A7_ArmIK
4. 能收到 rt/lowstate
5. Quest3 能打开 8012 页面并进入 VR
6. 确认无其他 lowcmd 发布器
7. 再执行实机遥操作
```

