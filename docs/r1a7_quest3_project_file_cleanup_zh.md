# R1-A7 Quest3 双臂与 Dex1 双夹爪项目文件整理说明

整理时间：2026-07-28 13:34

本次整理依据：

```text
/home/robot/桌面/jiqiren/R1-A7_Quest3_双臂与Dex1双夹爪_VR遥操作项目完整汇总与SOP_V2.0.docx
```

当前目标是让项目中与 Quest 3 手柄控制 R1-A7 双臂和 Dex1 双夹爪相关的正式入口清晰，减少历史备份、旧实验脚本、一次性修复脚本、旧日志和下载目录重复脚本对后续操作的干扰。

## 当前正式保留文件

正式联合控制入口：

```text
tools/r1a7_vr_dual_arm_g1ik_real.py
```

用途：

```text
Quest 3 左右手柄位姿 -> G1_29 IK -> R1-A7 双臂
Quest 左右食指扳机 -> Dex1 左右夹爪
```

该脚本当前已确认包含：

```text
--enable_gripper
--stale_pose_timeout
--rearm_hold_time
--pose_change_eps
Dex1 trigger control
```

夹爪单独测试入口：

```text
tools/r1a7_quest_dex1_trigger_only_test.py
```

用途：

```text
只测试 Quest 左右扳机控制 Dex1 左右夹爪，不发布 rt/lowcmd，不移动双臂。
```

Dex1 DDS 辅助测试：

```text
tools/dex1_1_gripper_dds.py
```

Dex1 服务启动与管理：

```text
scripts/run_dex1_1_service.sh
scripts/start_dex1_1_service_bg.sh
scripts/status_dex1_1_service_bg.sh
scripts/stop_dex1_1_service_bg.sh
```

当前 SOP：

```text
docs/r1a7_quest3_g1ik_dual_arm_teleop_zh.md
```

## 已归档位置

本次没有直接删除历史脚本，而是统一归档到：

```text
archive/cleanup_20260728_133418/
```

归档目录说明：

```text
backups/              历史 .bak、backup、pre-Dex1 等脚本备份
experimental_tools/   MuJoCo、preview、Cartesian、pose IK、right-arm-only 等早期实验脚本
root_oneoff/          根目录下一次性安装、修复、替换脚本
logs/                 旧运行日志
downloads/            /home/robot/下载 中已被 tools 正式文件替代的重复脚本
```

归档目录内的 `README.md` 记录了正式入口和恢复方式。

## 已清理内容

已清理 Python 缓存：

```text
__pycache__/
*.pyc
*.pyo
```

已归档明显历史备份：

```text
action_provider/*.bak*
action_provider/*.backup*
robots/*.bak*
robots/*.backup*
sim_main.py.*.bak
tools/r1a7_vr_dual_arm_g1ik_real_backup*.py
```

已归档不再作为正式入口的早期实验脚本：

```text
tools/r1a7_mujoco_unitree_g1_ik_controller.py
tools/r1a7_mujoco_vr_controller.py
tools/r1a7_vr_dual_arm_cartesian_real.py
tools/r1a7_vr_dual_arm_pose_real.py
tools/r1a7_vr_ik_preview.py
tools/r1a7_vr_right_arm_real.py
```

已归档下载目录重复脚本：

```text
/home/robot/下载/r1a7_quest_dex1_trigger_only_test.py
/home/robot/下载/r1a7_vr_dual_arm_g1ik_real.py
/home/robot/下载/r1a7_vr_dual_arm_g1ik_real_dex1_trigger.py
/home/robot/下载/r1a7_vr_dual_arm_g1ik_real_dex1_trigger (1).py
```

## 暂未清理内容

以下内容本次暂未处理，因为它们可能仍属于相机识别、IsaacLab 任务、右臂示教或 Dex1 维护链路：

```text
action_provider/
robots/
tasks/
dds/
scripts/run_r1a7_camera_real_control*.sh
scripts/run_r1a7_right_arm_teach_*.sh
tools/r1a7_camera_real_teleop.py
tools/r1a7_right_arm_teach.py
tools/r1a7_arm_sdk_status.py
tools/r1a7_dual_arm_ik.py
```

如果后续只保留“Quest 3 双臂 + Dex1 双夹爪”最小项目，可以再做第二轮清理，将相机识别、示教、IsaacLab 旧任务相关文件单独归档。

## 验证结果

已执行语法检查：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/bin/conda run --no-capture-output -n tv \
python -m py_compile \
  tools/r1a7_vr_dual_arm_g1ik_real.py \
  tools/r1a7_quest_dex1_trigger_only_test.py \
  tools/dex1_1_gripper_dds.py
```

结果：通过。

已确认正式联合脚本中存在：

```text
--enable_gripper
--stale_pose_timeout
--pose_change_eps
Dex1 trigger control
```

## 当前推荐运行入口

启动 Dex1 服务：

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/run_dex1_1_service.sh
```

启动最终联合控制：

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
