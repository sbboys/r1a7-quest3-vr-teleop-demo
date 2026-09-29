# R1-A7 VR 遥操作

## 目标

通过 Quest 3、TeleVuer 和 G1_29 IK 控制 R1-A7 双臂及 Dex1 夹爪，并作为真机运动的已验证基线。

## 代码位置

- 当前运行工具：`tools/r1a7_vr_dual_arm_g1ik_real.py`
- 动作提供器：`action_provider/`
- 设备/图像服务参考：`teleimager/`、`vr_teleop/`
- 启动和检查：`scripts/`
- 操作手册：`docs/r1a7_quest3_*.md`

真机控制前必须确认只有一个 `rt/lowcmd` 发布者，并按照运行手册执行停止和恢复流程。

