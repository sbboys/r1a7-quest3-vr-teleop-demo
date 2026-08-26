# R1-A7 Wrench Baseline Current System Inventory

Generated from the local checkout and prior verified handoff docs. No robot motion was commanded while creating this inventory.

## Repository

- Local path: `/home/robot/unitree_sim_isaaclab`
- Current working branch for this task: `r1a7-wrench-baseline`
- Origin remote: `https://github.com/unitreerobotics/unitree_sim_isaaclab.git`
- Demo remote: `git@github.com:sbboys/r1a7-quest3-vr-teleop-demo.git`
- Important warning: do not push private R1-A7 changes to the Unitree `origin`.

## Confirmed Runtime Files

- VR dual-arm and gripper control entry: `tools/r1a7_vr_dual_arm_g1ik_real.py`
- Right-arm DDS smoke test: `tools/r1a7_right_arm_comm.py`
- Internal lowcmd gripper test: `tools/r1a7_lowcmd_gripper_test.py`
- Demo guide: `docs/r1a7_quest3_dual_arm_dex1_demo_guide_zh.md`
- Handoff record: `docs/r1a7_quest3_vr_handoff_2026-07-31_zh.md`
- Camera/arm/gripper runbook: `docs/r1a7_camera_arm_gripper_runbook_zh.md`

## Confirmed Interfaces

- DDS domain: `0`
- Robot interface used in prior validation: `enx9c69d37d0967`
- State topic: `rt/lowstate`
- Command topic used by validated dual-arm/gripper path: `rt/lowcmd`
- Right-arm smoke-test default state topic: `rt/lowstate`
- Right-arm smoke-test default arm SDK topic: `rt/arm_sdk`

## Confirmed Joint Mapping

- Left arm motor indices: `15,16,17,18,19,20,21`
- Right arm motor indices: `22,23,24,25,26,27,28`
- Arm joint order:
  - `left_shoulder_pitch`
  - `left_shoulder_roll`
  - `left_shoulder_yaw`
  - `left_elbow`
  - `left_wrist_roll`
  - `left_wrist_pitch`
  - `left_wrist_yaw`
  - `right_shoulder_pitch`
  - `right_shoulder_roll`
  - `right_shoulder_yaw`
  - `right_elbow`
  - `right_wrist_roll`
  - `right_wrist_pitch`
  - `right_wrist_yaw`

## Confirmed Gripper Mapping

- Internal lowcmd Dex1 left gripper motor index: `31`
- Internal lowcmd Dex1 right gripper motor index: `33`
- Left open/close q from prior demo: `4.86` / `-0.08`
- Right open/close q from prior demo: `4.80` / `-0.20`
- Gripper velocity limit used in prior demo: `8.0`
- Gripper gains used in prior demo: `kp=25.0`, `kd=0.8`

## Confirmed Safety Behavior In Existing Code

- Existing true-motion scripts require explicit interactive `ENABLE`.
- Existing true-motion scripts release lowcmd gains on exit.
- Existing VR path has stale-pose and rearm-hold handling.
- Existing right-arm smoke test defaults to read-only monitor mode.

## Wrench Baseline Constraints

- New wrench code defaults to `backend=mock`.
- New wrench code defaults to `dry_run: true`.
- New wrench code defaults to `enable_robot_motion: false`.
- Real adapter is intentionally blocked until the lowcmd wrapper and current safety thresholds are explicitly confirmed.

## Unknown Or Unconfirmed

- Firmware version.
- Official units and safe thresholds for joint torque/current fields.
- Final wrench fixture coordinates.
- `base_to_nut` and `right_gripper_to_wrench_tool`.
- Reliable contact detection source for nut seating.
- Safe torque/current stop limits for tightening.

## Technician Questions

1. Confirm lowstate torque/current field units for R1-A7 arm and internal Dex1 motors.
2. Confirm recommended current or torque limits for low-speed contact and wrench rotation.
3. Confirm whether motor indices `31` and `33` remain stable across firmware modes.
4. Confirm whether `rt/lowcmd` is the preferred path for fixed-workcell scripted manipulation, or whether `rt/arm_sdk` should be used for arm joints.
5. Confirm the official safe-stop sequence after a contact or current-limit event.
