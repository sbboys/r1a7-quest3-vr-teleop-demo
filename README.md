# R1-A7 Quest 3 VR Teleoperation Demo

This private demo repository collects the working code and field documentation for R1-A7 teleoperation with Meta Quest 3.

Current validated demo:

```text
Quest 3 controller poses -> Unitree G1_29 IK -> R1-A7 dual arms
Quest 3 index triggers -> Dex1-1 left/right grippers
```

Validated runtime parameters:

```text
arm_velocity_limit=5.0
max_joint_offset_rad=2.00
ik_delta_scale=1.00
stale_pose_timeout=0.25
rearm_hold_time=1.0
```

## Repository Layout

```text
camera_teleop/
  Camera recognition and camera-pose teleoperation code.

vr_teleop/
  Quest 3, TeleVuer, G1_29 IK, R1-A7 dual-arm and Dex1 gripper teleoperation code.

simulation/
  IsaacLab / R1-A7 simulation configuration, DDS helpers, robot and task definitions.

tools/, action_provider/, robots/, tasks/, dds/
  Compatibility layout matching the original working tree and the documented run commands.

scripts/
  Field startup, stop, check, Dex1 service, camera and teach-in scripts.

docs/
  Demo guide, project summary, SOP, troubleshooting and runbook documents.

archive/
  Historical scripts, backups, logs and experiments kept for traceability.

third_party_notes/
  Notes and download/install helpers for large vendor dependencies that are not committed.
```

## Main Demo Entry

Primary real-robot script:

```text
tools/r1a7_vr_dual_arm_g1ik_real.py
```

Dex1 service launcher:

```text
scripts/run_dex1_1_service.sh
```

Quest 3 browser URL used in the field:

```text
https://192.168.1.127:8012/?ws=wss://192.168.1.127:8012
```

Robot control interface used in the field:

```text
enx9c69d37d0967
```

Robot control-side address used in the field:

```text
192.168.123.223
```

## Start Here

For a live demo, follow:

```text
docs/r1a7_quest3_dual_arm_dex1_demo_guide_zh.md
```

For the full project summary, follow:

```text
docs/r1a7_quest3_dual_arm_dex1_project_summary_zh.md
```

For file cleanup and repository organization notes, follow:

```text
docs/r1a7_quest3_project_file_cleanup_zh.md
```

## Safety

This repository contains real-robot control code. Before running any real-robot command:

```text
1. Clear the dual-arm and gripper workspaces.
2. Keep the emergency stop ready.
3. Confirm that no other process is publishing rt/lowcmd or rt/dex1/*/cmd.
4. Start with small, slow controller motions.
5. Stop immediately if direction, range, jitter or collision risk is abnormal.
```

## Notes

Large Orbbec SDK / Viewer binary packages are intentionally not committed. The relevant download and setup notes are kept under:

```text
third_party_notes/orbbec_gemini_336l/
```
