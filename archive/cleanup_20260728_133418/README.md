# R1-A7 Quest3 VR cleanup 20260728_133418

This archive contains files moved out of the active project tree during cleanup.
Nothing here is required by the current SOP path for Quest 3 dual-arm + Dex1 dual-gripper teleoperation.

Current active entry points:

- `tools/r1a7_vr_dual_arm_g1ik_real.py`: final real R1-A7 dual-arm + optional Dex1 trigger control.
- `tools/r1a7_quest_dex1_trigger_only_test.py`: Quest trigger-only Dex1 test, no `rt/lowcmd` arm control.
- `tools/dex1_1_gripper_dds.py`: Dex1 DDS helper/test.
- `scripts/run_dex1_1_service.sh`: foreground Dex1 service startup.
- `scripts/start_dex1_1_service_bg.sh`, `scripts/status_dex1_1_service_bg.sh`, `scripts/stop_dex1_1_service_bg.sh`: background Dex1 service management.
- `docs/r1a7_quest3_g1ik_dual_arm_teleop_zh.md`: current SOP summary.

Archive folders:

- `backups/`: old timestamped `.bak`, `.backup`, and pre-Dex1 versions.
- `experimental_tools/`: MuJoCo, preview, Cartesian, pose-IK, and right-arm-only scripts from earlier experiments.
- `root_oneoff/`: one-off installation/repair scripts from earlier IsaacLab/camera bring-up.
- `logs/`: old runtime logs.
- `downloads/`: duplicate downloaded script copies that were superseded by files in `tools/`.

To restore an archived file, move it back manually from this archive.
