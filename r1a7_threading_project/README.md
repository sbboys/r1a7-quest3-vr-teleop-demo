# R1-A7 Threading Project

This project is a separate workspace for the tweezer-assisted thin-wire grasping and threading task on Unitree R1-A7.

The first phase is hardware and calibration only:

- bind tweezers to the gripper without changing the wrench project;
- measure gripper command versus tweezer tip opening;
- verify that VR/manual control can grasp a thin wire repeatably;
- keep data, configs, and notes separate from `r1a7_wrench_project`.

Reuse the existing `tv` and `aloha` conda environments until this project needs dependencies that conflict with the existing R1-A7/ACT stack.

## R1-A7 arm console

The repository also contains the local 3D manual-control console developed for
R1-A7 arm bring-up and thin-wire task preparation:

- `console3d/`: the original full-featured console on port `8097`;
- `console3d_hierarchical/`: the current terminal-baseline console on port `8098`;
- `docs/r1a7_arm_console_work_summary_2026-09-22_zh.md`: complete Chinese work log, issue history, operating procedure, and safety boundaries;
- `docs/r1a7_arm_console_github_manifest_2026-09-22_zh.md`: selected upload manifest.

## Dual-camera visual servo archive

The 2026-09-24 to 2026-09-29 dual-camera visual servo work is documented in:

- `docs/r1a7_dual_camera_visual_servo_work_summary_2026-09-29_zh.md`: Chinese work summary, experiment results, file operations, run commands, risks, and next steps;
- `docs/r1a7_dual_camera_visual_servo_upload_manifest_2026-09-29_zh.md`: upload boundary, excluded large files, environment requirements, and reproducibility checks;
- `docs/source_records/`: the two original Word records used for the summary.

The active visual-servo files are under `vision_dual_camera/task6_threading/`.
The SAM2 source tree and model checkpoints remain local dependencies and are not
included in this snapshot.

## 2026-10-10 visual-servo and low-speed-control continuation

The work after the 2026-09-29 visual-servo archive is indexed in:

- `docs/r1a7_visual_servo_control_handoff_2026-10-10_zh.md`: Chinese handoff covering low-speed joint diagnosis, DQIK/Kd/POSVEL, W/U task analysis, Wrist Hole/Tip tracking repairs, and the verified 10 mm/s baseline;
- `docs/r1a7_visual_servo_control_upload_manifest_2026-10-10_zh.md`: selected source files, source-record hashes, excluded runtime data, and validation boundary;
- `docs/source_records/2026-10-07_to_2026-10-10/`: seven original Word work records.

The stable experimental robot controller is
`transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py`.
The active visual stack remains under `vision_dual_camera/task6_threading/` and
can be launched with `run_task6_visual_stack.sh`. The 10 mm/s configuration was
confirmed in the recorded test; 15 and 25 mm/s require new qualification after
the visual-state-machine fixes.

## Complete project snapshot

The complete engineering snapshot is documented in
`docs/r1a7_project_snapshot_2026-09-22_zh.md`. It indexes the controller,
console, gripper/tweezer calibration, single- and dual-camera visual guidance,
hand-eye/TCP calibration, research material, and archived source records.
