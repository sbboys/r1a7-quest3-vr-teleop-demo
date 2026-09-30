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
- `docs/r1a7_arm_console_github_manifest_2026-09-22_zh.md`: selected upload manifest;
- `docs/r1a7_arm_console_direction_switch_correction_2026-09-30_zh.md`: corrected web command timing for continuous `+Z` to `+X` direction changes.
