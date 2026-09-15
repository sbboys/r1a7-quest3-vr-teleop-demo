# R1-A7 Threading Project

This project is a separate workspace for the tweezer-assisted thin-wire grasping and threading task on Unitree R1-A7.

The first phase is hardware and calibration only:

- bind tweezers to the gripper without changing the wrench project;
- measure gripper command versus tweezer tip opening;
- verify that VR/manual control can grasp a thin wire repeatably;
- keep data, configs, and notes separate from `r1a7_wrench_project`.

Reuse the existing `tv` and `aloha` conda environments until this project needs dependencies that conflict with the existing R1-A7/ACT stack.

