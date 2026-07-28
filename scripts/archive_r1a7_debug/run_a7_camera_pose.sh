#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
PYTHON_CMD="${PYTHON_CMD:-python}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_CMD}" -u sim_main.py \
  --device cuda:0 \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-A7-Joint \
  --robot_type a7 \
  --action_source camera_pose \
  --camera_pose_robot_arm left \
  --camera_pose_human_hand right \
  --camera_pose_show \
  --camera_pose_mirror_view \
  --camera_pose_mirror_input \
  --camera_pose_debug \
  --camera_pose_direct_planar \
  --camera_pose_direct_view_vertical \
  --camera_pose_direct_view_horizontal \
  --camera_pose_direct_roll_gain 1.8 \
  --camera_pose_direct_yaw_gain 1.4 \
  --camera_pose_direct_pitch_gain 4.0 \
  --camera_pose_direct_elbow_gain 1.0 \
  --camera_pose_direct_depth_gain 1.8 \
  --camera_pose_direct_depth_sign -1.0 \
  --camera_pose_direct_pitch_depth_ratio 0.35 \
  --camera_pose_direct_elbow_depth_ratio -0.85 \
  --camera_pose_direct_elbow_vertical_ratio 0.15 \
  --camera_pose_direct_skeleton_lift_gain 1.10 \
  --camera_pose_direct_skeleton_side_roll_gain 0.95 \
  --camera_pose_direct_skeleton_side_yaw_gain 0.45 \
  --camera_pose_direct_skeleton_reach_pitch_gain 0.25 \
  --camera_pose_direct_skeleton_reach_yaw_gain 0.35 \
  --camera_pose_direct_skeleton_elbow_gain 0.85 \
  --camera_pose_direct_skeleton_lift_elbow_gain 0.20 \
  --camera_pose_direct_max_step 0.075 \
  --camera_pose_command_lead 0.32 \
  --camera_pose_lost_hold_s 0.55 \
  --camera_pose_lost_return_s 1.40 \
  --camera_pose_lock_wrist \
  --camera_pose_torso_safe \
  --camera_pose_min_visibility 0.05 \
  --camera_pose_min_wrist_shoulder_m 0.025 \
  --camera_pose_scale 0.8 \
  "$@"
