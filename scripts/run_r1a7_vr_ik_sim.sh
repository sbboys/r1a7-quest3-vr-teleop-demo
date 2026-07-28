#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
HOST_IP="${HOST_IP:-192.168.1.127}"
TASK="${TASK:-Isaac-PickPlace-Cylinder-A7-Joint}"
ROBOT_TYPE="${ROBOT_TYPE:-a7}"
VR_INPUT_MODE="${VR_INPUT_MODE:-hand}"
VR_DISPLAY_MODE="${VR_DISPLAY_MODE:-pass-through}"
STEP_HZ="${STEP_HZ:-25}"
RENDER_INTERVAL="${RENDER_INTERVAL:-2}"
CAMERA_WRITE_INTERVAL="${CAMERA_WRITE_INTERVAL:-4}"

export PYTHONNOUSERSITE=1
export XR_TELEOP_ROOT="${XR_TELEOP_ROOT:-/home/robot/xr_teleoperate}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u sim_main.py \
    --device cpu \
    --enable_cameras \
    --task "${TASK}" \
    --robot_type "${ROBOT_TYPE}" \
    --action_source vr_ik \
    --step_hz "${STEP_HZ}" \
    --render_interval "${RENDER_INTERVAL}" \
    --camera_write_interval "${CAMERA_WRITE_INTERVAL}" \
    --vr_ik_input_mode "${VR_INPUT_MODE}" \
    --vr_ik_display_mode "${VR_DISPLAY_MODE}" \
    --vr_ik_scale 0.18 \
    --vr_ik_max_delta_m 0.12 \
    --vr_ik_cartesian_step 0.010 \
    --vr_ik_joint_step 0.018 \
    --vr_ik_joint_speed 0.55 \
    --vr_ik_command_lead 0.08 \
    --vr_ik_damping 0.08 \
    --vr_ik_filter_alpha 0.55 \
    --public_ip "${HOST_IP}" \
    "$@"
