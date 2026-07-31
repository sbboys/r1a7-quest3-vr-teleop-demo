#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/r1a7_vr_right_arm_real.py \
    --interface "${ROBOT_IFACE}" \
    --domain_id 0 \
    --state_topic rt/lowstate \
    --command_topic rt/lowcmd \
    --enter_debug_mode \
    --swap_hands \
    --duration 8 \
    --scale 0.12 \
    --max_delta_m 0.04 \
    --kp 16.0 \
    --kd 1.0 \
    --hold_kp 10.0 \
    --hold_kd 0.8 \
    --max_speed_rad_s 0.08 \
    --max_command_lead 0.08 \
    "$@"
