#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/r1a7_right_arm_teach.py \
    --interface "${ROBOT_IFACE}" \
    --mode jog \
    --enter_debug_mode \
    --command_topic rt/lowcmd \
    --right_arm_indices 22,23,24,25,26,27,28 \
    --weight_index 31 \
    --kp 24.0 \
    --kd 1.4 \
    --hold_kp 16.0 \
    --hold_kd 1.0 \
    --max_speed_rad_s 0.20 \
    --step_rad 0.030 \
    --large_step_rad 0.100 \
    --record_hz 20.0 \
    --print_period 3.0 \
    --input_mode line \
    --label teach_jog \
    "$@"
