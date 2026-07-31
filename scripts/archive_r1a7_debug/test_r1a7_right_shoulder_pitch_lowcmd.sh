#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/r1a7_right_arm_comm.py \
    --interface "${ROBOT_IFACE}" \
    --domain_id 0 \
    --idl hg \
    --state_topic rt/lowstate \
    --command_topic rt/lowcmd \
    --enable_control \
    --debug_lowcmd \
    --enter_debug_mode \
    --mode test \
    --duration 12 \
    --right_arm_indices 22,23,24,25,26,27,28 \
    --weight_index 31 \
    --kp 28.0 \
    --kd 1.5 \
    --hold_kp 20.0 \
    --hold_kd 1.0 \
    --max_speed_rad_s 0.30 \
    --test_amplitude_deg 15.0 \
    --test_frequency_hz 0.08 \
    "$@"
