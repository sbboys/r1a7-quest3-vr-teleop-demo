#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"
SKIP_R1A7_PREFLIGHT="${SKIP_R1A7_PREFLIGHT:-0}"

if [[ "${SKIP_R1A7_PREFLIGHT}" != "1" ]]; then
  ./scripts/r1a7_real_arm_preflight.sh
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/r1a7_right_arm_comm.py \
    --interface "${ROBOT_IFACE}" \
    --idl hg \
    --state_topic rt/lowstate \
    --command_topic rt/arm_sdk \
    --r1_arm_sdk \
    --enter_debug_mode \
    --enable_control \
    --mode test \
    --duration 8 \
    --test_amplitude_deg 3 \
    --test_frequency_hz 0.08 \
    --kp 40.0 \
    --kd 2.0 \
    --max_speed_rad_s 0.15 \
    "$@"
