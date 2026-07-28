#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/dex1_1_gripper_dds.py \
    --interface "${ROBOT_IFACE}" \
    --side right \
    --mode cycle \
    --duration 10 \
    "$@"
