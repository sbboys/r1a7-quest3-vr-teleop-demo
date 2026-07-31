#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"
DURATION="${DURATION:-3}"
DEX1_MAX_STEP="${DEX1_MAX_STEP:-0.50}"
LOG_DIR="${LOG_DIR:-Log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/dex1_1_service.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/dex1_1_service.log}"
STOP_DEX1_ON_EXIT="${STOP_DEX1_ON_EXIT:-1}"

started_dex1=0

cleanup() {
  local status=$?
  if [[ "${started_dex1}" == "1" && "${STOP_DEX1_ON_EXIT}" == "1" ]]; then
    PID_FILE="${PID_FILE}" ./scripts/stop_dex1_1_service_bg.sh || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

if [[ ! -f "${PID_FILE}" ]] || ! ps -p "$(cat "${PID_FILE}" 2>/dev/null || echo 0)" >/dev/null 2>&1; then
  PID_FILE="${PID_FILE}" LOG_FILE="${LOG_FILE}" ./scripts/start_dex1_1_service_bg.sh
  started_dex1=1
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/dex1_1_gripper_dds.py \
    --interface "${ROBOT_IFACE}" \
    --side right \
    --mode open \
    --duration "${DURATION}" \
    --max_step "${DEX1_MAX_STEP}"
