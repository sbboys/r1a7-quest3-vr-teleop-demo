#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"
DEX1_DIR="${DEX1_DIR:-/home/robot/IsaacLab/unitree_robots/dex1_1_service}"
DURATION="${DURATION:-8}"
LOG_DIR="${LOG_DIR:-Log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/dex1_1_service.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/dex1_1_service.log}"
STOP_DEX1_ON_EXIT="${STOP_DEX1_ON_EXIT:-1}"

started_dex1=0

cleanup() {
  local status=$?
  if [[ -n "${test_pid:-}" ]] && ps -p "${test_pid}" >/dev/null 2>&1; then
    kill "${test_pid}" 2>/dev/null || true
    sleep 0.2
    kill -9 "${test_pid}" 2>/dev/null || true
  fi
  if [[ "${started_dex1}" == "1" && "${STOP_DEX1_ON_EXIT}" == "1" ]]; then
    PID_FILE="${PID_FILE}" ./scripts/stop_dex1_1_service_bg.sh || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

if [[ ! -x "${DEX1_DIR}/bin/test_dex1_1_gripper_server" ]]; then
  echo "[DEX1 OFFICIAL TEST] missing ${DEX1_DIR}/bin/test_dex1_1_gripper_server" >&2
  exit 2
fi

if [[ ! -f "${PID_FILE}" ]] || ! ps -p "$(cat "${PID_FILE}" 2>/dev/null || echo 0)" >/dev/null 2>&1; then
  PID_FILE="${PID_FILE}" LOG_FILE="${LOG_FILE}" ./scripts/start_dex1_1_service_bg.sh
  started_dex1=1
fi

echo "[DEX1 OFFICIAL TEST] running official right gripper cycle for ${DURATION}s"
"${DEX1_DIR}/bin/test_dex1_1_gripper_server" --network "${ROBOT_IFACE}" --right &
test_pid="$!"
sleep "${DURATION}"
