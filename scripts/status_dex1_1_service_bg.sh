#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PID_FILE="${PID_FILE:-Log/dex1_1_service.pid}"
LOG_FILE="${LOG_FILE:-Log/dex1_1_service.log}"

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
    echo "[DEX1 SERVICE BG] running: pid=${pid}"
  else
    echo "[DEX1 SERVICE BG] pid file exists but process is not running"
  fi
else
  echo "[DEX1 SERVICE BG] no pid file"
fi

ps -ef | rg -i 'dex1_1_gripper_server' || true
if [[ -f "${LOG_FILE}" ]]; then
  echo "[DEX1 SERVICE BG] recent log:"
  tail -n 40 "${LOG_FILE}" || true
fi
