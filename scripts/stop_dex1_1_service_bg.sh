#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PID_FILE="${PID_FILE:-Log/dex1_1_service.pid}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "[DEX1 SERVICE BG] no pid file: ${PID_FILE}"
  pkill -f dex1_1_gripper_server 2>/dev/null || true
  exit 0
fi

pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
  echo "[DEX1 SERVICE BG] stopping pid=${pid}"
  kill "${pid}" 2>/dev/null || true
  sleep 0.5
  if ps -p "${pid}" >/dev/null 2>&1; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
fi

rm -f "${PID_FILE}"
pkill -f dex1_1_gripper_server 2>/dev/null || true
echo "[DEX1 SERVICE BG] stopped"
