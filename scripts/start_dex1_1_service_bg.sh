#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="${LOG_DIR:-Log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/dex1_1_service.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/dex1_1_service.log}"
START_TIMEOUT_S="${DEX1_BG_START_TIMEOUT_S:-12}"

mkdir -p "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && ps -p "${old_pid}" >/dev/null 2>&1; then
    echo "[DEX1 SERVICE BG] already running: pid=${old_pid}"
    echo "[DEX1 SERVICE BG] log: ${LOG_FILE}"
    exit 0
  fi
fi

echo "[DEX1 SERVICE BG] starting Dex1_1 service in background"
echo "[DEX1 SERVICE BG] log: ${LOG_FILE}"

nohup ./scripts/run_dex1_1_service.sh >"${LOG_FILE}" 2>&1 &
pid="$!"
echo "${pid}" >"${PID_FILE}"

deadline=$((SECONDS + START_TIMEOUT_S))
started=0
while (( SECONDS < deadline )); do
  if ! ps -p "${pid}" >/dev/null 2>&1; then
    echo "[DEX1 SERVICE BG] failed to stay running. Recent log:" >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
    rm -f "${PID_FILE}"
    exit 2
  fi
  if grep -Fq "Dex1-1 Gripper Server started" "${LOG_FILE}" 2>/dev/null; then
    started=1
    break
  fi
  sleep 0.2
done

if [[ "${started}" != "1" ]]; then
  if ps -p "${pid}" >/dev/null 2>&1; then
    echo "[DEX1 SERVICE BG] service is still running but did not print the ready line within ${START_TIMEOUT_S}s."
    echo "[DEX1 SERVICE BG] continuing; the DDS state check will decide whether it is usable."
    echo "[DEX1 SERVICE BG] running: pid=${pid}"
    tail -n 40 "${LOG_FILE}" || true
    exit 0
  fi
  echo "[DEX1 SERVICE BG] service did not report ready within ${START_TIMEOUT_S}s and is not running. Recent log:" >&2
  tail -n 80 "${LOG_FILE}" >&2 || true
  rm -f "${PID_FILE}"
  exit 2
fi

echo "[DEX1 SERVICE BG] running: pid=${pid}"
tail -n 20 "${LOG_FILE}" || true
