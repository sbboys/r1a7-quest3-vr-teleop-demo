#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENABLE_DEX1="${ENABLE_DEX1:-1}"
DEX1_SIDE="${DEX1_SIDE:-right}"
CONTROL_CHANNEL="${CONTROL_CHANNEL:-lowcmd}"
CHECK_ARM_LOWSTATE="${CHECK_ARM_LOWSTATE:-1}"
STOP_DEX1_ON_EXIT="${STOP_DEX1_ON_EXIT:-1}"
LOG_DIR="${LOG_DIR:-Log}"
DEX1_PID_FILE="${DEX1_PID_FILE:-${LOG_DIR}/dex1_1_service.pid}"
DEX1_LOG_FILE="${DEX1_LOG_FILE:-${LOG_DIR}/dex1_1_service.log}"

started_dex1=0

check_dex1_motor_error() {
  local side="$1"
  local motor_id
  case "${side}" in
    right) motor_id=0 ;;
    left) motor_id=1 ;;
    *)
      echo "[R1-A7 ONE TERMINAL] unsupported DEX1_SIDE=${side}; use right or left" >&2
      return 2
      ;;
  esac

  # The Dex1 service can publish DDS state even when the motor is in protection.
  # Catch that case before starting camera control, otherwise grip commands are
  # sent but the gripper physically stays still.
  local debug_line
  debug_line="$(grep -E "Motor ${motor_id} debug .*merror=" "${DEX1_LOG_FILE}" 2>/dev/null | tail -n 1 || true)"
  if [[ -z "${debug_line}" ]]; then
    sleep 1.2
    debug_line="$(grep -E "Motor ${motor_id} debug .*merror=" "${DEX1_LOG_FILE}" 2>/dev/null | tail -n 1 || true)"
  fi

  if [[ -z "${debug_line}" ]]; then
    echo "[R1-A7 ONE TERMINAL] warning: no Dex1 motor-${motor_id} debug line yet; continuing with DDS state check only"
    return 0
  fi

  local merror
  merror="$(sed -n 's/.*merror=\([-0-9]\+\).*/\1/p' <<<"${debug_line}" | tail -n 1)"
  if [[ -n "${merror}" && "${merror}" != "0" ]]; then
    cat >&2 <<EOF
[R1-A7 ONE TERMINAL] Dex1 ${side} motor is not ready: merror=${merror}
[R1-A7 ONE TERMINAL] Latest motor line:
${debug_line}
[R1-A7 ONE TERMINAL] The camera task would send dex1_cmd, but the gripper motor will not move in this state.
[R1-A7 ONE TERMINAL] Stop the service, fully power-cycle the gripper/robot gripper power for 10-15s, then run:
  ./scripts/run_dex1_1_service.sh
Check that Motor ${motor_id} debug shows merror=0 before starting camera control.
EOF
    return 3
  fi

  echo "[R1-A7 ONE TERMINAL] Dex1 ${side} motor error check OK: merror=${merror}"
}

stop_dex1_if_needed() {
  if [[ "${started_dex1}" == "1" && "${STOP_DEX1_ON_EXIT}" == "1" ]]; then
    echo "[R1-A7 ONE TERMINAL] stopping background Dex1_1 service"
    PID_FILE="${DEX1_PID_FILE}" ./scripts/stop_dex1_1_service_bg.sh || true
    started_dex1=0
  fi
}

cleanup() {
  local status=$?
  stop_dex1_if_needed
  exit "${status}"
}
trap cleanup EXIT

if [[ "${ENABLE_DEX1}" == "1" ]]; then
  mkdir -p "${LOG_DIR}"
  if [[ -f "${DEX1_PID_FILE}" ]]; then
    old_pid="$(cat "${DEX1_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && ps -p "${old_pid}" >/dev/null 2>&1; then
      echo "[R1-A7 ONE TERMINAL] reusing background Dex1_1 service pid=${old_pid}"
      echo "[R1-A7 ONE TERMINAL] log: ${DEX1_LOG_FILE}"
    else
      rm -f "${DEX1_PID_FILE}"
    fi
  fi

  if [[ ! -f "${DEX1_PID_FILE}" ]]; then
    echo "[R1-A7 ONE TERMINAL] starting Dex1_1 service in background"
    PID_FILE="${DEX1_PID_FILE}" LOG_FILE="${DEX1_LOG_FILE}" ./scripts/start_dex1_1_service_bg.sh
    started_dex1=1
  fi

  echo "[R1-A7 ONE TERMINAL] checking ${DEX1_SIDE} Dex1_1 DDS state"
  if ! /home/robot/miniconda3/bin/conda run --no-capture-output -n isaaclab \
    python -u tools/dex1_1_gripper_dds.py \
      --interface "${ROBOT_IFACE:-enx9c69d37d0967}" \
      --side "${DEX1_SIDE}" \
      --mode monitor \
      --duration 2; then
    echo "[R1-A7 ONE TERMINAL] Dex1_1 DDS check failed. Recent service log:" >&2
    tail -n 80 "${DEX1_LOG_FILE}" >&2 || true
    exit 2
  fi

  if ! check_dex1_motor_error "${DEX1_SIDE}"; then
    exit 2
  fi
else
  echo "[R1-A7 ONE TERMINAL] ENABLE_DEX1=0; running arm control without gripper"
fi

if [[ "${CHECK_ARM_LOWSTATE}" == "1" ]]; then
  case "${CONTROL_CHANNEL}" in
    lowcmd|arm_sdk)
      ARM_STATE_TOPIC="rt/lowstate"
      ;;
    lf_lowcmd|lf_arm_sdk)
      ARM_STATE_TOPIC="rt/lf/lowstate"
      ;;
    *)
      echo "[R1-A7 ONE TERMINAL] unsupported CONTROL_CHANNEL=${CONTROL_CHANNEL}; use lowcmd, lf_lowcmd, arm_sdk, or lf_arm_sdk" >&2
      exit 2
      ;;
  esac

  echo "[R1-A7 ONE TERMINAL] checking R1-A7 arm DDS state: ${ARM_STATE_TOPIC}"
  if ! /home/robot/miniconda3/bin/conda run --no-capture-output -n isaaclab \
    python -u tools/r1a7_right_arm_comm.py \
      --interface "${ROBOT_IFACE:-enx9c69d37d0967}" \
      --domain_id 0 \
      --idl hg \
      --state_topic "${ARM_STATE_TOPIC}" \
      --mode monitor \
      --duration 3; then
    cat >&2 <<EOF
[R1-A7 ONE TERMINAL] arm lowstate check failed: no ${ARM_STATE_TOPIC} was received.
[R1-A7 ONE TERMINAL] This is why the camera task starts but the robot arm does not react.
[R1-A7 ONE TERMINAL] Try CONTROL_CHANNEL=lf_lowcmd, or check the robot body network/develop mode/DDS publisher.
EOF
    exit 2
  fi
fi

echo "[R1-A7 ONE TERMINAL] starting camera control"
echo "[R1-A7 ONE TERMINAL] when prompted, type ENABLE after the robot workspace is clear"
ENABLE_DEX1="${ENABLE_DEX1}" DEX1_SIDE="${DEX1_SIDE}" CONTROL_CHANNEL="${CONTROL_CHANNEL}" ./scripts/run_r1a7_camera_real_control.sh "$@"
