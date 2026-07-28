#!/usr/bin/env bash
set -euo pipefail

ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"
ROBOT_IP="${ROBOT_IP:-192.168.123.161}"
R1_LOCO_STATUS="${R1_LOCO_STATUS:-build/tools/r1a7_loco_status}"
R1A7_PREFLIGHT_STRICT="${R1A7_PREFLIGHT_STRICT:-1}"

echo "[R1-A7 PREFLIGHT] interface: ${ROBOT_IFACE}"

if ! ip -br addr show "${ROBOT_IFACE}" >/dev/null 2>&1; then
  echo "[R1-A7 PREFLIGHT] FAIL: network interface ${ROBOT_IFACE} not found." >&2
  exit 2
fi

ip -br addr show "${ROBOT_IFACE}"

if ! ping -c 1 -W 1 "${ROBOT_IP}" >/dev/null 2>&1; then
  echo "[R1-A7 PREFLIGHT] FAIL: robot ${ROBOT_IP} is not reachable from ${ROBOT_IFACE}." >&2
  exit 2
fi

if [[ ! -x "${R1_LOCO_STATUS}" ]]; then
  if [[ -x ./scripts/build_r1a7_loco_status.sh ]]; then
    ./scripts/build_r1a7_loco_status.sh
  fi
fi

if [[ ! -x "${R1_LOCO_STATUS}" ]]; then
  echo "[R1-A7 PREFLIGHT] WARN: ${R1_LOCO_STATUS} not found; skipping FSM RPC check." >&2
  exit 0
fi

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

if ! timeout 15s "${R1_LOCO_STATUS}" \
    --interface "${ROBOT_IFACE}" \
    --get \
    --timeout 4 | tee "${tmp}"; then
  echo "[R1-A7 PREFLIGHT] FAIL: R1 loco RPC query failed or timed out." >&2
  echo "[R1-A7 PREFLIGHT] rt/lowstate may still work, but body high-level control is not accepting RPC." >&2
  echo "[R1-A7 PREFLIGHT] Check robot app/control mode, body services, emergency stop, and reboot state." >&2
  exit 2
fi

ret_id="$(awk -F'ret=' '/get_fsm_id/ {split($2,a," "); print a[1]; exit}' "${tmp}")"
ret_mode="$(awk -F'ret=' '/get_fsm_mode/ {split($2,a," "); print a[1]; exit}' "${tmp}")"
fsm_id="$(awk -F'value=' '/get_fsm_id/ && /value=/ {split($2,a," "); print a[1]; exit}' "${tmp}")"
fsm_mode="$(awk -F'value=' '/get_fsm_mode/ && /value=/ {split($2,a," "); print a[1]; exit}' "${tmp}")"

if [[ "${ret_id:-}" != "0" ]] || [[ "${ret_mode:-}" != "0" ]]; then
  echo "[R1-A7 PREFLIGHT] FAIL: R1 loco RPC returned ret_id=${ret_id:-unknown} ret_mode=${ret_mode:-unknown}." >&2
  echo "[R1-A7 PREFLIGHT] The robot will usually ignore rt/arm_sdk while the body service is unavailable." >&2
  echo "[R1-A7 PREFLIGHT] Fix body-side control mode/services first, then retry camera control." >&2
  exit 2
fi

echo "[R1-A7 PREFLIGHT] fsm_id=${fsm_id:-unknown} fsm_mode=${fsm_mode:-unknown}"

if [[ ! "${fsm_id:-}" =~ ^[0-9]+$ ]] || (( fsm_id > 10000 )); then
  echo "[R1-A7 PREFLIGHT] FAIL: invalid or unavailable fsm_id from R1 loco RPC: ${fsm_id:-unknown}" >&2
  echo "[R1-A7 PREFLIGHT] Recheck body service state, robot app mode, and retry after the robot finishes booting." >&2
  exit 2
fi

case "${fsm_id:-unknown}" in
  0)
    echo "[R1-A7 PREFLIGHT] FAIL: fsm_id=0 means ZeroTorque/off-control state." >&2
    echo "[R1-A7 PREFLIGHT] Arm SDK commands on rt/arm_sdk will be ignored." >&2
    echo "[R1-A7 PREFLIGHT] With the workspace clear and emergency stop ready, run:" >&2
    echo "  cd /home/robot/unitree_sim_isaaclab" >&2
    echo "  ROBOT_IFACE=${ROBOT_IFACE} ./scripts/r1a7_start_body_for_arm_control.sh" >&2
    echo "[R1-A7 PREFLIGHT] To bypass only this check: SKIP_R1A7_PREFLIGHT=1 ..." >&2
    if [[ "${R1A7_PREFLIGHT_STRICT}" == "1" ]]; then
      exit 3
    fi
    ;;
  unknown)
    echo "[R1-A7 PREFLIGHT] WARN: could not parse fsm_id from R1 loco output." >&2
    ;;
  *)
    echo "[R1-A7 PREFLIGHT] OK: body FSM is not ZeroTorque."
    ;;
esac
