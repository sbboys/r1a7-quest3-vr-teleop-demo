#!/usr/bin/env bash
set -euo pipefail

ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"
R1_LOCO_STATUS="${R1_LOCO_STATUS:-build/tools/r1a7_loco_status}"
R1A7_BODY_ACTION="${R1A7_BODY_ACTION:-start}"

case "${R1A7_BODY_ACTION}" in
  start|stand_up)
    ;;
  *)
    echo "[R1-A7 BODY START] unsupported R1A7_BODY_ACTION=${R1A7_BODY_ACTION}; use start or stand_up" >&2
    exit 2
    ;;
esac

if [[ ! -x "${R1_LOCO_STATUS}" ]]; then
  if [[ -x ./scripts/build_r1a7_loco_status.sh ]]; then
    ./scripts/build_r1a7_loco_status.sh
  fi
fi

if [[ ! -x "${R1_LOCO_STATUS}" ]]; then
  echo "[R1-A7 BODY START] ${R1_LOCO_STATUS} not found or not executable." >&2
  exit 2
fi

echo "WARNING: This changes the real R1 body FSM state."
echo "Clear the robot workspace and keep the emergency stop ready."
echo "Action: ${R1A7_BODY_ACTION}"
read -r -p "Type ENABLE to continue: " answer
if [[ "${answer}" != "ENABLE" ]]; then
  echo "[R1-A7 BODY START] aborted"
  exit 2
fi

echo "[R1-A7 BODY START] state before:"
"${R1_LOCO_STATUS}" --interface "${ROBOT_IFACE}" --get --timeout 4 || \
  echo "[R1-A7 BODY START] WARN: state query before action failed; trying requested action anyway."

case "${R1A7_BODY_ACTION}" in
  start)
    echo "[R1-A7 BODY START] calling Start()"
    "${R1_LOCO_STATUS}" --interface "${ROBOT_IFACE}" --start --timeout 4
    ;;
  stand_up)
    echo "[R1-A7 BODY START] calling StandUp()"
    "${R1_LOCO_STATUS}" --interface "${ROBOT_IFACE}" --stand_up --timeout 4
    ;;
esac

sleep 1.0
echo "[R1-A7 BODY START] state after:"
"${R1_LOCO_STATUS}" --interface "${ROBOT_IFACE}" --get --timeout 4
