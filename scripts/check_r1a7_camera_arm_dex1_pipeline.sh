#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"
ROBOT_IP="${ROBOT_IP:-192.168.123.161}"
DEX1_SIDE="${DEX1_SIDE:-right}"

echo "[PIPELINE CHECK] PC DDS interface: ${ROBOT_IFACE}"
ip -br addr show "${ROBOT_IFACE}" || true
echo

echo "[PIPELINE CHECK] ping robot: ${ROBOT_IP}"
if ping -c 1 -W 1 "${ROBOT_IP}" >/dev/null 2>&1; then
  echo "ROBOT_REACHABLE ${ROBOT_IP}"
else
  echo "ROBOT_MISSING ${ROBOT_IP}"
fi
echo

echo "[PIPELINE CHECK] R1-A right arm DDS: rt/lowstate"
if "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/r1a7_right_arm_comm.py \
    --interface "${ROBOT_IFACE}" \
    --domain_id 0 \
    --idl hg \
    --state_topic rt/lowstate \
    --mode monitor \
    --duration 4; then
  ARM_OK=1
else
  ARM_OK=0
fi
echo

echo "[PIPELINE CHECK] Dex1_1 DDS: rt/dex1/${DEX1_SIDE}/state"
if "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/dex1_1_gripper_dds.py \
    --interface "${ROBOT_IFACE}" \
    --side "${DEX1_SIDE}" \
    --mode monitor \
    --duration 4; then
  DEX1_OK=1
else
  DEX1_OK=0
fi
echo

echo "[PIPELINE CHECK] summary"
if [[ "${ARM_OK}" -eq 1 ]]; then
  echo "ARM_DDS_OK"
else
  echo "ARM_DDS_FAIL: check robot power, develop mode, DDS domain/interface, and rt/lowstate."
fi

if [[ "${DEX1_OK}" -eq 1 ]]; then
  echo "DEX1_DDS_OK"
else
  echo "DEX1_DDS_FAIL: start dex1_1_gripper_server on the host that can see the gripper serial board."
fi

if [[ "${ARM_OK}" -eq 1 && "${DEX1_OK}" -eq 1 ]]; then
  cat <<'EOF'
PIPELINE_READY
Next safe checks:
  ./scripts/run_r1a7_camera_real_preview.sh --show
  ./scripts/test_dex1_1_right_gripper.sh

Real control:
  ./scripts/run_r1a7_camera_real_control.sh --show
EOF
else
  exit 2
fi
