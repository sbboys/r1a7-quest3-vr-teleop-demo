#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"
SIDE="${SIDE:-right}"
DURATION="${DURATION:-5}"

echo "[DEX1 BODY CHECK] PC DDS interface: ${ROBOT_IFACE}"
echo "[DEX1 BODY CHECK] monitoring rt/dex1/${SIDE}/state for ${DURATION}s"
echo "[DEX1 BODY CHECK] this does not move the gripper"

if ! "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/dex1_1_gripper_dds.py \
    --interface "${ROBOT_IFACE}" \
    --side "${SIDE}" \
    --mode monitor \
    --duration "${DURATION}" \
    "$@"; then
  cat <<EOF
[DEX1 BODY CHECK] no DDS state was received from the robot body.
[DEX1 BODY CHECK] On the robot body, check:
  1. ip -br addr
  2. Dex1_1 service is running on the 192.168.123.* interface:
     sudo ./bin/dex1_1_gripper_server --network <robot_192.168.123_interface>
  3. The service detected Motor ID 0 as Side: Right.
EOF
  exit 2
fi
