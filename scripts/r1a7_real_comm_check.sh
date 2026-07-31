#!/usr/bin/env bash
set -euo pipefail

IFACE="${1:-${ROBOT_IFACE:-enx9c69d37d0967}}"
PROJECT_DIR="/home/robot/unitree_sim_isaaclab"
CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"

cd "${PROJECT_DIR}"

echo "[R1-A7 CHECK] interface: ${IFACE}"
ip -br addr show "${IFACE}" || true
echo

echo "[R1-A7 CHECK] ping common Unitree development IPs"
for ip in 192.168.123.161 192.168.123.164 192.168.123.18; do
  if ping -c 1 -W 1 "${ip}" >/dev/null 2>&1; then
    echo "REACH ${ip}"
  else
    echo "MISS  ${ip}"
  fi
done
echo

echo "[R1-A7 CHECK] DDS monitor: hg / domain 0 / rt/lowstate"
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/r1a7_right_arm_comm.py \
    --interface "${IFACE}" \
    --domain_id 0 \
    --idl hg \
    --state_topic rt/lowstate \
    --mode monitor \
    --duration 5 || true
echo

echo "[R1-A7 CHECK] DDS monitor: hg / domain 0 / rt/lf/lowstate"
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/r1a7_right_arm_comm.py \
    --interface "${IFACE}" \
    --domain_id 0 \
    --idl hg \
    --state_topic rt/lf/lowstate \
    --mode monitor \
    --duration 5 || true
echo

echo "[R1-A7 CHECK] If both DDS monitors receive no data, verify robot power/develop mode/network cable and IP subnet."
