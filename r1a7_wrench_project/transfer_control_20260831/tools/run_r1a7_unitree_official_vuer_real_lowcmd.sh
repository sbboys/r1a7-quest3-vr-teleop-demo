#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="/home/robot/miniconda3/bin/conda"
HOST_IP="${R1A7_HOST_IP:-}"

if [[ -z "${HOST_IP}" ]]; then
  HOST_IP="$(ip -4 -brief address show up | awk '$1 ~ /^(wl|enx)/ && $3 ~ /^192[.]168[.]1[.]/ {split($3, a, "/"); print a[1]; exit}')"
fi
if [[ -z "${HOST_IP}" ]]; then
  HOST_IP="$(ip -4 -brief address show up | awk '$1 ~ /^wl/ {split($3, a, "/"); print a[1]; exit}')"
fi
if [[ -z "${HOST_IP}" ]]; then
  echo "Could not detect the Quest-facing Wi-Fi IP. Set R1A7_HOST_IP." >&2
  exit 1
fi

echo "Official default path: TeleVuer controller -> R1A7_ArmIK -> R1 rt/lowcmd"
echo "Real robot only. MuJoCo and rt/arm_sdk are not used."
echo "Quest URL: https://${HOST_IP}:8012/?ws=wss://${HOST_IP}:8012"

exec env PYTHONNOUSERSITE=1 AIOHTTP_NOSENDFILE=1 \
  XR_TELEOP_ROOT="/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/xr_teleoperate" \
  "${CONDA}" run --no-capture-output -n tv \
  python -u "${ROOT}/tools/r1a7_unitree_official_vuer_real_lowcmd.py" \
    --interface enp6s0 \
    --domain-id 0 \
    --host-ip "${HOST_IP}" \
    --ik-frequency 30 \
    --publish-frequency 250 \
    --max-joint-speed 2.0 \
    "$@"
