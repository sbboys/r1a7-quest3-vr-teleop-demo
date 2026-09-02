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

bash "${ROOT}/tools/build_dataset_setup_001_record_camera_episode_sync.sh" >/dev/null

echo "Recording copy: TeleVuer -> R1A7_ArmIK -> R1 rt/lowcmd + Unitree EpisodeWriter"
echo "One TeleVuer server and one rt/lowcmd publisher. MuJoCo and rt/arm_sdk are not used."
echo "Default joint speed: 0.8 rad/s"
echo "Press right A or left X to start teleoperation and synchronized recording together."
echo "Close the Quest page to stop and save; 3600 seconds is the safety maximum."
echo "Quest URL: https://${HOST_IP}:8012/?ws=wss://${HOST_IP}:8012"

exec env -u PYTHONPATH PYTHONNOUSERSITE=1 AIOHTTP_NOSENDFILE=1 \
  XR_TELEOP_ROOT="/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/xr_teleoperate" \
  "${CONDA}" run --no-capture-output -n tv \
  python -u "${ROOT}/tools/r1a7_unitree_official_vuer_real_lowcmd_record.py" \
    --interface enp6s0 \
    --domain-id 0 \
    --host-ip "${HOST_IP}" \
    --ik-frequency 30 \
    --publish-frequency 250 \
    --max-joint-speed 0.8 \
    --record-duration "${R1A7_RECORD_DURATION:-3600}" \
    --camera-warmup 3 \
    --record-frequency 15 \
    --record-task-dir "${R1A7_RECORD_TASK_DIR:-datasets/dataset_setup_001/unitree_xr_data/r1a7_charge_connector_coarse_approach}" \
    "$@"
