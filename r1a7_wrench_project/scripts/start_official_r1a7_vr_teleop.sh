#!/usr/bin/env bash
set -euo pipefail

OFFICIAL_XR_DIR="${OFFICIAL_XR_DIR:-/home/robot/xr_teleoperate_r1_official}"
CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-tv}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-enp6s0}"
HOST_IP="${HOST_IP:-192.168.1.103}"
IMG_SERVER_IP="${IMG_SERVER_IP:-192.168.123.164}"
INPUT_MODE="${INPUT_MODE:-controller}"
DISPLAY_MODE="${DISPLAY_MODE:-pass-through}"
EE="${EE:-dex1}"
FREQUENCY="${FREQUENCY:-30}"

if [[ ! -f "${OFFICIAL_XR_DIR}/teleop/teleop_hand_and_arm.py" ]]; then
  echo "Official xr_teleoperate checkout not found: ${OFFICIAL_XR_DIR}" >&2
  exit 1
fi

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable not found: ${CONDA_BIN}" >&2
  exit 1
fi

if ! ip link show "${NETWORK_INTERFACE}" >/dev/null 2>&1; then
  echo "DDS network interface not found: ${NETWORK_INTERFACE}" >&2
  ip -br link >&2
  exit 1
fi

if ! ip -br addr | grep -q "${HOST_IP}"; then
  echo "Host IP ${HOST_IP} is not present on this machine." >&2
  ip -br addr >&2
  exit 1
fi

if ss -lntp | grep -E ':(8012|60000|60001)\b'; then
  echo "A VR/image service port is already in use. Stop the old process before starting official teleop." >&2
  exit 1
fi

if ! ping -c 1 -W 1 "${IMG_SERVER_IP}" >/dev/null 2>&1; then
  echo "Warning: image server ${IMG_SERVER_IP} is not reachable." >&2
  echo "Official teleop may fall back to local camera config, but first-person images/recording need teleimager." >&2
fi

cat <<EOF
Official R1-A7 VR teleop launch
  official dir:       ${OFFICIAL_XR_DIR}
  network interface:  ${NETWORK_INTERFACE}
  host/Quest URL:     https://${HOST_IP}:8012/?ws=wss://${HOST_IP}:8012
  image server IP:    ${IMG_SERVER_IP}
  input mode:         ${INPUT_MODE}
  display mode:       ${DISPLAY_MODE}
  end effector:       ${EE}

This runs Unitree official xr_teleoperate only.
After startup, open the Quest browser URL above, enter VR, then press r in this terminal.
Press q in this terminal to exit.
EOF

export PYTHONNOUSERSITE=1
export PYTHONPATH="${OFFICIAL_XR_DIR}:${OFFICIAL_XR_DIR}/teleop:${OFFICIAL_XR_DIR}/teleop/televuer/src:${OFFICIAL_XR_DIR}/teleop/teleimager/src:${OFFICIAL_XR_DIR}/teleop/robot_control/dex-retargeting/src${PYTHONPATH:+:${PYTHONPATH}}"
export XR_TELEOP_CERT="${XR_TELEOP_CERT:-/home/robot/.config/xr_teleoperate/cert.pem}"
export XR_TELEOP_KEY="${XR_TELEOP_KEY:-/home/robot/.config/xr_teleoperate/key.pem}"

cmd=(
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}"
  python -u teleop_hand_and_arm.py
  --input-mode="${INPUT_MODE}"
  --display-mode="${DISPLAY_MODE}"
  --arm=R1_A7
  --img-server-ip="${IMG_SERVER_IP}"
  --network-interface="${NETWORK_INTERFACE}"
  --frequency="${FREQUENCY}"
)

if [[ "${EE}" != "none" ]]; then
  cmd+=(--ee="${EE}")
fi

cd "${OFFICIAL_XR_DIR}/teleop"
exec "${cmd[@]}" "$@"
