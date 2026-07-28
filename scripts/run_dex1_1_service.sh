#!/usr/bin/env bash
set -euo pipefail

DEX1_DIR="${DEX1_DIR:-/home/robot/IsaacLab/unitree_robots/dex1_1_service}"
ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"

cd "${DEX1_DIR}"

if [[ ! -e /usr/local/include/unitree/idl/go2/MotorCmds_.hpp ]]; then
  cat <<'EOF'
[DEX1 SERVICE] missing C++ unitree_sdk2 headers:
  /usr/local/include/unitree/idl/go2/MotorCmds_.hpp

Install the official C++ unitree_sdk2 first:
  cd ~
  git clone https://github.com/unitreerobotics/unitree_sdk2.git
  cd unitree_sdk2
  mkdir -p build
  cmake -S . -B build
  cmake --build build -j"$(nproc)"
  sudo cmake --install build

Then run this script again.
EOF
  exit 2
fi

if [[ ! -e /usr/include/libserialport.h && ! -e /usr/local/include/libserialport.h ]]; then
  cat <<'EOF'
[DEX1 SERVICE] missing libserialport header:
  libserialport.h

Install it first:
  sudo apt update
  sudo apt install libserialport-dev

The offline libserialport .deb files shipped in dex1_1_service/lib are arm64 packages.
Do not install them on this x86_64 PC.

Then run this script again.
EOF
  exit 2
fi

shopt -s nullglob
SERIAL_PORTS=(/dev/ttyUSB* /dev/ttyCH343USB* /dev/ttyACM*)
shopt -u nullglob
if [[ ${#SERIAL_PORTS[@]} -eq 0 ]]; then
  cat <<'EOF'
[DEX1 SERVICE] no local gripper serial device was found.

Expected one of:
  /dev/ttyUSB*
  /dev/ttyCH343USB*
  /dev/ttyACM*

If Dex1_1 is connected to the robot body, run this service on the robot body
computer that can see the gripper serial board, not on the camera PC.

After the body-side service is running, this PC should receive:
  rt/dex1/right/state

Check from this PC with:
  ./scripts/check_dex1_1_body_gripper.sh
EOF
  exit 3
fi

if [[ ! -x bin/dex1_1_gripper_server ]]; then
  echo "[DEX1 SERVICE] bin/dex1_1_gripper_server not found; building ..."
  mkdir -p build
  cmake -S . -B build
  cmake --build build -j"$(nproc)"
fi

echo "[DEX1 SERVICE] starting Dex1_1 server on ${ROBOT_IFACE}"
if [[ -r "${SERIAL_PORTS[0]}" && -w "${SERIAL_PORTS[0]}" ]]; then
  echo "[DEX1 SERVICE] serial ports are accessible by the current user; starting without sudo"
  exec ./bin/dex1_1_gripper_server --network "${ROBOT_IFACE}"
fi

cat <<'EOF'
[DEX1 SERVICE] serial ports are not accessible by the current user.
[DEX1 SERVICE] Enter sudo password if prompted, or add this user to dialout and re-login:
  sudo usermod -aG dialout "$USER"
EOF
exec sudo ./bin/dex1_1_gripper_server --network "${ROBOT_IFACE}"
