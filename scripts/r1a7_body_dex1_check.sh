#!/usr/bin/env bash
set -u

echo "[BODY DEX1 CHECK] host: $(hostname)"
echo "[BODY DEX1 CHECK] time: $(date '+%F %T')"

echo
echo "[1/6] network interfaces"
ip -br addr

echo
echo "[2/6] gripper serial devices"
shopt -s nullglob
SERIAL_PORTS=(/dev/ttyUSB* /dev/ttyCH343USB* /dev/ttyACM*)
shopt -u nullglob
if [[ ${#SERIAL_PORTS[@]} -eq 0 ]]; then
  echo "NO_SERIAL_DEVICE"
  echo "Expected one of: /dev/ttyUSB*, /dev/ttyCH343USB*, /dev/ttyACM*"
else
  ls -l "${SERIAL_PORTS[@]}"
fi

echo
echo "[3/6] usb devices"
if command -v lsusb >/dev/null 2>&1; then
  lsusb
else
  echo "lsusb not installed"
fi

echo
echo "[4/6] serial drivers"
lsmod | grep -E 'ch34|cp210|ftdi|cdc_acm|usbserial' || echo "NO_SERIAL_DRIVER_LOADED"

echo
echo "[5/6] dex1/gripper processes"
ps -ef | grep -E 'dex1_1_gripper_server|test_dex1|gripper' | grep -v grep || echo "NO_DEX1_PROCESS"

echo
echo "[6/6] dex1 service binaries"
for dir in \
  /home/robot/IsaacLab/unitree_robots/dex1_1_service \
  /home/unitree/dex1_1_service \
  /home/robot/dex1_1_service \
  "$HOME/dex1_1_service"; do
  if [[ -d "$dir" ]]; then
    echo "FOUND_DIR $dir"
    ls -l "$dir/bin/dex1_1_gripper_server" "$dir/build/dex1_1_gripper_server" 2>/dev/null || true
  fi
done

echo
if [[ ${#SERIAL_PORTS[@]} -eq 0 ]]; then
  echo "[RESULT] FAIL: this host cannot see the Dex1_1 serial board."
  echo "Check gripper power, USB/serial board cable, and whether you are on the correct body computer."
else
  echo "[RESULT] OK: serial device exists. Start dex1_1_gripper_server on the 192.168.123.* interface."
fi
