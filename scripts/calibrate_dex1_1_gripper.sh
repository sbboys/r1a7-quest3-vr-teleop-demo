#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEX1_DIR="${DEX1_DIR:-/home/robot/IsaacLab/unitree_robots/dex1_1_service}"
ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"

./scripts/stop_dex1_1_service_bg.sh || true

cat <<'EOF'
[DEX1 CALIBRATION] This will run the official Dex1_1 motor calibration.
[DEX1 CALIBRATION] The program will ask for each detected motor:
  1. Manually close the corresponding gripper tightly.
  2. Type s and press Enter to calibrate that motor.
  3. Type another key and press Enter to skip that motor.

Keep fingers and objects clear of the gripper mechanism.
EOF

cd "${DEX1_DIR}"
exec ./bin/dex1_1_gripper_server --network "${ROBOT_IFACE}" --calibration
