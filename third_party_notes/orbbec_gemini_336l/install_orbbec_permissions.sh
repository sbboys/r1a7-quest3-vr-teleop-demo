#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULES_SRC="$SCRIPT_DIR/OrbbecSDK_v2.8.7_202606161335_ab8672c_linux_x86_64/shared/99-obsensor-libusb.rules"
RULES_DST="/etc/udev/rules.d/99-obsensor-libusb.rules"

if [[ ! -f "$RULES_SRC" ]]; then
  echo "Missing rules file: $RULES_SRC" >&2
  exit 1
fi

sudo cp "$RULES_SRC" "$RULES_DST"
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Installed Orbbec udev rules to $RULES_DST"
echo "Unplug and replug the Gemini 336L if permissions do not update immediately."
