#!/usr/bin/env bash
set -euo pipefail

CONSOLE_ROOT="/home/robot/unitree_sim_isaaclab_console"
SERVER="$CONSOLE_ROOT/r1a7_threading_project/console3d_hierarchical/server.py"
PYTHON_BIN="/home/robot/miniconda3/envs/tv/bin/python"
LOG_DIR="/home/robot/.cache/r1a7-console"
LOG_FILE="$LOG_DIR/console-8098.log"
BASE_URL="http://127.0.0.1:8098"
UI_BUILD="20261013-visual-servo-drive"
UI_URL="$BASE_URL/?build=$UI_BUILD"
UNIT="r1a7-console-8098.service"

mkdir -p "$LOG_DIR"

if ! curl -fsS --max-time 1 "$BASE_URL/api/cameras" >/dev/null 2>&1; then
  systemctl --user stop "$UNIT" >/dev/null 2>&1 || true
  systemctl --user reset-failed "$UNIT" >/dev/null 2>&1 || true
  : >"$LOG_FILE"
  systemd-run --user \
    --unit="$UNIT" \
    --collect \
    --property=Restart=on-failure \
    --property=RestartSec=1 \
    --property="StandardOutput=append:$LOG_FILE" \
    --property="StandardError=append:$LOG_FILE" \
    --setenv="DISPLAY=${DISPLAY:-:0}" \
    "$PYTHON_BIN" -u "$SERVER" --host 127.0.0.1 --port 8098 \
    >/dev/null

  for _attempt in $(seq 1 30); do
    if curl -fsS --max-time 1 "$BASE_URL/api/cameras" >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
fi

if curl -fsS --max-time 1 "$BASE_URL/api/cameras" >/dev/null 2>&1; then
  xdg-open "$UI_URL" >/dev/null 2>&1 || true
  exit 0
fi

notify-send "R1-A7 控制台启动失败" "请查看 $LOG_FILE" >/dev/null 2>&1 || true
exit 1
