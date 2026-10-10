#!/usr/bin/env bash
set -u

PROJECT="$HOME/unitree_sim_isaaclab_threading/r1a7_threading_project"
PY="/home/robot/miniconda3/envs/tv_sam2/bin/python"

EXT="$PROJECT/vision_dual_camera/task6_threading/task6_external_ibvs_v4_aruco_edge.py"
WRIST="$PROJECT/vision_dual_camera/task6_threading/task6_wrist_ibvs_controller_v2_tip.py"
FUSION="$PROJECT/vision_dual_camera/task6_threading/task6_dual_ibvs_fusion.py"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="/tmp/r1a7_visual_test_${STAMP}"

mkdir -p "$LOGDIR"

echo "============================================================"
echo "R1-A7 dual-camera visual stack"
echo "LOGDIR: $LOGDIR"
echo "============================================================"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# JSON snapshot logger
# Records the RAW output of External / Wrist / Fusion at 20 Hz.
# No visual-control logic is modified.
# ------------------------------------------------------------

LOGGER="$LOGDIR/visual_json_logger.py"

cat > "$LOGGER" <<'PY'
import json
import time
from pathlib import Path

FILES = {
    "external": Path("/tmp/r1a7_ibvs_velocity.json"),
    "wrist": Path("/tmp/r1a7_wrist_ibvs_velocity.json"),
    "fusion": Path("/tmp/r1a7_dual_ibvs_velocity.json"),
}

OUT = Path("__OUTPUT_FILE__")

def read_json(path):
    try:
        text = path.read_text()
        data = json.loads(text)
        stat = path.stat()

        return {
            "exists": True,
            "mtime_ns": stat.st_mtime_ns,
            "data": data,
        }

    except Exception as exc:
        return {
            "exists": path.exists(),
            "error": repr(exc),
            "data": None,
        }

print("[LOGGER] output:", OUT, flush=True)

with OUT.open("w", buffering=1) as f:
    try:
        while True:
            row = {
                "wall_time": time.time(),
                "time_monotonic": time.monotonic(),
            }

            for name, path in FILES.items():
                row[name] = read_json(path)

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":")
                )
                + "\n"
            )

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

print("[LOGGER] stopped", flush=True)
PY

sed -i \
"s|__OUTPUT_FILE__|$LOGDIR/visual_data.jsonl|" \
"$LOGGER"

# ------------------------------------------------------------
# Start processes
# ------------------------------------------------------------

echo "[1/4] Starting External camera..."
"$PY" -u "$EXT" \
    > "$LOGDIR/external.log" 2>&1 &
PID_EXT=$!

sleep 1

echo "[2/4] Starting Wrist camera..."
"$PY" -u "$WRIST" \
    > "$LOGDIR/wrist.log" 2>&1 &
PID_WRIST=$!

sleep 1

echo "[3/4] Starting Fusion..."
"$PY" -u "$FUSION" \
    > "$LOGDIR/fusion.log" 2>&1 &
PID_FUSION=$!

sleep 1

echo "[4/4] Starting JSON data logger..."
"$PY" -u "$LOGGER" \
    > "$LOGDIR/logger.log" 2>&1 &
PID_LOGGER=$!

cleanup() {
    echo
    echo "============================================================"
    echo "Stopping visual stack..."
    echo "============================================================"

    kill "$PID_LOGGER" 2>/dev/null || true
    kill "$PID_FUSION" 2>/dev/null || true
    kill "$PID_WRIST" 2>/dev/null || true
    kill "$PID_EXT" 2>/dev/null || true

    wait "$PID_LOGGER" 2>/dev/null || true
    wait "$PID_FUSION" 2>/dev/null || true
    wait "$PID_WRIST" 2>/dev/null || true
    wait "$PID_EXT" 2>/dev/null || true

    echo
    echo "Saved:"
    echo "  $LOGDIR/external.log"
    echo "  $LOGDIR/wrist.log"
    echo "  $LOGDIR/fusion.log"
    echo "  $LOGDIR/visual_data.jsonl"
    echo
    echo "$LOGDIR" > /tmp/r1a7_last_visual_logdir.txt
    echo "Latest log directory:"
    echo "  $LOGDIR"
}

trap cleanup EXIT INT TERM

echo
echo "Processes:"
echo "  External PID = $PID_EXT"
echo "  Wrist    PID = $PID_WRIST"
echo "  Fusion   PID = $PID_FUSION"
echo "  Logger   PID = $PID_LOGGER"
echo

echo "Logs:"
echo "  External : $LOGDIR/external.log"
echo "  Wrist    : $LOGDIR/wrist.log"
echo "  Fusion   : $LOGDIR/fusion.log"
echo "  JSONL    : $LOGDIR/visual_data.jsonl"
echo

echo "$LOGDIR" > /tmp/r1a7_last_visual_logdir.txt

echo "Visual stack is running."
echo "Press Ctrl+C here to stop all four processes."
echo

while true; do
    sleep 1

    if ! kill -0 "$PID_EXT" 2>/dev/null; then
        echo "[ERROR] External process stopped."
        break
    fi

    if ! kill -0 "$PID_WRIST" 2>/dev/null; then
        echo "[ERROR] Wrist process stopped."
        break
    fi

    if ! kill -0 "$PID_FUSION" 2>/dev/null; then
        echo "[ERROR] Fusion process stopped."
        break
    fi

    if ! kill -0 "$PID_LOGGER" 2>/dev/null; then
        echo "[ERROR] Logger process stopped."
        break
    fi
done
