#!/usr/bin/env bash

set -u

PY="${R1A7_WIRE_PYTHON:-/home/robot/miniconda3/envs/tv_sam2/bin/python}"
export PYTHONNOUSERSITE=1

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../../.." && pwd)"
LOGDIR="${R1A7_WIRE_LOG_DIR:-$ROOT/r1a7_logs/wire_grasp_candidate}"

mkdir -p "$LOGDIR"

ENABLE_MOTION=0

if [ "${1:-}" = "--enable-motion" ]; then
    ENABLE_MOTION=1
fi

PIDS=""

cleanup()
{
    echo
    echo "[STOP] dual-camera wire grasp stack"

    for pid in $PIDS; do
        kill "$pid" 2>/dev/null || true
    done

    wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM


start_bg()
{
    name="$1"
    shift

    echo "[START] $name"

    "$@" >"$LOGDIR/$name.log" 2>&1 &

    pid=$!
    PIDS="$PIDS $pid"

    echo "        pid=$pid"
    echo "        log=$LOGDIR/$name.log"
}


echo
echo "======================================================"
echo "R1-A7 DUAL CAMERA WIRE GRASP STACK"
echo "======================================================"

if [ "$ENABLE_MOTION" -eq 1 ]; then
    echo "MODE: CONTROL OUTPUT ENABLED"
else
    echo "MODE: SHADOW ONLY"
fi

echo "======================================================"
echo


cd "$HERE"

start_bg \
    wrist_perception \
    "$PY" \
    "$HERE/right_wrist_wire_tweezer_v69_nomask.py"

sleep 1


start_bg \
    wrist_guidance \
    "$PY" \
    "$HERE/wrist_wire_grasp_guidance_bridge.py"

sleep 1


start_bg \
    external_guidance \
    bash -c \
    "cd '$ROOT' && '$PY' '$HERE/external_wire_grasp_guidance_v2_candidate.py'"

sleep 1


start_bg \
    fusion \
    "$PY" \
    "$HERE/wire_grasp_fusion_shadow_v1.py"

sleep 1


if [ "$ENABLE_MOTION" -eq 1 ]; then

    start_bg \
        control_gate \
        "$PY" \
        "$HERE/wire_grasp_control_gate_v1.py" \
        --enable-output

else

    start_bg \
        control_gate \
        "$PY" \
        "$HERE/wire_grasp_control_gate_v1.py"

fi


echo
echo "======================================================"
echo "STACK RUNNING"
echo
echo "Final command:"
echo "  /tmp/r1a7_dual_ibvs_velocity.json"
echo
echo "Logs:"
echo "  $LOGDIR"
echo
echo "Ctrl+C = stop complete stack"
echo "======================================================"
echo


while true; do

    sleep 2

    for pid in $PIDS; do

        if ! kill -0 "$pid" 2>/dev/null; then

            echo
            echo "[ERROR] process $pid exited"
            echo "[ERROR] stopping complete stack"

            exit 1

        fi

    done

done
