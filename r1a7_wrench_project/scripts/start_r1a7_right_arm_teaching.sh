#!/usr/bin/env bash

set -u
set -o pipefail

PROJECT_ROOT="$HOME/unitree_sim_isaaclab"

AUTO_HOME_SCRIPT="$PROJECT_ROOT/r1a7_wrench_project/scripts/auto_move_to_keyframe_v2_verified.py"

VR_SCRIPT="$PROJECT_ROOT/tools/r1a7_vr_dual_arm_g1ik_real.py"

LOG_ROOT="$PROJECT_ROOT/r1a7_wrench_project/results/startup_logs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

AUTO_HOME_LOG="$LOG_ROOT/auto_home_${TIMESTAMP}.log"


echo "============================================================"
echo " R1-A7 RIGHT-ARM WRENCH TASK STARTUP"
echo "============================================================"
echo
echo "Flow:"
echo "  AUTO_HOME V2"
echo "       -> AUTO_HOME_READY PASS"
echo "       -> release LowCmd"
echo "       -> start VR teleoperation"
echo "       -> press right A to enable teleop"
echo
echo "IMPORTANT:"
echo "  Keep emergency stop ready."
echo "  Keep robot workspace clear."
echo "============================================================"
echo


# ------------------------------------------------------------
# 1. Enter project
# ------------------------------------------------------------

cd "$PROJECT_ROOT" || exit 1


# ------------------------------------------------------------
# 2. Activate conda environment
# ------------------------------------------------------------

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "[STARTUP] ERROR: conda.sh not found."
    exit 1
fi

conda activate tv || {
    echo "[STARTUP] ERROR: failed to activate conda environment 'tv'."
    exit 1
}

export PYTHONNOUSERSITE=1


echo "[STARTUP] Python:"
which python
python --version
echo


# ------------------------------------------------------------
# 3. Check required files
# ------------------------------------------------------------

if [ ! -f "$AUTO_HOME_SCRIPT" ]; then
    echo "[STARTUP] ERROR: AUTO_HOME V2 script not found:"
    echo "  $AUTO_HOME_SCRIPT"
    exit 1
fi

if [ ! -f "$VR_SCRIPT" ]; then
    echo "[STARTUP] ERROR: VR script not found:"
    echo "  $VR_SCRIPT"
    exit 1
fi


# ------------------------------------------------------------
# 4. Refuse to start if another real controller is running
# ------------------------------------------------------------

if pgrep -f "auto_move_to_keyframe.*\.py" >/dev/null 2>&1; then
    echo
    echo "[STARTUP] ERROR:"
    echo "An AUTO_HOME/keyframe controller is already running."
    echo
    pgrep -af "auto_move_to_keyframe.*\.py"
    echo
    echo "Stop the existing controller before retrying."
    exit 1
fi

if pgrep -f "tools/r1a7_vr_dual_arm_g1ik_real.py" >/dev/null 2>&1; then
    echo
    echo "[STARTUP] ERROR:"
    echo "VR real-robot controller is already running."
    echo
    pgrep -af "tools/r1a7_vr_dual_arm_g1ik_real.py"
    echo
    echo "Stop the existing controller before retrying."
    exit 1
fi


# ------------------------------------------------------------
# 5. Syntax check before touching the robot
# ------------------------------------------------------------

echo "[STARTUP] Checking Python syntax..."

python -m py_compile "$AUTO_HOME_SCRIPT" || {
    echo "[STARTUP] ERROR: AUTO_HOME V2 syntax check failed."
    exit 1
}

python -m py_compile "$VR_SCRIPT" || {
    echo "[STARTUP] ERROR: VR script syntax check failed."
    exit 1
}

echo "[STARTUP] Syntax check PASS."
echo


# ------------------------------------------------------------
# 6. Create log directory
# ------------------------------------------------------------

mkdir -p "$LOG_ROOT"


# ------------------------------------------------------------
# 7. AUTO_HOME V2
# ------------------------------------------------------------

echo "============================================================"
echo " STAGE 1 / 2 : AUTO_HOME V2"
echo "============================================================"
echo
echo "The verified AUTO_HOME V2 controller will start."
echo
echo "When requested by the controller, type:"
echo
echo "    RECOVER HOME V2"
echo
echo "Do NOT start Quest arm teleoperation during this stage."
echo


python "$AUTO_HOME_SCRIPT" \
    HOME \
    --interface enp6s0 \
    --execute \
    2>&1 | tee "$AUTO_HOME_LOG"

AUTO_HOME_EXIT=${PIPESTATUS[0]}


# ------------------------------------------------------------
# 8. Check process exit code
# ------------------------------------------------------------

if [ "$AUTO_HOME_EXIT" -ne 0 ]; then
    echo
    echo "============================================================"
    echo " AUTO_HOME V2 FAILED"
    echo "============================================================"
    echo
    echo "Python exit code: $AUTO_HOME_EXIT"
    echo
    echo "VR WILL NOT START."
    echo
    echo "Log:"
    echo "  $AUTO_HOME_LOG"
    exit "$AUTO_HOME_EXIT"
fi


# ------------------------------------------------------------
# 9. Require explicit AUTO_HOME_READY PASS in log
# ------------------------------------------------------------

if ! grep -Eq "AUTO_HOME_READY[^A-Za-z0-9]*=?[^A-Za-z0-9]*PASS|AUTO_HOME_READY.*PASS" \
    "$AUTO_HOME_LOG"; then

    echo
    echo "============================================================"
    echo " AUTO_HOME VERIFICATION FAILED"
    echo "============================================================"
    echo
    echo "AUTO_HOME process exited, but:"
    echo
    echo "    AUTO_HOME_READY = PASS"
    echo
    echo "was not found in the log."
    echo
    echo "VR WILL NOT START."
    echo
    echo "Inspect:"
    echo "  $AUTO_HOME_LOG"
    exit 1
fi


echo
echo "============================================================"
echo " AUTO_HOME V2 PASS"
echo "============================================================"
echo
echo "AUTO_HOME_READY PASS confirmed."
echo


# ------------------------------------------------------------
# 10. Ensure AUTO_HOME publisher really exited
# ------------------------------------------------------------

sleep 1

if pgrep -f "auto_move_to_keyframe.*\.py" >/dev/null 2>&1; then
    echo "[STARTUP] ERROR:"
    echo "AUTO_HOME process is still present after completion:"
    pgrep -af "auto_move_to_keyframe.*\.py"
    echo
    echo "VR WILL NOT START."
    exit 1
fi


echo "[STARTUP] AUTO_HOME process exited."
echo "[STARTUP] Waiting 2 seconds before VR takeover..."
sleep 2


# ------------------------------------------------------------
# 11. Start VR
# ------------------------------------------------------------

echo
echo "============================================================"
echo " STAGE 2 / 2 : VR TELEOPERATION"
echo "============================================================"
echo
echo "Robot should now be at HOME."
echo
echo "Next:"
echo
echo "  1. Type ENABLE when the VR script requests confirmation."
echo "  2. Open Quest:"
echo "     https://192.168.1.103:8012/?ws=wss://192.168.1.103:8012"
echo "  3. Wait for fresh Quest controller poses."
echo "  4. Keep controllers still."
echo "  5. Press RIGHT A to enable teleoperation."
echo "  6. Test the right trigger/gripper first."
echo
echo "Press Ctrl+C to stop VR control."
echo "============================================================"
echo


exec python "$VR_SCRIPT" \
    --duration 0 \
    --enable_gripper \
    --gripper_mode lowcmd \
    --enable_gravity_comp \
    --gravity_comp_scale 0.20 \
    --gravity_comp_tau_limit 3.0 \
    --treat_motion_ready_as_fresh
