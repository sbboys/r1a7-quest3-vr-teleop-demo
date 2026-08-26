#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="${HOME}/unitree_sim_isaaclab"
VR_SCRIPT="${PROJECT_ROOT}/tools/r1a7_vr_dual_arm_g1ik_real.py"
AUTO_HOME_SCRIPT="${PROJECT_ROOT}/r1a7_wrench_project/scripts/auto_move_to_keyframe_v2_verified.py"
LOG_DIR="${PROJECT_ROOT}/r1a7_wrench_project/data/logs"

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [ -x "${HOME}/miniconda3/envs/tv/bin/python" ]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/tv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

VR_EXIT_HOME=42
VR_EXTRA_ARGS=("$@")
VR_LAST_PGID=""

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}" || exit 1
export PYTHONNOUSERSITE=1

wait_vr_released() {
    echo "Waiting for old VR process and port 8012 to be released..."

    local i vr_alive port_busy
    for i in $(seq 1 20); do
        vr_alive=0
        port_busy=0

        if pgrep -f 'python.*r1a7_vr_dual_arm_g1ik_real.py' >/dev/null 2>&1; then
            vr_alive=1
        fi

        if ss -ltnp 2>/dev/null | grep -q ':8012'; then
            port_busy=1
        fi

        if [ "${vr_alive}" -eq 0 ] && [ "${port_busy}" -eq 0 ]; then
            echo "VR process released and port 8012 is free."
            return 0
        fi

        echo "  waiting... ${i}/20 (vr_alive=${vr_alive}, port_busy=${port_busy})"
        sleep 0.5
    done

    echo "ERROR: old VR process or port 8012 is still active."
    pgrep -af 'r1a7_vr_dual_arm_g1ik_real.py' || true
    ss -ltnp 2>/dev/null | grep ':8012' || true
    return 1
}

check_conflicts() {
    local found
    found="$(pgrep -af 'python.*(r1a7_vr_dual_arm_g1ik_real.py|auto_move_to_keyframe_v2_verified.py)' || true)"
    if [ -n "${found}" ]; then
        echo "ERROR: conflicting robot-control process detected:"
        echo "${found}"
        return 1
    fi
    return 0
}

cleanup_vr_process_group() {
    local pgid="$1"
    local i

    if [ -z "${pgid}" ]; then
        echo "ERROR: VR process-group id is empty; refusing blind cleanup."
        return 1
    fi

    if ! [[ "${pgid}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid VR process-group id: ${pgid}"
        return 1
    fi

    echo "Cleaning VR process group PGID=${pgid} ..."

    # The VR main process may already have exited with code 42, but TeleVuer/Vuer
    # children can remain in the same process group.  Terminate only this VR group.
    if kill -0 -- "-${pgid}" 2>/dev/null; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
    fi

    for i in $(seq 1 15); do
        if ! kill -0 -- "-${pgid}" 2>/dev/null; then
            echo "VR process group ${pgid} released after TERM."
            return 0
        fi
        echo "  waiting for VR process group... ${i}/15"
        sleep 0.2
    done

    if kill -0 -- "-${pgid}" 2>/dev/null; then
        echo "VR process group ${pgid} still alive; sending final KILL."
        kill -KILL -- "-${pgid}" 2>/dev/null || true
        sleep 0.5
    fi

    if kill -0 -- "-${pgid}" 2>/dev/null; then
        echo "ERROR: VR process group ${pgid} still exists after KILL."
        return 1
    fi

    echo "VR process group ${pgid} fully released."
    return 0
}

run_auto_home_v2() {
    local reason="$1"
    local stamp log rc
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="${LOG_DIR}/auto_home_${reason}_${stamp}.log"

    echo
    echo "======================================================================"
    echo "VERIFIED AUTO_HOME V2 : ${reason}"
    echo "======================================================================"
    echo "Log: ${log}"
    echo "The verified controller still requires the manual phrase:"
    echo "  RECOVER HOME V2"
    echo "======================================================================"

    if ! check_conflicts; then
        return 1
    fi

    "${PYTHON_BIN}" "${AUTO_HOME_SCRIPT}" \
        HOME \
        --interface enp6s0 \
        --execute \
        2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}

    if [ "${rc}" -ne 0 ]; then
        echo "ERROR: AUTO_HOME V2 exited with code ${rc}."
        return "${rc}"
    fi

    if ! grep -Eq 'AUTO_HOME_READY[[:space:]]*=[[:space:]]*PASS' "${log}"; then
        echo "ERROR: AUTO_HOME_READY = PASS was not found."
        echo "VR will not start/restart."
        return 1
    fi

    echo
    echo "AUTO_HOME V2 PASS. Waiting for LowCmd ownership to clear..."
    sleep 2

    if ! check_conflicts; then
        echo "ERROR: a control process is still alive after AUTO_HOME V2."
        return 1
    fi

    echo "AUTO_HOME_READY = PASS; LowCmd owner cleared."
    return 0
}

run_vr() {
    local pgid_file rc
    pgid_file="$(mktemp /tmp/r1a7_vr_pgid.XXXXXX)"

    echo
    echo "======================================================================"
    echo "STARTING VR"
    echo "======================================================================"
    echo "Right A : arm/gripper teleop ON/OFF"
    echo "Right B : stop recording -> release LowCmd -> verified AUTO_HOME V2"
    echo "Left  X : start 60 s robot CSV + D435i episode"
    echo "======================================================================"

    # Start each VR session in its own Unix session/process group.  The small
    # wrapper writes its PID before exec(); after exec this same PID is the
    # Python VR process and also the PGID/session leader.  TeleVuer/Vuer child
    # processes inherit this process group unless they explicitly detach.
    setsid --wait bash -c '
        pgid_file="$1"
        shift
        printf "%s\n" "$$" > "$pgid_file"
        exec "$@"
    ' bash "${pgid_file}" \
        "${PYTHON_BIN}" "${VR_SCRIPT}" \
        --duration 0 \
        --treat_motion_ready_as_fresh \
        --debug_vr_data \
        --home_button right_ctrl_bButton \
        --record_button left_ctrl_aButton \
        --record_root r1a7_wrench_project/data/episodes \
        --record_episode_id right_safe_001 \
        --record_duration_s 60 \
        "${VR_EXTRA_ARGS[@]}"
    rc=$?

    if [ -s "${pgid_file}" ]; then
        VR_LAST_PGID="$(tr -dc '0-9' < "${pgid_file}")"
    else
        VR_LAST_PGID=""
    fi
    rm -f "${pgid_file}"

    echo "VR session PGID: ${VR_LAST_PGID:-UNKNOWN}"
    return "${rc}"
}

echo "======================================================================"
echo "R1-A7 SAFE VR / B-HOME LAUNCHER"
echo "======================================================================"
echo "Architecture:"
echo "  AUTO_HOME V2 -> VR -> B -> VR release -> AUTO_HOME V2 -> VR -> ..."
echo "Each VR session runs in its own process group."
echo "Only one process may own rt/lowcmd at any time."
echo "Python: ${PYTHON_BIN}"
echo "======================================================================"

# Safe initial entry: always establish the verified HOME before the first VR run.
if ! run_auto_home_v2 "initial"; then
    exit 1
fi

while true; do
    if ! check_conflicts; then
        exit 1
    fi

    run_vr
    vr_rc=$?

    echo
    echo "VR process exited with code ${vr_rc}."

    if [ "${vr_rc}" -eq 0 ]; then
        # Normal user exit: still clean any detached TeleVuer/Vuer children from
        # this specific VR process group before leaving the launcher.
        if [ -n "${VR_LAST_PGID}" ]; then
            cleanup_vr_process_group "${VR_LAST_PGID}" || true
        fi
        echo "Normal VR exit. Launcher finished."
        exit 0
    fi

    if [ "${vr_rc}" -ne "${VR_EXIT_HOME}" ]; then
        echo "ERROR: unexpected VR exit code ${vr_rc}."
        if [ -n "${VR_LAST_PGID}" ]; then
            cleanup_vr_process_group "${VR_LAST_PGID}" || true
        fi
        echo "AUTO_HOME V2 will NOT be started automatically."
        exit "${vr_rc}"
    fi

    echo
    echo "B-HOME request accepted."
    echo "VR cleanup returned exit code ${VR_EXIT_HOME}."

    if ! cleanup_vr_process_group "${VR_LAST_PGID}"; then
        echo "AUTO_HOME aborted: this VR process group could not be fully released."
        exit 1
    fi

    if ! wait_vr_released; then
        echo "AUTO_HOME aborted for safety."
        exit 1
    fi

    if ! check_conflicts; then
        echo "ERROR: VR/AUTO_HOME process conflict remains; refusing HOME motion."
        exit 1
    fi

    if ! run_auto_home_v2 "button_B"; then
        echo "ERROR: B-triggered AUTO_HOME V2 did not PASS."
        echo "VR will NOT restart."
        exit 1
    fi

    echo
    echo "HOME reached. Verifying old VR ownership is fully cleared before restart..."

    if ! wait_vr_released; then
        echo "VR restart aborted for safety."
        exit 1
    fi

    if ! check_conflicts; then
        echo "VR restart aborted: a robot-control process is still active."
        exit 1
    fi

    echo "Restarting VR process..."
    echo "If the Quest page does not reconnect automatically, reload/re-enter the VR page."
    sleep 2
done
