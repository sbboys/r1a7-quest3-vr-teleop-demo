#!/usr/bin/env bash
set -u

PRIMARY_REPO="${R1A7_PRIMARY_REPO:-/home/robot/unitree_sim_isaaclab_console}"
RESEARCH_REPO="${R1A7_RESEARCH_REPO:-/home/robot/unitree_sim_isaaclab_threading}"
ROBOT_IP="${R1A7_ROBOT_IP:-192.168.123.161}"
OUTPUT_PATH="${1:-}"

section() {
    printf '\n## %s\n' "$1"
}

collect_snapshot() {
    printf '# R1-A7 只读运行状态快照\n'
    printf 'generated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'hostname=%s\n' "$(hostname)"

    section "当前主工作区"
    printf 'path=%s\n' "$PRIMARY_REPO"
    if git -C "$PRIMARY_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf 'branch=%s\n' "$(git -C "$PRIMARY_REPO" branch --show-current)"
        printf 'head=%s\n' "$(git -C "$PRIMARY_REPO" rev-parse HEAD)"
        printf 'remote_demo=%s\n' "$(git -C "$PRIMARY_REPO" remote get-url demo 2>/dev/null || true)"
        printf 'status_begin\n'
        git -C "$PRIMARY_REPO" status --short
        printf 'status_end\n'
    else
        printf 'error=not_a_git_repository\n'
    fi

    section "研究参考区"
    printf 'path=%s\n' "$RESEARCH_REPO"
    if git -C "$RESEARCH_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf 'branch=%s\n' "$(git -C "$RESEARCH_REPO" branch --show-current)"
        printf 'head=%s\n' "$(git -C "$RESEARCH_REPO" rev-parse HEAD)"
        printf 'modified_count=%s\n' "$(git -C "$RESEARCH_REPO" status --short | wc -l)"
        printf 'note=reference_only_do_not_clean_automatically\n'
    else
        printf 'state=missing_or_not_git\n'
    fi

    section "网络（只读）"
    if command -v ip >/dev/null 2>&1; then
        ip -br -4 addr || true
        printf 'route_to_robot='
        ip route get "$ROBOT_IP" 2>&1 || true
    else
        printf 'ip_command=missing\n'
    fi

    section "8098 端口"
    if command -v ss >/dev/null 2>&1; then
        ss -ltnp '( sport = :8098 )' 2>&1 || true
    else
        printf 'ss_command=missing\n'
    fi

    section "相关进程（只读）"
    ps -eo pid=,ppid=,pgid=,etimes=,args= \
        | grep -E 'console3d|r1a7_.*control|lowcmd|teleop_hand_and_arm|task6_.*ibvs|r1a7_head_' \
        | grep -v -E 'grep -E|r1a7_context_snapshot' \
        || true

    section "控制器与锁关系（只读）"
    controller_pattern='r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py'
    controller_pids="$(pgrep -f "$controller_pattern" 2>/dev/null | tr '\n' ' ' || true)"
    controller_count="$(pgrep -fc "$controller_pattern" 2>/dev/null || true)"
    lock_holders=""
    if command -v fuser >/dev/null 2>&1 && [ -e /tmp/r1a7_vr_g1ik_real.lock ]; then
        lock_holders="$(fuser /tmp/r1a7_vr_g1ik_real.lock 2>/dev/null | xargs || true)"
    fi
    printf 'controller_process_count=%s\n' "${controller_count:-0}"
    printf 'controller_pids=%s\n' "${controller_pids:-none}"
    printf 'lowcmd_lock_holders=%s\n' "${lock_holders:-none}"
    printf '%s\n' \
        'note=多个同名 PID 可能是控制器主进程和派生子进程；必须结合 PPID、进程树、锁持有者和 DDS 发布者判断，不能仅按数量认定重复发布。'

    section "控制锁与相机服务状态"
    find /tmp -maxdepth 1 -type f \
        \( -name 'r1a7*.lock' -o -name 'r1a7_head_service_state.json' \) \
        -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null \
        | sort || true
    if [ -f /tmp/r1a7_head_service_state.json ]; then
        printf 'head_service_state='
        sed -n '1p' /tmp/r1a7_head_service_state.json
    fi

    section "视频设备稳定链接"
    if [ -d /dev/v4l/by-id ]; then
        find /dev/v4l/by-id -maxdepth 1 -type l -printf '%f -> %l\n' | sort
    else
        printf 'state=no_by_id_directory\n'
    fi

    section "解释边界"
    printf '%s\n' \
        '本文件仅记录生成时的只读状态。' \
        '它不证明机器人安全、LowCmd 发布者唯一、相机首帧有效或真机运动通过。'
}

if [ -n "$OUTPUT_PATH" ]; then
    mkdir -p "$(dirname "$OUTPUT_PATH")"
    collect_snapshot >"$OUTPUT_PATH"
    printf 'snapshot_written=%s\n' "$OUTPUT_PATH"
else
    collect_snapshot
fi
