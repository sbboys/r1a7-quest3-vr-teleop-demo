#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"
TEACH_PROFILE="${TEACH_PROFILE:-data/r1a7_teach/latest_profile.json}"
TEACH_PROFILE_BLEND="${TEACH_PROFILE_BLEND:-0.65}"

TEACH_ARGS=()
if [[ -f "${TEACH_PROFILE}" ]]; then
  TEACH_ARGS+=(
    --teach_profile "${TEACH_PROFILE}"
    --teach_profile_blend "${TEACH_PROFILE_BLEND}"
    --teach_elbow_limits
  )
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/r1a7_camera_real_teleop.py \
    --interface "${ROBOT_IFACE}" \
    --weight_index 31 \
    --robot_reference_q current \
    --duration 0 \
    --debug_pose \
    --camera_filter_alpha 0.18 \
    --max_human_delta_m 1.20 \
    --mirror_view \
    --direct_view_vertical \
    --direct_view_horizontal \
    --coordination_mode anatomic \
    --skeleton_blend 0.60 \
    --elbow_skeleton_blend 0.80 \
    --input_deadband 0.015 \
    --depth_deadband 0.025 \
    --skeleton_deadband 0.030 \
    --amplitude_scale 1.0 \
    --shoulder_pitch_sign 1.0 \
    --pitch_gain 4.5 \
    --depth_pitch_sign -1.0 \
    --depth_gain 2.8 \
    --pitch_depth_ratio 0.75 \
    --skeleton_lift_gain 2.6 \
    --skeleton_lift_sign -1.0 \
    --skeleton_reach_pitch_gain 0.70 \
    --teach_elbow_sign 1.0 \
    --elbow_vertical_ratio 0.00 \
    --elbow_depth_ratio -0.25 \
    --skeleton_elbow_sign 1.0 \
    --skeleton_elbow_gain 0.90 \
    --skeleton_lift_elbow_gain 0.00 \
    --pitch_min -1.35 \
    --pitch_max 1.45 \
    --elbow_min 0.35 \
    --elbow_max 1.75 \
    --shoulder_max_speed_rad_s 0.70 \
    --elbow_max_speed_rad_s 1.00 \
    --shoulder_direct_max_step 0.120 \
    --elbow_direct_max_step 0.160 \
    --shoulder_max_command_lead 0.85 \
    --elbow_max_command_lead 1.00 \
    "${TEACH_ARGS[@]}" \
    "$@"
