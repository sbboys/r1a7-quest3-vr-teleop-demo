#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
ROBOT_IFACE="${ROBOT_IFACE:-enx9c69d37d0967}"
TEACH_PROFILE="${TEACH_PROFILE:-data/r1a7_teach/latest_profile.json}"
TEACH_PROFILE_BLEND="${TEACH_PROFILE_BLEND:-0.65}"
HORIZONTAL_AMPLITUDE_SCALE="${HORIZONTAL_AMPLITUDE_SCALE:-1.8}"
TEACH_HORIZONTAL_ROLL_WEIGHT="${TEACH_HORIZONTAL_ROLL_WEIGHT:-0.30}"
TEACH_HORIZONTAL_SCALE="${TEACH_HORIZONTAL_SCALE:-0.28}"
TEACH_HORIZONTAL_YAW_WEIGHT="${TEACH_HORIZONTAL_YAW_WEIGHT:-0.85}"
TEACH_HORIZONTAL_YAW_SCALE="${TEACH_HORIZONTAL_YAW_SCALE:-0.20}"
TEACH_HORIZONTAL_ROLL_SIGN="${TEACH_HORIZONTAL_ROLL_SIGN:-1.0}"
TEACH_HORIZONTAL_YAW_SIGN="${TEACH_HORIZONTAL_YAW_SIGN:-1.0}"
TEACH_ELBOW_LIMITS="${TEACH_ELBOW_LIMITS:-0}"
TEACH_ELBOW_SCALE="${TEACH_ELBOW_SCALE:-2.40}"
ELBOW_SKELETON_BLEND="${ELBOW_SKELETON_BLEND:-0.50}"
SKELETON_ELBOW_GAIN="${SKELETON_ELBOW_GAIN:-0.45}"
ROLL_GAIN="${ROLL_GAIN:-3.0}"
YAW_GAIN="${YAW_GAIN:-3.8}"
SKELETON_SIDE_ROLL_GAIN="${SKELETON_SIDE_ROLL_GAIN:-1.40}"
SKELETON_SIDE_YAW_GAIN="${SKELETON_SIDE_YAW_GAIN:-1.60}"
ELBOW_DEPTH_RATIO="${ELBOW_DEPTH_RATIO:--0.10}"
ELBOW_MIN="${ELBOW_MIN:-0.20}"
ELBOW_MAX="${ELBOW_MAX:-1.55}"
ELBOW_MAX_SPEED_RAD_S="${ELBOW_MAX_SPEED_RAD_S:-0.55}"
ELBOW_DIRECT_MAX_STEP="${ELBOW_DIRECT_MAX_STEP:-0.080}"
ELBOW_MAX_COMMAND_LEAD="${ELBOW_MAX_COMMAND_LEAD:-0.45}"
SHOULDER_MAX_SPEED_RAD_S="${SHOULDER_MAX_SPEED_RAD_S:-1.00}"
SHOULDER_DIRECT_MAX_STEP="${SHOULDER_DIRECT_MAX_STEP:-0.160}"
SHOULDER_MAX_COMMAND_LEAD="${SHOULDER_MAX_COMMAND_LEAD:-1.05}"
ROLL_MIN="${ROLL_MIN:--0.85}"
ROLL_MAX="${ROLL_MAX:-0.35}"
YAW_MIN="${YAW_MIN:--1.15}"
YAW_MAX="${YAW_MAX:-1.15}"
LOWCMD_HOLD_INDICES="${LOWCMD_HOLD_INDICES:-22,23,24,25,26,27,28}"
DEX1_MAX_STEP="${DEX1_MAX_STEP:-0.35}"
DEX1_OPEN_ON_LOST_S="${DEX1_OPEN_ON_LOST_S:-1.0}"
DEX1_START_OPEN_S="${DEX1_START_OPEN_S:-1.0}"
DEX1_EXIT_OPEN_S="${DEX1_EXIT_OPEN_S:-1.5}"
DEX1_GRIP_OPEN_THRESHOLD="${DEX1_GRIP_OPEN_THRESHOLD:-0.55}"
DEX1_GRIP_CLOSE_THRESHOLD="${DEX1_GRIP_CLOSE_THRESHOLD:-0.75}"
DEX1_GRIP_OPEN_FRAMES="${DEX1_GRIP_OPEN_FRAMES:-2}"
DEX1_GRIP_CLOSE_FRAMES="${DEX1_GRIP_CLOSE_FRAMES:-5}"
DEX1_SIDE="${DEX1_SIDE:-right}"
ENABLE_DEX1="${ENABLE_DEX1:-1}"
CONTROL_CHANNEL="${CONTROL_CHANNEL:-arm_sdk}"
SKIP_R1A7_PREFLIGHT="${SKIP_R1A7_PREFLIGHT:-0}"

if [[ "${SKIP_R1A7_PREFLIGHT}" != "1" && ( "${CONTROL_CHANNEL}" == "arm_sdk" || "${CONTROL_CHANNEL}" == "lf_arm_sdk" ) ]]; then
  ./scripts/r1a7_real_arm_preflight.sh
elif [[ "${CONTROL_CHANNEL}" == "lowcmd" || "${CONTROL_CHANNEL}" == "lf_lowcmd" ]]; then
  echo "[R1-A7 CAMERA REAL] lowcmd mode: skipping R1 sport/loco preflight"
fi

TEACH_ARGS=()
if [[ -f "${TEACH_PROFILE}" ]]; then
  TEACH_ARGS+=(
    --teach_profile "${TEACH_PROFILE}"
    --teach_profile_blend "${TEACH_PROFILE_BLEND}"
  )
  if [[ "${TEACH_ELBOW_LIMITS}" == "1" ]]; then
    TEACH_ARGS+=(--teach_elbow_limits)
  fi
fi

DEX_ARGS=()
if [[ "${ENABLE_DEX1}" == "1" ]]; then
  DEX_ARGS+=(
    --enable_dex1
    --require_dex1
    --dex1_side "${DEX1_SIDE}"
    --dex1_binary
  )
fi

CONTROL_ARGS=()
case "${CONTROL_CHANNEL}" in
  lowcmd)
    CONTROL_ARGS+=(--debug_lowcmd --command_topic rt/lowcmd --lowcmd_hold_indices "${LOWCMD_HOLD_INDICES}")
    ;;
  lf_lowcmd)
    CONTROL_ARGS+=(--debug_lowcmd --state_topic rt/lf/lowstate --command_topic rt/lf/lowcmd --lowcmd_hold_indices "${LOWCMD_HOLD_INDICES}")
    ;;
  arm_sdk)
    CONTROL_ARGS+=(--r1_arm_sdk --command_topic rt/arm_sdk)
    ;;
  lf_arm_sdk)
    CONTROL_ARGS+=(--r1_arm_sdk --state_topic rt/lf/lowstate --command_topic rt/lf/arm_sdk)
    ;;
  *)
    echo "[R1-A7 CAMERA REAL] unsupported CONTROL_CHANNEL=${CONTROL_CHANNEL}; use lowcmd, lf_lowcmd, arm_sdk, or lf_arm_sdk" >&2
    exit 2
    ;;
esac

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/r1a7_camera_real_teleop.py \
    --interface "${ROBOT_IFACE}" \
    --enable_control \
    "${DEX_ARGS[@]}" \
    --enter_debug_mode \
    "${CONTROL_ARGS[@]}" \
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
    --elbow_skeleton_blend "${ELBOW_SKELETON_BLEND}" \
    --input_deadband 0.010 \
    --horizontal_amplitude_scale "${HORIZONTAL_AMPLITUDE_SCALE}" \
    --depth_deadband 0.025 \
    --skeleton_deadband 0.030 \
    --kp 32.0 \
    --kd 1.8 \
    --amplitude_scale 1.0 \
    --roll_gain "${ROLL_GAIN}" \
    --yaw_gain "${YAW_GAIN}" \
    --shoulder_pitch_sign 1.0 \
    --pitch_gain 4.5 \
    --depth_pitch_sign -1.0 \
    --depth_gain 2.8 \
    --pitch_depth_ratio 0.75 \
    --skeleton_lift_gain 2.6 \
    --skeleton_lift_sign -1.0 \
    --skeleton_side_roll_gain "${SKELETON_SIDE_ROLL_GAIN}" \
    --skeleton_side_yaw_gain "${SKELETON_SIDE_YAW_GAIN}" \
    --skeleton_reach_pitch_gain 0.70 \
    --teach_horizontal_scale "${TEACH_HORIZONTAL_SCALE}" \
    --teach_horizontal_roll_weight "${TEACH_HORIZONTAL_ROLL_WEIGHT}" \
    --teach_horizontal_yaw_scale "${TEACH_HORIZONTAL_YAW_SCALE}" \
    --teach_horizontal_yaw_weight "${TEACH_HORIZONTAL_YAW_WEIGHT}" \
    --teach_horizontal_roll_sign "${TEACH_HORIZONTAL_ROLL_SIGN}" \
    --teach_horizontal_yaw_sign "${TEACH_HORIZONTAL_YAW_SIGN}" \
    --teach_elbow_scale "${TEACH_ELBOW_SCALE}" \
    --teach_elbow_sign 1.0 \
    --elbow_vertical_ratio 0.00 \
    --elbow_depth_ratio "${ELBOW_DEPTH_RATIO}" \
    --skeleton_elbow_sign 1.0 \
    --skeleton_elbow_gain "${SKELETON_ELBOW_GAIN}" \
    --skeleton_lift_elbow_gain 0.00 \
    --pitch_min -1.35 \
    --pitch_max 1.45 \
    --roll_min "${ROLL_MIN}" \
    --roll_max "${ROLL_MAX}" \
    --yaw_min "${YAW_MIN}" \
    --yaw_max "${YAW_MAX}" \
    --elbow_min "${ELBOW_MIN}" \
    --elbow_max "${ELBOW_MAX}" \
    --max_speed_rad_s 0.25 \
    --shoulder_max_speed_rad_s "${SHOULDER_MAX_SPEED_RAD_S}" \
    --elbow_max_speed_rad_s "${ELBOW_MAX_SPEED_RAD_S}" \
    --direct_max_step 0.040 \
    --shoulder_direct_max_step "${SHOULDER_DIRECT_MAX_STEP}" \
    --elbow_direct_max_step "${ELBOW_DIRECT_MAX_STEP}" \
    --max_command_lead 0.25 \
    --shoulder_max_command_lead "${SHOULDER_MAX_COMMAND_LEAD}" \
    --elbow_max_command_lead "${ELBOW_MAX_COMMAND_LEAD}" \
    --dex1_max_step "${DEX1_MAX_STEP}" \
    --dex1_open_on_lost_s "${DEX1_OPEN_ON_LOST_S}" \
    --dex1_start_open_s "${DEX1_START_OPEN_S}" \
    --dex1_exit_open_s "${DEX1_EXIT_OPEN_S}" \
    --dex1_grip_open_threshold "${DEX1_GRIP_OPEN_THRESHOLD}" \
    --dex1_grip_close_threshold "${DEX1_GRIP_CLOSE_THRESHOLD}" \
    --dex1_grip_open_frames "${DEX1_GRIP_OPEN_FRAMES}" \
    --dex1_grip_close_frames "${DEX1_GRIP_CLOSE_FRAMES}" \
    "${TEACH_ARGS[@]}" \
    "$@"
