#!/usr/bin/env python3

import json
import signal
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

import r1a7_head_visual_servo_lowcmd as vs


WINDOW = "R1-A7 10cm Online Visual Servo"

# ------------------------------------------------------------
# Visual range
# Current calibration suggests 10 cm is roughly tens of pixels.
# This is only an image-space guard, not an exact metric conversion.
# ------------------------------------------------------------
MAX_TARGET_DISTANCE_PX = 45.0

# ------------------------------------------------------------
# Current-pose local calibration
# ------------------------------------------------------------
CALIB_MODEL_STEP_MM = 8.0
CALIB_HOLD_S = 2.0
RETURN_HOLD_S = 2.0
MIN_CALIB_IMAGE_MOTION_PX = 0.05
MAX_H_CONDITION = 20.0

# ------------------------------------------------------------
# Automatic coarse servo
# ------------------------------------------------------------
AUTO_MODEL_STEP_MM = 8.0
AUTO_HOLD_S = 2.0
AUTO_GAP_S = 0.30

# Coarse mode stops here.
# Fine servo will later handle the final alignment.
COARSE_STOP_ERROR_PX = 2.0

MIN_MEANINGFUL_DROP_PX = 0.05
MAX_ERROR_INCREASE_PX = 0.10
MAX_STALL_STEPS = 2
MAX_AUTO_STEPS = 40

# Online Broyden update strength.
ONLINE_ALPHA = 0.15
MIN_UPDATE_IMAGE_MOTION_PX = 0.04

# A step is useful only if the observed image displacement
# has a positive projection toward the target.
MIN_DIRECTIONAL_PROGRESS_PX = 0.02

# Reject online updates whose measured visual direction is
# grossly inconsistent with the current Jacobian prediction.
MIN_PREDICT_OBS_COSINE = 0.30

# Large unexpected wrist/tool rotation invalidates the local
# translation-only image model.
MAX_AUTO_ROTATION_DEG = 1.0


# ------------------------------------------------------------
# Robot safety guards
# ------------------------------------------------------------
MAX_STEP_JOINT_DELTA_RAD = 0.020
MAX_TOTAL_JOINT_DELTA_FROM_HOME_RAD = 0.20

# 10 cm physical workspace cap from experiment startup EE.
MAX_EE_FROM_HOME_M = 0.100

# XY-only control should not create large Z drift.
MAX_Z_DRIFT_FROM_HOME_M = 0.025

MAX_CORNERS = 40
MIN_FEATURES = 6
MIN_START_FEATURES = 8
MAX_TRACK_FAILS = 3

ONLINE_SAVE = Path(
    "/home/robot/unitree_sim_isaaclab_threading/"
    "r1a7_threading_project/vision/"
    "r1a7_head_online_command_jacobian.json"
)


def detect_features(gray, roi):
    x, y, w, h = roi

    mask = np.zeros_like(gray)
    mask[y:y + h, x:x + w] = 255

    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=5,
    )


def track_features(prev_gray, gray, prev_pts):
    if prev_gray is None or prev_pts is None:
        return None, None, 0

    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_pts,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if p1 is None or status is None:
        return None, None, 0

    pb, status_back, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        prev_gray,
        p1,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if pb is None or status_back is None:
        return None, None, 0

    p0 = prev_pts.reshape(-1, 2)
    p1f = p1.reshape(-1, 2)
    pbf = pb.reshape(-1, 2)

    good = (
        status.reshape(-1).astype(bool)
        & status_back.reshape(-1).astype(bool)
        & (np.linalg.norm(p0 - pbf, axis=1) < 1.0)
    )

    p0g = p0[good]
    p1g = p1f[good]

    if len(p0g) < MIN_FEATURES:
        return None, None, len(p0g)

    delta = np.median(
        p1g - p0g,
        axis=0,
    )

    return (
        p1g.reshape(-1, 1, 2).astype(np.float32),
        delta.astype(float),
        len(p1g),
    )


def roi_from_center(center, size, image_shape):
    w, h = size
    H, W = image_shape[:2]

    x = int(round(center[0] - w / 2.0))
    y = int(round(center[1] - h / 2.0))

    x = max(0, min(W - w, x))
    y = max(0, min(H - h, y))

    return x, y, w, h


def solve_cartesian_step(
    q_seed,
    dq_seed,
    left_target,
    right_rotation,
    right_position,
):
    ik = vs.R1A7_ArmIK(
        Unit_Test=False,
        Visualization=False,
    )

    ik.reset_target_calibration(
        q_seed
    )

    _, right_now = vs.get_ee_poses(
        ik,
        q_seed,
    )

    right_target = right_now.copy()

    right_target[:3, :3] = right_rotation
    right_target[:3, 3] = right_position

    q_goal, _ = ik.solve_ik(
        left_target,
        right_target,
        q_seed,
        dq_seed,
        position_only=False,
        max_joint_step=vs.IK_MAX_JOINT_STEP,
        rotation_weight=vs.IK_ROTATION_WEIGHT,
    )

    return np.asarray(
        q_goal,
        dtype=float,
    ).reshape(14)


def main():
    print(
        "R1-A7 10 cm ONLINE VISUAL SERVO",
        flush=True,
    )

    print(
        "8 mm coarse model steps + online Jacobian update.",
        flush=True,
    )

    vs.base.validate_robot_interface(
        vs.INTERFACE
    )

    guard = vs.acquire_lowcmd_guard(
        Path(__file__).name,
        topic=vs.COMMAND_TOPIC,
    )

    stopped = False

    def signal_handler(_sig=None, _frame=None):
        nonlocal stopped
        stopped = True

    signal.signal(
        signal.SIGINT,
        signal_handler,
    )

    signal.signal(
        signal.SIGTERM,
        signal_handler,
    )

    output = None

    try:
        # --------------------------------------------------
        # DDS
        # --------------------------------------------------
        vs.ChannelFactoryInitialize(
            vs.DOMAIN_ID,
            vs.INTERFACE,
        )

        crc = vs.CRC()

        state_buffer = vs.base.StateBuffer(
            crc
        )

        subscriber = vs.ChannelSubscriber(
            vs.STATE_TOPIC,
            vs.LowState_,
        )

        subscriber.Init(
            state_buffer.callback,
            10,
        )

        deadline = time.monotonic() + 10.0
        state = state_buffer.snapshot()

        while (
            state is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
            state = state_buffer.snapshot()

        if state is None:
            raise RuntimeError(
                "No valid rt/lowstate."
            )

        (
            upper_q,
            arm_q,
            arm_dq,
            gripper_q,
            gripper_dq,
            mode_machine,
            _received_at,
        ) = state

        q_home = np.asarray(
            arm_q,
            dtype=float,
        ).copy()

        vs.base.check_debug_mode(
            vs.MotionSwitcherClient
        )

        # --------------------------------------------------
        # FK model / experiment home
        # --------------------------------------------------
        ik_fk = vs.R1A7_ArmIK(
            Unit_Test=False,
            Visualization=False,
        )

        ik_fk.reset_target_calibration(
            q_home
        )

        left_home, right_home = (
            vs.get_ee_poses(
                ik_fk,
                q_home,
            )
        )

        home_rotation = (
            right_home[:3, :3].copy()
        )

        print(
            "Home right EE xyz:",
            np.round(
                right_home[:3, 3],
                6,
            ).tolist(),
            flush=True,
        )

        # --------------------------------------------------
        # LowCmd
        # --------------------------------------------------
        publisher = vs.ChannelPublisher(
            vs.COMMAND_TOPIC,
            vs.LowCmd_,
        )

        publisher.Init()

        output = vs.base.R1A7LowCmdOutput(
            publisher,
            vs.unitree_hg_msg_dds__LowCmd_,
            crc,
            vs.PUBLISH_FREQUENCY,
            vs.MAX_JOINT_SPEED,
            enable_gripper=True,
            gripper_kp=8.0,
            gripper_kd=0.4,
            gripper_speed=0.6,
            gripper_contact_hold=False,
        )

        # --------------------------------------------------
        # Camera
        # --------------------------------------------------
        video = vs.VideoClient()
        video.SetTimeout(3.0)
        video.Init()

        print()
        print(
            "Controls:",
            flush=True,
        )
        print(
            "  R : select rigid tweezer ROI",
            flush=True,
        )
        print(
            "  C : calibrate local X/Y visual response",
            flush=True,
        )
        print(
            "  T : select target region (up to ~10 cm scale)",
            flush=True,
        )
        print(
            "  A : start / stop automatic approach",
            flush=True,
        )
        print(
            "  H : return experiment home",
            flush=True,
        )
        print(
            "  Q / ESC : quit",
            flush=True,
        )

        print()
        print(
            "Keep emergency stop ready.",
            flush=True,
        )

        answer = input(
            "Type ENABLE to take over current posture: "
        ).strip()

        if answer.casefold() != "enable":
            print("Aborted.")
            return 2

        output.enable(
            upper_q,
            mode_machine,
            gripper_q=gripper_q,
            gripper_initial_mode="current",
            right_gripper_open_cap_current=True,
        )

        time.sleep(0.2)

        output.set_arm_goal(
            q_home,
            mode_machine,
        )

        print(
            "[ACTIVE] Current posture held.",
            flush=True,
        )

        cv2.namedWindow(
            WINDOW,
            cv2.WINDOW_NORMAL,
        )

        # --------------------------------------------------
        # Vision state
        # --------------------------------------------------
        roi = None
        roi_size = None

        points = None
        current_center = None

        target_roi = None
        target_center = None

        prev_gray = None
        feature_count = 0
        fail_count = 0

        image_crc_last = None

        # --------------------------------------------------
        # Online command -> image mapping
        #
        # delta_pixel = H_cmd @ delta_model_mm
        # --------------------------------------------------
        H_cmd = None
        H_valid = False

        # --------------------------------------------------
        # State machine
        # --------------------------------------------------
        phase = "IDLE"
        phase_started = 0.0

        calib_anchor_q = None
        calib_anchor_right = None
        calib_anchor_left = None
        calib_anchor_rotation = None

        calib_goal_x = None
        calib_goal_y = None

        calib_start_center = None
        calib_start_right = None

        calib_img_x = None
        calib_img_y = None

        calib_robot_x = None
        calib_robot_y = None

        auto_active = False
        auto_steps = 0
        stall_steps = 0
        next_auto_time = 0.0

        auto_command_mm = None
        auto_error_before = None
        auto_start_center = None
        auto_start_right = None

        # --------------------------------------------------
        # Helper: re-detect in current ROI position
        # --------------------------------------------------
        def redetect(gray):
            nonlocal points
            nonlocal feature_count
            nonlocal fail_count
            nonlocal prev_gray
            nonlocal roi

            if (
                current_center is None
                or roi_size is None
            ):
                return False

            roi = roi_from_center(
                current_center,
                roi_size,
                gray.shape,
            )

            candidate = detect_features(
                gray,
                roi,
            )

            feature_count = (
                0
                if candidate is None
                else len(candidate)
            )

            if (
                candidate is None
                or feature_count
                < MIN_START_FEATURES
            ):
                points = None
                return False

            points = candidate
            fail_count = 0
            prev_gray = gray.copy()

            return True

        # --------------------------------------------------
        # Main loop
        # --------------------------------------------------
        while not stopped:
            now = time.monotonic()

            state = state_buffer.snapshot()

            if state is None:
                raise RuntimeError(
                    "LowState disappeared."
                )

            (
                _upper_q,
                q_now,
                dq_now,
                _gripper_q,
                _gripper_dq,
                current_mode_machine,
                received_at,
            ) = state

            q_now = np.asarray(
                q_now,
                dtype=float,
            )

            dq_now = np.asarray(
                dq_now,
                dtype=float,
            )

            if (
                now - received_at
                > vs.LOWSTATE_TIMEOUT
            ):
                raise RuntimeError(
                    "LowState stale."
                )

            if output.error:
                raise RuntimeError(
                    output.error
                )

            left_now, right_now = (
                vs.get_ee_poses(
                    ik_fk,
                    q_now,
                )
            )

            ee_from_home = (
                right_now[:3, 3]
                - right_home[:3, 3]
            )

            ee_home_norm = float(
                np.linalg.norm(
                    ee_from_home
                )
            )

            if (
                ee_home_norm
                > MAX_EE_FROM_HOME_M
            ):
                auto_active = False

                output.set_arm_goal(
                    q_now,
                    current_mode_machine,
                )

                print(
                    "[SAFETY] EE exceeded 100 mm experiment range.",
                    flush=True,
                )
                print(
                    "Automatic motion stopped.",
                    flush=True,
                )

            if (
                abs(ee_from_home[2])
                > MAX_Z_DRIFT_FROM_HOME_M
            ):
                auto_active = False

                output.set_arm_goal(
                    q_now,
                    current_mode_machine,
                )

                print(
                    "[SAFETY] Z drift exceeded 25 mm.",
                    flush=True,
                )

            # --------------------------------------------------
            # Camera
            # --------------------------------------------------
            ret, data = video.GetImageSample()

            if (
                ret != 0
                or data is None
                or len(data) == 0
            ):
                continue

            raw = bytes(data)

            image_crc = zlib.crc32(
                raw
            )

            if image_crc == image_crc_last:
                continue

            image_crc_last = image_crc

            frame = cv2.imdecode(
                np.frombuffer(
                    raw,
                    dtype=np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                continue

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            # --------------------------------------------------
            # ROI tracking
            # --------------------------------------------------
            if (
                points is not None
                and prev_gray is not None
            ):
                new_points, delta, n_good = (
                    track_features(
                        prev_gray,
                        gray,
                        points,
                    )
                )

                if new_points is None:
                    fail_count += 1

                    print(
                        "[TRACK] transient failure "
                        f"{fail_count}/{MAX_TRACK_FAILS} "
                        f"(good={n_good})",
                        flush=True,
                    )

                    if (
                        fail_count
                        >= MAX_TRACK_FAILS
                    ):
                        auto_active = False
                        H_valid = False

                        output.set_arm_goal(
                            q_now,
                            current_mode_machine,
                        )

                        points = None
                        feature_count = 0

                        print(
                            "[STOP] Tracking lost. "
                            "Select R again.",
                            flush=True,
                        )

                else:
                    points = new_points
                    feature_count = n_good
                    fail_count = 0

                    if current_center is not None:
                        current_center = (
                            current_center
                            + delta
                        )

                    prev_gray = gray.copy()

            elif points is None:
                prev_gray = gray.copy()

            # --------------------------------------------------
            # Local calibration state machine
            # --------------------------------------------------
            if phase == "CAL_X_OUT":
                if (
                    now - phase_started
                    >= CALIB_HOLD_S
                    and current_center is not None
                ):
                    calib_img_x = (
                        current_center
                        - calib_start_center
                    )

                    calib_robot_x = (
                        right_now[:3, 3]
                        - calib_start_right[:3, 3]
                    )

                    print(
                        "[CAL X] image px =",
                        np.round(
                            calib_img_x,
                            5,
                        ).tolist(),
                        " actual EE mm =",
                        np.round(
                            calib_robot_x
                            * 1000.0,
                            4,
                        ).tolist(),
                        flush=True,
                    )

                    output.set_arm_goal(
                        calib_anchor_q,
                        current_mode_machine,
                    )

                    phase = "CAL_X_RETURN"
                    phase_started = now

            elif phase == "CAL_X_RETURN":
                if (
                    now - phase_started
                    >= RETURN_HOLD_S
                ):
                    if not redetect(gray):
                        phase = "IDLE"
                        H_valid = False

                        print(
                            "[CAL] Cannot re-detect ROI.",
                            flush=True,
                        )
                        continue

                    calib_start_center = (
                        current_center.copy()
                    )

                    calib_start_right = (
                        right_now.copy()
                    )

                    output.set_arm_goal(
                        calib_goal_y,
                        current_mode_machine,
                    )

                    phase = "CAL_Y_OUT"
                    phase_started = now

                    print(
                        "[CAL] +Y probe",
                        flush=True,
                    )

            elif phase == "CAL_Y_OUT":
                if (
                    now - phase_started
                    >= CALIB_HOLD_S
                    and current_center is not None
                ):
                    calib_img_y = (
                        current_center
                        - calib_start_center
                    )

                    calib_robot_y = (
                        right_now[:3, 3]
                        - calib_start_right[:3, 3]
                    )

                    print(
                        "[CAL Y] image px =",
                        np.round(
                            calib_img_y,
                            5,
                        ).tolist(),
                        " actual EE mm =",
                        np.round(
                            calib_robot_y
                            * 1000.0,
                            4,
                        ).tolist(),
                        flush=True,
                    )

                    output.set_arm_goal(
                        calib_anchor_q,
                        current_mode_machine,
                    )

                    phase = "CAL_Y_RETURN"
                    phase_started = now

            elif phase == "CAL_Y_RETURN":
                if (
                    now - phase_started
                    >= RETURN_HOLD_S
                ):
                    phase = "IDLE"

                    x_motion = float(
                        np.linalg.norm(
                            calib_img_x
                        )
                    )

                    y_motion = float(
                        np.linalg.norm(
                            calib_img_y
                        )
                    )

                    if (
                        x_motion
                        < MIN_CALIB_IMAGE_MOTION_PX
                        or y_motion
                        < MIN_CALIB_IMAGE_MOTION_PX
                    ):
                        H_valid = False

                        print(
                            "[CAL] FAILED: image probe "
                            "motion too small.",
                            flush=True,
                        )
                        continue

                    H_candidate = np.column_stack(
                        (
                            calib_img_x
                            / CALIB_MODEL_STEP_MM,
                            calib_img_y
                            / CALIB_MODEL_STEP_MM,
                        )
                    )

                    cond = float(
                        np.linalg.cond(
                            H_candidate
                        )
                    )

                    print()
                    print(
                        "[CAL] local H_cmd px/model-mm:",
                        flush=True,
                    )
                    print(
                        H_candidate,
                        flush=True,
                    )
                    print(
                        "[CAL] condition = "
                        f"{cond:.3f}",
                        flush=True,
                    )

                    if (
                        not np.isfinite(cond)
                        or cond
                        > MAX_H_CONDITION
                    ):
                        H_valid = False

                        print(
                            "[CAL] FAILED: bad condition.",
                            flush=True,
                        )
                        continue

                    H_cmd = H_candidate
                    H_valid = True
                    auto_active = False
                    auto_steps = 0
                    stall_steps = 0

                    print(
                        "[CAL] SUCCESS. "
                        "Now select T, then press A.",
                        flush=True,
                    )

            # --------------------------------------------------
            # Finish one AUTO step
            # --------------------------------------------------
            elif phase == "AUTO_MOVE":
                if (
                    now - phase_started
                    >= AUTO_HOLD_S
                    and current_center is not None
                ):
                    phase = "IDLE"

                    observed_img = (
                        current_center
                        - auto_start_center
                    )

                    actual_ee_step = (
                        right_now[:3, 3]
                        - auto_start_right[:3, 3]
                    )

                    error_after = (
                        target_center
                        - current_center
                    )

                    before_norm = float(
                        np.linalg.norm(
                            auto_error_before
                        )
                    )

                    after_norm = float(
                        np.linalg.norm(
                            error_after
                        )
                    )

                    error_drop = (
                        before_norm
                        - after_norm
                    )

                    print()
                    print(
                        f"[AUTO RESULT step={auto_steps + 1}]",
                        flush=True,
                    )

                    print(
                        "  command XY mm =",
                        np.round(
                            auto_command_mm,
                            4,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "  observed image delta px =",
                        np.round(
                            observed_img,
                            5,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "  actual EE step mm =",
                        np.round(
                            actual_ee_step
                            * 1000.0,
                            4,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        f"  error: "
                        f"{before_norm:.4f}px "
                        f"-> {after_norm:.4f}px "
                        f"drop={error_drop:+.4f}px",
                        flush=True,
                    )

                    # ------------------------------------------
                    # Directional validation BEFORE online update
                    # ------------------------------------------
                    target_direction = (
                        auto_error_before
                        / max(
                            before_norm,
                            1e-9,
                        )
                    )

                    directional_progress = float(
                        observed_img
                        @ target_direction
                    )

                    print(
                        "  directional progress = "
                        f"{directional_progress:+.4f} px",
                        flush=True,
                    )

                    observed_norm = float(
                        np.linalg.norm(
                            observed_img
                        )
                    )

                    predicted_img = (
                        H_cmd
                        @ auto_command_mm
                    )

                    predicted_norm = float(
                        np.linalg.norm(
                            predicted_img
                        )
                    )

                    predict_observe_cosine = -1.0

                    if (
                        observed_norm > 1e-9
                        and predicted_norm > 1e-9
                    ):
                        predict_observe_cosine = float(
                            (
                                predicted_img
                                @ observed_img
                            )
                            / (
                                predicted_norm
                                * observed_norm
                            )
                        )

                    print(
                        "  predicted image delta px =",
                        np.round(
                            predicted_img,
                            5,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "  predict/observe cosine = "
                        f"{predict_observe_cosine:+.3f}",
                        flush=True,
                    )

                    # A sample that moved away from the target
                    # must NEVER be used to adapt H.
                    sample_direction_valid = (
                        directional_progress
                        >= MIN_DIRECTIONAL_PROGRESS_PX
                    )

                    sample_model_valid = (
                        predict_observe_cosine
                        >= MIN_PREDICT_OBS_COSINE
                    )

                    allow_online_update = (
                        H_valid
                        and sample_direction_valid
                        and sample_model_valid
                        and observed_norm
                        >= MIN_UPDATE_IMAGE_MOTION_PX
                    )

                    if allow_online_update:
                        u = (
                            auto_command_mm.copy()
                        )

                        y = (
                            observed_img.copy()
                        )

                        residual = (
                            y - predicted_img
                        )

                        denom = float(
                            u @ u
                        )

                        H_candidate = (
                            H_cmd
                            + ONLINE_ALPHA
                            * np.outer(
                                residual,
                                u,
                            )
                            / denom
                        )

                        cond = float(
                            np.linalg.cond(
                                H_candidate
                            )
                        )

                        if (
                            np.isfinite(cond)
                            and cond
                            <= MAX_H_CONDITION
                        ):
                            H_cmd = H_candidate

                            ONLINE_SAVE.write_text(
                                json.dumps(
                                    {
                                        "H_pixel_per_model_mm":
                                            H_cmd.tolist(),
                                        "condition":
                                            cond,
                                        "last_command_mm":
                                            u.tolist(),
                                        "last_observed_pixel":
                                            y.tolist(),
                                        "directional_progress_px":
                                            directional_progress,
                                        "predict_observe_cosine":
                                            predict_observe_cosine,
                                        "auto_step":
                                            auto_steps + 1,
                                    },
                                    indent=2,
                                )
                            )

                            print(
                                "  [ONLINE H] accepted, "
                                f"alpha={ONLINE_ALPHA:.2f}, "
                                f"condition={cond:.3f}",
                                flush=True,
                            )

                            print(
                                H_cmd,
                                flush=True,
                            )

                        else:
                            print(
                                "  [ONLINE H] rejected: "
                                "ill-conditioned candidate.",
                                flush=True,
                            )

                    else:
                        print(
                            "  [ONLINE H] NOT updated.",
                            flush=True,
                        )

                        if not sample_direction_valid:
                            print(
                                "  reason: observed motion did "
                                "not progress toward target.",
                                flush=True,
                            )

                        elif not sample_model_valid:
                            print(
                                "  reason: observed direction "
                                "disagrees with Jacobian prediction.",
                                flush=True,
                            )

                    # ------------------------------------------
                    # Progress logic
                    # ------------------------------------------
                    if (
                        directional_progress
                        < MIN_DIRECTIONAL_PROGRESS_PX
                    ):
                        auto_active = False
                        H_valid = False

                        print(
                            "[STOP] Observed image motion did "
                            "not move toward the target.",
                            flush=True,
                        )

                        print(
                            "Bad sample was NOT used to update H.",
                            flush=True,
                        )

                        print(
                            "Press C to re-calibrate "
                            "at the current pose.",
                            flush=True,
                        )

                    elif (
                        error_drop
                        >= MIN_MEANINGFUL_DROP_PX
                    ):
                        stall_steps = 0
                        auto_steps += 1

                        print(
                            "[OK] Moving toward target.",
                            flush=True,
                        )

                    else:
                        stall_steps += 1
                        auto_steps += 1

                        print(
                            "[STALL] weak progress "
                            f"{stall_steps}/{MAX_STALL_STEPS}",
                            flush=True,
                        )

                        if (
                            stall_steps
                            >= MAX_STALL_STEPS
                        ):
                            auto_active = False
                            H_valid = False

                            print(
                                "[STOP] Repeated stall. "
                                "Press C to re-calibrate.",
                                flush=True,
                            )

                    next_auto_time = (
                        now + AUTO_GAP_S
                    )

            # --------------------------------------------------
            # Plan next automatic step
            # --------------------------------------------------
            if (
                phase == "IDLE"
                and auto_active
                and now >= next_auto_time
            ):
                if (
                    not H_valid
                    or target_center is None
                    or current_center is None
                ):
                    auto_active = False
                else:
                    error_px = (
                        target_center
                        - current_center
                    )

                    error_norm = float(
                        np.linalg.norm(
                            error_px
                        )
                    )

                    if (
                        error_norm
                        <= COARSE_STOP_ERROR_PX
                    ):
                        auto_active = False

                        print()
                        print(
                            "[TARGET] Coarse target reached: "
                            f"|e|={error_norm:.3f}px",
                            flush=True,
                        )

                        print(
                            "Stop coarse mode here. "
                            "Fine servo is the next stage.",
                            flush=True,
                        )

                    elif (
                        auto_steps
                        >= MAX_AUTO_STEPS
                    ):
                        auto_active = False

                        print(
                            "[STOP] Maximum auto steps reached.",
                            flush=True,
                        )

                    else:
                        try:
                            raw_command = (
                                np.linalg.solve(
                                    H_cmd,
                                    error_px,
                                )
                            )
                        except np.linalg.LinAlgError:
                            auto_active = False
                            H_valid = False

                            print(
                                "[STOP] H_cmd singular.",
                                flush=True,
                            )
                            continue

                        raw_norm = float(
                            np.linalg.norm(
                                raw_command
                            )
                        )

                        if (
                            not np.isfinite(raw_norm)
                            or raw_norm
                            < 1e-9
                        ):
                            auto_active = False
                            continue

                        # Direction only.
                        auto_command_mm = (
                            raw_command
                            / raw_norm
                            * AUTO_MODEL_STEP_MM
                        )

                        right_target_position = (
                            right_now[:3, 3].copy()
                        )

                        right_target_position[0] += (
                            auto_command_mm[0]
                            / 1000.0
                        )

                        right_target_position[1] += (
                            auto_command_mm[1]
                            / 1000.0
                        )

                        # Preserve the CURRENT measured
                        # end-effector orientation. Do not pull
                        # every step back toward startup rotation.
                        q_goal = solve_cartesian_step(
                            q_now,
                            dq_now,
                            left_now,
                            right_now[:3, :3].copy(),
                            right_target_position,
                        )

                        # Never move the left arm in this demo.
                        q_goal[:7] = q_home[:7]

                        step_joint_delta = float(
                            np.max(
                                np.abs(
                                    q_goal - q_now
                                )
                            )
                        )

                        total_joint_delta = float(
                            np.max(
                                np.abs(
                                    q_goal - q_home
                                )
                            )
                        )

                        print()
                        print(
                            f"[AUTO PLAN step={auto_steps + 1}]",
                            flush=True,
                        )

                        print(
                            "  image error px =",
                            np.round(
                                error_px,
                                4,
                            ).tolist(),
                            f"|e|={error_norm:.3f}",
                            flush=True,
                        )

                        print(
                            "  command XY mm =",
                            np.round(
                                auto_command_mm,
                                4,
                            ).tolist(),
                            f"norm={AUTO_MODEL_STEP_MM:.1f}",
                            flush=True,
                        )

                        print(
                            "  step joint delta = "
                            f"{step_joint_delta:.6f} rad",
                            flush=True,
                        )

                        print(
                            "  total joint delta home = "
                            f"{total_joint_delta:.6f} rad",
                            flush=True,
                        )

                        if (
                            step_joint_delta
                            > MAX_STEP_JOINT_DELTA_RAD
                        ):
                            auto_active = False

                            print(
                                "[SAFETY] Step joint delta "
                                "too large.",
                                flush=True,
                            )
                            continue

                        if (
                            total_joint_delta
                            > MAX_TOTAL_JOINT_DELTA_FROM_HOME_RAD
                        ):
                            auto_active = False

                            print(
                                "[SAFETY] Total joint range "
                                "limit reached.",
                                flush=True,
                            )
                            continue

                        auto_error_before = (
                            error_px.copy()
                        )

                        auto_start_center = (
                            current_center.copy()
                        )

                        auto_start_right = (
                            right_now.copy()
                        )

                        output.set_arm_goal(
                            q_goal,
                            current_mode_machine,
                        )

                        phase = "AUTO_MOVE"
                        phase_started = now

            # --------------------------------------------------
            # Display
            # --------------------------------------------------
            vis = frame.copy()

            if (
                current_center is not None
                and roi_size is not None
            ):
                x, y, w, h = roi_from_center(
                    current_center,
                    roi_size,
                    vis.shape,
                )

                cv2.rectangle(
                    vis,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                cv2.circle(
                    vis,
                    (
                        int(round(current_center[0])),
                        int(round(current_center[1])),
                    ),
                    4,
                    (0, 255, 0),
                    -1,
                )

            if target_roi is not None:
                tx, ty, tw, th = target_roi

                cv2.rectangle(
                    vis,
                    (tx, ty),
                    (tx + tw, ty + th),
                    (255, 0, 255),
                    2,
                )

                cv2.circle(
                    vis,
                    (
                        int(round(target_center[0])),
                        int(round(target_center[1])),
                    ),
                    4,
                    (255, 0, 255),
                    -1,
                )

            if (
                current_center is not None
                and target_center is not None
            ):
                cv2.arrowedLine(
                    vis,
                    (
                        int(round(current_center[0])),
                        int(round(current_center[1])),
                    ),
                    (
                        int(round(target_center[0])),
                        int(round(target_center[1])),
                    ),
                    (0, 255, 255),
                    2,
                    tipLength=0.15,
                )

            cv2.putText(
                vis,
                (
                    f"phase={phase} "
                    f"auto={auto_active} "
                    f"H={H_valid} "
                    f"features={feature_count}"
                ),
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                (
                    f"EE from home="
                    f"{ee_home_norm*1000.0:.1f} mm "
                    f"auto steps={auto_steps}"
                ),
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                "R:ROI  C:calib  T:target  A:auto  H:home",
                (15, vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                WINDOW,
                vis,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                break

            # --------------------------------------------------
            # R
            # --------------------------------------------------
            elif key in (
                ord("r"),
                ord("R"),
            ):
                if phase != "IDLE":
                    continue

                auto_active = False

                selected = cv2.selectROI(
                    "Select rigid tweezer ROI",
                    frame,
                    fromCenter=False,
                    showCrosshair=True,
                )

                cv2.destroyWindow(
                    "Select rigid tweezer ROI"
                )

                x, y, w, h = [
                    int(v)
                    for v in selected
                ]

                if w > 0 and h > 0:
                    roi = (
                        x, y, w, h
                    )

                    roi_size = (
                        w, h
                    )

                    current_center = np.asarray(
                        [
                            x + w / 2.0,
                            y + h / 2.0,
                        ],
                        dtype=float,
                    )

                    points = detect_features(
                        gray,
                        roi,
                    )

                    feature_count = (
                        0
                        if points is None
                        else len(points)
                    )

                    prev_gray = gray.copy()
                    fail_count = 0
                    H_valid = False

                    print(
                        "[ROI]",
                        roi,
                        "features=",
                        feature_count,
                        "center=",
                        np.round(
                            current_center,
                            3,
                        ).tolist(),
                        flush=True,
                    )

            # --------------------------------------------------
            # C
            # --------------------------------------------------
            elif key in (
                ord("c"),
                ord("C"),
            ):
                if phase != "IDLE":
                    continue

                if (
                    points is None
                    or current_center is None
                    or feature_count
                    < MIN_START_FEATURES
                ):
                    print(
                        "[CAL] Need valid R first.",
                        flush=True,
                    )
                    continue

                auto_active = False
                H_valid = False

                calib_anchor_q = (
                    q_now.copy()
                )

                calib_anchor_right = (
                    right_now.copy()
                )

                calib_anchor_left = (
                    left_now.copy()
                )

                calib_anchor_rotation = (
                    right_now[:3, :3].copy()
                )

                calib_start_center = (
                    current_center.copy()
                )

                calib_start_right = (
                    right_now.copy()
                )

                x_target = (
                    right_now[:3, 3].copy()
                )

                x_target[0] += (
                    CALIB_MODEL_STEP_MM
                    / 1000.0
                )

                y_target = (
                    right_now[:3, 3].copy()
                )

                y_target[1] += (
                    CALIB_MODEL_STEP_MM
                    / 1000.0
                )

                calib_goal_x = solve_cartesian_step(
                    q_now,
                    dq_now,
                    calib_anchor_left,
                    calib_anchor_rotation,
                    x_target,
                )

                calib_goal_y = solve_cartesian_step(
                    q_now,
                    dq_now,
                    calib_anchor_left,
                    calib_anchor_rotation,
                    y_target,
                )

                calib_goal_x[:7] = (
                    q_home[:7]
                )

                calib_goal_y[:7] = (
                    q_home[:7]
                )

                dx = float(
                    np.max(
                        np.abs(
                            calib_goal_x
                            - q_now
                        )
                    )
                )

                dy = float(
                    np.max(
                        np.abs(
                            calib_goal_y
                            - q_now
                        )
                    )
                )

                print()
                print(
                    "[CAL] Current-pose calibration",
                    flush=True,
                )

                print(
                    f"  X probe max joint delta = "
                    f"{dx:.6f} rad",
                    flush=True,
                )

                print(
                    f"  Y probe max joint delta = "
                    f"{dy:.6f} rad",
                    flush=True,
                )

                if (
                    dx > MAX_STEP_JOINT_DELTA_RAD
                    or dy > MAX_STEP_JOINT_DELTA_RAD
                ):
                    print(
                        "[CAL] Refused: probe IK too large.",
                        flush=True,
                    )
                    continue

                output.set_arm_goal(
                    calib_goal_x,
                    current_mode_machine,
                )

                phase = "CAL_X_OUT"
                phase_started = now

                print(
                    "[CAL] +X probe",
                    flush=True,
                )

            # --------------------------------------------------
            # T
            # --------------------------------------------------
            elif key in (
                ord("t"),
                ord("T"),
            ):
                if (
                    phase != "IDLE"
                    or current_center is None
                ):
                    continue

                auto_active = False

                selected = cv2.selectROI(
                    "Select TARGET region",
                    frame,
                    fromCenter=False,
                    showCrosshair=True,
                )

                cv2.destroyWindow(
                    "Select TARGET region"
                )

                tx, ty, tw, th = [
                    int(v)
                    for v in selected
                ]

                if tw > 0 and th > 0:
                    candidate = np.asarray(
                        [
                            tx + tw / 2.0,
                            ty + th / 2.0,
                        ],
                        dtype=float,
                    )

                    error = (
                        candidate
                        - current_center
                    )

                    distance = float(
                        np.linalg.norm(
                            error
                        )
                    )

                    if (
                        distance
                        > MAX_TARGET_DISTANCE_PX
                    ):
                        print(
                            "[TARGET] Too far: "
                            f"{distance:.2f}px > "
                            f"{MAX_TARGET_DISTANCE_PX:.1f}px",
                            flush=True,
                        )
                        continue

                    target_roi = (
                        tx, ty, tw, th
                    )

                    target_center = (
                        candidate
                    )

                    auto_steps = 0
                    stall_steps = 0

                    print(
                        "[TARGET] center=",
                        np.round(
                            target_center,
                            3,
                        ).tolist(),
                        "error=",
                        np.round(
                            error,
                            3,
                        ).tolist(),
                        f"|e|={distance:.3f}px",
                        flush=True,
                    )

            # --------------------------------------------------
            # A
            # --------------------------------------------------
            elif key in (
                ord("a"),
                ord("A"),
            ):
                if auto_active:
                    auto_active = False

                    output.set_arm_goal(
                        q_now,
                        current_mode_machine,
                    )

                    print(
                        "[AUTO] stopped by user.",
                        flush=True,
                    )

                else:
                    if not H_valid:
                        print(
                            "[AUTO] Need successful C calibration first.",
                            flush=True,
                        )
                        continue

                    if target_center is None:
                        print(
                            "[AUTO] Need T target first.",
                            flush=True,
                        )
                        continue

                    auto_active = True
                    auto_steps = 0
                    stall_steps = 0
                    next_auto_time = now

                    print(
                        "[AUTO] START",
                        flush=True,
                    )

            # --------------------------------------------------
            # H
            # --------------------------------------------------
            elif key in (
                ord("h"),
                ord("H"),
            ):
                auto_active = False
                H_valid = False

                output.set_arm_goal(
                    q_home,
                    current_mode_machine,
                )

                phase = "IDLE"

                print(
                    "[HOME] Returning to experiment home.",
                    flush=True,
                )

        return 0

    finally:
        if output is not None:
            output.close()

            print(
                "R1-A7 LowCmd publisher stopped.",
                flush=True,
            )

        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
