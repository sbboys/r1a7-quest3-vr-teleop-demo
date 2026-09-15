#!/usr/bin/env python3

import json
import signal
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

import r1a7_head_visual_servo_lowcmd as vs


WINDOW = "R1-A7 Absolute XY Visual Target"

SAVE_PATH = Path(
    "/home/robot/unitree_sim_isaaclab_threading/"
    "r1a7_threading_project/vision/"
    "r1a7_head_absolute_xy_homography.json"
)

# ============================================================
# FIRST-STAGE TEST WORKSPACE
# ============================================================

# First validation:
# +/-20 mm around startup XY -> approximately 40 x 40 mm.
#
# After this succeeds we will expand this to 50 mm.
CALIB_HALF_RANGE_MM = 20.0

# Internal Cartesian trajectory segment.
# This is NOT a visual-servo step.
PATH_STEP_MM = 5.0

# Absolute XY target tolerance.
WAYPOINT_TOL_MM = 2.0

# Calibration does not need to hit each theoretical grid point exactly.
# Homography stores ACTUAL pixel <-> ACTUAL FK XY correspondence.
CALIB_ACCEPT_TOL_MM = 4.0


STEP_HOLD_S = 1.2
SETTLE_S = 0.6
RETURN_HOLD_S = 2.0

MAX_NAV_STEPS_PER_TARGET = 24
MIN_PROGRESS_MM = 0.10
MAX_STALL_STEPS = 5

# After the nominal Cartesian reference reaches the physical target,
# allow a small bounded virtual lead to compensate LowCmd steady-state
# tracking error. This does NOT change the physical target.
BIAS_STEP_MM = 2.0
MAX_VIRTUAL_BIAS_MM = 10.0

# The commanded Cartesian reference must never run too far ahead
# of the actual FK position.
MAX_COMMAND_ACTUAL_LAG_MM = 25.0

# Joint-space difference between the cumulative command goal and
# actual measured q.
MAX_GOAL_ACTUAL_JOINT_GAP_RAD = 0.040


# ============================================================
# ROBOT SAFETY
# ============================================================

MAX_STEP_JOINT_DELTA_RAD = 0.020
MAX_TOTAL_JOINT_DELTA_RAD = 0.20

# +/-20-mm calibration corners are ~28 mm from center.
MAX_EE_FROM_HOME_M = 0.045

MAX_Z_DEVIATION_MM = 7.0
MAX_ROTATION_DEVIATION_DEG = 7.0

# ============================================================
# HOMOGRAPHY VALIDATION
# ============================================================

MIN_ROBOT_X_SPAN_MM = 25.0
MIN_ROBOT_Y_SPAN_MM = 25.0
MAX_HOMOGRAPHY_RMSE_MM = 3.0

# ============================================================
# TRACKER
# ============================================================

MAX_CORNERS = 40
MIN_FEATURES = 6
MIN_START_FEATURES = 8
MAX_TRACK_FAILS = 3


def detect_features(gray, roi):
    x, y, w, h = roi

    mask = np.zeros_like(gray)
    mask[y:y+h, x:x+w] = 255

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
        & (
            np.linalg.norm(
                p0 - pbf,
                axis=1,
            ) < 1.0
        )
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
        p1g.reshape(
            -1, 1, 2
        ).astype(np.float32),
        delta.astype(float),
        len(p1g),
    )


def roi_from_center(center, size, image_shape):
    w, h = size
    H, W = image_shape[:2]

    x = int(
        round(
            center[0] - w / 2.0
        )
    )

    y = int(
        round(
            center[1] - h / 2.0
        )
    )

    x = max(
        0,
        min(
            W - w,
            x,
        ),
    )

    y = max(
        0,
        min(
            H - h,
            y,
        ),
    )

    return (
        x,
        y,
        w,
        h,
    )


def rotation_error_deg(R_ref, R_now):
    R = R_ref.T @ R_now

    value = float(
        np.clip(
            (np.trace(R) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(value)
        )
    )


def fit_homography(samples):
    img = np.asarray(
        [
            [s["u"], s["v"]]
            for s in samples
        ],
        dtype=np.float64,
    )

    xy = np.asarray(
        [
            [s["x_mm"], s["y_mm"]]
            for s in samples
        ],
        dtype=np.float64,
    )

    H, mask = cv2.findHomography(
        img,
        xy,
        cv2.RANSAC,
        2.0,
    )

    if H is None:
        raise RuntimeError(
            "findHomography failed."
        )

    projected = cv2.perspectiveTransform(
        img.reshape(-1, 1, 2),
        H,
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected - xy,
        axis=1,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    return (
        H,
        np.linalg.inv(H),
        img,
        xy,
        errors,
        rmse,
        mask,
    )


def pixel_to_xy(H, pixel):
    p = np.asarray(
        pixel,
        dtype=np.float64,
    ).reshape(1, 1, 2)

    return cv2.perspectiveTransform(
        p,
        H,
    ).reshape(2)


def main():
    print(
        "R1-A7 CAMERA CLICK -> ABSOLUTE XY MOVE",
        flush=True,
    )

    print(
        "NO VR.",
        flush=True,
    )

    print(
        "Stage 1 workspace: +/-20 mm "
        "(approximately 40 x 40 mm).",
        flush=True,
    )

    print(
        "Coarse motion uses absolute robot XY targets, "
        "not online image Jacobian.",
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

    def stop_handler(_sig=None, _frame=None):
        nonlocal stopped
        stopped = True

    signal.signal(
        signal.SIGINT,
        stop_handler,
    )

    signal.signal(
        signal.SIGTERM,
        stop_handler,
    )

    output = None

    try:
        # ====================================================
        # DDS
        # ====================================================
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

        # ====================================================
        # FK reference
        # ====================================================
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

        home_xyz = (
            right_home[:3, 3].copy()
        )

        home_xy_mm = (
            home_xyz[:2]
            * 1000.0
        )

        home_rotation = (
            right_home[:3, :3].copy()
        )

        print(
            "Home right EE xyz:",
            np.round(
                home_xyz,
                6,
            ).tolist(),
            flush=True,
        )

        # ====================================================
        # LowCmd
        # ====================================================
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

        # ====================================================
        # Camera
        # ====================================================
        video = vs.VideoClient()
        video.SetTimeout(3.0)
        video.Init()

        print()
        print(
            "WARNING: REAL ROBOT AUTOMATIC MOTION.",
            flush=True,
        )

        print(
            "Clear at least a 10 cm region around the tweezer.",
            flush=True,
        )

        print(
            "Keep emergency stop ready.",
            flush=True,
        )

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
            "  C : automatically calibrate 9 planar points",
            flush=True,
        )

        print(
            "  T : select visual target",
            flush=True,
        )

        print(
            "  G : GO to absolute XY target",
            flush=True,
        )

        print(
            "  H : return startup arm q",
            flush=True,
        )

        print(
            "  Q / ESC : quit",
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
            "[ACTIVE] Startup posture held.",
            flush=True,
        )

        cv2.namedWindow(
            WINDOW,
            cv2.WINDOW_NORMAL,
        )

        # ====================================================
        # Tracker state
        # ====================================================
        roi = None
        roi_size = None

        points = None
        current_center = None

        prev_gray = None

        feature_count = 0
        fail_count = 0

        image_crc_last = None

        # ====================================================
        # Homography state
        # ====================================================
        samples = []

        H_img_to_xy = None
        H_xy_to_img = None
        H_valid = False
        calib_hull = None

        target_roi = None
        target_center = None
        target_xy_mm = None

        # ====================================================
        # NORMAL CLICK-TO-MOVE MODE:
        # load the one-time saved pixel -> robot XY calibration.
        # ====================================================
        if not SAVE_PATH.exists():
            raise RuntimeError(
                "Saved calibration file not found: "
                + str(SAVE_PATH)
            )

        calib_data = json.loads(
            SAVE_PATH.read_text()
        )

        if not bool(
            calib_data.get(
                "success",
                False,
            )
        ):
            raise RuntimeError(
                "Saved calibration is not marked SUCCESS."
            )

        H_img_to_xy = np.asarray(
            calib_data[
                "H_image_to_robot_xy_mm"
            ],
            dtype=np.float64,
        )

        H_xy_to_img = np.asarray(
            calib_data[
                "H_robot_xy_mm_to_image"
            ],
            dtype=np.float64,
        )

        samples = list(
            calib_data.get(
                "samples",
                []
            )
        )

        if len(samples) < 4:
            raise RuntimeError(
                "Saved calibration contains fewer than "
                "4 correspondences."
            )

        saved_img_points = np.asarray(
            [
                [
                    float(sample["u"]),
                    float(sample["v"]),
                ]
                for sample in samples
            ],
            dtype=np.float32,
        )

        calib_hull = cv2.convexHull(
            saved_img_points
        )

        H_valid = True

        print()
        print(
            "[CALIBRATION] Loaded saved pixel -> robot XY map:",
            flush=True,
        )
        print(
            SAVE_PATH,
            flush=True,
        )

        print(
            "[CALIBRATION] RMSE = "
            f"{float(calib_data.get('rmse_mm', -1.0)):.3f} mm",
            flush=True,
        )

        print(
            "[CALIBRATION] robot span X/Y = "
            f"{float(calib_data.get('robot_x_span_mm', -1.0)):.2f} / "
            f"{float(calib_data.get('robot_y_span_mm', -1.0)):.2f} mm",
            flush=True,
        )

        print(
            "[READY] Normal operation: "
            "ENABLE -> T -> select target -> G -> GO",
            flush=True,
        )

        # ====================================================
        # Automatic calibration grid
        #
        # Serpentine order to avoid huge jumps.
        # ====================================================
        h = CALIB_HALF_RANGE_MM

        calib_offsets_mm = [
            (0.0, 0.0),
            (-h, 0.0),
            (-h, -h),
            (0.0, -h),
            (+h, -h),
            (+h, 0.0),
            (+h, +h),
            (0.0, +h),
            (-h, +h),
        ]

        calib_index = 0

        # ====================================================
        # Navigation state
        # ====================================================
        phase = "IDLE"

        nav_purpose = None
        nav_target_xy_mm = None

        nav_step_count = 0
        nav_stall_count = 0

        nav_prev_distance_mm = None

        # Cumulative command reference.
        # This is deliberately separate from measured q / measured EE.
        nav_command_xy_mm = None
        nav_command_q = None
        nav_nominal_reached = False
        nav_bias_xy_mm = np.zeros(
            2,
            dtype=float,
        )

        phase_started = 0.0

        # ====================================================
        # Helpers
        # ====================================================
        def abort_motion(
            message,
            q_now,
            current_mode,
        ):
            nonlocal phase
            nonlocal nav_purpose

            output.set_arm_goal(
                q_now,
                current_mode,
            )

            phase = "IDLE"
            nav_purpose = None

            print(
                "[STOP]",
                message,
                flush=True,
            )

        def begin_navigation(
            target_xy,
            purpose,
            current_xy,
        ):
            nonlocal phase
            nonlocal nav_purpose
            nonlocal nav_target_xy_mm
            nonlocal nav_step_count
            nonlocal nav_stall_count
            nonlocal nav_prev_distance_mm
            nonlocal nav_command_xy_mm
            nonlocal nav_command_q
            nonlocal nav_nominal_reached
            nonlocal nav_bias_xy_mm

            nav_purpose = purpose

            nav_target_xy_mm = np.asarray(
                target_xy,
                dtype=float,
            ).copy()

            nav_step_count = 0
            nav_stall_count = 0

            nav_prev_distance_mm = float(
                np.linalg.norm(
                    nav_target_xy_mm
                    - current_xy
                )
            )

            # IMPORTANT:
            # The command reference starts at the measured pose once,
            # then advances cumulatively. It is NOT reset from measured
            # FK on every trajectory segment.
            nav_command_xy_mm = np.asarray(
                current_xy,
                dtype=float,
            ).copy()

            nav_command_q = np.asarray(
                q_now,
                dtype=float,
            ).copy()

            nav_nominal_reached = False

            nav_bias_xy_mm = np.zeros(
                2,
                dtype=float,
            )

            phase = "NAV_PLAN"

            print()
            print(
                f"[NAV] {purpose} target XY mm =",
                np.round(
                    nav_target_xy_mm,
                    3,
                ).tolist(),
                flush=True,
            )

            print(
                "[NAV] cumulative command reference enabled.",
                flush=True,
            )

        def save_calib_sample(
            right_now,
        ):
            nonlocal samples

            xy_mm = (
                right_now[:2, 3]
                * 1000.0
            )

            sample = {
                "u":
                    float(
                        current_center[0]
                    ),
                "v":
                    float(
                        current_center[1]
                    ),
                "x_mm":
                    float(
                        xy_mm[0]
                    ),
                "y_mm":
                    float(
                        xy_mm[1]
                    ),
                "z_mm":
                    float(
                        right_now[2, 3]
                        * 1000.0
                    ),
            }

            samples.append(
                sample
            )

            print(
                f"[SAMPLE {len(samples)}/9] "
                f"pixel=({sample['u']:.3f},"
                f"{sample['v']:.3f}) "
                f"XY=({sample['x_mm']:.3f},"
                f"{sample['y_mm']:.3f}) mm "
                f"Z={sample['z_mm']:.3f} mm",
                flush=True,
            )

        def finish_calibration():
            nonlocal H_img_to_xy
            nonlocal H_xy_to_img
            nonlocal H_valid
            nonlocal calib_hull

            (
                H,
                H_inv,
                img,
                xy,
                errors,
                rmse,
                mask,
            ) = fit_homography(
                samples
            )

            x_span = float(
                np.ptp(
                    xy[:, 0]
                )
            )

            y_span = float(
                np.ptp(
                    xy[:, 1]
                )
            )

            u_span = float(
                np.ptp(
                    img[:, 0]
                )
            )

            v_span = float(
                np.ptp(
                    img[:, 1]
                )
            )

            print()
            print(
                "========================================",
                flush=True,
            )

            print(
                "[CAL RESULT] H pixel -> robot XY mm:",
                flush=True,
            )

            print(
                H,
                flush=True,
            )

            print(
                "[CAL RESULT] errors mm =",
                np.round(
                    errors,
                    3,
                ).tolist(),
                flush=True,
            )

            print(
                f"[CAL RESULT] RMSE = "
                f"{rmse:.3f} mm",
                flush=True,
            )

            print(
                f"[CAL RESULT] robot span: "
                f"X={x_span:.2f} mm "
                f"Y={y_span:.2f} mm",
                flush=True,
            )

            print(
                f"[CAL RESULT] image span: "
                f"u={u_span:.3f} px "
                f"v={v_span:.3f} px",
                flush=True,
            )

            success = (
                rmse
                <= MAX_HOMOGRAPHY_RMSE_MM
                and x_span
                >= MIN_ROBOT_X_SPAN_MM
                and y_span
                >= MIN_ROBOT_Y_SPAN_MM
            )

            H_img_to_xy = H
            H_xy_to_img = H_inv

            calib_hull = cv2.convexHull(
                img.astype(
                    np.float32
                )
            )

            H_valid = bool(
                success
            )

            result = {
                "calib_half_range_mm":
                    CALIB_HALF_RANGE_MM,
                "H_image_to_robot_xy_mm":
                    H.tolist(),
                "H_robot_xy_mm_to_image":
                    H_inv.tolist(),
                "rmse_mm":
                    rmse,
                "robot_x_span_mm":
                    x_span,
                "robot_y_span_mm":
                    y_span,
                "image_u_span_px":
                    u_span,
                "image_v_span_px":
                    v_span,
                "samples":
                    samples,
                "success":
                    bool(success),
            }

            SAVE_PATH.write_text(
                json.dumps(
                    result,
                    indent=2,
                )
            )

            if success:
                print(
                    "[CAL RESULT] SUCCESS",
                    flush=True,
                )

                print(
                    "[CAL RESULT] T then G are enabled.",
                    flush=True,
                )

            else:
                print(
                    "[CAL RESULT] FAILED",
                    flush=True,
                )

                print(
                    "[CAL RESULT] Do NOT use G.",
                    flush=True,
                )

            print(
                "[CAL RESULT] saved:",
                SAVE_PATH,
                flush=True,
            )

            print(
                "========================================",
                flush=True,
            )

        # ====================================================
        # Main loop
        # ====================================================
        while not stopped:
            now = time.monotonic()

            state = state_buffer.snapshot()

            if state is None:
                continue

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

            current_xy_mm = (
                right_now[:2, 3]
                * 1000.0
            )

            ee_from_home = (
                right_now[:3, 3]
                - home_xyz
            )

            z_dev_mm = float(
                ee_from_home[2]
                * 1000.0
            )

            rot_dev_deg = rotation_error_deg(
                home_rotation,
                right_now[:3, :3],
            )

            # ------------------------------------------------
            # Global workspace guards
            # ------------------------------------------------
            if (
                np.linalg.norm(
                    ee_from_home
                )
                > MAX_EE_FROM_HOME_M
            ):
                abort_motion(
                    "EE exceeded first-stage workspace.",
                    q_now,
                    current_mode_machine,
                )

            if (
                abs(z_dev_mm)
                > MAX_Z_DEVIATION_MM
            ):
                abort_motion(
                    f"Z deviation too large: "
                    f"{z_dev_mm:+.2f} mm",
                    q_now,
                    current_mode_machine,
                )

            if (
                rot_dev_deg
                > MAX_ROTATION_DEVIATION_DEG
            ):
                abort_motion(
                    f"Tool rotation drift too large: "
                    f"{rot_dev_deg:.2f} deg",
                    q_now,
                    current_mode_machine,
                )

            # =================================================
            # RGB
            # =================================================
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

            if (
                image_crc
                == image_crc_last
            ):
                continue

            image_crc_last = (
                image_crc
            )

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

            # =================================================
            # Track ROI
            # =================================================
            if (
                points is not None
                and prev_gray is not None
            ):
                (
                    new_points,
                    delta,
                    n_good,
                ) = track_features(
                    prev_gray,
                    gray,
                    points,
                )

                if new_points is None:
                    fail_count += 1

                    print(
                        "[TRACK] transient failure "
                        f"{fail_count}/"
                        f"{MAX_TRACK_FAILS} "
                        f"(good={n_good})",
                        flush=True,
                    )

                    if (
                        fail_count
                        >= MAX_TRACK_FAILS
                    ):
                        abort_motion(
                            "ROI tracking lost. Press R again.",
                            q_now,
                            current_mode_machine,
                        )

                        points = None
                        feature_count = 0
                        H_valid = False

                else:
                    points = new_points
                    feature_count = n_good
                    fail_count = 0

                    if (
                        current_center
                        is not None
                    ):
                        current_center = (
                            current_center
                            + delta
                        )

                    prev_gray = (
                        gray.copy()
                    )

            elif points is None:
                prev_gray = (
                    gray.copy()
                )

            # =================================================
            # Absolute XY navigation
            # =================================================
            if phase == "NAV_PLAN":
                actual_remaining = (
                    nav_target_xy_mm
                    - current_xy_mm
                )

                actual_distance_mm = float(
                    np.linalg.norm(
                        actual_remaining
                    )
                )

                print(
                    f"[NAV] actual remaining = "
                    f"{actual_distance_mm:.3f} mm "
                    f"current XY="
                    f"{np.round(current_xy_mm,3).tolist()}",
                    flush=True,
                )

                # ------------------------------------------------
                # Physical target reached.
                # Freeze the measured posture instead of continuing
                # to hold any virtual bias beyond the target.
                # ------------------------------------------------
                accept_tolerance_mm = (
                    CALIB_ACCEPT_TOL_MM
                    if nav_purpose == "CAL"
                    else WAYPOINT_TOL_MM
                )

                if (
                    actual_distance_mm
                    <= accept_tolerance_mm
                ):
                    output.set_arm_goal(
                        q_now,
                        current_mode_machine,
                    )

                    print(
                        "[NAV] Physical target tolerance reached: "
                        f"{actual_distance_mm:.3f} mm <= "
                        f"{accept_tolerance_mm:.3f} mm "
                        f"(purpose={nav_purpose}).",
                        flush=True,
                    )

                    phase = "NAV_SETTLE"
                    phase_started = now

                else:
                    if (
                        nav_step_count
                        >= MAX_NAV_STEPS_PER_TARGET
                    ):
                        abort_motion(
                            "Too many cumulative trajectory steps.",
                            q_now,
                            current_mode_machine,
                        )
                        continue

                    # --------------------------------------------
                    # Stage A:
                    # advance the NOMINAL command reference toward
                    # the fixed physical target.
                    # --------------------------------------------
                    if not nav_nominal_reached:
                        command_remaining = (
                            nav_target_xy_mm
                            - nav_command_xy_mm
                        )

                        command_distance = float(
                            np.linalg.norm(
                                command_remaining
                            )
                        )

                        if (
                            command_distance
                            <= PATH_STEP_MM
                        ):
                            nav_command_xy_mm = (
                                nav_target_xy_mm.copy()
                            )

                            nav_nominal_reached = True

                            print(
                                "[NAV REF] nominal reference "
                                "reached physical target.",
                                flush=True,
                            )

                        else:
                            nav_command_xy_mm = (
                                nav_command_xy_mm
                                + command_remaining
                                / command_distance
                                * PATH_STEP_MM
                            )

                    # --------------------------------------------
                    # Stage B:
                    # nominal command already reached target but
                    # actual robot is still short.
                    #
                    # Add bounded virtual Cartesian bias toward the
                    # ACTUAL residual direction.
                    # --------------------------------------------
                    else:
                        actual_direction = (
                            actual_remaining
                            / actual_distance_mm
                        )

                        correction = min(
                            BIAS_STEP_MM,
                            actual_distance_mm,
                        )

                        nav_bias_xy_mm = (
                            nav_bias_xy_mm
                            + actual_direction
                            * correction
                        )

                        bias_norm = float(
                            np.linalg.norm(
                                nav_bias_xy_mm
                            )
                        )

                        if (
                            bias_norm
                            > MAX_VIRTUAL_BIAS_MM
                        ):
                            abort_motion(
                                "Virtual Cartesian bias exceeded "
                                f"{MAX_VIRTUAL_BIAS_MM:.1f} mm.",
                                q_now,
                                current_mode_machine,
                            )
                            continue

                        nav_command_xy_mm = (
                            nav_target_xy_mm
                            + nav_bias_xy_mm
                        )

                        print(
                            "[NAV REF] virtual bias mm =",
                            np.round(
                                nav_bias_xy_mm,
                                3,
                            ).tolist(),
                            f"|bias|={bias_norm:.3f}",
                            flush=True,
                        )

                    # --------------------------------------------
                    # Prevent desired reference from getting too far
                    # ahead of the real robot.
                    # --------------------------------------------
                    command_actual_lag = float(
                        np.linalg.norm(
                            nav_command_xy_mm
                            - current_xy_mm
                        )
                    )

                    print(
                        "[NAV REF] command XY mm =",
                        np.round(
                            nav_command_xy_mm,
                            3,
                        ).tolist(),
                        f"command/actual lag="
                        f"{command_actual_lag:.3f} mm",
                        flush=True,
                    )

                    if (
                        command_actual_lag
                        > MAX_COMMAND_ACTUAL_LAG_MM
                    ):
                        abort_motion(
                            "Command reference got too far ahead "
                            "of actual EE.",
                            q_now,
                            current_mode_machine,
                        )
                        continue

                    target_xyz = (
                        right_now[:3, 3]
                        .copy()
                    )

                    target_xyz[0] = (
                        nav_command_xy_mm[0]
                        / 1000.0
                    )

                    target_xyz[1] = (
                        nav_command_xy_mm[1]
                        / 1000.0
                    )

                    target_xyz[2] = (
                        home_xyz[2]
                    )

                    # --------------------------------------------
                    # CUMULATIVE IK
                    #
                    # Seed from previous COMMANDED q, not measured
                    # q_now. This is the critical change.
                    # --------------------------------------------
                    ik_step = vs.R1A7_ArmIK(
                        Unit_Test=False,
                        Visualization=False,
                    )

                    ik_step.reset_target_calibration(
                        nav_command_q
                    )

                    left_seed, right_seed = (
                        vs.get_ee_poses(
                            ik_step,
                            nav_command_q,
                        )
                    )

                    right_target = (
                        right_seed.copy()
                    )

                    right_target[:3, 3] = (
                        target_xyz
                    )

                    q_goal, _ = ik_step.solve_ik(
                        left_seed,
                        right_target,
                        nav_command_q,
                        np.zeros_like(
                            dq_now
                        ),
                        position_only=True,
                        max_joint_step=vs.IK_MAX_JOINT_STEP,
                        rotation_weight=0.0,
                    )

                    q_goal = np.asarray(
                        q_goal,
                        dtype=float,
                    ).reshape(14)

                    q_goal[:7] = (
                        q_home[:7]
                    )

                    # Difference between consecutive desired goals.
                    command_joint_step = float(
                        np.max(
                            np.abs(
                                q_goal
                                - nav_command_q
                            )
                        )
                    )

                    # Difference between new desired goal and actual q.
                    goal_actual_joint_gap = float(
                        np.max(
                            np.abs(
                                q_goal
                                - q_now
                            )
                        )
                    )

                    total_joint_delta = float(
                        np.max(
                            np.abs(
                                q_goal
                                - q_home
                            )
                        )
                    )

                    print(
                        f"[NAV PLAN] step="
                        f"{nav_step_count + 1} "
                        f"command_dq="
                        f"{command_joint_step:.6f} "
                        f"goal/actual_dq="
                        f"{goal_actual_joint_gap:.6f} "
                        f"total_dq="
                        f"{total_joint_delta:.6f}",
                        flush=True,
                    )

                    if (
                        command_joint_step
                        > MAX_STEP_JOINT_DELTA_RAD
                    ):
                        abort_motion(
                            "Cumulative IK command step "
                            "too large.",
                            q_now,
                            current_mode_machine,
                        )
                        continue

                    if (
                        goal_actual_joint_gap
                        > MAX_GOAL_ACTUAL_JOINT_GAP_RAD
                    ):
                        abort_motion(
                            "Desired joint goal got too far "
                            "ahead of measured q.",
                            q_now,
                            current_mode_machine,
                        )
                        continue

                    if (
                        total_joint_delta
                        > MAX_TOTAL_JOINT_DELTA_RAD
                    ):
                        abort_motion(
                            "Total joint displacement too large.",
                            q_now,
                            current_mode_machine,
                        )
                        continue

                    nav_prev_distance_mm = (
                        actual_distance_mm
                    )

                    # Store desired q for the NEXT IK solve.
                    nav_command_q = (
                        q_goal.copy()
                    )

                    output.set_arm_goal(
                        q_goal,
                        current_mode_machine,
                    )

                    nav_step_count += 1

                    phase = "NAV_WAIT"
                    phase_started = now

            elif phase == "NAV_WAIT":
                if (
                    now - phase_started
                    >= STEP_HOLD_S
                ):
                    new_distance = float(
                        np.linalg.norm(
                            nav_target_xy_mm
                            - current_xy_mm
                        )
                    )

                    progress = (
                        nav_prev_distance_mm
                        - new_distance
                    )

                    goal_actual_joint_gap = float(
                        np.max(
                            np.abs(
                                nav_command_q
                                - q_now
                            )
                        )
                    )

                    command_actual_lag = float(
                        np.linalg.norm(
                            nav_command_xy_mm
                            - current_xy_mm
                        )
                    )

                    print(
                        f"[NAV RESULT] "
                        f"progress={progress:+.3f} mm "
                        f"remaining={new_distance:.3f} mm "
                        f"cmd/actual={command_actual_lag:.3f} mm "
                        f"goal/actual_dq={goal_actual_joint_gap:.6f} "
                        f"EE XYZ mm="
                        f"{np.round(right_now[:3,3]*1000,3).tolist()}",
                        flush=True,
                    )

                    if (
                        goal_actual_joint_gap
                        > MAX_GOAL_ACTUAL_JOINT_GAP_RAD
                    ):
                        abort_motion(
                            "Joint tracking lag exceeded safety limit.",
                            q_now,
                            current_mode_machine,
                        )
                        continue

                    # Before the nominal reference reaches the final
                    # target we allow the cumulative reference to keep
                    # building, while command/actual lag provides the
                    # safety bound.
                    if (
                        progress
                        < MIN_PROGRESS_MM
                    ):
                        if nav_nominal_reached:
                            nav_stall_count += 1

                            print(
                                "[NAV] low progress after nominal "
                                f"target: {nav_stall_count}/"
                                f"{MAX_STALL_STEPS}",
                                flush=True,
                            )

                        else:
                            print(
                                "[NAV] low physical progress; "
                                "cumulative command reference "
                                "will continue.",
                                flush=True,
                            )

                    else:
                        nav_stall_count = 0

                    if (
                        nav_stall_count
                        >= MAX_STALL_STEPS
                    ):
                        abort_motion(
                            "Repeated stall even with cumulative "
                            "Cartesian bias.",
                            q_now,
                            current_mode_machine,
                        )
                        continue

                    phase = "NAV_PLAN"

            elif phase == "NAV_SETTLE":
                if (
                    now - phase_started
                    >= SETTLE_S
                ):
                    if nav_purpose == "CAL":
                        # --------------------------------------
                        # Calibration-point ROI recovery.
                        #
                        # Motion may temporarily reduce the number
                        # of LK features because of JPEG corruption
                        # or image blur. Do not throw away an
                        # otherwise successful robot move.
                        # --------------------------------------
                        need_redetect = (
                            current_center is None
                            or points is None
                            or feature_count < MIN_FEATURES
                        )

                        if (
                            need_redetect
                            and current_center is not None
                            and roi_size is not None
                        ):
                            recover_roi = roi_from_center(
                                current_center,
                                roi_size,
                                gray.shape,
                            )

                            recovered_points = detect_features(
                                gray,
                                recover_roi,
                            )

                            recovered_count = (
                                0
                                if recovered_points is None
                                else len(recovered_points)
                            )

                            print(
                                "[CAL ROI] re-detect at settled pose: "
                                f"features={recovered_count}",
                                flush=True,
                            )

                            if (
                                recovered_points is not None
                                and recovered_count >= MIN_FEATURES
                            ):
                                roi = recover_roi
                                points = recovered_points
                                feature_count = recovered_count
                                prev_gray = gray.copy()
                                fail_count = 0

                                print(
                                    "[CAL ROI] recovered.",
                                    flush=True,
                                )

                        # For continuous tracking, MIN_FEATURES is
                        # sufficient. MIN_START_FEATURES is only
                        # required when the user initially selects R.
                        if (
                            current_center is None
                            or points is None
                            or feature_count < MIN_FEATURES
                        ):
                            abort_motion(
                                "ROI could not be recovered at "
                                "calibration point.",
                                q_now,
                                current_mode_machine,
                            )

                            H_valid = False

                            print(
                                "[CAL] Robot reached the point, "
                                "but visual features were insufficient.",
                                flush=True,
                            )

                            print(
                                "Re-select R and restart C.",
                                flush=True,
                            )

                            continue

                        print(
                            "[CAL ROI] sample accepted with "
                            f"{feature_count} tracked features.",
                            flush=True,
                        )

                        save_calib_sample(
                            right_now
                        )

                        calib_index += 1

                        if (
                            calib_index
                            >= len(
                                calib_offsets_mm
                            )
                        ):
                            finish_calibration()

                            output.set_arm_goal(
                                q_home,
                                current_mode_machine,
                            )

                            phase = "CAL_RETURN"
                            phase_started = now

                        else:
                            offset = np.asarray(
                                calib_offsets_mm[
                                    calib_index
                                ],
                                dtype=float,
                            )

                            next_target = (
                                home_xy_mm
                                + offset
                            )

                            begin_navigation(
                                next_target,
                                "CAL",
                                current_xy_mm,
                            )

                    elif nav_purpose == "GO":
                        final_error = (
                            nav_target_xy_mm
                            - current_xy_mm
                        )

                        final_distance = float(
                            np.linalg.norm(
                                final_error
                            )
                        )

                        print()
                        print(
                            "[GO VERIFY] after settle:",
                            flush=True,
                        )

                        print(
                            "  target XY mm =",
                            np.round(
                                nav_target_xy_mm,
                                3,
                            ).tolist(),
                            flush=True,
                        )

                        print(
                            "  actual XY mm =",
                            np.round(
                                current_xy_mm,
                                3,
                            ).tolist(),
                            flush=True,
                        )

                        print(
                            "  residual XY mm =",
                            np.round(
                                final_error,
                                3,
                            ).tolist(),
                            f"|e|={final_distance:.3f} mm",
                            flush=True,
                        )

                        # Do not declare success merely because the
                        # robot briefly entered the tolerance band.
                        # Verify again after settling.
                        if (
                            final_distance
                            > WAYPOINT_TOL_MM
                        ):
                            print(
                                "[GO VERIFY] Settling drift moved "
                                "the EE outside tolerance.",
                                flush=True,
                            )

                            print(
                                "[GO VERIFY] Continuing absolute "
                                "XY correction.",
                                flush=True,
                            )

                            nav_prev_distance_mm = (
                                final_distance
                            )

                            phase = "NAV_PLAN"

                        else:
                            print(
                                "[GO RESULT] SUCCESS: absolute XY "
                                "target remains inside tolerance "
                                "after settling.",
                                flush=True,
                            )

                            if (
                                current_center is not None
                                and target_center is not None
                            ):
                                pixel_error = (
                                    target_center
                                    - current_center
                                )

                                print(
                                    "  final visual error px =",
                                    np.round(
                                        pixel_error,
                                        3,
                                    ).tolist(),
                                    f"|e|="
                                    f"{np.linalg.norm(pixel_error):.3f}",
                                    flush=True,
                                )

                            phase = "IDLE"
                            nav_purpose = None

            elif phase == "CAL_RETURN":
                if (
                    now - phase_started
                    >= RETURN_HOLD_S
                ):
                    _, returned_right = (
                        vs.get_ee_poses(
                            ik_fk,
                            q_now,
                        )
                    )

                    print(
                        "[CAL] returned toward startup q.",
                        flush=True,
                    )

                    print(
                        "[CAL] current EE error from home mm =",
                        np.round(
                            (
                                returned_right[:3, 3]
                                - home_xyz
                            )
                            * 1000.0,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    phase = "IDLE"
                    nav_purpose = None

            # =================================================
            # Display
            # =================================================
            vis = frame.copy()

            if (
                current_center is not None
                and roi_size is not None
            ):
                rw, rh = roi_size

                p0 = (
                    int(
                        round(
                            current_center[0]
                            - rw / 2
                        )
                    ),
                    int(
                        round(
                            current_center[1]
                            - rh / 2
                        )
                    ),
                )

                p1 = (
                    int(
                        round(
                            current_center[0]
                            + rw / 2
                        )
                    ),
                    int(
                        round(
                            current_center[1]
                            + rh / 2
                        )
                    ),
                )

                cv2.rectangle(
                    vis,
                    p0,
                    p1,
                    (0, 255, 0),
                    2,
                )

                cv2.circle(
                    vis,
                    (
                        int(
                            round(
                                current_center[0]
                            )
                        ),
                        int(
                            round(
                                current_center[1]
                            )
                        ),
                    ),
                    4,
                    (0, 255, 0),
                    -1,
                )

            if target_roi is not None:
                tx, ty, tw, th = (
                    target_roi
                )

                cv2.rectangle(
                    vis,
                    (tx, ty),
                    (tx + tw, ty + th),
                    (255, 0, 255),
                    2,
                )

            for i, s in enumerate(samples):
                p = (
                    int(round(s["u"])),
                    int(round(s["v"])),
                )

                cv2.circle(
                    vis,
                    p,
                    5,
                    (255, 0, 255),
                    2,
                )

                cv2.putText(
                    vis,
                    str(i + 1),
                    (
                        p[0] + 4,
                        p[1] - 4,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 255),
                    1,
                )

            cv2.putText(
                vis,
                (
                    f"phase={phase} "
                    f"samples={len(samples)}/9 "
                    f"features={feature_count} "
                    f"H={H_valid}"
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
                    f"EE XY mm="
                    f"({current_xy_mm[0]:+.1f},"
                    f"{current_xy_mm[1]:+.1f}) "
                    f"Zdev={z_dev_mm:+.1f} "
                    f"Rdev={rot_dev_deg:.1f}deg"
                ),
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                "R:ROI  C:auto-calib  T:target  G:go  H:home",
                (15, vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                WINDOW,
                vis,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # =================================================
            # Q
            # =================================================
            if key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                break

            # =================================================
            # R
            # =================================================
            elif key in (
                ord("r"),
                ord("R"),
            ):
                if phase != "IDLE":
                    print(
                        "[ROI] Cannot change ROI during motion.",
                        flush=True,
                    )
                    continue

                selected = cv2.selectROI(
                    "Select rigid tweezer ROI",
                    frame,
                    fromCenter=False,
                    showCrosshair=True,
                )

                cv2.destroyWindow(
                    "Select rigid tweezer ROI"
                )

                x, y, w, h_roi = [
                    int(v)
                    for v in selected
                ]

                if (
                    w > 0
                    and h_roi > 0
                ):
                    roi = (
                        x, y, w, h_roi
                    )

                    roi_size = (
                        w, h_roi
                    )

                    current_center = np.asarray(
                        [
                            x + w / 2.0,
                            y + h_roi / 2.0,
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

                    prev_gray = (
                        gray.copy()
                    )

                    fail_count = 0

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

            # =================================================
            # C - automatic 9 point calibration
            # =================================================
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
                        "[CAL] Need valid R first "
                        f"(features >= {MIN_START_FEATURES}).",
                        flush=True,
                    )
                    continue

                confirm = input(
                    "Type CALIBRATE to execute automatic "
                    "40 x 40 mm calibration: "
                ).strip()

                # OpenCV key C can occasionally remain in terminal
                # stdin, producing strings such as "CCALIBRATE".
                # Accept a confirmation whose tail is CALIBRATE.
                confirm_normalized = (
                    confirm.casefold()
                )

                if not confirm_normalized.endswith(
                    "calibrate"
                ):
                    print(
                        "[CAL] cancelled. "
                        f"Received input={confirm!r}",
                        flush=True,
                    )
                    continue

                print(
                    "[CAL] confirmation accepted. "
                    f"Received input={confirm!r}",
                    flush=True,
                )

                samples.clear()

                H_valid = False
                H_img_to_xy = None
                H_xy_to_img = None
                calib_hull = None

                target_roi = None
                target_center = None
                target_xy_mm = None

                calib_index = 0

                first_target = (
                    home_xy_mm
                    + np.asarray(
                        calib_offsets_mm[0],
                        dtype=float,
                    )
                )

                print()
                print(
                    "[CAL] START automatic 9-point calibration.",
                    flush=True,
                )

                begin_navigation(
                    first_target,
                    "CAL",
                    current_xy_mm,
                )

            # =================================================
            # T
            # =================================================
            elif key in (
                ord("t"),
                ord("T"),
            ):
                if phase != "IDLE":
                    continue

                if not H_valid:
                    print(
                        "[TARGET] Need successful C calibration first.",
                        flush=True,
                    )
                    continue

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

                if (
                    tw <= 0
                    or th <= 0
                ):
                    continue

                candidate = np.asarray(
                    [
                        tx + tw / 2.0,
                        ty + th / 2.0,
                    ],
                    dtype=float,
                )

                inside = cv2.pointPolygonTest(
                    calib_hull,
                    (
                        float(candidate[0]),
                        float(candidate[1]),
                    ),
                    False,
                )

                if inside < 0:
                    print(
                        "[TARGET] Rejected: target is outside "
                        "the calibrated image region.",
                        flush=True,
                    )
                    continue

                candidate_xy = pixel_to_xy(
                    H_img_to_xy,
                    candidate,
                )

                target_roi = (
                    tx, ty, tw, th
                )

                target_center = (
                    candidate
                )

                target_xy_mm = (
                    candidate_xy
                )

                print()
                print(
                    "[TARGET] pixel center =",
                    np.round(
                        target_center,
                        3,
                    ).tolist(),
                    flush=True,
                )

                print(
                    "[TARGET] absolute robot XY mm =",
                    np.round(
                        target_xy_mm,
                        3,
                    ).tolist(),
                    flush=True,
                )

                print(
                    "[TARGET] current robot XY mm =",
                    np.round(
                        current_xy_mm,
                        3,
                    ).tolist(),
                    flush=True,
                )

                print(
                    "[TARGET] required XY displacement mm =",
                    np.round(
                        target_xy_mm
                        - current_xy_mm,
                        3,
                    ).tolist(),
                    flush=True,
                )

            # =================================================
            # G
            # =================================================
            elif key in (
                ord("g"),
                ord("G"),
            ):
                if phase != "IDLE":
                    continue

                if (
                    not H_valid
                    or target_xy_mm
                    is None
                ):
                    print(
                        "[GO] Need successful C and T first.",
                        flush=True,
                    )
                    continue

                requested_distance = float(
                    np.linalg.norm(
                        target_xy_mm
                        - current_xy_mm
                    )
                )

                print()
                print(
                    f"[GO] Absolute target distance = "
                    f"{requested_distance:.2f} mm",
                    flush=True,
                )

                confirm = input(
                    "Type GO to move the real robot "
                    "to this absolute XY target: "
                ).strip()

                confirm_normalized = (
                    confirm.casefold()
                )

                if not confirm_normalized.endswith(
                    "go"
                ):
                    print(
                        "[GO] cancelled. "
                        f"Received input={confirm!r}",
                        flush=True,
                    )
                    continue

                print(
                    "[GO] confirmation accepted. "
                    f"Received input={confirm!r}",
                    flush=True,
                )

                begin_navigation(
                    target_xy_mm,
                    "GO",
                    current_xy_mm,
                )

            # =================================================
            # H
            # =================================================
            elif key in (
                ord("h"),
                ord("H"),
            ):
                phase = "IDLE"
                nav_purpose = None

                output.set_arm_goal(
                    q_home,
                    current_mode_machine,
                )

                print(
                    "[HOME] Returning to startup arm q.",
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
