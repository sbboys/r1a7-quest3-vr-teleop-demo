#!/usr/bin/env python3

import json
import signal
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

import r1a7_head_visual_servo_lowcmd as vs


WINDOW = "R1-A7 Repeated Visual Jacobian Calibration"

CALIB_STEP_M = 0.005
CALIB_REPEATS = 3

MOVE_HOLD_S = 2.0
RETURN_HOLD_S = 2.0

MIN_FEATURES = 6
MIN_START_FEATURES = 8
MAX_CORNERS = 40
MAX_TRACK_FAILS = 3

MIN_ACTUAL_EXCURSION_M = 0.00035
MIN_IMAGE_MOTION_PX = 0.03
MAX_JACOBIAN_CONDITION = 50.0

MAX_GOAL_JOINT_DELTA_RAD = 0.012

SAVE_PATH = Path(
    "/home/robot/unitree_sim_isaaclab_threading/"
    "r1a7_threading_project/vision/"
    "r1a7_head_local_jacobian_repeat.json"
)


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

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
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

    if next_pts is None or status is None:
        return None, None, 0

    back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        prev_gray,
        next_pts,
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

    if back_pts is None or back_status is None:
        return None, None, 0

    p0 = prev_pts.reshape(-1, 2)
    p1 = next_pts.reshape(-1, 2)
    pb = back_pts.reshape(-1, 2)

    s1 = status.reshape(-1).astype(bool)
    s2 = back_status.reshape(-1).astype(bool)

    fb_error = np.linalg.norm(
        p0 - pb,
        axis=1,
    )

    good = (
        s1
        & s2
        & (fb_error < 1.0)
    )

    p0_good = p0[good]
    p1_good = p1[good]

    if len(p0_good) < MIN_FEATURES:
        return None, None, len(p0_good)

    delta = np.median(
        p1_good - p0_good,
        axis=0,
    )

    return (
        p1_good.reshape(
            -1, 1, 2
        ).astype(np.float32),
        delta.astype(float),
        len(p1_good),
    )


def make_fixed_goal(
    q_home,
    dq_home,
    axis,
):
    """
    Fresh IK object for each axis.

    Important:
    Solve ONCE and then hold the returned joint goal.
    This matches the successful standalone 5-mm LowCmd tests.
    """

    ik_goal = vs.R1A7_ArmIK(
        Unit_Test=False,
        Visualization=False,
    )

    ik_goal.reset_target_calibration(
        q_home
    )

    left_home, right_home = (
        vs.get_ee_poses(
            ik_goal,
            q_home,
        )
    )

    target = right_home.copy()
    target[axis, 3] += CALIB_STEP_M

    q_goal, _ = ik_goal.solve_ik(
        left_home,
        target,
        q_home,
        dq_home,
        position_only=False,
        max_joint_step=vs.IK_MAX_JOINT_STEP,
        rotation_weight=vs.IK_ROTATION_WEIGHT,
    )

    q_goal = np.asarray(
        q_goal,
        dtype=float,
    ).reshape(14)

    _, predicted = vs.get_ee_poses(
        ik_goal,
        q_goal,
    )

    predicted_delta = (
        predicted[:3, 3]
        - right_home[:3, 3]
    )

    return (
        q_goal,
        predicted_delta,
    )


def main():
    print(
        "R1-A7 REPEATED VISUAL JACOBIAN CALIBRATION",
        flush=True,
    )

    print(
        "X x3 + Y x3, fixed IK joint targets.",
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
        # --------------------------------------------------
        # DDS state
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
            received_at,
        ) = state

        q_home = np.asarray(
            arm_q,
            dtype=float,
        ).copy()

        dq_home = np.asarray(
            arm_dq,
            dtype=float,
        ).copy()

        print(
            "Measured arm q:",
            np.round(
                q_home,
                5,
            ).tolist(),
            flush=True,
        )

        vs.base.check_debug_mode(
            vs.MotionSwitcherClient
        )

        # --------------------------------------------------
        # FK model
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

        print(
            "Home right EE xyz:",
            np.round(
                right_home[:3, 3],
                6,
            ).tolist(),
            flush=True,
        )

        # --------------------------------------------------
        # Fixed X/Y joint goals
        # --------------------------------------------------
        q_goal_x, pred_x = (
            make_fixed_goal(
                q_home,
                dq_home,
                0,
            )
        )

        q_goal_y, pred_y = (
            make_fixed_goal(
                q_home,
                dq_home,
                1,
            )
        )

        x_joint_delta = (
            q_goal_x - q_home
        )

        y_joint_delta = (
            q_goal_y - q_home
        )

        print()
        print(
            "Fixed +X goal max joint delta = "
            f"{np.max(np.abs(x_joint_delta)):.6f} rad",
            flush=True,
        )

        print(
            "Predicted +X EE delta mm =",
            np.round(
                pred_x * 1000.0,
                4,
            ).tolist(),
            flush=True,
        )

        print(
            "Fixed +Y goal max joint delta = "
            f"{np.max(np.abs(y_joint_delta)):.6f} rad",
            flush=True,
        )

        print(
            "Predicted +Y EE delta mm =",
            np.round(
                pred_y * 1000.0,
                4,
            ).tolist(),
            flush=True,
        )

        if (
            np.max(
                np.abs(
                    x_joint_delta
                )
            )
            > MAX_GOAL_JOINT_DELTA_RAD
        ):
            raise RuntimeError(
                "X fixed IK goal exceeds joint-delta guard."
            )

        if (
            np.max(
                np.abs(
                    y_joint_delta
                )
            )
            > MAX_GOAL_JOINT_DELTA_RAD
        ):
            raise RuntimeError(
                "Y fixed IK goal exceeds joint-delta guard."
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
        # RGB
        # --------------------------------------------------
        video = vs.VideoClient()
        video.SetTimeout(3.0)
        video.Init()

        print()
        print(
            "WARNING: REAL R1-A7 rt/lowcmd.",
            flush=True,
        )

        print(
            "Calibration sequence:",
            flush=True,
        )

        print(
            "  +X 5 mm command x3, return each time",
            flush=True,
        )

        print(
            "  +Y 5 mm command x3, return each time",
            flush=True,
        )

        print(
            "Actual FK displacement, NOT commanded "
            "5 mm, is used for J_img.",
            flush=True,
        )

        print(
            "Keep emergency stop ready.",
            flush=True,
        )

        answer = input(
            "Type ENABLE to take over at current pose: "
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

        print()
        print(
            "[ACTIVE] Current posture held.",
            flush=True,
        )

        print(
            "R = select rigid tweezer ROI",
            flush=True,
        )

        print(
            "J = start Xx3 + Yx3 calibration",
            flush=True,
        )

        print(
            "Q / ESC = stop",
            flush=True,
        )

        # --------------------------------------------------
        # Runtime state
        # --------------------------------------------------
        cv2.namedWindow(
            WINDOW,
            cv2.WINDOW_NORMAL,
        )

        prev_gray = None
        points = None
        roi = None
        feature_count = 0
        track_fail_count = 0

        image_crc_last = None

        phase = "IDLE"
        phase_started = 0.0

        axis = "X"
        repeat_index = 0

        sample_start_pos = None
        accum = np.zeros(
            2,
            dtype=float,
        )

        x_img_samples = []
        x_robot_samples = []

        y_img_samples = []
        y_robot_samples = []

        def redetect(gray):
            nonlocal points
            nonlocal feature_count
            nonlocal track_fail_count
            nonlocal prev_gray

            if roi is None:
                return False

            candidate = detect_features(
                gray,
                roi,
            )

            if (
                candidate is None
                or len(candidate)
                < MIN_START_FEATURES
            ):
                feature_count = (
                    0
                    if candidate is None
                    else len(candidate)
                )

                points = None

                return False

            points = candidate
            feature_count = len(points)
            track_fail_count = 0
            prev_gray = gray.copy()

            return True

        def begin_outbound(
            this_axis,
            gray,
            actual_right,
        ):
            nonlocal phase
            nonlocal phase_started
            nonlocal sample_start_pos
            nonlocal accum
            nonlocal axis

            axis = this_axis

            if not redetect(gray):
                raise RuntimeError(
                    "Cannot redetect enough rigid ROI "
                    "features before motion."
                )

            accum[:] = 0.0

            sample_start_pos = (
                actual_right[
                    :3, 3
                ].copy()
            )

            if axis == "X":
                output.set_arm_goal(
                    q_goal_x,
                    mode_machine,
                )
            else:
                output.set_arm_goal(
                    q_goal_y,
                    mode_machine,
                )

            phase = f"{axis}_OUT"
            phase_started = time.monotonic()

            print(
                f"[CALIB] {axis} repeat "
                f"{repeat_index + 1}/{CALIB_REPEATS} START",
                flush=True,
            )

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

            _, actual_right = (
                vs.get_ee_poses(
                    ik_fk,
                    q_now,
                )
            )

            # ----------------------------------------------
            # New RGB frame
            # ----------------------------------------------
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

            # ----------------------------------------------
            # Multi-feature LK
            # ----------------------------------------------
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
                    track_fail_count += 1

                    print(
                        "[TRACK] transient failure "
                        f"{track_fail_count}/"
                        f"{MAX_TRACK_FAILS} "
                        f"(good={n_good})",
                        flush=True,
                    )

                    if (
                        track_fail_count
                        >= MAX_TRACK_FAILS
                    ):
                        if phase != "IDLE":
                            print(
                                "[SAFETY] Tracking lost during "
                                "calibration. Returning home.",
                                flush=True,
                            )

                            output.set_arm_goal(
                                q_home,
                                current_mode_machine,
                            )

                            phase = "ABORT_RETURN"
                            phase_started = now

                        points = None
                        feature_count = 0
                        track_fail_count = 0

                else:
                    points = new_points
                    feature_count = n_good
                    track_fail_count = 0

                    if phase in (
                        "X_OUT",
                        "Y_OUT",
                    ):
                        accum += delta

                    prev_gray = gray.copy()

            elif points is None:
                prev_gray = gray.copy()

            # ----------------------------------------------
            # Calibration state machine
            # ----------------------------------------------
            if phase in (
                "X_OUT",
                "Y_OUT",
            ):
                if (
                    now - phase_started
                    >= MOVE_HOLD_S
                ):
                    d_img = accum.copy()

                    d_robot = (
                        actual_right[
                            :2, 3
                        ]
                        - sample_start_pos[:2]
                    )

                    print(
                        f"[CALIB {axis} "
                        f"{repeat_index + 1}] "
                        "image px =",
                        np.round(
                            d_img,
                            5,
                        ).tolist(),
                        " actual XY mm =",
                        np.round(
                            d_robot * 1000.0,
                            4,
                        ).tolist(),
                        flush=True,
                    )

                    if axis == "X":
                        x_img_samples.append(
                            d_img.copy()
                        )

                        x_robot_samples.append(
                            d_robot.copy()
                        )

                    else:
                        y_img_samples.append(
                            d_img.copy()
                        )

                        y_robot_samples.append(
                            d_robot.copy()
                        )

                    output.set_arm_goal(
                        q_home,
                        current_mode_machine,
                    )

                    phase = f"{axis}_RETURN"
                    phase_started = now

            elif phase in (
                "X_RETURN",
                "Y_RETURN",
            ):
                if (
                    now - phase_started
                    >= RETURN_HOLD_S
                ):
                    home_error = (
                        actual_right[
                            :3, 3
                        ]
                        - right_home[
                            :3, 3
                        ]
                    )

                    print(
                        f"[RETURN {axis} "
                        f"{repeat_index + 1}] "
                        "EE home error mm =",
                        np.round(
                            home_error
                            * 1000.0,
                            4,
                        ).tolist(),
                        flush=True,
                    )

                    repeat_index += 1

                    if (
                        repeat_index
                        < CALIB_REPEATS
                    ):
                        begin_outbound(
                            axis,
                            gray,
                            actual_right,
                        )

                    else:
                        if axis == "X":
                            axis = "Y"
                            repeat_index = 0

                            begin_outbound(
                                "Y",
                                gray,
                                actual_right,
                            )

                        else:
                            # ------------------------------
                            # Finished all six moves
                            # ------------------------------
                            phase = "IDLE"

                            x_img = np.asarray(
                                x_img_samples,
                                dtype=float,
                            )

                            x_robot = np.asarray(
                                x_robot_samples,
                                dtype=float,
                            )

                            y_img = np.asarray(
                                y_img_samples,
                                dtype=float,
                            )

                            y_robot = np.asarray(
                                y_robot_samples,
                                dtype=float,
                            )

                            x_img_med = np.median(
                                x_img,
                                axis=0,
                            )

                            x_robot_med = np.median(
                                x_robot,
                                axis=0,
                            )

                            y_img_med = np.median(
                                y_img,
                                axis=0,
                            )

                            y_robot_med = np.median(
                                y_robot,
                                axis=0,
                            )

                            print()
                            print(
                                "=============================="
                                "================",
                                flush=True,
                            )

                            print(
                                "[RESULT] X image samples px:",
                                flush=True,
                            )
                            print(
                                np.round(
                                    x_img,
                                    5,
                                ),
                                flush=True,
                            )

                            print(
                                "[RESULT] X robot samples mm:",
                                flush=True,
                            )
                            print(
                                np.round(
                                    x_robot
                                    * 1000.0,
                                    4,
                                ),
                                flush=True,
                            )

                            print(
                                "[RESULT] Y image samples px:",
                                flush=True,
                            )
                            print(
                                np.round(
                                    y_img,
                                    5,
                                ),
                                flush=True,
                            )

                            print(
                                "[RESULT] Y robot samples mm:",
                                flush=True,
                            )
                            print(
                                np.round(
                                    y_robot
                                    * 1000.0,
                                    4,
                                ),
                                flush=True,
                            )

                            print()
                            print(
                                "[RESULT] median X image px =",
                                np.round(
                                    x_img_med,
                                    5,
                                ).tolist(),
                                flush=True,
                            )

                            print(
                                "[RESULT] median X robot mm =",
                                np.round(
                                    x_robot_med
                                    * 1000.0,
                                    4,
                                ).tolist(),
                                flush=True,
                            )

                            print(
                                "[RESULT] median Y image px =",
                                np.round(
                                    y_img_med,
                                    5,
                                ).tolist(),
                                flush=True,
                            )

                            print(
                                "[RESULT] median Y robot mm =",
                                np.round(
                                    y_robot_med
                                    * 1000.0,
                                    4,
                                ).tolist(),
                                flush=True,
                            )

                            x_motion = float(
                                np.linalg.norm(
                                    x_robot_med
                                )
                            )

                            y_motion = float(
                                np.linalg.norm(
                                    y_robot_med
                                )
                            )

                            x_image_motion = float(
                                np.linalg.norm(
                                    x_img_med
                                )
                            )

                            y_image_motion = float(
                                np.linalg.norm(
                                    y_img_med
                                )
                            )

                            success = True

                            if (
                                x_motion
                                < MIN_ACTUAL_EXCURSION_M
                                or y_motion
                                < MIN_ACTUAL_EXCURSION_M
                            ):
                                success = False

                                print(
                                    "[RESULT] FAILED: actual "
                                    "robot displacement too small.",
                                    flush=True,
                                )

                            if (
                                x_image_motion
                                < MIN_IMAGE_MOTION_PX
                                or y_image_motion
                                < MIN_IMAGE_MOTION_PX
                            ):
                                success = False

                                print(
                                    "[RESULT] FAILED: image "
                                    "displacement too small.",
                                    flush=True,
                                )

                            D_robot = np.column_stack(
                                (
                                    x_robot_med,
                                    y_robot_med,
                                )
                            )

                            D_img = np.column_stack(
                                (
                                    x_img_med,
                                    y_img_med,
                                )
                            )

                            det_robot = float(
                                np.linalg.det(
                                    D_robot
                                )
                            )

                            print(
                                f"[RESULT] det(D_robot) = "
                                f"{det_robot:.9e}",
                                flush=True,
                            )

                            if abs(det_robot) < 1e-9:
                                success = False

                                print(
                                    "[RESULT] FAILED: "
                                    "robot calibration directions "
                                    "are singular.",
                                    flush=True,
                                )

                            if success:
                                J = (
                                    D_img
                                    @ np.linalg.inv(
                                        D_robot
                                    )
                                )

                                condition = float(
                                    np.linalg.cond(
                                        J
                                    )
                                )

                                print()
                                print(
                                    "[RESULT] J_img pixel/m:",
                                    flush=True,
                                )

                                print(
                                    J,
                                    flush=True,
                                )

                                print()
                                print(
                                    "[RESULT] J_img pixel/mm:",
                                    flush=True,
                                )

                                print(
                                    J / 1000.0,
                                    flush=True,
                                )

                                print(
                                    "[RESULT] condition = "
                                    f"{condition:.3f}",
                                    flush=True,
                                )

                                if (
                                    not np.isfinite(
                                        condition
                                    )
                                    or condition
                                    > MAX_JACOBIAN_CONDITION
                                ):
                                    success = False

                                    print(
                                        "[RESULT] FAILED: "
                                        "ill-conditioned J_img.",
                                        flush=True,
                                    )

                            if success:
                                result = {
                                    "calib_command_step_mm":
                                        CALIB_STEP_M
                                        * 1000.0,
                                    "repeats":
                                        CALIB_REPEATS,
                                    "x_image_samples_px":
                                        x_img.tolist(),
                                    "x_robot_samples_m":
                                        x_robot.tolist(),
                                    "y_image_samples_px":
                                        y_img.tolist(),
                                    "y_robot_samples_m":
                                        y_robot.tolist(),
                                    "x_image_median_px":
                                        x_img_med.tolist(),
                                    "x_robot_median_m":
                                        x_robot_med.tolist(),
                                    "y_image_median_px":
                                        y_img_med.tolist(),
                                    "y_robot_median_m":
                                        y_robot_med.tolist(),
                                    "J_pixel_per_meter":
                                        J.tolist(),
                                    "J_pixel_per_mm":
                                        (
                                            J / 1000.0
                                        ).tolist(),
                                    "condition":
                                        condition,
                                    "roi":
                                        list(roi),
                                }

                                SAVE_PATH.write_text(
                                    json.dumps(
                                        result,
                                        indent=2,
                                    )
                                )

                                print()
                                print(
                                    "[RESULT] SUCCESS",
                                    flush=True,
                                )

                                print(
                                    "[RESULT] saved:",
                                    SAVE_PATH,
                                    flush=True,
                                )

                            print(
                                "=============================="
                                "================",
                                flush=True,
                            )

            elif phase == "ABORT_RETURN":
                if (
                    now - phase_started
                    >= RETURN_HOLD_S
                ):
                    phase = "IDLE"

                    print(
                        "[CALIB] aborted and returned.",
                        flush=True,
                    )

            # ----------------------------------------------
            # Display
            # ----------------------------------------------
            vis = frame.copy()

            if roi is not None:
                x, y, w, h = roi

                cv2.rectangle(
                    vis,
                    (x, y),
                    (x+w, y+h),
                    (255, 255, 0),
                    2,
                )

            if points is not None:
                for point in points.reshape(
                    -1,
                    2,
                ):
                    cv2.circle(
                        vis,
                        (
                            int(round(point[0])),
                            int(round(point[1])),
                        ),
                        3,
                        (0, 255, 0),
                        -1,
                    )

            cv2.putText(
                vis,
                (
                    f"phase={phase} "
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
                    f"accum="
                    f"({accum[0]:+.4f},"
                    f"{accum[1]:+.4f}) px"
                ),
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                "R: ROI  J: Xx3+Yx3  Q/ESC: stop",
                (15, vis.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
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

            if key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                break

            elif key in (
                ord("r"),
                ord("R"),
            ):
                if phase != "IDLE":
                    print(
                        "[ROI] Cannot select ROI "
                        "during calibration.",
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

                x, y, w, h = [
                    int(v)
                    for v in selected
                ]

                if w > 0 and h > 0:
                    roi = (
                        x,
                        y,
                        w,
                        h,
                    )

                    ok = redetect(
                        gray
                    )

                    print(
                        "[ROI]",
                        roi,
                        "features=",
                        feature_count,
                        flush=True,
                    )

                    if not ok:
                        print(
                            "[ROI] Need >= "
                            f"{MIN_START_FEATURES} features.",
                            flush=True,
                        )

            elif key in (
                ord("j"),
                ord("J"),
            ):
                if phase != "IDLE":
                    print(
                        "[CALIB] already running.",
                        flush=True,
                    )
                    continue

                if (
                    roi is None
                    or points is None
                    or feature_count
                    < MIN_START_FEATURES
                ):
                    print(
                        "[CALIB] Need >= "
                        f"{MIN_START_FEATURES} "
                        "valid rigid features.",
                        flush=True,
                    )

                    continue

                confirm = input(
                    "Type CALIBRATE to execute "
                    "Xx3 + Yx3 real robot calibration: "
                ).strip()

                if (
                    confirm.casefold()
                    != "calibrate"
                ):
                    print(
                        "[CALIB] cancelled.",
                        flush=True,
                    )

                    continue

                # Refresh exact current home.
                state = state_buffer.snapshot()

                (
                    _upper_q,
                    q_now,
                    _dq_now,
                    _gripper_q,
                    _gripper_dq,
                    current_mode_machine,
                    _received_at,
                ) = state

                q_now = np.asarray(
                    q_now,
                    dtype=float,
                )

                # For this experiment the fixed joint goals
                # were computed from q_home at ENABLE.
                # Ensure robot is still close to that home.
                home_q_error = float(
                    np.max(
                        np.abs(
                            q_now - q_home
                        )
                    )
                )

                if home_q_error > 0.003:
                    print(
                        "[CALIB] Refusing start: current "
                        "arm moved too far from calibration home. "
                        f"max error={home_q_error:.6f} rad",
                        flush=True,
                    )

                    continue

                x_img_samples.clear()
                x_robot_samples.clear()
                y_img_samples.clear()
                y_robot_samples.clear()

                repeat_index = 0
                axis = "X"

                print()
                print(
                    "[CALIB] START Xx3 + Yx3",
                    flush=True,
                )

                begin_outbound(
                    "X",
                    gray,
                    actual_right,
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
