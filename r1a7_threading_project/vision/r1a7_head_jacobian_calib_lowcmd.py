#!/usr/bin/env python3

import json
import signal
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

import r1a7_head_visual_servo_lowcmd as vs


WINDOW = "R1-A7 Visual Jacobian Calibration"

CALIB_STEP_M = 0.005           # 2 mm
SETTLE_POS_M = 0.0006          # 0.6 mm
SETTLE_TIME_S = 0.35
PHASE_TIMEOUT_S = 10.0

MIN_FEATURES = 6
MIN_START_FEATURES = 8
MAX_TRACK_FAILS = 3
MAX_CORNERS = 40

MIN_ACTUAL_EXCURSION_M = 0.00035
MIN_IMAGE_MOTION_PX = 0.03
MAX_CONDITION = 50.0

SAVE_PATH = Path(
    "/home/robot/unitree_sim_isaaclab_threading/"
    "r1a7_threading_project/vision/"
    "r1a7_head_local_jacobian.json"
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
    if prev_pts is None:
        return None, None, 0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_pts,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
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
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
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
        p1_good.reshape(-1, 1, 2).astype(np.float32),
        delta.astype(float),
        len(p1_good),
    )


def main():
    vs.base.validate_robot_interface(
        vs.INTERFACE
    )

    print(
        f"Robot DDS interface: {vs.INTERFACE}",
        flush=True,
    )

    guard = vs.acquire_lowcmd_guard(
        Path(__file__).name,
        topic=vs.COMMAND_TOPIC,
    )

    stopped = False

    def stop_handler(_sig=None, _frame=None):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

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
                "No valid rt/lowstate. No command sent."
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

        print(
            "Measured arm q:",
            np.round(arm_q, 4).tolist(),
            flush=True,
        )

        vs.base.check_debug_mode(
            vs.MotionSwitcherClient
        )

        # --------------------------------------------------
        # IK / initial FK
        # --------------------------------------------------
        ik = vs.R1A7_ArmIK(
            Unit_Test=False,
            Visualization=False,
        )

        ik.reset_target_calibration(
            arm_q
        )

        left_target, right_home = (
            vs.get_ee_poses(
                ik,
                arm_q,
            )
        )

        left_target = left_target.copy()
        right_home = right_home.copy()
        right_target = right_home.copy()

        first_solution, _ = ik.solve_ik(
            left_target,
            right_target,
            arm_q,
            arm_dq,
            max_joint_step=vs.IK_MAX_JOINT_STEP,
            rotation_weight=vs.IK_ROTATION_WEIGHT,
        )

        first_solution = np.asarray(
            first_solution,
            dtype=float,
        ).reshape(14)

        start_error = float(
            np.max(
                np.abs(
                    first_solution - arm_q
                )
            )
        )

        print(
            f"Startup IK difference: "
            f"{start_error:.6f} rad",
            flush=True,
        )

        if start_error > 0.08:
            raise RuntimeError(
                "Startup IK difference too large."
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
        # Head RGB
        # --------------------------------------------------
        video = vs.VideoClient()
        video.SetTimeout(3.0)
        video.Init()

        print()
        print(
            "REAL ROBOT VISUAL JACOBIAN TEST",
            flush=True,
        )
        print(
            "No VR. No visual servo. "
            "Only +/- local calibration motion.",
            flush=True,
        )
        print(
            "Keep emergency stop ready.",
            flush=True,
        )

        answer = input(
            "Type ENABLE to hold current posture: "
        ).strip()

        if answer.casefold() != "enable":
            print("Aborted. No LowCmd sent.")
            return 2

        output.enable(
            upper_q,
            mode_machine,
            gripper_q=gripper_q,
            gripper_initial_mode="current",
            right_gripper_open_cap_current=True,
        )

        time.sleep(0.1)

        output.set_arm_goal(
            first_solution,
            mode_machine,
        )

        print(
            "[ACTIVE] Current posture taken over.",
            flush=True,
        )
        print(
            "R = select rigid tweezer ROI",
            flush=True,
        )
        print(
            "J = start +X/+Y 2 mm calibration",
            flush=True,
        )
        print(
            "Q / ESC = stop",
            flush=True,
        )

        # --------------------------------------------------
        # Vision state
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

        accum = np.zeros(2, dtype=float)

        last_crc = None

        # Calibration state
        phase = "IDLE"
        phase_started = 0.0
        settle_since = None

        calib_home = None
        phase_start_pos = None

        d_img_x = None
        d_img_y = None
        d_pos_x = None
        d_pos_y = None

        while not stopped:
            now = time.monotonic()

            # ----------------------------------------------
            # State safety
            # ----------------------------------------------
            state = state_buffer.snapshot()

            if state is None:
                raise RuntimeError(
                    "LowState disappeared."
                )

            (
                _upper_q,
                arm_q,
                arm_dq,
                gripper_q,
                gripper_dq,
                mode_machine,
                received_at,
            ) = state

            state_age = now - received_at

            if state_age > vs.LOWSTATE_TIMEOUT:
                raise RuntimeError(
                    f"LowState stale: {state_age:.3f}s"
                )

            if output.error:
                raise RuntimeError(
                    output.error
                )

            _, actual_right = (
                vs.get_ee_poses(
                    ik,
                    arm_q,
                )
            )

            # ----------------------------------------------
            # Camera
            # ----------------------------------------------
            ret, data = video.GetImageSample()

            if (
                ret != 0
                or data is None
                or len(data) == 0
            ):
                continue

            raw = bytes(data)
            image_crc = zlib.crc32(raw)

            if image_crc == last_crc:
                continue

            last_crc = image_crc

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
            # Multi-point tracking
            # ----------------------------------------------
            if (
                prev_gray is not None
                and points is not None
            ):
                new_points, delta, n_good = (
                    track_features(
                        prev_gray,
                        gray,
                        points,
                    )
                )

                if new_points is None:
                    track_fail_count += 1

                    print(
                        f"[TRACK] transient failure "
                        f"{track_fail_count}/{MAX_TRACK_FAILS} "
                        f"(good={n_good})",
                        flush=True,
                    )

                    # IMPORTANT:
                    # Do NOT update prev_gray here.
                    # The next good frame is allowed to track
                    # from the last valid image.

                    if track_fail_count >= MAX_TRACK_FAILS:

                        if phase != "IDLE":
                            # During a real calibration move we must
                            # not silently reinitialize features,
                            # because that would corrupt accumulated
                            # image displacement.
                            print(
                                "[SAFETY] Tracking failed for "
                                f"{MAX_TRACK_FAILS} consecutive frames "
                                "during calibration.",
                                flush=True,
                            )

                            print(
                                "[SAFETY] Calibration aborted; "
                                "returning right arm to calibration home.",
                                flush=True,
                            )

                            phase = "ABORT_RETURN"
                            right_target = right_home.copy()

                            points = None
                            feature_count = 0
                            track_fail_count = 0

                        elif roi is not None:
                            # Robot is not moving: safe to redetect.
                            redetected = detect_features(
                                gray,
                                roi,
                            )

                            if (
                                redetected is not None
                                and len(redetected)
                                >= MIN_START_FEATURES
                            ):
                                points = redetected
                                feature_count = len(points)
                                track_fail_count = 0
                                prev_gray = gray.copy()

                                print(
                                    "[TRACK] automatically redetected "
                                    f"{feature_count} rigid features.",
                                    flush=True,
                                )

                            else:
                                n_redetect = (
                                    0
                                    if redetected is None
                                    else len(redetected)
                                )

                                points = None
                                feature_count = n_redetect
                                track_fail_count = 0

                                print(
                                    "[TRACK] redetection insufficient: "
                                    f"{n_redetect} features. "
                                    "Press R and choose the ROI again.",
                                    flush=True,
                                )

                else:
                    track_fail_count = 0

                    points = new_points
                    feature_count = n_good

                    if phase in ("X_OUT", "Y_OUT"):
                        accum += delta

                    # Only advance the tracker reference image
                    # after a VALID optical-flow update.
                    prev_gray = gray.copy()

            # ----------------------------------------------
            # Calibration
            # ----------------------------------------------
            if phase != "IDLE":

                pos_error = float(
                    np.linalg.norm(
                        actual_right[:3, 3]
                        - right_target[:3, 3]
                    )
                )

                if pos_error < SETTLE_POS_M:
                    if settle_since is None:
                        settle_since = now
                else:
                    settle_since = None

                settled = (
                    settle_since is not None
                    and now - settle_since
                    >= SETTLE_TIME_S
                )

                timed_out = (
                    now - phase_started
                    >= PHASE_TIMEOUT_S
                )

                if phase == "X_OUT":
                    vs.solve_and_send(
                        ik,
                        output,
                        left_target,
                        right_target,
                        arm_q,
                        arm_dq,
                        mode_machine,
                    )

                    if settled or timed_out:
                        d_img_x = accum.copy()

                        d_pos_x = (
                            actual_right[:2, 3]
                            - phase_start_pos[:2]
                        )

                        print(
                            "[CALIB X] image px =",
                            np.round(
                                d_img_x, 4
                            ).tolist(),
                            " actual XY m =",
                            np.round(
                                d_pos_x, 6
                            ).tolist(),
                            flush=True,
                        )

                        right_target = (
                            calib_home.copy()
                        )

                        phase = "X_RETURN"
                        phase_started = now
                        settle_since = None

                elif phase == "X_RETURN":
                    vs.solve_and_send(
                        ik,
                        output,
                        left_target,
                        right_target,
                        arm_q,
                        arm_dq,
                        mode_machine,
                    )

                    if settled or timed_out:
                        # Redetect fresh features at home.
                        points = detect_features(
                            gray,
                            roi,
                        )

                        if (
                            points is None
                            or len(points) < MIN_FEATURES
                        ):
                            print(
                                "[CALIB] Cannot redetect "
                                "enough features at home.",
                                flush=True,
                            )
                            phase = "IDLE"
                            continue

                        feature_count = len(points)

                        accum[:] = 0.0

                        phase_start_pos = (
                            actual_right[
                                :3, 3
                            ].copy()
                        )

                        right_target = (
                            calib_home.copy()
                        )
                        right_target[
                            1, 3
                        ] += CALIB_STEP_M

                        phase = "Y_OUT"
                        phase_started = now
                        settle_since = None

                        print(
                            "[CALIB] +Y 2.0 mm",
                            flush=True,
                        )

                elif phase == "Y_OUT":
                    vs.solve_and_send(
                        ik,
                        output,
                        left_target,
                        right_target,
                        arm_q,
                        arm_dq,
                        mode_machine,
                    )

                    if settled or timed_out:
                        d_img_y = accum.copy()

                        d_pos_y = (
                            actual_right[:2, 3]
                            - phase_start_pos[:2]
                        )

                        print(
                            "[CALIB Y] image px =",
                            np.round(
                                d_img_y, 4
                            ).tolist(),
                            " actual XY m =",
                            np.round(
                                d_pos_y, 6
                            ).tolist(),
                            flush=True,
                        )

                        right_target = (
                            calib_home.copy()
                        )

                        phase = "Y_RETURN"
                        phase_started = now
                        settle_since = None

                elif phase == "Y_RETURN":
                    vs.solve_and_send(
                        ik,
                        output,
                        left_target,
                        right_target,
                        arm_q,
                        arm_dq,
                        mode_machine,
                    )

                    if settled or timed_out:
                        x_exc = float(
                            np.linalg.norm(
                                d_pos_x
                            )
                        )

                        y_exc = float(
                            np.linalg.norm(
                                d_pos_y
                            )
                        )

                        x_img = float(
                            np.linalg.norm(
                                d_img_x
                            )
                        )

                        y_img = float(
                            np.linalg.norm(
                                d_img_y
                            )
                        )

                        print(
                            f"[CALIB] actual X excursion = "
                            f"{x_exc*1000:.3f} mm",
                            flush=True,
                        )

                        print(
                            f"[CALIB] actual Y excursion = "
                            f"{y_exc*1000:.3f} mm",
                            flush=True,
                        )

                        print(
                            f"[CALIB] image X motion = "
                            f"{x_img:.3f} px",
                            flush=True,
                        )

                        print(
                            f"[CALIB] image Y motion = "
                            f"{y_img:.3f} px",
                            flush=True,
                        )

                        success = True

                        if (
                            x_exc
                            < MIN_ACTUAL_EXCURSION_M
                            or y_exc
                            < MIN_ACTUAL_EXCURSION_M
                        ):
                            success = False

                            print(
                                "[CALIB] FAILED: robot did not "
                                "complete enough of the 2 mm move.",
                                flush=True,
                            )

                        if (
                            x_img < MIN_IMAGE_MOTION_PX
                            or y_img < MIN_IMAGE_MOTION_PX
                        ):
                            success = False

                            print(
                                "[CALIB] FAILED: image motion "
                                "too small.",
                                flush=True,
                            )

                        if success:
                            D_img = np.column_stack(
                                (d_img_x, d_img_y)
                            )

                            D_robot = np.column_stack(
                                (d_pos_x, d_pos_y)
                            )

                            det_robot = float(
                                np.linalg.det(
                                    D_robot
                                )
                            )

                            if abs(det_robot) < 1e-9:
                                success = False
                                print(
                                    "[CALIB] FAILED: robot "
                                    "motion matrix singular.",
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
                                np.linalg.cond(J)
                            )

                            print(
                                "[CALIB] J_img pixel/m:",
                                flush=True,
                            )
                            print(
                                J,
                                flush=True,
                            )

                            print(
                                "[CALIB] J_img pixel/mm:",
                                flush=True,
                            )
                            print(
                                J / 1000.0,
                                flush=True,
                            )

                            print(
                                f"[CALIB] condition = "
                                f"{condition:.3f}",
                                flush=True,
                            )

                            if (
                                not np.isfinite(condition)
                                or condition
                                > MAX_CONDITION
                            ):
                                success = False

                                print(
                                    "[CALIB] FAILED: "
                                    "ill-conditioned Jacobian.",
                                    flush=True,
                                )

                        if success:
                            result = {
                                "J_pixel_per_meter":
                                    J.tolist(),
                                "J_pixel_per_mm":
                                    (J / 1000.0).tolist(),
                                "condition":
                                    condition,
                                "calib_step_command_mm":
                                    CALIB_STEP_M * 1000.0,
                                "actual_x_excursion_mm":
                                    x_exc * 1000.0,
                                "actual_y_excursion_mm":
                                    y_exc * 1000.0,
                                "image_x_motion_px":
                                    x_img,
                                "image_y_motion_px":
                                    y_img,
                                "right_home_xyz_m":
                                    calib_home[
                                        :3, 3
                                    ].tolist(),
                                "roi":
                                    list(roi),
                                "features":
                                    int(feature_count),
                            }

                            SAVE_PATH.write_text(
                                json.dumps(
                                    result,
                                    indent=2,
                                )
                            )

                            print(
                                "[CALIB] SUCCESS",
                                flush=True,
                            )

                            print(
                                "[CALIB] saved:",
                                SAVE_PATH,
                                flush=True,
                            )

                        phase = "IDLE"
                        right_target = (
                            calib_home.copy()
                        )

                elif phase == "ABORT_RETURN":
                    vs.solve_and_send(
                        ik,
                        output,
                        left_target,
                        right_target,
                        arm_q,
                        arm_dq,
                        mode_machine,
                    )

                    if settled or timed_out:
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
                for p in points.reshape(-1, 2):
                    cv2.circle(
                        vis,
                        (
                            int(round(p[0])),
                            int(round(p[1])),
                        ),
                        3,
                        (0, 255, 0),
                        -1,
                    )

            cv2.putText(
                vis,
                f"phase={phase} features={feature_count}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                f"accum=({accum[0]:+.3f},"
                f"{accum[1]:+.3f}) px",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                "R: ROI  J: calibrate  Q/ESC: stop",
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

            key = cv2.waitKey(1) & 0xFF

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
                        "[ROI] Cannot change ROI "
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
                        x, y, w, h
                    )

                    points = detect_features(
                        gray,
                        roi,
                    )

                    if points is None:
                        feature_count = 0
                    else:
                        feature_count = len(points)

                    print(
                        "[ROI]",
                        roi,
                        "features=",
                        feature_count,
                        flush=True,
                    )

                    accum[:] = 0.0
                    track_fail_count = 0

                    # Synchronize tracker frame.
                    prev_gray = gray.copy()

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
                    or feature_count < MIN_START_FEATURES
                ):
                    print(
                        f"[CALIB] Need >= {MIN_START_FEATURES} valid "
                        "rigid ROI features first.",
                        flush=True,
                    )
                    continue

                confirm = input(
                    "Type CALIBRATE for real "
                    "+X/+Y 2 mm motion: "
                ).strip()

                if confirm.casefold() != "calibrate":
                    print(
                        "[CALIB] cancelled.",
                        flush=True,
                    )
                    continue

                _, current_right = (
                    vs.get_ee_poses(
                        ik,
                        arm_q,
                    )
                )

                calib_home = (
                    current_right.copy()
                )

                right_home = (
                    current_right.copy()
                )

                right_target = (
                    calib_home.copy()
                )

                phase_start_pos = (
                    current_right[
                        :3, 3
                    ].copy()
                )

                accum[:] = 0.0

                right_target[
                    0, 3
                ] += CALIB_STEP_M

                phase = "X_OUT"
                phase_started = now
                settle_since = None

                print(
                    "[CALIB] START",
                    flush=True,
                )

                print(
                    "[CALIB] +X 2.0 mm",
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
    raise SystemExit(main())
