#!/usr/bin/env python3

import json
import signal
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

import r1a7_head_visual_servo_lowcmd as vs


WINDOW = "R1-A7 Click Target Visual Move"
ZOOM_WINDOW = "R1-A7 Local Zoom x15"

CALIB_PATH = Path(
    "/home/robot/unitree_sim_isaaclab_threading/"
    "r1a7_threading_project/vision/"
    "r1a7_head_local_jacobian_repeat.json"
)

MAX_CORNERS = 40
MIN_FEATURES = 6
MIN_START_FEATURES = 8
MAX_TRACK_FAILS = 3

# One press of M generates at most this much MODEL Cartesian motion.
MAX_MODEL_STEP_MM = 12.0

# Coarse visual demonstration:
# use image error only to determine DIRECTION.
# Every M sends a fixed-size Cartesian model command.
COARSE_MODEL_STEP_MM = 12.0

# Once this close in the image, stop coarse motion.
# Fine visual servo will handle the final part later.
COARSE_STOP_ERROR_PX = 2.0

# Changes smaller than this are not considered convincing progress.
MEANINGFUL_ERROR_DROP_PX = 0.05

# If error grows by more than this after one move, block further motion.
ERROR_INCREASE_STOP_PX = 0.05


# If the requested command is smaller than this, the current coarse
# experiment is too close to the R1-A7 small-signal dead zone.
MIN_MODEL_STEP_MM = 3.0

MOVE_HOLD_S = 2.0

# Local experiment only.
MAX_IMAGE_TARGET_DISTANCE_PX = 20.0

# For this demo the TARGET means target CENTER.
# The size of the purple rectangle is only a visual aid.
TARGET_CENTER_DEADBAND_PX = 0.20

MAX_EE_FROM_HOME_M = 0.030
MAX_IK_JOINT_DELTA_RAD = 0.020

ZOOM_HALF_SIZE = 20
ZOOM_SCALE = 15


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

    pb, back_status, _ = cv2.calcOpticalFlowPyrLK(
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

    if pb is None or back_status is None:
        return None, None, 0

    p0 = prev_pts.reshape(-1, 2)
    p1f = p1.reshape(-1, 2)
    pbf = pb.reshape(-1, 2)

    good = (
        status.reshape(-1).astype(bool)
        & back_status.reshape(-1).astype(bool)
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


def point_in_rect(pt, rect):
    if pt is None or rect is None:
        return False

    x, y, w, h = rect
    u, v = pt

    return (
        x <= u <= x + w
        and y <= v <= y + h
    )


def make_zoom(vis, center):
    if center is None:
        return None

    h, w = vis.shape[:2]

    cx = int(round(center[0]))
    cy = int(round(center[1]))

    x0 = max(0, cx - ZOOM_HALF_SIZE)
    x1 = min(w, cx + ZOOM_HALF_SIZE)

    y0 = max(0, cy - ZOOM_HALF_SIZE)
    y1 = min(h, cy + ZOOM_HALF_SIZE)

    crop = vis[y0:y1, x0:x1]

    if crop.size == 0:
        return None

    return cv2.resize(
        crop,
        None,
        fx=ZOOM_SCALE,
        fy=ZOOM_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )


def main():
    if not CALIB_PATH.exists():
        raise RuntimeError(
            f"Calibration file not found: {CALIB_PATH}"
        )

    calib = json.loads(
        CALIB_PATH.read_text()
    )

    step_mm = float(
        calib["calib_command_step_mm"]
    )

    x_img = np.asarray(
        calib["x_image_median_px"],
        dtype=float,
    )

    y_img = np.asarray(
        calib["y_image_median_px"],
        dtype=float,
    )

    # ------------------------------------------------------
    # Direct mapping:
    #
    # [du,dv] = H_cmd * [dXcmd,dYcmd]   (model command mm)
    #
    # This deliberately includes the measured R1-A7
    # under-response of the LowCmd position controller.
    # ------------------------------------------------------
    H_cmd = np.column_stack(
        (
            x_img / step_mm,
            y_img / step_mm,
        )
    )

    H_inv = np.linalg.inv(
        H_cmd
    )

    print(
        "Loaded direct command-to-image mapping "
        "(px / model-mm):"
    )
    print(H_cmd)

    print(
        "condition =",
        float(np.linalg.cond(H_cmd)),
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
            received_at,
        ) = state

        q_home = np.asarray(
            arm_q,
            dtype=float,
        ).copy()

        vs.base.check_debug_mode(
            vs.MotionSwitcherClient
        )

        ik_fk = vs.R1A7_ArmIK(
            Unit_Test=False,
            Visualization=False,
        )

        ik_fk.reset_target_calibration(
            q_home
        )

        _, right_home = vs.get_ee_poses(
            ik_fk,
            q_home,
        )

        print(
            "Home right EE xyz:",
            np.round(
                right_home[:3, 3],
                6,
            ).tolist(),
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
            "REAL ROBOT CLICK-TO-TARGET DEMO"
        )
        print(
            "R : select rigid tweezer ROI"
        )
        print(
            "T : select target region"
        )
        print(
            "M : execute ONE visual-guided move"
        )
        print(
            "H : return to experiment home"
        )
        print(
            "Q/ESC : stop"
        )
        print()
        print(
            "Each M is one bounded move only."
        )
        print(
            "Keep emergency stop ready."
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

        cv2.namedWindow(
            ZOOM_WINDOW,
            cv2.WINDOW_NORMAL,
        )

        prev_gray = None
        points = None

        rigid_roi = None
        rigid_roi_size = None

        current_center = None

        target_roi = None
        target_center = None

        feature_count = 0
        fail_count = 0

        image_crc_last = None

        phase = "IDLE"
        phase_started = 0.0

        error_before = None
        step_start_right = None
        current_goal_q = q_home.copy()

        movement_allowed = True

        while not stopped:
            now = time.monotonic()

            # ----------------------------------------------
            # Robot state
            # ----------------------------------------------
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

            _, actual_right = vs.get_ee_poses(
                ik_fk,
                q_now,
            )

            total_from_home = float(
                np.linalg.norm(
                    actual_right[:3, 3]
                    - right_home[:3, 3]
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
            # Multi-point rigid ROI tracking
            # ----------------------------------------------
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
                        f"[TRACK] transient failure "
                        f"{fail_count}/{MAX_TRACK_FAILS} "
                        f"(good={n_good})",
                        flush=True,
                    )

                    if (
                        fail_count
                        >= MAX_TRACK_FAILS
                    ):
                        print(
                            "[SAFETY] rigid ROI tracking lost.",
                            flush=True,
                        )

                        output.set_arm_goal(
                            q_now,
                            current_mode_machine,
                        )

                        points = None
                        feature_count = 0
                        phase = "IDLE"
                        movement_allowed = False

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

            # ----------------------------------------------
            # End one commanded move
            # ----------------------------------------------
            if (
                phase == "MOVE"
                and now - phase_started
                >= MOVE_HOLD_S
            ):
                phase = "IDLE"

                if (
                    current_center is not None
                    and target_center is not None
                ):
                    error_after = (
                        target_center
                        - current_center
                    )

                    before_norm = float(
                        np.linalg.norm(
                            error_before
                        )
                    )

                    after_norm = float(
                        np.linalg.norm(
                            error_after
                        )
                    )

                    ee_step = (
                        actual_right[:3, 3]
                        - step_start_right[:3, 3]
                    )

                    print()
                    print(
                        "[MOVE RESULT]",
                        flush=True,
                    )

                    print(
                        "  image error before =",
                        np.round(
                            error_before,
                            4,
                        ).tolist(),
                        f"|e|={before_norm:.4f}px",
                        flush=True,
                    )

                    print(
                        "  image error after  =",
                        np.round(
                            error_after,
                            4,
                        ).tolist(),
                        f"|e|={after_norm:.4f}px",
                        flush=True,
                    )

                    print(
                        "  actual EE step mm  =",
                        np.round(
                            ee_step * 1000.0,
                            4,
                        ).tolist(),
                        flush=True,
                    )

                    error_drop = (
                        before_norm
                        - after_norm
                    )

                    print(
                        "  error drop = "
                        f"{error_drop:+.4f} px",
                        flush=True,
                    )

                    if (
                        error_drop
                        < -ERROR_INCREASE_STOP_PX
                    ):
                        movement_allowed = False

                        print(
                            "[STOP] Image error increased "
                            "significantly.",
                            flush=True,
                        )

                        print(
                            "Further M moves are blocked. "
                            "Press H to return home or T to "
                            "choose a new target.",
                            flush=True,
                        )

                    elif (
                        error_drop
                        >= MEANINGFUL_ERROR_DROP_PX
                    ):
                        print(
                            "[OK] Meaningful motion toward target.",
                            flush=True,
                        )

                    else:
                        print(
                            "[WARN] Image change is small / "
                            "near the tracking-noise range.",
                            flush=True,
                        )

            # ----------------------------------------------
            # Display
            # ----------------------------------------------
            vis = frame.copy()

            if (
                current_center is not None
                and rigid_roi_size is not None
            ):
                rw, rh = rigid_roi_size

                x0 = int(
                    round(
                        current_center[0]
                        - rw / 2
                    )
                )
                y0 = int(
                    round(
                        current_center[1]
                        - rh / 2
                    )
                )

                x1 = int(
                    round(
                        current_center[0]
                        + rw / 2
                    )
                )
                y1 = int(
                    round(
                        current_center[1]
                        + rh / 2
                    )
                )

                cv2.rectangle(
                    vis,
                    (x0, y0),
                    (x1, y1),
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
                    (tx+tw, ty+th),
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
                    tipLength=0.2,
                )

                error = (
                    target_center
                    - current_center
                )

                cv2.putText(
                    vis,
                    (
                        f"image error="
                        f"({error[0]:+.3f},"
                        f"{error[1]:+.3f}) px "
                        f"|e|={np.linalg.norm(error):.3f}"
                    ),
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
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
                        2,
                        (0, 255, 0),
                        -1,
                    )

            cv2.putText(
                vis,
                (
                    f"phase={phase} "
                    f"features={feature_count} "
                    f"EEhome={total_from_home*1000:.2f}mm"
                ),
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                "R: rigid ROI  T: target ROI  M: one move  H: home",
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

            zoom_center = (
                target_center
                if target_center is not None
                else current_center
            )

            zoom = make_zoom(
                vis,
                zoom_center,
            )

            if zoom is not None:
                cv2.imshow(
                    ZOOM_WINDOW,
                    zoom,
                )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                break

            # ----------------------------------------------
            # R: rigid tweezer ROI
            # ----------------------------------------------
            elif key in (
                ord("r"),
                ord("R"),
            ):
                if phase != "IDLE":
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
                    rigid_roi = (
                        x, y, w, h
                    )

                    rigid_roi_size = (
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
                        rigid_roi,
                    )

                    feature_count = (
                        0
                        if points is None
                        else len(points)
                    )

                    prev_gray = gray.copy()
                    fail_count = 0
                    movement_allowed = True

                    print(
                        "[ROI]",
                        rigid_roi,
                        "features=",
                        feature_count,
                        "center=",
                        np.round(
                            current_center,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    if (
                        feature_count
                        < MIN_START_FEATURES
                    ):
                        print(
                            f"[ROI] Need >= "
                            f"{MIN_START_FEATURES} features.",
                            flush=True,
                        )

            # ----------------------------------------------
            # T: target region
            # ----------------------------------------------
            elif key in (
                ord("t"),
                ord("T"),
            ):
                if (
                    phase != "IDLE"
                    or current_center is None
                    or points is None
                    or feature_count
                    < MIN_START_FEATURES
                ):
                    print(
                        "[TARGET] Select a stable rigid ROI first.",
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

                if tw > 0 and th > 0:
                    candidate_center = np.asarray(
                        [
                            tx + tw / 2.0,
                            ty + th / 2.0,
                        ],
                        dtype=float,
                    )

                    e = (
                        candidate_center
                        - current_center
                    )

                    distance = float(
                        np.linalg.norm(e)
                    )

                    if (
                        distance
                        > MAX_IMAGE_TARGET_DISTANCE_PX
                    ):
                        print(
                            "[TARGET] Too far for this LOCAL demo: "
                            f"{distance:.3f}px > "
                            f"{MAX_IMAGE_TARGET_DISTANCE_PX:.1f}px",
                            flush=True,
                        )

                        print(
                            "Choose a target closer to the current ROI.",
                            flush=True,
                        )

                        target_roi = None
                        target_center = None

                    else:
                        target_roi = (
                            tx, ty, tw, th
                        )

                        target_center = (
                            candidate_center
                        )

                        movement_allowed = True

                        print(
                            "[TARGET]",
                            target_roi,
                            "center=",
                            np.round(
                                target_center,
                                3,
                            ).tolist(),
                            "error px=",
                            np.round(
                                e,
                                3,
                            ).tolist(),
                            f"|e|={distance:.3f}",
                            flush=True,
                        )

            # ----------------------------------------------
            # H: return experiment home
            # ----------------------------------------------
            elif key in (
                ord("h"),
                ord("H"),
            ):
                output.set_arm_goal(
                    q_home,
                    current_mode_machine,
                )

                current_goal_q = (
                    q_home.copy()
                )

                print(
                    "[HOME] Returning to startup arm q.",
                    flush=True,
                )

            # ----------------------------------------------
            # M: ONE visual-guided macro step
            # ----------------------------------------------
            elif key in (
                ord("m"),
                ord("M"),
            ):
                if phase != "IDLE":
                    print(
                        "[MOVE] Previous move still active.",
                        flush=True,
                    )
                    continue

                if not movement_allowed:
                    print(
                        "[MOVE] Blocked after previous bad step. "
                        "Press T or H.",
                        flush=True,
                    )
                    continue

                if (
                    current_center is None
                    or target_center is None
                    or points is None
                    or feature_count
                    < MIN_START_FEATURES
                ):
                    print(
                        "[MOVE] Need valid R and T first.",
                        flush=True,
                    )
                    continue

                center_error = (
                    target_center
                    - current_center
                )

                center_error_norm = float(
                    np.linalg.norm(
                        center_error
                    )
                )

                if (
                    center_error_norm
                    <= TARGET_CENTER_DEADBAND_PX
                ):
                    print(
                        "[TARGET] ROI center already reached "
                        "target center: "
                        f"|e|={center_error_norm:.3f}px",
                        flush=True,
                    )
                    continue

                if (
                    total_from_home
                    > MAX_EE_FROM_HOME_M
                ):
                    print(
                        "[SAFETY] Too far from experiment home. "
                        "Press H.",
                        flush=True,
                    )
                    continue

                error_px = (
                    target_center
                    - current_center
                )

                # --------------------------------------------------
                # COARSE MODE
                #
                # H_inv is used ONLY to estimate Cartesian direction.
                # We deliberately do NOT ask it to predict the whole
                # distance to a far target because the calibration is
                # local and the real R1-A7 response is pose-dependent.
                #
                # Every M therefore sends one fixed 12-mm model step.
                # --------------------------------------------------
                image_error_norm = float(
                    np.linalg.norm(
                        error_px
                    )
                )

                if (
                    image_error_norm
                    <= COARSE_STOP_ERROR_PX
                ):
                    print(
                        "[MOVE] Target is already inside coarse "
                        "deadband: "
                        f"|e|={image_error_norm:.3f}px <= "
                        f"{COARSE_STOP_ERROR_PX:.1f}px",
                        flush=True,
                    )
                    print(
                        "Use the later fine visual-servo mode "
                        "for final alignment.",
                        flush=True,
                    )
                    continue

                raw_direction_mm = (
                    H_inv
                    @ error_px
                )

                direction_norm = float(
                    np.linalg.norm(
                        raw_direction_mm
                    )
                )

                if (
                    not np.isfinite(
                        direction_norm
                    )
                    or direction_norm
                    < 1e-9
                ):
                    print(
                        "[MOVE] Invalid Cartesian direction.",
                        flush=True,
                    )
                    continue

                command_mm = (
                    raw_direction_mm
                    / direction_norm
                    * COARSE_MODEL_STEP_MM
                )

                # Fresh one-shot IK, matching the successful
                # 5-mm standalone experiments.
                ik_step = vs.R1A7_ArmIK(
                    Unit_Test=False,
                    Visualization=False,
                )

                ik_step.reset_target_calibration(
                    q_now
                )

                left_now, right_now = (
                    vs.get_ee_poses(
                        ik_step,
                        q_now,
                    )
                )

                right_target = (
                    right_now.copy()
                )

                right_target[
                    0, 3
                ] += command_mm[0] / 1000.0

                right_target[
                    1, 3
                ] += command_mm[1] / 1000.0

                q_goal, _ = ik_step.solve_ik(
                    left_now,
                    right_target,
                    q_now,
                    dq_now,
                    position_only=False,
                    max_joint_step=vs.IK_MAX_JOINT_STEP,
                    rotation_weight=vs.IK_ROTATION_WEIGHT,
                )

                q_goal = np.asarray(
                    q_goal,
                    dtype=float,
                ).reshape(14)

                # Do not let the left arm drift.
                q_goal[:7] = q_now[:7]

                max_joint_delta = float(
                    np.max(
                        np.abs(
                            q_goal - q_now
                        )
                    )
                )

                print()
                print(
                    "[MOVE PLAN]",
                    flush=True,
                )

                print(
                    "  image error px =",
                    np.round(
                        error_px,
                        4,
                    ).tolist(),
                    flush=True,
                )

                print(
                    "  fixed coarse XY command mm =",
                    np.round(
                        command_mm,
                        4,
                    ).tolist(),
                    flush=True,
                )

                print(
                    "  coarse command norm = "
                    f"{np.linalg.norm(command_mm):.3f} mm",
                    flush=True,
                )

                print(
                    "  max joint delta = "
                    f"{max_joint_delta:.6f} rad",
                    flush=True,
                )

                if (
                    max_joint_delta
                    > MAX_IK_JOINT_DELTA_RAD
                ):
                    print(
                        "[SAFETY] IK step rejected: "
                        "joint delta too large.",
                        flush=True,
                    )
                    continue

                error_before = (
                    error_px.copy()
                )

                step_start_right = (
                    actual_right.copy()
                )

                output.set_arm_goal(
                    q_goal,
                    current_mode_machine,
                )

                current_goal_q = (
                    q_goal.copy()
                )

                phase = "MOVE"
                phase_started = now

                print(
                    "[MOVE] ONE step sent. "
                    "Observe robot and zoom window.",
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
