#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Project paths
# ============================================================

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]

DATA_DIR = THIS_FILE.parent / "data"
IMAGE_DIR = DATA_DIR / "images"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Hand-eye constants
# ============================================================

RIGHT_WRIST_SN = "CPCBC53000C5"

STATE_TOPIC = "rt/lowstate"

COLOR_WIDTH = 1280
COLOR_HEIGHT = 800
COLOR_FPS = 30

# ChArUco:
# 6 x 8 squares
# square = 30 mm
# marker = 22 mm
CHARUCO_SQUARES_X = 6
CHARUCO_SQUARES_Y = 8
CHARUCO_SQUARE_LENGTH_M = 0.030
CHARUCO_MARKER_LENGTH_M = 0.022

MIN_CHARUCO_CORNERS = 20
MAX_RMSE_PX = 0.5
MAX_LOWSTATE_AGE_MS = 50.0


# ============================================================
# Verified Gemini 336L color intrinsic
# Color: 1280 x 800
#
# OpenCV distortion order:
# k1, k2, p1, p2, k3
# ============================================================

CAMERA_MATRIX = np.array(
    [
        [607.7092, 0.0, 640.3098],
        [0.0, 607.6644, 404.9614],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DIST_COEFFS = np.array(
    [
        -0.029733,
        0.033434,
        -0.000169,
        -0.000187,
        -0.011704,
    ],
    dtype=np.float64,
).reshape(-1, 1)


# ============================================================
# Load existing verified R1-A7 robot module
#
# We reuse:
#   - ChannelFactoryInitialize
#   - ChannelSubscriber
#   - LowState_
#   - CRC
#   - base.StateBuffer
#   - R1A7_ArmIK
#   - get_ee_poses()
#
# NO LowCmd publisher is created in this program.
# ============================================================

def load_robot_module():
    module_path = (
        REPO_ROOT
        / "r1a7_threading_project"
        / "vision"
        / "r1a7_head_visual_servo_lowcmd.py"
    )

    if not module_path.exists():
        raise FileNotFoundError(
            f"Robot module not found: {module_path}"
        )

    sys.path.insert(0, str(REPO_ROOT))

    spec = importlib.util.spec_from_file_location(
        "r1a7_existing_visual_servo",
        str(module_path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import robot module: {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


# ============================================================
# Orbbec helpers
# ============================================================

def normalize_serial(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def find_orbbec_device(target_serial):
    from pyorbbecsdk import Context

    context = Context()
    device_list = context.query_devices()
    count = device_list.get_count()

    print(f"[CAMERA] Orbbec device count: {count}")

    found_device = None

    for i in range(count):
        device = device_list.get_device_by_index(i)
        info = device.get_device_info()

        serial = normalize_serial(
            info.get_serial_number()
        )

        print(
            f"[CAMERA] Orbbec #{i}: SN={serial}"
        )

        if serial == target_serial:
            found_device = device

    if found_device is None:
        raise RuntimeError(
            "Right wrist Orbbec not found. "
            f"Expected SN={target_serial}"
        )

    print(
        "[CAMERA] Right wrist camera locked: "
        f"{target_serial}"
    )

    # Keep context alive together with device.
    return context, found_device


def color_frame_to_bgr(color_frame):
    width = int(color_frame.get_width())
    height = int(color_frame.get_height())

    data = color_frame.get_data()

    try:
        raw = np.frombuffer(
            data,
            dtype=np.uint8,
        )
    except TypeError:
        raw = np.asarray(
            data,
            dtype=np.uint8,
        ).reshape(-1)

    expected = width * height * 3

    if raw.size != expected:
        raise RuntimeError(
            "Unexpected RGB frame size: "
            f"got {raw.size}, expected {expected} "
            f"for {width}x{height}"
        )

    rgb = raw.reshape(
        height,
        width,
        3,
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    return bgr


# ============================================================
# ChArUco setup
# ============================================================


def create_charuco():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV has no cv2.aruco")

    aruco = cv2.aruco

    dictionary = aruco.getPredefinedDictionary(
        aruco.DICT_5X5_100
    )

    board = aruco.CharucoBoard(
        (
            CHARUCO_SQUARES_X,
            CHARUCO_SQUARES_Y,
        ),
        CHARUCO_SQUARE_LENGTH_M,
        CHARUCO_MARKER_LENGTH_M,
        dictionary,
    )

    detector_params = aruco.DetectorParameters()
    charuco_params = aruco.CharucoParameters()

    # Use verified 1280x800 wrist-camera calibration.
    charuco_params.cameraMatrix = CAMERA_MATRIX.copy()
    charuco_params.distCoeffs = DIST_COEFFS.copy()

    try:
        charuco_params.tryRefineMarkers = True
    except Exception:
        pass

    detector = aruco.CharucoDetector(
        board,
        charuco_params,
        detector_params,
    )

    return (
        aruco,
        dictionary,
        board,
        detector_params,
        detector,
    )


def detect_charuco_pose(
    image_bgr,
    aruco,
    dictionary,
    board,
    params,
    detector,
):
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    result = {
        "ok": False,
        "marker_corners": [],
        "marker_ids": None,
        "charuco_corners": None,
        "charuco_ids": None,
        "corner_count": 0,
        "rmse_px": None,
        "rvec": None,
        "tvec": None,
        "T_camera_board": None,
    }

    (
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
    ) = detector.detectBoard(gray)

    result["marker_corners"] = marker_corners
    result["marker_ids"] = marker_ids
    result["charuco_corners"] = charuco_corners
    result["charuco_ids"] = charuco_ids

    if (
        charuco_corners is None
        or charuco_ids is None
    ):
        return result

    corner_count = int(len(charuco_ids))
    result["corner_count"] = corner_count

    if corner_count < 4:
        return result

    # Convert ChArUco detections into matching
    # board 3D points and image 2D points.
    object_points, image_points = (
        board.matchImagePoints(
            charuco_corners,
            charuco_ids,
        )
    )

    object_points = np.asarray(
        object_points,
        dtype=np.float64,
    ).reshape(-1, 3)

    image_points = np.asarray(
        image_points,
        dtype=np.float64,
    ).reshape(-1, 2)

    if (
        len(object_points) < 4
        or len(image_points) < 4
    ):
        return result

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        CAMERA_MATRIX,
        DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok:
        return result

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        CAMERA_MATRIX,
        DIST_COEFFS,
    )

    projected = np.asarray(
        projected,
        dtype=np.float64,
    ).reshape(-1, 2)

    rmse_px = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (projected - image_points) ** 2,
                    axis=1,
                )
            )
        )
    )

    R_camera_board, _ = cv2.Rodrigues(
        rvec
    )

    T_camera_board = np.eye(
        4,
        dtype=np.float64,
    )

    T_camera_board[:3, :3] = (
        R_camera_board
    )

    T_camera_board[:3, 3] = (
        np.asarray(
            tvec,
            dtype=np.float64,
        ).reshape(3)
    )

    result.update(
        {
            "ok": True,
            "rmse_px": rmse_px,
            "rvec": np.asarray(
                rvec,
                dtype=np.float64,
            ).reshape(3),
            "tvec": np.asarray(
                tvec,
                dtype=np.float64,
            ).reshape(3),
            "T_camera_board":
                T_camera_board,
        }
    )

    return result




# ============================================================
# Sample helpers
# ============================================================

def next_sample_id():
    existing = sorted(
        DATA_DIR.glob(
            "sample_*.json"
        )
    )

    max_id = 0

    for path in existing:
        try:
            value = int(
                path.stem.split("_")[-1]
            )
            max_id = max(
                max_id,
                value,
            )
        except Exception:
            pass

    return max_id + 1


def matrix_valid(T):
    if T is None:
        return False

    T = np.asarray(
        T,
        dtype=float,
    )

    if T.shape != (4, 4):
        return False

    if not np.all(
        np.isfinite(T)
    ):
        return False

    return True


def save_sample(
    sample_id,
    image_bgr,
    camera_timestamp_us,
    arm_q,
    lowstate_age_ms,
    T_base_ee,
    detection,
):
    stem = (
        f"sample_{sample_id:04d}"
    )

    image_path = (
        IMAGE_DIR
        / f"{stem}.png"
    )

    json_path = (
        DATA_DIR
        / f"{stem}.json"
    )

    if not cv2.imwrite(
        str(image_path),
        image_bgr,
    ):
        raise RuntimeError(
            f"Failed to save image: {image_path}"
        )

    T_camera_board = (
        detection[
            "T_camera_board"
        ]
    )

    tvec = np.asarray(
        detection["tvec"],
        dtype=float,
    ).reshape(3)

    sample = {
        "sample_id": int(
            sample_id
        ),

        "host_time_ns": int(
            time.time_ns()
        ),

        "camera": {
            "serial_number":
                RIGHT_WRIST_SN,

            "width":
                COLOR_WIDTH,

            "height":
                COLOR_HEIGHT,

            "fps":
                COLOR_FPS,

            "timestamp_us":
                float(
                    camera_timestamp_us
                ),

            "camera_matrix":
                CAMERA_MATRIX.tolist(),

            "dist_coeffs":
                DIST_COEFFS
                .reshape(-1)
                .tolist(),
        },

        "board": {
            "type":
                "charuco",

            "squares_x":
                CHARUCO_SQUARES_X,

            "squares_y":
                CHARUCO_SQUARES_Y,

            "square_length_m":
                CHARUCO_SQUARE_LENGTH_M,

            "marker_length_m":
                CHARUCO_MARKER_LENGTH_M,

            "dictionary":
                "DICT_5X5_100",
        },

        "robot": {
            "state_topic":
                STATE_TOPIC,

            "q_arm_rad":
                np.asarray(
                    arm_q,
                    dtype=float,
                )
                .reshape(14)
                .tolist(),

            "lowstate_age_ms":
                float(
                    lowstate_age_ms
                ),

            "T_base_ee":
                np.asarray(
                    T_base_ee,
                    dtype=float,
                )
                .tolist(),
        },

        "vision": {
            "charuco_corner_count":
                int(
                    detection[
                        "corner_count"
                    ]
                ),

            "reprojection_rmse_px":
                float(
                    detection[
                        "rmse_px"
                    ]
                ),

            "board_translation_m":
                tvec.tolist(),

            "T_camera_board":
                np.asarray(
                    T_camera_board,
                    dtype=float,
                )
                .tolist(),
        },

        "image": str(
            image_path.relative_to(
                DATA_DIR
            )
        ),
    }

    temp_path = (
        json_path.with_suffix(
            ".json.tmp"
        )
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            sample,
            f,
            indent=2,
            ensure_ascii=False,
        )

    temp_path.replace(
        json_path
    )

    return json_path, image_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "R1-A7 right-wrist "
            "Eye-in-Hand sample collector"
        )
    )

    parser.add_argument(
        "--interface",
        default="enp6s0",
    )

    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    print()
    print(
        "=========================================="
    )
    print(
        " R1-A7 Eye-in-Hand Sample Collector"
    )
    print(
        " READ ONLY: NO LowCmd publisher"
    )
    print(
        "=========================================="
    )
    print()

    # --------------------------------------------------------
    # Load existing robot implementation
    # --------------------------------------------------------

    vs = load_robot_module()

    # --------------------------------------------------------
    # DDS / LowState
    # --------------------------------------------------------

    print(
        "[ROBOT] Initializing DDS..."
    )

    vs.ChannelFactoryInitialize(
        args.domain_id,
        args.interface,
    )

    crc = vs.CRC()

    state_buffer = (
        vs.base.StateBuffer(
            crc
        )
    )

    subscriber = (
        vs.ChannelSubscriber(
            STATE_TOPIC,
            vs.LowState_,
        )
    )

    subscriber.Init(
        state_buffer.callback,
        10,
    )

    print(
        "[ROBOT] Waiting for LowState..."
    )

    deadline = (
        time.monotonic()
        + 10.0
    )

    state = state_buffer.snapshot()

    while (
        state is None
        and time.monotonic()
        < deadline
    ):
        time.sleep(0.05)
        state = (
            state_buffer.snapshot()
        )

    if state is None:
        raise RuntimeError(
            "No valid rt/lowstate received."
        )

    print(
        "[ROBOT] LowState OK."
    )

    # --------------------------------------------------------
    # Robot model for FK only
    # --------------------------------------------------------

    print(
        "[ROBOT] Loading R1A7_ArmIK model "
        "for FK only..."
    )

    ik = vs.R1A7_ArmIK()

    print(
        "[ROBOT] FK model ready."
    )

    # --------------------------------------------------------
    # ChArUco
    # --------------------------------------------------------

    (
        aruco,
        dictionary,
        board,
        aruco_params,
        aruco_detector,
    ) = create_charuco()

    # --------------------------------------------------------
    # Right wrist Orbbec
    # --------------------------------------------------------

    from pyorbbecsdk import (
        Config,
        OBFormat,
        OBSensorType,
        Pipeline,
    )

    context, device = (
        find_orbbec_device(
            RIGHT_WRIST_SN
        )
    )

    pipeline = Pipeline(
        device
    )

    config = Config()

    profiles = (
        pipeline
        .get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
    )

    try:
        color_profile = (
            profiles
            .get_video_stream_profile(
                COLOR_WIDTH,
                COLOR_HEIGHT,
                OBFormat.RGB,
                COLOR_FPS,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Required wrist RGB profile "
            f"{COLOR_WIDTH}x{COLOR_HEIGHT}"
            f"@{COLOR_FPS} RGB unavailable. "
            "Do not fall back because current "
            "intrinsics are calibrated for "
            "1280x800."
        ) from exc

    config.enable_stream(
        color_profile
    )

    pipeline.start(
        config
    )

    print(
        "[CAMERA] RGB started: "
        f"{COLOR_WIDTH}x{COLOR_HEIGHT}"
        f"@{COLOR_FPS}"
    )

    print()
    print(
        "[CONTROL] S = save sample"
    )
    print(
        "[CONTROL] Q / ESC = quit"
    )
    print()

    sample_id = (
        next_sample_id()
    )

    window = (
        "R1-A7 Eye-in-Hand Collector"
    )

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL,
    )

    try:
        while True:
            frames = (
                pipeline.wait_for_frames(
                    1000
                )
            )

            if frames is None:
                continue

            color_frame = (
                frames.get_color_frame()
            )

            if color_frame is None:
                continue

            width = int(
                color_frame.get_width()
            )

            height = int(
                color_frame.get_height()
            )

            if (
                width != COLOR_WIDTH
                or height != COLOR_HEIGHT
            ):
                raise RuntimeError(
                    "Unexpected RGB resolution: "
                    f"{width}x{height}. "
                    "Expected 1280x800."
                )

            try:
                camera_timestamp_us = (
                    color_frame
                    .get_timestamp_us()
                )
            except Exception:
                camera_timestamp_us = (
                    time.time()
                    * 1e6
                )

            image = (
                color_frame_to_bgr(
                    color_frame
                )
            )

            detection = (
                detect_charuco_pose(
                    image,
                    aruco,
                    dictionary,
                    board,
                    aruco_params,
                    aruco_detector,
                )
            )

            display = image.copy()

            marker_ids = (
                detection[
                    "marker_ids"
                ]
            )

            marker_corners = (
                detection[
                    "marker_corners"
                ]
            )

            if (
                marker_ids is not None
                and len(marker_ids) > 0
            ):
                aruco.drawDetectedMarkers(
                    display,
                    marker_corners,
                    marker_ids,
                )

            if (
                detection[
                    "charuco_corners"
                ]
                is not None
            ):
                aruco.drawDetectedCornersCharuco(
                    display,
                    detection[
                        "charuco_corners"
                    ],
                    detection[
                        "charuco_ids"
                    ],
                )

            # Latest real robot state
            state = (
                state_buffer.snapshot()
            )

            arm_q = None
            lowstate_age_ms = float(
                "inf"
            )
            T_base_ee = None

            if state is not None:
                (
                    upper_q,
                    arm_q,
                    arm_dq,
                    gripper_q,
                    gripper_dq,
                    mode_machine,
                    received_at,
                ) = state

                lowstate_age_ms = (
                    1000.0
                    * (
                        time.monotonic()
                        - received_at
                    )
                )

                if (
                    arm_q is not None
                    and np.all(
                        np.isfinite(
                            np.asarray(
                                arm_q,
                                dtype=float,
                            )
                        )
                    )
                ):
                    _, T_base_ee = (
                        vs.get_ee_poses(
                            ik,
                            arm_q,
                        )
                    )

            # ------------------------------------------------
            # Validity gate
            # ------------------------------------------------

            vision_ok = (
                detection["ok"]
                and detection[
                    "corner_count"
                ]
                >= MIN_CHARUCO_CORNERS
                and detection[
                    "rmse_px"
                ]
                is not None
                and detection[
                    "rmse_px"
                ]
                <= MAX_RMSE_PX
            )

            robot_ok = (
                arm_q is not None
                and lowstate_age_ms
                <= MAX_LOWSTATE_AGE_MS
                and matrix_valid(
                    T_base_ee
                )
            )

            ready = (
                vision_ok
                and robot_ok
            )

            # ------------------------------------------------
            # Overlay
            # ------------------------------------------------

            status_lines = []

            status_lines.append(
                "RIGHT WRIST: "
                + RIGHT_WRIST_SN
            )

            status_lines.append(
                "Charuco corners: "
                f"{detection['corner_count']}"
            )

            if (
                detection[
                    "rmse_px"
                ]
                is None
            ):
                status_lines.append(
                    "RMSE: ---"
                )
            else:
                status_lines.append(
                    "RMSE: "
                    f"{detection['rmse_px']:.3f} px"
                )

            if detection[
                "tvec"
            ] is not None:
                xyz_mm = (
                    1000.0
                    * np.asarray(
                        detection[
                            "tvec"
                        ]
                    )
                )

                status_lines.append(
                    "Board camera XYZ: "
                    f"{xyz_mm[0]:+.1f}, "
                    f"{xyz_mm[1]:+.1f}, "
                    f"{xyz_mm[2]:+.1f} mm"
                )

            status_lines.append(
                "LowState age: "
                f"{lowstate_age_ms:.1f} ms"
            )

            status_lines.append(
                "Next sample: "
                f"{sample_id:04d}"
            )

            status_lines.append(
                "READY TO SAVE"
                if ready
                else "NOT READY"
            )

            y = 32

            for line in status_lines:
                cv2.putText(
                    display,
                    line,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

                y += 30

            if (
                detection["ok"]
                and detection[
                    "rvec"
                ]
                is not None
            ):
                cv2.drawFrameAxes(
                    display,
                    CAMERA_MATRIX,
                    DIST_COEFFS,
                    detection[
                        "rvec"
                    ].reshape(3, 1),
                    detection[
                        "tvec"
                    ].reshape(3, 1),
                    0.08,
                    2,
                )

            cv2.imshow(
                window,
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ------------------------------------------------
            # Quit
            # ------------------------------------------------

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            if key in (
                ord("s"),
                ord("S"),
            ):
                # Take a fresh robot snapshot at save time.
                state_save = (
                    state_buffer.snapshot()
                )

                if state_save is None:
                    print(
                        "[REJECT] No LowState."
                    )
                    continue

                (
                    upper_q,
                    arm_q_save,
                    arm_dq,
                    gripper_q,
                    gripper_dq,
                    mode_machine,
                    received_at_save,
                ) = state_save

                lowstate_age_save_ms = (
                    1000.0
                    * (
                        time.monotonic()
                        - received_at_save
                    )
                )

                try:
                    _, T_base_ee_save = (
                        vs.get_ee_poses(
                            ik,
                            arm_q_save,
                        )
                    )
                except Exception as exc:
                    print(
                        "[REJECT] FK failed:",
                        repr(exc),
                    )
                    continue

                vision_ok_save = (
                    detection["ok"]
                    and detection[
                        "corner_count"
                    ]
                    >= MIN_CHARUCO_CORNERS
                    and detection[
                        "rmse_px"
                    ]
                    is not None
                    and detection[
                        "rmse_px"
                    ]
                    <= MAX_RMSE_PX
                )

                robot_ok_save = (
                    lowstate_age_save_ms
                    <= MAX_LOWSTATE_AGE_MS
                    and matrix_valid(
                        T_base_ee_save
                    )
                    and np.all(
                        np.isfinite(
                            np.asarray(
                                arm_q_save,
                                dtype=float,
                            )
                        )
                    )
                )

                if not vision_ok_save:
                    print(
                        "[REJECT] Vision quality "
                        "not sufficient:",
                        "corners=",
                        detection[
                            "corner_count"
                        ],
                        "rmse=",
                        detection[
                            "rmse_px"
                        ],
                    )
                    continue

                if not robot_ok_save:
                    print(
                        "[REJECT] Robot state "
                        "not sufficient:",
                        "LowState age=",
                        f"{lowstate_age_save_ms:.1f} ms",
                    )
                    continue

                json_path, image_path = (
                    save_sample(
                        sample_id,
                        image,
                        camera_timestamp_us,
                        arm_q_save,
                        lowstate_age_save_ms,
                        T_base_ee_save,
                        detection,
                    )
                )

                print()
                print(
                    "======================================"
                )
                print(
                    f"[SAVED] sample_{sample_id:04d}"
                )
                print(
                    "======================================"
                )

                print(
                    "corners =",
                    detection[
                        "corner_count"
                    ],
                )

                print(
                    "rmse_px =",
                    f"{detection['rmse_px']:.4f}",
                )

                print(
                    "lowstate_age_ms =",
                    f"{lowstate_age_save_ms:.3f}",
                )

                print()
                print(
                    "T_base_ee ="
                )

                print(
                    np.array2string(
                        np.asarray(
                            T_base_ee_save
                        ),
                        precision=6,
                        suppress_small=True,
                    )
                )

                print()
                print(
                    "T_camera_board ="
                )

                print(
                    np.array2string(
                        np.asarray(
                            detection[
                                "T_camera_board"
                            ]
                        ),
                        precision=6,
                        suppress_small=True,
                    )
                )

                print()
                print(
                    "JSON:",
                    json_path,
                )

                print(
                    "Image:",
                    image_path,
                )

                print()

                sample_id += 1

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()

        print()
        print(
            "[DONE] Collector stopped."
        )
        print(
            "[SAFETY] No LowCmd publisher "
            "was created."
        )


if __name__ == "__main__":
    main()
