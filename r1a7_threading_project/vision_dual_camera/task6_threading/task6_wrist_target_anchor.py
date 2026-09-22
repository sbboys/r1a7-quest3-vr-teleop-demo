#!/usr/bin/env python3

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


THIS_FILE = Path(__file__).resolve()
HAND_EYE_DIR = THIS_FILE.parent

COLLECTOR_PATH = (
    HAND_EYE_DIR
    / "collect_handeye_samples.py"
)

CALIB_PATH = (
    Path(__file__).resolve().parents[1]
    / "handeye"
    / "r1a7_right_wrist_handeye.json"
)


OUTPUT_DIR = (
    HAND_EYE_DIR
    / "validation_live"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helpers
# ============================================================

def load_collector_module():
    import importlib.util
    import sys
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parents[1]
        / "handeye"
        / "collect_handeye_samples.py"
    )

    print(
        "[TASK6] collector module:",
        module_path,
    )

    if not module_path.exists():
        raise FileNotFoundError(
            f"Collector module not found: {module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "task6_collect_handeye_samples",
        str(module_path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot create import spec for: {module_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = module

    spec.loader.exec_module(
        module
    )

    return module


def load_handeye():
    with CALIB_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    T_ee_camera = np.asarray(
        data["T_ee_camera"],
        dtype=np.float64,
    )

    if T_ee_camera.shape != (4, 4):
        raise RuntimeError(
            "T_ee_camera must be 4x4"
        )

    if not np.all(
        np.isfinite(
            T_ee_camera
        )
    ):
        raise RuntimeError(
            "T_ee_camera contains NaN/Inf"
        )

    if data.get(
        "ee_frame"
    ) != "R_ee":
        raise RuntimeError(
            "Calibration ee_frame is not R_ee"
        )

    return data, T_ee_camera


def rotation_angle_deg(R):
    value = (
        np.trace(R) - 1.0
    ) / 2.0

    value = np.clip(
        value,
        -1.0,
        1.0,
    )

    return math.degrees(
        math.acos(value)
    )


def rotation_mean(rotations):
    M = np.mean(
        np.stack(
            rotations,
            axis=0,
        ),
        axis=0,
    )

    U, _, Vt = np.linalg.svd(
        M
    )

    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    return R


def compute_summary(samples):
    if len(samples) < 2:
        return None

    positions = np.array(
        [
            np.asarray(
                s["T_base_board"],
                dtype=np.float64,
            )[:3, 3]
            for s in samples
        ],
        dtype=np.float64,
    )

    rotations = [
        np.asarray(
            s["T_base_board"],
            dtype=np.float64,
        )[:3, :3]
        for s in samples
    ]

    mean_position = np.mean(
        positions,
        axis=0,
    )

    mean_rotation = rotation_mean(
        rotations
    )

    pos_errors_m = np.linalg.norm(
        positions
        - mean_position,
        axis=1,
    )

    rot_errors_deg = np.array(
        [
            rotation_angle_deg(
                mean_rotation.T
                @ R
            )
            for R in rotations
        ]
    )

    xyz_std_m = np.std(
        positions,
        axis=0,
    )

    return {
        "mean_board_position_base_m":
            mean_position.tolist(),

        "xyz_std_mm":
            (
                1000.0
                * xyz_std_m
            ).tolist(),

        "position_rms_mm":
            float(
                1000.0
                * np.sqrt(
                    np.mean(
                        pos_errors_m ** 2
                    )
                )
            ),

        "position_max_mm":
            float(
                1000.0
                * np.max(
                    pos_errors_m
                )
            ),

        "rotation_rms_deg":
            float(
                np.sqrt(
                    np.mean(
                        rot_errors_deg ** 2
                    )
                )
            ),

        "rotation_max_deg":
            float(
                np.max(
                    rot_errors_deg
                )
            ),
    }


def main():
    print()
    print(
        "============================================="
    )
    print(
        " R1-A7 Camera -> Base Live Validation"
    )
    print(
        " READ ONLY: NO LowCmd publisher"
    )
    print(
        "============================================="
    )
    print()

    # --------------------------------------------------------
    # Load already tested collector implementation
    # --------------------------------------------------------

    hc = load_collector_module()

    calib_data, T_ee_camera = (
        load_handeye()
    )

    print(
        "[CALIB] Loaded:",
        CALIB_PATH,
    )

    print(
        "[CALIB] Method:",
        calib_data.get(
            "method"
        ),
    )

    print()
    print(
        "T_ee_camera ="
    )

    print(
        np.array2string(
            T_ee_camera,
            precision=8,
            suppress_small=True,
        )
    )


    # ========================================================
    # TASK6_FIXED_TARGET_ANCHOR_V1
    #
    # The target identity comes ONLY from the fixed camera.
    # Wrist vision must never change this Base-frame target.
    # ========================================================

    task6_target_file = (
        "/tmp/r1a7_task6_hole_base.json"
    )

    try:
        with open(
            task6_target_file,
            "r",
            encoding="utf-8",
        ) as f:
            task6_target_data = json.load(f)
    except Exception as e:
        raise RuntimeError(
            "Cannot load TASK6 fixed-camera target: "
            f"{task6_target_file}: {e}"
        )

    task6_hole_base_m = np.asarray(
        task6_target_data["hole_base_m"],
        dtype=np.float64,
    ).reshape(3)

    print()
    print(
        "[TASK6] Fixed target identity loaded:"
    )

    print(
        "[TASK6] hole Base mm =",
        np.round(
            1000.0
            * task6_hole_base_m,
            3,
        ),
    )

    # --------------------------------------------------------
    # Robot DDS / FK
    # --------------------------------------------------------

    vs = hc.load_robot_module()

    print()
    print(
        "[ROBOT] Initializing DDS..."
    )

    vs.ChannelFactoryInitialize(
        0,
        "enp6s0",
    )

    crc = vs.CRC()

    state_buffer = (
        vs.base.StateBuffer(
            crc
        )
    )

    subscriber = (
        vs.ChannelSubscriber(
            hc.STATE_TOPIC,
            vs.LowState_,
        )
    )

    subscriber.Init(
        state_buffer.callback,
        10,
    )

    deadline = (
        time.monotonic()
        + 10.0
    )

    state = (
        state_buffer.snapshot()
    )

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
            "No valid LowState received."
        )

    print(
        "[ROBOT] LowState OK."
    )

    print(
        "[ROBOT] Loading FK model..."
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
        charuco_detector,
    ) = hc.create_charuco()

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    from pyorbbecsdk import (
        Config,
        OBFormat,
        OBSensorType,
        Pipeline,
    )

    context, device = (
        hc.find_orbbec_device(
            hc.RIGHT_WRIST_SN
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

    color_profile = (
        profiles
        .get_video_stream_profile(
            hc.COLOR_WIDTH,
            hc.COLOR_HEIGHT,
            OBFormat.RGB,
            hc.COLOR_FPS,
        )
    )

    config.enable_stream(
        color_profile
    )

    pipeline.start(
        config
    )

    print(
        "[CAMERA] RGB started:",
        f"{hc.COLOR_WIDTH}x"
        f"{hc.COLOR_HEIGHT}@"
        f"{hc.COLOR_FPS}",
    )

    print()
    print(
        "[CONTROL]"
    )
    print(
        "R = set current board pose as reference"
    )
    print(
        "S = save one validation pose"
    )
    print(
        "Q / ESC = quit"
    )
    print()

    reference_T_base_board = None

    saved_samples = []

    window = (
        "TASK6 Wrist Fixed-Target Anchor"
    )

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL,
    )

    task6_last_anchor_print = 0.0

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

            image = (
                hc.color_frame_to_bgr(
                    color_frame
                )
            )

            # --------------------------------------------------
            # TASK6:
            # ChArUco pose detection is deliberately disabled.
            #
            # This program is NOT a hand-eye validation program.
            # Target identity comes from the fixed-camera target.
            # Partial ChArUco visibility must never terminate the
            # wrist target-anchor process.
            # --------------------------------------------------

            from collections import defaultdict

            detection = defaultdict(
                lambda: None
            )

            detection.update(
                {
                    "ok": False,
                    "marker_ids": None,
                    "marker_corners": [],
                    "charuco_corners": None,
                    "charuco_ids": None,
                    "rvec": None,
                    "tvec": None,
                    "corner_count": 0,
                    "charuco_count": 0,
                    "reprojection_rmse_px": None,
                    "rmse_px": None,
                }
            )

            display = image.copy()

            # ----------------------------------------------
            # Draw detected features
            # ----------------------------------------------

            if (
                detection[
                    "marker_ids"
                ]
                is not None
            ):
                try:
                    aruco.drawDetectedMarkers(
                        display,
                        detection[
                            "marker_corners"
                        ],
                        detection[
                            "marker_ids"
                        ],
                    )
                except Exception:
                    pass

            if (
                detection[
                    "charuco_corners"
                ]
                is not None
            ):
                try:
                    aruco.drawDetectedCornersCharuco(
                        display,
                        detection[
                            "charuco_corners"
                        ],
                        detection[
                            "charuco_ids"
                        ],
                    )
                except Exception:
                    pass

            if (
                detection["ok"]
                and detection[
                    "rvec"
                ]
                is not None
            ):
                try:
                    cv2.drawFrameAxes(
                        display,
                        hc.CAMERA_MATRIX,
                        hc.DIST_COEFFS,
                        detection[
                            "rvec"
                        ].reshape(3, 1),
                        detection[
                            "tvec"
                        ].reshape(3, 1),
                        0.08,
                        2,
                    )
                except Exception:
                    pass

            # ----------------------------------------------
            # Robot state
            # ----------------------------------------------

            state = (
                state_buffer.snapshot()
            )

            T_base_ee = None
            T_base_camera = None
            T_base_board = None

            lowstate_age_ms = float(
                "inf"
            )

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

                try:
                    _, T_base_ee = (
                        vs.get_ee_poses(
                            ik,
                            arm_q,
                        )
                    )

                    T_base_camera = (
                        T_base_ee
                        @ T_ee_camera
                    )

                except Exception:
                    T_base_ee = None


            # ==================================================
            # TASK6_FIXED_TARGET_ANCHOR_V1
            #
            # Fixed target:
            #       P_hole_base
            #
            # Current wrist pose:
            #       T_base_camera
            #
            # Therefore:
            #       P_hole_camera
            #         = inv(T_base_camera) @ P_hole_base
            #
            # This prediction is recomputed every frame.
            # It NEVER depends on the previously detected hole.
            # ==================================================

            task6_predicted_uv = None
            task6_hole_camera_m = None

            if T_base_camera is not None:

                try:
                    T_camera_base = (
                        np.linalg.inv(
                            T_base_camera
                        )
                    )

                    p_base_h = np.array(
                        [
                            task6_hole_base_m[0],
                            task6_hole_base_m[1],
                            task6_hole_base_m[2],
                            1.0,
                        ],
                        dtype=np.float64,
                    )

                    p_camera_h = (
                        T_camera_base
                        @ p_base_h
                    )

                    task6_hole_camera_m = (
                        p_camera_h[:3]
                    )

                    Xc = float(
                        task6_hole_camera_m[0]
                    )

                    Yc = float(
                        task6_hole_camera_m[1]
                    )

                    Zc = float(
                        task6_hole_camera_m[2]
                    )

                    if (
                        np.isfinite(Xc)
                        and np.isfinite(Yc)
                        and np.isfinite(Zc)
                        and Zc > 0.0
                    ):

                        object_point = np.array(
                            [
                                [
                                    [
                                        Xc,
                                        Yc,
                                        Zc,
                                    ]
                                ]
                            ],
                            dtype=np.float64,
                        )

                        image_point, _ = (
                            cv2.projectPoints(
                                object_point,
                                np.zeros(
                                    (3, 1),
                                    dtype=np.float64,
                                ),
                                np.zeros(
                                    (3, 1),
                                    dtype=np.float64,
                                ),
                                hc.CAMERA_MATRIX,
                                hc.DIST_COEFFS,
                            )
                        )

                        uv = (
                            image_point
                            .reshape(-1, 2)[0]
                        )

                        u_pred = float(
                            uv[0]
                        )

                        v_pred = float(
                            uv[1]
                        )

                        task6_predicted_uv = (
                            u_pred,
                            v_pred,
                        )

                        image_h, image_w = (
                            display.shape[:2]
                        )

                        if (
                            0.0 <= u_pred < image_w
                            and
                            0.0 <= v_pred < image_h
                        ):

                            up = int(
                                round(u_pred)
                            )

                            vp = int(
                                round(v_pred)
                            )

                            # Yellow = geometric prediction.
                            cv2.drawMarker(
                                display,
                                (up, vp),
                                (0, 255, 255),
                                cv2.MARKER_CROSS,
                                34,
                                2,
                            )

                            cv2.circle(
                                display,
                                (up, vp),
                                18,
                                (0, 255, 255),
                                2,
                            )

                            cv2.putText(
                                display,
                                "TASK6 FIXED TARGET",
                                (
                                    max(
                                        10,
                                        up - 120,
                                    ),
                                    max(
                                        30,
                                        vp - 28,
                                    ),
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (0, 255, 255),
                                2,
                            )

                        cv2.putText(
                            display,
                            (
                                "Target predicted RGB: "
                                f"u={u_pred:.1f} "
                                f"v={v_pred:.1f}"
                            ),
                            (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.68,
                            (0, 255, 255),
                            2,
                        )

                        cv2.putText(
                            display,
                            (
                                "Target Camera XYZ mm: "
                                f"[{1000.0*Xc:.1f}, "
                                f"{1000.0*Yc:.1f}, "
                                f"{1000.0*Zc:.1f}]"
                            ),
                            (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.68,
                            (0, 255, 255),
                            2,
                        )

                        now_anchor = (
                            time.monotonic()
                        )

                        if (
                            now_anchor
                            - task6_last_anchor_print
                            >= 0.5
                        ):
                            print(
                                "[TASK6 ANCHOR] "
                                f"pred_uv="
                                f"({u_pred:.1f},"
                                f"{v_pred:.1f}) "
                                f"cameraXYZ_mm="
                                f"[{1000.0*Xc:.1f},"
                                f"{1000.0*Yc:.1f},"
                                f"{1000.0*Zc:.1f}]",
                                flush=True,
                            )

                            task6_last_anchor_print = (
                                now_anchor
                            )

                    else:

                        cv2.putText(
                            display,
                            "TASK6 target is behind wrist camera",
                            (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.68,
                            (0, 0, 255),
                            2,
                        )

                except Exception as e:

                    cv2.putText(
                        display,
                        (
                            "TASK6 projection error: "
                            f"{type(e).__name__}"
                        ),
                        (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.68,
                        (0, 0, 255),
                        2,
                    )

            # ----------------------------------------------
            # Camera -> Base transformation
            # ----------------------------------------------

            valid = (
                detection["ok"]
                and detection[
                    "corner_count"
                ] >= hc.MIN_CHARUCO_CORNERS
                and detection[
                    "rmse_px"
                ] is not None
                and detection[
                    "rmse_px"
                ] <= hc.MAX_RMSE_PX
                and T_base_ee is not None
                and lowstate_age_ms
                <= hc.MAX_LOWSTATE_AGE_MS
            )

            if valid:
                T_base_board = (
                    T_base_camera
                    @ detection[
                        "T_camera_board"
                    ]
                )

            # ----------------------------------------------
            # Overlay
            # ----------------------------------------------

            lines = []

            lines.append(
                "Camera->Base geometric validation"
            )

            lines.append(
                f"corners: "
                f"{detection['corner_count']}"
            )

            if (
                detection[
                    "rmse_px"
                ]
                is not None
            ):
                lines.append(
                    "RMSE: "
                    f"{detection['rmse_px']:.3f} px"
                )
            else:
                lines.append(
                    "RMSE: ---"
                )

            lines.append(
                "LowState age: "
                f"{lowstate_age_ms:.2f} ms"
            )

            if (
                detection[
                    "tvec"
                ]
                is not None
            ):
                c_xyz = (
                    1000.0
                    * np.asarray(
                        detection[
                            "tvec"
                        ]
                    )
                )

                lines.append(
                    "Board Camera XYZ [mm]: "
                    f"{c_xyz[0]:+.1f}, "
                    f"{c_xyz[1]:+.1f}, "
                    f"{c_xyz[2]:+.1f}"
                )

            if T_base_board is not None:
                b_xyz = (
                    1000.0
                    * T_base_board[
                        :3,
                        3
                    ]
                )

                lines.append(
                    "Board Base XYZ [mm]: "
                    f"{b_xyz[0]:+.1f}, "
                    f"{b_xyz[1]:+.1f}, "
                    f"{b_xyz[2]:+.1f}"
                )

                if (
                    reference_T_base_board
                    is not None
                ):
                    dp_mm = (
                        1000.0
                        * np.linalg.norm(
                            T_base_board[
                                :3,
                                3
                            ]
                            - reference_T_base_board[
                                :3,
                                3
                            ]
                        )
                    )

                    dR = (
                        reference_T_base_board[
                            :3,
                            :3
                        ].T
                        @ T_base_board[
                            :3,
                            :3
                        ]
                    )

                    dr_deg = (
                        rotation_angle_deg(
                            dR
                        )
                    )

                    lines.append(
                        "Delta REF: "
                        f"{dp_mm:.2f} mm, "
                        f"{dr_deg:.3f} deg"
                    )
                else:
                    lines.append(
                        "Reference: NOT SET "
                        "(press R)"
                    )

            lines.append(
                "VALID"
                if valid
                else "NOT VALID"
            )

            lines.append(
                f"Saved poses: "
                f"{len(saved_samples)}"
            )

            y = 30

            for line in lines:
                cv2.putText(
                    display,
                    line,
                    (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

                y += 28

            cv2.imshow(
                window,
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ----------------------------------------------
            # Quit
            # ----------------------------------------------

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            # ----------------------------------------------
            # Set reference
            # ----------------------------------------------

            if key in (
                ord("r"),
                ord("R"),
            ):
                if not valid:
                    print(
                        "[REF REJECT] "
                        "Current frame is not valid."
                    )
                    continue

                reference_T_base_board = (
                    T_base_board.copy()
                )

                xyz_mm = (
                    1000.0
                    * reference_T_base_board[
                        :3,
                        3
                    ]
                )

                print()
                print(
                    "[REFERENCE SET]"
                )

                print(
                    "Board Base XYZ [mm]: "
                    f"{xyz_mm[0]:+.3f}, "
                    f"{xyz_mm[1]:+.3f}, "
                    f"{xyz_mm[2]:+.3f}"
                )

            # ----------------------------------------------
            # Save validation pose
            # ----------------------------------------------

            if key in (
                ord("s"),
                ord("S"),
            ):

                if not valid:
                    print(
                        "[SAVE REJECT] "
                        "Current frame is not valid."
                    )
                    continue

                sample = {
                    "index":
                        len(saved_samples)
                        + 1,

                    "host_time_ns":
                        time.time_ns(),

                    "corners":
                        int(
                            detection[
                                "corner_count"
                            ]
                        ),

                    "rmse_px":
                        float(
                            detection[
                                "rmse_px"
                            ]
                        ),

                    "lowstate_age_ms":
                        float(
                            lowstate_age_ms
                        ),

                    "T_base_ee":
                        np.asarray(
                            T_base_ee
                        ).tolist(),

                    "T_base_camera":
                        np.asarray(
                            T_base_camera
                        ).tolist(),

                    "T_camera_board":
                        np.asarray(
                            detection[
                                "T_camera_board"
                            ]
                        ).tolist(),

                    "T_base_board":
                        np.asarray(
                            T_base_board
                        ).tolist(),
                }

                if (
                    reference_T_base_board
                    is not None
                ):
                    dp_mm = (
                        1000.0
                        * np.linalg.norm(
                            T_base_board[
                                :3,
                                3
                            ]
                            - reference_T_base_board[
                                :3,
                                3
                            ]
                        )
                    )

                    dR = (
                        reference_T_base_board[
                            :3,
                            :3
                        ].T
                        @ T_base_board[
                            :3,
                            :3
                        ]
                    )

                    sample[
                        "reference_position_error_mm"
                    ] = float(
                        dp_mm
                    )

                    sample[
                        "reference_rotation_error_deg"
                    ] = float(
                        rotation_angle_deg(
                            dR
                        )
                    )

                saved_samples.append(
                    sample
                )

                xyz_mm = (
                    1000.0
                    * T_base_board[
                        :3,
                        3
                    ]
                )

                print()
                print(
                    "====================================="
                )

                print(
                    "[SAVED VALIDATION POSE] "
                    f"{len(saved_samples):02d}"
                )

                print(
                    "Board Base XYZ [mm]: "
                    f"{xyz_mm[0]:+.3f}, "
                    f"{xyz_mm[1]:+.3f}, "
                    f"{xyz_mm[2]:+.3f}"
                )

                if (
                    reference_T_base_board
                    is not None
                ):
                    print(
                        "Delta REF: "
                        f"{sample['reference_position_error_mm']:.3f} mm, "
                        f"{sample['reference_rotation_error_deg']:.4f} deg"
                    )

                print(
                    "corners:",
                    detection[
                        "corner_count"
                    ],
                )

                print(
                    "rmse_px:",
                    f"{detection['rmse_px']:.4f}",
                )

                print(
                    "lowstate_age_ms:",
                    f"{lowstate_age_ms:.3f}",
                )

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()

    # ========================================================
    # Final statistics
    # ========================================================

    print()
    print(
        "============================================="
    )
    print(
        " Validation Summary"
    )
    print(
        "============================================="
    )

    summary = compute_summary(
        saved_samples
    )

    if summary is None:
        print(
            "Need at least 2 saved validation poses."
        )

    else:
        print(
            "Saved poses:",
            len(saved_samples),
        )

        print(
            "Mean Board Base XYZ [mm]: "
            f"{1000.0 * summary['mean_board_position_base_m'][0]:+.3f}, "
            f"{1000.0 * summary['mean_board_position_base_m'][1]:+.3f}, "
            f"{1000.0 * summary['mean_board_position_base_m'][2]:+.3f}"
        )

        print(
            "XYZ std [mm]: "
            f"{summary['xyz_std_mm'][0]:.3f}, "
            f"{summary['xyz_std_mm'][1]:.3f}, "
            f"{summary['xyz_std_mm'][2]:.3f}"
        )

        print(
            "Position RMS [mm]: "
            f"{summary['position_rms_mm']:.3f}"
        )

        print(
            "Position MAX [mm]: "
            f"{summary['position_max_mm']:.3f}"
        )

        print(
            "Rotation RMS [deg]: "
            f"{summary['rotation_rms_deg']:.4f}"
        )

        print(
            "Rotation MAX [deg]: "
            f"{summary['rotation_max_deg']:.4f}"
        )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        OUTPUT_DIR
        / f"validation_{timestamp}.json"
    )

    output = {
        "calibration_file":
            str(CALIB_PATH),

        "T_ee_camera":
            T_ee_camera.tolist(),

        "reference_T_base_board":
            (
                reference_T_base_board.tolist()
                if reference_T_base_board
                is not None
                else None
            ),

        "samples":
            saved_samples,

        "summary":
            summary,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Saved validation:",
        output_path,
    )

    print(
        "[SAFETY] No LowCmd publisher was created."
    )


if __name__ == "__main__":
    main()
