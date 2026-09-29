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
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

ROBOT_MODULE_PATH = (
    PROJECT_ROOT
    / "vision"
    / "r1a7_head_visual_servo_lowcmd.py"
)

INTRINSIC_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "fixed_camera_intrinsics"
    / "results"
    / "usb_zoom_camera_intrinsics.json"
)

EXTRINSIC_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "fixed_camera_extrinsic"
    / "usb_zoom_camera_base_extrinsic_frozen_20260920.json"
)

TCP_PATH = (
    PROJECT_ROOT
    / "tool_calibration"
    / "tcp_pivot"
    / "r1a7_right_tool_tcp.json"
)


# ============================================================
# Load existing verified robot implementation
# ============================================================

def load_robot_module():

    if not ROBOT_MODULE_PATH.exists():

        raise FileNotFoundError(
            f"Robot module not found: "
            f"{ROBOT_MODULE_PATH}"
        )

    # Existing project modules use both repository-level
    # and project-level imports.
    sys.path.insert(
        0,
        str(PROJECT_ROOT.parent),
    )

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

    spec = (
        importlib.util.spec_from_file_location(
            "r1a7_existing_visual_servo",
            str(ROBOT_MODULE_PATH),
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):

        raise RuntimeError(
            "Cannot load robot module."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# Calibration
# ============================================================

def load_calibration():

    # --------------------------------------------------------
    # Camera intrinsics
    # --------------------------------------------------------

    intr_data = json.loads(
        INTRINSIC_PATH.read_text(
            encoding="utf-8"
        )
    )

    calib = intr_data.get(
        "calibration",
        intr_data,
    )

    K = np.asarray(
        calib["camera_matrix"],
        dtype=np.float64,
    ).reshape(3, 3)

    dist = np.asarray(
        calib["dist_coeffs"],
        dtype=np.float64,
    ).reshape(-1, 1)

    calib_width = int(
        calib["image_width"]
    )

    calib_height = int(
        calib["image_height"]
    )

    # --------------------------------------------------------
    # Fixed external-camera extrinsic
    #
    # Stored convention:
    #
    #     P_base =
    #         T_base_camera @ P_camera
    #
    # Projection requires the inverse:
    #
    #     P_camera =
    #         T_camera_base @ P_base
    # --------------------------------------------------------

    ext_data = json.loads(
        EXTRINSIC_PATH.read_text(
            encoding="utf-8"
        )
    )

    T_base_camera = np.asarray(
        ext_data["T_base_camera"],
        dtype=np.float64,
    ).reshape(4, 4)

    T_camera_base = np.linalg.inv(
        T_base_camera
    )

    # --------------------------------------------------------
    # Tool TCP
    # --------------------------------------------------------

    tcp_data = json.loads(
        TCP_PATH.read_text(
            encoding="utf-8"
        )
    )

    p_tip_ee = np.asarray(
        tcp_data["P_tip_ee_m"],
        dtype=np.float64,
    ).reshape(3)

    return {
        "K":
        K,

        "dist":
        dist,

        "calib_width":
        calib_width,

        "calib_height":
        calib_height,

        "T_base_camera":
        T_base_camera,

        "T_camera_base":
        T_camera_base,

        "p_tip_ee":
        p_tip_ee,
    }


def scale_camera_matrix(
    K,
    calib_width,
    calib_height,
    runtime_width,
    runtime_height,
):

    sx = (
        float(runtime_width)
        /
        float(calib_width)
    )

    sy = (
        float(runtime_height)
        /
        float(calib_height)
    )

    K_runtime = (
        np.asarray(
            K,
            dtype=np.float64,
        )
        .copy()
    )

    K_runtime[0, 0] *= sx
    K_runtime[0, 2] *= sx

    K_runtime[1, 1] *= sy
    K_runtime[1, 2] *= sy

    return K_runtime


# ============================================================
# Geometry
# ============================================================

def calculate_tip_base(
    T_base_ee,
    p_tip_ee,
):

    T_base_ee = np.asarray(
        T_base_ee,
        dtype=np.float64,
    ).reshape(4, 4)

    R_base_ee = (
        T_base_ee[:3, :3]
    )

    p_ee_base = (
        T_base_ee[:3, 3]
    )

    # Existing project convention:
    #
    # p_tip_base =
    #     p_ee_base
    #     +
    #     R_base_ee @ p_tip_ee

    p_tip_base = (
        p_ee_base
        +
        R_base_ee
        @
        p_tip_ee
    )

    return p_tip_base


def base_point_to_camera(
    p_base,
    T_camera_base,
):

    p_h = np.ones(
        4,
        dtype=np.float64,
    )

    p_h[:3] = (
        np.asarray(
            p_base,
            dtype=np.float64,
        )
        .reshape(3)
    )

    p_camera_h = (
        T_camera_base
        @
        p_h
    )

    return (
        p_camera_h[:3]
    )


def project_camera_point(
    p_camera,
    K,
    dist,
):

    p_camera = np.asarray(
        p_camera,
        dtype=np.float64,
    ).reshape(3)

    # Camera optical convention requires positive Z.
    if (
        not np.all(
            np.isfinite(
                p_camera
            )
        )
        or
        p_camera[2] <= 1.0e-6
    ):

        return None

    object_points = (
        p_camera
        .reshape(1, 1, 3)
    )

    image_points, _ = (
        cv2.projectPoints(
            object_points,
            np.zeros(
                (3, 1),
                dtype=np.float64,
            ),
            np.zeros(
                (3, 1),
                dtype=np.float64,
            ),
            K,
            dist,
        )
    )

    uv = (
        image_points
        .reshape(2)
    )

    if not np.all(
        np.isfinite(
            uv
        )
    ):

        return None

    return uv


# ============================================================
# Drawing
# ============================================================

def draw_prediction(
    image,
    uv,
    roi_half_width,
    roi_half_height,
):

    h, w = image.shape[:2]

    u = int(
        round(
            float(uv[0])
        )
    )

    v = int(
        round(
            float(uv[1])
        )
    )

    inside = (
        0 <= u < w
        and
        0 <= v < h
    )

    # --------------------------------------------------------
    # Predicted SAM2 ROI
    # --------------------------------------------------------

    x1 = int(
        np.clip(
            u - roi_half_width,
            0,
            w - 1,
        )
    )

    y1 = int(
        np.clip(
            v - roi_half_height,
            0,
            h - 1,
        )
    )

    x2 = int(
        np.clip(
            u + roi_half_width,
            0,
            w - 1,
        )
    )

    y2 = int(
        np.clip(
            v + roi_half_height,
            0,
            h - 1,
        )
    )

    if inside:

        # Purple:
        # future automatically predicted SAM2 ROI.
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            2,
        )

        # Blue:
        # kinematic predicted tweezer tip.
        cv2.drawMarker(
            image,
            (u, v),
            (255, 0, 0),
            cv2.MARKER_CROSS,
            36,
            3,
        )

        cv2.putText(
            image,
            "FK PREDICTED TIP",
            (
                u + 15,
                max(
                    25,
                    v - 15,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2,
        )

    return (
        u,
        v,
        inside,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
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

    parser.add_argument(
        "--state-topic",
        default="rt/lowstate",
    )

    parser.add_argument(
        "--device",
        default="/dev/video16",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--roi-half-width",
        type=int,
        default=140,
    )

    parser.add_argument(
        "--roi-half-height",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--lowstate-timeout",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--print-period",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    print("")
    print(
        "=========================================="
    )
    print(
        " EXTERNAL FK TIP PROJECTION"
    )
    print(
        " READ ONLY - NO LowCmd publisher"
    )
    print(
        "=========================================="
    )
    print("")

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    calib = (
        load_calibration()
    )

    print(
        "[CALIB] intrinsics:",
        f"{calib['calib_width']}x"
        f"{calib['calib_height']}",
    )

    print(
        "[CALIB] P_tip_ee_m =",
        np.round(
            calib["p_tip_ee"],
            6,
        ).tolist(),
    )

    print(
        "[CALIB] T_base_camera translation =",
        np.round(
            calib[
                "T_base_camera"
            ][:3, 3],
            6,
        ).tolist(),
    )

    # --------------------------------------------------------
    # Existing verified robot module
    # --------------------------------------------------------

    print(
        "[ROBOT] loading existing module...",
        flush=True,
    )

    vs = (
        load_robot_module()
    )

    print(
        "[ROBOT] initializing DDS...",
        flush=True,
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
            args.state_topic,
            vs.LowState_,
        )
    )

    subscriber.Init(
        state_buffer.callback,
        10,
    )

    print(
        "[ROBOT] waiting for LowState...",
        flush=True,
    )

    deadline = (
        time.monotonic()
        +
        10.0
    )

    state = (
        state_buffer.snapshot()
    )

    while (
        state is None
        and
        time.monotonic()
        <
        deadline
    ):

        time.sleep(
            0.05
        )

        state = (
            state_buffer.snapshot()
        )

    if state is None:

        raise RuntimeError(
            "No valid rt/lowstate received."
        )

    print(
        "[ROBOT] LowState OK.",
        flush=True,
    )

    print(
        "[ROBOT] loading R1A7_ArmIK "
        "for FK only...",
        flush=True,
    )

    try:

        ik = vs.R1A7_ArmIK(
            Unit_Test=False,
            Visualization=False,
        )

    except TypeError:

        ik = (
            vs.R1A7_ArmIK()
        )

    print(
        "[ROBOT] FK model ready.",
        flush=True,
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        args.device,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open camera "
            f"{args.device}"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        args.width,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        args.height,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        args.fps,
    )

    actual_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        "[CAMERA]",
        args.device,
        f"{actual_width}x{actual_height}",
    )

    K_runtime = (
        scale_camera_matrix(
            calib["K"],
            calib["calib_width"],
            calib["calib_height"],
            actual_width,
            actual_height,
        )
    )

    print(
        "[CAMERA] runtime K ="
    )

    print(
        np.array2string(
            K_runtime,
            precision=4,
        )
    )

    print("")
    print(
        "Blue cross  = FK predicted TCP/tweezer tip"
    )
    print(
        "Purple box  = future SAM2 search ROI"
    )
    print(
        "Q / ESC     = quit"
    )
    print("")

    last_print = 0.0

    window = (
        "EXTERNAL FK TIP PROJECTION"
    )

    cv2.namedWindow(
        window
    )

    try:

        while True:

            ret, frame = (
                cap.read()
            )

            if not ret:

                continue

            display = (
                frame.copy()
            )

            state = (
                state_buffer.snapshot()
            )

            status = (
                "NO LOWSTATE"
            )

            uv = None

            p_tip_base = None
            p_tip_camera = None

            lowstate_age = None

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

                lowstate_age = (
                    time.monotonic()
                    -
                    received_at
                )

                if (
                    lowstate_age
                    <=
                    args.lowstate_timeout
                    and
                    np.all(
                        np.isfinite(
                            arm_q
                        )
                    )
                ):

                    try:

                        (
                            _,
                            T_base_ee,
                        ) = (
                            vs.get_ee_poses(
                                ik,
                                arm_q,
                            )
                        )

                        p_tip_base = (
                            calculate_tip_base(
                                T_base_ee,
                                calib[
                                    "p_tip_ee"
                                ],
                            )
                        )

                        p_tip_camera = (
                            base_point_to_camera(
                                p_tip_base,
                                calib[
                                    "T_camera_base"
                                ],
                            )
                        )

                        uv = (
                            project_camera_point(
                                p_tip_camera,
                                K_runtime,
                                calib[
                                    "dist"
                                ],
                            )
                        )

                        if uv is None:

                            status = (
                                "TIP BEHIND CAMERA "
                                "OR INVALID"
                            )

                        else:

                            (
                                u,
                                v,
                                inside,
                            ) = (
                                draw_prediction(
                                    display,
                                    uv,
                                    args.roi_half_width,
                                    args.roi_half_height,
                                )
                            )

                            if inside:

                                status = (
                                    "FK PROJECTION OK"
                                )

                            else:

                                status = (
                                    "PREDICTION "
                                    "OUTSIDE IMAGE"
                                )

                    except Exception as exc:

                        status = (
                            "FK/PROJECTION ERROR: "
                            +
                            type(exc).__name__
                        )

                else:

                    status = (
                        "LOWSTATE STALE"
                    )

            # ------------------------------------------------
            # Status display
            # ------------------------------------------------

            good = (
                status
                ==
                "FK PROJECTION OK"
            )

            cv2.putText(
                display,
                status,
                (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    (0, 255, 0)
                    if good
                    else
                    (0, 0, 255)
                ),
                2,
            )

            if uv is not None:

                cv2.putText(
                    display,
                    (
                        "pred uv = "
                        f"({uv[0]:.1f}, "
                        f"{uv[1]:.1f})"
                    ),
                    (25, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    2,
                )

            if (
                p_tip_camera
                is not None
            ):

                cv2.putText(
                    display,
                    (
                        "tip camera XYZ m = "
                        f"{p_tip_camera[0]:+.3f}, "
                        f"{p_tip_camera[1]:+.3f}, "
                        f"{p_tip_camera[2]:+.3f}"
                    ),
                    (25, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )

            if (
                lowstate_age
                is not None
            ):

                cv2.putText(
                    display,
                    (
                        "LowState age = "
                        f"{1000.0 * lowstate_age:.1f} ms"
                    ),
                    (25, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )

            now = (
                time.monotonic()
            )

            if (
                now
                -
                last_print
                >=
                args.print_period
            ):

                if (
                    uv is not None
                    and
                    p_tip_base
                    is not None
                    and
                    p_tip_camera
                    is not None
                ):

                    print(
                        "[FK PROJ] "
                        f"status={status} "
                        f"uv=("
                        f"{uv[0]:.1f},"
                        f"{uv[1]:.1f}) "
                        f"tip_base=("
                        f"{p_tip_base[0]:+.4f},"
                        f"{p_tip_base[1]:+.4f},"
                        f"{p_tip_base[2]:+.4f}) "
                        f"tip_cam=("
                        f"{p_tip_camera[0]:+.4f},"
                        f"{p_tip_camera[1]:+.4f},"
                        f"{p_tip_camera[2]:+.4f}) "
                        f"age_ms="
                        f"{1000.0 * lowstate_age:.1f}",
                        flush=True,
                    )

                else:

                    print(
                        "[FK PROJ] "
                        f"status={status}",
                        flush=True,
                    )

                last_print = now

            cv2.imshow(
                window,
                display,
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if (
                key == ord("q")
                or
                key == ord("Q")
                or
                key == 27
            ):

                break

    except KeyboardInterrupt:

        print("")
        print(
            "[FK PROJ] Ctrl+C",
            flush=True,
        )

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print(
            "[FK PROJ] stopped",
            flush=True,
        )


if __name__ == "__main__":

    main()
