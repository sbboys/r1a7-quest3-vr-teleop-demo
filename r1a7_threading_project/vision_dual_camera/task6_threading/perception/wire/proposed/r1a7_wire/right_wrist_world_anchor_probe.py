#!/usr/bin/env python3

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Locate project
# ============================================================

HERE = Path(__file__).resolve().parent

ROOT = None

for p in [HERE] + list(HERE.parents):
    if (
        (p / "vision_dual_camera").exists()
        and
        (p / "config").exists()
    ):
        ROOT = p
        break

if ROOT is None:
    raise RuntimeError(
        "Cannot locate r1a7_threading_project"
    )


HANDEYE_DIR = (
    ROOT /
    "vision_dual_camera" /
    "handeye"
)

sys.path.insert(
    0,
    str(HANDEYE_DIR)
)

import collect_handeye_samples as hc


CALIB_FILE = (
    HANDEYE_DIR /
    "r1a7_right_wrist_handeye.json"
)

OUTPUT_FILE = Path(
    "/tmp/r1a7_right_wrist_world_anchor.json"
)


# ============================================================
# Camera calibration already verified in project
# ============================================================

K = hc.CAMERA_MATRIX.copy()
D = hc.DIST_COEFFS.copy()

RIGHT_WRIST_SN = hc.RIGHT_WRIST_SN


# ============================================================
# Load Hand-Eye
# ============================================================

with CALIB_FILE.open(
    "r",
    encoding="utf-8"
) as f:

    calib = json.load(f)


T_EE_CAMERA = np.asarray(
    calib["T_ee_camera"],
    dtype=np.float64
)


# ============================================================
# Depth
# ============================================================

def depth_to_meters(
    depth_frame
):

    data = np.frombuffer(
        depth_frame.get_data(),
        dtype=np.uint16
    )

    h = int(
        depth_frame.get_height()
    )

    w = int(
        depth_frame.get_width()
    )

    if data.size != h * w:
        return None

    depth = (
        data.reshape(
            h,
            w
        )
        .astype(
            np.float32
        )
    )

    scale = float(
        depth_frame.get_depth_scale()
    )

    depth *= scale

    valid = depth[
        depth > 0
    ]

    # Some SDK builds expose mm-like values.
    if (
        valid.size
        and
        float(
            np.median(
                valid
            )
        )
        >
        10.0
    ):
        depth *= 0.001

    return depth


def robust_depth(
    depth,
    u,
    v
):

    """
    Thin wire itself may have invalid depth.

    For pre-grasp stage the free end is essentially on the
    tabletop, so estimate Z from the surrounding tabletop
    annulus rather than trusting one wire pixel.
    """

    h, w = depth.shape[:2]

    values = []

    for dy in range(
        -14,
        15
    ):

        for dx in range(
            -14,
            15
        ):

            r = (
                dx * dx
                +
                dy * dy
            )

            # Ignore very center pixels containing thin wire.
            if (
                r < 6 * 6
                or
                r > 14 * 14
            ):
                continue

            x = int(u + dx)
            y = int(v + dy)

            if (
                x < 0
                or
                y < 0
                or
                x >= w
                or
                y >= h
            ):
                continue

            z = float(
                depth[
                    y,
                    x
                ]
            )

            if (
                np.isfinite(z)
                and
                0.15 < z < 2.5
            ):
                values.append(
                    z
                )

    if len(values) < 20:
        return None

    values = np.asarray(
        values,
        dtype=np.float64
    )

    med = float(
        np.median(
            values
        )
    )

    mad = float(
        np.median(
            np.abs(
                values
                -
                med
            )
        )
    )

    if mad > 1e-6:

        good = (
            np.abs(
                values
                -
                med
            )
            <
            max(
                0.010,
                3.0 * mad
            )
        )

        values = values[
            good
        ]

    if len(values) < 10:
        return None

    return float(
        np.median(
            values
        )
    )


# ============================================================
# Pixel <-> camera
# ============================================================

def pixel_depth_to_camera(
    u,
    v,
    z
):

    uv = np.array(
        [
            [
                [
                    float(u),
                    float(v)
                ]
            ]
        ],
        dtype=np.float64
    )

    normalized = (
        cv2.undistortPoints(
            uv,
            K,
            D
        )
        .reshape(2)
    )

    x = (
        normalized[0]
        *
        z
    )

    y = (
        normalized[1]
        *
        z
    )

    return np.array(
        [
            x,
            y,
            z,
            1.0
        ],
        dtype=np.float64
    )


def camera_to_pixel(
    point_camera
):

    p = np.asarray(
        point_camera[:3],
        dtype=np.float64
    )

    if (
        not np.all(
            np.isfinite(p)
        )
        or
        p[2] <= 0.05
    ):
        return None

    image_points, _ = (
        cv2.projectPoints(
            p.reshape(
                1,
                1,
                3
            ),
            np.zeros(
                (3, 1),
                dtype=np.float64
            ),
            np.zeros(
                (3, 1),
                dtype=np.float64
            ),
            K,
            D
        )
    )

    uv = (
        image_points
        .reshape(2)
    )

    return (
        float(
            uv[0]
        ),
        float(
            uv[1]
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 72)
    print("R1-A7 RIGHT WRIST WORLD-ANCHOR PROBE")
    print("READ ONLY - NO LOWCMD")
    print("=" * 72)
    print(
        "Camera:",
        RIGHT_WRIST_SN
    )
    print(
        "Hand-eye:",
        CALIB_FILE
    )
    print()

    # --------------------------------------------------------
    # Robot state / FK
    # --------------------------------------------------------

    vs = hc.load_robot_module()

    print(
        "[ROBOT] DDS init..."
    )

    vs.ChannelFactoryInitialize(
        0,
        "enp6s0"
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
            vs.LowState_
        )
    )

    subscriber.Init(
        state_buffer.callback,
        10
    )

    deadline = (
        time.monotonic()
        +
        10.0
    )

    while (
        state_buffer.snapshot()
        is None
        and
        time.monotonic()
        <
        deadline
    ):

        time.sleep(
            0.05
        )

    if (
        state_buffer.snapshot()
        is None
    ):

        raise RuntimeError(
            "No rt/lowstate"
        )

    ik = vs.R1A7_ArmIK()

    print(
        "[ROBOT] FK ready."
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    from pyorbbecsdk import (
        AlignFilter,
        Config,
        OBFormat,
        OBFrameAggregateOutputMode,
        OBSensorType,
        OBStreamType,
        Pipeline,
    )

    context, device = (
        hc.find_orbbec_device(
            RIGHT_WRIST_SN
        )
    )

    pipeline = Pipeline(
        device
    )

    config = Config()

    color_profiles = (
        pipeline
        .get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
    )

    depth_profiles = (
        pipeline
        .get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )
    )

    color_profile = (
        color_profiles
        .get_video_stream_profile(
            hc.COLOR_WIDTH,
            hc.COLOR_HEIGHT,
            OBFormat.RGB,
            hc.COLOR_FPS
        )
    )

    depth_profile = (
        depth_profiles
        .get_default_video_stream_profile()
    )

    config.enable_stream(
        color_profile
    )

    config.enable_stream(
        depth_profile
    )

    try:

        config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode
            .FULL_FRAME_REQUIRE
        )

    except Exception:
        pass

    try:

        pipeline.enable_frame_sync()

    except Exception as exc:

        print(
            "[CAMERA] frame sync unavailable:",
            exc
        )

    pipeline.start(
        config
    )

    align_filter = AlignFilter(
        align_to_stream=
        OBStreamType.COLOR_STREAM
    )

    print(
        "[CAMERA] RGB-D started."
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    clicked = {
        "pending": None
    }

    anchor_base = None

    init_uv = None
    init_depth = None

    win = (
        "RIGHT WRIST WORLD ANCHOR"
    )

    cv2.namedWindow(
        win,
        cv2.WINDOW_NORMAL
    )

    def mouse_cb(
        event,
        x,
        y,
        flags,
        param
    ):

        if (
            event
            ==
            cv2.EVENT_LBUTTONDOWN
        ):

            clicked[
                "pending"
            ] = (
                int(x),
                int(y)
            )

    cv2.setMouseCallback(
        win,
        mouse_cb
    )

    def get_T_base_camera():

        state = (
            state_buffer.snapshot()
        )

        if state is None:
            return None, None

        (
            upper_q,
            arm_q,
            arm_dq,
            gripper_q,
            gripper_dq,
            mode_machine,
            received_at,
        ) = state

        age_ms = (
            1000.0
            *
            (
                time.monotonic()
                -
                received_at
            )
        )

        if (
            age_ms
            >
            100.0
        ):

            return None, age_ms

        try:

            _,
            T_base_ee = (
                vs.get_ee_poses(
                    ik,
                    arm_q
                )
            )

        except Exception:

            return None, age_ms

        T_base_camera = (
            np.asarray(
                T_base_ee,
                dtype=np.float64
            )
            @
            T_EE_CAMERA
        )

        return (
            T_base_camera,
            age_ms
        )

    try:

        print()
        print(
            "LEFT CLICK: select a STATIC point"
        )
        print(
            "Prefer the wire free tip on the tabletop."
        )
        print(
            "R: clear anchor"
        )
        print(
            "Q/ESC: quit"
        )
        print()

        while True:

            frames = (
                pipeline
                .wait_for_frames(
                    1000
                )
            )

            if frames is None:
                continue

            aligned = (
                align_filter.process(
                    frames
                )
            )

            if aligned is not None:
                frames = aligned

            color_frame = (
                frames.get_color_frame()
            )

            depth_frame = (
                frames.get_depth_frame()
            )

            if (
                color_frame is None
                or
                depth_frame is None
            ):
                continue

            frame = (
                hc.color_frame_to_bgr(
                    color_frame
                )
            )

            depth = (
                depth_to_meters(
                    depth_frame
                )
            )

            if depth is None:
                continue

            if (
                depth.shape[0]
                !=
                frame.shape[0]
                or
                depth.shape[1]
                !=
                frame.shape[1]
            ):

                cv2.putText(
                    frame,
                    (
                        "DEPTH ALIGN SIZE MISMATCH "
                        f"{depth.shape} vs "
                        f"{frame.shape[:2]}"
                    ),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

                cv2.imshow(
                    win,
                    frame
                )

                cv2.waitKey(1)

                continue

            T_base_camera, age_ms = (
                get_T_base_camera()
            )

            # ------------------------------------------------
            # New click -> world anchor
            # ------------------------------------------------

            if (
                clicked[
                    "pending"
                ]
                is not None
            ):

                u, v = (
                    clicked[
                        "pending"
                    ]
                )

                clicked[
                    "pending"
                ] = None

                z = robust_depth(
                    depth,
                    u,
                    v
                )

                if z is None:

                    print(
                        "[ANCHOR REJECT]"
                        " no valid local depth"
                    )

                elif (
                    T_base_camera
                    is None
                ):

                    print(
                        "[ANCHOR REJECT]"
                        " no valid robot pose"
                    )

                else:

                    point_camera = (
                        pixel_depth_to_camera(
                            u,
                            v,
                            z
                        )
                    )

                    anchor_base = (
                        T_base_camera
                        @
                        point_camera
                    )

                    init_uv = (
                        u,
                        v
                    )

                    init_depth = z

                    print()
                    print(
                        "[ANCHOR SET]"
                    )
                    print(
                        "uv =",
                        init_uv
                    )
                    print(
                        "depth =",
                        f"{z * 1000.0:.1f} mm"
                    )
                    print(
                        "base XYZ =",
                        (
                            anchor_base[
                                :3
                            ]
                            *
                            1000.0
                        ).round(
                            2
                        ).tolist()
                    )

            # ------------------------------------------------
            # Reproject fixed world point using CURRENT FK
            # ------------------------------------------------

            predicted_uv = None

            point_camera_now = None

            if (
                anchor_base is not None
                and
                T_base_camera is not None
            ):

                T_camera_base = (
                    np.linalg.inv(
                        T_base_camera
                    )
                )

                point_camera_now = (
                    T_camera_base
                    @
                    anchor_base
                )

                predicted_uv = (
                    camera_to_pixel(
                        point_camera_now
                    )
                )

            display = (
                frame.copy()
            )

            if predicted_uv is not None:

                pu = int(
                    round(
                        predicted_uv[0]
                    )
                )

                pv = int(
                    round(
                        predicted_uv[1]
                    )
                )

                if (
                    0 <= pu < display.shape[1]
                    and
                    0 <= pv < display.shape[0]
                ):

                    cv2.circle(
                        display,
                        (
                            pu,
                            pv
                        ),
                        9,
                        (0, 0, 255),
                        -1
                    )

                    cv2.drawMarker(
                        display,
                        (
                            pu,
                            pv
                        ),
                        (255, 255, 255),
                        cv2.MARKER_CROSS,
                        25,
                        2
                    )

                    cv2.putText(
                        display,
                        "WORLD PREDICT",
                        (
                            pu + 12,
                            pv - 12
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

            status = (
                "ANCHOR SET"
                if
                anchor_base is not None
                else
                "CLICK STATIC POINT"
            )

            cv2.putText(
                display,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (
                    (0, 255, 0)
                    if
                    anchor_base is not None
                    else
                    (0, 255, 255)
                ),
                2
            )

            if age_ms is not None:

                cv2.putText(
                    display,
                    f"LowState age={age_ms:.1f} ms",
                    (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2
                )

            if (
                point_camera_now
                is not None
            ):

                xyz = (
                    point_camera_now[
                        :3
                    ]
                    *
                    1000.0
                )

                cv2.putText(
                    display,
                    (
                        "pred camera XYZ="
                        f"{xyz[0]:+.1f},"
                        f"{xyz[1]:+.1f},"
                        f"{xyz[2]:+.1f} mm"
                    ),
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            output = {
                "enabled":
                    False,

                "camera_serial":
                    RIGHT_WRIST_SN,

                "timestamp":
                    time.time(),

                "anchor_valid":
                    anchor_base is not None,

                "init_uv":
                    (
                        list(init_uv)
                        if
                        init_uv is not None
                        else
                        None
                    ),

                "init_depth_m":
                    init_depth,

                "anchor_base_xyz_m":
                    (
                        anchor_base[
                            :3
                        ].tolist()
                        if
                        anchor_base is not None
                        else
                        None
                    ),

                "predicted_uv":
                    (
                        [
                            float(
                                predicted_uv[0]
                            ),
                            float(
                                predicted_uv[1]
                            )
                        ]
                        if
                        predicted_uv is not None
                        else
                        None
                    )
            }

            OUTPUT_FILE.write_text(
                json.dumps(
                    output,
                    indent=2
                )
            )

            cv2.imshow(
                win,
                display
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27
            ):
                break

            if key in (
                ord("r"),
                ord("R")
            ):

                anchor_base = None
                init_uv = None
                init_depth = None

                print(
                    "[ANCHOR] cleared"
                )

    finally:

        pipeline.stop()

        cv2.destroyAllWindows()

        print(
            "[STOP]"
        )


if __name__ == "__main__":
    main()
