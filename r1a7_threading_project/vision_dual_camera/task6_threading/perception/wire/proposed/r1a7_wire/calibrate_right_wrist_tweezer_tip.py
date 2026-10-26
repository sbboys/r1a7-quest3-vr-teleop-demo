#!/usr/bin/env python3

import json
import time
from pathlib import Path

import cv2
import numpy as np

from pyorbbecsdk import (
    Context,
    Pipeline,
    Config,
    OBSensorType,
    OBFormat,
)


SERIAL = "CPCBC53000C5"

ROOT = Path(__file__).resolve().parents[6]

OUTPUT = (
    ROOT
    /
    "vision_dual_camera"
    /
    "task6_threading"
    /
    "wrist_tip_calibration.json"
)


def find_device():

    ctx = Context()

    devices = ctx.query_devices()

    for i in range(
        devices.get_count()
    ):

        dev = (
            devices.get_device_by_index(
                i
            )
        )

        info = (
            dev.get_device_info()
        )

        sn = str(
            info.get_serial_number()
        )

        print(
            f"[CAMERA] #{i} SN={sn}"
        )

        if sn == SERIAL:

            print(
                "[CAMERA] RIGHT WRIST:",
                sn
            )

            return ctx, dev

    raise RuntimeError(
        f"Cannot find {SERIAL}"
    )


def frame_to_bgr(
    frame
):

    w = int(
        frame.get_width()
    )

    h = int(
        frame.get_height()
    )

    raw = np.frombuffer(
        frame.get_data(),
        dtype=np.uint8
    )

    if raw.size != w * h * 3:
        return None

    rgb = raw.reshape(
        h,
        w,
        3
    )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )


def main():

    ctx, device = (
        find_device()
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

    profile = (
        profiles.get_video_stream_profile(
            1280,
            800,
            OBFormat.RGB,
            30
        )
    )

    config.enable_stream(
        profile
    )

    pipeline.start(
        config
    )

    selected = {
        "uv": None
    }

    win = (
        "CALIBRATE RIGHT WRIST TWEEZER TIP"
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

            selected["uv"] = (
                int(x),
                int(y)
            )

            print(
                "[SELECT]",
                selected["uv"]
            )

    cv2.setMouseCallback(
        win,
        mouse_cb
    )

    try:

        print()
        print("=" * 70)
        print("RIGHT WRIST TWEEZER PIXEL CALIBRATION")
        print("=" * 70)
        print(
            "LEFT CLICK = select grasp-center between the two tweezer tips"
        )
        print(
            "ENTER      = save"
        )
        print(
            "R          = clear"
        )
        print(
            "Q/ESC      = quit without save"
        )
        print()

        while True:

            frames = (
                pipeline.wait_for_frames(
                    1000
                )
            )

            if frames is None:
                continue

            color = (
                frames.get_color_frame()
            )

            if color is None:
                continue

            frame = (
                frame_to_bgr(
                    color
                )
            )

            if frame is None:
                continue

            vis = frame.copy()

            if (
                selected["uv"]
                is not None
            ):

                u, v = (
                    selected["uv"]
                )

                cv2.drawMarker(
                    vis,
                    (
                        u,
                        v
                    ),
                    (0, 255, 255),
                    cv2.MARKER_CROSS,
                    26,
                    2
                )

                cv2.circle(
                    vis,
                    (
                        u,
                        v
                    ),
                    5,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    vis,
                    f"TIP CENTER ({u},{v})",
                    (
                        u + 10,
                        v - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

            cv2.putText(
                vis,
                "CLICK BETWEEN THE TWO PHYSICAL TWEEZER TIPS",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2
            )

            cv2.putText(
                vis,
                "ENTER=SAVE   R=CLEAR   Q=QUIT",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                win,
                vis
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

                selected["uv"] = None

                print(
                    "[CLEAR]"
                )

            if key in (
                10,
                13
            ):

                if (
                    selected["uv"]
                    is None
                ):

                    print(
                        "[SAVE] click tip center first"
                    )

                    continue

                u, v = (
                    selected["uv"]
                )

                data = {
                    "tip_pixel": [
                        int(u),
                        int(v)
                    ],

                    "camera":
                        SERIAL,

                    "definition":
                        "midpoint_between_two_physical_tweezer_tips",

                    "resolution": [
                        1280,
                        800
                    ],

                    "timestamp":
                        time.time()
                }

                OUTPUT.write_text(
                    json.dumps(
                        data,
                        indent=2
                    )
                )

                print()
                print(
                    "[SAVED]",
                    OUTPUT
                )

                print(
                    "[NEW FIXED TIP]",
                    (
                        u,
                        v
                    )
                )

                break

    finally:

        pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
