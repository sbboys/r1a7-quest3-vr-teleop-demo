#!/usr/bin/env python3

import json
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

TIP_FILE = (
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

    for i in range(devices.get_count()):

        dev = devices.get_device_by_index(i)

        sn = str(
            dev.get_device_info()
            .get_serial_number()
        )

        if sn == SERIAL:
            return ctx, dev

    raise RuntimeError(
        f"Cannot find camera {SERIAL}"
    )


def frame_to_bgr(frame):

    w = int(frame.get_width())
    h = int(frame.get_height())

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

    data = json.loads(
        TIP_FILE.read_text()
    )

    fixed_tip = tuple(
        int(v)
        for v in data["tip_pixel"]
    )

    print(
        "[FIXED TIP FROM FILE]",
        fixed_tip
    )

    ctx, device = find_device()

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

    win = "VERIFY FIXED TWEEZER TIP"

    cv2.namedWindow(
        win,
        cv2.WINDOW_NORMAL
    )

    try:

        while True:

            frames = pipeline.wait_for_frames(
                1000
            )

            if frames is None:
                continue

            color = frames.get_color_frame()

            if color is None:
                continue

            frame = frame_to_bgr(
                color
            )

            if frame is None:
                continue

            u, v = fixed_tip

            # Large fixed cross.
            cv2.drawMarker(
                frame,
                (u, v),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                35,
                3
            )

            cv2.circle(
                frame,
                (u, v),
                7,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                f"FIXED TIP ({u},{v})",
                (u + 12, v - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "NO SAM2 - RAW FIXED PIXEL ONLY",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                win,
                frame
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

    finally:

        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
