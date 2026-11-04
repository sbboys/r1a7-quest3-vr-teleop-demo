#!/usr/bin/env python3

import csv
import time
from datetime import datetime
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

ROOT = Path(
    "/home/robot/r1a7_datasets/wire_unet/right_wrist_open_001"
)

SESSION = datetime.now().strftime(
    "session_%Y%m%d_%H%M%S"
)

IMAGE_DIR = (
    ROOT
    / "images_raw"
    / SESSION
)

META_DIR = (
    ROOT
    / "metadata"
)

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

META_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = (
    META_DIR
    / f"{SESSION}.csv"
)

AUTO_INTERVAL_S = 0.50


def find_device(serial):

    ctx = Context()

    device_list = (
        ctx.query_devices()
    )

    count = (
        device_list.get_count()
    )

    print(
        "[INFO] Orbbec devices =",
        count
    )

    for i in range(count):

        device = (
            device_list.get_device_by_index(i)
        )

        info = (
            device.get_device_info()
        )

        try:
            sn = (
                info.get_serial_number()
            )
        except Exception:
            sn = ""

        print(
            f"[DEVICE {i}] serial={sn}"
        )

        if str(sn) == serial:

            print(
                "[OK] RIGHT WRIST FOUND:",
                serial
            )

            return device

    return None


def open_color_pipeline(device):

    pipeline = Pipeline(
        device
    )

    config = Config()

    profiles = (
        pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
    )

    profile = None

    # First try the resolution used by the current project.
    for fmt in (
        OBFormat.RGB,
        OBFormat.BGR,
        OBFormat.MJPG,
    ):

        try:

            profile = (
                profiles.get_video_stream_profile(
                    1280,
                    800,
                    fmt,
                    30,
                )
            )

            print(
                "[OK] requested COLOR profile:",
                "1280x800",
                fmt,
                "30fps"
            )

            break

        except Exception:
            profile = None


    # If the SDK/firmware exposes a different color format,
    # use its default COLOR profile instead of guessing a
    # /dev/video node.
    if profile is None:

        profile = (
            profiles.get_default_video_stream_profile()
        )

        print(
            "[INFO] using default COLOR profile"
        )


    config.enable_stream(
        profile
    )

    pipeline.start(
        config
    )

    return pipeline


def color_frame_to_bgr(
    frame,
):

    if frame is None:
        return None

    width = int(
        frame.get_width()
    )

    height = int(
        frame.get_height()
    )

    fmt = (
        frame.get_format()
    )

    data = np.frombuffer(
        frame.get_data(),
        dtype=np.uint8,
    )

    # RGB888
    if fmt == OBFormat.RGB:

        expected = (
            width
            *
            height
            *
            3
        )

        if data.size < expected:
            return None

        rgb = (
            data[:expected]
            .reshape(
                height,
                width,
                3,
            )
        )

        return cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )


    # BGR888
    if fmt == OBFormat.BGR:

        expected = (
            width
            *
            height
            *
            3
        )

        if data.size < expected:
            return None

        return (
            data[:expected]
            .reshape(
                height,
                width,
                3,
            )
            .copy()
        )


    # MJPEG/JPEG
    if fmt == OBFormat.MJPG:

        return cv2.imdecode(
            data,
            cv2.IMREAD_COLOR,
        )


    # Generic fallbacks.
    if data.size == (
        width
        *
        height
        *
        3
    ):

        image = data.reshape(
            height,
            width,
            3,
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )


    if data.size == (
        width
        *
        height
        *
        2
    ):

        yuyv = data.reshape(
            height,
            width,
            2,
        )

        return cv2.cvtColor(
            yuyv,
            cv2.COLOR_YUV2BGR_YUY2,
        )


    # Last attempt: compressed image.
    decoded = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    if decoded is not None:
        return decoded


    print(
        "[ERROR] unsupported COLOR frame:",
        "format=",
        fmt,
        "width=",
        width,
        "height=",
        height,
        "bytes=",
        data.size,
    )

    return None


device = find_device(
    SERIAL
)

if device is None:

    raise SystemExit(
        "ERROR: right wrist Orbbec not found"
    )


pipeline = open_color_pipeline(
    device
)


csv_file = CSV_PATH.open(
    "w",
    newline="",
)

writer = csv.writer(
    csv_file
)

writer.writerow(
    [
        "filename",
        "timestamp",
        "camera",
        "serial",
        "session",
        "capture_mode",
        "width",
        "height",
    ]
)


saved = 0
auto_mode = False
last_auto = 0.0

window = (
    "R1-A7 RIGHT WRIST OPEN DATA"
)

cv2.namedWindow(
    window,
    cv2.WINDOW_NORMAL,
)


def save_image(
    image,
    mode,
):

    global saved

    timestamp = (
        time.time()
    )

    stamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    filename = (
        f"right_wrist_open_"
        f"{saved:06d}_"
        f"{stamp}.png"
    )

    path = (
        IMAGE_DIR
        /
        filename
    )

    if not cv2.imwrite(
        str(path),
        image,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ],
    ):

        print(
            "[ERROR] save failed:",
            path
        )

        return


    h, w = (
        image.shape[:2]
    )

    writer.writerow(
        [
            filename,
            f"{timestamp:.6f}",
            "right_wrist",
            SERIAL,
            SESSION,
            mode,
            w,
            h,
        ]
    )

    csv_file.flush()

    saved += 1

    print(
        f"[SAVE] {saved:04d} "
        f"{mode} "
        f"{filename}",
        flush=True,
    )


print()
print(
    "=============================================="
)

print(
    "RIGHT WRIST OPEN-TWEEZER DATA COLLECTION"
)

print(
    "SPACE = save one frame"
)

print(
    "A     = AUTO 2 FPS ON/OFF"
)

print(
    "Q/ESC = quit"
)

print(
    "OUTPUT =",
    IMAGE_DIR
)

print(
    "=============================================="
)


try:

    while True:

        frames = (
            pipeline.wait_for_frames(
                100
            )
        )

        if frames is None:
            continue

        color_frame = (
            frames.get_color_frame()
        )

        image = color_frame_to_bgr(
            color_frame
        )

        if image is None:
            continue


        now = (
            time.monotonic()
        )

        if (
            auto_mode
            and
            now - last_auto
            >=
            AUTO_INTERVAL_S
        ):

            save_image(
                image,
                "auto",
            )

            last_auto = now


        display = (
            image.copy()
        )

        cv2.putText(
            display,
            (
                "RIGHT WRIST OPEN "
                f"| saved={saved} "
                f"| AUTO={'ON' if auto_mode else 'OFF'}"
            ),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "SPACE=save  A=auto  Q=quit",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            window,
            display,
        )

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == ord(" "):

            save_image(
                image,
                "manual",
            )


        elif key in (
            ord("a"),
            ord("A"),
        ):

            auto_mode = (
                not auto_mode
            )

            last_auto = 0.0

            print(
                "[AUTO]",
                "ON"
                if auto_mode
                else "OFF",
                flush=True,
            )


        elif key in (
            ord("q"),
            ord("Q"),
            27,
        ):

            break


finally:

    try:
        pipeline.stop()
    except Exception:
        pass

    csv_file.close()

    cv2.destroyAllWindows()


print()
print(
    "CAPTURE_FINISHED"
)

print(
    "saved =",
    saved
)

print(
    "images =",
    IMAGE_DIR
)

print(
    "metadata =",
    CSV_PATH
)
