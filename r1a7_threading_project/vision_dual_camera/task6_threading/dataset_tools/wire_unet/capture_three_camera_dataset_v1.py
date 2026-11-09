#!/usr/bin/env python3

import argparse
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


# ============================================================
# CAMERA CONFIG
# ============================================================

EXTERNAL_DEV = (
    "/dev/v4l/by-id/"
    "usb-ALP_USB_ZOOM_Camera_20685204b12d5283-video-index0"
)

RIGHT_SERIAL = "CPCBC53000C5"
LEFT_SERIAL = "CPCBC530007Z"

EXT_W = 1280
EXT_H = 720
EXT_FPS = 30

ORB_W = 1280
ORB_H = 800
ORB_FPS = 30


# ============================================================
# EXTERNAL CAMERA
# ============================================================

def open_external():

    cap = cv2.VideoCapture(
        EXTERNAL_DEV,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open external camera: {EXTERNAL_DEV}"
        )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        EXT_W,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        EXT_H,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        EXT_FPS,
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    print(
        "[External] opened:",
        EXTERNAL_DEV,
    )

    return cap


# ============================================================
# ORBBEC
# ============================================================

def get_device_by_serial(
    ctx,
    serial,
):

    devices = ctx.query_devices()
    count = devices.get_count()

    print(
        "[Orbbec] device count =",
        count,
    )

    for i in range(count):

        dev = devices.get_device_by_index(i)
        info = dev.get_device_info()

        try:
            sn = info.get_serial_number()
        except Exception:
            sn = ""

        print(
            f"[Orbbec device {i}] serial={sn}"
        )

        if str(sn) == serial:
            return dev

    raise RuntimeError(
        f"Orbbec serial not found: {serial}"
    )


def open_orbbec(
    ctx,
    serial,
):

    device = get_device_by_serial(
        ctx,
        serial,
    )

    pipeline = Pipeline(
        device
    )

    config = Config()

    profiles = (
        pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
    )

    profile = (
        profiles.get_video_stream_profile(
            ORB_W,
            ORB_H,
            OBFormat.RGB,
            ORB_FPS,
        )
    )

    config.enable_stream(
        profile
    )

    pipeline.start(
        config
    )

    print(
        f"[Orbbec] started {serial} "
        f"{ORB_W}x{ORB_H}@{ORB_FPS}"
    )

    return pipeline


def read_orbbec(
    pipeline,
):

    frames = pipeline.wait_for_frames(
        100
    )

    if frames is None:
        return None

    color = frames.get_color_frame()

    if color is None:
        return None

    data = np.frombuffer(
        color.get_data(),
        dtype=np.uint8,
    )

    expected = (
        ORB_W
        *
        ORB_H
        *
        3
    )

    if data.size != expected:

        print(
            "[WARN] Orbbec RGB size:",
            data.size,
            "expected:",
            expected,
        )

        return None

    rgb = data.reshape(
        ORB_H,
        ORB_W,
        3,
    )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )


# ============================================================
# DISPLAY
# ============================================================

def preview_tile(
    image,
    label,
):

    vis = cv2.resize(
        image,
        (480, 300),
    )

    cv2.putText(
        vis,
        label,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return vis


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root directory",
    )

    parser.add_argument(
        "--auto-interval",
        type=float,
        default=0.0,
        help=(
            "0 = manual only; "
            ">0 enables automatic capture every N seconds"
        ),
    )

    args = parser.parse_args()

    root = Path(
        args.root
    )

    session = datetime.now().strftime(
        "session_%Y%m%d_%H%M%S"
    )

    session_root = (
        root
        /
        "images_raw"
        /
        session
    )

    ext_dir = (
        session_root
        /
        "external"
    )

    right_dir = (
        session_root
        /
        "right_wrist"
    )

    left_dir = (
        session_root
        /
        "left_wrist"
    )

    meta_dir = (
        root
        /
        "metadata"
    )

    for p in (
        ext_dir,
        right_dir,
        left_dir,
        meta_dir,
    ):
        p.mkdir(
            parents=True,
            exist_ok=True,
        )

    csv_path = (
        meta_dir
        /
        f"{session}.csv"
    )

    print("=" * 72)
    print("R1-A7 THREE CAMERA DATASET CAPTURE")
    print("ROOT    =", root)
    print("SESSION =", session)
    print()
    print("Controls:")
    print("  S = save one synchronized group")
    print("  A = toggle auto capture")
    print("  Q = quit")
    print("=" * 72)

    ctx = Context()

    external = None
    right = None
    left = None

    count = 0
    auto_enabled = (
        args.auto_interval
        >
        0.0
    )

    last_auto = 0.0

    try:

        external = open_external()

        right = open_orbbec(
            ctx,
            RIGHT_SERIAL,
        )

        left = open_orbbec(
            ctx,
            LEFT_SERIAL,
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "sample_id",
                "wall_timestamp",
                "monotonic_timestamp",
                "external_file",
                "right_wrist_file",
                "left_wrist_file",
                "external_width",
                "external_height",
                "right_width",
                "right_height",
                "left_width",
                "left_height",
            ])

            while True:

                ok_ext, frame_ext = (
                    external.read()
                )

                frame_right = read_orbbec(
                    right
                )

                frame_left = read_orbbec(
                    left
                )

                if (
                    not ok_ext
                    or
                    frame_ext is None
                    or
                    frame_right is None
                    or
                    frame_left is None
                ):

                    print(
                        "[WARN] incomplete camera group"
                    )

                    continue

                vis = np.hstack([
                    preview_tile(
                        frame_ext,
                        "EXTERNAL",
                    ),
                    preview_tile(
                        frame_right,
                        "RIGHT WRIST",
                    ),
                    preview_tile(
                        frame_left,
                        "LEFT WRIST",
                    ),
                ])

                cv2.putText(
                    vis,
                    (
                        f"samples={count} "
                        f"auto={int(auto_enabled)}"
                    ),
                    (14, 585),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    "R1-A7 THREE CAMERA DATASET CAPTURE",
                    vis,
                )

                key = (
                    cv2.waitKey(1)
                    &
                    0xFF
                )

                now = time.monotonic()

                save_now = False

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    break

                if key in (
                    ord("a"),
                    ord("A"),
                ):

                    auto_enabled = (
                        not auto_enabled
                    )

                    last_auto = now

                    print(
                        "[AUTO]",
                        "ON"
                        if auto_enabled
                        else "OFF",
                    )

                if key in (
                    ord("s"),
                    ord("S"),
                    32,
                ):
                    save_now = True

                if (
                    auto_enabled
                    and
                    args.auto_interval
                    >
                    0.0
                    and
                    (
                        now
                        -
                        last_auto
                    )
                    >=
                    args.auto_interval
                ):

                    save_now = True
                    last_auto = now

                if not save_now:
                    continue

                count += 1

                sample_id = (
                    f"{count:06d}"
                )

                ext_path = (
                    ext_dir
                    /
                    f"{sample_id}.png"
                )

                right_path = (
                    right_dir
                    /
                    f"{sample_id}.png"
                )

                left_path = (
                    left_dir
                    /
                    f"{sample_id}.png"
                )

                ok1 = cv2.imwrite(
                    str(ext_path),
                    frame_ext,
                )

                ok2 = cv2.imwrite(
                    str(right_path),
                    frame_right,
                )

                ok3 = cv2.imwrite(
                    str(left_path),
                    frame_left,
                )

                if not (
                    ok1
                    and
                    ok2
                    and
                    ok3
                ):
                    raise RuntimeError(
                        "cv2.imwrite failed"
                    )

                wall_ts = time.time()
                mono_ts = time.monotonic()

                writer.writerow([
                    sample_id,
                    f"{wall_ts:.6f}",
                    f"{mono_ts:.6f}",
                    str(ext_path),
                    str(right_path),
                    str(left_path),
                    frame_ext.shape[1],
                    frame_ext.shape[0],
                    frame_right.shape[1],
                    frame_right.shape[0],
                    frame_left.shape[1],
                    frame_left.shape[0],
                ])

                f.flush()

                print(
                    "[SAVED]",
                    sample_id,
                )

    finally:

        print(
            "\n[INFO] stopping cameras..."
        )

        if external is not None:
            external.release()

        if right is not None:
            try:
                right.stop()
            except Exception:
                pass

        if left is not None:
            try:
                left.stop()
            except Exception:
                pass

        cv2.destroyAllWindows()

        print(
            "[DONE] samples =",
            count,
        )

        print(
            "[DONE] metadata =",
            csv_path,
        )


if __name__ == "__main__":
    main()
