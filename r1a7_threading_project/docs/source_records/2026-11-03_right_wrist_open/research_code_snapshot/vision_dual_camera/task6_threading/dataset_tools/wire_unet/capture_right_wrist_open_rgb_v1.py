#!/usr/bin/env python3

import csv
import glob
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2


SERIAL = "CPCBC53000C5"

ROOT = Path(
    "/home/robot/r1a7_datasets/wire_unet/right_wrist_open_001"
)

SESSION = datetime.now().strftime(
    "session_%Y%m%d_%H%M%S"
)

OUT_DIR = (
    ROOT
    / "images_raw"
    / SESSION
)

META_DIR = (
    ROOT
    / "metadata"
)

OUT_DIR.mkdir(
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


WIDTH = 1280
HEIGHT = 800
FPS = 30

# Auto collection rate.
AUTO_INTERVAL_S = 0.50


def udev_properties(
    dev,
):

    try:
        result = subprocess.run(
            [
                "udevadm",
                "info",
                "--query=property",
                f"--name={dev}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        return result.stdout

    except Exception:

        return ""


def find_candidates():

    override = os.environ.get(
        "RIGHT_WRIST_VIDEO"
    )

    if override:

        print(
            "[INFO] Manual device:",
            override,
        )

        return [
            override
        ]

    matched = []

    for dev in sorted(
        glob.glob(
            "/dev/video*"
        )
    ):

        props = udev_properties(
            dev
        )

        if SERIAL in props:

            matched.append(
                dev
            )

    return matched


def try_open(
    dev,
):

    print(
        f"[PROBE] {dev}"
    )

    cap = cv2.VideoCapture(
        dev,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():

        cap.release()

        print(
            f"[FAIL] cannot open {dev}"
        )

        return None


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FPS,
    )


    # Prefer MJPG for RGB UVC stream.
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
    )


    good_frame = None

    for _ in range(40):

        ok, frame = cap.read()

        if (
            ok
            and
            frame is not None
            and
            frame.ndim == 3
            and
            frame.shape[2] == 3
        ):

            good_frame = frame
            break


    if good_frame is None:

        cap.release()

        print(
            f"[FAIL] no 3-channel RGB frame from {dev}"
        )

        return None


    h, w = good_frame.shape[:2]

    print(
        f"[OK] {dev} "
        f"RGB {w}x{h}"
    )

    return cap


candidates = find_candidates()

print(
    "=============================================="
)

print(
    "R1-A7 RIGHT WRIST OPEN-TWEEZER RGB CAPTURE"
)

print(
    "SERIAL:",
    SERIAL,
)

print(
    "SESSION:",
    SESSION,
)

print(
    "OUTPUT:",
    OUT_DIR,
)

print(
    "=============================================="
)


if not candidates:

    print(
        "[ERROR] No /dev/video device matched serial",
        SERIAL,
    )

    print(
        "Run: v4l2-ctl --list-devices"
    )

    raise SystemExit(
        1
    )


print(
    "[INFO] candidates:",
    candidates,
)


cap = None
selected_device = None

for dev in candidates:

    test = try_open(
        dev
    )

    if test is not None:

        cap = test
        selected_device = dev
        break


if cap is None:

    print()
    print(
        "[ERROR] Serial was found but no RGB node could be opened."
    )

    print(
        "Candidates:",
        candidates,
    )

    print(
        "You can override with:"
    )

    print(
        "RIGHT_WRIST_VIDEO=/dev/videoX python3 capture_right_wrist_open_rgb_v1.py"
    )

    raise SystemExit(
        2
    )


csv_exists = CSV_PATH.exists()

csv_file = CSV_PATH.open(
    "a",
    newline="",
)

writer = csv.writer(
    csv_file
)

if not csv_exists:

    writer.writerow(
        [
            "filename",
            "timestamp",
            "session",
            "camera",
            "serial",
            "device",
            "width",
            "height",
            "capture_mode",
        ]
    )


saved_count = 0
auto_mode = False
last_auto_save = 0.0


def save_frame(
    frame,
    mode,
):

    global saved_count

    ts = time.time()

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"right_wrist_open_"
        f"{saved_count:06d}_"
        f"{stamp}.png"
    )

    path = (
        OUT_DIR
        / filename
    )

    ok = cv2.imwrite(
        str(path),
        frame,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ],
    )

    if not ok:

        print(
            "[ERROR] SAVE FAILED:",
            path,
        )

        return


    h, w = frame.shape[:2]

    writer.writerow(
        [
            filename,
            f"{ts:.6f}",
            SESSION,
            "right_wrist",
            SERIAL,
            selected_device,
            w,
            h,
            mode,
        ]
    )

    csv_file.flush()

    saved_count += 1

    print(
        f"[SAVE] #{saved_count:04d} "
        f"{mode} "
        f"{filename}"
    )


print()
print(
    "Controls:"
)

print(
    "  SPACE : save one image"
)

print(
    "  A     : toggle AUTO capture at 2 FPS"
)

print(
    "  Q/ESC : quit"
)

print()


window = (
    "R1-A7 Right Wrist OPEN Tweezer Data Capture"
)

cv2.namedWindow(
    window,
    cv2.WINDOW_NORMAL,
)


try:

    while True:

        ok, frame = cap.read()

        if (
            not ok
            or
            frame is None
        ):

            print(
                "[WARN] frame read failed"
            )

            time.sleep(
                0.02
            )

            continue


        now = time.monotonic()


        if (
            auto_mode
            and
            (
                now
                -
                last_auto_save
                >=
                AUTO_INTERVAL_S
            )
        ):

            save_frame(
                frame,
                "auto",
            )

            last_auto_save = now


        display = frame.copy()

        text1 = (
            f"RIGHT WRIST OPEN | "
            f"saved={saved_count} | "
            f"AUTO={'ON' if auto_mode else 'OFF'}"
        )

        cv2.putText(
            display,
            text1,
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "SPACE=save  A=auto  Q=quit",
            (25, 75),
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

            save_frame(
                frame,
                "manual",
            )


        elif key in (
            ord("a"),
            ord("A"),
        ):

            auto_mode = (
                not auto_mode
            )

            last_auto_save = 0.0

            print(
                "[AUTO]",
                "ON"
                if auto_mode
                else "OFF",
            )


        elif key in (
            ord("q"),
            ord("Q"),
            27,
        ):

            break


finally:

    cap.release()

    csv_file.close()

    cv2.destroyAllWindows()


print()
print(
    "=============================================="
)

print(
    "CAPTURE FINISHED"
)

print(
    "saved_count =",
    saved_count,
)

print(
    "images =",
    OUT_DIR,
)

print(
    "metadata =",
    CSV_PATH,
)

print(
    "=============================================="
)
