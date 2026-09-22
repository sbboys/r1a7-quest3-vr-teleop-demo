#!/usr/bin/env python3

import cv2
import time
from pathlib import Path

DEVICE = "/dev/video14"

WIDTH = 1280
HEIGHT = 720
FPS = 30

SQUARES_X = 6
SQUARES_Y = 8

SQUARE_LENGTH_M = 0.030
MARKER_LENGTH_M = 0.022

SAVE_DIR = Path(
    "calibration/fixed_camera_intrinsics/images"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

aruco = cv2.aruco

dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_5X5_100
)

# OpenCV version compatibility
try:
    board = aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary
    )
except Exception:
    board = aruco.CharucoBoard_create(
        SQUARES_X,
        SQUARES_Y,
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary
    )

try:
    detector_params = aruco.DetectorParameters()
except Exception:
    detector_params = (
        aruco.DetectorParameters_create()
    )

cap = cv2.VideoCapture(
    DEVICE,
    cv2.CAP_V4L2
)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(
        *"MJPG"
    )
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    FPS
)

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open {DEVICE}"
    )

print()
print("USB ZOOM Camera opened.")
print(
    "Resolution:",
    int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    ),
    "x",
    int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )
)
print(
    "FPS:",
    cap.get(
        cv2.CAP_PROP_FPS
    )
)
print()
print(
    "SPACE = save image"
)
print(
    "q     = exit"
)
print()

saved_count = len(
    list(
        SAVE_DIR.glob(
            "charuco_*.png"
        )
    )
)

while True:

    ok, frame = cap.read()

    if not ok:
        print(
            "Failed to read frame."
        )
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    corners, ids, rejected = (
        aruco.detectMarkers(
            gray,
            dictionary,
            parameters=detector_params
        )
    )

    charuco_corners = None
    charuco_ids = None
    charuco_count = 0

    display = frame.copy()

    if ids is not None:

        aruco.drawDetectedMarkers(
            display,
            corners,
            ids
        )

        result = (
            aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board
            )
        )

        if result is not None:
            retval = result[0]
            charuco_corners = result[1]
            charuco_ids = result[2]

            if (
                charuco_corners is not None
                and charuco_ids is not None
            ):
                charuco_count = len(
                    charuco_ids
                )

                aruco.drawDetectedCornersCharuco(
                    display,
                    charuco_corners,
                    charuco_ids,
                    (0, 255, 0)
                )

    status = (
        f"Charuco corners: "
        f"{charuco_count}/35"
    )

    cv2.putText(
        display,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0)
        if charuco_count >= 15
        else (0, 0, 255),
        2
    )

    cv2.putText(
        display,
        f"Saved: {saved_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        "SPACE=save  q=quit",
        (20, HEIGHT - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Fixed Camera Charuco Intrinsic Capture",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:

        if charuco_count < 12:
            print(
                "[SKIP] Only "
                f"{charuco_count} "
                "Charuco corners."
            )
            continue

        saved_count += 1

        filename = SAVE_DIR / (
            f"charuco_"
            f"{saved_count:03d}.png"
        )

        cv2.imwrite(
            str(filename),
            frame
        )

        print(
            "[SAVE]",
            filename,
            "| corners =",
            charuco_count
        )

        time.sleep(0.15)

cap.release()
cv2.destroyAllWindows()

print()
print(
    f"Total images: {saved_count}"
)
print(
    "Saved in:",
    SAVE_DIR
)
