#!/usr/bin/env python3

import cv2
import json
import numpy as np
from pathlib import Path


DEVICE = "/dev/video14"

CALIB_FILE = Path(
    "calibration/fixed_camera_intrinsics/results/"
    "usb_zoom_camera_intrinsics.json"
)

WIDTH = 1280
HEIGHT = 720
FPS = 30

SQUARES_X = 6
SQUARES_Y = 8

SQUARE_LENGTH_M = 0.030
MARKER_LENGTH_M = 0.022


# ------------------------------------------------------------
# Load calibration
# ------------------------------------------------------------

with CALIB_FILE.open(
    "r",
    encoding="utf-8"
) as f:
    calib = json.load(f)

c = calib["calibration"]

K = np.asarray(
    c["camera_matrix"],
    dtype=np.float64
)

dist = np.asarray(
    c["dist_coeffs"],
    dtype=np.float64
).reshape(-1, 1)

print()
print("Loaded calibration:")
print(CALIB_FILE)

print()
print("K:")
print(K)

print()
print("dist:")
print(dist.reshape(-1))


# ------------------------------------------------------------
# ChArUco board
# ------------------------------------------------------------

aruco = cv2.aruco

dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_5X5_100
)

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
    detector_params = aruco.DetectorParameters_create()


# OpenCV 4.x compatibility
if hasattr(board, "getChessboardCorners"):
    board_points = np.asarray(
        board.getChessboardCorners(),
        dtype=np.float32
    )
else:
    board_points = np.asarray(
        board.chessboardCorners,
        dtype=np.float32
    )


# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------

cap = cv2.VideoCapture(
    DEVICE,
    cv2.CAP_V4L2
)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
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


sample_errors = []

print()
print("====================================")
print("Intrinsic independent validation")
print("====================================")
print("SPACE = record current validation")
print("q     = finish")
print()


while True:

    ok, frame = cap.read()

    if not ok:
        print("Frame read failed")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    marker_corners, marker_ids, _ = (
        aruco.detectMarkers(
            gray,
            dictionary,
            parameters=detector_params
        )
    )

    display = frame.copy()

    current_rms = None
    current_max = None
    charuco_count = 0

    if (
        marker_ids is not None
        and len(marker_ids) > 0
    ):

        aruco.drawDetectedMarkers(
            display,
            marker_corners,
            marker_ids
        )

        ret = aruco.interpolateCornersCharuco(
            markerCorners=marker_corners,
            markerIds=marker_ids,
            image=gray,
            board=board
        )

        if (
            ret is not None
            and ret[1] is not None
            and ret[2] is not None
        ):

            charuco_corners = ret[1]
            charuco_ids = ret[2].reshape(-1)

            charuco_count = len(
                charuco_ids
            )

            cv2.cornerSubPix(
                gray,
                charuco_corners,
                (3, 3),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS
                    + cv2.TERM_CRITERIA_MAX_ITER,
                    50,
                    0.001
                )
            )

            aruco.drawDetectedCornersCharuco(
                display,
                charuco_corners,
                charuco_ids.reshape(-1, 1),
                (0, 255, 0)
            )

            if charuco_count >= 12:

                obj_pts = board_points[
                    charuco_ids
                ].reshape(-1, 3)

                img_pts = charuco_corners.reshape(
                    -1,
                    2
                )

                success, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    K,
                    dist,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:

                    projected, _ = cv2.projectPoints(
                        obj_pts,
                        rvec,
                        tvec,
                        K,
                        dist
                    )

                    projected = projected.reshape(
                        -1,
                        2
                    )

                    errors = np.linalg.norm(
                        projected - img_pts,
                        axis=1
                    )

                    current_rms = float(
                        np.sqrt(
                            np.mean(
                                errors ** 2
                            )
                        )
                    )

                    current_max = float(
                        np.max(errors)
                    )

                    cv2.drawFrameAxes(
                        display,
                        K,
                        dist,
                        rvec,
                        tvec,
                        0.06
                    )


    cv2.putText(
        display,
        f"Charuco: {charuco_count}/35",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    if current_rms is not None:

        cv2.putText(
            display,
            f"Reproj RMS: {current_rms:.3f} px",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Reproj MAX: {current_max:.3f} px",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

    cv2.putText(
        display,
        f"Saved validation samples: {len(sample_errors)}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        "SPACE=record   q=quit",
        (20, HEIGHT - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "USB ZOOM Camera Intrinsic Validation",
        display
    )

    key = cv2.waitKey(1) & 0xff

    if key == ord("q"):
        break

    if key == 32:

        if current_rms is None:
            print(
                "[SKIP] No valid pose"
            )
            continue

        sample_errors.append(
            {
                "corners": charuco_count,
                "rms_px": current_rms,
                "max_px": current_max
            }
        )

        print(
            f"[VALIDATION {len(sample_errors):02d}] "
            f"corners={charuco_count:2d} "
            f"RMS={current_rms:.4f} px "
            f"MAX={current_max:.4f} px"
        )


cap.release()
cv2.destroyAllWindows()


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("====================================")
print("Validation summary")
print("====================================")

print(
    "Samples:",
    len(sample_errors)
)

if sample_errors:

    rms_values = np.asarray(
        [
            x["rms_px"]
            for x in sample_errors
        ],
        dtype=np.float64
    )

    max_values = np.asarray(
        [
            x["max_px"]
            for x in sample_errors
        ],
        dtype=np.float64
    )

    print(
        f"Mean RMS : "
        f"{np.mean(rms_values):.6f} px"
    )

    print(
        f"RMS max  : "
        f"{np.max(rms_values):.6f} px"
    )

    print(
        f"RMS min  : "
        f"{np.min(rms_values):.6f} px"
    )

    print(
        f"Point MAX: "
        f"{np.max(max_values):.6f} px"
    )
