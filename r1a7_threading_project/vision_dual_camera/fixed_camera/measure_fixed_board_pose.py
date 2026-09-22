#!/usr/bin/env python3

import cv2
import json
import math
import numpy as np
from pathlib import Path
from datetime import datetime


DEVICE = "/dev/video14"

INTRINSIC_FILE = Path(
    "calibration/fixed_camera_intrinsics/results/"
    "usb_zoom_camera_intrinsics.json"
)

EXTRINSIC_FILE = Path(
    "calibration/fixed_camera_extrinsic/"
    "usb_zoom_camera_base_extrinsic_frozen_20260920.json"
)

OUTPUT_DIR = Path(
    "calibration/fixed_camera_cross_validation"
)

WIDTH = 1280
HEIGHT = 720
FPS = 30

SQUARES_X = 6
SQUARES_Y = 8
SQUARE_LENGTH_M = 0.030
MARKER_LENGTH_M = 0.022

MIN_CORNERS = 20


def rotation_mean(rotations):
    M = np.mean(
        np.stack(rotations, axis=0),
        axis=0
    )

    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def rotation_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)

    return math.degrees(
        math.acos(c)
    )


# ------------------------------------------------------------
# Load intrinsic
# ------------------------------------------------------------

with INTRINSIC_FILE.open(
    "r",
    encoding="utf-8"
) as f:
    intr = json.load(f)

K = np.asarray(
    intr["calibration"]["camera_matrix"],
    dtype=np.float64
)

dist = np.asarray(
    intr["calibration"]["dist_coeffs"],
    dtype=np.float64
).reshape(-1, 1)


# ------------------------------------------------------------
# Load frozen fixed-camera extrinsic
# ------------------------------------------------------------

with EXTRINSIC_FILE.open(
    "r",
    encoding="utf-8"
) as f:
    ext = json.load(f)

T_base_camera = np.asarray(
    ext["T_base_camera"],
    dtype=np.float64
)

if T_base_camera.shape != (4, 4):
    raise RuntimeError(
        "Invalid T_base_camera"
    )


print()
print("======================================")
print("Fixed Camera Board Pose Measurement")
print("======================================")

print()
print("Frozen T_base_camera:")
print(T_base_camera)


# ------------------------------------------------------------
# ChArUco
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
    params = aruco.DetectorParameters()
except Exception:
    params = aruco.DetectorParameters_create()

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
# Open camera
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


samples = []

print()
print("S = save current board pose")
print("Q = finish and calculate mean")
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
            parameters=params
        )
    )

    display = frame.copy()

    current = None
    count = 0

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

            count = len(charuco_ids)

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

            if count >= MIN_CORNERS:

                obj = board_points[
                    charuco_ids
                ].reshape(-1, 3)

                img = charuco_corners.reshape(
                    -1,
                    2
                )

                success, rvec, tvec = cv2.solvePnP(
                    obj,
                    img,
                    K,
                    dist,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:

                    R_camera_board, _ = (
                        cv2.Rodrigues(rvec)
                    )

                    T_camera_board = np.eye(
                        4,
                        dtype=np.float64
                    )

                    T_camera_board[
                        :3, :3
                    ] = R_camera_board

                    T_camera_board[
                        :3, 3
                    ] = tvec.reshape(3)

                    T_base_board = (
                        T_base_camera
                        @ T_camera_board
                    )

                    projected, _ = (
                        cv2.projectPoints(
                            obj,
                            rvec,
                            tvec,
                            K,
                            dist
                        )
                    )

                    projected = projected.reshape(
                        -1,
                        2
                    )

                    err = np.linalg.norm(
                        projected - img,
                        axis=1
                    )

                    rms = float(
                        np.sqrt(
                            np.mean(
                                err ** 2
                            )
                        )
                    )

                    current = {
                        "corners": int(count),
                        "rmse_px": rms,
                        "T_camera_board":
                            T_camera_board.copy(),
                        "T_base_board":
                            T_base_board.copy()
                    }

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
        f"Charuco: {count}/35",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0)
            if count >= MIN_CORNERS
            else (0, 0, 255),
        2
    )

    if current is not None:

        xyz = (
            current["T_base_board"][
                :3, 3
            ] * 1000.0
        )

        cv2.putText(
            display,
            f"RMS: {current['rmse_px']:.3f} px",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            (
                "Base XYZ: "
                f"{xyz[0]:+.1f}, "
                f"{xyz[1]:+.1f}, "
                f"{xyz[2]:+.1f} mm"
            ),
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

    cv2.putText(
        display,
        f"Saved: {len(samples)}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        "S=save   Q=finish",
        (20, HEIGHT - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Fixed Camera Board Pose",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key in (
        ord("q"),
        ord("Q"),
        27
    ):
        break

    if key in (
        ord("s"),
        ord("S")
    ):

        if current is None:
            print(
                "[SAVE REJECT] invalid pose"
            )
            continue

        samples.append(current)

        xyz = (
            current["T_base_board"][
                :3, 3
            ] * 1000.0
        )

        print(
            f"[SAMPLE {len(samples):02d}] "
            f"corners={current['corners']} "
            f"rmse={current['rmse_px']:.4f}px "
            f"Board Base XYZ="
            f"[{xyz[0]:+.3f}, "
            f"{xyz[1]:+.3f}, "
            f"{xyz[2]:+.3f}] mm"
        )


cap.release()
cv2.destroyAllWindows()


if len(samples) < 5:
    raise RuntimeError(
        "Need at least 5 samples."
    )


# ------------------------------------------------------------
# Average current board pose
# ------------------------------------------------------------

translations = np.stack(
    [
        s["T_base_board"][:3, 3]
        for s in samples
    ],
    axis=0
)

rotations = [
    s["T_base_board"][:3, :3]
    for s in samples
]

mean_t = translations.mean(
    axis=0
)

mean_R = rotation_mean(
    rotations
)

T_mean = np.eye(
    4,
    dtype=np.float64
)

T_mean[:3, :3] = mean_R
T_mean[:3, 3] = mean_t


position_errors_mm = np.asarray(
    [
        np.linalg.norm(
            t - mean_t
        ) * 1000.0
        for t in translations
    ]
)

rotation_errors_deg = np.asarray(
    [
        rotation_angle_deg(
            mean_R.T @ R
        )
        for R in rotations
    ]
)

rmse_values = np.asarray(
    [
        s["rmse_px"]
        for s in samples
    ]
)


print()
print("======================================")
print("FIXED CAMERA BOARD POSE RESULT")
print("======================================")

print()
print("T_base_board_mean:")
print(
    np.array2string(
        T_mean,
        precision=9,
        suppress_small=True
    )
)

print()
print(
    "Board Base XYZ [mm]:",
    np.round(
        mean_t * 1000.0,
        3
    )
)

print()
print(
    f"Repeat Position RMS : "
    f"{np.sqrt(np.mean(position_errors_mm**2)):.4f} mm"
)

print(
    f"Repeat Position MAX : "
    f"{position_errors_mm.max():.4f} mm"
)

print(
    f"Repeat Rotation RMS : "
    f"{np.sqrt(np.mean(rotation_errors_deg**2)):.4f} deg"
)

print(
    f"Repeat Rotation MAX : "
    f"{rotation_errors_deg.max():.4f} deg"
)

print(
    f"Reproj mean         : "
    f"{rmse_values.mean():.4f} px"
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_file = (
    OUTPUT_DIR
    / f"fixed_board_pose_{timestamp}.json"
)

output = {
    "source": "fixed_usb_camera",

    "intrinsic_file":
        str(INTRINSIC_FILE),

    "extrinsic_file":
        str(EXTRINSIC_FILE),

    "T_base_camera":
        T_base_camera.tolist(),

    "T_base_board_mean":
        T_mean.tolist(),

    "board_base_xyz_mm":
        (
            mean_t * 1000.0
        ).tolist(),

    "sample_count":
        len(samples),

    "repeat_position_rms_mm":
        float(
            np.sqrt(
                np.mean(
                    position_errors_mm ** 2
                )
            )
        ),

    "repeat_position_max_mm":
        float(
            position_errors_mm.max()
        ),

    "repeat_rotation_rms_deg":
        float(
            np.sqrt(
                np.mean(
                    rotation_errors_deg ** 2
                )
            )
        ),

    "repeat_rotation_max_deg":
        float(
            rotation_errors_deg.max()
        ),

    "reprojection_mean_px":
        float(
            rmse_values.mean()
        ),

    "samples": [
        {
            "corners":
                s["corners"],

            "rmse_px":
                s["rmse_px"],

            "T_camera_board":
                s["T_camera_board"].tolist(),

            "T_base_board":
                s["T_base_board"].tolist()
        }
        for s in samples
    ]
}

with output_file.open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )

print()
print("Saved:")
print(output_file)
