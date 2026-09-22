#!/usr/bin/env python3

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime


DEVICE = "/dev/video14"

INTRINSIC_FILE = Path(
    "calibration/fixed_camera_intrinsics/results/"
    "usb_zoom_camera_intrinsics.json"
)

WORKPLANE_FILE = Path(
    "calibration/r1a7_workplane_base.json"
)

OUTPUT_FILE = Path(
    "calibration/fixed_camera_extrinsic/"
    "usb_zoom_camera_base_extrinsic.json"
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
    M = np.zeros((3, 3), dtype=np.float64)

    for R in rotations:
        M += R

    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt

    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt

    return Rm


def rotation_error_deg(R_ref, R):
    dR = R_ref.T @ R

    c = (np.trace(dR) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)

    return float(
        np.degrees(
            np.arccos(c)
        )
    )


# ============================================================
# Load camera intrinsics
# ============================================================

with INTRINSIC_FILE.open(
    "r",
    encoding="utf-8"
) as f:
    intrinsic_data = json.load(f)

calib = intrinsic_data["calibration"]

K = np.asarray(
    calib["camera_matrix"],
    dtype=np.float64
)

dist = np.asarray(
    calib["dist_coeffs"],
    dtype=np.float64
).reshape(-1, 1)


# ============================================================
# Load T_base_board
# ============================================================

with WORKPLANE_FILE.open(
    "r",
    encoding="utf-8"
) as f:
    workplane_data = json.load(f)

T_base_board = np.asarray(
    workplane_data["T_base_board"],
    dtype=np.float64
)

if T_base_board.shape != (4, 4):
    raise RuntimeError(
        "T_base_board is not 4x4"
    )


print()
print("========================================")
print("Loaded calibration")
print("========================================")

print()
print("T_base_board:")
print(T_base_board)

print()
print("K:")
print(K)

print()
print("dist:")
print(dist.reshape(-1))


# ============================================================
# ChArUco board
# ============================================================

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


# ============================================================
# Open camera
# ============================================================

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

print()
print("========================================")
print("Fixed Camera -> Base Extrinsic")
print("========================================")
print("DO NOT move camera.")
print("DO NOT move ChArUco board.")
print()
print("SPACE = record sample")
print("q     = finish")
print()


samples = []


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

                object_points = board_points[
                    charuco_ids
                ].reshape(-1, 3)

                image_points = (
                    charuco_corners.reshape(-1, 2)
                )

                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
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

                    # Board -> Camera is known.
                    # Camera -> Base:
                    #
                    # T_base_camera =
                    # T_base_board @ inv(T_camera_board)

                    T_base_camera = (
                        T_base_board
                        @ np.linalg.inv(
                            T_camera_board
                        )
                    )

                    projected, _ = cv2.projectPoints(
                        object_points,
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
                        projected - image_points,
                        axis=1
                    )

                    reproj_rms = float(
                        np.sqrt(
                            np.mean(
                                errors ** 2
                            )
                        )
                    )

                    current = {
                        "corners": int(count),
                        "reproj_rms_px": reproj_rms,
                        "T_camera_board":
                            T_camera_board.copy(),
                        "T_base_camera":
                            T_base_camera.copy()
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
        0.8,
        (0, 255, 0)
            if count >= MIN_CORNERS
            else (0, 0, 255),
        2
    )

    if current is not None:
        cv2.putText(
            display,
            (
                "Reproj RMS: "
                f"{current['reproj_rms_px']:.3f} px"
            ),
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

    cv2.putText(
        display,
        f"Samples: {len(samples)}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        "SPACE=record  q=finish",
        (20, HEIGHT - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Fixed Camera Extrinsic Calibration",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:

        if current is None:
            print(
                "[SKIP] Invalid ChArUco pose"
            )
            continue

        samples.append(current)

        xyz_mm = (
            current["T_base_camera"][
                :3, 3
            ]
            * 1000.0
        )

        print(
            f"[SAMPLE {len(samples):02d}] "
            f"corners={current['corners']:2d} "
            f"reproj="
            f"{current['reproj_rms_px']:.4f} px  "
            f"camera Base XYZ="
            f"[{xyz_mm[0]:+.2f}, "
            f"{xyz_mm[1]:+.2f}, "
            f"{xyz_mm[2]:+.2f}] mm"
        )


cap.release()
cv2.destroyAllWindows()


# ============================================================
# Compute final mean
# ============================================================

if len(samples) < 10:
    raise RuntimeError(
        f"Only {len(samples)} samples. "
        "Collect at least 10; "
        "20-30 is recommended."
    )

translations = np.stack(
    [
        s["T_base_camera"][:3, 3]
        for s in samples
    ],
    axis=0
)

rotations = [
    s["T_base_camera"][:3, :3]
    for s in samples
]

mean_t = translations.mean(
    axis=0
)

mean_R = rotation_mean(
    rotations
)

T_base_camera_mean = np.eye(
    4,
    dtype=np.float64
)

T_base_camera_mean[
    :3, :3
] = mean_R

T_base_camera_mean[
    :3, 3
] = mean_t


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
        rotation_error_deg(
            mean_R,
            R
        )
        for R in rotations
    ]
)

reprojection_errors = np.asarray(
    [
        s["reproj_rms_px"]
        for s in samples
    ]
)


position_rms_mm = float(
    np.sqrt(
        np.mean(
            position_errors_mm ** 2
        )
    )
)

rotation_rms_deg = float(
    np.sqrt(
        np.mean(
            rotation_errors_deg ** 2
        )
    )
)


print()
print("========================================")
print("FINAL EXTRINSIC RESULT")
print("========================================")

print()
print("T_base_camera:")
print(
    np.array2string(
        T_base_camera_mean,
        precision=9,
        suppress_small=True
    )
)

print()
print(
    "Camera Base XYZ [mm]:",
    np.round(
        mean_t * 1000.0,
        3
    )
)

print()
print(
    f"Position RMS : "
    f"{position_rms_mm:.4f} mm"
)

print(
    f"Position MAX : "
    f"{position_errors_mm.max():.4f} mm"
)

print(
    f"Rotation RMS : "
    f"{rotation_rms_deg:.4f} deg"
)

print(
    f"Rotation MAX : "
    f"{rotation_errors_deg.max():.4f} deg"
)

print(
    f"Reproj mean  : "
    f"{reprojection_errors.mean():.4f} px"
)

print(
    f"Reproj max   : "
    f"{reprojection_errors.max():.4f} px"
)


# ============================================================
# Save result
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

output_samples = []

for i, s in enumerate(
    samples,
    start=1
):
    output_samples.append({
        "sample_index": i,
        "corners": s["corners"],
        "reproj_rms_px":
            s["reproj_rms_px"],
        "T_camera_board":
            s["T_camera_board"].tolist(),
        "T_base_camera":
            s["T_base_camera"].tolist()
    })


output = {
    "camera": {
        "name": "USB ZOOM Camera",
        "serial": "20685204b12d5283",
        "device_at_calibration":
            "/dev/video14"
    },

    "stream": {
        "width": WIDTH,
        "height": HEIGHT,
        "pixel_format": "MJPG",
        "fps": FPS
    },

    "lens_settings": {
        "zoom_absolute": 0,
        "focus_automatic_continuous": 0,
        "focus_absolute": 220
    },

    "frame_convention":
        "T_base_camera maps "
        "fixed-camera coordinates "
        "into R1-A7 Base coordinates",

    "T_base_camera":
        T_base_camera_mean.tolist(),

    "translation_base_m":
        mean_t.tolist(),

    "translation_base_mm":
        (
            mean_t * 1000.0
        ).tolist(),

    "validation": {
        "sample_count":
            len(samples),

        "position_rms_mm":
            position_rms_mm,

        "position_max_mm":
            float(
                position_errors_mm.max()
            ),

        "rotation_rms_deg":
            rotation_rms_deg,

        "rotation_max_deg":
            float(
                rotation_errors_deg.max()
            ),

        "reprojection_mean_px":
            float(
                reprojection_errors.mean()
            ),

        "reprojection_max_px":
            float(
                reprojection_errors.max()
            )
    },

    "source_T_base_board":
        T_base_board.tolist(),

    "intrinsic_file":
        str(INTRINSIC_FILE),

    "workplane_file":
        str(WORKPLANE_FILE),

    "samples":
        output_samples,

    "generated_at":
        datetime.now().isoformat(
            timespec="seconds"
        )
}

with OUTPUT_FILE.open(
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
print(OUTPUT_FILE)
