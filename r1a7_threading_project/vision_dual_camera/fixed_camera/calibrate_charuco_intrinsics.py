#!/usr/bin/env python3

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = Path(
    "calibration/fixed_camera_intrinsics/images"
)

OUTPUT_FILE = Path(
    "calibration/fixed_camera_intrinsics/results/"
    "usb_zoom_camera_intrinsics.json"
)

# ChArUco board:
# 6 x 8 squares -> (6-1)*(8-1) = 35 internal ChArUco corners
SQUARES_X = 6
SQUARES_Y = 8

SQUARE_LENGTH_M = 0.030
MARKER_LENGTH_M = 0.022

MIN_CHARUCO_CORNERS = 12


# ============================================================
# OpenCV / ChArUco setup
# ============================================================

print("OpenCV:", cv2.__version__)

if not hasattr(cv2, "aruco"):
    raise RuntimeError(
        "cv2.aruco is unavailable. "
        "opencv-contrib-python is required."
    )

aruco = cv2.aruco

dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_5X5_100
)

# Compatibility between OpenCV versions
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


# ============================================================
# Load images
# ============================================================

image_paths = sorted(
    IMAGE_DIR.glob("charuco_*.png")
)

print()
print("========================================")
print("USB ZOOM Camera Intrinsic Calibration")
print("========================================")
print("Image directory :", IMAGE_DIR)
print("Images found    :", len(image_paths))

if len(image_paths) < 15:
    raise RuntimeError(
        f"Only {len(image_paths)} images found. "
        "At least 15 usable images are required."
    )


all_charuco_corners = []
all_charuco_ids = []

used_images = []
corner_counts = []

image_size = None


# ============================================================
# Detect ChArUco corners
# ============================================================

for path in image_paths:

    image = cv2.imread(str(path))

    if image is None:
        print(
            f"[SKIP] {path.name}: cannot read"
        )
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    if image_size is None:
        image_size = (w, h)

    elif image_size != (w, h):
        print(
            f"[SKIP] {path.name}: "
            f"resolution {w}x{h} differs from "
            f"{image_size[0]}x{image_size[1]}"
        )
        continue

    marker_corners, marker_ids, rejected = (
        aruco.detectMarkers(
            gray,
            dictionary,
            parameters=detector_params
        )
    )

    if marker_ids is None or len(marker_ids) == 0:
        print(
            f"[SKIP] {path.name}: "
            "no ArUco markers"
        )
        continue

    result = aruco.interpolateCornersCharuco(
        markerCorners=marker_corners,
        markerIds=marker_ids,
        image=gray,
        board=board
    )

    if result is None:
        print(
            f"[SKIP] {path.name}: "
            "ChArUco interpolation failed"
        )
        continue

    retval = result[0]
    charuco_corners = result[1]
    charuco_ids = result[2]

    if (
        charuco_corners is None
        or charuco_ids is None
    ):
        print(
            f"[SKIP] {path.name}: "
            "no ChArUco corners"
        )
        continue

    count = len(charuco_ids)

    if count < MIN_CHARUCO_CORNERS:
        print(
            f"[SKIP] {path.name}: "
            f"only {count} corners"
        )
        continue

    # Refine ChArUco corners to sub-pixel precision
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

    all_charuco_corners.append(
        charuco_corners
    )

    all_charuco_ids.append(
        charuco_ids
    )

    used_images.append(
        path.name
    )

    corner_counts.append(
        int(count)
    )

    print(
        f"[USE ] {path.name:<20s} "
        f"corners = {count:2d}/35"
    )


# ============================================================
# Check dataset
# ============================================================

num_views = len(
    all_charuco_corners
)

print()
print("========================================")
print("Detection summary")
print("========================================")
print("Images found :", len(image_paths))
print("Usable views :", num_views)
print("Image size   :", image_size)

if num_views < 15:
    raise RuntimeError(
        f"Only {num_views} usable views. "
        "Collect more diverse ChArUco images."
    )


# ============================================================
# Intrinsic calibration
# ============================================================

criteria = (
    cv2.TERM_CRITERIA_EPS
    + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-9
)

flags = 0

per_view_errors = None

if hasattr(
    aruco,
    "calibrateCameraCharucoExtended"
):

    result = (
        aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=flags,
            criteria=criteria
        )
    )

    rms = float(result[0])
    camera_matrix = result[1]
    dist_coeffs = result[2]
    rvecs = result[3]
    tvecs = result[4]

    # OpenCV normally returns:
    # 0 ret
    # 1 cameraMatrix
    # 2 distCoeffs
    # 3 rvecs
    # 4 tvecs
    # 5 stdDeviationsIntrinsics
    # 6 stdDeviationsExtrinsics
    # 7 perViewErrors
    if len(result) >= 8:
        per_view_errors = np.asarray(
            result[7],
            dtype=np.float64
        ).reshape(-1)

else:

    result = aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
        criteria=criteria
    )

    rms = float(result[0])
    camera_matrix = result[1]
    dist_coeffs = result[2]
    rvecs = result[3]
    tvecs = result[4]


K = np.asarray(
    camera_matrix,
    dtype=np.float64
)

dist = np.asarray(
    dist_coeffs,
    dtype=np.float64
).reshape(-1)

fx = float(K[0, 0])
fy = float(K[1, 1])
cx = float(K[0, 2])
cy = float(K[1, 2])


# ============================================================
# Print result
# ============================================================

print()
print("========================================")
print("Calibration result")
print("========================================")

print(
    f"Usable views : {num_views}"
)

print(
    f"RMS          : {rms:.6f} px"
)

print()
print("Camera matrix K:")
print(
    np.array2string(
        K,
        precision=8,
        suppress_small=True
    )
)

print()
print("Distortion coefficients:")
print(
    np.array2string(
        dist,
        precision=10,
        suppress_small=True
    )
)

print()
print(
    f"fx = {fx:.6f}"
)
print(
    f"fy = {fy:.6f}"
)
print(
    f"cx = {cx:.6f}"
)
print(
    f"cy = {cy:.6f}"
)


# ============================================================
# Per-view errors
# ============================================================

per_view_list = []

if (
    per_view_errors is not None
    and len(per_view_errors) == num_views
):

    print()
    print("========================================")
    print("Per-view reprojection errors")
    print("========================================")

    for (
        name,
        count,
        error
    ) in zip(
        used_images,
        corner_counts,
        per_view_errors
    ):

        item = {
            "image": name,
            "charuco_corners": int(count),
            "error_px": float(error)
        }

        per_view_list.append(item)

    ranking = sorted(
        per_view_list,
        key=lambda x: x["error_px"],
        reverse=True
    )

    print()
    print("Worst 10 views:")

    for item in ranking[:10]:
        print(
            f"{item['image']:<20s} "
            f"corners={item['charuco_corners']:2d} "
            f"error={item['error_px']:.6f} px"
        )

    print()

    print(
        "Per-view mean:",
        f"{np.mean(per_view_errors):.6f} px"
    )

    print(
        "Per-view max :",
        f"{np.max(per_view_errors):.6f} px"
    )

    print(
        "Per-view min :",
        f"{np.min(per_view_errors):.6f} px"
    )


# ============================================================
# Save JSON
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

output = {
    "camera": {
        "name": "USB ZOOM Camera",
        "serial": "20685204b12d5283",
        "device_at_calibration": "/dev/video14"
    },

    "stream": {
        "width": int(image_size[0]),
        "height": int(image_size[1]),
        "pixel_format": "MJPG",
        "fps": 30
    },

    "lens_settings": {
        "zoom_absolute": 0,
        "focus_automatic_continuous": 0,
        "focus_absolute": 220
    },

    "board": {
        "type": "ChArUco",
        "dictionary": "DICT_5X5_100",
        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,
        "square_length_m": SQUARE_LENGTH_M,
        "marker_length_m": MARKER_LENGTH_M,
        "internal_charuco_corners": 35
    },

    "calibration": {
        "rms_reprojection_error_px": rms,

        "camera_matrix": (
            K.tolist()
        ),

        "dist_coeffs": (
            dist.tolist()
        ),

        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,

        "image_width": int(
            image_size[0]
        ),

        "image_height": int(
            image_size[1]
        ),

        "images_found": len(
            image_paths
        ),

        "usable_views": num_views
    },

    "used_images": [
        {
            "image": name,
            "charuco_corners": count
        }
        for name, count in zip(
            used_images,
            corner_counts
        )
    ],

    "generated_at": datetime.now().isoformat(
        timespec="seconds"
    )
}

if per_view_list:
    output[
        "per_view_reprojection_errors"
    ] = per_view_list

    output[
        "per_view_error_mean_px"
    ] = float(
        np.mean(
            per_view_errors
        )
    )

    output[
        "per_view_error_max_px"
    ] = float(
        np.max(
            per_view_errors
        )
    )

OUTPUT_FILE.write_text(
    json.dumps(
        output,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print()
print("========================================")
print("Saved calibration")
print("========================================")
print(OUTPUT_FILE)
