#!/usr/bin/env python3

import json
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

CAMERA_DEVICE = "/dev/video16"
CAMERA_INDEX = 16

INTRINSICS_JSON = Path(
    "calibration/fixed_camera_intrinsics/results/"
    "usb_zoom_camera_intrinsics.json"
)

OUTPUT_JSON = Path(
    "calibration/tweezer_marker_tip_calibration.json"
)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30

ARUCO_ID = 20
MARKER_SIZE_M = 0.010

WINDOW = "ARUCO MARKER TO TIP CALIBRATION"

AXIS_LENGTH_M = 0.008

MIN_SAMPLES = 8

# The manual measurement already verified approximately:
# Marker X = 0 mm
# Marker Y = +85 mm
# Marker Z = -25 mm
#
# This is only used for visualization before calibration.
MANUAL_INITIAL_TIP_MARKER = np.asarray(
    [
        0.000,
        0.085,
        -0.025,
    ],
    dtype=np.float64,
)


# ============================================================
# Camera intrinsics
# ============================================================

def load_intrinsics():

    with open(
        INTRINSICS_JSON,
        "r",
    ) as f:

        data = json.load(f)

    calibration = data["calibration"]

    K = np.asarray(
        calibration["camera_matrix"],
        dtype=np.float64,
    )

    dist = np.asarray(
        calibration["dist_coeffs"],
        dtype=np.float64,
    ).reshape(-1, 1)

    print("[CAMERA]")
    print(
        " serial =",
        data["camera"]["serial"],
    )
    print(
        " size   =",
        calibration["image_width"],
        "x",
        calibration["image_height"],
    )
    print(
        " zoom   =",
        data["lens_settings"]["zoom_absolute"],
    )
    print(
        " focus  =",
        data["lens_settings"]["focus_absolute"],
    )

    return K, dist, data


# ============================================================
# Camera
# ============================================================

def open_camera():

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():

        cap.release()

        cap = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_V4L2,
        )

    if not cap.isOpened():

        raise RuntimeError(
            "Cannot open external camera"
        )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FRAME_FPS,
    )

    print(
        "[CAM] opened:",
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
        ),
    )

    return cap


# ============================================================
# ArUco model
# ============================================================

HALF = MARKER_SIZE_M / 2.0

OBJECT_POINTS = np.asarray(
    [
        [-HALF, +HALF, 0.0],
        [+HALF, +HALF, 0.0],
        [+HALF, -HALF, 0.0],
        [-HALF, -HALF, 0.0],
    ],
    dtype=np.float64,
)

ARUCO_DICT = (
    cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_5X5_50
    )
)

ARUCO_PARAMS = (
    cv2.aruco.DetectorParameters()
)

ARUCO_DETECTOR = (
    cv2.aruco.ArucoDetector(
        ARUCO_DICT,
        ARUCO_PARAMS,
    )
)


# ============================================================
# Detection
# ============================================================

def detect_marker(gray):

    corners, ids, _ = (
        ARUCO_DETECTOR.detectMarkers(
            gray
        )
    )

    if ids is None:
        return None

    ids = ids.reshape(-1)

    for i, marker_id in enumerate(ids):

        if int(marker_id) != ARUCO_ID:
            continue

        pts = np.asarray(
            corners[i],
            dtype=np.float32,
        ).reshape(4, 2)

        refined = pts.reshape(
            4,
            1,
            2,
        ).copy()

        cv2.cornerSubPix(
            gray,
            refined,
            (5, 5),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS
                |
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.01,
            ),
        )

        return (
            refined
            .reshape(4, 2)
            .astype(np.float64)
        )

    return None


# ============================================================
# Marker pose
# ============================================================

def solve_marker_pose(
    image_points,
    K,
    dist,
):

    result = cv2.solvePnPGeneric(
        OBJECT_POINTS,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not bool(result[0]):
        return None

    rvecs = result[1]
    tvecs = result[2]

    candidates = []

    for rvec, tvec in zip(
        rvecs,
        tvecs,
    ):

        rvec = np.asarray(
            rvec,
            dtype=np.float64,
        ).reshape(3, 1)

        tvec = np.asarray(
            tvec,
            dtype=np.float64,
        ).reshape(3, 1)

        if float(tvec[2, 0]) <= 0:
            continue

        projected, _ = (
            cv2.projectPoints(
                OBJECT_POINTS,
                rvec,
                tvec,
                K,
                dist,
            )
        )

        projected = projected.reshape(
            4,
            2,
        )

        error = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            projected
                            -
                            image_points
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )
        )

        candidates.append(
            (
                error,
                rvec,
                tvec,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0]
    )

    error, rvec, tvec = candidates[0]

    R, _ = cv2.Rodrigues(
        rvec
    )

    return {
        "R": R,
        "rvec": rvec,
        "tvec": tvec,
        "marker_reproj_px": error,
    }


# ============================================================
# Pixel ray
# ============================================================

def pixel_to_ray(
    tip_uv,
    K,
    dist,
):

    uv = np.asarray(
        [[tip_uv]],
        dtype=np.float64,
    )

    undistorted = cv2.undistortPoints(
        uv,
        K,
        dist,
    )

    x = float(
        undistorted[0, 0, 0]
    )

    y = float(
        undistorted[0, 0, 1]
    )

    ray = np.asarray(
        [x, y, 1.0],
        dtype=np.float64,
    )

    ray /= np.linalg.norm(
        ray
    )

    return ray


# ============================================================
# Marker -> fixed tip solve
# ============================================================

def solve_tip_marker(
    samples,
    K,
    dist,
):

    A_list = []
    b_list = []

    I = np.eye(
        3,
        dtype=np.float64,
    )

    for sample in samples:

        R = np.asarray(
            sample["R"],
            dtype=np.float64,
        ).reshape(3, 3)

        t = np.asarray(
            sample["tvec"],
            dtype=np.float64,
        ).reshape(3)

        tip_uv = np.asarray(
            sample["tip_uv"],
            dtype=np.float64,
        ).reshape(2)

        ray = pixel_to_ray(
            tip_uv,
            K,
            dist,
        )

        # R * P_tip_marker + t must lie on the
        # camera ray corresponding to the clicked tip.
        #
        # (I - dd^T)(R P + t) = 0
        P_perp = (
            I
            -
            np.outer(
                ray,
                ray,
            )
        )

        A_list.append(
            P_perp @ R
        )

        b_list.append(
            -P_perp @ t
        )

    A = np.vstack(
        A_list
    )

    b = np.concatenate(
        b_list
    )

    p_tip_marker, residuals, rank, singular_values = (
        np.linalg.lstsq(
            A,
            b,
            rcond=None,
        )
    )

    return {
        "p_tip_marker":
            p_tip_marker.reshape(3),

        "rank":
            int(rank),

        "singular_values":
            singular_values,
    }


# ============================================================
# Project fixed tip
# ============================================================

def project_tip(
    p_tip_marker,
    pose,
    K,
    dist,
):

    projected, _ = cv2.projectPoints(
        np.asarray(
            p_tip_marker,
            dtype=np.float64,
        ).reshape(1, 3),

        pose["rvec"],
        pose["tvec"],
        K,
        dist,
    )

    return projected.reshape(2)


# ============================================================
# Evaluate calibration
# ============================================================

def evaluate_solution(
    p_tip_marker,
    samples,
    K,
    dist,
):

    errors = []
    predictions = []

    for sample in samples:

        pose = {
            "rvec":
                np.asarray(
                    sample["rvec"],
                    dtype=np.float64,
                ).reshape(3, 1),

            "tvec":
                np.asarray(
                    sample["tvec"],
                    dtype=np.float64,
                ).reshape(3, 1),
        }

        predicted = project_tip(
            p_tip_marker,
            pose,
            K,
            dist,
        )

        gt = np.asarray(
            sample["tip_uv"],
            dtype=np.float64,
        )

        error = float(
            np.linalg.norm(
                predicted - gt
            )
        )

        predictions.append(
            predicted
        )

        errors.append(
            error
        )

    errors = np.asarray(
        errors,
        dtype=np.float64,
    )

    return {
        "errors":
            errors,

        "predictions":
            predictions,

        "mean":
            float(
                np.mean(errors)
            ),

        "rms":
            float(
                np.sqrt(
                    np.mean(
                        errors ** 2
                    )
                )
            ),

        "median":
            float(
                np.median(errors)
            ),

        "max":
            float(
                np.max(errors)
            ),
    }


# ============================================================
# Frozen-frame click
# ============================================================

def click_tip(
    frame,
    corners,
    pose,
    K,
    dist,
):

    win = "CLICK TRUE TWEEZER TIP"

    base = frame.copy()

    polygon = (
        corners
        .astype(np.int32)
        .reshape(-1, 1, 2)
    )

    cv2.polylines(
        base,
        [polygon],
        True,
        (0, 255, 0),
        2,
    )

    cv2.drawFrameAxes(
        base,
        K,
        dist,
        pose["rvec"],
        pose["tvec"],
        AXIS_LENGTH_M,
        2,
    )

    clicked = []

    def mouse_callback(
        event,
        x,
        y,
        flags,
        userdata,
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            clicked.clear()

            clicked.append(
                [
                    float(x),
                    float(y),
                ]
            )

    cv2.namedWindow(
        win,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        win,
        mouse_callback,
    )

    while True:

        shown = base.copy()

        cv2.putText(
            shown,
            "CLICK THE SAME PHYSICAL TWEEZER TIP",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            shown,
            "ENTER=save  R=redo  ESC=cancel",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        if clicked:

            u = int(
                round(
                    clicked[0][0]
                )
            )

            v = int(
                round(
                    clicked[0][1]
                )
            )

            cv2.drawMarker(
                shown,
                (u, v),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                22,
                2,
            )

            cv2.putText(
                shown,
                f"TIP ({u},{v})",
                (
                    u + 10,
                    v - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        cv2.imshow(
            win,
            shown,
        )

        key = cv2.waitKey(20) & 0xFF

        if (
            key in (13, 32)
            and clicked
        ):

            cv2.destroyWindow(
                win
            )

            return np.asarray(
                clicked[0],
                dtype=np.float64,
            )

        if key in (
            ord("r"),
            ord("R"),
        ):

            clicked.clear()

        if key == 27:

            cv2.destroyWindow(
                win
            )

            return None


# ============================================================
# Save
# ============================================================

def save_calibration(
    p_tip_marker,
    evaluation,
    samples,
    camera_data,
):

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "marker": {
            "dictionary":
                "DICT_5X5_50",

            "id":
                ARUCO_ID,

            "size_m":
                MARKER_SIZE_M,
        },

        "camera": {
            "name":
                camera_data[
                    "camera"
                ]["name"],

            "serial":
                camera_data[
                    "camera"
                ]["serial"],

            "width":
                FRAME_WIDTH,

            "height":
                FRAME_HEIGHT,

            "zoom_absolute":
                camera_data[
                    "lens_settings"
                ]["zoom_absolute"],

            "focus_absolute":
                camera_data[
                    "lens_settings"
                ]["focus_absolute"],
        },

        "manual_initial_tip_marker_mm":
            (
                MANUAL_INITIAL_TIP_MARKER
                *
                1000.0
            ).tolist(),

        "tip_marker_m":
            np.asarray(
                p_tip_marker
            ).tolist(),

        "tip_marker_mm":
            (
                np.asarray(
                    p_tip_marker
                )
                *
                1000.0
            ).tolist(),

        "sample_count":
            len(samples),

        "tip_reprojection_error_px": {
            "mean":
                evaluation["mean"],

            "rms":
                evaluation["rms"],

            "median":
                evaluation["median"],

            "max":
                evaluation["max"],
        },

        "samples": [],

        "generated_at":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
    }

    for i, sample in enumerate(samples):

        output["samples"].append(
            {
                "index":
                    i + 1,

                "tip_uv":
                    sample["tip_uv"],

                "rvec":
                    sample["rvec"],

                "tvec":
                    sample["tvec"],

                "marker_reproj_px":
                    sample[
                        "marker_reproj_px"
                    ],

                "tip_reproj_error_px":
                    float(
                        evaluation[
                            "errors"
                        ][i]
                    ),
            }
        )

    with open(
        OUTPUT_JSON,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        "[SAVE]",
        OUTPUT_JSON,
    )


# ============================================================
# Main
# ============================================================

def main():

    K, dist, camera_data = (
        load_intrinsics()
    )

    cap = open_camera()

    samples = []

    solved_tip = None

    print()
    print(
        "=============================================="
    )
    print(
        "ARUCO MARKER -> TWEEZER TIP CALIBRATION"
    )
    print(
        "=============================================="
    )
    print(
        "Marker = DICT_5X5_50 ID20"
    )
    print(
        "Marker size = 10 mm"
    )
    print(
        "Manual initial tip [mm] =",
        MANUAL_INITIAL_TIP_MARKER
        *
        1000.0,
    )
    print()
    print(
        "SPACE : capture pose + click true tip"
    )
    print(
        "F     : solve Marker -> Tip calibration"
    )
    print(
        "D     : delete last sample"
    )
    print(
        "R     : clear all samples"
    )
    print(
        "Q/ESC : quit"
    )
    print()

    try:

        while True:

            ok, frame = cap.read()

            if (
                not ok
                or
                frame is None
            ):
                continue

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            corners = detect_marker(
                gray
            )

            pose = None

            if corners is not None:

                pose = solve_marker_pose(
                    corners,
                    K,
                    dist,
                )

            display = frame.copy()

            if pose is not None:

                marker_polygon = (
                    corners
                    .astype(np.int32)
                    .reshape(-1, 1, 2)
                )

                cv2.polylines(
                    display,
                    [marker_polygon],
                    True,
                    (0, 255, 0),
                    2,
                )

                cv2.drawFrameAxes(
                    display,
                    K,
                    dist,
                    pose["rvec"],
                    pose["tvec"],
                    AXIS_LENGTH_M,
                    2,
                )

                cv2.putText(
                    display,
                    (
                        "ARUCO POSE OK "
                        f"reproj="
                        f"{pose['marker_reproj_px']:.3f}px"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 0),
                    2,
                )

                # Before final calibration, show the manual
                # [0, +85, -25] mm prediction.
                manual_uv = project_tip(
                    MANUAL_INITIAL_TIP_MARKER,
                    pose,
                    K,
                    dist,
                )

                mu = int(
                    round(
                        manual_uv[0]
                    )
                )

                mv = int(
                    round(
                        manual_uv[1]
                    )
                )

                cv2.drawMarker(
                    display,
                    (mu, mv),
                    (255, 0, 255),
                    cv2.MARKER_CROSS,
                    18,
                    1,
                )

                cv2.putText(
                    display,
                    "MANUAL",
                    (
                        mu + 8,
                        mv - 8,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 255),
                    1,
                )

                if solved_tip is not None:

                    predicted_uv = (
                        project_tip(
                            solved_tip,
                            pose,
                            K,
                            dist,
                        )
                    )

                    pu = int(
                        round(
                            predicted_uv[0]
                        )
                    )

                    pv = int(
                        round(
                            predicted_uv[1]
                        )
                    )

                    cv2.drawMarker(
                        display,
                        (pu, pv),
                        (0, 165, 255),
                        cv2.MARKER_CROSS,
                        26,
                        2,
                    )

                    cv2.putText(
                        display,
                        "CALIBRATED 3D TIP",
                        (
                            pu + 10,
                            pv - 10,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2,
                    )

            else:

                cv2.putText(
                    display,
                    "ARUCO POSE LOST",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                display,
                (
                    f"samples={len(samples)}  "
                    "SPACE=capture  F=solve"
                ),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                WINDOW,
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            if key in (
                ord("d"),
                ord("D"),
            ):

                if samples:

                    samples.pop()

                    solved_tip = None

                    print(
                        "[DELETE] sample count =",
                        len(samples),
                    )

            if key in (
                ord("r"),
                ord("R"),
            ):

                samples.clear()

                solved_tip = None

                print(
                    "[RESET] all samples cleared"
                )

            if key == 32:

                if pose is None:

                    print(
                        "[CAPTURE] ArUco pose unavailable"
                    )

                    continue

                frozen_frame = frame.copy()
                frozen_corners = corners.copy()

                frozen_pose = {
                    "R":
                        pose["R"].copy(),

                    "rvec":
                        pose["rvec"].copy(),

                    "tvec":
                        pose["tvec"].copy(),

                    "marker_reproj_px":
                        pose[
                            "marker_reproj_px"
                        ],
                }

                tip_uv = click_tip(
                    frozen_frame,
                    frozen_corners,
                    frozen_pose,
                    K,
                    dist,
                )

                if tip_uv is None:

                    print(
                        "[CAPTURE] cancelled"
                    )

                    continue

                sample = {
                    "R":
                        frozen_pose[
                            "R"
                        ].tolist(),

                    "rvec":
                        frozen_pose[
                            "rvec"
                        ].reshape(3).tolist(),

                    "tvec":
                        frozen_pose[
                            "tvec"
                        ].reshape(3).tolist(),

                    "tip_uv":
                        tip_uv.tolist(),

                    "marker_reproj_px":
                        float(
                            frozen_pose[
                                "marker_reproj_px"
                            ]
                        ),
                }

                samples.append(
                    sample
                )

                solved_tip = None

                print(
                    f"[CAPTURE] sample={len(samples)} "
                    f"tip=({tip_uv[0]:.1f},"
                    f"{tip_uv[1]:.1f}) "
                    f"marker_reproj="
                    f"{frozen_pose['marker_reproj_px']:.3f}px"
                )

            if key in (
                ord("f"),
                ord("F"),
            ):

                if len(samples) < MIN_SAMPLES:

                    print(
                        "[SOLVE] Need at least",
                        MIN_SAMPLES,
                        "samples. Current =",
                        len(samples),
                    )

                    continue

                solved = solve_tip_marker(
                    samples,
                    K,
                    dist,
                )

                p_tip_marker = solved[
                    "p_tip_marker"
                ]

                evaluation = (
                    evaluate_solution(
                        p_tip_marker,
                        samples,
                        K,
                        dist,
                    )
                )

                solved_tip = (
                    p_tip_marker.copy()
                )

                print()
                print(
                    "=============================================="
                )
                print(
                    "MARKER -> TIP CALIBRATION RESULT"
                )
                print(
                    "=============================================="
                )
                print(
                    "samples =",
                    len(samples),
                )
                print(
                    "rank =",
                    solved["rank"],
                )
                print()
                print(
                    "manual initial [mm] =",
                    MANUAL_INITIAL_TIP_MARKER
                    *
                    1000.0,
                )
                print(
                    "P_tip_marker [m]  =",
                    p_tip_marker,
                )
                print(
                    "P_tip_marker [mm] =",
                    p_tip_marker
                    *
                    1000.0,
                )
                print()
                print(
                    "tip reprojection error:"
                )
                print(
                    f" mean   = "
                    f"{evaluation['mean']:.3f} px"
                )
                print(
                    f" rms    = "
                    f"{evaluation['rms']:.3f} px"
                )
                print(
                    f" median = "
                    f"{evaluation['median']:.3f} px"
                )
                print(
                    f" max    = "
                    f"{evaluation['max']:.3f} px"
                )
                print()
                print(
                    "Per-sample errors:"
                )

                for i, error in enumerate(
                    evaluation["errors"]
                ):

                    print(
                        f" {i+1:02d}: "
                        f"{error:.3f} px"
                    )

                print(
                    "=============================================="
                )

                save_calibration(
                    p_tip_marker,
                    evaluation,
                    samples,
                    camera_data,
                )

                print()
                print(
                    "[LIVE] Orange cross = "
                    "CALIBRATED 3D TIP"
                )

    except KeyboardInterrupt:

        print(
            "\n[CALIB] Ctrl+C"
        )

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print(
            "[CALIB] stopped"
        )


if __name__ == "__main__":
    main()
