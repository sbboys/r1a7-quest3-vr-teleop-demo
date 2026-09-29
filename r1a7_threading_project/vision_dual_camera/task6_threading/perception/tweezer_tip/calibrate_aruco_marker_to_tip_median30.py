#!/usr/bin/env python3

import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


# ============================================================
# Configuration
# ============================================================

CAMERA_DEVICE = "/dev/video16"

INTRINSICS_JSON = Path(
    "calibration/fixed_camera_intrinsics/results/"
    "usb_zoom_camera_intrinsics.json"
)

OUTPUT_JSON = Path(
    "calibration/tweezer_marker_tip_calibration_median30.json"
)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30

ARUCO_ID = 20
MARKER_SIZE_M = 0.020

MEDIAN_FRAMES = 30
MIN_SAMPLES = 10

WINDOW = "MARKER TO TIP MEDIAN30 CALIBRATION"

# Hand measured initial estimate.
# This is NOT the final calibration result.
MANUAL_INITIAL_TIP_MARKER = np.asarray(
    [
        0.000,
        0.085,
        -0.025,
    ],
    dtype=np.float64,
)


# ============================================================
# Intrinsics
# ============================================================

def load_intrinsics():

    with open(INTRINSICS_JSON, "r") as f:
        data = json.load(f)

    cal = data["calibration"]

    K = np.asarray(
        cal["camera_matrix"],
        dtype=np.float64,
    )

    dist = np.asarray(
        cal["dist_coeffs"],
        dtype=np.float64,
    ).reshape(-1, 1)

    return K, dist, data


# ============================================================
# ArUco
# ============================================================

half = MARKER_SIZE_M / 2.0

OBJECT_POINTS = np.asarray(
    [
        [-half, +half, 0.0],
        [+half, +half, 0.0],
        [+half, -half, 0.0],
        [-half, -half, 0.0],
    ],
    dtype=np.float64,
)

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_5X5_50
)

ARUCO_PARAMS = cv2.aruco.DetectorParameters()

ARUCO_DETECTOR = cv2.aruco.ArucoDetector(
    ARUCO_DICT,
    ARUCO_PARAMS,
)


def detect_marker(gray):

    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(
        gray
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
# PnP candidates
# ============================================================

def solve_pose_candidates(
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
        return []

    candidates = []

    for branch_id, (rvec, tvec) in enumerate(
        zip(
            result[1],
            result[2],
        )
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

        projected, _ = cv2.projectPoints(
            OBJECT_POINTS,
            rvec,
            tvec,
            K,
            dist,
        )

        projected = projected.reshape(4, 2)

        reproj = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            projected
                            -
                            image_points
                        ) ** 2,
                        axis=1,
                    )
                )
            )
        )

        R, _ = cv2.Rodrigues(
            rvec
        )

        candidates.append(
            {
                "branch_id": int(branch_id),
                "rvec": rvec,
                "tvec": tvec,
                "R": R,
                "marker_reproj_px": reproj,
            }
        )

    return candidates


# ============================================================
# Projection
# ============================================================

def project_tip(
    p_marker,
    pose,
    K,
    dist,
):

    uv, _ = cv2.projectPoints(
        np.asarray(
            p_marker,
            dtype=np.float64,
        ).reshape(1, 3),
        pose["rvec"],
        pose["tvec"],
        K,
        dist,
    )

    return uv.reshape(2)


# ============================================================
# Collect 30 static frames
# ============================================================

def collect_median30(
    cap,
):

    corner_samples = []

    last_frame = None

    valid = 0
    attempts = 0
    max_attempts = 180

    print()
    print(
        "[MED30] collecting 30 valid frames..."
    )
    print(
        "[MED30] KEEP ROBOT COMPLETELY STILL"
    )

    while (
        valid < MEDIAN_FRAMES
        and
        attempts < max_attempts
    ):

        attempts += 1

        ok, frame = cap.read()

        if not ok:
            continue

        last_frame = frame.copy()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        corners = detect_marker(
            gray
        )

        shown = frame.copy()

        if corners is not None:

            corner_samples.append(
                corners.copy()
            )

            valid += 1

            cv2.polylines(
                shown,
                [
                    corners
                    .astype(np.int32)
                    .reshape(-1, 1, 2)
                ],
                True,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            shown,
            (
                f"MED30 CAPTURE "
                f"{valid}/{MEDIAN_FRAMES}"
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            shown,
            "DO NOT MOVE ROBOT",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 0, 255),
            2,
        )

        cv2.imshow(
            WINDOW,
            shown,
        )

        cv2.waitKey(1)

    if valid < MEDIAN_FRAMES:

        print(
            "[MED30] FAILED. valid=",
            valid,
        )

        return None, None

    stack = np.stack(
        corner_samples,
        axis=0,
    )

    median_corners = np.median(
        stack,
        axis=0,
    )

    print(
        "[MED30] completed:",
        valid,
        "frames"
    )

    return (
        median_corners,
        last_frame,
    )


# ============================================================
# Click tip
# ============================================================

def click_tip(
    frame,
    median_corners,
):

    win = "CLICK TRUE TIP"

    clicked = []

    base = frame.copy()

    cv2.polylines(
        base,
        [
            median_corners
            .astype(np.int32)
            .reshape(-1, 1, 2)
        ],
        True,
        (0, 255, 255),
        2,
    )

    def callback(
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
        callback,
    )

    while True:

        shown = base.copy()

        cv2.putText(
            shown,
            "CLICK THE SAME PHYSICAL TIP",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            shown,
            "ENTER=save   R=redo   ESC=cancel",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        if clicked:

            u = int(round(clicked[0][0]))
            v = int(round(clicked[0][1]))

            cv2.drawMarker(
                shown,
                (u, v),
                (0, 165, 255),
                cv2.MARKER_CROSS,
                24,
                2,
            )

        cv2.imshow(
            win,
            shown,
        )

        key = cv2.waitKey(20) & 0xFF

        if key == 13 and clicked:

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
# Select IPPE branch using measured physical prior
# ============================================================

def choose_pose_branch(
    candidates,
    clicked_tip,
    K,
    dist,
):

    if not candidates:
        return None

    scored = []

    for candidate in candidates:

        prior_uv = project_tip(
            MANUAL_INITIAL_TIP_MARKER,
            candidate,
            K,
            dist,
        )

        prior_tip_error = float(
            np.linalg.norm(
                prior_uv
                -
                clicked_tip
            )
        )

        candidate = dict(
            candidate
        )

        candidate[
            "manual_prior_tip_error_px"
        ] = prior_tip_error

        candidate[
            "manual_prior_uv"
        ] = prior_uv

        scored.append(
            candidate
        )

    scored.sort(
        key=lambda x:
        x[
            "manual_prior_tip_error_px"
        ]
    )

    print()
    print(
        "[IPPE] candidates:"
    )

    for c in scored:

        print(
            " branch=",
            c["branch_id"],
            " marker_reproj=",
            f"{c['marker_reproj_px']:.4f}px",
            " manual_tip_error=",
            f"{c['manual_prior_tip_error_px']:.2f}px",
        )

    chosen = scored[0]

    print(
        "[IPPE] selected branch =",
        chosen["branch_id"],
    )

    return chosen


# ============================================================
# Pixel optimization
# ============================================================

def residual_vector(
    p_marker,
    samples,
    K,
    dist,
):

    residuals = []

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

        pred = project_tip(
            p_marker,
            pose,
            K,
            dist,
        )

        gt = np.asarray(
            sample["tip_uv"],
            dtype=np.float64,
        )

        residuals.extend(
            [
                pred[0] - gt[0],
                pred[1] - gt[1],
            ]
        )

    return np.asarray(
        residuals,
        dtype=np.float64,
    )


def evaluate(
    p_marker,
    samples,
    K,
    dist,
):

    errors = []
    vectors = []

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

        pred = project_tip(
            p_marker,
            pose,
            K,
            dist,
        )

        gt = np.asarray(
            sample["tip_uv"],
            dtype=np.float64,
        )

        vec = pred - gt

        vectors.append(
            vec
        )

        errors.append(
            float(
                np.linalg.norm(
                    vec
                )
            )
        )

    errors = np.asarray(
        errors,
        dtype=np.float64,
    )

    vectors = np.asarray(
        vectors,
        dtype=np.float64,
    )

    return {
        "errors": errors,

        "mean":
            float(np.mean(errors)),

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
            float(np.max(errors)),

        "mean_du":
            float(
                np.mean(
                    vectors[:, 0]
                )
            ),

        "mean_dv":
            float(
                np.mean(
                    vectors[:, 1]
                )
            ),
    }


def solve_calibration(
    samples,
    K,
    dist,
):

    # Stage 1:
    # standard pixel least squares
    result_ls = least_squares(
        residual_vector,
        MANUAL_INITIAL_TIP_MARKER,
        args=(
            samples,
            K,
            dist,
        ),
        loss="linear",
        method="trf",
        x_scale="jac",
        max_nfev=5000,
    )

    p_ls = np.asarray(
        result_ls.x,
        dtype=np.float64,
    )

    # Stage 2:
    # robust refinement
    result_robust = least_squares(
        residual_vector,
        p_ls,
        args=(
            samples,
            K,
            dist,
        ),
        loss="soft_l1",
        f_scale=3.0,
        method="trf",
        x_scale="jac",
        max_nfev=5000,
    )

    p_robust = np.asarray(
        result_robust.x,
        dtype=np.float64,
    )

    return (
        p_ls,
        p_robust,
    )


def print_result(
    title,
    p_marker,
    ev,
):

    print()
    print(
        "=============================================="
    )
    print(title)
    print(
        "=============================================="
    )

    print(
        "P_tip_marker [mm] =",
        p_marker * 1000.0,
    )

    print(
        f"mean   = {ev['mean']:.3f} px"
    )

    print(
        f"rms    = {ev['rms']:.3f} px"
    )

    print(
        f"median = {ev['median']:.3f} px"
    )

    print(
        f"max    = {ev['max']:.3f} px"
    )

    print(
        f"mean du = {ev['mean_du']:+.3f} px"
    )

    print(
        f"mean dv = {ev['mean_dv']:+.3f} px"
    )

    print()
    print(
        "Per-sample errors:"
    )

    for i, err in enumerate(
        ev["errors"]
    ):

        print(
            f" {i+1:02d}: "
            f"{err:.3f} px"
        )


# ============================================================
# Save
# ============================================================

def save_result(
    p_ls,
    p_robust,
    ev_ls,
    ev_robust,
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

        "median_frames":
            MEDIAN_FRAMES,

        "manual_initial_tip_marker_mm":
            (
                MANUAL_INITIAL_TIP_MARKER
                *
                1000.0
            ).tolist(),

        "least_squares_tip_marker_mm":
            (
                p_ls * 1000.0
            ).tolist(),

        "robust_tip_marker_mm":
            (
                p_robust * 1000.0
            ).tolist(),

        "robust_tip_marker_m":
            p_robust.tolist(),

        "least_squares_error_px": {
            "mean": ev_ls["mean"],
            "rms": ev_ls["rms"],
            "median": ev_ls["median"],
            "max": ev_ls["max"],
        },

        "robust_error_px": {
            "mean": ev_robust["mean"],
            "rms": ev_robust["rms"],
            "median": ev_robust["median"],
            "max": ev_robust["max"],
        },

        "sample_count":
            len(samples),

        "samples":
            samples,

        "camera_serial":
            camera_data[
                "camera"
            ]["serial"],

        "generated_at":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
    }

    with open(
        OUTPUT_JSON,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
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

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2,
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

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open camera"
        )

    samples = []

    print()
    print(
        "=============================================="
    )
    print(
        "20 mm MARKER -> TIP MEDIAN30 CALIBRATION"
    )
    print(
        "=============================================="
    )

    print(
        "Marker = DICT_5X5_50 ID20"
    )

    print(
        "Marker size = 20 mm"
    )

    print(
        "Median frames per pose =",
        MEDIAN_FRAMES,
    )

    print()
    print(
        "SPACE = collect 30 frames + click Tip"
    )
    print(
        "F     = solve calibration"
    )
    print(
        "D     = delete last sample"
    )
    print(
        "R     = reset all"
    )
    print(
        "Q/ESC = quit"
    )
    print()

    while True:

        ok, frame = cap.read()

        if not ok:
            continue

        shown = frame.copy()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        corners = detect_marker(
            gray
        )

        if corners is not None:

            cv2.polylines(
                shown,
                [
                    corners
                    .astype(np.int32)
                    .reshape(-1, 1, 2)
                ],
                True,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                shown,
                "ARUCO OK",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

        else:

            cv2.putText(
                shown,
                "ARUCO LOST",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            shown,
            (
                f"samples={len(samples)} "
                "SPACE=capture F=solve"
            ),
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            WINDOW,
            shown,
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

                print(
                    "[DELETE] samples =",
                    len(samples),
                )

        if key in (
            ord("r"),
            ord("R"),
        ):

            samples.clear()

            print(
                "[RESET] all samples cleared"
            )

        if key == 32:

            print()
            print(
                "----------------------------------------------"
            )
            print(
                "[CAPTURE] Pose",
                len(samples) + 1
            )
            print(
                "Robot must remain completely still."
            )

            median_corners, frozen_frame = (
                collect_median30(
                    cap
                )
            )

            if median_corners is None:
                continue

            candidates = solve_pose_candidates(
                median_corners,
                K,
                dist,
            )

            if not candidates:

                print(
                    "[CAPTURE] PnP failed"
                )

                continue

            tip_uv = click_tip(
                frozen_frame,
                median_corners,
            )

            if tip_uv is None:

                print(
                    "[CAPTURE] cancelled"
                )

                continue

            pose = choose_pose_branch(
                candidates,
                tip_uv,
                K,
                dist,
            )

            if pose is None:
                continue

            sample = {
                "index":
                    len(samples) + 1,

                "tip_uv":
                    tip_uv.tolist(),

                "rvec":
                    pose[
                        "rvec"
                    ].reshape(3).tolist(),

                "tvec":
                    pose[
                        "tvec"
                    ].reshape(3).tolist(),

                "marker_reproj_px":
                    float(
                        pose[
                            "marker_reproj_px"
                        ]
                    ),

                "ippe_branch":
                    int(
                        pose[
                            "branch_id"
                        ]
                    ),

                "manual_prior_tip_error_px":
                    float(
                        pose[
                            "manual_prior_tip_error_px"
                        ]
                    ),
            }

            samples.append(
                sample
            )

            print(
                "[CAPTURE] saved sample =",
                len(samples),
            )

            print(
                "[CAPTURE] tip_uv =",
                tip_uv,
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

            p_ls, p_robust = solve_calibration(
                samples,
                K,
                dist,
            )

            ev_ls = evaluate(
                p_ls,
                samples,
                K,
                dist,
            )

            ev_robust = evaluate(
                p_robust,
                samples,
                K,
                dist,
            )

            print()
            print(
                "##############################################"
            )
            print(
                "MEDIAN30 MARKER -> TIP CALIBRATION RESULT"
            )
            print(
                "##############################################"
            )

            print(
                "samples =",
                len(samples),
            )

            print(
                "manual initial [mm] =",
                MANUAL_INITIAL_TIP_MARKER
                *
                1000.0,
            )

            print_result(
                "PIXEL LEAST SQUARES",
                p_ls,
                ev_ls,
            )

            print_result(
                "ROBUST PIXEL REFINEMENT",
                p_robust,
                ev_robust,
            )

            save_result(
                p_ls,
                p_robust,
                ev_ls,
                ev_robust,
                samples,
                camera_data,
            )

    cap.release()
    cv2.destroyAllWindows()

    print(
        "[CALIB] stopped"
    )


if __name__ == "__main__":
    main()
