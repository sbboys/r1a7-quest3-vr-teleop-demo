#!/usr/bin/env python3

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


INPUT_JSON = Path(
    "calibration/tweezer_marker_tip_calibration.json"
)

OUTPUT_JSON = Path(
    "calibration/tweezer_marker_tip_calibration_refined.json"
)

INTRINSICS_JSON = Path(
    "calibration/fixed_camera_intrinsics/results/"
    "usb_zoom_camera_intrinsics.json"
)


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

    return K, dist


def load_calibration():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    samples = data["samples"]

    p0 = np.asarray(
        data["tip_marker_m"],
        dtype=np.float64,
    )

    return data, samples, p0


def project_point(
    p_marker,
    sample,
    K,
    dist,
):
    rvec = np.asarray(
        sample["rvec"],
        dtype=np.float64,
    ).reshape(3, 1)

    tvec = np.asarray(
        sample["tvec"],
        dtype=np.float64,
    ).reshape(3, 1)

    projected, _ = cv2.projectPoints(
        np.asarray(
            p_marker,
            dtype=np.float64,
        ).reshape(1, 3),
        rvec,
        tvec,
        K,
        dist,
    )

    return projected.reshape(2)


def residual_vector(
    p_marker,
    samples,
    K,
    dist,
):
    residuals = []

    for sample in samples:
        pred = project_point(
            p_marker,
            sample,
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
    predictions = []

    for sample in samples:
        pred = project_point(
            p_marker,
            sample,
            K,
            dist,
        )

        gt = np.asarray(
            sample["tip_uv"],
            dtype=np.float64,
        )

        vec = pred - gt
        err = float(
            np.linalg.norm(vec)
        )

        errors.append(err)
        vectors.append(vec)
        predictions.append(pred)

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
        "vectors": vectors,
        "predictions": predictions,
        "mean": float(np.mean(errors)),
        "rms": float(
            np.sqrt(
                np.mean(errors ** 2)
            )
        ),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
        "mean_du": float(
            np.mean(vectors[:, 0])
        ),
        "mean_dv": float(
            np.mean(vectors[:, 1])
        ),
    }


def print_result(
    title,
    p_marker,
    result,
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
        "P_tip_marker [m]  =",
        p_marker,
    )

    print(
        "P_tip_marker [mm] =",
        p_marker * 1000.0,
    )

    print()

    print(
        f"mean   = {result['mean']:.3f} px"
    )

    print(
        f"rms    = {result['rms']:.3f} px"
    )

    print(
        f"median = {result['median']:.3f} px"
    )

    print(
        f"max    = {result['max']:.3f} px"
    )

    print(
        f"mean du = {result['mean_du']:+.3f} px"
    )

    print(
        f"mean dv = {result['mean_dv']:+.3f} px"
    )

    print()
    print("Per-sample errors:")

    for i, err in enumerate(
        result["errors"]
    ):
        print(
            f" {i+1:02d}: {err:.3f} px"
        )


def main():
    K, dist = load_intrinsics()

    original_data, samples, p0 = (
        load_calibration()
    )

    print()
    print(
        "=============================================="
    )
    print(
        "MARKER -> TIP PIXEL DOMAIN REFINEMENT"
    )
    print(
        "=============================================="
    )

    print(
        "samples =",
        len(samples),
    )

    print(
        "initial P_tip_marker [mm] =",
        p0 * 1000.0,
    )

    before = evaluate(
        p0,
        samples,
        K,
        dist,
    )

    print_result(
        "BEFORE REFINEMENT",
        p0,
        before,
    )

    # --------------------------------------------------------
    # Ordinary image-pixel least squares
    # --------------------------------------------------------

    ls_result = least_squares(
        residual_vector,
        p0,
        args=(
            samples,
            K,
            dist,
        ),
        method="trf",
        loss="linear",
        x_scale="jac",
        max_nfev=5000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )

    p_ls = np.asarray(
        ls_result.x,
        dtype=np.float64,
    )

    eval_ls = evaluate(
        p_ls,
        samples,
        K,
        dist,
    )

    print_result(
        "PIXEL LEAST SQUARES",
        p_ls,
        eval_ls,
    )

    # --------------------------------------------------------
    # Robust refinement
    #
    # Large-error samples get less influence.
    # --------------------------------------------------------

    robust_result = least_squares(
        residual_vector,
        p_ls,
        args=(
            samples,
            K,
            dist,
        ),
        method="trf",
        loss="soft_l1",
        f_scale=4.0,
        x_scale="jac",
        max_nfev=5000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )

    p_robust = np.asarray(
        robust_result.x,
        dtype=np.float64,
    )

    eval_robust = evaluate(
        p_robust,
        samples,
        K,
        dist,
    )

    print_result(
        "ROBUST PIXEL REFINEMENT",
        p_robust,
        eval_robust,
    )

    output = dict(
        original_data
    )

    output[
        "pixel_refinement"
    ] = {
        "initial_tip_marker_m":
            p0.tolist(),

        "initial_tip_marker_mm":
            (
                p0 * 1000.0
            ).tolist(),

        "least_squares_tip_marker_m":
            p_ls.tolist(),

        "least_squares_tip_marker_mm":
            (
                p_ls * 1000.0
            ).tolist(),

        "least_squares_error_px": {
            "mean":
                eval_ls["mean"],
            "rms":
                eval_ls["rms"],
            "median":
                eval_ls["median"],
            "max":
                eval_ls["max"],
            "mean_du":
                eval_ls["mean_du"],
            "mean_dv":
                eval_ls["mean_dv"],
        },

        "robust_tip_marker_m":
            p_robust.tolist(),

        "robust_tip_marker_mm":
            (
                p_robust * 1000.0
            ).tolist(),

        "robust_error_px": {
            "mean":
                eval_robust["mean"],
            "rms":
                eval_robust["rms"],
            "median":
                eval_robust["median"],
            "max":
                eval_robust["max"],
            "mean_du":
                eval_robust["mean_du"],
            "mean_dv":
                eval_robust["mean_dv"],
        },
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
        "=============================================="
    )
    print(
        "[SAVE]",
        OUTPUT_JSON,
    )
    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()
