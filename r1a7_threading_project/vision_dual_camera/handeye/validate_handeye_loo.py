#!/usr/bin/env python3

import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


METHODS = {
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def load_samples():
    samples = []

    for path in sorted(DATA_DIR.glob("sample_*.json")):
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)

        samples.append({
            "id": int(d["sample_id"]),
            "T_be": np.asarray(
                d["robot"]["T_base_ee"],
                dtype=np.float64,
            ),
            "T_cb": np.asarray(
                d["vision"]["T_camera_board"],
                dtype=np.float64,
            ),
        })

    return samples


def solve(samples, method):
    R_g2b = []
    t_g2b = []
    R_t2c = []
    t_t2c = []

    for s in samples:
        T_be = s["T_be"]
        T_cb = s["T_cb"]

        R_g2b.append(T_be[:3, :3])
        t_g2b.append(T_be[:3, 3].reshape(3, 1))

        R_t2c.append(T_cb[:3, :3])
        t_t2c.append(T_cb[:3, 3].reshape(3, 1))

    R, t = cv2.calibrateHandEye(
        R_g2b,
        t_g2b,
        R_t2c,
        t_t2c,
        method=method,
    )

    T = np.eye(4)
    T[:3, :3] = np.asarray(R).reshape(3, 3)
    T[:3, 3] = np.asarray(t).reshape(3)

    return T


def rot_mean(rotations):
    M = np.mean(
        np.stack(rotations, axis=0),
        axis=0,
    )

    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def rot_error_deg(R_ref, R):
    R_err = R_ref.T @ R

    c = np.clip(
        (np.trace(R_err) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    return math.degrees(
        math.acos(c)
    )


def main():
    samples = load_samples()

    print("Samples:", len(samples))

    for name, method in METHODS.items():

        print()
        print("=" * 72)
        print(name)
        print("=" * 72)

        rows = []

        for i, holdout in enumerate(samples):

            train = [
                s for j, s in enumerate(samples)
                if j != i
            ]

            T_ec = solve(
                train,
                method,
            )

            train_board = [
                s["T_be"]
                @ T_ec
                @ s["T_cb"]
                for s in train
            ]

            mean_p = np.mean(
                np.array(
                    [T[:3, 3] for T in train_board]
                ),
                axis=0,
            )

            mean_R = rot_mean(
                [T[:3, :3] for T in train_board]
            )

            T_test_board = (
                holdout["T_be"]
                @ T_ec
                @ holdout["T_cb"]
            )

            pos_err_mm = (
                1000.0
                * np.linalg.norm(
                    T_test_board[:3, 3]
                    - mean_p
                )
            )

            rot_err_deg = rot_error_deg(
                mean_R,
                T_test_board[:3, :3],
            )

            rows.append(
                (
                    holdout["id"],
                    pos_err_mm,
                    rot_err_deg,
                )
            )

            print(
                f"sample_{holdout['id']:04d}  "
                f"pos={pos_err_mm:7.3f} mm  "
                f"rot={rot_err_deg:7.4f} deg"
            )

        p = np.array(
            [x[1] for x in rows]
        )

        r = np.array(
            [x[2] for x in rows]
        )

        worst_p = rows[int(np.argmax(p))]
        worst_r = rows[int(np.argmax(r))]

        print()
        print("SUMMARY")
        print(
            f"Position mean : {np.mean(p):.3f} mm"
        )
        print(
            f"Position RMS  : "
            f"{np.sqrt(np.mean(p**2)):.3f} mm"
        )
        print(
            f"Position max  : {np.max(p):.3f} mm "
            f"(sample_{worst_p[0]:04d})"
        )

        print(
            f"Rotation mean : {np.mean(r):.4f} deg"
        )
        print(
            f"Rotation RMS  : "
            f"{np.sqrt(np.mean(r**2)):.4f} deg"
        )
        print(
            f"Rotation max  : {np.max(r):.4f} deg "
            f"(sample_{worst_r[0]:04d})"
        )


if __name__ == "__main__":
    main()
