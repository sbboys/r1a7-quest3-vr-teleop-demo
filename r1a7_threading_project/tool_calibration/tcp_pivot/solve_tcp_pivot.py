#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = ROOT / "tcp_pivot_result.json"


def load_samples():
    paths = sorted(
        DATA_DIR.glob("sample_*.json")
    )

    if len(paths) < 4:
        raise RuntimeError(
            f"Need >= 4 samples, found {len(paths)}"
        )

    samples = []

    for path in paths:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            d = json.load(f)

        T = np.asarray(
            d["measurement"]["T_base_ee"],
            dtype=np.float64,
        )

        if T.shape != (4, 4):
            raise RuntimeError(
                f"{path.name}: invalid T_base_ee"
            )

        if not np.all(
            np.isfinite(T)
        ):
            raise RuntimeError(
                f"{path.name}: NaN/Inf"
            )

        samples.append(
            {
                "id": int(
                    d["sample_id"]
                ),
                "path": path,
                "T_base_ee": T,
                "data": d,
            }
        )

    return samples


def solve_pivot(samples):
    """
    For every sample:

        R_i * p_tip_ee + t_i = p_pivot_base

    Rearrange:

        [R_i  -I] [p_tip_ee    ] = -t_i
                  [p_pivot_base]

    Solve all samples by least squares.
    """

    A_rows = []
    b_rows = []

    for sample in samples:
        T = sample[
            "T_base_ee"
        ]

        R = T[:3, :3]
        t = T[:3, 3]

        A_rows.append(
            np.hstack(
                [
                    R,
                    -np.eye(3),
                ]
            )
        )

        b_rows.append(
            -t
        )

    A = np.vstack(
        A_rows
    )

    b = np.concatenate(
        b_rows
    )

    (
        x,
        residuals,
        rank,
        singular_values,
    ) = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    p_tip_ee = x[:3]
    p_pivot_base = x[3:]

    return {
        "A": A,
        "b": b,
        "x": x,
        "p_tip_ee": p_tip_ee,
        "p_pivot_base": p_pivot_base,
        "rank": int(rank),
        "singular_values":
            singular_values,
    }


def evaluate(
    samples,
    p_tip_ee,
    p_pivot_base,
):
    pivot_points = []
    rows = []

    for sample in samples:
        T = sample[
            "T_base_ee"
        ]

        R = T[:3, :3]
        t = T[:3, 3]

        p_tip_base = (
            R @ p_tip_ee
            + t
        )

        error = (
            p_tip_base
            - p_pivot_base
        )

        error_norm = float(
            np.linalg.norm(
                error
            )
        )

        pivot_points.append(
            p_tip_base
        )

        rows.append(
            {
                "sample_id":
                    sample["id"],

                "predicted_tip_base_m":
                    p_tip_base.tolist(),

                "error_xyz_mm":
                    (
                        1000.0
                        * error
                    ).tolist(),

                "error_norm_mm":
                    1000.0
                    * error_norm,
            }
        )

    pivot_points = np.asarray(
        pivot_points,
        dtype=np.float64,
    )

    errors_m = np.asarray(
        [
            np.asarray(
                r["error_xyz_mm"]
            )
            / 1000.0
            for r in rows
        ]
    )

    norms_m = np.linalg.norm(
        errors_m,
        axis=1,
    )

    xyz_std_mm = (
        1000.0
        * np.std(
            pivot_points,
            axis=0,
        )
    )

    rms_mm = float(
        1000.0
        * np.sqrt(
            np.mean(
                norms_m ** 2
            )
        )
    )

    mean_mm = float(
        1000.0
        * np.mean(
            norms_m
        )
    )

    max_index = int(
        np.argmax(
            norms_m
        )
    )

    max_mm = float(
        1000.0
        * norms_m[
            max_index
        ]
    )

    return {
        "rows": rows,

        "xyz_std_mm":
            xyz_std_mm,

        "position_mean_mm":
            mean_mm,

        "position_rms_mm":
            rms_mm,

        "position_max_mm":
            max_mm,

        "worst_sample":
            rows[
                max_index
            ]["sample_id"],
    }


def leave_one_out(
    samples,
    full_p_tip,
):
    rows = []

    for i, holdout in enumerate(
        samples
    ):
        train = [
            sample
            for j, sample
            in enumerate(samples)
            if j != i
        ]

        result = solve_pivot(
            train
        )

        p_tip = (
            result[
                "p_tip_ee"
            ]
        )

        p_pivot = (
            result[
                "p_pivot_base"
            ]
        )

        T = holdout[
            "T_base_ee"
        ]

        predicted = (
            T[:3, :3]
            @ p_tip
            + T[:3, 3]
        )

        holdout_error_m = float(
            np.linalg.norm(
                predicted
                - p_pivot
            )
        )

        tip_shift_m = float(
            np.linalg.norm(
                p_tip
                - full_p_tip
            )
        )

        rows.append(
            {
                "sample_id":
                    holdout["id"],

                "holdout_error_mm":
                    1000.0
                    * holdout_error_m,

                "p_tip_shift_mm":
                    1000.0
                    * tip_shift_m,
            }
        )

    errors_mm = np.asarray(
        [
            r[
                "holdout_error_mm"
            ]
            for r in rows
        ],
        dtype=np.float64,
    )

    shifts_mm = np.asarray(
        [
            r[
                "p_tip_shift_mm"
            ]
            for r in rows
        ],
        dtype=np.float64,
    )

    worst_index = int(
        np.argmax(
            errors_mm
        )
    )

    return {
        "rows": rows,

        "position_rms_mm":
            float(
                np.sqrt(
                    np.mean(
                        errors_mm ** 2
                    )
                )
            ),

        "position_max_mm":
            float(
                np.max(
                    errors_mm
                )
            ),

        "worst_sample":
            rows[
                worst_index
            ]["sample_id"],

        "p_tip_shift_rms_mm":
            float(
                np.sqrt(
                    np.mean(
                        shifts_mm ** 2
                    )
                )
            ),

        "p_tip_shift_max_mm":
            float(
                np.max(
                    shifts_mm
                )
            ),
    }


def main():
    print()
    print(
        "=============================================="
    )
    print(
        " R1-A7 TCP / P_tip Pivot Solver"
    )
    print(
        "=============================================="
    )
    print()

    samples = load_samples()

    print(
        "Samples loaded:",
        len(samples),
    )

    solution = solve_pivot(
        samples
    )

    p_tip = (
        solution[
            "p_tip_ee"
        ]
    )

    p_pivot = (
        solution[
            "p_pivot_base"
        ]
    )

    singular_values = (
        solution[
            "singular_values"
        ]
    )

    condition_number = float(
        singular_values[0]
        / singular_values[-1]
    )

    evaluation = evaluate(
        samples,
        p_tip,
        p_pivot,
    )

    loo = leave_one_out(
        samples,
        p_tip,
    )

    p_tip_mm = (
        1000.0
        * p_tip
    )

    p_pivot_mm = (
        1000.0
        * p_pivot
    )

    print()
    print(
        "=============================================="
    )
    print(
        " Solution"
    )
    print(
        "=============================================="
    )

    print()
    print(
        "Matrix rank:",
        solution["rank"],
    )

    print(
        "Condition number:",
        f"{condition_number:.4f}",
    )

    print()
    print(
        "P_tip in R_ee [mm]:"
    )

    print(
        f"  X = {p_tip_mm[0]:+.6f}"
    )

    print(
        f"  Y = {p_tip_mm[1]:+.6f}"
    )

    print(
        f"  Z = {p_tip_mm[2]:+.6f}"
    )

    print(
        "  norm = "
        f"{np.linalg.norm(p_tip_mm):.6f} mm"
    )

    print()
    print(
        "Pivot point in Base [mm]:"
    )

    print(
        f"  X = {p_pivot_mm[0]:+.6f}"
    )

    print(
        f"  Y = {p_pivot_mm[1]:+.6f}"
    )

    print(
        f"  Z = {p_pivot_mm[2]:+.6f}"
    )

    print()
    print(
        "=============================================="
    )
    print(
        " Per-sample residuals"
    )
    print(
        "=============================================="
    )

    for row in evaluation[
        "rows"
    ]:
        print(
            f"sample_{row['sample_id']:04d}  "
            f"error={row['error_norm_mm']:7.3f} mm"
        )

    print()
    print(
        "=============================================="
    )
    print(
        " Full-dataset validation"
    )
    print(
        "=============================================="
    )

    print(
        "XYZ std [mm]: "
        f"{evaluation['xyz_std_mm'][0]:.3f}, "
        f"{evaluation['xyz_std_mm'][1]:.3f}, "
        f"{evaluation['xyz_std_mm'][2]:.3f}"
    )

    print(
        "Position mean [mm]: "
        f"{evaluation['position_mean_mm']:.3f}"
    )

    print(
        "Position RMS [mm]: "
        f"{evaluation['position_rms_mm']:.3f}"
    )

    print(
        "Position MAX [mm]: "
        f"{evaluation['position_max_mm']:.3f}"
    )

    print(
        "Worst sample: "
        f"sample_{evaluation['worst_sample']:04d}"
    )

    print()
    print(
        "=============================================="
    )
    print(
        " Leave-One-Out"
    )
    print(
        "=============================================="
    )

    for row in loo[
        "rows"
    ]:
        print(
            f"sample_{row['sample_id']:04d}  "
            f"holdout={row['holdout_error_mm']:7.3f} mm  "
            f"tip_shift={row['p_tip_shift_mm']:6.3f} mm"
        )

    print()
    print(
        "LOO Position RMS [mm]: "
        f"{loo['position_rms_mm']:.3f}"
    )

    print(
        "LOO Position MAX [mm]: "
        f"{loo['position_max_mm']:.3f}"
    )

    print(
        "LOO worst sample: "
        f"sample_{loo['worst_sample']:04d}"
    )

    print(
        "P_tip shift RMS [mm]: "
        f"{loo['p_tip_shift_rms_mm']:.3f}"
    )

    print(
        "P_tip shift MAX [mm]: "
        f"{loo['p_tip_shift_max_mm']:.3f}"
    )

    output = {
        "robot":
            "Unitree R1-A7",

        "calibration_type":
            "tcp_pivot",

        "ee_frame":
            "R_ee",

        "sample_count":
            len(samples),

        "solution_equation":
            (
                "R_base_ee * P_tip_ee "
                "+ t_base_ee "
                "= P_pivot_base"
            ),

        "matrix_rank":
            solution["rank"],

        "condition_number":
            condition_number,

        "P_tip_ee_m":
            p_tip.tolist(),

        "P_tip_ee_mm":
            p_tip_mm.tolist(),

        "P_tip_norm_mm":
            float(
                np.linalg.norm(
                    p_tip_mm
                )
            ),

        "P_pivot_base_m":
            p_pivot.tolist(),

        "P_pivot_base_mm":
            p_pivot_mm.tolist(),

        "validation": {
            "xyz_std_mm":
                evaluation[
                    "xyz_std_mm"
                ].tolist(),

            "position_mean_mm":
                evaluation[
                    "position_mean_mm"
                ],

            "position_rms_mm":
                evaluation[
                    "position_rms_mm"
                ],

            "position_max_mm":
                evaluation[
                    "position_max_mm"
                ],

            "worst_sample":
                evaluation[
                    "worst_sample"
                ],
        },

        "leave_one_out": {
            "position_rms_mm":
                loo[
                    "position_rms_mm"
                ],

            "position_max_mm":
                loo[
                    "position_max_mm"
                ],

            "worst_sample":
                loo[
                    "worst_sample"
                ],

            "p_tip_shift_rms_mm":
                loo[
                    "p_tip_shift_rms_mm"
                ],

            "p_tip_shift_max_mm":
                loo[
                    "p_tip_shift_max_mm"
                ],
        },

        "per_sample":
            evaluation[
                "rows"
            ],

        "status":
            (
                "candidate_not_yet_validated_"
                "for_autonomous_motion"
            ),
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Saved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
