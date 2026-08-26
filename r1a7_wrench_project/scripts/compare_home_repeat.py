#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


ARM_NAMES = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",

    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


def load_home(keyframes_path: Path):

    with keyframes_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = yaml.safe_load(f)

    home = data["keyframes"]["HOME"]

    left = home["left_joint_position"]
    right = home["right_joint_position"]

    q_home = np.asarray(
        left + right,
        dtype=float,
    )

    if len(q_home) != 14:
        raise RuntimeError(
            f"HOME should contain 14 arm joints, got {len(q_home)}"
        )

    left_gripper = home.get(
        "left_gripper_position"
    )

    right_gripper = home.get(
        "right_gripper_position"
    )

    return (
        q_home,
        left_gripper,
        right_gripper,
    )


def load_repeat_episode(
    episode_dir: Path,
    trim_seconds=1.0,
    static_dq_threshold=0.05,
):

    states_csv = (
        episode_dir / "states.csv"
    )

    if not states_csv.exists():
        raise FileNotFoundError(
            states_csv
        )

    timestamps = []
    q_list = []
    dq_list = []
    left_gripper = []
    right_gripper = []

    total_rows = 0
    valid_rows = 0

    with states_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            total_rows += 1

            if (
                row.get("communication_ok")
                != "True"
            ):
                continue

            try:
                q = json.loads(
                    row["joint_position"]
                )

                dq = json.loads(
                    row["joint_velocity"]
                )

                gripper = json.loads(
                    row["gripper_position"]
                )

            except Exception:
                continue

            if (
                len(q) != 14
                or len(dq) != 14
            ):
                continue

            timestamps.append(
                float(
                    row["timestamp_monotonic"]
                )
            )

            q_list.append(q)
            dq_list.append(dq)

            left_gripper.append(
                float(
                    gripper.get(
                        "left",
                        np.nan,
                    )
                )
            )

            right_gripper.append(
                float(
                    gripper.get(
                        "right",
                        np.nan,
                    )
                )
            )

            valid_rows += 1

    if valid_rows < 10:
        raise RuntimeError(
            "Too few valid robot-state rows"
        )

    t = np.asarray(
        timestamps,
        dtype=float,
    )

    q = np.asarray(
        q_list,
        dtype=float,
    )

    dq = np.asarray(
        dq_list,
        dtype=float,
    )

    left_gripper = np.asarray(
        left_gripper,
        dtype=float,
    )

    right_gripper = np.asarray(
        right_gripper,
        dtype=float,
    )

    # --------------------------------------------------
    # Remove startup/end transient
    # --------------------------------------------------

    start_t = (
        t[0] + trim_seconds
    )

    end_t = (
        t[-1] - trim_seconds
    )

    central_mask = (
        (t >= start_t)
        & (t <= end_t)
    )

    if (
        np.count_nonzero(
            central_mask
        )
        < 10
    ):
        raise RuntimeError(
            "Not enough central samples"
        )

    q_c = q[central_mask]
    dq_c = dq[central_mask]

    left_g_c = (
        left_gripper[
            central_mask
        ]
    )

    right_g_c = (
        right_gripper[
            central_mask
        ]
    )

    # --------------------------------------------------
    # Only retain sufficiently static samples
    # --------------------------------------------------

    max_abs_dq_per_sample = np.max(
        np.abs(dq_c),
        axis=1,
    )

    static_mask = (
        max_abs_dq_per_sample
        < static_dq_threshold
    )

    static_count = (
        np.count_nonzero(
            static_mask
        )
    )

    if static_count < 20:
        raise RuntimeError(
            "Too few static samples. "
            "Robot may not have been held still."
        )

    q_static = (
        q_c[static_mask]
    )

    left_g_static = (
        left_g_c[static_mask]
    )

    right_g_static = (
        right_g_c[static_mask]
    )

    q_repeat = np.median(
        q_static,
        axis=0,
    )

    q_p05 = np.percentile(
        q_static,
        5,
        axis=0,
    )

    q_p95 = np.percentile(
        q_static,
        95,
        axis=0,
    )

    q_span = (
        q_p95 - q_p05
    )

    return {
        "q_repeat": q_repeat,
        "left_gripper": float(
            np.nanmedian(
                left_g_static
            )
        ),
        "right_gripper": float(
            np.nanmedian(
                right_g_static
            )
        ),
        "static_samples": static_count,
        "central_samples": len(q_c),
        "p95_max_abs_dq": float(
            np.percentile(
                max_abs_dq_per_sample,
                95,
            )
        ),
        "max_q_span": float(
            np.max(q_span)
        ),
        "total_rows": total_rows,
        "valid_rows": valid_rows,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "episode",
        type=Path,
        help="repeat episode directory",
    )

    parser.add_argument(
        "--keyframes",
        type=Path,
        default=Path(
            "r1a7_wrench_project/"
            "config/keyframes.yaml"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help=(
            "maximum allowed absolute "
            "joint error in rad"
        ),
    )

    args = parser.parse_args()

    q_home, home_left_grip, home_right_grip = (
        load_home(
            args.keyframes
        )
    )

    repeat = load_repeat_episode(
        args.episode
    )

    q_repeat = repeat[
        "q_repeat"
    ]

    error = (
        q_repeat - q_home
    )

    abs_error = np.abs(
        error
    )

    max_error = float(
        np.max(abs_error)
    )

    max_index = int(
        np.argmax(abs_error)
    )

    mean_error = float(
        np.mean(abs_error)
    )

    rms_error = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    left_max_error = float(
        np.max(
            abs_error[:7]
        )
    )

    right_max_error = float(
        np.max(
            abs_error[7:]
        )
    )

    passed = (
        max_error
        < args.threshold
    )

    print()
    print("=" * 104)

    print(
        "R1-A7 HOME REPEATABILITY CHECK"
    )

    print("=" * 104)

    print(
        f"Reference HOME : "
        f"{args.keyframes}"
    )

    print(
        f"Repeat episode : "
        f"{args.episode}"
    )

    print(
        f"Threshold      : "
        f"{args.threshold:.6f} rad "
        f"({np.degrees(args.threshold):.3f} deg)"
    )

    print()

    print(
        f"Valid rows     : "
        f"{repeat['valid_rows']} / "
        f"{repeat['total_rows']}"
    )

    print(
        f"Static samples : "
        f"{repeat['static_samples']} / "
        f"{repeat['central_samples']}"
    )

    print(
        f"P95 max |dq|   : "
        f"{repeat['p95_max_abs_dq']:.6f} rad/s"
    )

    print(
        f"Max q span     : "
        f"{repeat['max_q_span']:.6f} rad"
    )

    print()

    print(
        f"{'joint':28s}"
        f"{'HOME(rad)':>13s}"
        f"{'repeat(rad)':>14s}"
        f"{'error(rad)':>13s}"
        f"{'error(deg)':>13s}"
        f"{'result':>10s}"
    )

    print("-" * 104)

    for i, name in enumerate(
        ARM_NAMES
    ):

        joint_pass = (
            abs_error[i]
            < args.threshold
        )

        result = (
            "PASS"
            if joint_pass
            else "FAIL"
        )

        print(
            f"{name:28s}"
            f"{q_home[i]:+13.6f}"
            f"{q_repeat[i]:+14.6f}"
            f"{error[i]:+13.6f}"
            f"{np.degrees(error[i]):+13.3f}"
            f"{result:>10s}"
        )

    print()
    print("-" * 104)

    print(
        f"Left arm max error  : "
        f"{left_max_error:.6f} rad "
        f"({np.degrees(left_max_error):.3f} deg)"
    )

    print(
        f"Right arm max error : "
        f"{right_max_error:.6f} rad "
        f"({np.degrees(right_max_error):.3f} deg)"
    )

    print(
        f"Mean absolute error : "
        f"{mean_error:.6f} rad "
        f"({np.degrees(mean_error):.3f} deg)"
    )

    print(
        f"RMS joint error     : "
        f"{rms_error:.6f} rad "
        f"({np.degrees(rms_error):.3f} deg)"
    )

    print(
        f"MAX absolute error  : "
        f"{max_error:.6f} rad "
        f"({np.degrees(max_error):.3f} deg)"
    )

    print(
        f"Worst joint         : "
        f"{ARM_NAMES[max_index]}"
    )

    print()

    print(
        "Gripper:"
    )

    if home_left_grip is not None:

        print(
            f"  left HOME   = "
            f"{float(home_left_grip):+.6f}"
        )

        print(
            f"  left repeat = "
            f"{repeat['left_gripper']:+.6f}"
        )

    if home_right_grip is not None:

        print(
            f"  right HOME   = "
            f"{float(home_right_grip):+.6f}"
        )

        print(
            f"  right repeat = "
            f"{repeat['right_gripper']:+.6f}"
        )

    print()
    print("=" * 104)

    if passed:

        print(
            "HOME_READY = PASS"
        )

        print(
            "All 14 arm joints are within "
            "the allowed HOME error."
        )

    else:

        print(
            "HOME_READY = FAIL"
        )

        print(
            f"Worst joint: "
            f"{ARM_NAMES[max_index]}"
        )

        print(
            f"|error| = "
            f"{max_error:.6f} rad "
            f">= "
            f"{args.threshold:.6f} rad"
        )

    print("=" * 104)
    print()

    return (
        0
        if passed
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())

