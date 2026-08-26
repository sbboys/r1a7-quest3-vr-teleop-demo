#!/usr/bin/env python3

import csv
import json
import sys
from pathlib import Path

import numpy as np


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


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python check_home_episode.py "
            "data/episodes/experiment_home_001"
        )
        return 1

    episode_dir = Path(sys.argv[1])
    states_csv = episode_dir / "states.csv"

    if not states_csv.exists():
        raise FileNotFoundError(states_csv)

    timestamps = []
    q_list = []
    dq_list = []
    tau_list = []
    left_gripper = []
    right_gripper = []

    total_rows = 0
    invalid_rows = 0

    with states_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            total_rows += 1

            if row.get("communication_ok") != "True":
                invalid_rows += 1
                continue

            try:
                q = json.loads(
                    row["joint_position"]
                )
                dq = json.loads(
                    row["joint_velocity"]
                )
                tau = json.loads(
                    row["joint_torque_or_current"]
                )
                gripper = json.loads(
                    row["gripper_position"]
                )
            except Exception:
                invalid_rows += 1
                continue

            if len(q) != 14 or len(dq) != 14:
                invalid_rows += 1
                continue

            timestamps.append(
                float(row["timestamp_monotonic"])
            )

            q_list.append(q)
            dq_list.append(dq)

            if len(tau) == 14:
                tau_list.append(tau)
            else:
                tau_list.append(
                    [float("nan")] * 14
                )

            left_gripper.append(
                float(
                    gripper.get(
                        "left",
                        float("nan"),
                    )
                )
            )

            right_gripper.append(
                float(
                    gripper.get(
                        "right",
                        float("nan"),
                    )
                )
            )

    if len(q_list) < 10:
        raise RuntimeError(
            "Too few valid samples"
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

    tau = np.asarray(
        tau_list,
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
    # Remove approximately first and last 1 second.
    # --------------------------------------------------

    start_t = t[0] + 1.0
    end_t = t[-1] - 1.0

    mask = (
        (t >= start_t)
        & (t <= end_t)
    )

    if np.count_nonzero(mask) < 10:
        raise RuntimeError(
            "Not enough central-window samples"
        )

    t_c = t[mask]
    q_c = q[mask]
    dq_c = dq[mask]
    tau_c = tau[mask]

    left_g_c = left_gripper[mask]
    right_g_c = right_gripper[mask]

    # --------------------------------------------------
    # Static test
    # --------------------------------------------------

    sample_max_abs_dq = np.max(
        np.abs(dq_c),
        axis=1,
    )

    static_mask = (
        sample_max_abs_dq < 0.05
    )

    static_fraction = (
        np.count_nonzero(static_mask)
        / len(static_mask)
    )

    # Use only static samples for HOME median,
    # provided enough samples exist.
    if np.count_nonzero(static_mask) >= 20:
        q_static = q_c[static_mask]
        tau_static = tau_c[static_mask]

        left_g_static = (
            left_g_c[static_mask]
        )
        right_g_static = (
            right_g_c[static_mask]
        )

    else:
        q_static = q_c
        tau_static = tau_c

        left_g_static = left_g_c
        right_g_static = right_g_c

    # --------------------------------------------------
    # Robust HOME candidate
    # --------------------------------------------------

    q_median = np.median(
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

    q_p90_span = q_p95 - q_p05

    dq_abs_p95 = np.percentile(
        np.abs(dq_c),
        95,
        axis=0,
    )

    tau_median = np.nanmedian(
        tau_static,
        axis=0,
    )

    left_g_median = np.nanmedian(
        left_g_static
    )

    right_g_median = np.nanmedian(
        right_g_static
    )

    print()
    print("=" * 72)
    print("R1-A7 EXPERIMENT_HOME STATIC QUALITY")
    print("=" * 72)

    print(
        f"episode           : {episode_dir}"
    )

    print(
        f"total CSV rows    : {total_rows}"
    )

    print(
        f"invalid rows      : {invalid_rows}"
    )

    print(
        f"valid rows        : {len(q)}"
    )

    print(
        f"central samples   : {len(q_c)}"
    )

    print(
        f"duration valid    : "
        f"{t[-1] - t[0]:.3f} s"
    )

    print(
        f"static samples    : "
        f"{np.count_nonzero(static_mask)} "
        f"/ {len(static_mask)}"
    )

    print(
        f"static fraction   : "
        f"{100.0 * static_fraction:.2f}%"
    )

    print(
        f"max |dq| overall  : "
        f"{np.max(sample_max_abs_dq):.6f} rad/s"
    )

    print(
        f"P95 max |dq|      : "
        f"{np.percentile(sample_max_abs_dq, 95):.6f} rad/s"
    )

    print(
        f"max P95-P05 q span: "
        f"{np.max(q_p90_span):.6f} rad"
    )

    print()
    print(
        "Joint statistics:"
    )

    print(
        f"{'joint':28s} "
        f"{'q_median':>11s} "
        f"{'q_span90':>11s} "
        f"{'dq_p95':>11s} "
        f"{'tau_med':>11s}"
    )

    print("-" * 78)

    for i, name in enumerate(
        ARM_NAMES
    ):

        print(
            f"{name:28s} "
            f"{q_median[i]:+11.6f} "
            f"{q_p90_span[i]:11.6f} "
            f"{dq_abs_p95[i]:11.6f} "
            f"{tau_median[i]:+11.6f}"
        )

    print()
    print(
        "Gripper candidate:"
    )

    print(
        f"left  = "
        f"{left_g_median:+.9f}"
    )

    print(
        f"right = "
        f"{right_g_median:+.9f}"
    )

    print()
    print(
        "EXPERIMENT_HOME candidate:"
    )

    print(
        "left_joint_position:"
    )

    print(
        "  ["
        + ", ".join(
            f"{v:.9f}"
            for v in q_median[:7]
        )
        + "]"
    )

    print(
        "right_joint_position:"
    )

    print(
        "  ["
        + ", ".join(
            f"{v:.9f}"
            for v in q_median[7:]
        )
        + "]"
    )

    print()
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
