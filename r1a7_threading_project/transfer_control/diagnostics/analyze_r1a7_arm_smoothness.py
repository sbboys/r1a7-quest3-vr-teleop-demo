#!/usr/bin/env python3
"""Offline analysis for r1a7_arm_smoothness_baseline.py CSV logs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ARM_NAMES = (
    "L_SHOULDER_PITCH", "L_SHOULDER_ROLL", "L_SHOULDER_YAW",
    "L_ELBOW", "L_WRIST_ROLL", "L_WRIST_PITCH", "L_WRIST_YAW",
    "R_SHOULDER_PITCH", "R_SHOULDER_ROLL", "R_SHOULDER_YAW",
    "R_ELBOW", "R_WRIST_ROLL", "R_WRIST_PITCH", "R_WRIST_YAW",
)


def rms(x: np.ndarray, axis=0):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    with args.csv.expanduser().open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 20:
        raise RuntimeError(f"log contains only {len(rows)} samples")

    t = np.asarray([float(r["time_monotonic"]) for r in rows], dtype=float)
    q_cmd = np.asarray([json.loads(r["q_cmd"]) for r in rows], dtype=float)
    q = np.asarray([json.loads(r["q_actual"]) for r in rows], dtype=float)
    dq = np.asarray([json.loads(r["dq_actual"]) for r in rows], dtype=float)

    dt = np.diff(t)
    median_dt = float(np.median(dt))
    fs = 1.0 / median_dt
    # Use logged velocity to avoid differentiating encoder position once more than needed.
    accel = np.diff(dq, axis=0) / dt[:, None]
    dt_a = (dt[1:] + dt[:-1]) * 0.5
    jerk = np.diff(accel, axis=0) / dt_a[:, None]

    tracking_deg = np.rad2deg(q_cmd - q)
    dq_deg_s = np.rad2deg(dq)
    accel_deg_s2 = np.rad2deg(accel)
    jerk_deg_s3 = np.rad2deg(jerk)

    cmd_atv_deg = np.mean(np.abs(np.diff(np.rad2deg(q_cmd), axis=0)), axis=0)
    actual_atv_deg = np.mean(np.abs(np.diff(np.rad2deg(q), axis=0)), axis=0)
    tracking_rms_deg = rms(tracking_deg)
    speed_rms_deg_s = rms(dq_deg_s)
    accel_rms_deg_s2 = rms(accel_deg_s2)
    jerk_rms_deg_s3 = rms(jerk_deg_s3)

    print(f"file={args.csv.expanduser().resolve()}")
    print(f"samples={len(t)} duration_s={t[-1]-t[0]:.3f}")
    print(f"sample_rate_hz={fs:.2f} dt_p95_ms={np.percentile(dt,95)*1000.0:.3f}")
    print("top_joint_smoothness_metrics:")
    score = tracking_rms_deg + 0.02 * jerk_rms_deg_s3
    for i in np.argsort(score)[::-1][:7]:
        print(
            f"  {ARM_NAMES[i]:18s} "
            f"track_rms={tracking_rms_deg[i]:8.4f} deg "
            f"dq_rms={speed_rms_deg_s[i]:8.4f} deg/s "
            f"acc_rms={accel_rms_deg_s2[i]:10.3f} deg/s^2 "
            f"jerk_rms={jerk_rms_deg_s3[i]:12.3f} deg/s^3 "
            f"cmd_ATV={cmd_atv_deg[i]:8.5f} deg "
            f"actual_ATV={actual_atv_deg[i]:8.5f} deg"
        )
    print(f"global_tracking_rms_deg={float(rms(tracking_deg.reshape(-1))):.6f}")
    print(f"global_jerk_rms_deg_s3={float(rms(jerk_deg_s3.reshape(-1))):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
