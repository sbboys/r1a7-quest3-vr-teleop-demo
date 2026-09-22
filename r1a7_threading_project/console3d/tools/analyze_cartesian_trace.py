#!/usr/bin/env python3
"""Summarize a Cartesian console diagnostic trace without connecting to the robot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


JOINT_NAMES = (
    "L_SHOULDER_PITCH",
    "L_SHOULDER_ROLL",
    "L_SHOULDER_YAW",
    "L_ELBOW",
    "L_WRIST_ROLL",
    "L_WRIST_PITCH",
    "L_WRIST_YAW",
    "R_SHOULDER_PITCH",
    "R_SHOULDER_ROLL",
    "R_SHOULDER_YAW",
    "R_ELBOW",
    "R_WRIST_ROLL",
    "R_WRIST_PITCH",
    "R_WRIST_YAW",
)

DIRECTION_ACTIVE_AXES = {
    "w": np.asarray([True, False, False]),
    "s": np.asarray([True, False, False]),
    "a": np.asarray([False, True, False]),
    "d": np.asarray([False, True, False]),
    "u": np.asarray([False, False, True]),
    "j": np.asarray([False, False, True]),
    "i": np.asarray([True, False, True]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    return parser.parse_args()


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with path.expanduser().open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) < 10:
        raise RuntimeError(f"trace contains only {len(rows)} samples")

    scalar_names = (
        "time_monotonic",
        "loop_dt_s",
        "ik_solve_ms",
        "cartesian_speed_mm_s",
        "cartesian_lag_mm",
    )
    array_names = (
        "arm_q_actual",
        "arm_dq_actual",
        "arm_q_ik_raw",
        "arm_q_stream_goal",
        "arm_q_lowcmd_sent",
    )
    result = {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in scalar_names
    }
    result.update(
        {
            name: np.asarray([json.loads(row[name]) for row in rows], dtype=float)
            for name in array_names
        }
    )
    result["moving"] = np.asarray(
        [bool(row["jog_direction"]) or float(row["cartesian_speed_mm_s"]) > 0.2 for row in rows],
        dtype=bool,
    )
    result["jog_direction"] = np.asarray(
        [row["jog_direction"].strip() for row in rows], dtype=str
    )
    for name in ("command_xyz_mm", "actual_xyz_mm"):
        result[name] = np.asarray(
            [json.loads(row[name]) for row in rows], dtype=float
        )
    return result


def direction_segments(directions: np.ndarray) -> list[tuple[int, int, str]]:
    segments: list[tuple[int, int, str]] = []
    start = 0
    while start < len(directions):
        direction = str(directions[start])
        end = start + 1
        while end < len(directions) and directions[end] == direction:
            end += 1
        if direction in DIRECTION_ACTIVE_AXES:
            segments.append((start, end, direction))
        start = end
    return segments


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def dominant_frequency(values: np.ndarray, sample_rate_hz: float) -> tuple[float, float]:
    if len(values) < 30 or sample_rate_hz <= 0.0:
        return float("nan"), float("nan")
    centered = values - np.mean(values, axis=0, keepdims=True)
    window = np.hanning(len(centered))[:, None]
    spectrum = np.abs(np.fft.rfft(centered * window, axis=0))
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    band = (frequencies >= 1.0) & (frequencies <= min(12.0, sample_rate_hz * 0.45))
    if not np.any(band):
        return float("nan"), float("nan")
    band_spectrum = spectrum[band]
    flat_index = int(np.argmax(band_spectrum))
    frequency_index, joint_index = np.unravel_index(flat_index, band_spectrum.shape)
    return float(frequencies[band][frequency_index]), float(joint_index)


def main() -> int:
    args = parse_args()
    data = load_trace(args.trace)
    moving = data["moving"]
    if int(np.count_nonzero(moving)) < 30:
        moving = np.ones_like(moving, dtype=bool)

    times = data["time_monotonic"][moving]
    intervals = np.diff(times)
    median_dt = float(np.median(intervals))
    sample_rate_hz = 1.0 / median_dt
    q_actual = data["arm_q_actual"][moving]
    dq_actual = data["arm_dq_actual"][moving]
    q_raw = data["arm_q_ik_raw"][moving]
    q_stream = data["arm_q_stream_goal"][moving]
    q_sent = data["arm_q_lowcmd_sent"][moving]

    tracking_error_deg = np.rad2deg(q_sent - q_actual)
    raw_step_deg = np.rad2deg(np.diff(q_raw, axis=0))
    stream_step_deg = np.rad2deg(np.diff(q_stream, axis=0))
    actual_speed_deg_s = np.rad2deg(dq_actual)

    dq_rms = np.sqrt(np.mean(actual_speed_deg_s**2, axis=0))
    error_rms = np.sqrt(np.mean(tracking_error_deg**2, axis=0))
    raw_step_rms = np.sqrt(np.mean(raw_step_deg**2, axis=0))
    stream_step_rms = np.sqrt(np.mean(stream_step_deg**2, axis=0))
    dominant_hz, dominant_joint = dominant_frequency(dq_actual, sample_rate_hz)

    print(f"trace={args.trace.expanduser().resolve()}")
    print(f"samples={len(times)} duration_s={times[-1] - times[0]:.3f}")
    print(
        f"sample_rate_hz={sample_rate_hz:.2f} "
        f"loop_dt_p95_ms={percentile(intervals, 95) * 1000.0:.2f} "
        f"ik_p50_ms={percentile(data['ik_solve_ms'][moving], 50):.2f} "
        f"ik_p95_ms={percentile(data['ik_solve_ms'][moving], 95):.2f}"
    )
    print(
        f"cartesian_speed_p95_mm_s={percentile(data['cartesian_speed_mm_s'][moving], 95):.2f} "
        f"lag_p50_mm={percentile(data['cartesian_lag_mm'][moving], 50):.2f} "
        f"lag_p95_mm={percentile(data['cartesian_lag_mm'][moving], 95):.2f}"
    )
    if np.isfinite(dominant_hz):
        print(
            f"dominant_motion_frequency_hz={dominant_hz:.2f} "
            f"joint={JOINT_NAMES[int(dominant_joint)]}"
        )

    axis_lock_metrics = []
    for start, end, direction in direction_segments(data["jog_direction"]):
        if end - start < max(3, int(round(sample_rate_hz * 0.5))):
            continue
        inactive = ~DIRECTION_ACTIVE_AXES[direction]
        command = data["command_xyz_mm"][start:end]
        if np.any(inactive):
            locked_axis_drift_mm = float(
                np.max(np.abs(command[:, inactive] - command[0, inactive]))
            )
        else:
            locked_axis_drift_mm = 0.0
        axis_lock_metrics.append((locked_axis_drift_mm, direction, end - start))

    if axis_lock_metrics:
        worst_drift, worst_direction, worst_samples = max(axis_lock_metrics)
        print(
            "axis_lock_worst_command_drift_mm="
            f"{worst_drift:.3f} direction={worst_direction} "
            f"duration_s={worst_samples / sample_rate_hz:.2f}"
        )

    print("top_joint_metrics:")
    ranking = np.argsort(dq_rms + error_rms)[::-1]
    for index in ranking[:7]:
        print(
            f"  {JOINT_NAMES[index]:18s} "
            f"actual_dq_rms={dq_rms[index]:7.3f} deg/s "
            f"tracking_error_rms={error_rms[index]:7.3f} deg "
            f"raw_ik_step_rms={raw_step_rms[index]:7.4f} deg "
            f"stream_step_rms={stream_step_rms[index]:7.4f} deg"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
