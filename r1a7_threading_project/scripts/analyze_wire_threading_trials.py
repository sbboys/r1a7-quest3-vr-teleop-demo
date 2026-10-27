#!/usr/bin/env python3
"""Summarize R1-A7 wire-threading trial CSV files without robot access."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "trial_id",
    "condition_id",
    "method_id",
    "duration_s",
    "first_attempt_success",
    "final_success",
    "failure_code",
}

NUMERIC_METRICS = (
    "duration_s",
    "wire_iou",
    "wire_cldice",
    "tip_error_px",
    "hole_error_px",
    "visual_error_mm",
    "valid_observation_rate",
    "valid_update_rate_hz",
    "capture_to_control_latency_p95_ms",
    "fusion_time_delta_p95_ms",
    "alignment_error_mm",
    "direction_error_deg",
    "hold_rms_mm",
    "stall_ratio_pct",
    "jump_count",
    "max_jump_mm",
    "grasp_slip_mm",
    "stop_command_latency_ms",
    "stop_overtravel_mm",
    "protrusion_mm",
)

TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n"}


def parse_bool(value: str, field: str, trial_id: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"trial {trial_id!r}: {field} must be one of 0/1/true/false/yes/no"
    )


def optional_float(value: str, field: str, trial_id: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"trial {trial_id!r}: invalid {field}={value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"trial {trial_id!r}: non-finite {field}={value!r}")
    return number


def nearest_rank(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("nearest_rank requires at least one value")
    index = max(1, math.ceil(fraction * len(ordered))) - 1
    return ordered[index]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no trial rows")
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        trial_id = row["trial_id"].strip()
        if not trial_id:
            raise ValueError(f"line {line_number}: trial_id is empty")
        if trial_id in seen:
            raise ValueError(f"duplicate trial_id: {trial_id}")
        seen.add(trial_id)
        row["trial_id"] = trial_id
    return rows


def numeric_summary(rows: list[dict[str, str]], field: str) -> dict[str, float | int] | None:
    values: list[float] = []
    for row in rows:
        if field not in row:
            continue
        value = optional_float(row.get(field, ""), field, row["trial_id"])
        if value is not None:
            values.append(value)
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95_nearest_rank": nearest_rank(values, 0.95),
        "max": max(values),
    }


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    first = [
        parse_bool(row["first_attempt_success"], "first_attempt_success", row["trial_id"])
        for row in rows
    ]
    final = [
        parse_bool(row["final_success"], "final_success", row["trial_id"])
        for row in rows
    ]
    first_count = sum(first)
    final_count = sum(final)
    failures = Counter(
        row["failure_code"].strip() or "UNSPECIFIED"
        for row, succeeded in zip(rows, final)
        if not succeeded
    )
    metrics = {
        field: summary
        for field in NUMERIC_METRICS
        if (summary := numeric_summary(rows, field)) is not None
    }
    total = len(rows)
    return {
        "n_trials": total,
        "first_attempt_success": {
            "successes": first_count,
            "rate": first_count / total,
            "wilson_95_ci": wilson_interval(first_count, total),
        },
        "final_success": {
            "successes": final_count,
            "rate": final_count / total,
            "wilson_95_ci": wilson_interval(final_count, total),
        },
        "failure_counts": dict(sorted(failures.items())),
        "metrics": metrics,
    }


def grouped_summary(rows: list[dict[str, str]], group_fields: list[str]) -> dict[str, object]:
    for field in group_fields:
        if field not in rows[0]:
            raise ValueError(f"group-by column not found: {field}")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[field].strip() or "UNSPECIFIED" for field in group_fields)
        groups.setdefault(key, []).append(row)
    return {
        "group_by": group_fields,
        "groups": {
            " | ".join(key): summarize(group_rows)
            for key, group_rows in sorted(groups.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize independent R1-A7 wire-threading trials."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["method_id", "condition_id"],
        help="CSV columns used for grouped results",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    report = {
        "source_csv": str(args.csv_path.resolve()),
        "quantile_method": "nearest_rank",
        "confidence_interval": "wilson_95",
        "overall": summarize(rows),
        "grouped": grouped_summary(rows, args.group_by),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
