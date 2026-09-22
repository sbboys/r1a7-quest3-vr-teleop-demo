#!/usr/bin/env python3

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

src = ROOT / "tcp_pivot_result.json"
dst = ROOT / "r1a7_right_tool_tcp.json"

with src.open("r", encoding="utf-8") as f:
    result = json.load(f)

output = {
    "robot": "Unitree R1-A7",

    "tool_side": "right",

    "ee_frame": "R_ee",

    "ee_definition": {
        "parent_joint": "right_wrist_yaw_joint",
        "local_translation_m": [
            0.05,
            0.0,
            0.0
        ]
    },

    "tcp_definition": {
        "name": "P_tip",
        "meaning": (
            "tool working point expressed "
            "in R_ee coordinates"
        )
    },

    "P_tip_ee_m":
        result["P_tip_ee_m"],

    "P_tip_ee_mm":
        result["P_tip_ee_mm"],

    "P_tip_norm_mm":
        result["P_tip_norm_mm"],

    "calibration_pivot_base_m":
        result["P_pivot_base_m"],

    "calibration_pivot_base_mm":
        result["P_pivot_base_mm"],

    "sample_count":
        result["sample_count"],

    "validation": {
        "full_position_rms_mm":
            result["validation"]["position_rms_mm"],

        "full_position_max_mm":
            result["validation"]["position_max_mm"],

        "loo_position_rms_mm":
            result["leave_one_out"]["position_rms_mm"],

        "loo_position_max_mm":
            result["leave_one_out"]["position_max_mm"],

        "loo_p_tip_shift_rms_mm":
            result["leave_one_out"]["p_tip_shift_rms_mm"],

        "loo_p_tip_shift_max_mm":
            result["leave_one_out"]["p_tip_shift_max_mm"]
    },

    "status":
        "accepted_for_independent_geometric_validation"
}

with dst.open("w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )

print("Saved:", dst)
print()
print(
    json.dumps(
        output,
        indent=2,
        ensure_ascii=False
    )
)
