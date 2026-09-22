#!/usr/bin/env python3

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

src = ROOT / "handeye_result.json"
dst = ROOT / "r1a7_right_wrist_handeye.json"

with src.open("r", encoding="utf-8") as f:
    result = json.load(f)

METHOD = "DANIILIDIS"

method_result = result["methods"][METHOD]

if not method_result.get("success", False):
    raise RuntimeError(
        f"{METHOD} result is not valid."
    )

output = {
    "camera_role": "right_wrist",
    "camera_model": "Orbbec Gemini 336L",
    "camera_serial": "CPCBC53000C5",

    "robot": "Unitree R1-A7",

    "ee_frame": "R_ee",

    "ee_definition": {
        "parent_joint": "right_wrist_yaw_joint",
        "local_translation_m": [
            0.05,
            0.0,
            0.0
        ]
    },

    "method": METHOD,

    "sample_count": result["sample_count"],

    "transform_definition": {
        "name": "T_ee_camera",
        "meaning": (
            "camera frame expressed relative to R_ee; "
            "P_ee = T_ee_camera @ P_camera"
        )
    },

    "T_ee_camera": method_result["T_ee_camera"],

    "translation_mm":
        method_result["translation_mm"],

    "translation_norm_mm":
        method_result["translation_norm_mm"],

    "validation": {
        "full_dataset_position_rms_mm":
            method_result["validation"][
                "position_rms_mm"
            ],

        "full_dataset_position_max_mm":
            method_result["validation"][
                "position_max_mm"
            ],

        "full_dataset_rotation_rms_deg":
            method_result["validation"][
                "rotation_rms_deg"
            ],

        "full_dataset_rotation_max_deg":
            method_result["validation"][
                "rotation_max_deg"
            ],

        "loo_position_rms_mm": 4.086,
        "loo_position_max_mm": 5.856,
        "loo_rotation_rms_deg": 0.4296,
        "loo_rotation_max_deg": 0.6254
    },

    "status":
        "accepted_for_geometric_validation_not_yet_for_autonomous_motion"
}

with dst.open(
    "w",
    encoding="utf-8"
) as f:
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
