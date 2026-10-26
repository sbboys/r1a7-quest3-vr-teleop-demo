#!/usr/bin/env python3

import json
import math
import os
import time
from pathlib import Path


WIRE_FILE = Path(
    "/tmp/r1a7_right_wrist_wire_v69.json"
)

TIP_FILE = Path(
    "/tmp/r1a7_dynamic_tweezer_tip.json"
)

OUTPUT_FILE = Path(
    "/tmp/r1a7_wrist_wire_grasp_guidance.json"
)

MAX_AGE_S = 0.20

KP_Y = 0.08
KP_Z = 0.08

MAX_AXIS_MM_S = 10.0


def read_json(path):
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None


def clip(v, limit):
    return max(
        -limit,
        min(limit, v),
    )


def atomic_write(path, data):
    tmp = Path(str(path) + ".tmp")

    tmp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def main():

    print(
        "[WRIST GUIDANCE] "
        "INSERTION-STAGE DYNAMIC TWEEZER TIP",
        flush=True,
    )

    print(
        "[WRIST GUIDANCE] "
        "fixed (739,530) calibration DISABLED",
        flush=True,
    )

    while True:

        now = time.time()

        w = read_json(
            WIRE_FILE
        )

        t = read_json(
            TIP_FILE
        )

        valid = False

        target = None
        tip = None

        wire_age = math.inf
        tip_age = math.inf

        if w is not None:

            wire_age = (
                now
                -
                float(
                    w.get(
                        "timestamp",
                        0.0,
                    )
                )
            )

            target = w.get(
                "grasp_target_uv"
            )

        if t is not None:

            tip_age = (
                now
                -
                float(
                    t.get(
                        "timestamp",
                        0.0,
                    )
                )
            )

            tip = t.get(
                "tip_uv"
            )

        valid = bool(
            w is not None
            and
            t is not None
            and
            wire_age >= 0.0
            and
            wire_age <= MAX_AGE_S
            and
            tip_age >= 0.0
            and
            tip_age <= MAX_AGE_S
            and
            bool(
                w.get(
                    "grasp_guidance_valid",
                    False,
                )
            )
            and
            bool(
                t.get(
                    "valid",
                    False,
                )
            )
            and
            isinstance(
                target,
                list,
            )
            and
            len(target) == 2
            and
            isinstance(
                tip,
                list,
            )
            and
            len(tip) == 2
        )

        if valid:

            du = (
                float(target[0])
                -
                float(tip[0])
            )

            dv = (
                float(target[1])
                -
                float(tip[1])
            )

            # Same wrist image -> robot mapping:
            # image +U -> robot -Y
            # image +V -> robot -Z
            vy = clip(
                -KP_Y * du,
                MAX_AXIS_MM_S,
            )

            vz = clip(
                -KP_Z * dv,
                MAX_AXIS_MM_S,
            )

        else:

            du = None
            dv = None

            vy = 0.0
            vz = 0.0

        out = {
            "timestamp": now,

            "enabled": False,
            "shadow_mode": True,

            "source":
                "right_wrist_wire_dynamic_insertion_tip",

            "input_file":
                str(WIRE_FILE),

            "tip_input_file":
                str(TIP_FILE),

            "input_age_s":
                wire_age
                if math.isfinite(wire_age)
                else None,

            "tip_age_s":
                tip_age
                if math.isfinite(tip_age)
                else None,

            "guidance_valid":
                bool(valid),

            # Keep compatibility field name,
            # but its value is now DYNAMIC.
            "tweezer_grasp_center_uv":
                tip
                if valid
                else None,

            "tweezer_tip_uv":
                tip
                if valid
                else None,

            "tweezer_tip_state":
                (
                    t.get("state")
                    if t is not None
                    else None
                ),

            "grasp_target_uv":
                target
                if valid
                else None,

            "du_px":
                du,

            "dv_px":
                dv,

            "proposed_vx_mm_s":
                0.0,

            "proposed_vy_mm_s":
                float(vy),

            "proposed_vz_mm_s":
                float(vz),
        }

        atomic_write(
            OUTPUT_FILE,
            out,
        )

        time.sleep(
            0.02
        )


if __name__ == "__main__":
    main()
