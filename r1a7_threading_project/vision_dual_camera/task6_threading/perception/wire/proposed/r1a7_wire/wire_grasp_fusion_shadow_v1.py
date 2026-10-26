#!/usr/bin/env python3

import json
import math
import os
import time
from pathlib import Path


EXT_FILE = Path(
    "/tmp/r1a7_external_wire_grasp_v1.json"
)

WRIST_FILE = Path(
    "/tmp/r1a7_wrist_wire_grasp_guidance.json"
)

OUTPUT = Path(
    "/tmp/r1a7_wire_grasp_fusion_shadow_v1.json"
)


MAX_AGE = 0.40

MAX_VECTOR_MM_S = 10.0


def read(path):

    try:
        return json.loads(
            path.read_text()
        )
    except Exception:
        return {}


def age(data, now):

    try:
        return max(
            0.0,
            now -
            float(
                data["timestamp"]
            )
        )
    except Exception:
        return float("inf")


def valid(data, a):

    return bool(
        data
        and
        a <= MAX_AGE
        and
        data.get(
            "guidance_valid",
            False
        )
    )


def vector_limit(
    x,
    y,
    z
):

    n = math.sqrt(
        x*x +
        y*y +
        z*z
    )

    if (
        n <= MAX_VECTOR_MM_S
        or
        n <= 1e-9
    ):
        return (
            x,
            y,
            z
        )

    k = (
        MAX_VECTOR_MM_S
        /
        n
    )

    return (
        x*k,
        y*k,
        z*k
    )


def atomic(data):

    tmp = Path(
        str(OUTPUT) + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            data,
            indent=2
        )
    )

    os.replace(
        tmp,
        OUTPUT
    )


last_mode = None


while True:

    now = time.time()

    ext = read(
        EXT_FILE
    )

    wrist = read(
        WRIST_FILE
    )

    ea = age(
        ext,
        now
    )

    wa = age(
        wrist,
        now
    )

    ev = valid(
        ext,
        ea
    )

    wv = valid(
        wrist,
        wa
    )


    vx = 0.0
    vy = 0.0
    vz = 0.0


    if ev and wv:

        mode = "DUAL"

        vx = float(
            ext.get(
                "proposed_vx_mm_s",
                0.0
            )
        )

        vy = float(
            wrist.get(
                "proposed_vy_mm_s",
                0.0
            )
        )

        ez = float(
            ext.get(
                "proposed_vz_mm_s",
                0.0
            )
        )

        wz = float(
            wrist.get(
                "proposed_vz_mm_s",
                0.0
            )
        )

        # If both cameras agree on Z:
        # wrist gets more fine-control weight.
        if (
            ez == 0.0
            or
            wz == 0.0
            or
            ez * wz > 0
        ):

            vz = (
                0.35 * ez
                +
                0.65 * wz
            )

        else:

            # Disagreement means do NOT guess Z.
            vz = 0.0

            mode = (
                "DUAL_Z_CONFLICT"
            )


    elif ev:

        mode = (
            "EXTERNAL_ONLY"
        )

        vx = float(
            ext.get(
                "proposed_vx_mm_s",
                0.0
            )
        )

        vy = 0.0

        vz = float(
            ext.get(
                "proposed_vz_mm_s",
                0.0
            )
        )


    elif wv:

        mode = (
            "WRIST_ONLY"
        )

        vx = 0.0

        vy = float(
            wrist.get(
                "proposed_vy_mm_s",
                0.0
            )
        )

        vz = float(
            wrist.get(
                "proposed_vz_mm_s",
                0.0
            )
        )


    else:

        mode = "STOP"


    (
        vx,
        vy,
        vz
    ) = vector_limit(
        vx,
        vy,
        vz
    )


    # ========================================================
    # ALWAYS SHADOW.
    #
    # proposed_* is diagnostic only.
    # actual enabled output remains false.
    # ========================================================

    out = {
        "timestamp":
            now,

        "enabled":
            False,

        "shadow_mode":
            True,

        "ready_for_motion":
            bool(
                ev or wv
            ),

        "fusion_mode":
            mode,

        "external_valid":
            ev,

        "wrist_valid":
            wv,

        "external_age_s":
            (
                ea
                if math.isfinite(ea)
                else None
            ),

        "wrist_age_s":
            (
                wa
                if math.isfinite(wa)
                else None
            ),

        "proposed_vx_mm_s":
            float(vx),

        "proposed_vy_mm_s":
            float(vy),

        "proposed_vz_mm_s":
            float(vz),

        "external":
            ext,

        "wrist":
            wrist,
    }


    atomic(
        out
    )


    if mode != last_mode:

        print(
            "[WIRE GRASP FUSION]",
            last_mode,
            "->",
            mode,
            flush=True
        )

        last_mode = mode


    print(
        f"[{mode}] "
        f"EXT={'OK' if ev else 'NO'} "
        f"WRIST={'OK' if wv else 'NO'} "
        f"proposed="
        f"({vx:+.2f},"
        f"{vy:+.2f},"
        f"{vz:+.2f}) mm/s",
        flush=True
    )


    time.sleep(
        0.05
    )
