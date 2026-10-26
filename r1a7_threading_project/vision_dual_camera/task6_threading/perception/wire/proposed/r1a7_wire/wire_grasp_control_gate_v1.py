#!/usr/bin/env python3

import argparse
import json
import math
import os
import time
from pathlib import Path


INPUT = Path(
    "/tmp/r1a7_wire_grasp_fusion_shadow_v1.json"
)

OUTPUT = Path(
    "/tmp/r1a7_dual_ibvs_velocity.json"
)


# ============================================================
# First physical-motion test limits
# ============================================================

MAX_INPUT_AGE_S = 0.20


# Do not move if image error is absurdly large.
MAX_EXTERNAL_ERROR_PX = 350.0
MAX_WRIST_ERROR_PX = 350.0


def read_json(path):

    try:
        return json.loads(
            path.read_text()
        )
    except Exception:
        return {}


def atomic_write(data):

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


def clip(v, limit):

    return max(
        -float(limit),
        min(
            float(limit),
            float(v)
        )
    )


def vector_limit(
    vx,
    vy,
    vz,
    limit
):

    n = math.sqrt(
        vx * vx
        +
        vy * vy
        +
        vz * vz
    )

    if (
        n <= limit
        or
        n <= 1e-9
    ):
        return (
            vx,
            vy,
            vz
        )

    k = limit / n

    return (
        vx * k,
        vy * k,
        vz * k
    )


def slew(
    target,
    previous,
    step
):

    delta = clip(
        target - previous,
        step
    )

    return (
        previous
        +
        delta
    )


def image_error(source):

    try:

        du = float(
            source["du_px"]
        )

        dv = float(
            source["dv_px"]
        )

        return math.hypot(
            du,
            dv
        )

    except Exception:

        return float("inf")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--enable-output",
        action="store_true",
        help=(
            "Allow enabled=true to be published. "
            "The existing robot driver deadman is still required."
        )
    )

    args = parser.parse_args()


    print()
    print("=" * 72)
    print("R1-A7 DUAL CAMERA WIRE-GRASP CONTROL GATE")
    print("=" * 72)
    print("Input :", INPUT)
    print("Output:", OUTPUT)
    print("Velocity shaping: PASSTHROUGH (fusion velocity unchanged)")
    print(
        "Publish motion enabled:",
        args.enable_output
    )
    print()
    print(
        "IMPORTANT: existing robot deadman must still be held."
    )
    print(
        "No automatic gripper close."
    )
    print("=" * 72)
    print()


    last_vx = 0.0
    last_vy = 0.0
    last_vz = 0.0

    last_state = None


    while True:

        now = time.time()

        d = read_json(
            INPUT
        )


        try:

            input_age = (
                now -
                float(
                    d["timestamp"]
                )
            )

        except Exception:

            input_age = float("inf")


        ext = d.get(
            "external",
            {}
        )

        wrist = d.get(
            "wrist",
            {}
        )


        external_valid = bool(
            d.get(
                "external_valid",
                False
            )
        )

        wrist_valid = bool(
            d.get(
                "wrist_valid",
                False
            )
        )


        fusion_mode = str(
            d.get(
                "fusion_mode",
                "STOP"
            )
        )


        ext_error = image_error(
            ext
        )

        wrist_error = image_error(
            wrist
        )


        # Mode-aware validity checks.
        # Blind-grace motion remains disabled.
        source_fresh = bool(
            math.isfinite(input_age)
            and 0.0 <= input_age <= MAX_INPUT_AGE_S
        )

        if fusion_mode == "DUAL":
            perception_ok = bool(
                source_fresh
                and external_valid
                and wrist_valid
                and ext_error <= MAX_EXTERNAL_ERROR_PX
                and wrist_error <= MAX_WRIST_ERROR_PX
            )
        elif fusion_mode == "WRIST_ONLY":
            perception_ok = bool(
                source_fresh
                and not external_valid
                and wrist_valid
                and wrist_error <= MAX_WRIST_ERROR_PX
            )
        elif fusion_mode == "EXTERNAL_ONLY":
            perception_ok = bool(
                source_fresh
                and external_valid
                and not wrist_valid
                and ext_error <= MAX_EXTERNAL_ERROR_PX
            )
        else:
            perception_ok = False


        mode_speed_limit_mm_s = 0.0

        if perception_ok:
            try:
                requested_vx = float(d.get("proposed_vx_mm_s", 0.0))
                requested_vy = float(d.get("proposed_vy_mm_s", 0.0))
                requested_vz = float(d.get("proposed_vz_mm_s", 0.0))
            except (TypeError, ValueError):
                perception_ok = False

        if perception_ok:
            requested = (requested_vx, requested_vy, requested_vz)

            if not all(math.isfinite(v) for v in requested):
                perception_ok = False
            else:
                vx, vy, vz = requested
                mode_speed_limit_mm_s = 10.0

                speed_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
                if speed_norm > mode_speed_limit_mm_s and speed_norm > 0.0:
                    scale = mode_speed_limit_mm_s / speed_norm
                    vx *= scale
                    vy *= scale
                    vz *= scale

                gate_state = (
                    "DUAL_READY"
                    if fusion_mode == "DUAL"
                    else f"{fusion_mode}_READY"
                )

        if not perception_ok:
            vx = vy = vz = 0.0
            mode_speed_limit_mm_s = 0.0
            gate_state = "STOP"

        last_vx = float(vx)
        last_vy = float(vy)
        last_vz = float(vz)


        output_enabled = bool(
            args.enable_output
            and
            perception_ok
        )


        out = {

            "timestamp":
                now,

            "enabled":
                output_enabled,

            "source":
                "wire_grasp_dual_camera_v1",

            "ready_for_motion":
                perception_ok,

            "fusion_mode":
                fusion_mode,

            "max_speed_mm_s":
                float(mode_speed_limit_mm_s),

            "gate_state":
                gate_state,

            "external_valid":
                external_valid,

            "wrist_valid":
                wrist_valid,

            "input_age_s":
                (
                    input_age
                    if math.isfinite(input_age)
                    else None
                ),

            "external_error_px":
                (
                    ext_error
                    if math.isfinite(ext_error)
                    else None
                ),

            "wrist_error_px":
                (
                    wrist_error
                    if math.isfinite(wrist_error)
                    else None
                ),

            # Standard fields consumed by the old
            # deadman visual-servo drive.
            "vx_mm_s":
                float(vx)
                if output_enabled
                else 0.0,

            "vy_mm_s":
                float(vy)
                if output_enabled
                else 0.0,

            "vz_mm_s":
                float(vz)
                if output_enabled
                else 0.0,

            # Always preserve diagnostics.
            "proposed_vx_mm_s":
                float(vx),

            "proposed_vy_mm_s":
                float(vy),

            "proposed_vz_mm_s":
                float(vz),

            "tip_confidence":
                (
                    1.0
                    if perception_ok
                    else 0.0
                ),

            "hole_confidence":
                (
                    1.0
                    if perception_ok
                    else 0.0
                ),

            "du_px":
                wrist.get(
                    "du_px"
                ),

            "dv_px":
                wrist.get(
                    "dv_px"
                ),
        }


        atomic_write(
            out
        )


        if gate_state != last_state:

            print()
            print(
                "[CONTROL GATE]",
                last_state,
                "->",
                gate_state,
                flush=True
            )

            last_state = gate_state


        print(
            f"[{gate_state}] "
            f"EXT={'OK' if external_valid else 'NO'} "
            f"WRIST={'OK' if wrist_valid else 'NO'} "
            f"mode={fusion_mode} "
            f"age={input_age:.3f}s "
            f"v=({vx:+.2f},"
            f"{vy:+.2f},"
            f"{vz:+.2f}) "
            f"enabled={output_enabled}",
            flush=True
        )


        time.sleep(
            0.02
        )


if __name__ == "__main__":
    main()
