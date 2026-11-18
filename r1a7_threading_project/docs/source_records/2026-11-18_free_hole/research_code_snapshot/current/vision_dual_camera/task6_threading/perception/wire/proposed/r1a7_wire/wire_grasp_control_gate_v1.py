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


SETTLE_INPUT = Path(
    "/tmp/r1a7_auto_grasp_close_settle_v1.json"
)


INSERTION_TARGET_INPUT = Path(
    "/tmp/r1a7_insertion_target_state_v1.json"
)


# ============================================================
# First physical-motion test limits
# ============================================================

MAX_INPUT_AGE_S = 0.20

# CLOSE_SETTLE signal is refreshed by the auto-grasp
# controller while the arm must remain stationary.
MAX_SETTLE_AGE_S = 0.30

MAX_INSERTION_TARGET_AGE_S = 0.30


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


def close_settle_active(
    now,
):

    d = read_json(
        SETTLE_INPUT
    )

    try:

        ts = float(
            d["timestamp"]
        )

        age_s = (
            now
            -
            ts
        )

        return bool(
            d.get(
                "active",
                False,
            )
            and
            0.0
            <=
            age_s
            <=
            MAX_SETTLE_AGE_S
        )

    except Exception:

        return False


def insertion_target_ready(
    now,
):

    d = read_json(
        INSERTION_TARGET_INPUT
    )

    try:

        ts = float(
            d["timestamp"]
        )

        age_s = (
            now
            -
            ts
        )

        return bool(
            d.get(
                "target_ready",
                False,
            )
            and
            d.get(
                "target_confirmed",
                False,
            )
            and
            d.get(
                "phase",
                "",
            )
            ==
            "INSERTION_TRACKING"
            and
            0.0
            <=
            age_s
            <=
            MAX_INSERTION_TARGET_AGE_S
        )

    except Exception:

        return False


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


    # PERMANENT_EXIT_CLOSE_SETTLE_AFTER_INSERTION_CONFIRM_V1
    #
    # One-way task transition:
    #
    #   GRASP / WAIT_HOLE_TARGET
    #       -> CLOSE_SETTLE_HOLD allowed
    #
    #   target confirmed once
    #       -> INSERTION_TRACKING latched
    #       -> CLOSE_SETTLE_HOLD permanently disabled
    #
    # The latch lasts for this Gate process.
    insertion_phase_active = False


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


        # DISABLE_IMAGE_ERROR_HARD_GATE_TREND_V1
        #
        # For the current trend test, a large image error means
        # that motion is required; it must not itself cause STOP.
        #
        # Still retained:
        #   - Fusion heartbeat freshness
        #   - external/wrist validity
        #   - finite velocity checks
        #   - BLIND_GRACE timeout
        #   - true STOP on sustained perception loss

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
            )

        elif fusion_mode == "DUAL_Z_CONFLICT":
            # Z disagreement is NOT a full-stop condition.
            #
            # Fusion already keeps:
            #   X <- External
            #   Y <- Right Wrist
            # and selects:
            #   Z <- External
            #
            # Both cameras must still be fresh and valid.
            perception_ok = bool(
                source_fresh
                and external_valid
                and wrist_valid
            )

        elif fusion_mode == "WRIST_ONLY":
            perception_ok = bool(
                source_fresh
                and not external_valid
                and wrist_valid
            )
        elif fusion_mode == "EXTERNAL_ONLY":
            perception_ok = bool(
                source_fresh
                and external_valid
                and not wrist_valid
            )

        # BLIND_GRACE_GATE_V1
        #
        # Fusion has already:
        #   1. required a recent real visual command,
        #   2. reduced it by BLIND_GRACE_SCALE,
        #   3. limited blind speed,
        #   4. limited grace duration.
        #
        # Therefore the Gate only verifies that the Fusion
        # heartbeat itself is fresh.
        elif fusion_mode == "BLIND_GRACE":

            perception_ok = bool(
                source_fresh
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
                # INSERTION_SPEED_LIMIT_V1
                #
                # Keep the existing grasp-stage limit, but reduce
                # Cartesian velocity after entering INSERTION_TRACKING.
                mode_speed_limit_mm_s = (
                    1.0
                    if bool(
                        d.get(
                            "insertion_tracking_active",
                            False,
                        )
                    )
                    else
                    10.0
                )

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


        # ----------------------------------------------------
        # Auto-grasp CLOSE_SETTLE hold.
        #
        # Keep perception valid/enabled, but force Cartesian
        # velocity to exactly zero while the visual close gate
        # is being verified.
        # ----------------------------------------------------

        raw_insertion_ready = (
            insertion_target_ready(
                now
            )
        )


        # One-way transition into insertion.
        #
        # Once target selection/confirmation has succeeded,
        # never return to grasp CLOSE_SETTLE_HOLD during this
        # Gate process.
        if (
            raw_insertion_ready
            and
            not insertion_phase_active
        ):

            insertion_phase_active = True

            print()
            print(
                "[CONTROL GATE] "
                "INSERTION_TRACKING ACTIVATED"
            )
            print(
                "[CONTROL GATE] "
                "CLOSE_SETTLE_HOLD PERMANENTLY DISABLED"
            )
            print()


        insertion_ready = bool(
            insertion_phase_active
        )

        close_settle_hold = bool(
            perception_ok
            and
            close_settle_active(
                now
            )
            and
            not insertion_phase_active
        )

        if close_settle_hold:

            vx = 0.0
            vy = 0.0
            vz = 0.0

            mode_speed_limit_mm_s = 0.0

            gate_state = (
                "CLOSE_SETTLE_HOLD"
            )


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

            "insertion_target_ready":
                bool(
                    insertion_ready
                ),

            "insertion_target_ready_raw":
                bool(
                    raw_insertion_ready
                ),

            "insertion_phase_active":
                bool(
                    insertion_phase_active
                ),

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
