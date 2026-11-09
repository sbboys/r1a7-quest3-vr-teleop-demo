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


# GRASP_EXTERNAL_NEAR_Z_PRIORITY_V1
#
# During GRASP:
#
# Far from the External grasp target:
#   X <- External
#   Y <- Right Wrist
#   Z <- External/Wrist weighted fusion
#
# Once External is close to its fixed grasp target:
#   X <- External
#   Y <- Right Wrist
#   Z <- External ONLY
#
# This prevents a drifting Wrist grasp target from repeatedly
# moving the arm vertically after External is already aligned.
#
# Hysteresis prevents rapid mode switching near the threshold.
GRASP_EXT_NEAR_ENTER_PX = 45.0
GRASP_EXT_NEAR_EXIT_PX = 65.0

grasp_ext_near = False

INSERTION_STATE_FILE = Path(
    "/tmp/r1a7_insertion_target_state_v1.json"
)


# BLIND_GRACE_TREND_V1
#
# Trend-test purpose:
# bridge short visual dropouts with the most recent valid
# Cartesian direction instead of immediately outputting STOP.
#
# Real sustained visual loss still causes STOP.
BLIND_GRACE_TIME_S = 0.40
BLIND_GRACE_SCALE = 0.40
BLIND_GRACE_MAX_SPEED_MM_S = 2.0

last_valid_time = None
last_valid_velocity = (
    0.0,
    0.0,
    0.0,
)


def image_error_px(source):
    try:
        du = float(
            source.get(
                "du_px",
                source.get("du")
            )
        )
        dv = float(
            source.get(
                "dv_px",
                source.get("dv")
            )
        )
    except (TypeError, ValueError):
        return float("inf")

    if not (
        math.isfinite(du)
        and
        math.isfinite(dv)
    ):
        return float("inf")

    return math.sqrt(
        du * du
        +
        dv * dv
    )


def insertion_tracking_active():
    """
    Do not apply the grasp-stage External-Z priority after
    insertion has been confirmed.

    We intentionally use the phase value itself rather than a
    very short heartbeat timeout, because INSERTION_TRACKING is
    a task phase, not a one-frame condition.
    """
    d = read(
        INSERTION_STATE_FILE
    )

    return str(
        d.get(
            "phase",
            ""
        )
    ) == "INSERTION_TRACKING"


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


    # --------------------------------------------------------
    # GRASP External-near latch.
    #
    # Only meaningful while both visual sources are available.
    # Once insertion starts, restore the normal insertion
    # fusion behavior.
    # --------------------------------------------------------

    ext_error_px = image_error_px(
        ext
    )

    insertion_active = (
        insertion_tracking_active()
    )


    if insertion_active:

        grasp_ext_near = False

    elif ev:

        if grasp_ext_near:

            if (
                ext_error_px
                >=
                GRASP_EXT_NEAR_EXIT_PX
            ):
                grasp_ext_near = False

        else:

            if (
                ext_error_px
                <=
                GRASP_EXT_NEAR_ENTER_PX
            ):
                grasp_ext_near = True


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

        # ====================================================
        # GRASP_EXTERNAL_NEAR_Z_PRIORITY_V1
        #
        # During final GRASP alignment, once External says that
        # its fixed grasp point is already close, Wrist is only
        # allowed to control Base-Y (left/right).
        #
        # Base-Z (up/down) is then controlled by External only.
        #
        # During INSERTION_TRACKING the original dual-camera
        # Z fusion is restored.
        # ====================================================

        if (
            grasp_ext_near
            and
            not insertion_active
        ):

            # External:
            #   X unchanged
            #   Z full authority
            #
            # Wrist:
            #   Y unchanged
            #   Z weight = 0
            vz = ez

        elif (
            ez == 0.0
            or
            wz == 0.0
            or
            ez * wz > 0
        ):

            # Normal far-range / insertion Z fusion.
            vz = (
                0.35 * ez
                +
                0.65 * wz
            )

        else:

            # Z disagreement:
            # keep External-X and Wrist-Y active,
            # trust External for Base-Z.
            vz = ez

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

        # Both visual sources are currently invalid.
        #
        # Do NOT immediately inject (0,0,0) for a short
        # dropout.  Reuse the most recent VALID direction
        # at reduced speed.
        if last_valid_time is None:

            grace_age_s = float("inf")

        else:

            grace_age_s = (
                now
                -
                last_valid_time
            )


        if (
            0.0
            <= grace_age_s
            <= BLIND_GRACE_TIME_S
        ):

            mode = "BLIND_GRACE"

            vx = (
                last_valid_velocity[0]
                *
                BLIND_GRACE_SCALE
            )

            vy = (
                last_valid_velocity[1]
                *
                BLIND_GRACE_SCALE
            )

            vz = (
                last_valid_velocity[2]
                *
                BLIND_GRACE_SCALE
            )


            # Additional blind-motion vector limit.
            n = math.sqrt(
                vx*vx
                +
                vy*vy
                +
                vz*vz
            )

            if (
                n
                >
                BLIND_GRACE_MAX_SPEED_MM_S
                and
                n > 1e-9
            ):

                k = (
                    BLIND_GRACE_MAX_SPEED_MM_S
                    /
                    n
                )

                vx *= k
                vy *= k
                vz *= k

        else:

            mode = "STOP"

            vx = 0.0
            vy = 0.0
            vz = 0.0


    (
        vx,
        vy,
        vz
    ) = vector_limit(
        vx,
        vy,
        vz
    )


    # Update last known GOOD visual velocity only from an
    # actually valid visual mode.  BLIND_GRACE itself must
    # never refresh the grace timer.
    if mode in (
        "DUAL",
        "DUAL_Z_CONFLICT",
        "EXTERNAL_ONLY",
        "WRIST_ONLY",
    ):

        last_valid_time = now

        last_valid_velocity = (
            float(vx),
            float(vy),
            float(vz),
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
                ev
                or
                wv
                or
                mode == "BLIND_GRACE"
            ),

        "fusion_mode":
            mode,

        "external_valid":
            ev,

        "wrist_valid":
            wv,

        "grasp_ext_near":
            bool(
                grasp_ext_near
            ),

        "external_error_px":
            (
                float(ext_error_px)
                if math.isfinite(ext_error_px)
                else None
            ),

        "insertion_tracking_active":
            bool(
                insertion_active
            ),

        "z_control_source":
            (
                "EXTERNAL_ONLY_GRASP_NEAR"
                if (
                    grasp_ext_near
                    and
                    not insertion_active
                )
                else
                "DUAL_NORMAL"
            ),

        "external_z_weight":
            (
                1.0
                if (
                    grasp_ext_near
                    and
                    not insertion_active
                )
                else
                0.35
            ),

        "wrist_z_weight":
            (
                0.0
                if (
                    grasp_ext_near
                    and
                    not insertion_active
                )
                else
                0.65
            ),

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
        f"{vz:+.2f}) mm/s "
        f"EXT_ERR={ext_error_px:.1f}px "
        f"Z={'EXT_ONLY' if (grasp_ext_near and not insertion_active) else 'DUAL'}",
        flush=True
    )


    time.sleep(
        0.05
    )
