#!/usr/bin/env python3

import json
import math
import os
import time
from pathlib import Path


EXT_FILE = Path("/tmp/r1a7_ibvs_velocity.json")
WRIST_FILE = Path("/tmp/r1a7_wrist_ibvs_velocity.json")
OUT_FILE = Path("/tmp/r1a7_dual_ibvs_velocity.json")


# ============================================================
# Global
# ============================================================

# First test with True.
# After confirming state switching, change to False.
SHADOW_MODE = False

EXT_MAX_AGE_S = 0.35
WRIST_MAX_AGE_S = 0.35

EXT_Z_WEIGHT = 0.7
WRIST_Z_WEIGHT = 0.3

# Vision confidence thresholds
TIP_CONF_MIN = 0.5
HOLE_CONF_MIN = 0.5

# Motion direction consistency
DIRECTION_CONFLICT_DOT = -0.2


# ============================================================
# Single-camera fallback
# ============================================================

# If only one camera survives, continue moving,
# but use a lower fallback speed.
SINGLE_CAMERA_MAX_MM_S = 2.0


# ============================================================
# Both-camera-loss grace mode
# ============================================================

# Both cameras lost:
# continue briefly instead of instantly stopping.
BLIND_GRACE_S = 0.30

# Blind continuation must be slow.
BLIND_MAX_SPEED_MM_S = 0.50

# If we were already near the target, do NOT blind continue.
# Use cached last valid image error.
NEAR_TARGET_ERROR_PX = 80.0


def read_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def source_age(data, now):
    try:
        return max(
            0.0,
            now - float(data["timestamp"])
        )
    except Exception:
        return float("inf")


def source_valid(data, age, max_age):

    if not (
        bool(data)
        and bool(data.get("enabled", False))
        and math.isfinite(age)
        and age <= max_age
    ):
        return False

    tip_conf = float(
        data.get(
            "tip_confidence",
            0.0
        )
    )

    hole_conf = float(
        data.get(
            "hole_confidence",
            0.0
        )
    )

    return (
        tip_conf >= TIP_CONF_MIN
        and
        hole_conf >= HOLE_CONF_MIN
    )



def visual_confidence(data):

    if not data:
        return 0.0

    tip = float(
        data.get(
            "tip_confidence",
            0.0
        )
    )

    hole = float(
        data.get(
            "hole_confidence",
            0.0
        )
    )

    return max(
        0.0,
        min(
            1.0,
            0.5 * (tip + hole)
        )
    )



def direction_consistent(v1, v2):

    if abs(v1) < 1e-6 or abs(v2) < 1e-6:
        return True

    return (
        (v1 * v2)
        >
        0
    )


def clip(v, lim):
    return max(
        -float(lim),
        min(float(lim), float(v))
    )


def vector_limit(vx, vy, vz, max_speed):
    norm = math.sqrt(
        vx * vx +
        vy * vy +
        vz * vz
    )

    if norm <= max_speed or norm <= 1e-9:
        return vx, vy, vz

    s = max_speed / norm

    return (
        vx * s,
        vy * s,
        vz * s,
    )


def error_norm(data):
    du = data.get(
        "du_px",
        data.get("du")
    )

    dv = data.get(
        "dv_px",
        data.get("dv")
    )

    try:
        return math.hypot(
            float(du),
            float(dv),
        )
    except Exception:
        return float("inf")


def atomic_write(data):
    tmp = Path(str(OUT_FILE) + ".tmp")

    tmp.write_text(
        json.dumps(
            data,
            indent=2,
        )
    )

    os.replace(
        tmp,
        OUT_FILE,
    )


def main():

    print(
        "[FAULT-TOLERANT DUAL IBVS START]",
        flush=True,
    )

    last_mode = None

    # Last reliable Cartesian velocity.
    last_safe_vx = 0.0
    last_safe_vy = 0.0
    last_safe_vz = 0.0

    last_ext_error = float("inf")
    last_wrist_error = float("inf")

    both_lost_since = None

    while True:

        now = time.time()

        ext = read_json(EXT_FILE)
        wrist = read_json(WRIST_FILE)

        ext_age = source_age(
            ext,
            now,
        )

        wrist_age = source_age(
            wrist,
            now,
        )

        ext_valid = source_valid(
            ext,
            ext_age,
            EXT_MAX_AGE_S,
        )

        wrist_valid = source_valid(
            wrist,
            wrist_age,
            WRIST_MAX_AGE_S,
        )

        # ----------------------------------------------------
        # Read source velocities
        # ----------------------------------------------------

        vx_ext = (
            float(ext.get("vx_mm_s", 0.0))
            if ext_valid else 0.0
        )

        vz_ext = (
            float(ext.get("vz_mm_s", 0.0))
            if ext_valid else 0.0
        )

        vy_wrist = (
            float(wrist.get("vy_mm_s", 0.0))
            if wrist_valid else 0.0
        )

        vz_wrist = (
            float(wrist.get("vz_mm_s", 0.0))
            if wrist_valid else 0.0
        )

        if ext_valid:
            last_ext_error = error_norm(ext)

        if wrist_valid:
            last_wrist_error = error_norm(wrist)

        # ====================================================
        # MODE 1: both valid
        # ====================================================

        if ext_valid and wrist_valid:

            mode = "DUAL"

            both_lost_since = None

            proposed_vx = vx_ext
            proposed_vy = vy_wrist

            ext_conf = visual_confidence(ext)
            wrist_conf = visual_confidence(wrist)

            conf_sum = (
                ext_conf
                +
                wrist_conf
            )

            if conf_sum > 1e-6:

                ext_weight = (
                    ext_conf
                    /
                    conf_sum
                )

                wrist_weight = (
                    wrist_conf
                    /
                    conf_sum
                )

            else:

                ext_weight = EXT_Z_WEIGHT
                wrist_weight = WRIST_Z_WEIGHT


            # ============================================
            # Motion direction consistency protection
            #
            # If both cameras agree:
            #     confidence weighted fusion
            #
            # If directions conflict:
            #     trust the higher-confidence source
            # ============================================

            if direction_consistent(
                vz_ext,
                vz_wrist
            ):

                proposed_vz = (
                    ext_weight * vz_ext
                    +
                    wrist_weight * vz_wrist
                )

            else:

                if ext_conf >= wrist_conf:

                    proposed_vz = vz_ext

                    mode = "DUAL_EXT_PRIORITY"

                else:

                    proposed_vz = vz_wrist

                    mode = "DUAL_WRIST_PRIORITY"

            ready = True

        # ====================================================
        # MODE 2: only external valid
        # ====================================================

        elif ext_valid and not wrist_valid:

            mode = "EXTERNAL_ONLY"

            both_lost_since = None

            # External controls X/Z.
            # Wrist Y information is unavailable -> Y=0.
            proposed_vx = clip(
                vx_ext,
                SINGLE_CAMERA_MAX_MM_S,
            )

            proposed_vy = 0.0

            proposed_vz = clip(
                vz_ext,
                SINGLE_CAMERA_MAX_MM_S,
            )

            ready = True

        # ====================================================
        # MODE 3: only wrist valid
        # ====================================================

        elif wrist_valid and not ext_valid:

            mode = "WRIST_ONLY"

            both_lost_since = None

            # External X information is unavailable -> X=0.
            proposed_vx = 0.0

            proposed_vy = clip(
                vy_wrist,
                SINGLE_CAMERA_MAX_MM_S,
            )

            proposed_vz = clip(
                vz_wrist,
                SINGLE_CAMERA_MAX_MM_S,
            )

            ready = True

        # ====================================================
        # MODE 4: both lost
        # ====================================================

        else:

            if both_lost_since is None:
                both_lost_since = now

            lost_time = (
                now - both_lost_since
            )

            near_target = (
                last_ext_error
                <= NEAR_TARGET_ERROR_PX
                or
                last_wrist_error
                <= NEAR_TARGET_ERROR_PX
            )

            if (
                lost_time <= BLIND_GRACE_S
                and
                not near_target
            ):

                mode = "BLIND_GRACE"

                # Gradually decay:
                # 1.0 -> 0.0 during the grace interval.
                scale = max(
                    0.0,
                    1.0
                    -
                    lost_time
                    /
                    BLIND_GRACE_S
                )

                # Do NOT continue blind Y correction.
                # Y requires wrist information.
                proposed_vx = (
                    last_safe_vx
                    *
                    scale
                )

                proposed_vy = 0.0

                proposed_vz = (
                    last_safe_vz
                    *
                    scale
                )

                (
                    proposed_vx,
                    proposed_vy,
                    proposed_vz,
                ) = vector_limit(
                    proposed_vx,
                    proposed_vy,
                    proposed_vz,
                    BLIND_MAX_SPEED_MM_S,
                )

                ready = True

            else:

                mode = "STOP"

                proposed_vx = 0.0
                proposed_vy = 0.0
                proposed_vz = 0.0

                ready = False

        # ----------------------------------------------------
        # Save last reliable velocity.
        #
        # Do NOT learn from blind motion itself.
        # ----------------------------------------------------

        if mode in (
            "DUAL",
            "EXTERNAL_ONLY",
            "WRIST_ONLY",
        ):

            last_safe_vx = float(
                proposed_vx
            )

            last_safe_vy = float(
                proposed_vy
            )

            last_safe_vz = float(
                proposed_vz
            )

        # ----------------------------------------------------
        # Actual output
        # ----------------------------------------------------

        if SHADOW_MODE or not ready:

            enabled = False

            vx = 0.0
            vy = 0.0
            vz = 0.0

        else:

            enabled = True

            vx = proposed_vx
            vy = proposed_vy
            vz = proposed_vz

        out = {
            "timestamp": now,

            "enabled": enabled,
            "shadow_mode": SHADOW_MODE,
            "ready_for_motion": ready,

            "fusion_mode": mode,

            "external_valid": ext_valid,
            "wrist_valid": wrist_valid,

            "external_age_s": ext_age,
            "wrist_age_s": wrist_age,

            "last_external_error_px": (
                last_ext_error
                if math.isfinite(last_ext_error)
                else None
            ),

            "last_wrist_error_px": (
                last_wrist_error
                if math.isfinite(last_wrist_error)
                else None
            ),

            "vx_mm_s": float(vx),
            "vy_mm_s": float(vy),
            "vz_mm_s": float(vz),

            "proposed_vx_mm_s":
                float(proposed_vx),

            "proposed_vy_mm_s":
                float(proposed_vy),

            "proposed_vz_mm_s":
                float(proposed_vz),

            "external": ext,
            "wrist": wrist,
        }

        atomic_write(out)

        if mode != last_mode:

            print(
                f"[FUSION MODE] "
                f"{last_mode} -> {mode}",
                flush=True,
            )

            last_mode = mode

        print(
            f"[{mode}] "
            f"EXT={'OK' if ext_valid else 'NO'} "
            f"age={ext_age:.3f} | "
            f"WRIST={'OK' if wrist_valid else 'NO'} "
            f"age={wrist_age:.3f} | "
            f"v=({vx:+.3f},"
            f"{vy:+.3f},"
            f"{vz:+.3f}) mm/s",
            flush=True,
        )

        time.sleep(0.01)


if __name__ == "__main__":
    main()
