#!/usr/bin/env python3

from collections import deque

import numpy as np

from wire_local_detector_v65 import (
    WireLocalDetectorV65 as RawWireLocalDetectorV65
)


# ============================================================
# Temporal stabilization
#
# Important:
# - History is used only for filtering / candidate validation.
# - If CURRENT frame has no valid wire evidence, output remains
#   invalid.  Old target is NEVER reused as robot observation.
# ============================================================

MEDIAN_WINDOW = 5
EMA_ALPHA = 0.30

MAX_JUMP_PX = 15.0

RELOCK_FRAMES = 3
RELOCK_RADIUS_PX = 6.0
RELOCK_MAX_ANCHOR_PX = 40.0


class StablePoint:
    def __init__(self):
        self.raw_window = deque(
            maxlen=MEDIAN_WINDOW
        )

        self.ema = None

        self.relock = []

        self.state = "INIT"
        self.jump_px = None
        self.awaiting_relock = False
        self.reacquire_anchor = None


    def reset(self):
        self.raw_window.clear()
        self.ema = None
        self.relock.clear()
        self.state = "RESET"
        self.jump_px = None
        self.awaiting_relock = False
        self.reacquire_anchor = None


    def update(
        self,
        point,
        current_valid,
    ):
        # A lost observation invalidates temporal samples.
        # Keep only a separate anchor for guarded reacquisition.
        def invalidate(state):
            if self.ema is not None:
                self.reacquire_anchor = self.ema.copy()

            self.raw_window.clear()
            self.ema = None
            self.relock.clear()
            self.awaiting_relock = True
            self.state = state
            self.jump_px = None
            return None

        if not current_valid or point is None:
            return invalidate("NO_CURRENT_OBSERVATION")

        try:
            raw = np.asarray(point, dtype=np.float64)
        except (TypeError, ValueError):
            return invalidate("INVALID_POINT")

        if raw.shape != (2,) or not np.all(np.isfinite(raw)):
            return invalidate("NONFINITE_OR_BAD_SHAPE")

        # After loss, require a stable 3-frame cluster.
        if self.awaiting_relock:
            self.relock.append(raw.copy())
            self.relock = self.relock[-RELOCK_FRAMES:]

            if len(self.relock) < RELOCK_FRAMES:
                self.state = (
                    f"REACQUIRE_WAIT "
                    f"{len(self.relock)}/{RELOCK_FRAMES}"
                )
                return None

            arr = np.stack(self.relock, axis=0)
            center = np.mean(arr, axis=0)
            spread = float(np.max(
                np.linalg.norm(arr - center, axis=1)
            ))

            anchor_dist = (
                float(np.linalg.norm(
                    center - self.reacquire_anchor
                ))
                if self.reacquire_anchor is not None
                else 0.0
            )

            # After tracking loss, the object may have moved in
            # image coordinates. Require a stable current cluster;
            # keep the old anchor for diagnostics, not as a hard gate.
            if (
                spread <= RELOCK_RADIUS_PX
                and (
                    self.reacquire_anchor is None
                    or anchor_dist <= RELOCK_MAX_ANCHOR_PX
                )
            ):
                self.ema = center.copy()
                self.raw_window.clear()
                for sample in arr:
                    self.raw_window.append(sample.copy())

                self.relock.clear()
                self.awaiting_relock = False
                self.reacquire_anchor = None
                self.jump_px = anchor_dist
                self.state = (
                    f"RELOCK_AFTER_LOSS "
                    f"spread={spread:.1f} "
                    f"anchor={anchor_dist:.1f}"
                )
                return self._pixel(self.ema)

            self.state = (
                f"REACQUIRE_REJECT "
                f"spread={spread:.1f} "
                f"anchor={anchor_dist:.1f}"
            )

            # Keep the latest two samples to test the next cluster.
            self.relock = self.relock[-(RELOCK_FRAMES - 1):]
            return None

        # Initial observation before any loss.
        if self.ema is None:
            self.raw_window.append(raw.copy())
            self.ema = raw.copy()
            self.state = "INIT_ACCEPT"
            self.jump_px = 0.0
            return self._pixel(self.ema)

        # Gate the raw candidate BEFORE inserting it into the
        # normal median window, so rejected jumps cannot poison it.
        jump = float(np.linalg.norm(raw - self.ema))
        self.jump_px = jump

        if jump <= MAX_JUMP_PX:
            self.raw_window.append(raw.copy())
            arr = np.stack(list(self.raw_window), axis=0)
            median_point = np.median(arr, axis=0)

            self.ema = (
                EMA_ALPHA * median_point
                + (1.0 - EMA_ALPHA) * self.ema
            )
            self.relock.clear()
            self.state = f"TRACK jump={jump:.1f}"
            return self._pixel(self.ema)

        # Large jumps must form a stable cluster and remain close
        # enough to the previous accepted anchor.
        self.relock.append(raw.copy())
        self.relock = self.relock[-RELOCK_FRAMES:]

        self.state = (
            f"RELOCK_WAIT {len(self.relock)}/"
            f"{RELOCK_FRAMES} jump={jump:.1f}"
        )

        if len(self.relock) < RELOCK_FRAMES:
            return None

        arr = np.stack(self.relock, axis=0)
        center = np.mean(arr, axis=0)
        spread = float(np.max(
            np.linalg.norm(arr - center, axis=1)
        ))
        anchor_dist = float(np.linalg.norm(center - self.ema))

        if (
            spread <= RELOCK_RADIUS_PX
            and anchor_dist <= RELOCK_MAX_ANCHOR_PX
        ):
            self.ema = center.copy()
            self.raw_window.clear()
            for sample in arr:
                self.raw_window.append(sample.copy())

            self.relock.clear()
            self.state = (
                f"RELOCK spread={spread:.1f} "
                f"anchor={anchor_dist:.1f}"
            )
            self.jump_px = anchor_dist
            return self._pixel(self.ema)

        self.state = (
            f"RELOCK_REJECT spread={spread:.1f} "
            f"anchor={anchor_dist:.1f}"
        )

        self.relock = self.relock[-(RELOCK_FRAMES - 1):]
        return None



    @staticmethod
    def _pixel(p):

        return [
            int(
                round(
                    float(p[0])
                )
            ),
            int(
                round(
                    float(p[1])
                )
            ),
        ]


class WireLocalDetectorV65:
    """
    Drop-in wrapper.

    Existing callers can keep:

        WireLocalDetectorV65(...)
        set_seed_tip(...)
        detect(...)

    but returned wire_tip_uv / grasp_target_uv are stabilized.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        self.raw = (
            RawWireLocalDetectorV65(
                *args,
                **kwargs,
            )
        )

        self.tip_filter = (
            StablePoint()
        )

        self.grasp_filter = (
            StablePoint()
        )


    def __getattr__(
        self,
        name,
    ):

        return getattr(
            self.raw,
            name,
        )


    def set_seed_tip(
        self,
        *args,
        **kwargs,
    ):

        self.tip_filter.reset()
        self.grasp_filter.reset()

        return self.raw.set_seed_tip(
            *args,
            **kwargs,
        )


    def detect(
        self,
        *args,
        **kwargs,
    ):

        result = self.raw.detect(
            *args,
            **kwargs,
        )


        result = dict(
            result
        )


        raw_tip = result.get(
            "wire_tip_uv"
        )

        raw_grasp = result.get(
            "grasp_target_uv"
        )


        result[
            "raw_wire_tip_uv"
        ] = raw_tip

        result[
            "raw_grasp_target_uv"
        ] = raw_grasp


        raw_tip_valid = bool(
            result.get(
                "tip_valid",
                False,
            )
        )


        raw_grasp_valid = bool(
            result.get(
                "grasp_guidance_valid",
                False,
            )
        )


        stable_tip = (
            self.tip_filter.update(
                raw_tip,
                raw_tip_valid,
            )
        )


        stable_grasp = (
            self.grasp_filter.update(
                raw_grasp,
                raw_grasp_valid,
            )
        )


        result[
            "wire_tip_filter_state"
        ] = self.tip_filter.state

        result[
            "grasp_filter_state"
        ] = self.grasp_filter.state


        result[
            "wire_tip_filter_jump_px"
        ] = self.tip_filter.jump_px

        result[
            "grasp_filter_jump_px"
        ] = self.grasp_filter.jump_px


        # ----------------------------------------------------
        # Current-frame evidence is still mandatory.
        # ----------------------------------------------------

        if (
            raw_tip_valid
            and
            stable_tip is not None
        ):
            result[
                "wire_tip_uv"
            ] = stable_tip

            result[
                "tip_valid"
            ] = True

        else:
            result[
                "wire_tip_uv"
            ] = None

            result[
                "tip_valid"
            ] = False


        if (
            raw_grasp_valid
            and
            stable_grasp is not None
        ):
            result[
                "grasp_target_uv"
            ] = stable_grasp

            result[
                "grasp_guidance_valid"
            ] = True

        else:
            result[
                "grasp_target_uv"
            ] = None

            result[
                "grasp_guidance_valid"
            ] = False


        return result
