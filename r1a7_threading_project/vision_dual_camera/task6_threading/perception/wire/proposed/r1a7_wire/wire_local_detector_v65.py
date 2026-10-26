#!/usr/bin/env python3

import numpy as np

from wire_local_detector_v64 import (
    WireLocalDetectorV64,
)


class WireLocalDetectorV65(
    WireLocalDetectorV64
):
    """
    V6.5

    Fixes:
      1. ROI-clipped endpoint cannot become physical free tip.
      2. When free tip is occluded, do not redefine a visible
         segment endpoint as the new free tip.
      3. Keep last_confirmed_free_tip_uv only as history.
      4. Stabilize grasp target while current-image wire
         guidance remains valid.
    """

    def __init__(
        self,
        local_length_px=180.0,
        grasp_offset_px=55.0,
        grasp_band_start_px=35.0,
        grasp_band_end_px=90.0,
    ):

        super().__init__(
            local_length_px=local_length_px,
            grasp_offset_px=grasp_offset_px,
            grasp_band_start_px=grasp_band_start_px,
            grasp_band_end_px=grasp_band_end_px,
        )

        self.confirmed_free_tip_uv = None
        self.last_guidance_target_uv = None

        self.roi_edge_margin_px = 14.0

        # When tweezer reaches the free-tip area, stop
        # updating physical free-tip identity.
        self.tip_tweezer_occlusion_px = 26.0

        # Only used to smooth two CURRENT-image targets.
        self.target_ema_alpha = 0.65

        # An occlusion-mode target may not suddenly jump.
        self.occluded_target_max_jump_px = 35.0


    def set_seed_tip(
        self,
        uv
    ):

        super().set_seed_tip(
            uv
        )

        self.confirmed_free_tip_uv = (
            np.asarray(
                uv,
                dtype=np.float64
            ).copy()
        )

        self.last_guidance_target_uv = None


    @staticmethod
    def _point_distance(
        a,
        b
    ):

        return float(
            np.linalg.norm(
                np.asarray(
                    a,
                    dtype=np.float64
                )
                -
                np.asarray(
                    b,
                    dtype=np.float64
                )
            )
        )


    def _tip_near_roi_boundary(
        self,
        tip,
        roi
    ):

        if tip is None:
            return True

        u, v = [
            float(x)
            for x in tip
        ]

        x, y, w, h = [
            float(x)
            for x in roi
        ]

        m = float(
            self.roi_edge_margin_px
        )

        return bool(
            u <= x + m
            or
            u >= x + w - m
            or
            v <= y + m
            or
            v >= y + h - m
        )


    def _stabilize_grasp_target(
        self,
        result
    ):

        target = result.get(
            "grasp_target_uv"
        )

        guidance = bool(
            result.get(
                "grasp_guidance_valid",
                False
            )
        )

        if (
            not guidance
            or
            target is None
        ):
            # Keep the previous target for diagnostics only.
            # Never treat it as a current visual observation.
            held = (
                np.rint(self.last_guidance_target_uv)
                .astype(int)
                .tolist()
                if self.last_guidance_target_uv is not None
                else None
            )

            result["grasp_target_uv"] = held
            result["grasp_guidance_valid"] = False
            result["grasp_target_measurement_valid"] = False
            result["grasp_target_held"] = held is not None
            result["grasp_target_state"] = "HOLD_INVALID"

            return result

        current = np.asarray(
            target,
            dtype=np.float64
        )

        tip_visible = bool(
            result.get(
                "tip_valid",
                False
            )
        )

        # Validate the current physical free tip BEFORE initializing
        # or updating the grasp-target history.
        if not tip_visible:
            held = (
                np.rint(self.last_guidance_target_uv)
                .astype(int)
                .tolist()
                if self.last_guidance_target_uv is not None
                else None
            )

            jump = (
                float(np.linalg.norm(
                    current - self.last_guidance_target_uv
                ))
                if self.last_guidance_target_uv is not None
                else None
            )

            result["grasp_target_uv"] = held
            result["grasp_guidance_valid"] = False
            result["grasp_target_measurement_valid"] = False
            result["grasp_target_held"] = held is not None
            result["grasp_target_state"] = "HOLD_INVALID"
            result["grasp_target_jump_px"] = jump

            # Never initialize or update target history from
            # an occluded or untrusted endpoint.
            return result

        if self.last_guidance_target_uv is None:
            self.last_guidance_target_uv = current.copy()
            result["grasp_target_measurement_valid"] = True
            result["grasp_target_held"] = False
            result["grasp_target_state"] = "TRACK_INIT"
            result["grasp_target_jump_px"] = 0.0
            return result

        jump = float(
            np.linalg.norm(
                current
                -
                self.last_guidance_target_uv
            )
        )

        # Normal tracking: current physical free tip is trusted.
        if tip_visible:
            result["grasp_target_measurement_valid"] = True
            result["grasp_target_held"] = False

            filtered = (
                self.target_ema_alpha
                *
                current
                +
                (
                    1.0
                    -
                    self.target_ema_alpha
                )
                *
                self.last_guidance_target_uv
            )

            self.last_guidance_target_uv = (
                filtered.copy()
            )

            result[
                "grasp_target_uv"
            ] = (
                np.rint(
                    filtered
                )
                .astype(int)
                .tolist()
            )

            result[
                "grasp_target_jump_px"
            ] = jump

            return result

        # During occlusion, a sudden target jump means the
        # visible segment identity is uncertain.
        if (
            jump
            >
            self.occluded_target_max_jump_px
        ):

            result[
                "grasp_guidance_valid"
            ] = False

            result[
                "grasp_target_uv"
            ] = None

            result[
                "state"
            ] = (
                "OCCLUDED_TARGET_JUMP_REJECT"
            )

            result[
                "grasp_target_jump_px"
            ] = jump

            return result

        filtered = (
            self.target_ema_alpha
            *
            current
            +
            (
                1.0
                -
                self.target_ema_alpha
            )
            *
            self.last_guidance_target_uv
        )

        self.last_guidance_target_uv = (
            filtered.copy()
        )

        result[
            "grasp_target_uv"
        ] = (
            np.rint(
                filtered
            )
            .astype(int)
            .tolist()
        )

        result[
            "grasp_target_jump_px"
        ] = jump

        return result


    def detect(
        self,
        frame,
        roi,
        tweezer_exclusion_mask=None,
        tweezer_tip=None,
    ):

        result = super().detect(
            frame,
            roi,
            tweezer_exclusion_mask=
                tweezer_exclusion_mask,
            tweezer_tip=
                tweezer_tip,
        )

        # ====================================================
        # A. Normal detector says the free tip is visible.
        # Verify that it is actually a physical endpoint.
        # ====================================================

        if (
            result.get(
                "tip_valid",
                False
            )
            and
            result.get(
                "wire_tip_uv"
            )
            is not None
        ):

            tip = np.asarray(
                result[
                    "wire_tip_uv"
                ],
                dtype=np.float64
            )

            # ----------------------------------------------
            # ROI clipping protection.
            #
            # A centerline endpoint near the crop boundary
            # is only a CROP endpoint, not a physical tip.
            # ----------------------------------------------

            if self._tip_near_roi_boundary(
                tip,
                roi
            ):

                result[
                    "tip_valid"
                ] = False

                result[
                    "valid"
                ] = False

                result[
                    "wire_tip_uv"
                ] = None

                result[
                    "state"
                ] = (
                    "ROI_CLIPPED_VISIBLE_SEGMENT"
                )

                result[
                    "visibility_mode"
                ] = (
                    "TIP_UNTRUSTED_ROI_EDGE"
                )

                result[
                    "grasp_guidance_valid"
                ] = bool(
                    result.get(
                        "grasp_target_uv"
                    )
                    is not None
                    and
                    len(
                        result.get(
                            "centerline",
                            []
                        )
                    )
                    >=
                    2
                )

            # ----------------------------------------------
            # Tweezer is already covering the free-tip zone.
            # Freeze free-tip identity instead of allowing
            # segmentation geometry to redefine the endpoint.
            # ----------------------------------------------

            elif (
                tweezer_tip is not None
                and
                self._point_distance(
                    tip,
                    tweezer_tip
                )
                <
                self.tip_tweezer_occlusion_px
            ):

                result[
                    "tip_valid"
                ] = False

                result[
                    "valid"
                ] = False

                result[
                    "wire_tip_uv"
                ] = None

                result[
                    "state"
                ] = (
                    "TIP_NEAR_TWEEZER"
                )

                result[
                    "visibility_mode"
                ] = (
                    "TIP_OCCLUDED"
                )

                result[
                    "grasp_guidance_valid"
                ] = bool(
                    result.get(
                        "grasp_target_uv"
                    )
                    is not None
                    and
                    len(
                        result.get(
                            "centerline",
                            []
                        )
                    )
                    >=
                    2
                )

            else:

                # This is a genuine currently visible
                # physical free endpoint.
                self.confirmed_free_tip_uv = (
                    tip.copy()
                )

        # ====================================================
        # B. Always expose the historical confirmed tip
        # separately.
        #
        # Do NOT pretend it is the current visible tip.
        # ====================================================

        result[
            "last_confirmed_free_tip_uv"
        ] = (
            np.rint(
                self.confirmed_free_tip_uv
            )
            .astype(int)
            .tolist()
            if
            self.confirmed_free_tip_uv
            is not None
            else
            None
        )

        # ====================================================
        # C. Stabilize current-image grasp target.
        # ====================================================

        result = (
            self._stabilize_grasp_target(
                result
            )
        )

        return result
