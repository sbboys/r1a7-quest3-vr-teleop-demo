#!/usr/bin/env python3

import cv2
import numpy as np

from tweezer_mask_tracker_v67 import (
    TweezerMaskTrackerV67,
)


class TweezerMaskTrackerV68(
    TweezerMaskTrackerV67
):
    """
    V6.8

    Rigid right-wrist tool model.

    The calibrated pixel is authoritative:
        tweezer_grasp_center_uv = fixed calibrated pixel

    SAM2 is used ONLY for:
        - tweezer semantic mask
        - wire exclusion mask

    SAM2 / PCA / refiner are NOT allowed to move the
    grasp-center pixel.
    """

    def __init__(
        self,
        init_frame
    ):

        super().__init__(
            init_frame
        )

        # The V6.7 nominal tip was loaded from
        # wrist_tip_calibration.json.
        self.fixed_grasp_center = (
            self.nominal_tip.copy()
        )

        print()
        print(
            "[TWEEZER V68] FIXED GRASP CENTER =",
            tuple(
                np.rint(
                    self.fixed_grasp_center
                ).astype(int)
            ),
            flush=True
        )

        print(
            "[TWEEZER V68] SAM2 = MASK ONLY",
            flush=True
        )


    def update(
        self,
        frame
    ):

        # Let previous code obtain a SAM2 semantic mask.
        result = super().update(
            frame
        )

        fixed_xy = tuple(
            np.rint(
                self.fixed_grasp_center
            ).astype(int)
        )

        # ====================================================
        # CRITICAL V6.8 RULE:
        #
        # No visual detector may redefine the physical
        # tweezer grasp center.
        # ====================================================

        self.filtered_tip = (
            self.fixed_grasp_center.copy()
        )

        result[
            "tip"
        ] = fixed_xy

        result[
            "tweezer_grasp_center_uv"
        ] = [
            int(fixed_xy[0]),
            int(fixed_xy[1]),
        ]

        result[
            "fixed_tip_uv"
        ] = [
            int(fixed_xy[0]),
            int(fixed_xy[1]),
        ]

        result[
            "tip_offset_from_fixed_px"
        ] = 0.0

        # ====================================================
        # Rebuild exclusion mask around the FIXED grasp center.
        #
        # Do not use a drifting SAM2/PCA tip for this aperture.
        # ====================================================

        raw_mask = result.get(
            "raw_mask"
        )

        if raw_mask is not None:

            raw_mask = np.asarray(
                raw_mask,
                dtype=np.uint8
            )

            kernel = (
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (13, 13)
                )
            )

            exclusion = cv2.dilate(
                raw_mask,
                kernel,
                iterations=1
            )

            # Open a small window at the TRUE physical
            # grasp center so the wire can enter the jaws.
            cv2.circle(
                exclusion,
                fixed_xy,
                16,
                0,
                -1
            )

            result[
                "exclusion_mask"
            ] = exclusion

            self.last_exclusion_mask = (
                exclusion.copy()
            )

        # Mask state and grasp-center state are now separated.
        if result.get(
            "valid",
            False
        ):

            result[
                "state"
            ] = "TWEEZER_MASK_TRACK_FIXED_CENTER"

        return result
