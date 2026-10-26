#!/usr/bin/env python3

import cv2
import numpy as np

from wire_local_detector_v62 import (
    WireLocalDetectorV62,
)


class WireLocalDetectorV63(
    WireLocalDetectorV62
):
    """
    V6.3

    Adds an explicit tweezer exclusion mask.

    The wire detector is not allowed to build ridge/skeleton
    candidates on pixels belonging to the tweezer.

    This solves the semantic identity-switch problem:
        wire -> tweezer jaw
    """

    def __init__(
        self,
        local_length_px=120.0,
        grasp_offset_px=30.0,
    ):

        super().__init__(
            local_length_px=local_length_px,
            grasp_offset_px=grasp_offset_px,
        )

        self._exclusion_crop = None
        self._mask_fraction = 0.0


    def _wire_response(
        self,
        crop
    ):

        gray, response = (
            super()._wire_response(
                crop
            )
        )

        if (
            self._exclusion_crop is not None
            and
            self._exclusion_crop.shape[:2]
            ==
            response.shape[:2]
        ):

            forbidden = (
                self._exclusion_crop
                >
                0
            )

            self._mask_fraction = float(
                np.mean(
                    forbidden
                )
            )

            # Hard negative mask.
            response[
                forbidden
            ] = 0.0

        else:

            self._mask_fraction = 0.0

        return (
            gray,
            response
        )


    def detect(
        self,
        frame,
        roi,
        tweezer_exclusion_mask=None,
    ):

        x, y, w, h = [
            int(v)
            for v in roi
        ]

        if (
            tweezer_exclusion_mask
            is not None
        ):

            self._exclusion_crop = (
                tweezer_exclusion_mask[
                    y:y+h,
                    x:x+w
                ]
            )

        else:

            self._exclusion_crop = None

        result = super().detect(
            frame,
            roi
        )

        result[
            "tweezer_mask_fraction"
        ] = float(
            self._mask_fraction
        )

        return result
