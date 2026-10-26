#!/usr/bin/env python3

import numpy as np

from wire_local_detector_v61 import (
    WireLocalDetectorV61,
)


class WireLocalDetectorV62(
    WireLocalDetectorV61
):
    """
    V6.2

    Critical fix:
      seed_tip / last_valid_tip are FULL-IMAGE coordinates.

      Candidate endpoints inside _candidate_components()
      are ROI-LOCAL coordinates.

    V6.2 temporarily converts the reference tip into ROI-local
    coordinates only while candidate extraction/ranking runs.

    After candidate extraction the reference is restored to
    full-image coordinates, so V5/V6 jump gates still operate
    correctly in full-image coordinates.
    """

    def __init__(
        self,
        local_length_px=120.0,
        grasp_offset_px=30.0,
    ):

        super().__init__(
            local_length_px=
                local_length_px,
            grasp_offset_px=
                grasp_offset_px,
        )

        self._roi_origin = np.array(
            [0.0, 0.0],
            dtype=np.float64
        )


    def _candidate_components(
        self,
        gray,
        response,
    ):

        offset = (
            self._roi_origin.copy()
        )

        seed_full = (
            None
            if self.seed_tip is None
            else
            np.asarray(
                self.seed_tip,
                dtype=np.float64
            ).copy()
        )

        last_full = (
            None
            if self.last_valid_tip is None
            else
            np.asarray(
                self.last_valid_tip,
                dtype=np.float64
            ).copy()
        )

        try:

            # --------------------------------------------
            # FULL IMAGE -> ROI LOCAL
            # --------------------------------------------

            if seed_full is not None:

                self.seed_tip = (
                    seed_full
                    -
                    offset
                )

            if last_full is not None:

                self.last_valid_tip = (
                    last_full
                    -
                    offset
                )

            candidates = (
                super()
                ._candidate_components(
                    gray,
                    response
                )
            )

        finally:

            # --------------------------------------------
            # Restore FULL IMAGE coordinates.
            # --------------------------------------------

            self.seed_tip = (
                seed_full
            )

            self.last_valid_tip = (
                last_full
            )

        if not candidates:

            # Do not leave stale diagnostic values.
            self.last_rank_score = 0.0
            self.last_profile_score = 0.0
            self.last_two_sided_fraction = 0.0
            self.last_edge_overlap = 0.0
            self.debug_path_local = None

        return candidates


    def detect(
        self,
        frame,
        roi
    ):

        x, y, w, h = [
            int(v)
            for v in roi
        ]

        self._roi_origin = np.array(
            [
                float(x),
                float(y)
            ],
            dtype=np.float64
        )

        return super().detect(
            frame,
            roi
        )
