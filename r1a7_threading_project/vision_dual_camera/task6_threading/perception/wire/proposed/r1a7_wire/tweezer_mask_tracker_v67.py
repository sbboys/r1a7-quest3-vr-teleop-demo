#!/usr/bin/env python3

import json
import numpy as np

from tweezer_mask_tracker_v66 import (
    TweezerMaskTrackerV66,
    wrist,
)


class TweezerMaskTrackerV67(
    TweezerMaskTrackerV66
):
    """
    V6.7

    Restore the original R1-A7 wrist-tip philosophy:

      - camera and tweezer are rigidly mounted
      - calibrated tip pixel is the primary reference
      - SAM2 is mainly used for tweezer MASK
      - image refinement is only allowed inside a tiny gate
      - a bad SAM2/PCA candidate can never redefine the tip

    Previous Task6 value:
        TIP_MAX_JUMP_PX = 8 px
    """

    def __init__(
        self,
        init_frame
    ):

        # Still initialize SAM2 mask ROI / prompts.
        super().__init__(
            init_frame
        )

        # ----------------------------------------------------
        # Load the already existing fixed wrist-tip calibration.
        # ----------------------------------------------------

        if not wrist.TIP_FILE.exists():

            raise RuntimeError(
                f"Missing fixed tip calibration: "
                f"{wrist.TIP_FILE}"
            )

        data = json.loads(
            wrist.TIP_FILE.read_text()
        )

        fixed_tip = np.asarray(
            data["tip_pixel"],
            dtype=np.float64
        )

        self.nominal_tip = (
            fixed_tip.copy()
        )

        self.filtered_tip = (
            fixed_tip.copy()
        )

        # Restore old strict Task6 gate.
        self.max_nominal_drift_px = 8.0

        # Similar to old TIP_EMA_ALPHA = 0.4.
        self.ema_alpha = 0.40

        print()
        print(
            "[TWEEZER V67] FIXED calibrated tip =",
            tuple(
                np.rint(
                    self.nominal_tip
                ).astype(int)
            ),
            flush=True
        )

        print(
            "[TWEEZER V67] candidate gate = +/- 8 px",
            flush=True
        )

        print(
            "[TWEEZER V67] SAM2 controls MASK, "
            "not global tip position",
            flush=True
        )


    def update(
        self,
        frame
    ):

        result = super().update(
            frame
        )

        # ----------------------------------------------------
        # Never allow SAM2/PCA to move the reported tip far
        # away from the rigid calibrated location.
        # ----------------------------------------------------

        tip_xy = tuple(
            np.rint(
                self.filtered_tip
            ).astype(int)
        )

        result[
            "tip"
        ] = tip_xy

        result[
            "fixed_tip_uv"
        ] = (
            np.rint(
                self.nominal_tip
            ).astype(int).tolist()
        )

        result[
            "tip_offset_from_fixed_px"
        ] = float(
            np.linalg.norm(
                self.filtered_tip
                -
                self.nominal_tip
            )
        )

        return result
