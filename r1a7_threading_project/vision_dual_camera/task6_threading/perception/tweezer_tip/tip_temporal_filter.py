#!/usr/bin/env python3

import numpy as np


class TipTemporalFilter:

    def __init__(
        self,
        alpha=0.4,
        max_jump_px=8.0,
    ):
        self.alpha = float(alpha)
        self.max_jump_px = float(max_jump_px)

        self.tip = None
        self.gap = None

    def reset(self):
        self.tip = None
        self.gap = None

    def update(
        self,
        tip_uv,
        jaw_gap_px=None,
    ):
        tip = np.asarray(
            tip_uv,
            dtype=np.float64,
        )

        if self.tip is None:

            self.tip = tip.copy()

            if jaw_gap_px is not None:
                self.gap = float(
                    jaw_gap_px
                )

            return {
                "valid": True,
                "filtered_tip_uv":
                self.tip.tolist(),
                "filtered_gap_px":
                self.gap,
                "rejected": False,
            }

        jump = float(
            np.linalg.norm(
                tip - self.tip
            )
        )

        # 明显异常跳点先拒绝。
        if jump > self.max_jump_px:

            return {
                "valid": True,
                "filtered_tip_uv":
                self.tip.tolist(),
                "filtered_gap_px":
                self.gap,
                "rejected": True,
                "raw_jump_px":
                jump,
            }

        a = self.alpha

        self.tip = (
            a * tip
            +
            (1.0 - a) * self.tip
        )

        if jaw_gap_px is not None:

            gap = float(
                jaw_gap_px
            )

            if self.gap is None:
                self.gap = gap
            else:
                self.gap = (
                    a * gap
                    +
                    (1.0 - a)
                    * self.gap
                )

        return {
            "valid": True,

            "filtered_tip_uv": [
                float(self.tip[0]),
                float(self.tip[1]),
            ],

            "filtered_gap_px":
            (
                float(self.gap)
                if self.gap is not None
                else None
            ),

            "rejected": False,

            "raw_jump_px":
            jump,
        }
