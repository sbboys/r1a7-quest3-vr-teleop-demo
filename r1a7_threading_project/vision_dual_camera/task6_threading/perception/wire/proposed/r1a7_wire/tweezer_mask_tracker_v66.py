#!/usr/bin/env python3

import sys
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent

TASK6_DIR = None

for p in [HERE] + list(HERE.parents):

    candidate = (
        p /
        "task6_wrist_ibvs_controller_v2_tip.py"
    )

    if candidate.exists():

        TASK6_DIR = p
        break


if TASK6_DIR is None:

    raise RuntimeError(
        "Cannot locate task6_wrist_ibvs_controller_v2_tip.py"
    )


sys.path.insert(
    0,
    str(TASK6_DIR)
)


import task6_wrist_ibvs_controller_v2_tip as wrist


class TweezerMaskTrackerV66:
    """
    Tweezer and wrist camera are rigidly mounted.

    Therefore the image location of the physical tweezer tip
    should only change slightly.

    V6.6:
      - fixed SAM2 search box
      - fixed nominal tweezer tip
      - hard tip jump rejection
      - EMA small correction
      - mask confined to original tool ROI
      - short mask hold when SAM2 temporarily fails
    """

    def __init__(
        self,
        init_frame,
    ):

        print(
            "[TWEEZER V66] Loading SAM2...",
            flush=True
        )

        self.detector = (
            wrist.Sam2PcaTweezerDetector(
                checkpoint_path=
                    wrist.SAM2_CHECKPOINT
            )
        )

        self.refiner = (
            wrist.TweezerTipRefinerV2()
        )

        print(
            "[TWEEZER V66] SAM2 ready.",
            flush=True
        )

        init = wrist.initialize_live_tip(
            init_frame
        )

        if init is None:

            raise RuntimeError(
                "Tweezer initialization cancelled"
            )

        (
            self.tip_box,
            self.positive_points,
            self.tip_hint
        ) = init

        self.nominal_tip = np.asarray(
            self.tip_hint,
            dtype=np.float64
        )

        self.filtered_tip = (
            self.nominal_tip.copy()
        )

        # Because camera and tweezer are rigidly attached,
        # this can be much stricter than wire motion gates.
        self.max_nominal_drift_px = 32.0

        self.ema_alpha = 0.25

        self.last_raw_mask = None
        self.last_exclusion_mask = None

        self.mask_hold_count = 0
        self.max_mask_hold_frames = 5

        self.fixed_roi_margin_px = 25

        print(
            "[TWEEZER V66] nominal tip =",
            tuple(
                np.rint(
                    self.nominal_tip
                )
                .astype(int)
            ),
            flush=True
        )


    def _fixed_tool_roi_mask(
        self,
        shape
    ):

        h, w = shape[:2]

        box = np.asarray(
            self.tip_box,
            dtype=np.float64
        ).reshape(-1)

        if len(box) < 4:

            return np.full(
                (h, w),
                255,
                dtype=np.uint8
            )

        x1, y1, x2, y2 = (
            box[:4]
        )

        m = float(
            self.fixed_roi_margin_px
        )

        x1 = max(
            0,
            int(
                np.floor(
                    x1 - m
                )
            )
        )

        y1 = max(
            0,
            int(
                np.floor(
                    y1 - m
                )
            )
        )

        x2 = min(
            w - 1,
            int(
                np.ceil(
                    x2 + m
                )
            )
        )

        y2 = min(
            h - 1,
            int(
                np.ceil(
                    y2 + m
                )
            )
        )

        mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        cv2.rectangle(
            mask,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            255,
            -1
        )

        return mask


    @staticmethod
    def _mask_iou(
        a,
        b
    ):

        if (
            a is None
            or
            b is None
        ):

            return 1.0

        aa = a > 0
        bb = b > 0

        union = int(
            np.count_nonzero(
                aa | bb
            )
        )

        if union == 0:
            return 0.0

        inter = int(
            np.count_nonzero(
                aa & bb
            )
        )

        return float(
            inter / union
        )


    def _hold_result(
        self,
        reason
    ):

        self.mask_hold_count += 1

        if (
            self.last_exclusion_mask
            is None
            or
            self.mask_hold_count
            >
            self.max_mask_hold_frames
        ):

            return {
                "valid": False,
                "state":
                    "TWEEZER_LOST:"
                    +
                    reason,
                "tip": None,
                "raw_mask": None,
                "exclusion_mask": None,
                "tip_error_from_nominal_px":
                    None,
            }

        tip = tuple(
            np.rint(
                self.filtered_tip
            )
            .astype(int)
        )

        return {
            "valid": True,
            "state":
                "TWEEZER_HOLD:"
                +
                reason,
            "tip": tip,
            "raw_mask":
                self.last_raw_mask,
            "exclusion_mask":
                self.last_exclusion_mask,
            "tip_error_from_nominal_px":
                float(
                    np.linalg.norm(
                        self.filtered_tip
                        -
                        self.nominal_tip
                    )
                ),
        }


    def update(
        self,
        frame
    ):

        try:

            coarse = self.detector.detect(
                frame,
                box_xyxy=
                    self.tip_box,

                # Critical change:
                # do not chase the previous erroneous result.
                # Always use rigid nominal tip as SAM2 hint.
                tip_hint=(
                    float(
                        self.nominal_tip[0]
                    ),
                    float(
                        self.nominal_tip[1]
                    )
                ),

                positive_points=
                    self.positive_points,
            )

        except Exception as exc:

            return self._hold_result(
                "SAM2_ERROR_"
                +
                type(exc).__name__
            )


        if not coarse.get(
            "valid",
            False
        ):

            return self._hold_result(
                "SAM2_INVALID"
            )


        raw = np.asarray(
            coarse["mask"]
        )

        raw = np.squeeze(
            raw
        )

        if (
            raw.shape[:2]
            !=
            frame.shape[:2]
        ):

            raw = cv2.resize(
                raw.astype(
                    np.uint8
                ),
                (
                    frame.shape[1],
                    frame.shape[0]
                ),
                interpolation=
                    cv2.INTER_NEAREST
            )


        raw_mask = (
            raw > 0
        ).astype(
            np.uint8
        ) * 255


        # ----------------------------------------------------
        # Tool is rigidly attached:
        # SAM2 segmentation is not allowed to wander outside
        # the original tweezer ROI.
        # ----------------------------------------------------

        fixed_roi = (
            self._fixed_tool_roi_mask(
                frame.shape
            )
        )

        raw_mask = cv2.bitwise_and(
            raw_mask,
            fixed_roi
        )


        area = int(
            np.count_nonzero(
                raw_mask
            )
        )

        if area < 80:

            return self._hold_result(
                "MASK_TOO_SMALL"
            )


        iou = self._mask_iou(
            raw_mask,
            self.last_raw_mask
        )

        if (
            self.last_raw_mask
            is not None
            and
            iou < 0.12
        ):

            return self._hold_result(
                f"MASK_IOU_{iou:.2f}"
            )


        candidate = np.asarray(
            coarse["tip_uv"],
            dtype=np.float64
        )


        # Try the existing local edge refiner.
        try:

            refined = self.refiner.refine(
                frame,
                raw_mask > 0,
                coarse["tip_uv"],
                coarse["direction"],
            )

            if refined.get(
                "valid",
                False
            ):

                candidate = np.asarray(
                    refined[
                        "refined_tip_uv"
                    ],
                    dtype=np.float64
                )

        except Exception:

            pass


        nominal_error = float(
            np.linalg.norm(
                candidate
                -
                self.nominal_tip
            )
        )


        # ----------------------------------------------------
        # Rigid camera-tool geometry gate.
        # ----------------------------------------------------

        if (
            nominal_error
            >
            self.max_nominal_drift_px
        ):

            tip_state = (
                "TIP_REJECT_HOLD"
            )

            # Do NOT update filtered tip with bad candidate.

        else:

            self.filtered_tip = (
                self.ema_alpha
                *
                candidate
                +
                (
                    1.0
                    -
                    self.ema_alpha
                )
                *
                self.filtered_tip
            )

            tip_state = (
                "TWEEZER_TRACK"
            )


        tip_xy = tuple(
            np.rint(
                self.filtered_tip
            )
            .astype(int)
        )


        # ----------------------------------------------------
        # Build wire exclusion mask.
        # ----------------------------------------------------

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


        # Allow the real wire to enter the jaw-tip area.
        cv2.circle(
            exclusion,
            tip_xy,
            16,
            0,
            -1
        )


        self.last_raw_mask = (
            raw_mask.copy()
        )

        self.last_exclusion_mask = (
            exclusion.copy()
        )

        self.mask_hold_count = 0


        return {
            "valid": True,

            "state":
                tip_state,

            "tip":
                tip_xy,

            "raw_mask":
                raw_mask,

            "exclusion_mask":
                exclusion,

            "tip_error_from_nominal_px":
                float(
                    np.linalg.norm(
                        self.filtered_tip
                        -
                        self.nominal_tip
                    )
                ),

            "raw_candidate_error_px":
                nominal_error,

            "mask_iou":
                float(iou),
        }
