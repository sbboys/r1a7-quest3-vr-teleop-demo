#!/usr/bin/env python3

import cv2
import numpy as np

from wire_local_detector_v5 import (
    WireLocalDetectorV5,
)


class WireLocalDetectorV6(
    WireLocalDetectorV5
):
    """
    V6:
      - still image-driven detection every frame
      - transverse line-profile test
      - long straight-edge penalty
      - output smoothing ONLY after a valid current-frame detection

    Previous frame is NOT used to generate a new wire.
    """

    def __init__(
        self,
        local_length_px=130.0,
        grasp_offset_px=30.0,
    ):

        super().__init__(
            local_length_px=
                local_length_px,
            grasp_offset_px=
                grasp_offset_px,
        )

        self.smoothed_path = None

        self.last_profile_score = 0.0
        self.last_two_sided_fraction = 0.0
        self.last_edge_overlap = 0.0


    # ========================================================
    # Bilinear-ish nearest sampling
    # ========================================================

    @staticmethod
    def _sample_gray(
        gray,
        p
    ):

        x = int(
            round(
                float(p[0])
            )
        )

        y = int(
            round(
                float(p[1])
            )
        )

        h, w = gray.shape[:2]

        if (
            x < 0
            or y < 0
            or x >= w
            or y >= h
        ):
            return None

        return float(
            gray[y, x]
        )


    # ========================================================
    # Thin bright wire cross-section test
    #
    # For a real thin bright wire:
    #
    #       center
    #         /\
    #  ------/  \------
    #
    # both sides should usually be darker.
    #
    # A cardboard boundary is generally only a one-sided step.
    # ========================================================

    def _profile_score(
        self,
        gray,
        path,
        normal_offset=4.0,
    ):

        if (
            path is None
            or len(path) < 12
        ):

            return (
                0.0,
                0.0
            )

        path = np.asarray(
            path,
            dtype=np.float64
        )

        gains = []

        good_count = 0
        valid_count = 0

        step = max(
            2,
            len(path) // 30
        )

        for i in range(
            3,
            len(path) - 3,
            step
        ):

            tangent = (
                path[i + 3]
                -
                path[i - 3]
            )

            n = float(
                np.linalg.norm(
                    tangent
                )
            )

            if n < 1e-6:
                continue

            tangent /= n

            normal = np.array(
                [
                    -tangent[1],
                    tangent[0]
                ],
                dtype=np.float64
            )

            center = self._sample_gray(
                gray,
                path[i]
            )

            left = self._sample_gray(
                gray,
                path[i]
                +
                normal
                *
                normal_offset
            )

            right = self._sample_gray(
                gray,
                path[i]
                -
                normal
                *
                normal_offset
            )

            if (
                center is None
                or left is None
                or right is None
            ):
                continue

            # The weaker of the two sides is decisive.
            gain_left = (
                center - left
            )

            gain_right = (
                center - right
            )

            two_side_gain = min(
                gain_left,
                gain_right
            )

            gains.append(
                two_side_gain
            )

            valid_count += 1

            if (
                gain_left > 5.0
                and
                gain_right > 5.0
            ):

                good_count += 1

        if valid_count < 3:

            return (
                0.0,
                0.0
            )

        gains = np.asarray(
            gains,
            dtype=np.float64
        )

        two_sided_fraction = float(
            good_count
            /
            valid_count
        )

        positive_gain = float(
            np.mean(
                np.clip(
                    gains,
                    0.0,
                    30.0
                )
            )
        )

        contrast_score = float(
            np.clip(
                positive_gain
                /
                18.0,
                0.0,
                1.0
            )
        )

        score = (
            0.70
            *
            two_sided_fraction
            +
            0.30
            *
            contrast_score
        )

        return (
            float(score),
            float(
                two_sided_fraction
            )
        )


    # ========================================================
    # Detect long straight structural boundaries
    #
    # Cardboard edges usually form long Hough segments.
    # This is a SOFT penalty, not a hard rejection, because
    # the wire itself may sometimes be fairly straight.
    # ========================================================

    @staticmethod
    def _long_edge_mask(
        gray
    ):

        edges = cv2.Canny(
            gray,
            55,
            150
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=40,
            minLineLength=65,
            maxLineGap=10
        )

        mask = np.zeros_like(
            gray,
            dtype=np.uint8
        )

        if lines is None:

            return mask

        for line in lines:

            x1, y1, x2, y2 = (
                line[0]
            )

            length = float(
                np.hypot(
                    x2 - x1,
                    y2 - y1
                )
            )

            if length < 65.0:
                continue

            cv2.line(
                mask,
                (
                    int(x1),
                    int(y1)
                ),
                (
                    int(x2),
                    int(y2)
                ),
                255,
                7,
                cv2.LINE_AA
            )

        return mask


    @staticmethod
    def _path_mask_overlap(
        path,
        mask
    ):

        if (
            path is None
            or len(path) == 0
        ):

            return 0.0

        pts = np.rint(
            path
        ).astype(
            np.int32
        )

        xs = np.clip(
            pts[:, 0],
            0,
            mask.shape[1] - 1
        )

        ys = np.clip(
            pts[:, 1],
            0,
            mask.shape[0] - 1
        )

        return float(
            np.mean(
                mask[
                    ys,
                    xs
                ]
                >
                0
            )
        )


    # ========================================================
    # Re-score V5 components
    # ========================================================

    def _candidate_components(
        self,
        gray,
        response,
    ):

        candidates = (
            super()
            ._candidate_components(
                gray,
                response
            )
        )

        if not candidates:

            self.last_profile_score = 0.0
            self.last_two_sided_fraction = 0.0
            self.last_edge_overlap = 0.0

            return []

        long_edges = (
            self._long_edge_mask(
                gray
            )
        )

        for candidate in candidates:

            path = candidate[
                "path"
            ]

            (
                profile_score,
                two_sided_fraction
            ) = (
                self._profile_score(
                    gray,
                    path
                )
            )

            edge_overlap = (
                self._path_mask_overlap(
                    path,
                    long_edges
                )
            )

            original_score = float(
                candidate["score"]
            )

            # --------------------------------------------
            # Important V6 ranking:
            #
            # Frangi alone gets only 35%.
            #
            # The transverse physical appearance of a wire
            # now contributes 50%.
            #
            # Long structural boundaries lose score.
            # --------------------------------------------

            score = (
                0.35
                *
                original_score

                +

                0.50
                *
                profile_score

                +

                0.15
                *
                (
                    1.0
                    -
                    edge_overlap
                )
            )

            candidate[
                "v5_score"
            ] = original_score

            candidate[
                "profile_score"
            ] = float(
                profile_score
            )

            candidate[
                "two_sided_fraction"
            ] = float(
                two_sided_fraction
            )

            candidate[
                "straight_edge_overlap"
            ] = float(
                edge_overlap
            )

            candidate[
                "score"
            ] = float(
                score
            )

        candidates.sort(
            key=lambda c:
            c["score"],
            reverse=True
        )

        best = candidates[0]

        self.last_profile_score = float(
            best[
                "profile_score"
            ]
        )

        self.last_two_sided_fraction = float(
            best[
                "two_sided_fraction"
            ]
        )

        self.last_edge_overlap = float(
            best[
                "straight_edge_overlap"
            ]
        )

        return candidates


    # ========================================================
    # Resample path to fixed point count
    # ========================================================

    @staticmethod
    def _resample_path(
        path,
        count=40
    ):

        path = np.asarray(
            path,
            dtype=np.float64
        )

        if len(path) < 2:
            return None

        seg = np.linalg.norm(
            path[1:]
            -
            path[:-1],
            axis=1
        )

        s = np.concatenate(
            [
                [0.0],
                np.cumsum(seg)
            ]
        )

        if s[-1] < 1e-6:
            return None

        target_s = np.linspace(
            0.0,
            s[-1],
            count
        )

        x = np.interp(
            target_s,
            s,
            path[:, 0]
        )

        y = np.interp(
            target_s,
            s,
            path[:, 1]
        )

        return np.column_stack(
            [
                x,
                y
            ]
        )


    # ========================================================
    # V6 detect
    # ========================================================

    def detect(
        self,
        frame,
        roi
    ):

        result = super().detect(
            frame,
            roi
        )

        result[
            "profile_score"
        ] = float(
            self.last_profile_score
        )

        result[
            "two_sided_fraction"
        ] = float(
            self.last_two_sided_fraction
        )

        result[
            "straight_edge_overlap"
        ] = float(
            self.last_edge_overlap
        )

        if not result[
            "valid"
        ]:

            return result

        # --------------------------------------------
        # Strong V6 physical-wire gate
        # --------------------------------------------

        if (
            self.last_profile_score
            <
            0.28
        ):

            result[
                "valid"
            ] = False

            result[
                "state"
            ] = "PROFILE_REJECT"

            result[
                "centerline"
            ] = []

            result[
                "wire_tip_uv"
            ] = None

            result[
                "grasp_target_uv"
            ] = None

            result[
                "wire_direction_uv"
            ] = None

            return result

        path = self._resample_path(
            result[
                "centerline"
            ],
            count=40
        )

        if path is None:

            return result

        # --------------------------------------------
        # Smooth ONLY between two valid image detections.
        #
        # No prediction occurs here.
        # --------------------------------------------

        if (
            self.smoothed_path
            is not None
            and
            self.smoothed_path.shape
            ==
            path.shape
        ):

            mean_distance = float(
                np.mean(
                    np.linalg.norm(
                        path
                        -
                        self.smoothed_path,
                        axis=1
                    )
                )
            )

            if mean_distance < 35.0:

                path = (
                    0.55
                    *
                    path
                    +
                    0.45
                    *
                    self.smoothed_path
                )

        self.smoothed_path = (
            path.copy()
        )

        result[
            "centerline"
        ] = (
            np.rint(
                path
            )
            .astype(int)
            .tolist()
        )

        tip = path[0]

        result[
            "wire_tip_uv"
        ] = (
            np.rint(
                tip
            )
            .astype(int)
            .tolist()
        )

        target = (
            self._grasp_target(
                path
            )
        )

        result[
            "grasp_target_uv"
        ] = (
            np.rint(
                target
            )
            .astype(int)
            .tolist()
            if target is not None
            else None
        )

        direction = (
            self._local_direction(
                path
            )
        )

        result[
            "wire_direction_uv"
        ] = (
            direction.astype(
                float
            ).tolist()
            if direction is not None
            else None
        )

        return result
