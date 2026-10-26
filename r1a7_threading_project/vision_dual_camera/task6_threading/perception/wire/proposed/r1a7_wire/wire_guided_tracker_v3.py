#!/usr/bin/env python3

import numpy as np

from wire_guided_tracker_v2 import (
    GuidedWireTrackerV2,
    grasp_target_from_centerline,
)


class GuidedWireTrackerV3(GuidedWireTrackerV2):

    def __init__(
        self,
        sample_step_px=4.0,
        roi_margin_px=50,
        normal_search_px=10,
    ):

        super().__init__(
            sample_step_px=sample_step_px,
            roi_margin_px=roi_margin_px,
            normal_search_px=normal_search_px,
        )

        self.tip_search_normal_px = 5
        self.tip_max_correction_px = 16.0

        self.min_ridge_support_fraction = 0.55
        self.min_tip_ridge_score = 0.065

        self._v3_support_fraction = 0.0
        self._v3_tip_score = 0.0
        self._v3_tip_shift = 0.0


    # ========================================================
    # Ridge sampling
    # ========================================================

    @staticmethod
    def _ridge_value(
        ridge_pack,
        point
    ):

        if ridge_pack is None:
            return 0.0

        ridge, x0, y0 = ridge_pack

        x = int(round(float(point[0]))) - x0
        y = int(round(float(point[1]))) - y0

        h, w = ridge.shape[:2]

        if (
            x < 0
            or y < 0
            or x >= w
            or y >= h
        ):
            return 0.0

        return float(
            ridge[y, x]
        )


    # ========================================================
    # Dedicated free-tip refinement
    #
    # Interior of the wire gives us a stable local direction.
    # Starting from inside the wire, walk OUTWARD until ridge
    # support disappears. The last supported position is the
    # physical free endpoint.
    # ========================================================

    def _refine_free_tip(
        self,
        predicted,
        refined,
        ridge_pack,
    ):

        if (
            ridge_pack is None
            or len(refined) < 6
        ):
            return (
                refined[0].copy(),
                0.0,
                0.0,
            )

        anchor_index = min(
            6,
            len(refined) - 1
        )

        anchor = (
            refined[anchor_index]
            .astype(np.float64)
        )

        inward = (
            refined[anchor_index]
            -
            refined[1]
        ).astype(
            np.float64
        )

        n = float(
            np.linalg.norm(
                inward
            )
        )

        if n < 1e-6:
            return (
                refined[0].copy(),
                0.0,
                0.0,
            )

        inward /= n

        normal = np.array(
            [
                -inward[1],
                inward[0],
            ],
            dtype=np.float64,
        )

        # Expected distance from interior anchor back to tip.
        expected_distance = 0.0

        for i in range(
            anchor_index
        ):

            expected_distance += float(
                np.linalg.norm(
                    refined[i + 1]
                    -
                    refined[i]
                )
            )

        max_walk = int(
            min(
                55.0,
                expected_distance
                +
                self.tip_max_correction_px
            )
        )

        samples = []

        for s in range(
            0,
            max_walk + 1
        ):

            center = (
                anchor
                -
                inward
                *
                float(s)
            )

            best_score = -1.0
            best_point = center.copy()

            for normal_offset in range(
                -self.tip_search_normal_px,
                self.tip_search_normal_px + 1
            ):

                candidate = (
                    center
                    +
                    normal
                    *
                    float(normal_offset)
                )

                score = (
                    self._ridge_value(
                        ridge_pack,
                        candidate
                    )
                )

                if score > best_score:

                    best_score = score
                    best_point = (
                        candidate.copy()
                    )

            samples.append(
                (
                    float(s),
                    float(
                        max(
                            0.0,
                            best_score
                        )
                    ),
                    best_point,
                )
            )

        if not samples:

            return (
                refined[0].copy(),
                0.0,
                0.0,
            )

        inside_limit = max(
            5,
            int(
                round(
                    expected_distance
                    *
                    0.55
                )
            )
        )

        inside_scores = [
            score
            for s, score, p
            in samples
            if s <= inside_limit
        ]

        if inside_scores:

            reference = float(
                np.percentile(
                    inside_scores,
                    60
                )
            )

        else:

            reference = 0.10

        threshold = max(
            0.055,
            0.35 * reference
        )

        last_good = None

        consecutive_bad = 0

        minimum_walk_before_stop = max(
            5,
            int(
                expected_distance
                *
                0.45
            )
        )

        for s, score, point in samples:

            if score >= threshold:

                last_good = (
                    s,
                    score,
                    point
                )

                consecutive_bad = 0

            else:

                consecutive_bad += 1

                if (
                    s >= minimum_walk_before_stop
                    and
                    consecutive_bad >= 3
                ):
                    break

        if last_good is None:

            return (
                refined[0].copy(),
                0.0,
                0.0,
            )

        _, tip_score, tip_candidate = (
            last_good
        )

        tip_shift = float(
            np.linalg.norm(
                tip_candidate
                -
                predicted[0]
            )
        )

        # Do not allow endpoint finder to jump to another edge.
        if (
            tip_shift
            >
            self.tip_max_correction_px
        ):

            return (
                refined[0].copy(),
                float(tip_score),
                float(tip_shift),
            )

        return (
            tip_candidate.astype(
                np.float32
            ),
            float(tip_score),
            float(tip_shift),
        )


    # ========================================================
    # Extend V2 normal refinement with:
    #   1. ridge support measurement
    #   2. dedicated free endpoint refinement
    # ========================================================

    def _refine_normal(
        self,
        predicted,
        ridge_pack
    ):

        refined, base_score = (
            super()._refine_normal(
                predicted,
                ridge_pack
            )
        )

        # --------------------------------------------
        # Dedicated tip correction
        # --------------------------------------------

        (
            corrected_tip,
            tip_score,
            tip_shift,
        ) = self._refine_free_tip(
            predicted,
            refined,
            ridge_pack,
        )

        refined[0] = (
            corrected_tip
        )

        # --------------------------------------------
        # How much of the entire path is actually on
        # a ridge?
        # --------------------------------------------

        scores = []

        for p in refined:

            scores.append(
                self._ridge_value(
                    ridge_pack,
                    p
                )
            )

        scores = np.asarray(
            scores,
            dtype=np.float64
        )

        if len(scores):

            support_fraction = float(
                np.mean(
                    scores >= 0.055
                )
            )

            mean_score = float(
                np.mean(
                    scores
                )
            )

        else:

            support_fraction = 0.0
            mean_score = 0.0

        self._v3_support_fraction = (
            support_fraction
        )

        self._v3_tip_score = (
            tip_score
        )

        self._v3_tip_shift = (
            tip_shift
        )

        return (
            refined,
            max(
                base_score,
                mean_score
            )
        )


    # ========================================================
    # IMPORTANT V3 change:
    #
    # Invalid frame is NEVER allowed to become the next
    # tracking reference.
    #
    # V2 HOLD_FLOW could slowly drag the line away.
    # V3 restores the last CONFIRMED path when a frame fails.
    # ========================================================

    def update(
        self,
        frame
    ):

        if self.points is None:

            return super().update(
                frame
            )

        old_points = (
            self.points.copy()
        )

        old_gray = (
            self.prev_gray.copy()
            if self.prev_gray is not None
            else None
        )

        old_length = (
            self.prev_length
        )

        result = (
            super().update(
                frame
            )
        )

        new_missed = (
            self.missed
        )

        new_lost = (
            self.lost
        )

        # Additional V3 acceptance gate.
        if result.get(
            "valid",
            False
        ):

            v3_ok = (
                self._v3_support_fraction
                >=
                self.min_ridge_support_fraction
                and
                self._v3_tip_score
                >=
                self.min_tip_ridge_score
                and
                self._v3_tip_shift
                <=
                self.tip_max_correction_px
            )

            if v3_ok:

                result[
                    "ridge_support_fraction"
                ] = float(
                    self._v3_support_fraction
                )

                result[
                    "tip_ridge_score"
                ] = float(
                    self._v3_tip_score
                )

                result[
                    "tip_correction_px"
                ] = float(
                    self._v3_tip_shift
                )

                return result

            # Reject the entire frame.
            new_missed = max(
                1,
                new_missed + 1
            )

            new_lost = (
                new_missed >= 3
            )

            result_state = (
                "V3_RIDGE_GATE"
            )

        else:

            result_state = (
                result.get(
                    "state",
                    "INVALID"
                )
            )

        # --------------------------------------------
        # Restore last confirmed visual track
        # --------------------------------------------

        self.points = (
            old_points
        )

        self.prev_gray = (
            old_gray
        )

        self.prev_length = (
            old_length
        )

        self.missed = (
            new_missed
        )

        self.lost = (
            new_lost
        )

        frozen = self._make_result(
            False,
            result_state,
            float(
                result.get(
                    "confidence",
                    0.0
                )
            ),
            float(
                result.get(
                    "flow_fraction",
                    0.0
                )
            ),
            float(
                result.get(
                    "ridge_score",
                    0.0
                )
            ),
            float(
                result.get(
                    "motion_px",
                    0.0
                )
            ),
        )

        frozen[
            "ridge_support_fraction"
        ] = float(
            self._v3_support_fraction
        )

        frozen[
            "tip_ridge_score"
        ] = float(
            self._v3_tip_score
        )

        frozen[
            "tip_correction_px"
        ] = float(
            self._v3_tip_shift
        )

        return frozen
