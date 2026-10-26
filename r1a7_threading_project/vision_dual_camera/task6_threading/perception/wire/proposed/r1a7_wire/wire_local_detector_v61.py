#!/usr/bin/env python3

import numpy as np

from wire_local_detector_v5 import WireLocalDetectorV5
from wire_local_detector_v6 import WireLocalDetectorV6


class WireLocalDetectorV61(WireLocalDetectorV6):
    """
    V6.1

    Fixes:
      1. V6 ranking score no longer replaces V5 quality score.
      2. Candidate must stay near the previous confirmed free tip.
      3. Cardboard long-edge rejection is kept, but profile threshold
         is no longer so aggressive that the real wire disappears.
      4. Invalid candidate can be returned as debug geometry, but is
         NEVER a valid wire result.
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

        self.debug_path_local = None
        self.last_rank_score = 0.0


    def _candidate_components(
        self,
        gray,
        response,
    ):

        candidates = (
            WireLocalDetectorV6
            ._candidate_components(
                self,
                gray,
                response
            )
        )

        self.debug_path_local = None
        self.last_rank_score = 0.0

        if not candidates:
            return []

        # --------------------------------------------------
        # Important fix:
        #
        # V6 overwrote candidate["score"] with its new
        # ranking score.
        #
        # But V5 detect() later uses candidate["score"]
        # again for its quality gate.
        #
        # Restore V5 score for the V5 quality gate and keep
        # V6 score separately as rank_score.
        # --------------------------------------------------

        for c in candidates:

            c["rank_score"] = float(
                c.get(
                    "score",
                    0.0
                )
            )

            c["score"] = float(
                c.get(
                    "v5_score",
                    c["rank_score"]
                )
            )

        # --------------------------------------------------
        # Spatial association:
        #
        # Do not allow a far paper edge to win and then get
        # rejected AFTER selection. Select only candidates
        # reasonably close to the currently known wire end.
        # --------------------------------------------------

        reference = (
            self.last_valid_tip
            if self.last_valid_tip is not None
            else self.seed_tip
        )

        if reference is not None:

            radius = (
                70.0
                if self.last_valid_tip is not None
                else 95.0
            )

            nearby = []

            for c in candidates:

                d = c.get(
                    "tip_distance"
                )

                if (
                    d is None
                    or float(d) <= radius
                ):
                    nearby.append(c)

            candidates = nearby

        if not candidates:

            return []

        candidates.sort(
            key=lambda c:
            c.get(
                "rank_score",
                0.0
            ),
            reverse=True
        )

        best = candidates[0]

        self.debug_path_local = (
            np.asarray(
                best["path"],
                dtype=np.float64
            ).copy()
        )

        self.last_rank_score = float(
            best.get(
                "rank_score",
                0.0
            )
        )

        self.last_profile_score = float(
            best.get(
                "profile_score",
                0.0
            )
        )

        self.last_two_sided_fraction = float(
            best.get(
                "two_sided_fraction",
                0.0
            )
        )

        self.last_edge_overlap = float(
            best.get(
                "straight_edge_overlap",
                0.0
            )
        )

        return candidates


    def detect(
        self,
        frame,
        roi
    ):

        # --------------------------------------------------
        # Deliberately bypass V6.detect()
        #
        # We want V5 current-frame detection combined with
        # V6.1 candidate ranking.
        # --------------------------------------------------

        result = (
            WireLocalDetectorV5
            .detect(
                self,
                frame,
                roi
            )
        )

        result["rank_score"] = float(
            self.last_rank_score
        )

        result["profile_score"] = float(
            self.last_profile_score
        )

        result["two_sided_fraction"] = float(
            self.last_two_sided_fraction
        )

        result["straight_edge_overlap"] = float(
            self.last_edge_overlap
        )

        # --------------------------------------------------
        # Debug candidate.
        #
        # Useful when valid=False:
        # it shows what the detector currently THINKS is the
        # best candidate without declaring it to be a wire.
        # --------------------------------------------------

        if self.debug_path_local is not None:

            x, y, w, h = [
                int(v)
                for v in roi
            ]

            debug = (
                self.debug_path_local.copy()
            )

            debug[:, 0] += x
            debug[:, 1] += y

            result[
                "debug_centerline"
            ] = (
                np.rint(debug)
                .astype(int)
                .tolist()
            )

        else:

            result[
                "debug_centerline"
            ] = []

        if not result["valid"]:
            return result

        # --------------------------------------------------
        # Paper/cardboard rejection.
        #
        # Do NOT use an aggressive profile threshold yet.
        # First collect real statistics from this camera.
        # --------------------------------------------------

        paper_like = (
            self.last_edge_overlap > 0.58
            and
            self.last_two_sided_fraction < 0.28
            and
            self.last_profile_score < 0.30
        )

        extremely_weak_wire_profile = (
            self.last_profile_score < 0.06
            and
            self.last_two_sided_fraction < 0.08
        )

        if paper_like:

            result["valid"] = False
            result["state"] = "PAPER_EDGE_REJECT"

            result["centerline"] = []
            result["wire_tip_uv"] = None
            result["grasp_target_uv"] = None
            result["wire_direction_uv"] = None

            return result

        if extremely_weak_wire_profile:

            result["valid"] = False
            result["state"] = "PROFILE_REJECT"

            result["centerline"] = []
            result["wire_tip_uv"] = None
            result["grasp_target_uv"] = None
            result["wire_direction_uv"] = None

            return result

        # --------------------------------------------------
        # Smooth only two ACTUAL valid image detections.
        # No prediction.
        # --------------------------------------------------

        path = self._resample_path(
            result["centerline"],
            count=40
        )

        if path is None:
            return result

        if (
            self.smoothed_path is not None
            and
            self.smoothed_path.shape
            ==
            path.shape
        ):

            distance = float(
                np.mean(
                    np.linalg.norm(
                        path
                        -
                        self.smoothed_path,
                        axis=1
                    )
                )
            )

            if distance < 30.0:

                path = (
                    0.65 * path
                    +
                    0.35 * self.smoothed_path
                )

        self.smoothed_path = (
            path.copy()
        )

        result["centerline"] = (
            np.rint(path)
            .astype(int)
            .tolist()
        )

        tip = path[0]

        result["wire_tip_uv"] = (
            np.rint(tip)
            .astype(int)
            .tolist()
        )

        target = (
            self._grasp_target(
                path
            )
        )

        result["grasp_target_uv"] = (
            np.rint(target)
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

        result["wire_direction_uv"] = (
            direction
            .astype(float)
            .tolist()
            if direction is not None
            else None
        )

        return result
