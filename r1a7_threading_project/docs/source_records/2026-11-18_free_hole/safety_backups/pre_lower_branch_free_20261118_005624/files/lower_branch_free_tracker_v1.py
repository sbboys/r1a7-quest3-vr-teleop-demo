#!/usr/bin/env python3

# LOWER_BRANCH_FREE_TRACKER_V1
#
# Standalone experimental module.
#
# IMPORTANT:
#   - no robot interface
#   - no Gate output
#   - no Fusion output
#   - no JSON control output
#   - no modification of production programs
#
# Purpose:
#   After grasp, FREE must remain on the same physical branch
#   extending from the gripper toward the initially confirmed
#   lower/free side of the wire.

from __future__ import annotations

import math


class LowerBranchFreeTracker:
    def __init__(
        self,
        *,
        branch_min_projection_px=4.0,
        prediction_gate_px=25.0,
        endpoint_match_px=12.0,
        alpha=0.35,
        lost_after=5,
    ):
        self.branch_min_projection_px = float(
            branch_min_projection_px
        )

        self.prediction_gate_px = float(
            prediction_gate_px
        )

        self.endpoint_match_px = float(
            endpoint_match_px
        )

        self.alpha = float(alpha)
        self.lost_after = int(lost_after)

        self.initialized = False

        self.grasp_uv = None
        self.free_uv = None

        # Unit vector:
        # grasp -> physical FREE branch.
        self.branch_dir = None

        # FREE velocity in image pixels/frame.
        self.velocity_uv = (0.0, 0.0)

        self.bad_count = 0
        self.frame_count = 0


    @staticmethod
    def _dist(a, b):
        return math.hypot(
            float(a[0]) - float(b[0]),
            float(a[1]) - float(b[1]),
        )


    @staticmethod
    def _normalize(v):
        n = math.hypot(v[0], v[1])

        if n < 1e-6:
            raise ValueError(
                "grasp_uv and free_uv are too close"
            )

        return (
            float(v[0]) / n,
            float(v[1]) / n,
        )


    def initialize(
        self,
        grasp_uv,
        free_uv,
    ):
        """
        Initialize with a known-correct grasp point and FREE point.

        This determines which physical half/branch of the wire is valid.
        """

        grasp_uv = (
            float(grasp_uv[0]),
            float(grasp_uv[1]),
        )

        free_uv = (
            float(free_uv[0]),
            float(free_uv[1]),
        )

        self.branch_dir = self._normalize(
            (
                free_uv[0] - grasp_uv[0],
                free_uv[1] - grasp_uv[1],
            )
        )

        self.grasp_uv = grasp_uv
        self.free_uv = free_uv

        self.velocity_uv = (0.0, 0.0)

        self.bad_count = 0
        self.frame_count = 0
        self.initialized = True


    def branch_projection(
        self,
        grasp_uv,
        candidate_uv,
    ):
        """
        Positive:
            candidate lies on the initialized FREE half.

        Negative:
            candidate lies on the opposite/gripper upper half.
        """

        dx = (
            float(candidate_uv[0])
            -
            float(grasp_uv[0])
        )

        dy = (
            float(candidate_uv[1])
            -
            float(grasp_uv[1])
        )

        return (
            dx * self.branch_dir[0]
            +
            dy * self.branch_dir[1]
        )


    def predict(
        self,
        grasp_uv,
        robot_image_delta_uv=None,
    ):
        """
        Predict next FREE position.

        Main prediction:
            previous FREE
            + grasp-point image motion

        Optionally:
            robot_image_delta_uv can be supplied later from
            calibrated actual XYZ -> image motion.
        """

        if not self.initialized:
            return None

        grasp_uv = (
            float(grasp_uv[0]),
            float(grasp_uv[1]),
        )

        grasp_du = (
            grasp_uv[0]
            -
            self.grasp_uv[0]
        )

        grasp_dv = (
            grasp_uv[1]
            -
            self.grasp_uv[1]
        )

        if robot_image_delta_uv is not None:
            motion_du = float(
                robot_image_delta_uv[0]
            )

            motion_dv = float(
                robot_image_delta_uv[1]
            )

        else:
            motion_du = grasp_du
            motion_dv = grasp_dv

        return (
            self.free_uv[0]
            + motion_du
            + self.velocity_uv[0],

            self.free_uv[1]
            + motion_dv
            + self.velocity_uv[1],
        )


    def choose_candidate(
        self,
        *,
        grasp_uv,
        endpoint_candidates,
        robot_image_delta_uv=None,
    ):
        """
        endpoint_candidates:
            iterable of (u, v)

        Selection policy:

        1. Reject opposite-half candidates.
        2. Compare remaining candidates with motion prediction.
        3. Accept only candidates within prediction gate.
        4. Prefer smallest prediction residual.

        No candidate:
            PREDICT_ONLY / LOST

        Prediction is never returned as a valid control FREE.
        """

        if not self.initialized:
            raise RuntimeError(
                "tracker must be initialized first"
            )

        grasp_uv = (
            float(grasp_uv[0]),
            float(grasp_uv[1]),
        )

        pred = self.predict(
            grasp_uv,
            robot_image_delta_uv,
        )

        valid = []

        for candidate in endpoint_candidates:
            candidate = (
                float(candidate[0]),
                float(candidate[1]),
            )

            projection = self.branch_projection(
                grasp_uv,
                candidate,
            )

            # Hard physical branch rejection.
            if (
                projection
                <
                self.branch_min_projection_px
            ):
                continue

            residual = self._dist(
                candidate,
                pred,
            )

            if (
                residual
                >
                self.prediction_gate_px
            ):
                continue

            valid.append(
                (
                    residual,
                    projection,
                    candidate,
                )
            )

        if not valid:
            self.bad_count += 1

            if self.bad_count >= self.lost_after:
                state = "LOST"
            else:
                state = "PREDICT_ONLY"

            return {
                "state": state,
                "identity_valid": False,
                "free_uv_control": None,
                "predicted_uv": pred,
                "motion_residual_px": None,
                "branch_projection_px": None,
                "candidate_count": 0,
                "bad_count": self.bad_count,
            }

        # Closest candidate to motion prediction.
        valid.sort(
            key=lambda item: item[0]
        )

        residual, projection, chosen = valid[0]

        old_free = self.free_uv

        # Filter only after identity has been validated.
        a = self.alpha

        filtered = (
            (1.0 - a) * pred[0]
            + a * chosen[0],

            (1.0 - a) * pred[1]
            + a * chosen[1],
        )

        observed_velocity = (
            filtered[0] - old_free[0],
            filtered[1] - old_free[1],
        )

        # Slow velocity update.
        self.velocity_uv = (
            0.7 * self.velocity_uv[0]
            + 0.3 * observed_velocity[0],

            0.7 * self.velocity_uv[1]
            + 0.3 * observed_velocity[1],
        )

        self.grasp_uv = grasp_uv
        self.free_uv = filtered

        self.bad_count = 0
        self.frame_count += 1

        return {
            "state": "LOCKED",
            "identity_valid": True,
            "free_uv_control": filtered,
            "raw_candidate_uv": chosen,
            "predicted_uv": pred,
            "motion_residual_px": residual,
            "branch_projection_px": projection,
            "candidate_count": len(valid),
            "bad_count": 0,
        }


def skeleton_endpoint_candidates(
    skeleton_points,
    neighbor_radius_px=1.5,
):
    """
    Simple topology utility for offline testing.

    Given skeleton pixel coordinates [(u,v), ...],
    return degree<=1 endpoint candidates.

    Production integration can replace this with the existing
    V3 skeleton graph implementation if available.
    """

    pts = [
        (float(p[0]), float(p[1]))
        for p in skeleton_points
    ]

    endpoints = []

    for i, p in enumerate(pts):
        degree = 0

        for j, q in enumerate(pts):
            if i == j:
                continue

            if (
                math.hypot(
                    p[0] - q[0],
                    p[1] - q[1],
                )
                <= neighbor_radius_px
            ):
                degree += 1

        if degree <= 1:
            endpoints.append(p)

    return endpoints
