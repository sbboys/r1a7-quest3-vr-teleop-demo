#!/usr/bin/env python3

import cv2
import numpy as np

from wire_guided_tracker_v3 import (
    GuidedWireTrackerV3,
    grasp_target_from_centerline,
)


class EgoMotionWireTrackerV4(GuidedWireTrackerV3):
    """
    Wrist-camera thin-wire tracker.

    Key difference from V3:

    The camera is rigidly mounted on the robot wrist.
    Therefore robot-arm motion causes global image motion.

    V4 estimates that global motion from background features
    and moves the previous wire model with the scene BEFORE
    applying local ridge refinement.

    Invalid frames may update PREDICTION geometry but NEVER
    become valid grasp observations.
    """

    def __init__(
        self,
        sample_step_px=4.0,
        roi_margin_px=45,
        normal_search_px=8,
    ):

        super().__init__(
            sample_step_px=sample_step_px,
            roi_margin_px=roi_margin_px,
            normal_search_px=normal_search_px,
        )

        # Maximum number of consecutive frames for which
        # ego-motion prediction may continue without a
        # confirmed wire observation.
        self.max_predict_only_frames = 6

        self.predict_only_frames = 0

        self.ego_inliers = 0
        self.ego_inlier_ratio = 0.0
        self.ego_dx = 0.0
        self.ego_dy = 0.0

    # ========================================================
    # Initialization
    # ========================================================

    def initialize(
        self,
        frame,
        control_points
    ):

        result = super().initialize(
            frame,
            control_points
        )

        self.predict_only_frames = 0

        return result

    # ========================================================
    # Background feature mask
    #
    # Exclude:
    #   1. the current wire corridor
    #   2. lower-right tool/tweezer area
    #
    # We want the TABLE / world background motion,
    # not the camera-fixed tweezer.
    # ========================================================

    def _background_mask(
        self,
        gray
    ):

        h, w = gray.shape[:2]

        mask = np.full(
            (h, w),
            255,
            dtype=np.uint8
        )

        # Avoid extreme upper edge.
        cv2.rectangle(
            mask,
            (0, 0),
            (w, int(0.10 * h)),
            0,
            -1
        )

        # Right wrist tweezer is normally in the lower-right.
        # Camera and tweezer are rigidly attached, so tweezer
        # pixels should NOT define world ego motion.
        cv2.rectangle(
            mask,
            (
                int(0.62 * w),
                int(0.42 * h)
            ),
            (
                w,
                h
            ),
            0,
            -1
        )

        # Exclude current wire itself.
        if (
            self.points is not None
            and
            len(self.points) >= 2
        ):

            pts = np.rint(
                self.points
            ).astype(
                np.int32
            )

            cv2.polylines(
                mask,
                [
                    pts.reshape(
                        -1,
                        1,
                        2
                    )
                ],
                False,
                0,
                55,
                cv2.LINE_AA
            )

        return mask

    # ========================================================
    # Estimate camera-induced image motion
    # ========================================================

    def _estimate_ego_motion(
        self,
        prev_gray,
        gray
    ):

        mask = self._background_mask(
            prev_gray
        )

        p0 = cv2.goodFeaturesToTrack(
            prev_gray,
            mask=mask,
            maxCorners=250,
            qualityLevel=0.01,
            minDistance=12,
            blockSize=7
        )

        if (
            p0 is None
            or
            len(p0) < 20
        ):

            return None

        p1, st1, _ = (
            cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                p0,
                None,
                winSize=(25, 25),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    |
                    cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01
                )
            )
        )

        if (
            p1 is None
            or
            st1 is None
        ):

            return None

        p0_back, st2, _ = (
            cv2.calcOpticalFlowPyrLK(
                gray,
                prev_gray,
                p1,
                None,
                winSize=(25, 25),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    |
                    cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01
                )
            )
        )

        if (
            p0_back is None
            or
            st2 is None
        ):

            return None

        a = p0.reshape(
            -1,
            2
        )

        b = p1.reshape(
            -1,
            2
        )

        back = p0_back.reshape(
            -1,
            2
        )

        fb = np.linalg.norm(
            back - a,
            axis=1
        )

        good = (
            st1.reshape(-1).astype(bool)
            &
            st2.reshape(-1).astype(bool)
            &
            (fb < 1.5)
        )

        a = a[good]
        b = b[good]

        if len(a) < 15:
            return None

        # Full affine rather than translation only:
        # wrist movement can create scale + rotation locally.
        M, inliers = cv2.estimateAffine2D(
            a,
            b,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=2000,
            confidence=0.99,
            refineIters=10
        )

        if (
            M is None
            or
            inliers is None
        ):

            return None

        inliers = (
            inliers.reshape(-1)
            .astype(bool)
        )

        n_inliers = int(
            np.count_nonzero(
                inliers
            )
        )

        ratio = float(
            n_inliers
            /
            max(
                1,
                len(inliers)
            )
        )

        if (
            n_inliers < 12
            or
            ratio < 0.45
        ):

            return None

        # Translation component.
        dx = float(
            M[0, 2]
        )

        dy = float(
            M[1, 2]
        )

        translation = float(
            np.hypot(
                dx,
                dy
            )
        )

        # Slow robot motion should not create enormous
        # frame-to-frame displacement.
        if translation > 80.0:
            return None

        # Affine scale sanity check.
        sx = float(
            np.linalg.norm(
                M[:, 0]
            )
        )

        sy = float(
            np.linalg.norm(
                M[:, 1]
            )
        )

        if not (
            0.80 <= sx <= 1.20
            and
            0.80 <= sy <= 1.20
        ):

            return None

        self.ego_inliers = n_inliers
        self.ego_inlier_ratio = ratio
        self.ego_dx = dx
        self.ego_dy = dy

        return M

    # ========================================================
    # Transform the previous wire using camera ego motion
    # ========================================================

    @staticmethod
    def _transform_points(
        points,
        M
    ):

        p = np.asarray(
            points,
            dtype=np.float32
        )

        ones = np.ones(
            (
                len(p),
                1
            ),
            dtype=np.float32
        )

        ph = np.concatenate(
            [
                p,
                ones
            ],
            axis=1
        )

        out = (
            M
            @
            ph.T
        ).T

        return out.astype(
            np.float32
        )

    # ========================================================
    # Update
    # ========================================================

    def update(
        self,
        frame
    ):

        if (
            self.points is None
            or
            self.prev_gray is None
        ):

            return self._make_v4_result(
                False,
                "NOT_INITIALIZED",
                0.0,
                0.0
            )

        if self.lost:

            return self._make_v4_result(
                False,
                "LOST_REINIT_REQUIRED",
                0.0,
                0.0
            )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # 1. Estimate WORLD/BACKGROUND motion caused by wrist.
        # ----------------------------------------------------

        M = self._estimate_ego_motion(
            self.prev_gray,
            gray
        )

        if M is None:

            self.missed += 1

            if self.missed >= 3:
                self.lost = True

            return self._make_v4_result(
                False,
                "EGO_LOST",
                0.0,
                0.0
            )

        # ----------------------------------------------------
        # 2. Move previous wire WITH the background.
        # ----------------------------------------------------

        predicted = self._transform_points(
            self.points,
            M
        )

        # ----------------------------------------------------
        # 3. Local ridge correction around predicted wire.
        # ----------------------------------------------------

        ridge_pack = self._make_ridge_roi(
            gray,
            predicted
        )

        refined, ridge_score = (
            self._refine_normal(
                predicted,
                ridge_pack
            )
        )

        refined = self._smooth(
            refined
        )

        new_length = self._polyline_length(
            refined
        )

        old_length = self._polyline_length(
            predicted
        )

        length_ratio = (
            new_length
            /
            max(
                1e-6,
                old_length
            )
        )

        # Local visual correction should remain small because
        # global camera displacement was already compensated.
        correction = np.linalg.norm(
            refined - predicted,
            axis=1
        )

        mean_correction = float(
            np.mean(
                correction
            )
        )

        max_correction = float(
            np.max(
                correction
            )
        )

        pred_dir = self._direction(
            predicted
        )

        refined_dir = self._direction(
            refined
        )

        angle_deg = 0.0

        if (
            pred_dir is not None
            and
            refined_dir is not None
        ):

            dot = float(
                np.clip(
                    np.dot(
                        pred_dir,
                        refined_dir
                    ),
                    -1.0,
                    1.0
                )
            )

            angle_deg = float(
                np.degrees(
                    np.arccos(
                        dot
                    )
                )
            )

        # ----------------------------------------------------
        # 4. Strong visual acceptance gates.
        # ----------------------------------------------------

        geometry_ok = (
            0.82
            <=
            length_ratio
            <=
            1.18
            and
            mean_correction
            <=
            7.0
            and
            max_correction
            <=
            16.0
            and
            angle_deg
            <=
            25.0
        )

        ridge_ok = (
            ridge_score
            >=
            0.075
            and
            self._v3_support_fraction
            >=
            0.55
        )

        tip_ok = (
            self._v3_tip_score
            >=
            0.055
            and
            self._v3_tip_shift
            <=
            self.tip_max_correction_px
        )

        valid = (
            geometry_ok
            and
            ridge_ok
            and
            tip_ok
        )

        ego_conf = np.clip(
            self.ego_inlier_ratio,
            0.0,
            1.0
        )

        ridge_conf = np.clip(
            ridge_score / 0.25,
            0.0,
            1.0
        )

        support_conf = np.clip(
            self._v3_support_fraction,
            0.0,
            1.0
        )

        confidence = float(
            0.35 * ego_conf
            +
            0.35 * ridge_conf
            +
            0.30 * support_conf
        )

        # ----------------------------------------------------
        # 5a. CONFIRMED visual wire.
        # ----------------------------------------------------

        if valid:

            self.points = refined.astype(
                np.float32
            )

            self.prev_gray = gray

            self.prev_length = (
                new_length
            )

            self.missed = 0

            self.predict_only_frames = 0

            return self._make_v4_result(
                True,
                "TRACK",
                confidence,
                ridge_score
            )

        # ----------------------------------------------------
        # 5b. Not visually confirmed.
        #
        # IMPORTANT:
        # Follow camera ego motion, but remain INVALID.
        #
        # This avoids the V3 problem where the gray path stays
        # fixed in pixel space while the wrist camera moves.
        # ----------------------------------------------------

        self.points = predicted.astype(
            np.float32
        )

        self.prev_gray = gray

        self.prev_length = (
            self._polyline_length(
                self.points
            )
        )

        self.predict_only_frames += 1

        self.missed += 1

        if (
            self.predict_only_frames
            >
            self.max_predict_only_frames
        ):

            self.lost = True

            state = (
                "LOST_REINIT_REQUIRED"
            )

        else:

            state = (
                "EGO_PREDICT"
            )

        return self._make_v4_result(
            False,
            state,
            confidence,
            ridge_score
        )

    # ========================================================

    def _make_v4_result(
        self,
        valid,
        state,
        confidence,
        ridge_score,
    ):

        result = self._make_result(
            valid,
            state,
            confidence,
            0.0,
            ridge_score,
            float(
                np.hypot(
                    self.ego_dx,
                    self.ego_dy
                )
            )
        )

        result[
            "ego_inliers"
        ] = int(
            self.ego_inliers
        )

        result[
            "ego_inlier_ratio"
        ] = float(
            self.ego_inlier_ratio
        )

        result[
            "ego_dx_px"
        ] = float(
            self.ego_dx
        )

        result[
            "ego_dy_px"
        ] = float(
            self.ego_dy
        )

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

        result[
            "predict_only_frames"
        ] = int(
            self.predict_only_frames
        )

        return result
