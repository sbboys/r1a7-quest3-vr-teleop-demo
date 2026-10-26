#!/usr/bin/env python3

import cv2
import numpy as np

from skimage.filters import frangi


class GuidedWireTrackerV2:
    """
    R1-A7 local thin-wire tracker.

    Design:
      1. User initializes several ordered points along the real wire.
      2. Lucas-Kanade predicts the wire in the next frame.
      3. Frangi ridge is computed ONLY near the previous/predicted wire.
      4. Each point is refined only along its local normal direction.
      5. Strong temporal/geometric gates reject jumps.
      6. No global graph reacquisition is allowed.

    Point order:
      points[0] = free wire tip
      points[-1] = inward direction along wire
    """

    def __init__(
        self,
        sample_step_px=4.0,
        roi_margin_px=45,
        normal_search_px=6,
    ):
        self.sample_step_px = float(sample_step_px)
        self.roi_margin_px = int(roi_margin_px)
        self.normal_search_px = int(normal_search_px)

        self.prev_gray = None
        self.points = None
        self.prev_length = None

        self.missed = 0
        self.lost = False

    # --------------------------------------------------------

    @staticmethod
    def _polyline_length(points):

        if points is None or len(points) < 2:
            return 0.0

        d = points[1:] - points[:-1]

        return float(
            np.sum(
                np.linalg.norm(
                    d,
                    axis=1
                )
            )
        )

    # --------------------------------------------------------

    def _densify(self, control_points):

        cps = np.asarray(
            control_points,
            dtype=np.float32
        ).reshape(-1, 2)

        output = []

        for i in range(
            len(cps) - 1
        ):

            p0 = cps[i]
            p1 = cps[i + 1]

            dist = float(
                np.linalg.norm(
                    p1 - p0
                )
            )

            n = max(
                2,
                int(
                    np.ceil(
                        dist /
                        self.sample_step_px
                    )
                )
            )

            for j in range(n):

                t = j / float(n)

                output.append(
                    p0 * (1.0 - t)
                    +
                    p1 * t
                )

        output.append(
            cps[-1]
        )

        return np.asarray(
            output,
            dtype=np.float32
        )

    # --------------------------------------------------------

    @staticmethod
    def _smooth(points):

        if len(points) < 5:
            return points.copy()

        p = points.astype(
            np.float32
        )

        out = p.copy()

        # Do not move the free endpoint aggressively.
        for i in range(
            1,
            len(p) - 1
        ):

            i0 = max(
                0,
                i - 2
            )

            i1 = min(
                len(p),
                i + 3
            )

            out[i] = np.mean(
                p[i0:i1],
                axis=0
            )

        return out

    # --------------------------------------------------------

    def initialize(
        self,
        frame,
        control_points,
    ):

        if len(control_points) < 4:

            raise ValueError(
                "Need at least 4 ordered wire points"
            )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        self.points = self._densify(
            control_points
        )

        self.points = self._smooth(
            self.points
        )

        self.prev_gray = gray

        self.prev_length = (
            self._polyline_length(
                self.points
            )
        )

        self.missed = 0
        self.lost = False

        return self._make_result(
            True,
            "INIT",
            1.0,
            1.0,
            1.0,
            0.0
        )

    # --------------------------------------------------------

    def _make_ridge_roi(
        self,
        gray,
        predicted
    ):

        h, w = gray.shape[:2]

        xmin = max(
            0,
            int(
                np.floor(
                    np.min(
                        predicted[:, 0]
                    )
                )
            )
            -
            self.roi_margin_px
        )

        xmax = min(
            w,
            int(
                np.ceil(
                    np.max(
                        predicted[:, 0]
                    )
                )
            )
            +
            self.roi_margin_px
            +
            1
        )

        ymin = max(
            0,
            int(
                np.floor(
                    np.min(
                        predicted[:, 1]
                    )
                )
            )
            -
            self.roi_margin_px
        )

        ymax = min(
            h,
            int(
                np.ceil(
                    np.max(
                        predicted[:, 1]
                    )
                )
            )
            +
            self.roi_margin_px
            +
            1
        )

        if (
            xmax <= xmin
            or
            ymax <= ymin
        ):
            return None

        crop = gray[
            ymin:ymax,
            xmin:xmax
        ]

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(
            crop
        )

        image_f = (
            enhanced.astype(
                np.float32
            )
            /
            255.0
        )

        ridge = frangi(
            image_f,
            sigmas=[
                0.8,
                1.2,
                1.6,
                2.0,
            ],
            black_ridges=False
        )

        ridge = np.nan_to_num(
            ridge,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(
            np.float32
        )

        scale = float(
            np.percentile(
                ridge,
                99.5
            )
        )

        if scale > 1e-8:

            ridge = np.clip(
                ridge / scale,
                0.0,
                1.0
            )

        return (
            ridge,
            xmin,
            ymin
        )

    # --------------------------------------------------------

    def _refine_normal(
        self,
        predicted,
        ridge_pack
    ):

        if ridge_pack is None:
            return predicted.copy(), 0.0

        ridge, x0, y0 = ridge_pack

        rh, rw = ridge.shape[:2]

        refined = predicted.copy()

        scores = []

        for i in range(
            len(predicted)
        ):

            if i == 0:

                tangent = (
                    predicted[
                        min(
                            2,
                            len(predicted) - 1
                        )
                    ]
                    -
                    predicted[0]
                )

            elif i == len(
                predicted
            ) - 1:

                tangent = (
                    predicted[-1]
                    -
                    predicted[
                        max(
                            0,
                            len(predicted) - 3
                        )
                    ]
                )

            else:

                tangent = (
                    predicted[
                        min(
                            len(predicted) - 1,
                            i + 2
                        )
                    ]
                    -
                    predicted[
                        max(
                            0,
                            i - 2
                        )
                    ]
                )

            n = float(
                np.linalg.norm(
                    tangent
                )
            )

            if n < 1e-6:

                scores.append(
                    0.0
                )

                continue

            tangent = tangent / n

            normal = np.array(
                [
                    -tangent[1],
                    tangent[0]
                ],
                dtype=np.float32
            )

            best_score = -1.0
            best_point = predicted[i]

            for offset in range(
                -self.normal_search_px,
                self.normal_search_px + 1
            ):

                candidate = (
                    predicted[i]
                    +
                    normal
                    *
                    float(offset)
                )

                rx = int(
                    round(
                        candidate[0]
                    )
                ) - x0

                ry = int(
                    round(
                        candidate[1]
                    )
                ) - y0

                if (
                    rx < 0
                    or
                    ry < 0
                    or
                    rx >= rw
                    or
                    ry >= rh
                ):
                    continue

                score = float(
                    ridge[
                        ry,
                        rx
                    ]
                )

                if score > best_score:

                    best_score = score
                    best_point = candidate

            # Do not snap onto a random weak ridge.
            if best_score >= 0.08:

                refined[i] = best_point

            scores.append(
                max(
                    0.0,
                    best_score
                )
            )

        mean_score = float(
            np.mean(
                scores
            )
        ) if scores else 0.0

        return (
            refined,
            mean_score
        )

    # --------------------------------------------------------

    @staticmethod
    def _direction(points):

        if (
            points is None
            or
            len(points) < 4
        ):
            return None

        d = (
            points[3]
            -
            points[0]
        )

        n = float(
            np.linalg.norm(
                d
            )
        )

        if n < 1e-6:
            return None

        return d / n

    # --------------------------------------------------------

    def update(
        self,
        frame
    ):

        if (
            self.points is None
            or
            self.prev_gray is None
        ):

            return self._make_result(
                False,
                "NOT_INITIALIZED",
                0.0,
                0.0,
                0.0,
                0.0
            )

        if self.lost:

            return self._make_result(
                False,
                "LOST_REINIT_REQUIRED",
                0.0,
                0.0,
                0.0,
                0.0
            )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        p0 = self.points.reshape(
            -1,
            1,
            2
        ).astype(
            np.float32
        )

        p1, st1, _ = (
            cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                p0,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    |
                    cv2.TERM_CRITERIA_COUNT,
                    25,
                    0.02
                )
            )
        )

        if (
            p1 is None
            or
            st1 is None
        ):

            self.missed += 1

            if self.missed >= 3:
                self.lost = True

            return self._make_result(
                False,
                "FLOW_LOST",
                0.0,
                0.0,
                0.0,
                0.0
            )

        p0_back, st2, _ = (
            cv2.calcOpticalFlowPyrLK(
                gray,
                self.prev_gray,
                p1,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    |
                    cv2.TERM_CRITERIA_COUNT,
                    25,
                    0.02
                )
            )
        )

        next_points = p1.reshape(
            -1,
            2
        )

        back_points = (
            p0_back.reshape(
                -1,
                2
            )
            if p0_back is not None
            else self.points.copy()
        )

        fb_error = np.linalg.norm(
            back_points
            -
            self.points,
            axis=1
        )

        good = (
            st1.reshape(-1).astype(bool)
            &
            st2.reshape(-1).astype(bool)
            &
            (fb_error < 2.0)
        )

        good_fraction = float(
            np.mean(
                good
            )
        )

        if np.count_nonzero(
            good
        ) < max(
            6,
            int(
                0.45
                *
                len(
                    self.points
                )
            )
        ):

            self.missed += 1

            if self.missed >= 3:
                self.lost = True

            return self._make_result(
                False,
                "FLOW_WEAK",
                0.0,
                good_fraction,
                0.0,
                0.0
            )

        flows = (
            next_points[good]
            -
            self.points[good]
        )

        median_flow = np.median(
            flows,
            axis=0
        )

        median_motion = float(
            np.linalg.norm(
                median_flow
            )
        )

        # Large one-frame global motion is rejected.
        if median_motion > 25.0:

            self.missed += 1

            if self.missed >= 3:
                self.lost = True

            return self._make_result(
                False,
                "MOTION_GATE",
                0.0,
                good_fraction,
                0.0,
                median_motion
            )

        predicted = (
            self.points
            +
            median_flow
        )

        predicted[
            good
        ] = next_points[
            good
        ]

        ridge_pack = (
            self._make_ridge_roi(
                gray,
                predicted
            )
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

        new_length = (
            self._polyline_length(
                refined
            )
        )

        length_ratio = (
            new_length
            /
            max(
                1e-6,
                self.prev_length
            )
        )

        tip_jump = float(
            np.linalg.norm(
                refined[0]
                -
                self.points[0]
            )
        )

        old_dir = self._direction(
            self.points
        )

        new_dir = self._direction(
            refined
        )

        angle_deg = 0.0

        if (
            old_dir is not None
            and
            new_dir is not None
        ):

            dot = float(
                np.clip(
                    np.dot(
                        old_dir,
                        new_dir
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

        geometry_ok = (
            0.70
            <=
            length_ratio
            <=
            1.35
            and
            tip_jump
            <=
            20.0
            and
            angle_deg
            <=
            40.0
        )

        ridge_ok = (
            ridge_score
            >=
            0.08
        )

        flow_ok = (
            good_fraction
            >=
            0.55
        )

        valid = (
            geometry_ok
            and
            ridge_ok
            and
            flow_ok
        )

        flow_conf = np.clip(
            good_fraction,
            0.0,
            1.0
        )

        ridge_conf = np.clip(
            ridge_score / 0.30,
            0.0,
            1.0
        )

        geom_conf = np.clip(
            1.0
            -
            angle_deg / 45.0,
            0.0,
            1.0
        )

        confidence = float(
            0.45 * flow_conf
            +
            0.35 * ridge_conf
            +
            0.20 * geom_conf
        )

        if valid:

            self.points = (
                refined.astype(
                    np.float32
                )
            )

            self.prev_gray = gray

            self.prev_length = (
                new_length
            )

            self.missed = 0

            return self._make_result(
                True,
                "TRACK",
                confidence,
                good_fraction,
                ridge_score,
                median_motion
            )

        # ----------------------------------------------------
        # Failure handling:
        #
        # DO NOT globally search another ridge.
        # Carry only a small robust global image displacement.
        # Publication remains invalid during this state.
        # ----------------------------------------------------

        self.missed += 1

        if (
            good_fraction >= 0.65
            and
            median_motion <= 12.0
            and
            self.missed < 3
        ):

            self.points = (
                self.points
                +
                median_flow
            ).astype(
                np.float32
            )

            self.prev_gray = gray

            return self._make_result(
                False,
                "HOLD_FLOW",
                confidence,
                good_fraction,
                ridge_score,
                median_motion
            )

        if self.missed >= 3:
            self.lost = True

        return self._make_result(
            False,
            "GATE_REJECT",
            confidence,
            good_fraction,
            ridge_score,
            median_motion
        )

    # --------------------------------------------------------

    def _make_result(
        self,
        valid,
        state,
        confidence,
        flow_fraction,
        ridge_score,
        motion_px,
    ):

        if self.points is None:

            centerline = []
            tip = None
            direction = None

        else:

            centerline = (
                np.rint(
                    self.points
                )
                .astype(int)
                .tolist()
            )

            tip = (
                np.rint(
                    self.points[0]
                )
                .astype(int)
                .tolist()
            )

            d = self._direction(
                self.points
            )

            direction = (
                d.astype(float).tolist()
                if d is not None
                else None
            )

        return {
            "valid":
                bool(valid),

            "state":
                str(state),

            "tip":
                tip,

            "centerline":
                centerline,

            "direction":
                direction,

            "confidence":
                float(confidence),

            "flow_fraction":
                float(flow_fraction),

            "ridge_score":
                float(ridge_score),

            "motion_px":
                float(motion_px),

            "missed":
                int(self.missed),

            "needs_reinit":
                bool(self.lost),
        }


def grasp_target_from_centerline(
    centerline,
    arc_length_px=30.0,
):

    pts = np.asarray(
        centerline,
        dtype=np.float64
    ).reshape(-1, 2)

    if len(pts) < 4:
        return None

    lengths = np.linalg.norm(
        pts[1:] - pts[:-1],
        axis=1
    )

    cumulative = np.concatenate(
        [
            [0.0],
            np.cumsum(
                lengths
            )
        ]
    )

    if cumulative[-1] < arc_length_px:
        return None

    idx = int(
        np.searchsorted(
            cumulative,
            arc_length_px
        )
    )

    idx = min(
        idx,
        len(pts) - 1
    )

    return pts[idx]
