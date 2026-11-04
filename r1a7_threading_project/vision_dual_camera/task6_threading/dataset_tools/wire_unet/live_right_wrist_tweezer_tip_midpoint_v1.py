#!/usr/bin/env python3

import cv2
import numpy as np

import live_hybrid_locked_wire_live_tweezer_tip_v2 as v2


# ============================================================
# RIGHT WRIST OPEN-TWEEZER PARAMETERS
# ============================================================

# A component smaller than this is treated as segmentation noise.
OPEN_MIN_COMPONENT_PIXELS = 80

# Second blade must be a meaningful fraction of the largest blade.
OPEN_SECOND_COMPONENT_RATIO = 0.20

# Two detected front tips must be separated by at least this many
# pixels before we classify the tweezer as OPEN.
OPEN_TIP_SEPARATION_MIN_PX = 8.0


class RightWristTweezerTipMidpointV1:
    """
    Right Wrist only.

    CLOSED:
        fall back to the original TweezerTipTrackerV2.

    OPEN:
        detect two significant tool components,
        estimate one forward tip from each component,
        and use their midpoint as the final TIP.
    """

    def __init__(
        self,
        camera_name,
    ):
        self.camera_name = camera_name

        self.closed_tracker = (
            v2.TweezerTipTrackerV2(
                camera_name
            )
        )

    def clear(self):
        self.closed_tracker.clear()

    @staticmethod
    def _component_tip(
        component_mask,
        grasp_yx,
    ):
        """
        Estimate the forward-most tip of ONE blade.

        Uses PCA exactly for the same geometric reason as the
        original V2 tracker:
          - estimate blade longitudinal direction
          - orient it toward GRASP
          - choose the contour extreme in that direction
        """

        ys, xs = np.where(
            component_mask
        )

        if len(xs) < OPEN_MIN_COMPONENT_PIXELS:
            return None

        xy = np.column_stack(
            (
                xs.astype(np.float32),
                ys.astype(np.float32),
            )
        )

        center = np.mean(
            xy,
            axis=0,
        )

        centered = (
            xy
            -
            center
        )

        cov = (
            centered.T
            @
            centered
        )

        cov /= max(
            len(centered) - 1,
            1,
        )

        eigvals, eigvecs = (
            np.linalg.eigh(
                cov
            )
        )

        order = np.argsort(
            eigvals
        )

        axis = eigvecs[
            :,
            order[-1]
        ].astype(
            np.float32
        )

        axis /= max(
            float(
                np.linalg.norm(axis)
            ),
            1e-6,
        )

        gy = float(
            grasp_yx[0]
        )

        gx = float(
            grasp_yx[1]
        )

        grasp_xy = np.array(
            [gx, gy],
            dtype=np.float32,
        )

        # Orient PCA axis toward GRASP.
        if (
            float(
                np.dot(
                    axis,
                    grasp_xy - center,
                )
            )
            <
            0.0
        ):
            axis = -axis

        # Use component contour instead of filled pixels.
        contours, _ = cv2.findContours(
            component_mask.astype(
                np.uint8
            ),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            return None

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        if contour is None or len(contour) < 1:
            return None

        contour_xy = (
            contour[
                :,
                0,
                :
            ].astype(
                np.float32
            )
        )

        projection = (
            contour_xy
            -
            center
        ) @ axis

        max_projection = float(
            np.max(
                projection
            )
        )

        # Keep a very thin extreme-front band.
        front = contour_xy[
            projection
            >=
            (
                max_projection
                -
                v2.TIP_FRONT_BAND_PX
            )
        ]

        if len(front) < 1:
            return None

        # Average only the extreme contour pixels.
        tip_xy = np.mean(
            front,
            axis=0,
        )

        return {
            "tip_yx": (
                float(tip_xy[1]),
                float(tip_xy[0]),
            ),

            "center_xy":
                center,

            "axis_xy":
                axis,

            "front_pixels":
                int(len(front)),

            "component_pixels":
                int(len(xs)),
        }

    def update(
        self,
        tool_mask,
        grasp_yx,
    ):

        # ----------------------------------------------------
        # Always compute original V2 result first.
        # This is the CLOSED fallback.
        # ----------------------------------------------------

        closed_result = (
            self.closed_tracker.update(
                tool_mask,
                grasp_yx,
            )
        )

        # Only Right Wrist gets special OPEN logic.
        if self.camera_name != "right_wrist":
            return closed_result

        if grasp_yx is None:
            return closed_result

        tool = np.asarray(
            tool_mask,
            dtype=bool,
        )

        # ----------------------------------------------------
        # Find tool connected components.
        # ----------------------------------------------------

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                tool.astype(
                    np.uint8
                ),
                connectivity=8,
            )
        )

        components = []

        for label in range(
            1,
            num_labels,
        ):

            area = int(
                stats[
                    label,
                    cv2.CC_STAT_AREA,
                ]
            )

            if area < OPEN_MIN_COMPONENT_PIXELS:
                continue

            components.append(
                (
                    area,
                    label,
                )
            )

        components.sort(
            reverse=True
        )

        # One meaningful component:
        # keep original CLOSED behavior.
        if len(components) < 2:

            closed_result[
                "tip_mode"
            ] = "SINGLE_CLOSED"

            closed_result[
                "tip_pair_yx"
            ] = None

            closed_result[
                "tip_pair_separation_px"
            ] = None

            return closed_result

        area1, label1 = components[0]
        area2, label2 = components[1]

        # Avoid treating tiny noise as a second blade.
        if (
            area2
            <
            area1
            *
            OPEN_SECOND_COMPONENT_RATIO
        ):

            closed_result[
                "tip_mode"
            ] = "SINGLE_CLOSED"

            closed_result[
                "tip_pair_yx"
            ] = None

            closed_result[
                "tip_pair_separation_px"
            ] = None

            return closed_result

        comp1 = (
            labels
            ==
            label1
        )

        comp2 = (
            labels
            ==
            label2
        )

        tip1 = self._component_tip(
            comp1,
            grasp_yx,
        )

        tip2 = self._component_tip(
            comp2,
            grasp_yx,
        )

        if (
            tip1 is None
            or
            tip2 is None
        ):

            closed_result[
                "tip_mode"
            ] = "SINGLE_FALLBACK"

            closed_result[
                "tip_pair_yx"
            ] = None

            closed_result[
                "tip_pair_separation_px"
            ] = None

            return closed_result

        y1, x1 = tip1[
            "tip_yx"
        ]

        y2, x2 = tip2[
            "tip_yx"
        ]

        separation = float(
            np.hypot(
                x2 - x1,
                y2 - y1,
            )
        )

        # ----------------------------------------------------
        # CLOSED:
        # two components may exist because of segmentation
        # details, but their tips are essentially coincident.
        # ----------------------------------------------------

        if (
            separation
            <
            OPEN_TIP_SEPARATION_MIN_PX
        ):

            closed_result[
                "tip_mode"
            ] = "SINGLE_CLOSED"

            closed_result[
                "tip_pair_yx"
            ] = [
                [y1, x1],
                [y2, x2],
            ]

            closed_result[
                "tip_pair_separation_px"
            ] = separation

            return closed_result

        # ----------------------------------------------------
        # OPEN:
        #
        # TIP = midpoint of the two physical blade tips.
        # ----------------------------------------------------

        mid_y = (
            y1 + y2
        ) / 2.0

        mid_x = (
            x1 + x2
        ) / 2.0

        gy = float(
            grasp_yx[0]
        )

        gx = float(
            grasp_yx[1]
        )

        distance = float(
            np.hypot(
                gx - mid_x,
                gy - mid_y,
            )
        )

        result = dict(
            closed_result
        )

        result[
            "valid"
        ] = True

        result[
            "tip_yx"
        ] = (
            float(mid_y),
            float(mid_x),
        )

        result[
            "raw_tip_yx"
        ] = (
            float(mid_y),
            float(mid_x),
        )

        result[
            "tip_grasp_distance_px"
        ] = distance

        result[
            "front_pixels"
        ] = (
            int(
                tip1[
                    "front_pixels"
                ]
            )
            +
            int(
                tip2[
                    "front_pixels"
                ]
            )
        )

        result[
            "status"
        ] = "TIP_OPEN_MIDPOINT"

        result[
            "tip_mode"
        ] = "OPEN_MIDPOINT"

        result[
            "tip_pair_yx"
        ] = [
            [
                float(y1),
                float(x1),
            ],
            [
                float(y2),
                float(x2),
            ],
        ]

        result[
            "tip_pair_separation_px"
        ] = separation

        result[
            "blade1_pixels"
        ] = int(area1)

        result[
            "blade2_pixels"
        ] = int(area2)

        return result
