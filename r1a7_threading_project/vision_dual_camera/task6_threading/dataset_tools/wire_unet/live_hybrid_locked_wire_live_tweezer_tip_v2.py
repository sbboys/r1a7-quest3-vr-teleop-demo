#!/usr/bin/env python3

import cv2
import numpy as np

import live_hybrid_locked_wire_live_tweezer_tip_v1 as v1


# ============================================================
# TIP V2 PARAMETERS
# ============================================================

# Only use the extreme front edge.
TIP_FRONT_BAND_PX = 1.25

TIP_MIN_TOOL_PIXELS = 10

TIP_COLOR = (
    255,
    0,
    255,
)


# ============================================================
# TIP V2
#
# Key changes:
#
# 1. NO temporal median filtering.
# 2. Use tool contour, not filled front region center.
# 3. PCA estimates tweezer longitudinal direction.
# 4. GRASP determines which PCA direction is "forward".
# 5. TIP is on the extreme front contour.
# ============================================================

class TweezerTipTrackerV2:

    def __init__(
        self,
        camera_name,
    ):

        self.camera_name = camera_name


    def clear(
        self,
    ):

        # V2 intentionally has no temporal position history.
        pass


    def update(
        self,
        tool_mask,
        grasp_yx,
    ):

        result = {
            "valid":
                False,

            "tip_yx":
                None,

            "raw_tip_yx":
                None,

            "grasp_yx":
                grasp_yx,

            "tip_grasp_distance_px":
                None,

            "tool_pixels":
                0,

            "front_pixels":
                0,

            "axis_ratio":
                None,

            "status":
                "INVALID",
        }

        # ----------------------------------------------------
        # Need GRASP target to decide which side is forward.
        # ----------------------------------------------------

        if grasp_yx is None:

            result[
                "status"
            ] = "NO_GRASP_TARGET"

            return result

        tool = np.asarray(
            tool_mask,
            dtype=bool,
        )

        tool_pixels = int(
            np.count_nonzero(
                tool
            )
        )

        result[
            "tool_pixels"
        ] = tool_pixels

        if (
            tool_pixels
            <
            TIP_MIN_TOOL_PIXELS
        ):

            result[
                "status"
            ] = "NO_TOOL"

            return result

        # ----------------------------------------------------
        # Keep largest connected tool component.
        #
        # This prevents small segmentation noise from becoming
        # a false "front-most" point.
        # ----------------------------------------------------

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                tool.astype(
                    np.uint8
                ),
                connectivity=8,
            )
        )

        if num_labels <= 1:

            result[
                "status"
            ] = "NO_COMPONENT"

            return result

        areas = stats[
            1:,
            cv2.CC_STAT_AREA,
        ]

        best_label = (
            1
            +
            int(
                np.argmax(
                    areas
                )
            )
        )

        tool_main = (
            labels
            ==
            best_label
        )

        ys, xs = np.where(
            tool_main
        )

        if len(xs) < TIP_MIN_TOOL_PIXELS:

            result[
                "status"
            ] = "TOOL_COMPONENT_TOO_SMALL"

            return result

        # ----------------------------------------------------
        # PCA on full tool region.
        #
        # Coordinates are XY.
        # ----------------------------------------------------

        xy = np.column_stack(
            (
                xs.astype(
                    np.float32
                ),
                ys.astype(
                    np.float32
                ),
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

        major_value = float(
            eigvals[
                order[-1]
            ]
        )

        minor_value = float(
            eigvals[
                order[-2]
            ]
        )

        axis_ratio = (
            major_value
            /
            max(
                minor_value,
                1e-6,
            )
        )

        result[
            "axis_ratio"
        ] = axis_ratio

        # Principal longitudinal axis.
        axis = eigvecs[
            :,
            order[-1]
        ].astype(
            np.float32
        )

        axis /= max(
            float(
                np.linalg.norm(
                    axis
                )
            ),
            1e-6,
        )

        # ----------------------------------------------------
        # Orient the PCA axis toward GRASP.
        # ----------------------------------------------------

        gy = float(
            grasp_yx[0]
        )

        gx = float(
            grasp_yx[1]
        )

        grasp_xy = np.array(
            [
                gx,
                gy,
            ],
            dtype=np.float32,
        )

        center_to_grasp = (
            grasp_xy
            -
            center
        )

        if (
            float(
                np.dot(
                    axis,
                    center_to_grasp,
                )
            )
            <
            0.0
        ):

            axis = -axis

        # Orthogonal direction.
        normal = np.array(
            [
                -axis[1],
                axis[0],
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Extract the actual OUTER contour.
        # ----------------------------------------------------

        contours, _ = (
            cv2.findContours(
                tool_main.astype(
                    np.uint8
                ),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
        )

        if not contours:

            result[
                "status"
            ] = "NO_CONTOUR"

            return result

        contour_xy = np.concatenate(
            [
                c.reshape(
                    -1,
                    2,
                )
                for c
                in contours
            ],
            axis=0,
        ).astype(
            np.float32
        )

        if len(contour_xy) == 0:

            result[
                "status"
            ] = "NO_CONTOUR"

            return result

        contour_centered = (
            contour_xy
            -
            center
        )

        # Longitudinal projection.
        forward_projection = (
            contour_centered
            @
            axis
        )

        max_forward = float(
            np.max(
                forward_projection
            )
        )

        # ----------------------------------------------------
        # EXTREME FRONT EDGE ONLY.
        #
        # Not 12 px anymore.
        # Only ~1.25 px from the true leading edge.
        # ----------------------------------------------------

        front_select = (
            forward_projection
            >=
            (
                max_forward
                -
                TIP_FRONT_BAND_PX
            )
        )

        front_xy = contour_xy[
            front_select
        ]

        result[
            "front_pixels"
        ] = int(
            len(
                front_xy
            )
        )

        if len(front_xy) == 0:

            result[
                "status"
            ] = "NO_FRONT_EDGE"

            return result

        # ----------------------------------------------------
        # Determine CENTER of the extreme front edge.
        #
        # We use median lateral position, while longitudinal
        # coordinate remains at the true maximum front plane.
        # ----------------------------------------------------

        front_centered = (
            front_xy
            -
            center
        )

        lateral_projection = (
            front_centered
            @
            normal
        )

        lateral_center = float(
            np.median(
                lateral_projection
            )
        )

        ideal_tip_xy = (
            center
            +
            max_forward
            * axis
            +
            lateral_center
            * normal
        )

        # ----------------------------------------------------
        # Snap back to an ACTUAL front contour pixel.
        #
        # Thus the displayed TIP is guaranteed to sit on the
        # recognized blue tool boundary.
        # ----------------------------------------------------

        delta = (
            front_xy
            -
            ideal_tip_xy
        )

        distance_sq = np.sum(
            delta
            *
            delta,
            axis=1,
        )

        best_index = int(
            np.argmin(
                distance_sq
            )
        )

        tip_xy = (
            front_xy[
                best_index
            ]
        )

        tip_x = float(
            tip_xy[0]
        )

        tip_y = float(
            tip_xy[1]
        )

        # ----------------------------------------------------
        # NO temporal position smoothing.
        #
        # Current frame result is the displayed result.
        # ----------------------------------------------------

        distance_to_grasp = float(
            np.sqrt(
                (
                    tip_x
                    -
                    gx
                ) ** 2
                +
                (
                    tip_y
                    -
                    gy
                ) ** 2
            )
        )

        result.update(
            {
                "valid":
                    True,

                "tip_yx":
                    (
                        tip_y,
                        tip_x,
                    ),

                "raw_tip_yx":
                    (
                        tip_y,
                        tip_x,
                    ),

                "tip_grasp_distance_px":
                    distance_to_grasp,

                "status":
                    "TIP_FRONT_EDGE_V2",
            }
        )

        return result


# ============================================================
# SMALL PRECISE TIP DISPLAY
# ============================================================

def draw_tip_v2(
    vis,
    result,
):

    if not result[
        "valid"
    ]:

        cv2.putText(
            vis,
            "TIP INVALID: "
            +
            result[
                "status"
            ],
            (
                12,
                133,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                0,
                165,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        return vis

    tip_y, tip_x = (
        result[
            "tip_yx"
        ]
    )

    grasp_y, grasp_x = (
        result[
            "grasp_yx"
        ]
    )

    tx = int(
        round(
            tip_x
        )
    )

    ty = int(
        round(
            tip_y
        )
    )

    gx = int(
        round(
            grasp_x
        )
    )

    gy = int(
        round(
            grasp_y
        )
    )

    # --------------------------------------------------------
    # Small cross instead of a huge ring.
    # Center of the cross is the actual TIP pixel.
    # --------------------------------------------------------

    cv2.drawMarker(
        vis,
        (
            tx,
            ty,
        ),
        TIP_COLOR,
        markerType=cv2.MARKER_CROSS,
        markerSize=9,
        thickness=1,
        line_type=cv2.LINE_AA,
    )

    # Tiny center point.
    cv2.circle(
        vis,
        (
            tx,
            ty,
        ),
        1,
        TIP_COLOR,
        -1,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        "TIP",
        (
            tx + 6,
            ty - 6,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        TIP_COLOR,
        1,
        cv2.LINE_AA,
    )

    # TIP -> GRASP.
    cv2.line(
        vis,
        (
            tx,
            ty,
        ),
        (
            gx,
            gy,
        ),
        TIP_COLOR,
        1,
        cv2.LINE_AA,
    )

    d = result[
        "tip_grasp_distance_px"
    ]

    ratio = result.get(
        "axis_ratio"
    )

    line1 = (
        f"TIP=({tx},{ty})  "
        f"d={d:.1f}px"
    )

    if ratio is not None:

        line1 += (
            f"  axis={ratio:.1f}"
        )

    cv2.putText(
        vis,
        line1,
        (
            12,
            133,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        TIP_COLOR,
        1,
        cv2.LINE_AA,
    )

    return vis


# ============================================================
# PATCH V1 HYBRID FRAMEWORK
#
# Keep all existing:
# - camera opening
# - wire locking
# - manual FREE calibration
# - right-wrist temporal tracking
#
# Replace only the TIP layer.
# ============================================================

v1.TweezerTipTracker = (
    TweezerTipTrackerV2
)

v1.draw_tip = (
    draw_tip_v2
)


if __name__ == "__main__":

    print()
    print(
        "TIP V2 ACTIVE:"
    )

    print(
        "  temporal smoothing = OFF"
    )

    print(
        "  front edge band    =",
        TIP_FRONT_BAND_PX,
        "px",
    )

    print(
        "  algorithm          = PCA + extreme contour"
    )

    print()

    v1.main()
