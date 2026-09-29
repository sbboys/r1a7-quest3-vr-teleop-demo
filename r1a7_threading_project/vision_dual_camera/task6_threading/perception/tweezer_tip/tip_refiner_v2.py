#!/usr/bin/env python3

import cv2
import numpy as np


class TweezerTipRefinerV2:

    """
    V2 principle:

    SAM2/PCA:
        only provides coarse tip + shaft direction.

    Fine tip:
        uses ORIGINAL RGB edges.
        DOES NOT clip tip edges using SAM2 mask.

    In PCA-aligned coordinates:
        +x = toward tweezer tips

        upper jaw ---->
                       *
        ---------------- center axis
                       *
        lower jaw ---->

    Each jaw edge component is tracked from the shaft-side
    toward its forward-most endpoint.
    """

    def __init__(
        self,
        back=50,
        front=40,
        half_width=40,
        canny_low=30,
        canny_high=90,
        x_back=35,
        x_front=30,
        y_band=28,
    ):

        self.back = int(back)
        self.front = int(front)
        self.half_width = int(
            half_width
        )

        self.canny_low = int(
            canny_low
        )

        self.canny_high = int(
            canny_high
        )

        self.x_back = int(
            x_back
        )

        self.x_front = int(
            x_front
        )

        self.y_band = int(
            y_band
        )

    def _make_oriented_patch(
        self,
        image,
        coarse_tip,
        direction,
    ):

        tip = np.asarray(
            coarse_tip,
            dtype=np.float64,
        )

        d = np.asarray(
            direction,
            dtype=np.float64,
        )

        d /= (
            np.linalg.norm(d)
            +
            1e-12
        )

        n = np.array(
            [
                -d[1],
                d[0],
            ],
            dtype=np.float64,
        )

        width = (
            self.back
            +
            self.front
            +
            1
        )

        height = (
            2
            *
            self.half_width
            +
            1
        )

        cols, rows = np.meshgrid(
            np.arange(
                width,
                dtype=np.float32,
            ),
            np.arange(
                height,
                dtype=np.float32,
            ),
        )

        x_local = (
            cols
            -
            float(self.back)
        )

        y_local = (
            rows
            -
            float(
                self.half_width
            )
        )

        map_x = (
            tip[0]
            +
            x_local * d[0]
            +
            y_local * n[0]
        ).astype(
            np.float32
        )

        map_y = (
            tip[1]
            +
            x_local * d[1]
            +
            y_local * n[1]
        ).astype(
            np.float32
        )

        patch = cv2.remap(
            image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=
            cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return (
            patch,
            d,
            n,
        )

    def _find_side_tip(
        self,
        edges,
        upper,
    ):

        h, w = edges.shape

        back = self.back
        half = self.half_width

        roi = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        x0 = max(
            0,
            back
            -
            self.x_back,
        )

        x1 = min(
            w,
            back
            +
            self.x_front
            +
            1,
        )

        if upper:

            y0 = max(
                0,
                half
                -
                self.y_band,
            )

            y1 = half

        else:

            y0 = min(
                h,
                half + 1,
            )

            y1 = min(
                h,
                half
                +
                self.y_band
                +
                1,
            )

        roi[
            y0:y1,
            x0:x1
        ] = (
            edges[
                y0:y1,
                x0:x1
            ]
            >
            0
        ).astype(
            np.uint8
        )

        # Connect fragmented jaw edges.
        connected = cv2.dilate(
            roi,
            np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
            iterations=1,
        )

        connected = (
            cv2.morphologyEx(
                connected,
                cv2.MORPH_CLOSE,
                np.ones(
                    (3, 5),
                    dtype=np.uint8,
                ),
            )
        )

        (
            count,
            labels,
            stats,
            centroids,
        ) = (
            cv2.connectedComponentsWithStats(
                connected,
                connectivity=8,
            )
        )

        best = None

        for label in range(
            1,
            count,
        ):

            (
                bx,
                by,
                bw,
                bh,
                area,
            ) = stats[
                label
            ]

            if area < 10:
                continue

            component = (
                labels
                ==
                label
            )

            cy, cx = np.where(
                component
            )

            # Component must connect back toward
            # the visible jaw/shaft.
            seed_ok = np.any(
                (
                    cx
                    >=
                    back - 30
                )
                &
                (
                    cx
                    <=
                    back - 5
                )
            )

            if not seed_ok:
                continue

            ey, ex = np.where(
                (roi > 0)
                &
                component
            )

            if len(ex) < 5:
                continue

            extent = int(
                ex.max()
                -
                ex.min()
            )

            if extent < 8:
                continue

            xmax = int(
                ex.max()
            )

            # Prefer the component which reaches
            # furthest toward the real tip.
            score = (
                xmax,
                extent,
                len(ex),
            )

            if (
                best is None
                or
                score
                >
                best["score"]
            ):

                front_sel = (
                    ex
                    >=
                    xmax - 2
                )

                tip_x = float(
                    np.median(
                        ex[
                            front_sel
                        ]
                    )
                )

                tip_y = float(
                    np.median(
                        ey[
                            front_sel
                        ]
                    )
                )

                best = {
                    "score":
                    score,

                    "tip":
                    (
                        tip_x,
                        tip_y,
                    ),

                    "edge_count":
                    int(
                        len(ex)
                    ),

                    "extent":
                    extent,

                    "xmax":
                    xmax,
                }

        return (
            best,
            roi,
            connected,
        )

    def _patch_to_global(
        self,
        coarse_tip,
        direction,
        normal,
        point,
    ):

        px, py = point

        x_local = (
            px
            -
            self.back
        )

        y_local = (
            py
            -
            self.half_width
        )

        return (
            np.asarray(
                coarse_tip,
                dtype=np.float64,
            )
            +
            x_local
            *
            direction
            +
            y_local
            *
            normal
        )

    def refine(
        self,
        frame_bgr,
        mask_u8,
        coarse_tip,
        direction,
        return_debug=False,
    ):

        # mask_u8 is intentionally NOT used
        # for fine edge clipping in V2.
        #
        # It remains in the API for compatibility
        # with the existing pipeline.

        patch, d, n = (
            self._make_oriented_patch(
                frame_bgr,
                coarse_tip,
                direction,
            )
        )

        gray = cv2.cvtColor(
            patch,
            cv2.COLOR_BGR2GRAY,
        )

        blur = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        edges = cv2.Canny(
            blur,
            self.canny_low,
            self.canny_high,
        )

        upper, upper_roi, upper_conn = (
            self._find_side_tip(
                edges,
                upper=True,
            )
        )

        lower, lower_roi, lower_conn = (
            self._find_side_tip(
                edges,
                upper=False,
            )
        )

        if (
            upper is None
            or
            lower is None
        ):

            result = {
                "valid": False,
                "reason":
                "jaw_component_not_found",
            }

            if return_debug:

                result.update(
                    {
                        "patch":
                        patch,

                        "edges":
                        edges,
                    }
                )

            return result

        upper_patch = (
            upper["tip"]
        )

        lower_patch = (
            lower["tip"]
        )

        jaw1 = (
            self._patch_to_global(
                coarse_tip,
                d,
                n,
                upper_patch,
            )
        )

        jaw2 = (
            self._patch_to_global(
                coarse_tip,
                d,
                n,
                lower_patch,
            )
        )

        refined = (
            jaw1
            +
            jaw2
        ) / 2.0

        jaw_gap = float(
            np.linalg.norm(
                jaw1
                -
                jaw2
            )
        )

        longitudinal_diff = abs(
            upper_patch[0]
            -
            lower_patch[0]
        )

        mean_forward = (
            (
                upper_patch[0]
                +
                lower_patch[0]
            )
            /
            2.0
            -
            self.back
        )

        # Simple geometric confidence.
        #
        # Not a calibrated probability.
        longitudinal_score = max(
            0.0,
            1.0
            -
            longitudinal_diff
            /
            15.0,
        )

        forward_score = (
            1.0
            if mean_forward >= -5.0
            else
            max(
                0.0,
                1.0
                -
                (
                    -5.0
                    -
                    mean_forward
                )
                /
                15.0
            )
        )

        confidence = float(
            longitudinal_score
            *
            forward_score
        )

        result = {
            "valid": True,

            "jaw_tip_1_uv": [
                float(jaw1[0]),
                float(jaw1[1]),
            ],

            "jaw_tip_2_uv": [
                float(jaw2[0]),
                float(jaw2[1]),
            ],

            "refined_tip_uv": [
                float(refined[0]),
                float(refined[1]),
            ],

            "jaw_gap_px":
            jaw_gap,

            "jaw1_patch_uv": [
                float(
                    upper_patch[0]
                ),
                float(
                    upper_patch[1]
                ),
            ],

            "jaw2_patch_uv": [
                float(
                    lower_patch[0]
                ),
                float(
                    lower_patch[1]
                ),
            ],

            "longitudinal_diff_px":
            float(
                longitudinal_diff
            ),

            "mean_forward_px":
            float(
                mean_forward
            ),

            "confidence":
            confidence,
        }

        if return_debug:

            result.update(
                {
                    "patch":
                    patch,

                    "edges":
                    edges,

                    "upper_roi":
                    upper_roi,

                    "lower_roi":
                    lower_roi,

                    "upper_connected":
                    upper_conn,

                    "lower_connected":
                    lower_conn,
                }
            )

        return result
