#!/usr/bin/env python3

import numpy as np

from wire_local_detector_v61 import WireLocalDetectorV61
from wire_local_detector_v63 import WireLocalDetectorV63


class WireLocalDetectorV64(WireLocalDetectorV63):
    """
    V6.4

    Two visual modes:

    1. TIP_VISIBLE
       Normal free-tip detection.

    2. OCCLUDED_VISIBLE_SEGMENT
       Free tip is hidden by the approaching tweezer, but a
       currently visible wire segment is still detected from
       the CURRENT image.

    No FK/depth propagation is used for wire identity.
    """

    def __init__(
        self,
        local_length_px=180.0,
        grasp_offset_px=55.0,
        grasp_band_start_px=35.0,
        grasp_band_end_px=90.0,
    ):

        super().__init__(
            local_length_px=local_length_px,
            grasp_offset_px=grasp_offset_px,
        )

        self.grasp_band_start_px = float(
            grasp_band_start_px
        )

        self.grasp_band_end_px = float(
            grasp_band_end_px
        )


    # ========================================================
    # Arc-length utilities
    # ========================================================

    @staticmethod
    def _arc_coordinates(path):

        p = np.asarray(
            path,
            dtype=np.float64
        ).reshape(-1, 2)

        if len(p) < 2:
            return p, None

        ds = np.linalg.norm(
            p[1:] - p[:-1],
            axis=1
        )

        s = np.concatenate(
            [
                [0.0],
                np.cumsum(ds)
            ]
        )

        return p, s


    def _grasp_band_from_tip(
        self,
        path
    ):

        p, s = self._arc_coordinates(
            path
        )

        if (
            s is None
            or
            s[-1] <
            self.grasp_band_start_px
        ):
            return None

        mask = (
            (s >= self.grasp_band_start_px)
            &
            (s <= self.grasp_band_end_px)
        )

        band = p[mask]

        if len(band) < 2:
            return None

        return band


    @staticmethod
    def _nearest_index(
        path,
        point
    ):

        p = np.asarray(
            path,
            dtype=np.float64
        )

        q = np.asarray(
            point,
            dtype=np.float64
        )

        d = np.linalg.norm(
            p - q,
            axis=1
        )

        return int(
            np.argmin(d)
        )


    @staticmethod
    def _local_segment_around_index(
        path,
        index,
        radius_px=40.0,
    ):

        p = np.asarray(
            path,
            dtype=np.float64
        )

        if len(p) < 2:
            return None

        selected = [
            p[index]
        ]

        length = 0.0
        i = index

        while i > 0:

            ds = float(
                np.linalg.norm(
                    p[i] - p[i - 1]
                )
            )

            if length + ds > radius_px:
                break

            selected.insert(
                0,
                p[i - 1]
            )

            length += ds
            i -= 1

        length = 0.0
        i = index

        while i < len(p) - 1:

            ds = float(
                np.linalg.norm(
                    p[i + 1] - p[i]
                )
            )

            if length + ds > radius_px:
                break

            selected.append(
                p[i + 1]
            )

            length += ds
            i += 1

        if len(selected) < 2:
            return None

        return np.asarray(
            selected,
            dtype=np.float64
        )


    # ========================================================
    # Free-tip hidden fallback
    # ========================================================

    def _detect_visible_segment(
        self,
        frame,
        roi,
        tweezer_exclusion_mask,
        tweezer_tip,
    ):

        if tweezer_tip is None:
            return None

        x, y, w, h = [
            int(v)
            for v in roi
        ]

        crop = frame[
            y:y+h,
            x:x+w
        ]

        if crop.size == 0:
            return None

        if (
            tweezer_exclusion_mask
            is not None
        ):

            self._exclusion_crop = (
                tweezer_exclusion_mask[
                    y:y+h,
                    x:x+w
                ]
            )

        else:

            self._exclusion_crop = None

        gray, response = (
            self._wire_response(
                crop
            )
        )

        # ----------------------------------------------------
        # In occlusion mode:
        # do NOT require the component endpoint to coincide
        # with the old free-tip coordinate.
        #
        # We still require CURRENT-FRAME ridge evidence.
        # ----------------------------------------------------

        old_seed = self.seed_tip
        old_last = self.last_valid_tip

        try:

            self.seed_tip = None
            self.last_valid_tip = None

            candidates = (
                WireLocalDetectorV61
                ._candidate_components(
                    self,
                    gray,
                    response
                )
            )

        finally:

            self.seed_tip = old_seed
            self.last_valid_tip = old_last

        if not candidates:
            return None

        tool_tip = np.asarray(
            tweezer_tip,
            dtype=np.float64
        )

        best = None
        best_score = -1e9

        for c in candidates:

            local_path = np.asarray(
                c["path"],
                dtype=np.float64
            )

            if len(local_path) < 15:
                continue

            full_path = (
                local_path
                +
                np.array(
                    [x, y],
                    dtype=np.float64
                )
            )

            distance = np.min(
                np.linalg.norm(
                    full_path
                    -
                    tool_tip,
                    axis=1
                )
            )

            # In the final grasp stage the visible wire
            # should be reasonably near the tweezer.
            if distance > 120.0:
                continue

            rank = float(
                c.get(
                    "rank_score",
                    c.get(
                        "score",
                        0.0
                    )
                )
            )

            profile = float(
                c.get(
                    "profile_score",
                    0.0
                )
            )

            proximity = float(
                np.exp(
                    -
                    (
                        distance / 70.0
                    ) ** 2
                )
            )

            score = (
                rank
                +
                0.25 * proximity
                +
                0.10 * profile
            )

            if score > best_score:

                best_score = score

                best = {
                    "path": full_path,
                    "distance":
                        float(distance),
                    "score":
                        float(score),
                }

        if best is None:
            return None

        path = best["path"]

        nearest_index = (
            self._nearest_index(
                path,
                tool_tip
            )
        )

        target = (
            path[
                nearest_index
            ]
        )

        visible_grasp_segment = (
            self._local_segment_around_index(
                path,
                nearest_index,
                radius_px=35.0
            )
        )

        return {
            "valid": False,

            # Important:
            # free-tip localization is NOT valid.
            "tip_valid": False,

            # But CURRENT IMAGE still contains a usable wire
            # segment for grasp alignment.
            "grasp_guidance_valid": True,

            "state":
                "OCCLUDED_VISIBLE_SEGMENT",

            "visibility_mode":
                "TIP_OCCLUDED",

            "wire_tip_uv": None,

            "centerline":
                np.rint(
                    path
                ).astype(
                    int
                ).tolist(),

            "grasp_target_uv":
                np.rint(
                    target
                ).astype(
                    int
                ).tolist(),

            "grasp_segment_uv":
                (
                    np.rint(
                        visible_grasp_segment
                    )
                    .astype(int)
                    .tolist()
                    if
                    visible_grasp_segment
                    is not None
                    else
                    []
                ),

            "occlusion_distance_px":
                float(
                    best["distance"]
                ),

            "score":
                float(
                    best["score"]
                ),

            "candidate_count":
                int(
                    len(candidates)
                ),
        }


    # ========================================================
    # Public detector
    # ========================================================

    def detect(
        self,
        frame,
        roi,
        tweezer_exclusion_mask=None,
        tweezer_tip=None,
    ):

        result = super().detect(
            frame,
            roi,
            tweezer_exclusion_mask=
                tweezer_exclusion_mask
        )

        # ----------------------------------------------------
        # Normal mode: free tip visible
        # ----------------------------------------------------

        if result.get(
            "valid",
            False
        ):

            path = np.asarray(
                result[
                    "centerline"
                ],
                dtype=np.float64
            )

            band = (
                self._grasp_band_from_tip(
                    path
                )
            )

            if band is not None:

                # Select by wire geometry, not moving tweezer proximity.
                target = band[
                    len(band) // 2
                ]

                result[
                    "grasp_target_uv"
                ] = (
                    np.rint(
                        target
                    )
                    .astype(int)
                    .tolist()
                )

                result[
                    "grasp_segment_uv"
                ] = (
                    np.rint(
                        band
                    )
                    .astype(int)
                    .tolist()
                )

            else:

                result[
                    "grasp_segment_uv"
                ] = []

            result[
                "tip_valid"
            ] = True

            result[
                "grasp_guidance_valid"
            ] = (
                result.get(
                    "grasp_target_uv"
                )
                is not None
            )

            result[
                "visibility_mode"
            ] = "TIP_VISIBLE"

            return result

        # ----------------------------------------------------
        # Tip lost:
        # try CURRENT-IMAGE visible-segment recognition.
        # ----------------------------------------------------

        fallback = (
            self._detect_visible_segment(
                frame,
                roi,
                tweezer_exclusion_mask,
                tweezer_tip
            )
        )

        if fallback is not None:
            return fallback

        result[
            "tip_valid"
        ] = False

        result[
            "grasp_guidance_valid"
        ] = False

        result[
            "visibility_mode"
        ] = "WIRE_LOST"

        result[
            "grasp_segment_uv"
        ] = []

        return result
