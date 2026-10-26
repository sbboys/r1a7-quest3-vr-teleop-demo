#!/usr/bin/env python3

import heapq

import cv2
import numpy as np

from scipy import ndimage
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize


class WireLocalDetectorV5:
    """
    Image-driven thin-wire detector.

    IMPORTANT:
      - every frame is detected from RGB again
      - NO optical-flow propagation
      - NO FK propagation
      - NO depth dependency
      - previous tip is only a soft association cue
      - invalid detection is never propagated

    Intended output:
      wire free tip
      local wire centerline
      local wire direction
      grasp target
    """

    def __init__(
        self,
        local_length_px=130.0,
        grasp_offset_px=30.0,
    ):

        self.local_length_px = float(
            local_length_px
        )

        self.grasp_offset_px = float(
            grasp_offset_px
        )

        self.seed_tip = None
        self.last_valid_tip = None

        self.confirm_candidate = None
        self.confirm_count = 0


    # ========================================================
    # Initialization
    # ========================================================

    def set_seed_tip(
        self,
        uv
    ):

        self.seed_tip = np.asarray(
            uv,
            dtype=np.float64
        )

        self.last_valid_tip = None

        self.confirm_candidate = None
        self.confirm_count = 0


    # ========================================================
    # Image enhancement
    # ========================================================

    @staticmethod
    def _normalize01(x):

        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(
            np.float32
        )

        p1 = float(
            np.percentile(
                x,
                1
            )
        )

        p99 = float(
            np.percentile(
                x,
                99.5
            )
        )

        if (
            not np.isfinite(p99)
            or
            p99 <= p1 + 1e-8
        ):

            return np.zeros_like(
                x,
                dtype=np.float32
            )

        return np.clip(
            (x - p1)
            /
            (p99 - p1),
            0.0,
            1.0
        )


    def _wire_response(
        self,
        crop
    ):

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(
            gray
        )

        f = (
            enhanced.astype(
                np.float32
            )
            /
            255.0
        )

        # Multi-scale bright ridge response.
        ridge = frangi(
            f,
            sigmas=(
                0.8,
                1.2,
                1.6,
                2.2,
            ),
            black_ridges=False
        )

        ridge = self._normalize01(
            ridge
        )

        # Bright thin-structure enhancement.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11)
        )

        tophat = cv2.morphologyEx(
            enhanced,
            cv2.MORPH_TOPHAT,
            kernel
        )

        tophat = self._normalize01(
            tophat
        )

        response = (
            0.72 * ridge
            +
            0.28 * tophat
        )

        return (
            gray,
            response
        )


    # ========================================================
    # Skeleton helpers
    # ========================================================

    @staticmethod
    def _neighbors(
        p,
        pixels
    ):

        y, x = p

        for dy in (-1, 0, 1):

            for dx in (-1, 0, 1):

                if (
                    dx == 0
                    and
                    dy == 0
                ):
                    continue

                q = (
                    y + dy,
                    x + dx
                )

                if q in pixels:

                    yield (
                        q,
                        1.41421356
                        if
                        dx != 0
                        and
                        dy != 0
                        else
                        1.0
                    )


    @classmethod
    def _endpoints(
        cls,
        skeleton
    ):

        coords = np.argwhere(
            skeleton
        )

        pixels = {
            (
                int(y),
                int(x)
            )
            for y, x in coords
        }

        endpoints = []

        for p in pixels:

            count = sum(
                1
                for _ in
                cls._neighbors(
                    p,
                    pixels
                )
            )

            if count == 1:

                endpoints.append(
                    p
                )

        return (
            pixels,
            endpoints
        )


    @classmethod
    def _geodesic_from_endpoint(
        cls,
        skeleton,
        start
    ):

        pixels, endpoints = (
            cls._endpoints(
                skeleton
            )
        )

        if (
            start not in pixels
            or
            len(pixels) < 2
        ):

            return None

        dist = {
            start: 0.0
        }

        parent = {}

        queue = [
            (
                0.0,
                start
            )
        ]

        while queue:

            d, p = heapq.heappop(
                queue
            )

            if d != dist.get(
                p,
                None
            ):
                continue

            for q, cost in cls._neighbors(
                p,
                pixels
            ):

                nd = d + cost

                if nd < dist.get(
                    q,
                    float("inf")
                ):

                    dist[q] = nd
                    parent[q] = p

                    heapq.heappush(
                        queue,
                        (
                            nd,
                            q
                        )
                    )

        candidates = [
            p
            for p in endpoints
            if p != start
            and p in dist
        ]

        if not candidates:

            candidates = [
                p
                for p in pixels
                if p in dist
            ]

        if not candidates:
            return None

        end = max(
            candidates,
            key=lambda p:
            dist[p]
        )

        path = [
            end
        ]

        p = end

        while p != start:

            if p not in parent:
                return None

            p = parent[p]

            path.append(
                p
            )

        path.reverse()

        # y,x -> x,y
        return np.asarray(
            [
                (
                    float(x),
                    float(y)
                )
                for y, x
                in path
            ],
            dtype=np.float64
        )


    # ========================================================
    # Path helpers
    # ========================================================

    @staticmethod
    def _path_length(
        path
    ):

        if (
            path is None
            or len(path) < 2
        ):
            return 0.0

        return float(
            np.sum(
                np.linalg.norm(
                    path[1:]
                    -
                    path[:-1],
                    axis=1
                )
            )
        )


    @staticmethod
    def _truncate_path(
        path,
        max_length
    ):

        if (
            path is None
            or len(path) < 2
        ):

            return path

        result = [
            path[0]
        ]

        length = 0.0

        for i in range(
            1,
            len(path)
        ):

            segment = float(
                np.linalg.norm(
                    path[i]
                    -
                    path[i - 1]
                )
            )

            if (
                length + segment
                >
                max_length
            ):
                break

            result.append(
                path[i]
            )

            length += segment

        return np.asarray(
            result,
            dtype=np.float64
        )


    @staticmethod
    def _smoothness(
        path
    ):

        if len(path) < 8:
            return 0.0

        step = max(
            2,
            len(path) // 20
        )

        pts = path[
            ::step
        ]

        if len(pts) < 4:
            return 0.5

        d = (
            pts[1:]
            -
            pts[:-1]
        )

        n = np.linalg.norm(
            d,
            axis=1
        )

        good = n > 1e-6

        d = d[good]
        n = n[good]

        if len(d) < 3:
            return 0.5

        d = (
            d
            /
            n[:, None]
        )

        dots = np.sum(
            d[1:]
            *
            d[:-1],
            axis=1
        )

        dots = np.clip(
            dots,
            -1.0,
            1.0
        )

        angles = np.arccos(
            dots
        )

        mean_angle = float(
            np.mean(
                angles
            )
        )

        return float(
            np.exp(
                -mean_angle
                /
                0.45
            )
        )


    def _choose_tip_endpoint(
        self,
        endpoints_xy
    ):

        if not endpoints_xy:
            return None

        reference = None

        if (
            self.last_valid_tip
            is not None
        ):

            reference = (
                self.last_valid_tip
            )

        elif (
            self.seed_tip
            is not None
        ):

            reference = (
                self.seed_tip
            )

        if reference is None:

            return endpoints_xy[0]

        return min(
            endpoints_xy,
            key=lambda p:
            np.linalg.norm(
                np.asarray(
                    p,
                    dtype=np.float64
                )
                -
                reference
            )
        )


    # ========================================================
    # Candidate extraction
    # ========================================================

    def _candidate_components(
        self,
        gray,
        response,
    ):

        positive = response[
            response > 0
        ]

        if positive.size == 0:
            return []

        threshold = max(
            0.16,
            float(
                np.percentile(
                    positive,
                    91.5
                )
            )
        )

        mask = (
            response
            >=
            threshold
        )

        mask = remove_small_objects(
            mask,
            min_size=20
        )

        # Tiny gap repair only.
        mask_u8 = (
            mask.astype(
                np.uint8
            )
            *
            255
        )

        mask_u8 = cv2.morphologyEx(
            mask_u8,
            cv2.MORPH_CLOSE,
            np.ones(
                (3, 3),
                dtype=np.uint8
            ),
            iterations=1
        )

        mask = (
            mask_u8 > 0
        )

        labels, count = (
            ndimage.label(
                mask
            )
        )

        candidates = []

        for label_id in range(
            1,
            count + 1
        ):

            component = (
                labels
                ==
                label_id
            )

            area = int(
                np.count_nonzero(
                    component
                )
            )

            if (
                area < 25
                or
                area > 6000
            ):
                continue

            skeleton = skeletonize(
                component
            )

            skel_length = int(
                np.count_nonzero(
                    skeleton
                )
            )

            if skel_length < 35:
                continue

            pixels, endpoints = (
                self._endpoints(
                    skeleton
                )
            )

            if len(endpoints) < 1:
                continue

            endpoints_xy = [
                (
                    float(x),
                    float(y)
                )
                for y, x in endpoints
            ]

            tip_xy = (
                self._choose_tip_endpoint(
                    endpoints_xy
                )
            )

            start_yx = (
                int(round(
                    tip_xy[1]
                )),
                int(round(
                    tip_xy[0]
                ))
            )

            path = (
                self._geodesic_from_endpoint(
                    skeleton,
                    start_yx
                )
            )

            if (
                path is None
                or len(path) < 20
            ):
                continue

            path_length = (
                self._path_length(
                    path
                )
            )

            if path_length < 45.0:
                continue

            local_path = (
                self._truncate_path(
                    path,
                    self.local_length_px
                )
            )

            if (
                local_path is None
                or
                len(local_path) < 15
            ):
                continue

            # Mean line-response on local path.
            xs = np.clip(
                np.rint(
                    local_path[:, 0]
                ).astype(int),
                0,
                response.shape[1] - 1
            )

            ys = np.clip(
                np.rint(
                    local_path[:, 1]
                ).astype(int),
                0,
                response.shape[0] - 1
            )

            ridge_score = float(
                np.mean(
                    response[
                        ys,
                        xs
                    ]
                )
            )

            # Approximate width.
            thickness = float(
                area
                /
                max(
                    1,
                    skel_length
                )
            )

            thin_score = float(
                np.exp(
                    -
                    (
                        max(
                            0.0,
                            thickness - 2.0
                        )
                        /
                        3.0
                    ) ** 2
                )
            )

            smooth_score = (
                self._smoothness(
                    local_path
                )
            )

            endpoint_score = (
                1.0
                if
                2 <= len(endpoints) <= 6
                else
                0.45
            )

            # Soft temporal association only.
            association_score = 0.5

            reference = (
                self.last_valid_tip
                if
                self.last_valid_tip
                is not None
                else
                self.seed_tip
            )

            tip_distance = None

            if reference is not None:

                tip_distance = float(
                    np.linalg.norm(
                        local_path[0]
                        -
                        reference
                    )
                )

                association_score = float(
                    np.exp(
                        -
                        (
                            tip_distance
                            /
                            90.0
                        ) ** 2
                    )
                )

            score = (
                0.34 * ridge_score
                +
                0.22 * thin_score
                +
                0.20 * smooth_score
                +
                0.10 * endpoint_score
                +
                0.14 * association_score
            )

            candidates.append(
                {
                    "score":
                        float(score),

                    "ridge_score":
                        ridge_score,

                    "thickness":
                        thickness,

                    "smooth_score":
                        smooth_score,

                    "tip_distance":
                        tip_distance,

                    "path":
                        local_path,
                }
            )

        candidates.sort(
            key=lambda c:
            c["score"],
            reverse=True
        )

        return candidates


    # ========================================================
    # Grasp geometry
    # ========================================================

    def _grasp_target(
        self,
        path
    ):

        if (
            path is None
            or len(path) < 2
        ):
            return None

        length = 0.0

        for i in range(
            1,
            len(path)
        ):

            segment = float(
                np.linalg.norm(
                    path[i]
                    -
                    path[i - 1]
                )
            )

            length += segment

            if (
                length
                >=
                self.grasp_offset_px
            ):

                return (
                    path[i].copy()
                )

        return None


    @staticmethod
    def _local_direction(
        path
    ):

        if (
            path is None
            or len(path) < 8
        ):
            return None

        index = min(
            len(path) - 1,
            12
        )

        d = (
            path[index]
            -
            path[0]
        )

        n = float(
            np.linalg.norm(
                d
            )
        )

        if n < 1e-6:
            return None

        return (
            d / n
        )


    # ========================================================
    # Main detection
    # ========================================================

    def detect(
        self,
        frame,
        roi
    ):

        x, y, w, h = [
            int(v)
            for v in roi
        ]

        H, W = frame.shape[:2]

        x = max(
            0,
            min(
                x,
                W - 1
            )
        )

        y = max(
            0,
            min(
                y,
                H - 1
            )
        )

        w = max(
            1,
            min(
                w,
                W - x
            )
        )

        h = max(
            1,
            min(
                h,
                H - y
            )
        )

        crop = frame[
            y:y+h,
            x:x+w
        ]

        gray, response = (
            self._wire_response(
                crop
            )
        )

        candidates = (
            self._candidate_components(
                gray,
                response
            )
        )

        if not candidates:

            self.confirm_count = 0

            return {
                "valid": False,
                "state": "NO_CANDIDATE",
                "candidate_count": 0,
                "centerline": [],
                "wire_tip_uv": None,
                "grasp_target_uv": None,
                "wire_direction_uv": None,
                "score": 0.0,
            }

        best = candidates[0]

        path = best[
            "path"
        ].copy()

        # ROI -> full image
        path[:, 0] += x
        path[:, 1] += y

        tip = (
            path[0].copy()
        )

        # Hard association gate:
        # previous state may reject an impossible jump,
        # but NEVER generates a new position itself.
        if (
            self.last_valid_tip
            is not None
        ):

            jump = float(
                np.linalg.norm(
                    tip
                    -
                    self.last_valid_tip
                )
            )

            if jump > 90.0:

                self.confirm_count = 0

                return {
                    "valid": False,
                    "state":
                        f"JUMP_REJECT:{jump:.1f}",
                    "candidate_count":
                        len(candidates),
                    "centerline": [],
                    "wire_tip_uv": None,
                    "grasp_target_uv": None,
                    "wire_direction_uv": None,
                    "score":
                        float(
                            best["score"]
                        ),
                }

        candidate_good = (
            best["score"] >= 0.46
            and
            best["ridge_score"] >= 0.18
            and
            best["thickness"] <= 7.0
        )

        if not candidate_good:

            self.confirm_count = 0

            return {
                "valid": False,
                "state": "QUALITY_REJECT",
                "candidate_count":
                    len(candidates),
                "centerline": [],
                "wire_tip_uv": None,
                "grasp_target_uv": None,
                "wire_direction_uv": None,
                "score":
                    float(
                        best["score"]
                    ),
            }

        # Multi-frame confirmation.
        if (
            self.confirm_candidate
            is None
        ):

            self.confirm_candidate = (
                tip.copy()
            )

            self.confirm_count = 1

        else:

            d = float(
                np.linalg.norm(
                    tip
                    -
                    self.confirm_candidate
                )
            )

            if d <= 22.0:

                self.confirm_candidate = (
                    0.5
                    *
                    self.confirm_candidate
                    +
                    0.5
                    *
                    tip
                )

                self.confirm_count += 1

            else:

                self.confirm_candidate = (
                    tip.copy()
                )

                self.confirm_count = 1

        confirmed = (
            self.confirm_count >= 3
        )

        if confirmed:

            self.last_valid_tip = (
                tip.copy()
            )

        target = (
            self._grasp_target(
                path
            )
        )

        direction = (
            self._local_direction(
                path
            )
        )

        return {
            "valid":
                bool(confirmed),

            "state":
                (
                    "TRACK"
                    if confirmed
                    else
                    f"CONFIRM:{self.confirm_count}/3"
                ),

            "candidate_count":
                int(
                    len(candidates)
                ),

            "centerline":
                (
                    np.rint(
                        path
                    )
                    .astype(int)
                    .tolist()
                    if confirmed
                    else
                    []
                ),

            "wire_tip_uv":
                (
                    np.rint(
                        tip
                    )
                    .astype(int)
                    .tolist()
                    if confirmed
                    else
                    None
                ),

            "grasp_target_uv":
                (
                    np.rint(
                        target
                    )
                    .astype(int)
                    .tolist()
                    if
                    confirmed
                    and
                    target is not None
                    else
                    None
                ),

            "wire_direction_uv":
                (
                    direction.astype(
                        float
                    ).tolist()
                    if
                    confirmed
                    and
                    direction is not None
                    else
                    None
                ),

            "score":
                float(
                    best["score"]
                ),

            "ridge_score":
                float(
                    best["ridge_score"]
                ),

            "thickness":
                float(
                    best["thickness"]
                ),

            "smooth_score":
                float(
                    best["smooth_score"]
                ),
        }
