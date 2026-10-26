#!/usr/bin/env python3

import cv2
import heapq
import numpy as np

from skimage.filters import frangi

from wire_roi import make_wire_roi


class WireDetectorCore:

    def __init__(
        self,
        roi_margin=80,
        keypoint_step=20,
    ):
        self.roi_margin = int(roi_margin)
        self.keypoint_step = int(keypoint_step)


    # ========================================================
    # Image enhancement
    # ========================================================

    def _ridge_response(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8,8),
        )

        enhanced = clahe.apply(gray)

        ridge = frangi(
            enhanced,
            sigmas=(1,2,3,4),
            black_ridges=False,
        )

        ridge = cv2.normalize(
            ridge,
            None,
            0.0,
            1.0,
            cv2.NORM_MINMAX,
        ).astype(np.float32)

        return enhanced, ridge


    # ========================================================
    # Seeded Dijkstra
    # ========================================================

    def _shortest_path(
        self,
        ridge,
        start_uv,
        goal_uv,
    ):

        h, w = ridge.shape

        sx = int(np.clip(round(start_uv[0]), 0, w-1))
        sy = int(np.clip(round(start_uv[1]), 0, h-1))

        gx = int(np.clip(round(goal_uv[0]), 0, w-1))
        gy = int(np.clip(round(goal_uv[1]), 0, h-1))

        start = (sx,sy)
        goal = (gx,gy)

        # High ridge -> low cost.
        #
        # Keep a non-zero base cost so the path cannot
        # wander arbitrarily through background.
        cost_map = (
            0.15
            +
            0.85 * (1.0 - ridge)
        )

        pq = [(0.0, start)]

        dist = {
            start: 0.0
        }

        parent = {}

        visited = set()

        neighbours = (
            (-1,-1), (0,-1), (1,-1),
            (-1, 0),          (1, 0),
            (-1, 1), (0, 1), (1, 1),
        )

        while pq:

            cost, node = heapq.heappop(pq)

            if node in visited:
                continue

            visited.add(node)

            if node == goal:
                break

            x,y = node

            for dx,dy in neighbours:

                nx = x + dx
                ny = y + dy

                if not (
                    0 <= nx < w
                    and
                    0 <= ny < h
                ):
                    continue

                step = 1.41421356 if dx != 0 and dy != 0 else 1.0

                local_cost = (
                    float(cost_map[y,x])
                    +
                    float(cost_map[ny,nx])
                ) * 0.5

                new_cost = (
                    cost
                    +
                    step * local_cost
                )

                nxt = (nx,ny)

                if (
                    nxt not in dist
                    or
                    new_cost < dist[nxt]
                ):
                    dist[nxt] = new_cost
                    parent[nxt] = node

                    heapq.heappush(
                        pq,
                        (new_cost, nxt)
                    )

        if goal != start and goal not in parent:
            return []

        path = [goal]

        cur = goal

        while cur != start:

            cur = parent[cur]
            path.append(cur)

        path.reverse()

        return [
            [int(x), int(y)]
            for x,y in path
        ]


    # ========================================================
    # Main
    # ========================================================

    def detect(
        self,
        frame,
        wire_tip_seed_uv,
        grasp_uv,
    ):

        result = {
            "wire_valid": False,
            "wire_tip_uv": None,
            "wire_grasp_uv": None,
            "centerline": [],
            "keypoints": [],
            "confidence": 0.0,
            "roi": None,
        }

        if frame is None:
            return result

        if (
            wire_tip_seed_uv is None
            or
            grasp_uv is None
        ):
            return result

        roi = make_wire_roi(
            frame.shape,
            wire_tip_seed_uv,
            grasp_uv,
            margin=self.roi_margin,
        )

        x0,y0,x1,y1 = roi

        crop = frame[
            y0:y1,
            x0:x1
        ]

        if crop.size == 0:
            return result

        enhanced, ridge = self._ridge_response(
            crop
        )

        tip_local = (
            float(wire_tip_seed_uv[0]) - x0,
            float(wire_tip_seed_uv[1]) - y0,
        )

        grasp_local = (
            float(grasp_uv[0]) - x0,
            float(grasp_uv[1]) - y0,
        )

        path_local = self._shortest_path(
            ridge,
            tip_local,
            grasp_local,
        )

        if len(path_local) < 5:
            return result

        path_global = [
            [
                int(p[0] + x0),
                int(p[1] + y0),
            ]
            for p in path_local
        ]

        # ----------------------------------------------------
        # Confidence from mean ridge response along path
        # ----------------------------------------------------

        scores = []

        for x,y in path_local:

            if (
                0 <= x < ridge.shape[1]
                and
                0 <= y < ridge.shape[0]
            ):
                scores.append(
                    float(ridge[y,x])
                )

        confidence = (
            float(np.mean(scores))
            if scores
            else 0.0
        )

        keypoints = (
            path_global[
                ::max(1,self.keypoint_step)
            ]
        )

        if keypoints[-1] != path_global[-1]:
            keypoints.append(
                path_global[-1]
            )

        result.update(
            {
                "wire_valid":
                    True,

                "wire_tip_uv":
                    path_global[0],

                "wire_grasp_uv":
                    path_global[-1],

                "centerline":
                    path_global,

                "keypoints":
                    keypoints,

                "confidence":
                    confidence,

                "roi":
                    [x0,y0,x1,y1],

                "path_length_px":
                    len(path_global),
            }
        )

        return result


def draw_wire_result(
    image,
    result,
):

    out = image.copy()

    if not result.get(
        "wire_valid",
        False,
    ):
        return out

    centerline = result.get(
        "centerline",
        [],
    )

    for i in range(
        1,
        len(centerline),
    ):
        p0 = tuple(
            map(int, centerline[i-1])
        )

        p1 = tuple(
            map(int, centerline[i])
        )

        cv2.line(
            out,
            p0,
            p1,
            (0,255,0),
            2,
        )

    for p in result.get(
        "keypoints",
        [],
    ):
        cv2.circle(
            out,
            tuple(map(int,p)),
            3,
            (0,255,255),
            -1,
        )

    tip = result.get(
        "wire_tip_uv"
    )

    grasp = result.get(
        "wire_grasp_uv"
    )

    if tip is not None:
        cv2.circle(
            out,
            tuple(map(int,tip)),
            7,
            (0,255,255),
            -1,
        )

    if grasp is not None:
        cv2.circle(
            out,
            tuple(map(int,grasp)),
            7,
            (255,0,0),
            -1,
        )

    roi = result.get(
        "roi"
    )

    if roi is not None:

        x0,y0,x1,y1 = map(
            int,
            roi
        )

        cv2.rectangle(
            out,
            (x0,y0),
            (x1,y1),
            (255,255,0),
            1,
        )

    text = (
        f"WIRE conf="
        f"{result.get('confidence',0.0):.2f} "
        f"pts="
        f"{len(centerline)}"
    )

    cv2.putText(
        out,
        text,
        (10,25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,255),
        2,
        cv2.LINE_AA,
    )

    return out
