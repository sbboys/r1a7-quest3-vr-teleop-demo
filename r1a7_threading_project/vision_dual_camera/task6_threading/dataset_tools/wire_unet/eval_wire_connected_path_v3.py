#!/usr/bin/env python3

from pathlib import Path
import argparse
import heapq
import json
import math

import cv2
import numpy as np
import torch

from overfit_wire_unet import UNet
from train_wire_unet_v1 import CAMERAS
from eval_wire_threshold_sweep import (
    tiled_logits,
    threshold_prediction,
)
from eval_wire_main_cluster_v2 import (
    keep_main_cluster,
)
from eval_wire_skeleton_endpoints_v2 import (
    skeletonize,
    endpoint_error,
)


NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
)


def nearest_pair(a, b, chunk=256):

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    best_d2 = float("inf")
    best_a = None
    best_b = None

    for start in range(0, len(a), chunk):

        aa = a[start:start + chunk]

        diff = (
            aa[:, None, :]
            - b[None, :, :]
        )

        d2 = np.sum(
            diff * diff,
            axis=2,
        )

        idx = int(np.argmin(d2))

        i, j = np.unravel_index(
            idx,
            d2.shape,
        )

        value = float(
            d2[i, j]
        )

        if value < best_d2:

            best_d2 = value

            best_a = tuple(
                int(x)
                for x in aa[i]
            )

            best_b = tuple(
                int(x)
                for x in b[j]
            )

    return (
        math.sqrt(best_d2),
        best_a,
        best_b,
    )


class DSU:

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):

        while self.parent[x] != x:

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x

    def union(self, a, b):

        a = self.find(a)
        b = self.find(b)

        if a == b:
            return False

        self.parent[b] = a
        return True


def connect_skeleton_components(
    skel,
    max_gap,
):

    n, labels = cv2.connectedComponents(
        skel.astype(np.uint8),
        connectivity=8,
    )

    components = []

    for label in range(1, n):

        coords = np.argwhere(
            labels == label
        )

        if len(coords):
            components.append(coords)

    if len(components) <= 1:

        return (
            skel.copy(),
            [],
            True,
            len(components),
        )

    edges = []

    for i in range(
        len(components)
    ):

        for j in range(
            i + 1,
            len(components),
        ):

            d, p, q = nearest_pair(
                components[i],
                components[j],
            )

            edges.append(
                (
                    d,
                    i,
                    j,
                    p,
                    q,
                )
            )

    edges.sort(
        key=lambda x: x[0]
    )

    dsu = DSU(
        len(components)
    )

    connected = (
        skel.astype(np.uint8)
        .copy()
    )

    bridges = []

    for d, i, j, p, q in edges:

        if d > max_gap:
            continue

        if not dsu.union(i, j):
            continue

        # p/q = (y, x)
        cv2.line(
            connected,
            (p[1], p[0]),
            (q[1], q[0]),
            1,
            1,
            cv2.LINE_8,
        )

        bridges.append(
            {
                "distance_px": d,
                "p_yx": [
                    p[0],
                    p[1],
                ],
                "q_yx": [
                    q[0],
                    q[1],
                ],
            }
        )

    roots = {
        dsu.find(i)
        for i in range(
            len(components)
        )
    }

    success = (
        len(roots) == 1
    )

    return (
        connected > 0,
        bridges,
        success,
        len(components),
    )


def build_graph(skel):

    ys, xs = np.where(skel)

    pixels = {
        (int(y), int(x))
        for y, x in zip(
            ys,
            xs,
        )
    }

    graph = {}

    for y, x in pixels:

        p = (y, x)

        neighbors = []

        for dy, dx in NEIGHBORS:

            q = (
                y + dy,
                x + dx,
            )

            if q not in pixels:
                continue

            weight = (
                math.sqrt(2.0)
                if dy != 0
                and dx != 0
                else 1.0
            )

            neighbors.append(
                (q, weight)
            )

        graph[p] = neighbors

    return graph


def graph_components(graph):

    remaining = set(
        graph.keys()
    )

    components = []

    while remaining:

        start = next(
            iter(remaining)
        )

        stack = [start]
        comp = set()

        while stack:

            p = stack.pop()

            if p in comp:
                continue

            comp.add(p)

            for q, _ in graph[p]:

                if q not in comp:
                    stack.append(q)

        remaining -= comp
        components.append(comp)

    return components


def dijkstra(
    graph,
    start,
    allowed,
):

    dist = {
        start: 0.0
    }

    parent = {
        start: None
    }

    heap = [
        (0.0, start)
    ]

    while heap:

        d, p = heapq.heappop(
            heap
        )

        if d != dist[p]:
            continue

        for q, w in graph[p]:

            if q not in allowed:
                continue

            nd = d + w

            if (
                q not in dist
                or nd < dist[q]
            ):

                dist[q] = nd
                parent[q] = p

                heapq.heappush(
                    heap,
                    (nd, q),
                )

    farthest = max(
        dist,
        key=dist.get,
    )

    return (
        farthest,
        dist,
        parent,
    )


def longest_geodesic_path(skel):

    graph = build_graph(
        skel
    )

    if not graph:
        return [], None, None, 0.0

    best_path = []
    best_a = None
    best_b = None
    best_length = -1.0

    for component in graph_components(
        graph
    ):

        start = next(
            iter(component)
        )

        a, _, _ = dijkstra(
            graph,
            start,
            component,
        )

        b, dist, parent = dijkstra(
            graph,
            a,
            component,
        )

        length = dist[b]

        path = []

        p = b

        while p is not None:

            path.append(p)

            if p == a:
                break

            p = parent[p]

        path.reverse()

        if length > best_length:

            best_length = length
            best_path = path
            best_a = a
            best_b = b

    return (
        best_path,
        best_a,
        best_b,
        best_length,
    )


def path_to_gt_stats(
    path,
    gt_wire,
):

    if (
        not path
        or not gt_wire.any()
    ):
        return None

    dist_to_gt = cv2.distanceTransform(
        (~gt_wire).astype(
            np.uint8
        ),
        cv2.DIST_L2,
        5,
    )

    values = np.array(
        [
            dist_to_gt[y, x]
            for y, x in path
        ],
        dtype=np.float32,
    )

    return {
        "mean_px": float(
            np.mean(values)
        ),
        "median_px": float(
            np.median(values)
        ),
        "p95_px": float(
            np.percentile(
                values,
                95,
            )
        ),
        "max_px": float(
            np.max(values)
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.86,
    )

    parser.add_argument(
        "--bridge",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--max-gap",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--tile",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=384,
    )

    args = parser.parse_args()

    root = Path(
        args.dataset
    )

    split = json.loads(
        (
            root
            / "splits_sample_level_seed42.json"
        ).read_text()
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = UNet(
        in_channels=3,
        num_classes=3,
        base=32,
    ).to(device)

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    out_dir = (
        root
        / "runs"
        / "unet_train_val_v2_resume"
        / "connected_path_v3"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "R1-A7 CONNECTED WIRE PATH V3"
    )
    print("=" * 72)

    print(
        "threshold =",
        args.threshold,
    )

    print(
        "cluster bridge =",
        args.bridge,
    )

    print(
        "component max gap =",
        args.max_gap,
    )

    print(
        "val masks =",
        len(split["val"]) * 3,
    )

    print(
        "test USED = 0"
    )

    print()

    records = []

    for sid in split["val"]:

        for camera in CAMERAS:

            image = cv2.imread(
                str(
                    root
                    / "images"
                    / camera
                    / f"{sid}.png"
                ),
                cv2.IMREAD_COLOR,
            )

            mask = cv2.imread(
                str(
                    root
                    / "masks"
                    / camera
                    / f"{sid}.png"
                ),
                cv2.IMREAD_UNCHANGED,
            )

            if (
                image is None
                or mask is None
            ):
                raise RuntimeError(
                    f"read failure "
                    f"{camera} {sid}"
                )

            logits = tiled_logits(
                model,
                image,
                device,
                tile=args.tile,
                stride=args.stride,
            )

            pred = threshold_prediction(
                logits,
                args.threshold,
            )

            valid = (
                mask != 255
            )

            gt_wire = (
                (mask == 1)
                & valid
            )

            pred_wire = (
                (pred == 1)
                & valid
            )

            pred_main = keep_main_cluster(
                pred_wire,
                args.bridge,
            )

            pred_skel = skeletonize(
                pred_main
            )

            (
                pred_connected,
                pred_bridges,
                pred_connect_ok,
                pred_components,
            ) = connect_skeleton_components(
                pred_skel,
                args.max_gap,
            )

            (
                pred_path,
                pa,
                pb,
                pred_path_length,
            ) = longest_geodesic_path(
                pred_connected
            )

            # GT mask can itself contain interruptions because
            # hidden wire was intentionally not annotated.
            # Therefore this endpoint is only a mask-derived
            # visible-span proxy, not physical endpoint ground truth.
            gt_skel = skeletonize(
                gt_wire
            )

            (
                gt_connected,
                gt_bridges,
                gt_connect_ok,
                gt_components,
            ) = connect_skeleton_components(
                gt_skel,
                args.max_gap,
            )

            (
                gt_path,
                ga,
                gb,
                gt_path_length,
            ) = longest_geodesic_path(
                gt_connected
            )

            ep_error = None

            if (
                pred_connect_ok
                and gt_connect_ok
                and pa is not None
                and pb is not None
                and ga is not None
                and gb is not None
            ):

                ep_error = endpoint_error(
                    (pa, pb),
                    (ga, gb),
                )

            path_stats = path_to_gt_stats(
                pred_path,
                gt_wire,
            )

            max_pred_bridge = (
                max(
                    [
                        x["distance_px"]
                        for x
                        in pred_bridges
                    ],
                    default=0.0,
                )
            )

            rec = {
                "sample_id":
                    sid,
                "camera":
                    camera,

                "pred_components_before":
                    pred_components,

                "pred_bridge_count":
                    len(pred_bridges),

                "pred_connect_ok":
                    pred_connect_ok,

                "max_pred_bridge_px":
                    max_pred_bridge,

                "pred_path_length_px":
                    pred_path_length,

                "gt_components_before":
                    gt_components,

                "gt_bridge_count":
                    len(gt_bridges),

                "gt_connect_ok":
                    gt_connect_ok,

                "gt_path_length_px":
                    gt_path_length,

                "path_to_gt":
                    path_stats,

                "endpoint_proxy_error":
                    ep_error,
            }

            records.append(
                rec
            )

            vis = image.copy()

            # Connected main path: cyan
            for y, x in pred_path:

                cv2.circle(
                    vis,
                    (x, y),
                    1,
                    (255, 255, 0),
                    -1,
                )

            # Virtual bridges: magenta
            for b in pred_bridges:

                p = b["p_yx"]
                q = b["q_yx"]

                cv2.line(
                    vis,
                    (p[1], p[0]),
                    (q[1], q[0]),
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Pred endpoints: red
            if pa is not None:

                cv2.circle(
                    vis,
                    (pa[1], pa[0]),
                    7,
                    (0, 0, 255),
                    2,
                )

            if pb is not None:

                cv2.circle(
                    vis,
                    (pb[1], pb[0]),
                    7,
                    (0, 0, 255),
                    2,
                )

            # GT proxy endpoints: green
            if ga is not None:

                cv2.circle(
                    vis,
                    (ga[1], ga[0]),
                    6,
                    (0, 255, 0),
                    2,
                )

            if gb is not None:

                cv2.circle(
                    vis,
                    (gb[1], gb[0]),
                    6,
                    (0, 255, 0),
                    2,
                )

            cv2.imwrite(
                str(
                    out_dir
                    / f"{camera}_{sid}.jpg"
                ),
                vis,
            )

            p95 = (
                path_stats["p95_px"]
                if path_stats
                is not None
                else float("nan")
            )

            ep = (
                ep_error["mean_px"]
                if ep_error
                is not None
                else float("nan")
            )

            print(
                camera,
                sid,
                f"comp={pred_components}",
                f"bridges={len(pred_bridges)}",
                f"connect={pred_connect_ok}",
                f"path={pred_path_length:.1f}px",
                f"pathP95={p95:.2f}px",
                f"EPproxy={ep:.2f}px",
            )

    valid_path = [
        x
        for x in records
        if x["path_to_gt"]
        is not None
    ]

    p95_values = np.array(
        [
            x["path_to_gt"]["p95_px"]
            for x in valid_path
        ],
        dtype=np.float32,
    )

    valid_ep = [
        x
        for x in records
        if x["endpoint_proxy_error"]
        is not None
    ]

    ep_values = np.array(
        [
            x[
                "endpoint_proxy_error"
            ]["mean_px"]
            for x in valid_ep
        ],
        dtype=np.float32,
    )

    print()
    print("=" * 72)
    print(
        "CONNECTED PATH SUMMARY"
    )
    print("=" * 72)

    print(
        "prediction connected =",
        sum(
            x["pred_connect_ok"]
            for x in records
        ),
        "/",
        len(records),
    )

    if len(p95_values):

        print(
            "path->GT P95 metric:"
        )

        print(
            "  median =",
            float(
                np.median(
                    p95_values
                )
            ),
        )

        print(
            "  p95    =",
            float(
                np.percentile(
                    p95_values,
                    95,
                )
            ),
        )

        print(
            "  max    =",
            float(
                np.max(
                    p95_values
                )
            ),
        )

    print()
    print(
        "endpoint proxy valid =",
        len(valid_ep),
        "/",
        len(records),
    )

    if len(ep_values):

        print(
            "endpoint proxy mean error:"
        )

        print(
            "  mean   =",
            float(
                np.mean(
                    ep_values
                )
            ),
        )

        print(
            "  median =",
            float(
                np.median(
                    ep_values
                )
            ),
        )

        print(
            "  p95    =",
            float(
                np.percentile(
                    ep_values,
                    95,
                )
            ),
        )

        print(
            "  max    =",
            float(
                np.max(
                    ep_values
                )
            ),
        )

    records.sort(
        key=lambda x:
            (
                x["path_to_gt"]["p95_px"]
                if x["path_to_gt"]
                is not None
                else 999999
            ),
        reverse=True,
    )

    print()
    print(
        "===== WORST PATH->GT CASES ====="
    )

    for x in records[:10]:

        print(
            x["camera"],
            x["sample_id"],
            "components=",
            x[
                "pred_components_before"
            ],
            "path_to_gt=",
            x["path_to_gt"],
            "endpoint_proxy=",
            x[
                "endpoint_proxy_error"
            ],
        )

    out = (
        out_dir
        / "connected_path_eval.json"
    )

    out.write_text(
        json.dumps(
            {
                "threshold":
                    args.threshold,
                "cluster_bridge":
                    args.bridge,
                "component_max_gap":
                    args.max_gap,
                "records":
                    records,
            },
            indent=2,
        )
    )

    print()
    print(
        "OUTPUT =",
        out_dir,
    )


if __name__ == "__main__":
    main()
