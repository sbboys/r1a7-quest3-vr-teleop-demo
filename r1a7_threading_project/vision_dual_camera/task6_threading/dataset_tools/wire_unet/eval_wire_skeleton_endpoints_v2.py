#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
from collections import deque

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


NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
)


def skeletonize(mask):

    src = (
        mask.astype(np.uint8)
        * 255
    )

    if (
        hasattr(cv2, "ximgproc")
        and hasattr(
            cv2.ximgproc,
            "thinning",
        )
    ):
        skel = cv2.ximgproc.thinning(
            src,
            thinningType=
                cv2.ximgproc.THINNING_ZHANGSUEN,
        )

        return skel > 0

    # Fallback: morphological skeleton.
    img = mask.astype(np.uint8).copy()

    skel = np.zeros_like(img)

    element = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (3, 3),
    )

    while np.any(img):

        eroded = cv2.erode(
            img,
            element,
        )

        opened = cv2.dilate(
            eroded,
            element,
        )

        temp = cv2.subtract(
            img,
            opened,
        )

        skel = cv2.bitwise_or(
            skel,
            temp,
        )

        img = eroded

    return skel > 0


def neighbors_of(p, pixels):

    y, x = p

    out = []

    for dy, dx in NEIGHBORS:

        q = (
            y + dy,
            x + dx,
        )

        if q in pixels:
            out.append(q)

    return out


def build_graph(skel):

    ys, xs = np.where(skel)

    pixels = {
        (int(y), int(x))
        for y, x in zip(ys, xs)
    }

    graph = {}

    for p in pixels:
        graph[p] = neighbors_of(
            p,
            pixels,
        )

    return graph


def bfs_farthest(graph, start):

    q = deque([start])

    dist = {
        start: 0
    }

    parent = {
        start: None
    }

    farthest = start

    while q:

        p = q.popleft()

        if dist[p] > dist[farthest]:
            farthest = p

        for n in graph[p]:

            if n in dist:
                continue

            dist[n] = (
                dist[p] + 1
            )

            parent[n] = p

            q.append(n)

    return farthest, dist, parent


def longest_path(skel):
    """
    Approximate skeleton diameter:
      arbitrary node -> farthest A
      A -> farthest B
      reconstruct A-B path
    """

    graph = build_graph(skel)

    if not graph:
        return [], None, None

    start = next(iter(graph))

    a, _, _ = bfs_farthest(
        graph,
        start,
    )

    b, _, parent = bfs_farthest(
        graph,
        a,
    )

    path = []

    p = b

    while p is not None:

        path.append(p)

        if p == a:
            break

        p = parent.get(p)

    path.reverse()

    return path, a, b


def endpoint_error(
    pred_endpoints,
    gt_endpoints,
):
    """
    Two endpoints are unordered.
    Return best pairing mean/max distance.
    """

    if (
        pred_endpoints is None
        or gt_endpoints is None
    ):
        return None

    p1, p2 = pred_endpoints
    g1, g2 = gt_endpoints

    p1 = np.array(
        [p1[1], p1[0]],
        dtype=np.float32,
    )

    p2 = np.array(
        [p2[1], p2[0]],
        dtype=np.float32,
    )

    g1 = np.array(
        [g1[1], g1[0]],
        dtype=np.float32,
    )

    g2 = np.array(
        [g2[1], g2[0]],
        dtype=np.float32,
    )

    d11 = float(
        np.linalg.norm(p1 - g1)
    )

    d22 = float(
        np.linalg.norm(p2 - g2)
    )

    d12 = float(
        np.linalg.norm(p1 - g2)
    )

    d21 = float(
        np.linalg.norm(p2 - g1)
    )

    option1 = (
        d11 + d22,
        [d11, d22],
    )

    option2 = (
        d12 + d21,
        [d12, d21],
    )

    best = min(
        option1,
        option2,
        key=lambda x: x[0],
    )

    return {
        "d1_px": best[1][0],
        "d2_px": best[1][1],
        "mean_px":
            float(np.mean(best[1])),
        "max_px":
            float(np.max(best[1])),
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

    root = Path(args.dataset)

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
        / "skeleton_endpoint_eval"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    print("=" * 70)
    print(
        "R1-A7 WIRE SKELETON / ENDPOINT EVAL"
    )
    print("=" * 70)

    print(
        "threshold =",
        args.threshold,
    )

    print(
        "bridge    =",
        args.bridge,
    )

    print(
        "val masks =",
        len(split["val"]) * 3,
    )

    print(
        "test USED = 0"
    )

    print()

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

            valid = mask != 255

            pred_wire = (
                (pred == 1)
                & valid
            )

            gt_wire = (
                (mask == 1)
                & valid
            )

            pred_main = keep_main_cluster(
                pred_wire,
                args.bridge,
            )

            pred_skel = skeletonize(
                pred_main
            )

            gt_skel = skeletonize(
                gt_wire
            )

            pred_path, pa, pb = (
                longest_path(
                    pred_skel
                )
            )

            gt_path, ga, gb = (
                longest_path(
                    gt_skel
                )
            )

            error = endpoint_error(
                (
                    (pa, pb)
                    if pa is not None
                    and pb is not None
                    else None
                ),
                (
                    (ga, gb)
                    if ga is not None
                    and gb is not None
                    else None
                ),
            )

            rec = {
                "sample_id": sid,
                "camera": camera,
                "pred_path_length_px":
                    len(pred_path),
                "gt_path_length_px":
                    len(gt_path),
                "endpoint_error":
                    error,
            }

            records.append(rec)

            vis = image.copy()

            # Draw predicted main path in cyan.
            for y, x in pred_path:
                cv2.circle(
                    vis,
                    (x, y),
                    1,
                    (255, 255, 0),
                    -1,
                )

            # Pred endpoints: red.
            if pa is not None:
                cv2.circle(
                    vis,
                    (pa[1], pa[0]),
                    6,
                    (0, 0, 255),
                    2,
                )

            if pb is not None:
                cv2.circle(
                    vis,
                    (pb[1], pb[0]),
                    6,
                    (0, 0, 255),
                    2,
                )

            # GT endpoints: green.
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

            if error is None:
                print(
                    camera,
                    sid,
                    "NO ENDPOINT"
                )
            else:
                print(
                    camera,
                    sid,
                    f"mean={error['mean_px']:.2f}px",
                    f"max={error['max_px']:.2f}px",
                    f"path={len(pred_path)}"
                )

    valid_records = [
        x
        for x in records
        if x["endpoint_error"]
        is not None
    ]

    errors = np.array(
        [
            x[
                "endpoint_error"
            ]["mean_px"]
            for x in valid_records
        ],
        dtype=np.float32,
    )

    print()
    print("=" * 70)
    print(
        "ENDPOINT SUMMARY"
    )
    print("=" * 70)

    print(
        "valid =",
        len(valid_records),
        "/",
        len(records),
    )

    if len(errors):

        print(
            "mean_px =",
            float(
                np.mean(errors)
            )
        )

        print(
            "median_px =",
            float(
                np.median(errors)
            )
        )

        print(
            "p95_px =",
            float(
                np.percentile(
                    errors,
                    95,
                )
            )
        )

        print(
            "max_px =",
            float(
                np.max(errors)
            )
        )

    records.sort(
        key=lambda x:
            (
                x["endpoint_error"][
                    "mean_px"
                ]
                if x[
                    "endpoint_error"
                ]
                is not None
                else 999999
            ),
        reverse=True,
    )

    print()
    print(
        "===== WORST ENDPOINT CASES ====="
    )

    for x in records[:10]:

        print(
            x["camera"],
            x["sample_id"],
            x["endpoint_error"],
        )

    (
        out_dir
        / "endpoint_eval.json"
    ).write_text(
        json.dumps(
            {
                "threshold":
                    args.threshold,
                "bridge":
                    args.bridge,
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
