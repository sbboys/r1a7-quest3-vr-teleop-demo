#!/usr/bin/env python3

from pathlib import Path
import argparse
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
)
from eval_wire_connected_path_v3 import (
    connect_skeleton_components,
    longest_geodesic_path,
)


def point_xy(p):
    # path points are (y, x)
    return np.array(
        [float(p[1]), float(p[0])],
        dtype=np.float32,
    )


def endpoint_tangent(path, from_start, span_px):
    """
    Return unit vector pointing FROM endpoint INTO the wire.

    span_px controls how much local centerline is used.
    """

    if len(path) < 2:
        return None

    pts = (
        path
        if from_start
        else list(reversed(path))
    )

    p0 = point_xy(
        pts[0]
    )

    chosen = None
    accumulated = 0.0
    prev = p0

    for p in pts[1:]:

        cur = point_xy(p)

        accumulated += float(
            np.linalg.norm(
                cur - prev
            )
        )

        prev = cur
        chosen = cur

        if accumulated >= span_px:
            break

    if chosen is None:
        return None

    v = chosen - p0

    n = float(
        np.linalg.norm(v)
    )

    if n < 1e-6:
        return None

    return v / n


def angle_deg(a, b):

    if a is None or b is None:
        return None

    dot = float(
        np.clip(
            np.dot(a, b),
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(dot)
        )
    )


def endpoint_pairing(
    pred_path,
    gt_path,
):
    """
    Returns mapping:
      pred endpoint 0 -> GT endpoint ?
      pred endpoint 1 -> GT endpoint ?
    """

    if (
        len(pred_path) < 2
        or len(gt_path) < 2
    ):
        return None

    pred_ep = [
        point_xy(pred_path[0]),
        point_xy(pred_path[-1]),
    ]

    gt_ep = [
        point_xy(gt_path[0]),
        point_xy(gt_path[-1]),
    ]

    d00 = float(
        np.linalg.norm(
            pred_ep[0] - gt_ep[0]
        )
    )

    d11 = float(
        np.linalg.norm(
            pred_ep[1] - gt_ep[1]
        )
    )

    d01 = float(
        np.linalg.norm(
            pred_ep[0] - gt_ep[1]
        )
    )

    d10 = float(
        np.linalg.norm(
            pred_ep[1] - gt_ep[0]
        )
    )

    if d00 + d11 <= d01 + d10:
        mapping = [0, 1]
        errors = [d00, d11]
    else:
        mapping = [1, 0]
        errors = [d01, d10]

    return {
        "mapping": mapping,
        "endpoint_errors_px": errors,
    }


def distance_to_mask(
    endpoint,
    mask,
):

    if (
        endpoint is None
        or not mask.any()
    ):
        return None

    dist = cv2.distanceTransform(
        (~mask).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )

    y, x = endpoint

    return float(
        dist[y, x]
    )


def tool_side_from_path(
    path,
    tool_mask,
):

    if (
        len(path) < 2
        or not tool_mask.any()
    ):
        return None

    endpoints = [
        path[0],
        path[-1],
    ]

    distances = [
        distance_to_mask(
            endpoints[0],
            tool_mask,
        ),
        distance_to_mask(
            endpoints[1],
            tool_mask,
        ),
    ]

    if (
        distances[0] is None
        or distances[1] is None
    ):
        return None

    tool_side = int(
        np.argmin(distances)
    )

    free_side = 1 - tool_side

    return {
        "tool_side_index":
            tool_side,

        "free_side_index":
            free_side,

        "distances_to_tool_px":
            distances,

        "distance_separation_px":
            float(
                abs(
                    distances[0]
                    - distances[1]
                )
            ),
    }


def path_support_fraction(
    path,
    original_wire,
    radius=2,
):

    if not path:
        return 0.0

    k = 2 * radius + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k),
    )

    supported = cv2.dilate(
        original_wire.astype(
            np.uint8
        ),
        kernel,
    ).astype(bool)

    count = 0

    for y, x in path:

        if supported[y, x]:
            count += 1

    return float(
        count / len(path)
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
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
        "--tangent-span",
        type=float,
        default=20.0,
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
        / f"endpoint_tangent_v4_{args.split}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    tangent_errors = []
    endpoint_errors = []

    semantic_valid = 0
    semantic_correct = 0

    print("=" * 72)
    print(
        "R1-A7 WIRE ENDPOINT / TANGENT V4"
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
        "tangent span =",
        args.tangent_span,
    )

    print(
        "split =",
        args.split,
    )

    print(
        "split masks =",
        len(split[args.split]) * 3,
    )

    print()

    for sid in split[args.split]:

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

            pred_tool = (
                (pred == 2)
                & valid
            )

            gt_wire = (
                (mask == 1)
                & valid
            )

            gt_tool = (
                (mask == 2)
                & valid
            )

            # --------------------------------------------------
            # Prediction geometry
            # --------------------------------------------------

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

            # --------------------------------------------------
            # GT visible-path proxy
            # --------------------------------------------------

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

            pairing = endpoint_pairing(
                pred_path,
                gt_path,
            )

            pred_tangents = [
                endpoint_tangent(
                    pred_path,
                    True,
                    args.tangent_span,
                ),
                endpoint_tangent(
                    pred_path,
                    False,
                    args.tangent_span,
                ),
            ]

            gt_tangents = [
                endpoint_tangent(
                    gt_path,
                    True,
                    args.tangent_span,
                ),
                endpoint_tangent(
                    gt_path,
                    False,
                    args.tangent_span,
                ),
            ]

            sample_tangent_errors = [
                None,
                None,
            ]

            if pairing is not None:

                for pred_idx in (0, 1):

                    gt_idx = pairing[
                        "mapping"
                    ][pred_idx]

                    err = angle_deg(
                        pred_tangents[
                            pred_idx
                        ],
                        gt_tangents[
                            gt_idx
                        ],
                    )

                    sample_tangent_errors[
                        pred_idx
                    ] = err

                    if err is not None:
                        tangent_errors.append(
                            err
                        )

                endpoint_errors.extend(
                    pairing[
                        "endpoint_errors_px"
                    ]
                )

            # --------------------------------------------------
            # Tool-side / free-side endpoint candidate
            # --------------------------------------------------

            pred_semantic = (
                tool_side_from_path(
                    pred_path,
                    pred_tool,
                )
            )

            gt_semantic = (
                tool_side_from_path(
                    gt_path,
                    gt_tool,
                )
            )

            semantic_match = None

            if (
                pairing is not None
                and pred_semantic
                is not None
                and gt_semantic
                is not None
            ):

                semantic_valid += 1

                pred_tool_idx = (
                    pred_semantic[
                        "tool_side_index"
                    ]
                )

                mapped_gt_idx = (
                    pairing[
                        "mapping"
                    ][pred_tool_idx]
                )

                semantic_match = (
                    mapped_gt_idx
                    ==
                    gt_semantic[
                        "tool_side_index"
                    ]
                )

                if semantic_match:
                    semantic_correct += 1

            support = (
                path_support_fraction(
                    pred_path,
                    pred_main,
                    radius=2,
                )
            )

            max_bridge = max(
                [
                    x["distance_px"]
                    for x
                    in pred_bridges
                ],
                default=0.0,
            )

            rec = {
                "sample_id":
                    sid,

                "camera":
                    camera,

                "pred_connect_ok":
                    pred_connect_ok,

                "pred_components":
                    pred_components,

                "pred_bridge_count":
                    len(pred_bridges),

                "max_bridge_px":
                    float(max_bridge),

                "path_length_px":
                    float(
                        pred_path_length
                    ),

                "path_support_fraction":
                    support,

                "pred_endpoints_yx":
                    (
                        [
                            list(pred_path[0]),
                            list(pred_path[-1]),
                        ]
                        if len(pred_path) >= 2
                        else None
                    ),

                "pred_tangents_xy":
                    [
                        (
                            x.tolist()
                            if x is not None
                            else None
                        )
                        for x
                        in pred_tangents
                    ],

                "endpoint_pairing":
                    pairing,

                "tangent_errors_deg":
                    sample_tangent_errors,

                "pred_tool_semantic":
                    pred_semantic,

                "gt_tool_semantic":
                    gt_semantic,

                "tool_side_match":
                    semantic_match,
            }

            records.append(
                rec
            )

            # --------------------------------------------------
            # Visual QA
            # --------------------------------------------------

            vis = image.copy()

            for y, x in pred_path:

                cv2.circle(
                    vis,
                    (x, y),
                    1,
                    (255, 255, 0),
                    -1,
                )

            endpoints = (
                [
                    pred_path[0],
                    pred_path[-1],
                ]
                if len(pred_path) >= 2
                else []
            )

            for idx, p in enumerate(
                endpoints
            ):

                y, x = p

                cv2.circle(
                    vis,
                    (x, y),
                    7,
                    (0, 0, 255),
                    2,
                )

                tangent = (
                    pred_tangents[idx]
                )

                if tangent is not None:

                    end_xy = (
                        np.array(
                            [x, y],
                            dtype=np.float32,
                        )
                        + tangent
                        * 35.0
                    )

                    cv2.arrowedLine(
                        vis,
                        (x, y),
                        (
                            int(
                                round(
                                    end_xy[0]
                                )
                            ),
                            int(
                                round(
                                    end_xy[1]
                                )
                            ),
                        ),
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                        tipLength=0.25,
                    )

            if pred_semantic is not None:

                ti = pred_semantic[
                    "tool_side_index"
                ]

                fi = pred_semantic[
                    "free_side_index"
                ]

                ty, tx = endpoints[ti]
                fy, fx = endpoints[fi]

                cv2.putText(
                    vis,
                    "TOOL",
                    (tx + 8, ty - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    vis,
                    "FREE",
                    (fx + 8, fy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imwrite(
                str(
                    out_dir
                    / f"{camera}_{sid}.jpg"
                ),
                vis,
            )

            tangent_text = (
                ",".join(
                    "NA"
                    if x is None
                    else f"{x:.1f}"
                    for x
                    in sample_tangent_errors
                )
            )

            tool_text = (
                "NA"
                if semantic_match is None
                else str(
                    semantic_match
                )
            )

            print(
                camera,
                sid,
                f"path={pred_path_length:.1f}px",
                f"support={support:.3f}",
                f"maxBridge={max_bridge:.1f}px",
                f"tanErr=[{tangent_text}]deg",
                f"toolSide={tool_text}",
            )

    tangent_arr = np.asarray(
        tangent_errors,
        dtype=np.float32,
    )

    endpoint_arr = np.asarray(
        endpoint_errors,
        dtype=np.float32,
    )

    print()
    print("=" * 72)
    print(
        "ENDPOINT / TANGENT SUMMARY"
    )
    print("=" * 72)

    print(
        "endpoint count =",
        len(endpoint_arr),
    )

    if len(endpoint_arr):

        print(
            "endpoint error:"
        )

        print(
            "  median =",
            float(
                np.median(
                    endpoint_arr
                )
            ),
            "px",
        )

        print(
            "  p95    =",
            float(
                np.percentile(
                    endpoint_arr,
                    95,
                )
            ),
            "px",
        )

        print(
            "  max    =",
            float(
                np.max(
                    endpoint_arr
                )
            ),
            "px",
        )

    print()

    print(
        "tangent count =",
        len(tangent_arr),
    )

    if len(tangent_arr):

        print(
            "tangent angle error:"
        )

        print(
            "  median =",
            float(
                np.median(
                    tangent_arr
                )
            ),
            "deg",
        )

        print(
            "  p95    =",
            float(
                np.percentile(
                    tangent_arr,
                    95,
                )
            ),
            "deg",
        )

        print(
            "  max    =",
            float(
                np.max(
                    tangent_arr
                )
            ),
            "deg",
        )

    print()

    print(
        "tool-side semantic valid =",
        semantic_valid,
    )

    if semantic_valid:

        print(
            "tool-side semantic correct =",
            semantic_correct,
            "/",
            semantic_valid,
        )

        print(
            "tool-side accuracy =",
            float(
                semantic_correct
                / semantic_valid
            ),
        )

    # Per-camera tangent summary
    print()
    print(
        "===== PER CAMERA TANGENT ====="
    )

    for camera in CAMERAS:

        values = []

        for rec in records:

            if rec["camera"] != camera:
                continue

            for x in rec[
                "tangent_errors_deg"
            ]:

                if x is not None:
                    values.append(x)

        arr = np.asarray(
            values,
            dtype=np.float32,
        )

        if len(arr):

            print(
                camera,
                f"median={np.median(arr):.2f}deg",
                f"p95={np.percentile(arr,95):.2f}deg",
                f"max={np.max(arr):.2f}deg",
            )

    out = (
        out_dir
        / "endpoint_tangent_eval.json"
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

                "tangent_span_px":
                    args.tangent_span,

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
