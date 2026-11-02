#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import torch

from overfit_wire_unet import UNet
from train_wire_unet_v1 import CAMERAS
from eval_wire_threshold_sweep import (
    tiled_logits,
    threshold_prediction,
)


AREA_THRESHOLDS = (
    1, 2, 3, 5, 8, 12, 20, 30, 50
)


def dilate_bool(mask, radius):
    if radius <= 0:
        return mask.copy()

    k = 2 * radius + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k),
    )

    return cv2.dilate(
        mask.astype(np.uint8),
        kernel,
    ).astype(bool)


def tolerance_metrics(gt, pred, radius):

    gt_d = dilate_bool(gt, radius)
    pred_d = dilate_bool(pred, radius)

    pred_n = int(pred.sum())
    gt_n = int(gt.sum())

    matched_pred = int(
        (pred & gt_d).sum()
    )

    matched_gt = int(
        (gt & pred_d).sum()
    )

    precision = (
        matched_pred / pred_n
        if pred_n > 0
        else 0.0
    )

    recall = (
        matched_gt / gt_n
        if gt_n > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1


def keep_components_by_area(mask, min_area):

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
    )

    out = np.zeros_like(
        mask,
        dtype=bool,
    )

    for label in range(1, n):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        if area >= min_area:
            out[labels == label] = True

    return out


def component_records(
    pred_wire,
    gt_wire,
    sid,
    camera,
):

    n, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            pred_wire.astype(np.uint8),
            connectivity=8,
        )
    )

    if gt_wire.any():

        dist_to_gt = cv2.distanceTransform(
            (~gt_wire).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

    else:
        dist_to_gt = np.full(
            gt_wire.shape,
            9999.0,
            dtype=np.float32,
        )

    records = []

    for label in range(1, n):

        comp = labels == label

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP
            ]
        )

        w = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH
            ]
        )

        h = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT
            ]
        )

        distances = (
            dist_to_gt[comp]
            .astype(np.float32)
        )

        near2 = float(
            np.mean(
                distances <= 2.0
            )
        )

        rec = {
            "sample_id": sid,
            "camera": camera,
            "component": label,
            "area": area,
            "bbox": [x, y, w, h],
            "centroid": [
                float(
                    centroids[label][0]
                ),
                float(
                    centroids[label][1]
                ),
            ],
            "distance_min_px":
                float(
                    distances.min()
                ),
            "distance_median_px":
                float(
                    np.median(
                        distances
                    )
                ),
            "distance_max_px":
                float(
                    distances.max()
                ),
            "near_gt_2px_fraction":
                near2,
        }

        records.append(rec)

    return records


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
        / "component_analysis_t086"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_components = []

    cache = []

    print("=" * 68)
    print(
        "R1-A7 WIRE COMPONENT ANALYSIS"
    )
    print("=" * 68)

    print(
        "threshold =",
        args.threshold,
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
                    f"read failed "
                    f"{sid} {camera}"
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

            gt_wire = (
                (mask == 1)
                & valid
            )

            pred_wire = (
                (pred == 1)
                & valid
            )

            comps = component_records(
                pred_wire,
                gt_wire,
                sid,
                camera,
            )

            all_components.extend(
                comps
            )

            cache.append(
                {
                    "sample_id": sid,
                    "camera": camera,
                    "gt": gt_wire,
                    "pred": pred_wire,
                }
            )

            # Diagnostic visualization:
            # red boxes = components entirely >3 px from GT.
            vis = image.copy()

            for rec in comps:

                if (
                    rec[
                        "distance_min_px"
                    ] > 3.0
                ):

                    x, y, w, h = (
                        rec["bbox"]
                    )

                    cv2.rectangle(
                        vis,
                        (x, y),
                        (
                            x + w - 1,
                            y + h - 1,
                        ),
                        (0, 0, 255),
                        2,
                    )

                    cv2.putText(
                        vis,
                        f"A={rec['area']}",
                        (
                            x,
                            max(
                                15,
                                y - 5,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

            cv2.imwrite(
                str(
                    out_dir
                    / f"{camera}_{sid}.jpg"
                ),
                vis,
            )

            print(
                "done",
                camera,
                sid,
                "components=",
                len(comps),
            )

    # --------------------------------------------------------
    # Component statistics
    # --------------------------------------------------------

    far_components = [
        x
        for x in all_components
        if x["distance_min_px"] > 3.0
    ]

    far_components.sort(
        key=lambda x:
            (
                x["area"],
                x["distance_max_px"],
            ),
        reverse=True,
    )

    print()
    print("=" * 68)
    print(
        "FAR COMPONENTS (>3 px FROM GT)"
    )
    print("=" * 68)

    print(
        "all components =",
        len(all_components),
    )

    print(
        "far components =",
        len(far_components),
    )

    print()

    for x in far_components[:30]:

        print(
            x["camera"],
            x["sample_id"],
            f"area={x['area']}",
            f"minD={x['distance_min_px']:.1f}",
            f"medD={x['distance_median_px']:.1f}",
            f"maxD={x['distance_max_px']:.1f}",
        )

    # --------------------------------------------------------
    # Pure inference-valid area filtering experiment.
    # No GT used for the filtering itself.
    # --------------------------------------------------------

    print()
    print("=" * 68)
    print(
        "MIN COMPONENT AREA SWEEP"
    )
    print("=" * 68)

    sweep = []

    for min_area in AREA_THRESHOLDS:

        totals = {
            "exact_mp": 0,
            "exact_pred": 0,
            "exact_mg": 0,
            "exact_gt": 0,

            "tol2_mp": 0,
            "tol2_pred": 0,
            "tol2_mg": 0,
            "tol2_gt": 0,
        }

        for item in cache:

            gt = item["gt"]

            pred = keep_components_by_area(
                item["pred"],
                min_area,
            )

            # exact
            p0, r0, _ = (
                tolerance_metrics(
                    gt,
                    pred,
                    0,
                )
            )

            # ±2 px
            p2, r2, _ = (
                tolerance_metrics(
                    gt,
                    pred,
                    2,
                )
            )

            pred_n = int(
                pred.sum()
            )

            gt_n = int(
                gt.sum()
            )

            totals[
                "exact_mp"
            ] += int(
                round(
                    p0 * pred_n
                )
            )

            totals[
                "exact_pred"
            ] += pred_n

            totals[
                "exact_mg"
            ] += int(
                round(
                    r0 * gt_n
                )
            )

            totals[
                "exact_gt"
            ] += gt_n

            totals[
                "tol2_mp"
            ] += int(
                round(
                    p2 * pred_n
                )
            )

            totals[
                "tol2_pred"
            ] += pred_n

            totals[
                "tol2_mg"
            ] += int(
                round(
                    r2 * gt_n
                )
            )

            totals[
                "tol2_gt"
            ] += gt_n

        def calc(
            matched_pred,
            pred_n,
            matched_gt,
            gt_n,
        ):
            p = (
                matched_pred / pred_n
                if pred_n > 0
                else 0.0
            )

            r = (
                matched_gt / gt_n
                if gt_n > 0
                else 0.0
            )

            f1 = (
                2 * p * r
                / (p + r)
                if p + r > 0
                else 0.0
            )

            return p, r, f1

        p0, r0, f0 = calc(
            totals["exact_mp"],
            totals["exact_pred"],
            totals["exact_mg"],
            totals["exact_gt"],
        )

        p2, r2, f2 = calc(
            totals["tol2_mp"],
            totals["tol2_pred"],
            totals["tol2_mg"],
            totals["tol2_gt"],
        )

        rec = {
            "min_area":
                min_area,
            "exact": {
                "precision": p0,
                "recall": r0,
                "f1": f0,
            },
            "tol2": {
                "precision": p2,
                "recall": r2,
                "f1": f2,
            },
        }

        sweep.append(rec)

        print(
            f"min_area={min_area:2d} "
            f"exact_F1={f0:.4f} "
            f"P={p0:.4f} "
            f"R={r0:.4f} "
            f"| ±2px_F1={f2:.4f} "
            f"P={p2:.4f} "
            f"R={r2:.4f}"
        )

    out = {
        "checkpoint":
            args.checkpoint,
        "threshold":
            args.threshold,
        "all_component_count":
            len(all_components),
        "far_component_count":
            len(far_components),
        "far_components":
            far_components,
        "area_filter_sweep":
            sweep,
    }

    (
        out_dir
        / "component_analysis.json"
    ).write_text(
        json.dumps(
            out,
            indent=2,
        )
    )

    print()
    print(
        "WROTE:",
        out_dir
        / "component_analysis.json",
    )

    print(
        "VIS DIR:",
        out_dir,
    )


if __name__ == "__main__":
    main()
