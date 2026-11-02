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


BRIDGE_RADII = (
    1, 2, 3, 4, 5, 8, 12, 16, 24
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


def keep_main_cluster(pred, bridge_radius):
    """
    1. Dilate predicted wire so nearby fragments can join.
    2. Find connected clusters on dilated mask.
    3. Score each cluster by ORIGINAL predicted wire pixels.
    4. Keep original pixels belonging to largest cluster.

    No GT is used here, so this is valid for real inference.
    """

    if not pred.any():
        return pred.copy()

    bridged = dilate_bool(
        pred,
        bridge_radius,
    )

    n, labels = cv2.connectedComponents(
        bridged.astype(np.uint8),
        connectivity=8,
    )

    if n <= 2:
        return pred.copy()

    best_label = None
    best_score = -1

    for label in range(1, n):

        score = int(
            (
                pred
                & (labels == label)
            ).sum()
        )

        if score > best_score:
            best_score = score
            best_label = label

    keep_region = (
        labels == best_label
    )

    return (
        pred
        & keep_region
    )


def match_counts(gt, pred, radius):

    gt_d = dilate_bool(
        gt,
        radius,
    )

    pred_d = dilate_bool(
        pred,
        radius,
    )

    return {
        "matched_pred": int(
            (pred & gt_d).sum()
        ),
        "pred": int(
            pred.sum()
        ),
        "matched_gt": int(
            (gt & pred_d).sum()
        ),
        "gt": int(
            gt.sum()
        ),
    }


def add_counts(dst, src):

    for k in dst:
        dst[k] += src[k]


def empty_counts():
    return {
        "matched_pred": 0,
        "pred": 0,
        "matched_gt": 0,
        "gt": 0,
    }


def finalize(x):

    precision = (
        x["matched_pred"]
        / x["pred"]
        if x["pred"] > 0
        else 0.0
    )

    recall = (
        x["matched_gt"]
        / x["gt"]
        if x["gt"] > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_cache(cache, bridge_radius):

    overall_exact = empty_counts()
    overall_tol2 = empty_counts()

    per_camera = {
        cam: {
            "exact": empty_counts(),
            "tol2": empty_counts(),
        }
        for cam in CAMERAS
    }

    retained = 0
    original = 0

    for item in cache:

        gt = item["gt"]
        pred_raw = item["pred"]

        if bridge_radius is None:
            pred = pred_raw
        else:
            pred = keep_main_cluster(
                pred_raw,
                bridge_radius,
            )

        original += int(
            pred_raw.sum()
        )

        retained += int(
            pred.sum()
        )

        e = match_counts(
            gt,
            pred,
            0,
        )

        t = match_counts(
            gt,
            pred,
            2,
        )

        add_counts(
            overall_exact,
            e,
        )

        add_counts(
            overall_tol2,
            t,
        )

        add_counts(
            per_camera[
                item["camera"]
            ]["exact"],
            e,
        )

        add_counts(
            per_camera[
                item["camera"]
            ]["tol2"],
            t,
        )

    result = {
        "exact":
            finalize(
                overall_exact
            ),
        "tol2":
            finalize(
                overall_tol2
            ),
        "retained_fraction":
            (
                retained / original
                if original > 0
                else 0.0
            ),
        "per_camera": {},
    }

    for camera in CAMERAS:

        result[
            "per_camera"
        ][camera] = {
            "exact":
                finalize(
                    per_camera[
                        camera
                    ]["exact"]
                ),
            "tol2":
                finalize(
                    per_camera[
                        camera
                    ]["tol2"]
                ),
        }

    return result


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

    print("=" * 70)
    print(
        "R1-A7 WIRE MAIN-CLUSTER EVALUATION"
    )
    print("=" * 70)

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

    cache = []

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
                    f"read failure: "
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

            gt_wire = (
                (mask == 1)
                & valid
            )

            pred_wire = (
                (pred == 1)
                & valid
            )

            cache.append(
                {
                    "sample_id": sid,
                    "camera": camera,
                    "gt": gt_wire,
                    "pred": pred_wire,
                }
            )

            print(
                "cached",
                camera,
                sid,
            )

    print()
    print("=" * 70)
    print(
        "MAIN CLUSTER SWEEP"
    )
    print("=" * 70)

    results = []

    baseline = evaluate_cache(
        cache,
        None,
    )

    print(
        "baseline     "
        f"exact_F1={baseline['exact']['f1']:.4f} "
        f"P={baseline['exact']['precision']:.4f} "
        f"R={baseline['exact']['recall']:.4f} "
        f"| ±2px_F1={baseline['tol2']['f1']:.4f} "
        f"P={baseline['tol2']['precision']:.4f} "
        f"R={baseline['tol2']['recall']:.4f}"
    )

    for radius in BRIDGE_RADII:

        r = evaluate_cache(
            cache,
            radius,
        )

        results.append(
            {
                "bridge_radius":
                    radius,
                "result":
                    r,
            }
        )

        print(
            f"bridge={radius:2d}px "
            f"exact_F1={r['exact']['f1']:.4f} "
            f"P={r['exact']['precision']:.4f} "
            f"R={r['exact']['recall']:.4f} "
            f"| ±2px_F1={r['tol2']['f1']:.4f} "
            f"P={r['tol2']['precision']:.4f} "
            f"R={r['tol2']['recall']:.4f} "
            f"| retained={r['retained_fraction']:.4f}"
        )

    best = max(
        results,
        key=lambda x:
            x["result"]["tol2"]["f1"],
    )

    radius = best[
        "bridge_radius"
    ]

    r = best["result"]

    print()
    print("=" * 70)
    print(
        "BEST MAIN CLUSTER"
    )
    print("=" * 70)

    print(
        "bridge radius =",
        radius,
    )

    print(
        "exact =",
        r["exact"],
    )

    print(
        "±2px  =",
        r["tol2"],
    )

    print(
        "retained fraction =",
        f"{r['retained_fraction']:.4f}",
    )

    print()
    print(
        "===== PER CAMERA ====="
    )

    for camera in CAMERAS:

        x = r[
            "per_camera"
        ][camera]

        print(
            camera,
            f"exact_F1={x['exact']['f1']:.4f}",
            f"±2px_F1={x['tol2']['f1']:.4f}",
            f"±2_P={x['tol2']['precision']:.4f}",
            f"±2_R={x['tol2']['recall']:.4f}",
        )

    out = (
        root
        / "runs"
        / "unet_train_val_v2_resume"
        / "wire_main_cluster_eval.json"
    )

    out.write_text(
        json.dumps(
            {
                "checkpoint":
                    args.checkpoint,
                "threshold":
                    args.threshold,
                "baseline":
                    baseline,
                "sweep":
                    results,
                "best":
                    best,
            },
            indent=2,
        )
    )

    print()
    print(
        "WROTE:",
        out,
    )


if __name__ == "__main__":
    main()
