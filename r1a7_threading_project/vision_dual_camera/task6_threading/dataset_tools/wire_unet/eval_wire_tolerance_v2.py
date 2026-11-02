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


RADII = (0, 1, 2, 3, 4)


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


def init_acc():
    return {
        r: {
            "matched_pred": 0,
            "pred": 0,
            "matched_gt": 0,
            "gt": 0,
        }
        for r in RADII
    }


def add_tolerance(acc, gt, pred):

    gt_count = int(gt.sum())
    pred_count = int(pred.sum())

    for r in RADII:

        if r == 0:
            gt_dil = gt
            pred_dil = pred
        else:
            gt_dil = dilate_bool(
                gt,
                r,
            )
            pred_dil = dilate_bool(
                pred,
                r,
            )

        matched_pred = int(
            (pred & gt_dil).sum()
        )

        matched_gt = int(
            (gt & pred_dil).sum()
        )

        acc[r]["matched_pred"] += (
            matched_pred
        )

        acc[r]["pred"] += (
            pred_count
        )

        acc[r]["matched_gt"] += (
            matched_gt
        )

        acc[r]["gt"] += (
            gt_count
        )


def finalize_tolerance(acc):

    out = {}

    for r in RADII:

        x = acc[r]

        p = (
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
            2.0 * p * recall
            / (p + recall)
            if p + recall > 0
            else 0.0
        )

        out[str(r)] = {
            "precision": p,
            "recall": recall,
            "f1": f1,
        }

    return out


def pixel_distances(gt, pred):
    """
    Returns:
      pred_to_gt distances:
          each predicted wire pixel -> nearest GT wire pixel
      gt_to_pred distances:
          each GT wire pixel -> nearest predicted wire pixel
    """

    pred_to_gt = np.array(
        [],
        dtype=np.float32,
    )

    gt_to_pred = np.array(
        [],
        dtype=np.float32,
    )

    if gt.any():

        # distanceTransform measures distance
        # to nearest zero pixel.
        dist_to_gt = cv2.distanceTransform(
            (~gt).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

        if pred.any():
            pred_to_gt = (
                dist_to_gt[pred]
                .astype(np.float32)
            )

    if pred.any():

        dist_to_pred = cv2.distanceTransform(
            (~pred).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

        if gt.any():
            gt_to_pred = (
                dist_to_pred[gt]
                .astype(np.float32)
            )

    return (
        pred_to_gt,
        gt_to_pred,
    )


def summarize_distance(arr):

    if len(arr) == 0:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    return {
        "mean": float(
            np.mean(arr)
        ),
        "median": float(
            np.median(arr)
        ),
        "p95": float(
            np.percentile(
                arr,
                95,
            )
        ),
        "max": float(
            np.max(arr)
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

    overall = init_acc()

    per_camera = {
        camera: init_acc()
        for camera in CAMERAS
    }

    overall_pred_to_gt = []
    overall_gt_to_pred = []

    camera_pred_to_gt = {
        camera: []
        for camera in CAMERAS
    }

    camera_gt_to_pred = {
        camera: []
        for camera in CAMERAS
    }

    print("=" * 68)
    print(
        "R1-A7 WIRE TOLERANCE EVALUATION"
    )
    print("=" * 68)

    print(
        "checkpoint =",
        args.checkpoint,
    )

    print(
        "threshold  =",
        args.threshold,
    )

    print(
        "val masks   =",
        len(split["val"]) * 3,
    )

    print(
        "test USED   = 0"
    )

    print()

    for sid in split["val"]:

        for camera in CAMERAS:

            image_path = (
                root
                / "images"
                / camera
                / f"{sid}.png"
            )

            mask_path = (
                root
                / "masks"
                / camera
                / f"{sid}.png"
            )

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR,
            )

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_UNCHANGED,
            )

            if (
                image is None
                or mask is None
            ):
                raise RuntimeError(
                    f"read failure "
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

            add_tolerance(
                overall,
                gt_wire,
                pred_wire,
            )

            add_tolerance(
                per_camera[camera],
                gt_wire,
                pred_wire,
            )

            p2g, g2p = pixel_distances(
                gt_wire,
                pred_wire,
            )

            if len(p2g):
                overall_pred_to_gt.extend(
                    p2g.tolist()
                )

                camera_pred_to_gt[
                    camera
                ].extend(
                    p2g.tolist()
                )

            if len(g2p):
                overall_gt_to_pred.extend(
                    g2p.tolist()
                )

                camera_gt_to_pred[
                    camera
                ].extend(
                    g2p.tolist()
                )

            print(
                "done",
                camera,
                sid,
            )

    results = {}

    for name in (
        "overall",
        *CAMERAS,
    ):

        if name == "overall":
            tol = finalize_tolerance(
                overall
            )

            p2g = (
                overall_pred_to_gt
            )

            g2p = (
                overall_gt_to_pred
            )

        else:
            tol = finalize_tolerance(
                per_camera[name]
            )

            p2g = (
                camera_pred_to_gt[
                    name
                ]
            )

            g2p = (
                camera_gt_to_pred[
                    name
                ]
            )

        results[name] = {
            "tolerance":
                tol,
            "pred_to_gt_distance_px":
                summarize_distance(
                    p2g
                ),
            "gt_to_pred_distance_px":
                summarize_distance(
                    g2p
                ),
        }

    print()
    print("=" * 68)
    print(
        "TOLERANCE RESULTS"
    )
    print("=" * 68)

    for name in (
        "overall",
        *CAMERAS,
    ):

        print()
        print(
            "-----",
            name,
            "-----"
        )

        r = results[name]

        for radius in RADII:

            m = r[
                "tolerance"
            ][str(radius)]

            print(
                f"±{radius}px "
                f"F1={m['f1']:.4f} "
                f"P={m['precision']:.4f} "
                f"R={m['recall']:.4f}"
            )

        print(
            "pred->GT distance:",
            r[
                "pred_to_gt_distance_px"
            ],
        )

        print(
            "GT->pred distance:",
            r[
                "gt_to_pred_distance_px"
            ],
        )

    out = (
        root
        / "runs"
        / "unet_train_val_v2_resume"
        / "wire_tolerance_eval_t086.json"
    )

    out.write_text(
        json.dumps(
            {
                "checkpoint":
                    args.checkpoint,
                "threshold":
                    args.threshold,
                "radii_px":
                    list(RADII),
                "results":
                    results,
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
