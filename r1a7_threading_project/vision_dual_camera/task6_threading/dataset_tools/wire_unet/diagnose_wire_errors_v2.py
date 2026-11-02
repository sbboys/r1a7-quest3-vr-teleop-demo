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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)

    args = parser.parse_args()

    root = Path(args.dataset)

    split = json.loads(
        (root / "splits_sample_level_seed42.json").read_text()
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
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

    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = (
        root
        / "runs"
        / "unet_train_val_v2_resume"
        / "error_analysis_t086"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

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

            if image is None or mask is None:
                raise RuntimeError(
                    f"read failed: {sid} {camera}"
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

            tp = (
                gt_wire
                & pred_wire
            )

            fp = (
                (~gt_wire)
                & pred_wire
                & valid
            )

            fn = (
                gt_wire
                & (~pred_wire)
                & valid
            )

            tp_n = int(tp.sum())
            fp_n = int(fp.sum())
            fn_n = int(fn.sum())

            precision = (
                tp_n / (tp_n + fp_n)
                if tp_n + fp_n > 0
                else 0.0
            )

            recall = (
                tp_n / (tp_n + fn_n)
                if tp_n + fn_n > 0
                else 0.0
            )

            dice = (
                2 * tp_n
                / (
                    2 * tp_n
                    + fp_n
                    + fn_n
                )
                if (
                    2 * tp_n
                    + fp_n
                    + fn_n
                ) > 0
                else 0.0
            )

            record = {
                "sample_id": sid,
                "camera": camera,
                "tp": tp_n,
                "fp": fp_n,
                "fn": fn_n,
                "precision": precision,
                "recall": recall,
                "dice": dice,
            }

            records.append(record)

            # Visualization only.
            tp_vis = cv2.dilate(
                tp.astype(np.uint8),
                kernel,
            ).astype(bool)

            fp_vis = cv2.dilate(
                fp.astype(np.uint8),
                kernel,
            ).astype(bool)

            fn_vis = cv2.dilate(
                fn.astype(np.uint8),
                kernel,
            ).astype(bool)

            overlay = image.copy()

            # BGR
            overlay[tp_vis] = (0, 255, 0)
            overlay[fp_vis] = (0, 0, 255)
            overlay[fn_vis] = (0, 255, 255)

            vis = cv2.addWeighted(
                image,
                0.55,
                overlay,
                0.45,
                0,
            )

            text = (
                f"{camera} {sid} "
                f"D={dice:.3f} "
                f"P={precision:.3f} "
                f"R={recall:.3f} "
                f"FP={fp_n}"
            )

            cv2.putText(
                vis,
                text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out = (
                out_dir
                / f"{camera}_{sid}.jpg"
            )

            cv2.imwrite(
                str(out),
                vis,
            )

    records.sort(
        key=lambda x: x["fp"],
        reverse=True,
    )

    summary = {
        "threshold": args.threshold,
        "checkpoint": args.checkpoint,
        "records": records,
    }

    (
        out_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        "===== TOP FALSE POSITIVE CASES ====="
    )

    for x in records[:15]:
        print(
            x["camera"],
            x["sample_id"],
            f"FP={x['fp']}",
            f"D={x['dice']:.4f}",
            f"P={x['precision']:.4f}",
            f"R={x['recall']:.4f}",
        )

    print()
    print("OUTPUT =", out_dir)


if __name__ == "__main__":
    main()
