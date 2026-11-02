#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import torch

from overfit_wire_unet import UNet
from train_wire_unet_v1 import (
    CAMERAS,
    positions,
    update_confusion,
    metrics_from_confusion,
)


@torch.no_grad()
def tiled_logits(
    model,
    image_bgr,
    device,
    tile=512,
    stride=384,
):
    h, w = image_bgr.shape[:2]

    ys = positions(h, tile, stride)
    xs = positions(w, tile, stride)

    logits_sum = torch.zeros(
        (3, h, w),
        dtype=torch.float32,
        device=device,
    )

    counts = torch.zeros(
        (1, h, w),
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    for y0 in ys:
        for x0 in xs:

            crop = image_bgr[
                y0:y0+tile,
                x0:x0+tile,
            ]

            crop = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB,
            )

            crop = (
                crop.astype(np.float32)
                / 255.0
            )

            crop = np.transpose(
                crop,
                (2, 0, 1),
            )

            x = torch.from_numpy(
                crop.copy()
            ).unsqueeze(0).to(device)

            with torch.cuda.amp.autocast(
                enabled=device.type == "cuda"
            ):
                logits = model(x)[0]

            logits_sum[
                :,
                y0:y0+tile,
                x0:x0+tile,
            ] += logits.float()

            counts[
                :,
                y0:y0+tile,
                x0:x0+tile,
            ] += 1.0

    return (
        logits_sum
        / counts.clamp_min(1.0)
    )


def threshold_prediction(
    logits,
    threshold,
):
    probs = torch.softmax(
        logits,
        dim=0,
    )

    # If not accepted as wire,
    # choose between background and tool.
    nonwire = torch.stack(
        [
            logits[0],
            logits[2],
        ],
        dim=0,
    ).argmax(dim=0)

    pred = torch.where(
        nonwire == 0,
        torch.zeros_like(nonwire),
        torch.full_like(nonwire, 2),
    )

    wire_accept = (
        (probs[1] >= threshold)
        & (probs[1] >= probs[0])
        & (probs[1] >= probs[2])
    )

    pred[wire_accept] = 1

    return pred.cpu().numpy()


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

    # Cache val logits once.
    cache = []

    print("Caching validation logits...")

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

            logits = tiled_logits(
                model,
                image,
                device,
                tile=args.tile,
                stride=args.stride,
            )

            cache.append(
                (
                    sid,
                    camera,
                    mask,
                    logits,
                )
            )

    thresholds = np.arange(
        0.30,
        0.91,
        0.02,
    )

    results = []

    print()
    print(
        "threshold  wire_Dice  "
        "wire_P  wire_R  mean_fg"
    )

    for threshold in thresholds:

        conf = np.zeros(
            (3, 3),
            dtype=np.int64,
        )

        for _, _, mask, logits in cache:

            pred = threshold_prediction(
                logits,
                float(threshold),
            )

            update_confusion(
                conf,
                mask,
                pred,
            )

        metrics = metrics_from_confusion(
            conf
        )

        w = metrics["wire"]

        results.append(
            {
                "threshold":
                    float(threshold),
                "metrics":
                    metrics,
            }
        )

        print(
            f"{threshold:8.2f} "
            f"{w['dice']:10.4f} "
            f"{w['precision']:7.4f} "
            f"{w['recall']:7.4f} "
            f"{metrics['mean_fg_dice']:8.4f}"
        )

    best = max(
        results,
        key=lambda x:
            x["metrics"]["wire"]["dice"],
    )

    best_t = best["threshold"]

    print()
    print("=" * 60)
    print(
        "BEST WIRE THRESHOLD =",
        f"{best_t:.2f}",
    )

    print(
        "BEST OVERALL WIRE =",
        best["metrics"]["wire"],
    )

    print(
        "BEST MEAN FG =",
        f"{best['metrics']['mean_fg_dice']:.4f}",
    )

    print()
    print(
        "===== PER CAMERA AT BEST THRESHOLD ====="
    )

    for camera in CAMERAS:

        conf = np.zeros(
            (3, 3),
            dtype=np.int64,
        )

        for _, cam, mask, logits in cache:

            if cam != camera:
                continue

            pred = threshold_prediction(
                logits,
                best_t,
            )

            update_confusion(
                conf,
                mask,
                pred,
            )

        m = metrics_from_confusion(
            conf
        )

        w = m["wire"]
        t = m["tool"]

        print(
            camera,
            f"wire_D={w['dice']:.4f}",
            f"P={w['precision']:.4f}",
            f"R={w['recall']:.4f}",
            f"tool_D={t['dice']:.4f}",
            f"mean={m['mean_fg_dice']:.4f}",
        )

    out = (
        root
        / "runs"
        / "unet_train_val_v2_resume"
        / "wire_threshold_sweep.json"
    )

    out.write_text(
        json.dumps(
            {
                "checkpoint":
                    args.checkpoint,
                "best":
                    best,
                "results":
                    results,
            },
            indent=2,
        )
    )

    print()
    print("WROTE:", out)


if __name__ == "__main__":
    main()
