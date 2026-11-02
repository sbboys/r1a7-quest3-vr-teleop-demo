#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
import torch

from overfit_wire_unet import UNet
from eval_wire_threshold_sweep import tiled_logits


DATASET = Path(
    "/home/robot/r1a7_datasets/wire_unet/pilot_001"
)

CHECKPOINT = (
    DATASET
    / "runs"
    / "unet_train_val_v2_resume"
    / "best.pt"
)

CAMERA = "right_wrist"
SID = "000041"

OUT = (
    DATASET
    / "runs"
    / "unet_train_val_v2_resume"
    / "failure_analysis_000041"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

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
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    image = cv2.imread(
        str(
            DATASET
            / "images"
            / CAMERA
            / f"{SID}.png"
        ),
        cv2.IMREAD_COLOR,
    )

    mask = cv2.imread(
        str(
            DATASET
            / "masks"
            / CAMERA
            / f"{SID}.png"
        ),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None or mask is None:
        raise RuntimeError(
            "failed to load image/mask"
        )

    logits = tiled_logits(
        model,
        image,
        device,
        tile=512,
        stride=384,
    )

    prob = torch.softmax(
        logits,
        dim=0,
    )[1].detach().cpu().numpy()

    gt_wire = mask == 1

    print(
        "wire probability on GT pixels:"
    )

    values = prob[gt_wire]

    print(
        "count  =",
        len(values),
    )

    print(
        "mean   =",
        float(values.mean()),
    )

    print(
        "median =",
        float(np.median(values)),
    )

    print(
        "p10    =",
        float(np.percentile(values, 10)),
    )

    print(
        "p25    =",
        float(np.percentile(values, 25)),
    )

    print(
        "p75    =",
        float(np.percentile(values, 75)),
    )

    print(
        "p90    =",
        float(np.percentile(values, 90)),
    )

    # --------------------------------------------------
    # Probability heatmap
    # --------------------------------------------------

    heat = np.clip(
        prob * 255.0,
        0,
        255,
    ).astype(np.uint8)

    heat_color = cv2.applyColorMap(
        heat,
        cv2.COLORMAP_JET,
    )

    blend = cv2.addWeighted(
        image,
        0.55,
        heat_color,
        0.45,
        0,
    )

    cv2.imwrite(
        str(
            OUT
            / "wire_probability_heatmap.jpg"
        ),
        blend,
    )

    # --------------------------------------------------
    # Threshold comparison
    # --------------------------------------------------

    thresholds = [
        0.30,
        0.50,
        0.70,
        0.86,
    ]

    for t in thresholds:

        pred = prob >= t

        vis = image.copy()

        # GT = green
        gt_vis = cv2.dilate(
            gt_wire.astype(np.uint8),
            np.ones(
                (5, 5),
                np.uint8,
            ),
        ).astype(bool)

        # prediction = red
        pred_vis = cv2.dilate(
            pred.astype(np.uint8),
            np.ones(
                (3, 3),
                np.uint8,
            ),
        ).astype(bool)

        overlay = vis.copy()

        overlay[gt_vis] = (
            0,
            255,
            0,
        )

        overlay[pred_vis] = (
            0,
            0,
            255,
        )

        vis = cv2.addWeighted(
            image,
            0.55,
            overlay,
            0.45,
            0,
        )

        cv2.putText(
            vis,
            f"wire threshold={t:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(
            str(
                OUT
                / f"threshold_{t:.2f}.jpg"
            ),
            vis,
        )

        gt_recall = float(
            pred[gt_wire].mean()
        )

        print(
            f"threshold={t:.2f}",
            f"GT-pixel recall={gt_recall:.4f}",
        )

    print()
    print(
        "OUTPUT =",
        OUT,
    )


if __name__ == "__main__":
    main()
