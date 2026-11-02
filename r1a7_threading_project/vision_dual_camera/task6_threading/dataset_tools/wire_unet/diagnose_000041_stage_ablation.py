#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
import torch

from overfit_wire_unet import UNet
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

THRESHOLD = 0.86
CLUSTER_BRIDGE = 24
MAX_GAP = 50.0

OUT = (
    DATASET
    / "runs"
    / "unet_train_val_v2_resume"
    / "failure_analysis_000041_stages"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def recall_on_gt(pred, gt):
    if not gt.any():
        return 0.0
    return float(
        pred[gt].mean()
    )


def save_overlay(
    name,
    image,
    gt,
    pred,
):

    gt_d = cv2.dilate(
        gt.astype(np.uint8),
        np.ones((5, 5), np.uint8),
    ).astype(bool)

    pred_d = cv2.dilate(
        pred.astype(np.uint8),
        np.ones((3, 3), np.uint8),
    ).astype(bool)

    overlay = image.copy()

    # GT green
    overlay[gt_d] = (
        0, 255, 0
    )

    # prediction red
    overlay[pred_d] = (
        0, 0, 255
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
        name,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(
            OUT / f"{name}.jpg"
        ),
        vis,
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
            "image/mask read failed"
        )

    valid = mask != 255

    gt_wire = (
        (mask == 1)
        & valid
    )

    logits = tiled_logits(
        model,
        image,
        device,
        tile=512,
        stride=384,
    )

    probs = torch.softmax(
        logits,
        dim=0,
    ).detach().cpu().numpy()

    p_bg = probs[0]
    p_wire = probs[1]
    p_tool = probs[2]

    argmax = np.argmax(
        probs,
        axis=0,
    )

    # --------------------------------------------------
    # Stage A:
    # probability only
    # --------------------------------------------------

    prob_only = (
        p_wire >= THRESHOLD
    ) & valid

    # --------------------------------------------------
    # Stage B:
    # actual production wire decision
    # --------------------------------------------------

    pred_classes = threshold_prediction(
        logits,
        THRESHOLD,
    )

    actual_wire = (
        (pred_classes == 1)
        & valid
    )

    # Equivalent diagnostic expression
    argmax_wire = (
        (argmax == 1)
        & (p_wire >= THRESHOLD)
        & valid
    )

    print(
        "actual == explicit argmax:",
        bool(
            np.array_equal(
                actual_wire,
                argmax_wire,
            )
        ),
    )

    # --------------------------------------------------
    # Stage C:
    # main cluster
    # --------------------------------------------------

    main_wire = keep_main_cluster(
        actual_wire,
        CLUSTER_BRIDGE,
    )

    # --------------------------------------------------
    # Stage D:
    # skeleton
    # --------------------------------------------------

    skel = skeletonize(
        main_wire
    )

    # --------------------------------------------------
    # Stage E:
    # connected skeleton
    # --------------------------------------------------

    (
        connected,
        bridges,
        connect_ok,
        components,
    ) = connect_skeleton_components(
        skel,
        MAX_GAP,
    )

    # --------------------------------------------------
    # Stage F:
    # final longest path
    # --------------------------------------------------

    (
        path,
        pa,
        pb,
        path_length,
    ) = longest_geodesic_path(
        connected
    )

    final_path = np.zeros(
        gt_wire.shape,
        dtype=bool,
    )

    for y, x in path:
        final_path[y, x] = True

    # 2 px path tolerance
    final_path_d2 = cv2.dilate(
        final_path.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        ),
    ).astype(bool)

    # --------------------------------------------------
    # Stage metrics
    # --------------------------------------------------

    stages = [
        (
            "01_prob_only_t086",
            prob_only,
        ),
        (
            "02_actual_argmax_t086",
            actual_wire,
        ),
        (
            "03_main_cluster",
            main_wire,
        ),
        (
            "04_skeleton",
            skel,
        ),
        (
            "05_connected_skeleton",
            connected,
        ),
        (
            "06_final_path",
            final_path,
        ),
    ]

    print()
    print(
        "===== GT WIRE RECALL BY STAGE ====="
    )

    for name, pred in stages:

        r = recall_on_gt(
            pred,
            gt_wire,
        )

        print(
            f"{name:28s}",
            f"pixels={int(pred.sum()):5d}",
            f"GT_recall={r:.4f}",
        )

        save_overlay(
            name,
            image,
            gt_wire,
            pred,
        )

    print()
    print(
        "final path ±2px GT recall =",
        recall_on_gt(
            final_path_d2,
            gt_wire,
        ),
    )

    print(
        "components before connect =",
        components,
    )

    print(
        "bridges =",
        len(bridges),
    )

    print(
        "connect_ok =",
        connect_ok,
    )

    print(
        "path_length =",
        path_length,
    )

    print(
        "endpoints yx =",
        pa,
        pb,
    )

    # --------------------------------------------------
    # Critical diagnostic:
    # pixels where wire probability >= .86,
    # but production rule rejects wire.
    # --------------------------------------------------

    rejected = (
        gt_wire
        & prob_only
        & (~actual_wire)
    )

    print()
    print(
        "===== HIGH-PROBABILITY GT WIRE REJECTED BY ARGMAX ====="
    )

    print(
        "count =",
        int(
            rejected.sum()
        ),
    )

    if rejected.any():

        winners = argmax[
            rejected
        ]

        print(
            "winner background =",
            int(
                np.sum(
                    winners == 0
                )
            ),
        )

        print(
            "winner tool =",
            int(
                np.sum(
                    winners == 2
                )
            ),
        )

        print(
            "median P(bg) =",
            float(
                np.median(
                    p_bg[rejected]
                )
            ),
        )

        print(
            "median P(wire) =",
            float(
                np.median(
                    p_wire[rejected]
                )
            ),
        )

        print(
            "median P(tool) =",
            float(
                np.median(
                    p_tool[rejected]
                )
            ),
        )

    # --------------------------------------------------
    # Pixels lost specifically by main-cluster stage
    # --------------------------------------------------

    lost_cluster = (
        gt_wire
        & actual_wire
        & (~main_wire)
    )

    print()
    print(
        "GT wire pixels lost by main cluster =",
        int(
            lost_cluster.sum()
        ),
    )

    print()
    print(
        "OUTPUT =",
        OUT,
    )


if __name__ == "__main__":
    main()
