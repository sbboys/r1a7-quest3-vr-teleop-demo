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
from eval_wire_skeleton_endpoints_v2 import skeletonize
from eval_wire_connected_path_v3 import (
    connect_skeleton_components,
    longest_geodesic_path,
)


ROOT = Path(
    "/home/robot/r1a7_datasets/wire_unet/pilot_001"
)

CAMERA = "right_wrist"
SID = "000041"

CKPT = (
    ROOT
    / "runs"
    / "unet_train_val_v2_resume"
    / "best.pt"
)

THRESHOLDS = [
    0.30,
    0.50,
    0.70,
    0.86,
]

FREE_SPAN_PX = 80.0


def xy_distance_to_mask(point_yx, mask):

    dist = cv2.distanceTransform(
        (~mask).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )

    y, x = point_yx

    return float(dist[y, x])


def take_path_span(path, from_start, span_px):

    pts = (
        path
        if from_start
        else list(reversed(path))
    )

    out = [
        pts[0]
    ]

    total = 0.0

    for a, b in zip(
        pts[:-1],
        pts[1:],
    ):

        dy = b[0] - a[0]
        dx = b[1] - a[1]

        total += float(
            np.hypot(
                dy,
                dx,
            )
        )

        out.append(b)

        if total >= span_px:
            break

    return out


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
        CKPT,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    image = cv2.imread(
        str(
            ROOT
            / "images"
            / CAMERA
            / f"{SID}.png"
        )
    )

    mask = cv2.imread(
        str(
            ROOT
            / "masks"
            / CAMERA
            / f"{SID}.png"
        ),
        cv2.IMREAD_UNCHANGED,
    )

    gt_wire = (
        mask == 1
    )

    gt_tool = (
        mask == 2
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

    p_wire = probs[1]

    # ------------------------------------------------
    # Get GT visible wire path
    # ------------------------------------------------

    gt_skel = skeletonize(
        gt_wire
    )

    (
        gt_connected,
        _,
        _,
        _,
    ) = connect_skeleton_components(
        gt_skel,
        50.0,
    )

    (
        gt_path,
        _,
        _,
        _,
    ) = longest_geodesic_path(
        gt_connected
    )

    if len(gt_path) < 2:
        raise RuntimeError(
            "GT path unavailable"
        )

    # FREE = endpoint closer to tool/tweezer
    d0 = xy_distance_to_mask(
        gt_path[0],
        gt_tool,
    )

    d1 = xy_distance_to_mask(
        gt_path[-1],
        gt_tool,
    )

    free_from_start = (
        d0 <= d1
    )

    free_path = take_path_span(
        gt_path,
        free_from_start,
        FREE_SPAN_PX,
    )

    free_mask = np.zeros(
        gt_wire.shape,
        dtype=bool,
    )

    for y, x in free_path:
        free_mask[y, x] = True

    vals = p_wire[
        free_mask
    ]

    print("=" * 68)
    print(
        "right_wrist 000041 FREE-END DIAGNOSIS"
    )
    print("=" * 68)

    print(
        "GT endpoint distances to tool:",
        d0,
        d1,
    )

    print(
        "FREE endpoint =",
        "path[0]"
        if free_from_start
        else "path[-1]",
    )

    print(
        "FREE diagnostic span =",
        len(free_path),
        "path pixels",
    )

    print()
    print(
        "wire probability on FREE span:"
    )

    print(
        "  min    =",
        float(
            np.min(vals)
        ),
    )

    print(
        "  p10    =",
        float(
            np.percentile(
                vals,
                10,
            )
        ),
    )

    print(
        "  median =",
        float(
            np.median(vals)
        ),
    )

    print(
        "  p90    =",
        float(
            np.percentile(
                vals,
                90,
            )
        ),
    )

    print(
        "  max    =",
        float(
            np.max(vals)
        ),
    )

    print()
    print(
        "threshold     prob-only-R    production-R"
    )

    for t in THRESHOLDS:

        prob_only = (
            p_wire >= t
        )

        pred = threshold_prediction(
            logits,
            t,
        )

        production = (
            pred == 1
        )

        r_prob = float(
            prob_only[
                free_mask
            ].mean()
        )

        r_prod = float(
            production[
                free_mask
            ].mean()
        )

        print(
            f"{t:8.2f}"
            f"{r_prob:16.4f}"
            f"{r_prod:16.4f}"
        )


if __name__ == "__main__":
    main()
