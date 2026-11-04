#!/usr/bin/env python3

from pathlib import Path
import json
import cv2
import numpy as np
import torch

from overfit_wire_unet import UNet

from train_wire_unet_v1 import (
    tiled_predict,
    update_confusion,
    metrics_from_confusion,
)

OLD_CKPT = Path(
    "/home/robot/r1a7_datasets/wire_unet/"
    "pilot_001/runs/unet_train_val_v2_resume/best.pt"
)

NEW_CKPT = Path(
    "/home/robot/r1a7_datasets/wire_unet/"
    "right_wrist_open_001/runs/v3_mixed_replay/best.pt"
)

ROOT = Path(
    "/home/robot/r1a7_datasets/wire_unet/"
    "right_wrist_open_001"
)

SPLIT = ROOT / "splits_right_wrist_open_grouped_v1.json"

OUT_DIR = ROOT / "eval_v2_v3_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

split = json.loads(SPLIT.read_text())
test_ids = split["test"]


def load_model(path):
    ckpt = torch.load(
        path,
        map_location=device,
    )

    model = UNet(
        in_channels=3,
        num_classes=3,
        base=32,
    ).to(device)

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    return model


def one_metrics(target, pred):

    conf = np.zeros(
        (3, 3),
        dtype=np.int64,
    )

    update_confusion(
        conf,
        target,
        pred,
    )

    return metrics_from_confusion(conf)


def color_mask(mask):

    out = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    # wire green
    out[mask == 1] = (0, 255, 0)

    # tool red
    out[mask == 2] = (0, 0, 255)

    # ignore magenta
    out[mask == 255] = (255, 0, 255)

    return out


print("device =", device)
print("test ids =", test_ids)

v2 = load_model(OLD_CKPT)
v3 = load_model(NEW_CKPT)

conf_v2 = np.zeros(
    (3, 3),
    dtype=np.int64,
)

conf_v3 = np.zeros(
    (3, 3),
    dtype=np.int64,
)

rows = []

for sid in test_ids:

    ip = (
        ROOT
        / "images"
        / "right_wrist"
        / f"{sid}.png"
    )

    mp = (
        ROOT
        / "masks"
        / "right_wrist"
        / f"{sid}.png"
    )

    image = cv2.imread(
        str(ip),
        cv2.IMREAD_COLOR,
    )

    target = cv2.imread(
        str(mp),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None or target is None:
        raise RuntimeError(
            f"read failure {sid}"
        )

    pred_v2 = tiled_predict(
        v2,
        image,
        device,
        tile=512,
        stride=384,
    )

    pred_v3 = tiled_predict(
        v3,
        image,
        device,
        tile=512,
        stride=384,
    )

    update_confusion(
        conf_v2,
        target,
        pred_v2,
    )

    update_confusion(
        conf_v3,
        target,
        pred_v3,
    )

    m2 = one_metrics(
        target,
        pred_v2,
    )

    m3 = one_metrics(
        target,
        pred_v3,
    )

    rows.append({
        "sid": sid,
        "v2_wire":
            m2["wire"]["dice"],
        "v2_tool":
            m2["tool"]["dice"],
        "v2_mean":
            m2["mean_fg_dice"],
        "v3_wire":
            m3["wire"]["dice"],
        "v3_tool":
            m3["tool"]["dice"],
        "v3_mean":
            m3["mean_fg_dice"],
    })

    gt_color = color_mask(target)
    v2_color = color_mask(pred_v2)
    v3_color = color_mask(pred_v3)

    gt_overlay = cv2.addWeighted(
        image, 0.65,
        gt_color, 0.35,
        0
    )

    v2_overlay = cv2.addWeighted(
        image, 0.65,
        v2_color, 0.35,
        0
    )

    v3_overlay = cv2.addWeighted(
        image, 0.65,
        v3_color, 0.35,
        0
    )

    panel = np.hstack([
        image,
        gt_overlay,
        v2_overlay,
        v3_overlay,
    ])

    cv2.putText(
        panel,
        "ORIGINAL",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2,
    )

    w = image.shape[1]

    cv2.putText(
        panel,
        "GT",
        (w + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2,
    )

    cv2.putText(
        panel,
        "V2",
        (2*w + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2,
    )

    cv2.putText(
        panel,
        "V3",
        (3*w + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2,
    )

    cv2.imwrite(
        str(
            OUT_DIR
            / f"{sid}_compare.jpg"
        ),
        panel
    )


overall_v2 = metrics_from_confusion(
    conf_v2
)

overall_v3 = metrics_from_confusion(
    conf_v3
)

print()
print("=" * 72)
print("PER IMAGE")
print("=" * 72)

for r in rows:
    print(
        r["sid"],
        f"V2 mean={r['v2_mean']:.4f}",
        f"V3 mean={r['v3_mean']:.4f}",
        f"delta={r['v3_mean']-r['v2_mean']:+.4f}",
        f"| wire {r['v2_wire']:.4f}->{r['v3_wire']:.4f}",
        f"| tool {r['v2_tool']:.4f}->{r['v3_tool']:.4f}",
    )

print()
print("=" * 72)
print("OVERALL TEST")
print("=" * 72)

print(
    "V2 wire =",
    f"{overall_v2['wire']['dice']:.6f}"
)

print(
    "V2 tool =",
    f"{overall_v2['tool']['dice']:.6f}"
)

print(
    "V2 mean =",
    f"{overall_v2['mean_fg_dice']:.6f}"
)

print()

print(
    "V3 wire =",
    f"{overall_v3['wire']['dice']:.6f}"
)

print(
    "V3 tool =",
    f"{overall_v3['tool']['dice']:.6f}"
)

print(
    "V3 mean =",
    f"{overall_v3['mean_fg_dice']:.6f}"
)

print()

print(
    "DELTA mean =",
    f"{overall_v3['mean_fg_dice'] - overall_v2['mean_fg_dice']:+.6f}"
)

result = {
    "test_ids": test_ids,
    "v2": overall_v2,
    "v3": overall_v3,
    "per_image": rows,
}

(
    OUT_DIR
    / "metrics.json"
).write_text(
    json.dumps(
        result,
        indent=2
    )
)

print()
print(
    "OUTPUT =",
    OUT_DIR
)
