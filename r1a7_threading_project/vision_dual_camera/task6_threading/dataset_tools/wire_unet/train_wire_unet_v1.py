#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import random

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from overfit_wire_unet import (
    UNet,
    foreground_dice_loss,
    seed_all,
)


CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)


# ============================================================
# Train dataset
# ============================================================

class TrainDataset(Dataset):

    def __init__(
        self,
        root,
        sample_ids,
        crop_size=512,
    ):
        self.root = Path(root)
        self.crop_size = int(crop_size)
        self.items = []

        for sid in sample_ids:
            for camera in CAMERAS:
                self.items.append(
                    (sid, camera)
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        sid, camera = self.items[index]

        image_path = (
            self.root
            / "images"
            / camera
            / f"{sid}.png"
        )

        mask_path = (
            self.root
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
                f"read failure: {sid} {camera}"
            )

        h, w = mask.shape
        c = self.crop_size

        # Class-balanced positive crop center.
        # Deliberately only used for TRAIN.
        available = []

        if np.any(mask == 1):
            available.append(1)

        if np.any(mask == 2):
            available.append(2)

        if available:
            cls = random.choice(available)

            ys, xs = np.where(
                mask == cls
            )

            k = random.randrange(
                len(xs)
            )

            cx = int(xs[k])
            cy = int(ys[k])

        else:
            cx = w // 2
            cy = h // 2

        # Mild location jitter.
        cx += random.randint(-64, 64)
        cy += random.randint(-64, 64)

        x0 = cx - c // 2
        y0 = cy - c // 2

        x0 = max(
            0,
            min(x0, w - c),
        )

        y0 = max(
            0,
            min(y0, h - c),
        )

        image = image[
            y0:y0+c,
            x0:x0+c,
        ]

        mask = mask[
            y0:y0+c,
            x0:x0+c,
        ]

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = (
            image.astype(np.float32)
            / 255.0
        )

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        return {
            "image": torch.from_numpy(
                image.copy()
            ),
            "mask": torch.from_numpy(
                mask.astype(
                    np.int64
                ).copy()
            ),
        }


# ============================================================
# Validation
# ============================================================

def positions(length, tile, stride):

    if length <= tile:
        return [0]

    result = list(
        range(
            0,
            length - tile + 1,
            stride,
        )
    )

    last = length - tile

    if result[-1] != last:
        result.append(last)

    return result


@torch.no_grad()
def tiled_predict(
    model,
    image_bgr,
    device,
    tile=512,
    stride=384,
):

    h, w = image_bgr.shape[:2]

    ys = positions(
        h,
        tile,
        stride,
    )

    xs = positions(
        w,
        tile,
        stride,
    )

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
                crop.astype(
                    np.float32
                )
                / 255.0
            )

            crop = np.transpose(
                crop,
                (2, 0, 1),
            )

            x = torch.from_numpy(
                crop.copy()
            ).unsqueeze(0).to(
                device
            )

            with torch.cuda.amp.autocast(
                enabled=(
                    device.type
                    == "cuda"
                )
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

    logits_sum /= counts.clamp_min(
        1.0
    )

    pred = torch.argmax(
        logits_sum,
        dim=0,
    )

    return pred.cpu().numpy()


def update_confusion(
    confusion,
    target,
    pred,
):

    valid = target != 255

    target = target[valid]
    pred = pred[valid]

    good = (
        (target >= 0)
        & (target < 3)
        & (pred >= 0)
        & (pred < 3)
    )

    x = (
        target[good] * 3
        + pred[good]
    )

    bincount = np.bincount(
        x,
        minlength=9,
    ).reshape(3, 3)

    confusion += bincount


def metrics_from_confusion(conf):

    result = {}

    for cls, name in (
        (1, "wire"),
        (2, "tool"),
    ):

        tp = float(
            conf[cls, cls]
        )

        fp = float(
            conf[:, cls].sum()
            - tp
        )

        fn = float(
            conf[cls, :].sum()
            - tp
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        dice = (
            2.0 * tp
            / (
                2.0 * tp
                + fp
                + fn
            )
            if (
                2.0 * tp
                + fp
                + fn
            ) > 0
            else 0.0
        )

        iou = (
            tp
            / (
                tp
                + fp
                + fn
            )
            if (
                tp
                + fp
                + fn
            ) > 0
            else 0.0
        )

        result[name] = {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
        }

    result["mean_fg_dice"] = (
        result["wire"]["dice"]
        + result["tool"]["dice"]
    ) / 2.0

    return result


@torch.no_grad()
def validate(
    model,
    root,
    sample_ids,
    device,
    tile=512,
    stride=384,
):

    root = Path(root)

    overall = np.zeros(
        (3, 3),
        dtype=np.int64,
    )

    per_camera = {
        cam: np.zeros(
            (3, 3),
            dtype=np.int64,
        )
        for cam in CAMERAS
    }

    for sid in sample_ids:

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
                    f"VAL read failure "
                    f"{sid} {camera}"
                )

            pred = tiled_predict(
                model,
                image,
                device,
                tile=tile,
                stride=stride,
            )

            update_confusion(
                overall,
                mask,
                pred,
            )

            update_confusion(
                per_camera[camera],
                mask,
                pred,
            )

    result = {
        "overall":
            metrics_from_confusion(
                overall
            )
    }

    for camera in CAMERAS:
        result[camera] = (
            metrics_from_confusion(
                per_camera[camera]
            )
        )

    return result


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--val-stride",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    seed_all(42)

    root = Path(
        args.dataset
    )

    split = json.loads(
        (
            root
            / "splits_sample_level_seed42.json"
        ).read_text()
    )

    stats = json.loads(
        (
            root
            / "train_class_statistics.json"
        ).read_text()
    )

    weights = (
        stats[
            "candidate_weights"
        ][
            "sqrt_inverse_clipped_normalized"
        ]
    )

    train_dataset = TrainDataset(
        root,
        split["train"],
        crop_size=args.crop_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
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

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=(
                device.type
                == "cuda"
            )
        )
    )

    run_dir = (
        root
        / "runs"
        / "unet_train_val_v1"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 64)
    print("R1-A7 WIRE U-NET TRAIN/VAL V1")
    print("=" * 64)

    print(
        "device          =",
        device,
    )

    print(
        "train masks     =",
        len(train_dataset),
    )

    print(
        "val masks       =",
        len(split["val"]) * 3,
    )

    print(
        "test masks USED = 0"
    )

    print(
        "crop size       =",
        args.crop_size,
    )

    print(
        "val mode        = "
        "full-image sliding window"
    )

    print(
        "val stride      =",
        args.val_stride,
    )

    print(
        "class weights   =",
        weights,
    )

    print(
        "lr              =",
        args.lr,
    )

    print(
        "epochs          =",
        args.epochs,
    )

    print()

    best = -1.0
    best_epoch = -1
    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            image = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            target = batch[
                "mask"
            ].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=(
                    device.type
                    == "cuda"
                )
            ):

                logits = model(image)

                ce = F.cross_entropy(
                    logits,
                    target,
                    weight=class_weights,
                    ignore_index=255,
                )

                dice = (
                    foreground_dice_loss(
                        logits,
                        target,
                    )
                )

                loss = (
                    0.5 * ce
                    + 0.5 * dice
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                float(loss.item())
                * image.shape[0]
            )

        train_loss = (
            running_loss
            / len(train_dataset)
        )

        val = validate(
            model,
            root,
            split["val"],
            device,
            tile=args.crop_size,
            stride=args.val_stride,
        )

        wire = val[
            "overall"
        ]["wire"]

        tool = val[
            "overall"
        ]["tool"]

        mean_fg = val[
            "overall"
        ]["mean_fg_dice"]

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.6f} "
            f"VAL_wire_dice="
            f"{wire['dice']:.4f} "
            f"VAL_tool_dice="
            f"{tool['dice']:.4f} "
            f"VAL_mean_fg="
            f"{mean_fg:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss":
                    train_loss,
                "val": val,
            }
        )

        if mean_fg > best:

            best = mean_fg
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model":
                        model.state_dict(),
                    "optimizer":
                        optimizer.state_dict(),
                    "val": val,
                    "class_weights":
                        weights,
                    "crop_size":
                        args.crop_size,
                    "val_stride":
                        args.val_stride,
                },
                run_dir
                / "best.pt",
            )

            (
                run_dir
                / "best_metrics.json"
            ).write_text(
                json.dumps(
                    {
                        "epoch":
                            epoch,
                        "val":
                            val,
                    },
                    indent=2,
                )
            )

        (
            run_dir
            / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            )
        )

    print()
    print("=" * 64)
    print(
        "BEST EPOCH =",
        best_epoch,
    )
    print(
        "BEST VAL MEAN FG DICE =",
        f"{best:.6f}",
    )
    print(
        "checkpoint =",
        run_dir / "best.pt",
    )
    print("=" * 64)

    best_json = json.loads(
        (
            run_dir
            / "best_metrics.json"
        ).read_text()
    )

    print()
    print(
        "===== BEST PER-CAMERA VAL ====="
    )

    for camera in (
        "external",
        "right_wrist",
        "left_wrist",
    ):

        m = best_json[
            "val"
        ][camera]

        print(
            camera,
            "wire_dice=",
            f"{m['wire']['dice']:.4f}",
            "tool_dice=",
            f"{m['tool']['dice']:.4f}",
            "mean_fg=",
            f"{m['mean_fg_dice']:.4f}",
        )


if __name__ == "__main__":
    main()
