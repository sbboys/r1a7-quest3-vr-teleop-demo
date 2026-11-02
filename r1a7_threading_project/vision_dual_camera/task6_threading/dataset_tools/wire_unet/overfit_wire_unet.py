#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import random

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WireDataset(Dataset):

    def __init__(self, root, sample_ids, crop_size=512):
        self.root = Path(root)
        self.crop_size = int(crop_size)
        self.items = []

        for sid in sample_ids:
            for camera in CAMERAS:

                image = (
                    self.root
                    / "images"
                    / camera
                    / f"{sid}.png"
                )

                mask = (
                    self.root
                    / "masks"
                    / camera
                    / f"{sid}.png"
                )

                if not image.exists():
                    raise FileNotFoundError(image)

                if not mask.exists():
                    raise FileNotFoundError(mask)

                self.items.append(
                    {
                        "sample_id": sid,
                        "camera": camera,
                        "image": image,
                        "mask": mask,
                    }
                )

    def __len__(self):
        return len(self.items)

    def foreground_crop(self, image, mask):

        h, w = mask.shape
        c = self.crop_size

        if c > h or c > w:
            raise RuntimeError(
                f"crop_size={c} exceeds image {w}x{h}"
            )

        fg = (mask == 1) | (mask == 2)

        ys, xs = np.where(fg)

        if len(xs) > 0:
            cx = float(xs.mean())
            cy = float(ys.mean())
        else:
            cx = w / 2.0
            cy = h / 2.0

        x0 = int(round(cx - c / 2))
        y0 = int(round(cy - c / 2))

        x0 = max(0, min(x0, w - c))
        y0 = max(0, min(y0, h - c))

        return (
            image[y0:y0 + c, x0:x0 + c],
            mask[y0:y0 + c, x0:x0 + c],
        )

    def __getitem__(self, index):

        item = self.items[index]

        image = cv2.imread(
            str(item["image"]),
            cv2.IMREAD_COLOR,
        )

        mask = cv2.imread(
            str(item["mask"]),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise RuntimeError(
                f"cannot read {item['image']}"
            )

        if mask is None:
            raise RuntimeError(
                f"cannot read {item['mask']}"
            )

        if mask.ndim != 2:
            raise RuntimeError(
                f"mask is not 2D: {item['mask']}"
            )

        values = set(
            int(v) for v in np.unique(mask)
        )

        if not values.issubset(
            {0, 1, 2, 255}
        ):
            raise RuntimeError(
                f"invalid labels {values}: "
                f"{item['mask']}"
            )

        image, mask = self.foreground_crop(
            image,
            mask,
        )

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

        image = torch.from_numpy(
            image.copy()
        )

        mask = torch.from_numpy(
            mask.astype(np.int64).copy()
        )

        return {
            "image": image,
            "mask": mask,
            "sample_id": item["sample_id"],
            "camera": item["camera"],
        }


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_ch + skip_ch,
            out_ch,
        )

    def forward(self, x, skip):

        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(
            [skip, x],
            dim=1,
        )

        return self.conv(x)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_classes=3,
        base=32,
    ):
        super().__init__()

        self.enc1 = DoubleConv(
            in_channels,
            base,
        )

        self.enc2 = DoubleConv(
            base,
            base * 2,
        )

        self.enc3 = DoubleConv(
            base * 2,
            base * 4,
        )

        self.enc4 = DoubleConv(
            base * 4,
            base * 8,
        )

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            base * 8,
            base * 16,
        )

        self.up4 = UpBlock(
            base * 16,
            base * 8,
            base * 8,
        )

        self.up3 = UpBlock(
            base * 8,
            base * 4,
            base * 4,
        )

        self.up2 = UpBlock(
            base * 4,
            base * 2,
            base * 2,
        )

        self.up1 = UpBlock(
            base * 2,
            base,
            base,
        )

        self.head = nn.Conv2d(
            base,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(
            self.pool(e4)
        )

        x = self.up4(b, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)

        return self.head(x)


def foreground_dice_loss(
    logits,
    target,
    eps=1e-6,
):

    probs = torch.softmax(
        logits,
        dim=1,
    )

    valid = (
        target != 255
    ).float()

    losses = []

    for cls in (1, 2):

        pred = (
            probs[:, cls]
            * valid
        )

        truth = (
            (target == cls).float()
            * valid
        )

        intersection = (
            pred * truth
        ).sum()

        denominator = (
            pred.sum()
            + truth.sum()
        )

        dice = (
            2.0 * intersection + eps
        ) / (
            denominator + eps
        )

        losses.append(
            1.0 - dice
        )

    return torch.stack(
        losses
    ).mean()


def hard_stats(logits, target):

    pred = torch.argmax(
        logits,
        dim=1,
    )

    valid = target != 255

    result = {}

    for cls in (1, 2):

        p = (
            (pred == cls)
            & valid
        )

        t = (
            (target == cls)
            & valid
        )

        intersection = int(
            (p & t).sum().item()
        )

        pred_count = int(
            p.sum().item()
        )

        truth_count = int(
            t.sum().item()
        )

        result[cls] = (
            intersection,
            pred_count,
            truth_count,
        )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    args = parser.parse_args()

    seed_all(42)

    root = Path(args.dataset)

    split = json.loads(
        (
            root
            / "splits_sample_level_seed42.json"
        ).read_text()
    )

    sample_ids = split["train"][
        :args.sample_count
    ]

    dataset = WireDataset(
        root,
        sample_ids,
        crop_size=args.crop_size,
    )

    loader = DataLoader(
        dataset,
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

    print("=" * 60)
    print("R1-A7 WIRE U-NET OVERFIT TEST")
    print("=" * 60)
    print("device       =", device)
    print("sample_ids   =", sample_ids)
    print("images       =", len(dataset))
    print("crop_size    =", args.crop_size)
    print("batch_size   =", args.batch_size)
    print("epochs       =", args.epochs)
    print("learning_rate=", args.lr)
    print()

    model = UNet(
        in_channels=3,
        num_classes=3,
        base=32,
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("model_parameters =", parameters)
    print()

    class_weights = torch.tensor(
        [
            0.0980024184,
            2.0873056338,
            0.8146919479,
        ],
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            device.type == "cuda"
        )
    )

    run_dir = (
        root
        / "runs"
        / "unet_overfit_12"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_mean = -1.0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0.0

        stats = {
            1: [0, 0, 0],
            2: [0, 0, 0],
        }

        for batch in loader:

            image = batch["image"].to(
                device,
                non_blocking=True,
            )

            target = batch["mask"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=(
                    device.type == "cuda"
                )
            ):

                logits = model(image)

                ce_loss = F.cross_entropy(
                    logits,
                    target,
                    weight=class_weights,
                    ignore_index=255,
                )

                dice_loss = (
                    foreground_dice_loss(
                        logits,
                        target,
                    )
                )

                loss = (
                    0.5 * ce_loss
                    + 0.5 * dice_loss
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            total_loss += (
                float(loss.item())
                * image.shape[0]
            )

            batch_stats = hard_stats(
                logits.detach(),
                target,
            )

            for cls in (1, 2):
                for i in range(3):
                    stats[cls][i] += (
                        batch_stats[cls][i]
                    )

        epoch_loss = (
            total_loss
            / len(dataset)
        )

        dices = {}

        for cls in (1, 2):

            inter, pred_n, true_n = (
                stats[cls]
            )

            dices[cls] = (
                2.0 * inter + 1e-6
            ) / (
                pred_n
                + true_n
                + 1e-6
            )

        mean_fg = (
            dices[1]
            + dices[2]
        ) / 2.0

        if mean_fg > best_mean:

            best_mean = mean_fg

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": (
                        optimizer.state_dict()
                    ),
                    "wire_dice": dices[1],
                    "tool_dice": dices[2],
                    "mean_fg_dice": mean_fg,
                    "sample_ids": sample_ids,
                    "crop_size": (
                        args.crop_size
                    ),
                    "class_weights": (
                        class_weights
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                },
                run_dir / "best.pt",
            )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == args.epochs
        ):

            print(
                f"epoch={epoch:03d} "
                f"loss={epoch_loss:.6f} "
                f"wire_dice={dices[1]:.4f} "
                f"tool_dice={dices[2]:.4f} "
                f"mean_fg={mean_fg:.4f}"
            )

    print()
    print(
        "BEST mean_fg_dice =",
        f"{best_mean:.6f}",
    )

    print(
        "checkpoint =",
        run_dir / "best.pt",
    )

    print(
        "OVERFIT TEST COMPLETE"
    )


if __name__ == "__main__":
    main()
