#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import json
import random
import time

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

CLASS_NAMES = {
    0: "background",
    1: "board",
    2: "hole",
}

IGNORE_LABEL = 255


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# U-Net
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
    ):
        super().__init__()

        groups = 8

        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                min(groups, out_ch),
                out_ch,
            ),
            nn.ReLU(
                inplace=True,
            ),

            nn.Conv2d(
                out_ch,
                out_ch,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                min(groups, out_ch),
                out_ch,
            ),
            nn.ReLU(
                inplace=True,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(x)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_classes=3,
        base=32,
    ):
        super().__init__()

        self.enc1 = ConvBlock(
            in_channels,
            base,
        )

        self.enc2 = ConvBlock(
            base,
            base * 2,
        )

        self.enc3 = ConvBlock(
            base * 2,
            base * 4,
        )

        self.enc4 = ConvBlock(
            base * 4,
            base * 8,
        )

        self.pool = nn.MaxPool2d(
            2
        )

        self.bottleneck = ConvBlock(
            base * 8,
            base * 16,
        )

        self.up4 = nn.ConvTranspose2d(
            base * 16,
            base * 8,
            2,
            stride=2,
        )

        self.dec4 = ConvBlock(
            base * 16,
            base * 8,
        )

        self.up3 = nn.ConvTranspose2d(
            base * 8,
            base * 4,
            2,
            stride=2,
        )

        self.dec3 = ConvBlock(
            base * 8,
            base * 4,
        )

        self.up2 = nn.ConvTranspose2d(
            base * 4,
            base * 2,
            2,
            stride=2,
        )

        self.dec2 = ConvBlock(
            base * 4,
            base * 2,
        )

        self.up1 = nn.ConvTranspose2d(
            base * 2,
            base,
            2,
            stride=2,
        )

        self.dec1 = ConvBlock(
            base * 2,
            base,
        )

        self.head = nn.Conv2d(
            base,
            num_classes,
            1,
        )

    def forward(
        self,
        x,
    ):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)

        d4 = self.dec4(
            torch.cat(
                [d4, e4],
                dim=1,
            )
        )

        d3 = self.up3(d4)

        d3 = self.dec3(
            torch.cat(
                [d3, e3],
                dim=1,
            )
        )

        d2 = self.up2(d3)

        d2 = self.dec2(
            torch.cat(
                [d2, e2],
                dim=1,
            )
        )

        d1 = self.up1(d2)

        d1 = self.dec1(
            torch.cat(
                [d1, e1],
                dim=1,
            )
        )

        return self.head(d1)


# ============================================================
# Dataset
# ============================================================

def build_samples(
    root,
    session,
    sample_ids,
):

    result = []

    for sid in sample_ids:

        for camera in CAMERAS:

            image_path = (
                session
                / camera
                / f"{sid}.png"
            )

            mask_path = (
                root
                / "masks_v3"
                / camera
                / f"{sid}.png"
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    image_path
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    mask_path
                )

            result.append({
                "sample_id": sid,
                "camera": camera,
                "image_path": image_path,
                "mask_path": mask_path,
            })

    return result


class HoleCropDataset(Dataset):

    def __init__(
        self,
        samples,
        crop_size=512,
        repeats=3,
    ):
        self.samples = samples
        self.crop_size = crop_size
        self.repeats = repeats

    def __len__(self):

        return (
            len(self.samples)
            *
            self.repeats
        )

    def choose_center(
        self,
        mask,
    ):

        r = random.random()

        # 65%: center crop around Class 2 hole.
        if r < 0.65:

            ys, xs = np.where(
                mask == 2
            )

            if len(xs) > 0:

                i = random.randrange(
                    len(xs)
                )

                return (
                    int(xs[i]),
                    int(ys[i]),
                )

        # 25%: board region.
        if r < 0.90:

            ys, xs = np.where(
                mask == 1
            )

            if len(xs) > 0:

                i = random.randrange(
                    len(xs)
                )

                return (
                    int(xs[i]),
                    int(ys[i]),
                )

        # 10%, or fallback: random.
        h, w = mask.shape[:2]

        return (
            random.randrange(w),
            random.randrange(h),
        )

    def crop(
        self,
        image,
        mask,
        cx,
        cy,
    ):

        size = self.crop_size

        h, w = mask.shape[:2]

        # Small random displacement avoids always putting
        # the selected pixel exactly at crop center.
        jitter = size // 6

        cx += random.randint(
            -jitter,
            jitter,
        )

        cy += random.randint(
            -jitter,
            jitter,
        )

        x0 = cx - size // 2
        y0 = cy - size // 2

        x0 = max(
            0,
            min(
                x0,
                w - size,
            ),
        )

        y0 = max(
            0,
            min(
                y0,
                h - size,
            ),
        )

        image = image[
            y0:y0 + size,
            x0:x0 + size,
        ]

        mask = mask[
            y0:y0 + size,
            x0:x0 + size,
        ]

        return image, mask

    def augment(
        self,
        image,
        mask,
    ):

        if random.random() < 0.5:

            image = np.ascontiguousarray(
                image[:, ::-1]
            )

            mask = np.ascontiguousarray(
                mask[:, ::-1]
            )

        if random.random() < 0.7:

            alpha = random.uniform(
                0.80,
                1.20,
            )

            beta = random.uniform(
                -18.0,
                18.0,
            )

            image = np.clip(
                image.astype(
                    np.float32
                )
                *
                alpha
                +
                beta,
                0,
                255,
            ).astype(
                np.uint8
            )

        if random.random() < 0.15:

            image = cv2.GaussianBlur(
                image,
                (3, 3),
                0,
            )

        return image, mask

    def __getitem__(
        self,
        index,
    ):

        item = self.samples[
            index
            %
            len(self.samples)
        ]

        image = cv2.imread(
            str(
                item["image_path"]
            ),
            cv2.IMREAD_COLOR,
        )

        mask = cv2.imread(
            str(
                item["mask_path"]
            ),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise RuntimeError(
                item["image_path"]
            )

        if mask is None:
            raise RuntimeError(
                item["mask_path"]
            )

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        cx, cy = self.choose_center(
            mask
        )

        image, mask = self.crop(
            image,
            mask,
            cx,
            cy,
        )

        image, mask = self.augment(
            image,
            mask,
        )

        image = (
            image.astype(
                np.float32
            )
            /
            255.0
        )

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        return (
            torch.from_numpy(
                image
            ).float(),
            torch.from_numpy(
                mask.astype(
                    np.int64
                )
            ).long(),
        )


# ============================================================
# Loss
# ============================================================

def foreground_dice_loss(
    logits,
    target,
):

    valid = (
        target
        !=
        IGNORE_LABEL
    )

    safe_target = target.clone()

    safe_target[
        ~valid
    ] = 0

    probs = torch.softmax(
        logits,
        dim=1,
    )

    onehot = F.one_hot(
        safe_target,
        num_classes=3,
    ).permute(
        0,
        3,
        1,
        2,
    ).float()

    valid_f = (
        valid.unsqueeze(1)
        .float()
    )

    probs = (
        probs
        *
        valid_f
    )

    onehot = (
        onehot
        *
        valid_f
    )

    dice_values = []

    eps = 1e-6

    for cls in (
        1,
        2,
    ):

        p = probs[
            :,
            cls,
        ]

        t = onehot[
            :,
            cls,
        ]

        intersection = (
            p * t
        ).sum()

        denominator = (
            p.sum()
            +
            t.sum()
        )

        dice = (
            2.0
            *
            intersection
            +
            eps
        ) / (
            denominator
            +
            eps
        )

        dice_values.append(
            dice
        )

    mean_dice = torch.stack(
        dice_values
    ).mean()

    return (
        1.0
        -
        mean_dice
    )


# ============================================================
# Sliding-window validation
# ============================================================

@torch.no_grad()
def predict_sliding(
    model,
    image,
    device,
    tile=512,
    stride=384,
):

    h, w = image.shape[:2]

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image_rgb = (
        image_rgb.astype(
            np.float32
        )
        /
        255.0
    )

    ys = list(
        range(
            0,
            max(
                h - tile,
                0,
            )
            + 1,
            stride,
        )
    )

    xs = list(
        range(
            0,
            max(
                w - tile,
                0,
            )
            + 1,
            stride,
        )
    )

    if not ys:
        ys = [0]

    if not xs:
        xs = [0]

    if ys[-1] != h - tile:
        ys.append(
            max(
                0,
                h - tile,
            )
        )

    if xs[-1] != w - tile:
        xs.append(
            max(
                0,
                w - tile,
            )
        )

    logits_sum = np.zeros(
        (
            3,
            h,
            w,
        ),
        dtype=np.float32,
    )

    counts = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.float32,
    )

    for y0 in ys:

        for x0 in xs:

            crop = image_rgb[
                y0:y0 + tile,
                x0:x0 + tile,
            ]

            ch, cw = crop.shape[:2]

            if (
                ch != tile
                or
                cw != tile
            ):

                padded = np.zeros(
                    (
                        tile,
                        tile,
                        3,
                    ),
                    dtype=np.float32,
                )

                padded[
                    :ch,
                    :cw,
                ] = crop

                crop = padded

            x = np.transpose(
                crop,
                (2, 0, 1),
            )

            x = torch.from_numpy(
                x
            ).unsqueeze(
                0
            ).float().to(
                device
            )

            logits = model(x)[0]

            logits = (
                logits.detach()
                .float()
                .cpu()
                .numpy()
            )

            logits = logits[
                :,
                :ch,
                :cw,
            ]

            logits_sum[
                :,
                y0:y0 + ch,
                x0:x0 + cw,
            ] += logits

            counts[
                y0:y0 + ch,
                x0:x0 + cw,
            ] += 1.0

    logits_sum /= np.maximum(
        counts[None, :, :],
        1.0,
    )

    return np.argmax(
        logits_sum,
        axis=0,
    ).astype(
        np.uint8
    )


def empty_stats():

    return {
        1: {
            "inter": 0,
            "pred": 0,
            "gt": 0,
        },
        2: {
            "inter": 0,
            "pred": 0,
            "gt": 0,
        },
    }


def update_stats(
    stats,
    pred,
    gt,
):

    valid = (
        gt
        !=
        IGNORE_LABEL
    )

    for cls in (
        1,
        2,
    ):

        p = (
            (pred == cls)
            &
            valid
        )

        g = (
            (gt == cls)
            &
            valid
        )

        stats[cls][
            "inter"
        ] += int(
            np.count_nonzero(
                p & g
            )
        )

        stats[cls][
            "pred"
        ] += int(
            np.count_nonzero(
                p
            )
        )

        stats[cls][
            "gt"
        ] += int(
            np.count_nonzero(
                g
            )
        )


def stats_dice(
    stats,
    cls,
):

    s = stats[cls]

    denom = (
        s["pred"]
        +
        s["gt"]
    )

    if denom == 0:
        return 1.0

    return (
        2.0
        *
        s["inter"]
        /
        denom
    )


@torch.no_grad()
def validate(
    model,
    samples,
    device,
):

    model.eval()

    total_stats = empty_stats()

    camera_stats = {
        camera:
            empty_stats()
        for camera in CAMERAS
    }

    for item in samples:

        image = cv2.imread(
            str(
                item["image_path"]
            )
        )

        gt = cv2.imread(
            str(
                item["mask_path"]
            ),
            cv2.IMREAD_UNCHANGED,
        )

        if gt.ndim == 3:
            gt = gt[:, :, 0]

        pred = predict_sliding(
            model,
            image,
            device,
        )

        update_stats(
            total_stats,
            pred,
            gt,
        )

        update_stats(
            camera_stats[
                item["camera"]
            ],
            pred,
            gt,
        )

    board = stats_dice(
        total_stats,
        1,
    )

    hole = stats_dice(
        total_stats,
        2,
    )

    mean_fg = (
        board
        +
        hole
    ) / 2.0

    # Hole is the primary task.
    score = (
        0.70
        *
        hole
        +
        0.30
        *
        mean_fg
    )

    result = {
        "board_dice":
            board,

        "hole_dice":
            hole,

        "mean_fg_dice":
            mean_fg,

        "score":
            score,

        "per_camera": {},
    }

    for camera in CAMERAS:

        b = stats_dice(
            camera_stats[
                camera
            ],
            1,
        )

        h = stats_dice(
            camera_stats[
                camera
            ],
            2,
        )

        result[
            "per_camera"
        ][camera] = {
            "board_dice":
                b,

            "hole_dice":
                h,

            "mean_fg_dice":
                (b + h) / 2.0,
        }

    return result


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/hole_tube_001"
        ),
    )

    parser.add_argument(
        "--session",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/hole_tube_001/"
            "images_raw/"
            "session_20261105_105208"
        ),
    )

    parser.add_argument(
        "--split",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/hole_tube_001/"
            "splits_hole_board_grouped_v1.json"
        ),
    )

    parser.add_argument(
        "--run-dir",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/hole_tube_001/"
            "runs/"
            "hole_board_unet_v1"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    root = Path(
        args.root
    )

    session = Path(
        args.session
    )

    split_path = Path(
        args.split
    )

    run_dir = Path(
        args.run_dir
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split = json.loads(
        split_path.read_text()
    )

    train_samples = build_samples(
        root,
        session,
        split["train"],
    )

    val_samples = build_samples(
        root,
        session,
        split["val"],
    )

    test_samples = build_samples(
        root,
        session,
        split["test"],
    )

    print(
        "=============================================="
    )

    print(
        "R1-A7 HOLE BOARD U-NET V1"
    )

    print(
        "=============================================="
    )

    print(
        "train groups =",
        len(
            split["train"]
        ),
        "images =",
        len(train_samples),
    )

    print(
        "val groups   =",
        len(
            split["val"]
        ),
        "images =",
        len(val_samples),
    )

    print(
        "test groups  =",
        len(
            split["test"]
        ),
        "images =",
        len(test_samples),
        "(RESERVED / NOT USED)",
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "device =",
        device,
    )

    if device.type == "cuda":

        print(
            "gpu =",
            torch.cuda.get_device_name(
                0
            ),
        )

    train_dataset = HoleCropDataset(
        train_samples,
        crop_size=args.crop_size,
        repeats=args.repeats,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
        drop_last=True,
    )

    model = UNet(
        in_channels=3,
        num_classes=3,
        base=32,
    ).to(
        device
    )

    # Background intentionally downweighted.
    # Hole intentionally upweighted.
    class_weights = torch.tensor(
        [
            0.20,
            1.00,
            5.00,
        ],
        dtype=torch.float32,
        device=device,
    )

    ce_loss = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=IGNORE_LABEL,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )
    )

    use_amp = (
        device.type
        ==
        "cuda"
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=use_amp
        )
    )

    config = {
        "architecture":
            "UNet_GroupNorm",

        "base":
            32,

        "num_classes":
            3,

        "classes":
            CLASS_NAMES,

        "ignore_label":
            255,

        "crop_size":
            args.crop_size,

        "crop_sampling":
            {
                "hole": 0.65,
                "board": 0.25,
                "random": 0.10,
            },

        "class_weights":
            [
                0.20,
                1.00,
                5.00,
            ],

        "loss":
            "0.5 CE + 0.5 foreground Dice",

        "selection_score":
            (
                "0.70 HoleDice + "
                "0.30 MeanFGDice"
            ),

        "train_groups":
            split["train"],

        "val_groups":
            split["val"],

        "test_groups":
            split["test"],
    }

    (
        run_dir
        /
        "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        )
        +
        "\n"
    )

    history_path = (
        run_dir
        /
        "history.csv"
    )

    best_score = -1.0

    history_fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_ce",
        "train_dice_loss",
        "val_board_dice",
        "val_hole_dice",
        "val_mean_fg_dice",
        "val_score",
        "external_board",
        "external_hole",
        "right_wrist_board",
        "right_wrist_hole",
        "left_wrist_board",
        "left_wrist_hole",
    ]

    with history_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history_fields,
        )

        writer.writeheader()

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        loss_sum = 0.0
        ce_sum = 0.0
        dice_sum = 0.0
        batches = 0

        t0 = time.time()

        for images, masks in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):

                logits = model(
                    images
                )

                ce = ce_loss(
                    logits,
                    masks,
                )

                dice = (
                    foreground_dice_loss(
                        logits,
                        masks,
                    )
                )

                loss = (
                    0.5 * ce
                    +
                    0.5 * dice
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            loss_sum += float(
                loss.item()
            )

            ce_sum += float(
                ce.item()
            )

            dice_sum += float(
                dice.item()
            )

            batches += 1

        val = validate(
            model,
            val_samples,
            device,
        )

        lr_now = optimizer.param_groups[
            0
        ]["lr"]

        train_loss = (
            loss_sum
            /
            max(
                batches,
                1,
            )
        )

        train_ce = (
            ce_sum
            /
            max(
                batches,
                1,
            )
        )

        train_dice = (
            dice_sum
            /
            max(
                batches,
                1,
            )
        )

        elapsed = (
            time.time()
            -
            t0
        )

        print()
        print(
            f"[EPOCH {epoch:03d}/{args.epochs}] "
            f"loss={train_loss:.5f} "
            f"ce={train_ce:.5f} "
            f"dice_loss={train_dice:.5f} "
            f"lr={lr_now:.7f} "
            f"time={elapsed:.1f}s"
        )

        print(
            "VAL:",
            f"board={val['board_dice']:.4f}",
            f"hole={val['hole_dice']:.4f}",
            f"mean_fg={val['mean_fg_dice']:.4f}",
            f"score={val['score']:.4f}",
        )

        for camera in CAMERAS:

            m = val[
                "per_camera"
            ][camera]

            print(
                f"  {camera:12s}",
                f"board={m['board_dice']:.4f}",
                f"hole={m['hole_dice']:.4f}",
                f"mean={m['mean_fg_dice']:.4f}",
            )

        row = {
            "epoch":
                epoch,

            "lr":
                lr_now,

            "train_loss":
                train_loss,

            "train_ce":
                train_ce,

            "train_dice_loss":
                train_dice,

            "val_board_dice":
                val["board_dice"],

            "val_hole_dice":
                val["hole_dice"],

            "val_mean_fg_dice":
                val["mean_fg_dice"],

            "val_score":
                val["score"],

            "external_board":
                val["per_camera"][
                    "external"
                ]["board_dice"],

            "external_hole":
                val["per_camera"][
                    "external"
                ]["hole_dice"],

            "right_wrist_board":
                val["per_camera"][
                    "right_wrist"
                ]["board_dice"],

            "right_wrist_hole":
                val["per_camera"][
                    "right_wrist"
                ]["hole_dice"],

            "left_wrist_board":
                val["per_camera"][
                    "left_wrist"
                ]["board_dice"],

            "left_wrist_hole":
                val["per_camera"][
                    "left_wrist"
                ]["hole_dice"],
        }

        with history_path.open(
            "a",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=history_fields,
            )

            writer.writerow(
                row
            )

        checkpoint = {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "metrics":
                val,

            "architecture": {
                "name":
                    "UNet",

                "in_channels":
                    3,

                "num_classes":
                    3,

                "base":
                    32,

                "normalization":
                    "GroupNorm",
            },

            "class_names":
                CLASS_NAMES,

            "ignore_label":
                255,
        }

        torch.save(
            checkpoint,
            run_dir
            /
            "last.pt",
        )

        if (
            val["score"]
            >
            best_score
        ):

            best_score = (
                val["score"]
            )

            torch.save(
                checkpoint,
                run_dir
                /
                "best.pt",
            )

            print(
                "*** NEW BEST ***",
                f"score={best_score:.4f}",
                f"hole={val['hole_dice']:.4f}",
            )

        scheduler.step()

    print()
    print(
        "=============================================="
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "best score =",
        round(
            best_score,
            6,
        ),
    )

    print(
        "best checkpoint =",
        run_dir
        /
        "best.pt",
    )

    print(
        "history =",
        history_path,
    )

    print(
        "TEST SET WAS NOT USED."
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()
