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

from train_wire_unet_v1 import (
    validate,
    tiled_predict,
    update_confusion,
    metrics_from_confusion,
)


OLD_CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)


class MixedTrainDataset(Dataset):

    def __init__(
        self,
        old_root,
        old_ids,
        new_root,
        new_ids,
        crop_size=512,
        new_repeat=3,
    ):
        self.crop_size = int(crop_size)

        self.items = []

        old_root = Path(old_root)
        new_root = Path(new_root)

        # Old pilot replay: all 3 cameras.
        for sid in old_ids:
            for camera in OLD_CAMERAS:
                self.items.append({
                    "root": old_root,
                    "sid": sid,
                    "camera": camera,
                    "source": "old",
                })

        # New domain: right wrist only, repeated.
        for _ in range(int(new_repeat)):
            for sid in new_ids:
                self.items.append({
                    "root": new_root,
                    "sid": sid,
                    "camera": "right_wrist",
                    "source": "new",
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        item = self.items[index]

        root = item["root"]
        sid = item["sid"]
        camera = item["camera"]

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
                f"read failure: "
                f"{item['source']} "
                f"{sid} {camera}"
            )

        if mask.ndim != 2:
            raise RuntimeError(
                f"invalid mask ndim: "
                f"{sid} {camera} {mask.shape}"
            )

        h, w = mask.shape
        c = self.crop_size

        # Same class-balanced positive crop
        # used by V1/V2.
        available = []

        if np.any(mask == 1):
            available.append(1)

        if np.any(mask == 2):
            available.append(2)

        if available:

            cls = random.choice(
                available
            )

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

        # Same V1/V2 location jitter.
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


@torch.no_grad()
def validate_new_right_wrist(
    model,
    root,
    sample_ids,
    device,
    tile=512,
    stride=384,
):

    root = Path(root)

    confusion = np.zeros(
        (3, 3),
        dtype=np.int64,
    )

    used = 0

    for sid in sample_ids:

        image_path = (
            root
            / "images"
            / "right_wrist"
            / f"{sid}.png"
        )

        mask_path = (
            root
            / "masks"
            / "right_wrist"
            / f"{sid}.png"
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        target = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None or target is None:
            raise RuntimeError(
                f"new val read failure: {sid}"
            )

        pred = tiled_predict(
            model,
            image,
            device,
            tile=tile,
            stride=stride,
        )

        update_confusion(
            confusion,
            target,
            pred,
        )

        used += 1

    metrics = metrics_from_confusion(
        confusion
    )

    metrics["count"] = used
    metrics["confusion"] = (
        confusion.tolist()
    )

    return metrics


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--old-dataset",
        required=True,
    )

    parser.add_argument(
        "--new-dataset",
        required=True,
    )

    parser.add_argument(
        "--resume",
        required=True,
    )

    parser.add_argument(
        "--new-split",
        default=(
            "splits_right_wrist_open_grouped_v1.json"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
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
        default=2e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--new-repeat",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--old-score-weight",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--new-score-weight",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--old-max-drop",
        type=float,
        default=0.020,
    )

    args = parser.parse_args()

    seed_all(44)

    old_root = Path(
        args.old_dataset
    )

    new_root = Path(
        args.new_dataset
    )

    resume_path = Path(
        args.resume
    )

    if not resume_path.exists():
        raise FileNotFoundError(
            resume_path
        )

    old_split = json.loads(
        (
            old_root
            / "splits_sample_level_seed42.json"
        ).read_text()
    )

    new_split = json.loads(
        (
            new_root
            / args.new_split
        ).read_text()
    )

    stats = json.loads(
        (
            old_root
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

    train_dataset = MixedTrainDataset(
        old_root=old_root,
        old_ids=old_split["train"],
        new_root=new_root,
        new_ids=new_split["train"],
        crop_size=args.crop_size,
        new_repeat=args.new_repeat,
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

    ckpt = torch.load(
        resume_path,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    # V3 domain adaptation:
    # keep model weights,
    # deliberately reset optimizer moments.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=(
                device.type == "cuda"
            )
        )
    )

    run_dir = (
        new_root
        / "runs"
        / "v3_mixed_replay"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "R1-A7 WIRE U-NET V3 "
        "RIGHT-WRIST OPEN MIXED REPLAY"
    )
    print("=" * 72)

    print("device             =", device)
    print("old dataset        =", old_root)
    print("new dataset        =", new_root)
    print("resume             =", resume_path)

    print(
        "old train images   =",
        len(old_split["train"]) * 3
    )

    print(
        "new train unique   =",
        len(new_split["train"])
    )

    print(
        "new repeat         =",
        args.new_repeat
    )

    print(
        "total train items  =",
        len(train_dataset)
    )

    print(
        "old val images     =",
        len(old_split["val"]) * 3
    )

    print(
        "new val images     =",
        len(new_split["val"])
    )

    print(
        "new test RESERVED  =",
        len(new_split["test"])
    )

    print(
        "class weights      =",
        weights
    )

    print(
        "lr                 =",
        args.lr
    )

    print(
        "epochs             =",
        args.epochs
    )

    print()

    # ---------------------------------------------------------
    # Baseline evaluation before any V3 update.
    # ---------------------------------------------------------

    print(
        "[BASELINE] evaluating V2 on old val..."
    )

    old_val = validate(
        model,
        old_root,
        old_split["val"],
        device,
        tile=args.crop_size,
        stride=args.val_stride,
    )

    print(
        "[BASELINE] evaluating V2 on new val..."
    )

    new_val = validate_new_right_wrist(
        model,
        new_root,
        new_split["val"],
        device,
        tile=args.crop_size,
        stride=args.val_stride,
    )

    old_base = float(
        old_val[
            "overall"
        ][
            "mean_fg_dice"
        ]
    )

    new_base = float(
        new_val[
            "mean_fg_dice"
        ]
    )

    old_floor = (
        old_base
        - args.old_max_drop
    )

    best_score = (
        args.old_score_weight
        * old_base
        +
        args.new_score_weight
        * new_base
    )

    start_epoch = int(
        ckpt.get(
            "epoch",
            0
        )
    )

    best_epoch = start_epoch

    baseline = {
        "source_checkpoint":
            str(resume_path),

        "source_epoch":
            start_epoch,

        "old_val":
            old_val,

        "new_val":
            new_val,

        "combined_score":
            best_score,

        "old_guard_floor":
            old_floor,
    }

    (
        run_dir
        / "baseline_metrics.json"
    ).write_text(
        json.dumps(
            baseline,
            indent=2
        )
    )

    # Save V2 weights as the initial V3 baseline.
    torch.save(
        {
            "epoch":
                start_epoch,

            "model":
                model.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "old_val":
                old_val,

            "new_val":
                new_val,

            "combined_score":
                best_score,

            "class_weights":
                weights,

            "crop_size":
                args.crop_size,

            "val_stride":
                args.val_stride,

            "learning_rate":
                args.lr,

            "source_checkpoint":
                str(resume_path),

            "source":
                "V3 baseline V2 weights",
        },
        run_dir / "best.pt",
    )

    print()
    print(
        "BASE old mean_fg =",
        f"{old_base:.6f}"
    )

    print(
        "BASE new mean_fg =",
        f"{new_base:.6f}"
    )

    print(
        "BASE combined    =",
        f"{best_score:.6f}"
    )

    print(
        "OLD guard floor =",
        f"{old_floor:.6f}"
    )

    print()

    history = []

    for local_epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch = (
            start_epoch
            + local_epoch
        )

        model.train()

        running_loss = 0.0
        seen = 0

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

                logits = model(
                    image
                )

                ce = F.cross_entropy(
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
                    0.5 * ce
                    +
                    0.5 * dice_loss
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            bs = image.shape[0]

            running_loss += (
                float(loss.item())
                * bs
            )

            seen += bs

        train_loss = (
            running_loss
            / max(seen, 1)
        )

        old_val = validate(
            model,
            old_root,
            old_split["val"],
            device,
            tile=args.crop_size,
            stride=args.val_stride,
        )

        new_val = validate_new_right_wrist(
            model,
            new_root,
            new_split["val"],
            device,
            tile=args.crop_size,
            stride=args.val_stride,
        )

        old_mean = float(
            old_val[
                "overall"
            ][
                "mean_fg_dice"
            ]
        )

        new_mean = float(
            new_val[
                "mean_fg_dice"
            ]
        )

        score = (
            args.old_score_weight
            * old_mean
            +
            args.new_score_weight
            * new_mean
        )

        old_guard_ok = (
            old_mean >= old_floor
        )

        improved = (
            old_guard_ok
            and
            score > best_score
        )

        flag = (
            " *BEST*"
            if improved
            else ""
        )

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.6f} "
            f"old={old_mean:.4f} "
            f"new={new_mean:.4f} "
            f"score={score:.4f} "
            f"old_guard={int(old_guard_ok)}"
            f"{flag}"
        )

        history.append({
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "old_val":
                old_val,

            "new_val":
                new_val,

            "combined_score":
                score,

            "old_guard_ok":
                old_guard_ok,
        })

        (
            run_dir
            / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2
            )
        )

        if improved:

            best_score = score
            best_epoch = epoch

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model":
                        model.state_dict(),

                    "optimizer":
                        optimizer.state_dict(),

                    "old_val":
                        old_val,

                    "new_val":
                        new_val,

                    "combined_score":
                        score,

                    "class_weights":
                        weights,

                    "crop_size":
                        args.crop_size,

                    "val_stride":
                        args.val_stride,

                    "learning_rate":
                        args.lr,

                    "source_checkpoint":
                        str(resume_path),

                    "new_repeat":
                        args.new_repeat,
                },
                run_dir / "best.pt",
            )

            (
                run_dir
                / "best_metrics.json"
            ).write_text(
                json.dumps(
                    {
                        "epoch":
                            epoch,

                        "old_val":
                            old_val,

                        "new_val":
                            new_val,

                        "combined_score":
                            score,
                    },
                    indent=2
                )
            )

    print()
    print("=" * 72)

    print(
        "BASE EPOCH      =",
        start_epoch
    )

    print(
        "V3 BEST EPOCH   =",
        best_epoch
    )

    print(
        "V3 BEST SCORE   =",
        f"{best_score:.6f}"
    )

    print(
        "CHECKPOINT      =",
        run_dir / "best.pt"
    )

    print(
        "NEW TEST USED   = 0"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
