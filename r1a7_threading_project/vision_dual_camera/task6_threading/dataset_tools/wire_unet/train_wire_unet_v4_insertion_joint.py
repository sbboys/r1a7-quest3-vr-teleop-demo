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
    tiled_predict,
    update_confusion,
    metrics_from_confusion,
)


CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)


def make_items(
    root,
    sample_ids,
    cameras,
    source,
    repeat=1,
):
    root = Path(root)

    items = []

    for _ in range(int(repeat)):
        for sid in sample_ids:
            for camera in cameras:
                items.append({
                    "root": root,
                    "sid": sid,
                    "camera": camera,
                    "source": source,
                })

    return items


class MixedWireDataset(Dataset):

    def __init__(
        self,
        items,
        crop_size=512,
    ):
        self.items = list(items)
        self.crop_size = int(crop_size)

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

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        h, w = mask.shape
        c = self.crop_size

        if h < c or w < c:
            raise RuntimeError(
                f"image smaller than crop: "
                f"{sid} {camera} "
                f"{w}x{h} crop={c}"
            )

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

        cx += random.randint(
            -64,
            64,
        )

        cy += random.randint(
            -64,
            64,
        )

        x0 = cx - c // 2
        y0 = cy - c // 2

        x0 = max(
            0,
            min(
                x0,
                w - c,
            ),
        )

        y0 = max(
            0,
            min(
                y0,
                h - c,
            ),
        )

        image = image[
            y0:y0 + c,
            x0:x0 + c,
        ]

        mask = mask[
            y0:y0 + c,
            x0:x0 + c,
        ]

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        return {
            "image":
                torch.from_numpy(
                    image.copy()
                ),

            "mask":
                torch.from_numpy(
                    mask.astype(
                        np.int64
                    ).copy()
                ),
        }


@torch.no_grad()
def validate_views(
    model,
    root,
    sample_ids,
    cameras,
    device,
    tile=512,
    stride=384,
):
    root = Path(root)

    total = np.zeros(
        (3, 3),
        dtype=np.int64,
    )

    camera_confusions = {
        camera:
            np.zeros(
                (3, 3),
                dtype=np.int64,
            )
        for camera in cameras
    }

    used = 0

    for sid in sample_ids:

        for camera in cameras:

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

            target = cv2.imread(
                str(mask_path),
                cv2.IMREAD_UNCHANGED,
            )

            if image is None or target is None:
                raise RuntimeError(
                    f"validation read failure: "
                    f"{sid} {camera}"
                )

            if target.ndim == 3:
                target = target[:, :, 0]

            pred = tiled_predict(
                model,
                image,
                device,
                tile=tile,
                stride=stride,
            )

            update_confusion(
                total,
                target,
                pred,
            )

            update_confusion(
                camera_confusions[
                    camera
                ],
                target,
                pred,
            )

            used += 1

    metrics = metrics_from_confusion(
        total
    )

    metrics["count"] = used

    metrics["confusion"] = (
        total.tolist()
    )

    metrics["per_camera"] = {}

    for camera in cameras:

        m = metrics_from_confusion(
            camera_confusions[
                camera
            ]
        )

        m["confusion"] = (
            camera_confusions[
                camera
            ].tolist()
        )

        metrics[
            "per_camera"
        ][camera] = m

    return metrics


def mean_fg(metrics):

    if "mean_fg_dice" in metrics:
        return float(
            metrics[
                "mean_fg_dice"
            ]
        )

    return float(
        metrics[
            "overall"
        ][
            "mean_fg_dice"
        ]
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot-root",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/pilot_001"
        ),
    )

    parser.add_argument(
        "--open-root",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/right_wrist_open_001"
        ),
    )

    parser.add_argument(
        "--insertion-root",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/insertion_joint_001/"
            "training_runtime"
        ),
    )

    parser.add_argument(
        "--resume",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/right_wrist_open_001/"
            "runs/v3_mixed_replay/best.pt"
        ),
    )

    parser.add_argument(
        "--run-dir",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/insertion_joint_001/"
            "training_runtime/runs/"
            "wire_v4_insertion_replay_seed44"
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
        default=1e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--pilot-repeat",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--open-repeat",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--insertion-repeat",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--pilot-score-weight",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--open-score-weight",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--insertion-score-weight",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--pilot-max-drop",
        type=float,
        default=0.030,
    )

    parser.add_argument(
        "--open-max-drop",
        type=float,
        default=0.030,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=44,
    )

    args = parser.parse_args()

    seed_all(
        args.seed
    )

    pilot_root = Path(
        args.pilot_root
    )

    open_root = Path(
        args.open_root
    )

    insertion_root = Path(
        args.insertion_root
    )

    resume_path = Path(
        args.resume
    )

    run_dir = Path(
        args.run_dir
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pilot_split = json.loads(
        (
            pilot_root
            / "splits_sample_level_seed42.json"
        ).read_text()
    )

    open_split = json.loads(
        (
            open_root
            / "splits_right_wrist_open_grouped_v1.json"
        ).read_text()
    )

    insertion_split = json.loads(
        (
            insertion_root
            / "splits_insertion_joint_grouped_seed44_v1.json"
        ).read_text()
    )

    stats = json.loads(
        (
            pilot_root
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

    items = []

    items += make_items(
        pilot_root,
        pilot_split["train"],
        CAMERAS,
        "pilot",
        args.pilot_repeat,
    )

    items += make_items(
        open_root,
        open_split["train"],
        ("right_wrist",),
        "right_wrist_open",
        args.open_repeat,
    )

    items += make_items(
        insertion_root,
        insertion_split["train"],
        CAMERAS,
        "insertion",
        args.insertion_repeat,
    )

    train_dataset = MixedWireDataset(
        items,
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

    ckpt = torch.load(
        resume_path,
        map_location=device,
    )

    if "model" in ckpt:
        state = ckpt["model"]

    elif "model_state_dict" in ckpt:
        state = ckpt[
            "model_state_dict"
        ]

    else:
        raise RuntimeError(
            "resume checkpoint has "
            "no model/model_state_dict"
        )

    model.load_state_dict(
        state
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=(
                args.weight_decay
            ),
        )
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=(
                device.type
                == "cuda"
            )
        )
    )

    print("=" * 78)
    print(
        "R1-A7 WIRE U-NET V4 "
        "INSERTION MIXED REPLAY"
    )
    print("=" * 78)

    print("device =", device)

    print(
        "pilot train items =",
        len(
            pilot_split[
                "train"
            ]
        )
        * 3
        * args.pilot_repeat,
    )

    print(
        "open train items =",
        len(
            open_split[
                "train"
            ]
        )
        * args.open_repeat,
    )

    print(
        "insertion train items =",
        len(
            insertion_split[
                "train"
            ]
        )
        * 3
        * args.insertion_repeat,
    )

    print(
        "total train items =",
        len(train_dataset),
    )

    print(
        "insertion test RESERVED =",
        len(
            insertion_split[
                "test"
            ]
        )
        * 3,
    )

    print(
        "resume =",
        resume_path,
    )

    print()

    print(
        "[BASELINE] pilot val"
    )

    pilot_val = validate_views(
        model,
        pilot_root,
        pilot_split["val"],
        CAMERAS,
        device,
        tile=args.crop_size,
        stride=args.val_stride,
    )

    print(
        "[BASELINE] right-wrist-open val"
    )

    open_val = validate_views(
        model,
        open_root,
        open_split["val"],
        ("right_wrist",),
        device,
        tile=args.crop_size,
        stride=args.val_stride,
    )

    print(
        "[BASELINE] insertion val"
    )

    insertion_val = validate_views(
        model,
        insertion_root,
        insertion_split["val"],
        CAMERAS,
        device,
        tile=args.crop_size,
        stride=args.val_stride,
    )

    pilot_base = mean_fg(
        pilot_val
    )

    open_base = mean_fg(
        open_val
    )

    insertion_base = mean_fg(
        insertion_val
    )

    pilot_floor = (
        pilot_base
        - args.pilot_max_drop
    )

    open_floor = (
        open_base
        - args.open_max_drop
    )

    best_score = (
        args.pilot_score_weight
        * pilot_base
        +
        args.open_score_weight
        * open_base
        +
        args.insertion_score_weight
        * insertion_base
    )

    baseline = {
        "source_checkpoint":
            str(resume_path),

        "pilot_val":
            pilot_val,

        "open_val":
            open_val,

        "insertion_val":
            insertion_val,

        "combined_score":
            best_score,

        "pilot_guard_floor":
            pilot_floor,

        "open_guard_floor":
            open_floor,

        "insertion_test_used":
            0,
    }

    (
        run_dir
        / "baseline_metrics.json"
    ).write_text(
        json.dumps(
            baseline,
            indent=2,
        )
        + "\n"
    )

    torch.save(
        {
            "epoch":
                int(
                    ckpt.get(
                        "epoch",
                        0,
                    )
                ),

            "model":
                model.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "pilot_val":
                pilot_val,

            "open_val":
                open_val,

            "insertion_val":
                insertion_val,

            "combined_score":
                best_score,

            "source_checkpoint":
                str(resume_path),

            "source":
                "V4 baseline V3 weights",
        },
        run_dir / "best.pt",
    )

    print()
    print(
        "BASE pilot     =",
        f"{pilot_base:.6f}",
    )

    print(
        "BASE open      =",
        f"{open_base:.6f}",
    )

    print(
        "BASE insertion =",
        f"{insertion_base:.6f}",
    )

    print(
        "BASE combined  =",
        f"{best_score:.6f}",
    )

    print(
        "pilot floor    =",
        f"{pilot_floor:.6f}",
    )

    print(
        "open floor     =",
        f"{open_floor:.6f}",
    )

    print()

    start_epoch = int(
        ckpt.get(
            "epoch",
            0,
        )
    )

    best_epoch = start_epoch

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

            image = (
                batch[
                    "image"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            target = (
                batch[
                    "mask"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with (
                torch.cuda.amp.autocast(
                    enabled=(
                        device.type
                        == "cuda"
                    )
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
                float(
                    loss.item()
                )
                * bs
            )

            seen += bs

        train_loss = (
            running_loss
            / max(
                seen,
                1,
            )
        )

        pilot_val = validate_views(
            model,
            pilot_root,
            pilot_split["val"],
            CAMERAS,
            device,
            tile=args.crop_size,
            stride=args.val_stride,
        )

        open_val = validate_views(
            model,
            open_root,
            open_split["val"],
            ("right_wrist",),
            device,
            tile=args.crop_size,
            stride=args.val_stride,
        )

        insertion_val = validate_views(
            model,
            insertion_root,
            insertion_split["val"],
            CAMERAS,
            device,
            tile=args.crop_size,
            stride=args.val_stride,
        )

        pilot_mean = mean_fg(
            pilot_val
        )

        open_mean = mean_fg(
            open_val
        )

        insertion_mean = mean_fg(
            insertion_val
        )

        score = (
            args.pilot_score_weight
            * pilot_mean
            +
            args.open_score_weight
            * open_mean
            +
            args.insertion_score_weight
            * insertion_mean
        )

        pilot_guard_ok = (
            pilot_mean
            >= pilot_floor
        )

        open_guard_ok = (
            open_mean
            >= open_floor
        )

        improved = (
            pilot_guard_ok
            and
            open_guard_ok
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
            f"pilot={pilot_mean:.4f} "
            f"open={open_mean:.4f} "
            f"insert={insertion_mean:.4f} "
            f"score={score:.4f} "
            f"guards="
            f"{int(pilot_guard_ok)}"
            f"{int(open_guard_ok)}"
            f"{flag}"
        )

        history.append({
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "pilot_val":
                pilot_val,

            "open_val":
                open_val,

            "insertion_val":
                insertion_val,

            "combined_score":
                score,

            "pilot_guard_ok":
                pilot_guard_ok,

            "open_guard_ok":
                open_guard_ok,
        })

        (
            run_dir
            / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            )
            + "\n"
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

                    "pilot_val":
                        pilot_val,

                    "open_val":
                        open_val,

                    "insertion_val":
                        insertion_val,

                    "combined_score":
                        score,

                    "source_checkpoint":
                        str(
                            resume_path
                        ),

                    "insertion_test_used":
                        0,
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

                        "pilot_val":
                            pilot_val,

                        "open_val":
                            open_val,

                        "insertion_val":
                            insertion_val,

                        "combined_score":
                            score,
                    },
                    indent=2,
                )
                + "\n"
            )

    print()
    print("=" * 78)
    print(
        "V4 BEST EPOCH =",
        best_epoch,
    )
    print(
        "V4 BEST SCORE =",
        f"{best_score:.6f}",
    )
    print(
        "CHECKPOINT =",
        run_dir / "best.pt",
    )
    print(
        "INSERTION TEST USED = 0"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
