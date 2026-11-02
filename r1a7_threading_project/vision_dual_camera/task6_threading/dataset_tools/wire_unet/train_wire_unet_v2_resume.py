#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from overfit_wire_unet import (
    UNet,
    foreground_dice_loss,
    seed_all,
)

from train_wire_unet_v1 import (
    TrainDataset,
    validate,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--resume",
        required=True,
    )

    parser.add_argument(
        "--extra-epochs",
        type=int,
        default=50,
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
        default=5e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    # Different from V1, but deterministic.
    seed_all(43)

    root = Path(args.dataset)
    resume_path = Path(args.resume)

    if not resume_path.exists():
        raise FileNotFoundError(
            resume_path
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

    ckpt = torch.load(
        resume_path,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    # Resume AdamW moments from V1.
    if "optimizer" in ckpt:
        optimizer.load_state_dict(
            ckpt["optimizer"]
        )

    # Override old LR after loading optimizer state.
    for group in optimizer.param_groups:
        group["lr"] = args.lr
        group["weight_decay"] = (
            args.weight_decay
        )

    start_epoch = int(
        ckpt.get("epoch", 0)
    )

    if "val" not in ckpt:
        raise RuntimeError(
            "resume checkpoint has no val metrics"
        )

    baseline_val = ckpt["val"]

    baseline_best = float(
        baseline_val[
            "overall"
        ][
            "mean_fg_dice"
        ]
    )

    run_dir = (
        root
        / "runs"
        / "unet_train_val_v2_resume"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve V1 best as V2 baseline.
    shutil.copy2(
        resume_path,
        run_dir / "best.pt",
    )

    (
        run_dir
        / "baseline_resume_metrics.json"
    ).write_text(
        json.dumps(
            {
                "source_checkpoint":
                    str(resume_path),
                "source_epoch":
                    start_epoch,
                "val":
                    baseline_val,
            },
            indent=2,
        )
    )

    (
        run_dir
        / "best_metrics.json"
    ).write_text(
        json.dumps(
            {
                "epoch":
                    start_epoch,
                "val":
                    baseline_val,
                "source":
                    "V1 resume baseline",
            },
            indent=2,
        )
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=(
                device.type
                == "cuda"
            )
        )
    )

    print("=" * 68)
    print(
        "R1-A7 WIRE U-NET V2 RESUME"
    )
    print("=" * 68)

    print(
        "device              =",
        device,
    )

    print(
        "resume checkpoint   =",
        resume_path,
    )

    print(
        "resume epoch        =",
        start_epoch,
    )

    print(
        "baseline val mean   =",
        f"{baseline_best:.6f}",
    )

    print(
        "train masks         =",
        len(train_dataset),
    )

    print(
        "val masks           =",
        len(split["val"]) * 3,
    )

    print(
        "test masks USED     = 0"
    )

    print(
        "learning rate       =",
        args.lr,
    )

    print(
        "extra epochs        =",
        args.extra_epochs,
    )

    print(
        "final epoch target  =",
        start_epoch
        + args.extra_epochs,
    )

    print(
        "class weights       =",
        weights,
    )

    print()

    best = baseline_best
    best_epoch = start_epoch
    history = []

    for local_epoch in range(
        1,
        args.extra_epochs + 1,
    ):

        epoch = (
            start_epoch
            + local_epoch
        )

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

                dice_loss = (
                    foreground_dice_loss(
                        logits,
                        target,
                    )
                )

                loss = (
                    0.5 * ce
                    + 0.5 * dice_loss
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
        ][
            "mean_fg_dice"
        ]

        improved = (
            mean_fg > best
        )

        flag = (
            " *BEST*"
            if improved
            else ""
        )

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.6f} "
            f"VAL_wire_dice="
            f"{wire['dice']:.4f} "
            f"VAL_tool_dice="
            f"{tool['dice']:.4f} "
            f"VAL_mean_fg="
            f"{mean_fg:.4f}"
            f"{flag}"
        )

        history.append(
            {
                "epoch":
                    epoch,
                "train_loss":
                    train_loss,
                "val":
                    val,
            }
        )

        if improved:

            best = mean_fg
            best_epoch = epoch

            torch.save(
                {
                    "epoch":
                        epoch,
                    "model":
                        model.state_dict(),
                    "optimizer":
                        optimizer.state_dict(),
                    "val":
                        val,
                    "class_weights":
                        weights,
                    "crop_size":
                        args.crop_size,
                    "val_stride":
                        args.val_stride,
                    "learning_rate":
                        args.lr,
                    "resumed_from":
                        str(resume_path),
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
                        "val":
                            val,
                        "resumed_from":
                            str(
                                resume_path
                            ),
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
    print("=" * 68)
    print(
        "V1 BASELINE EPOCH =",
        start_epoch,
    )

    print(
        "V1 BASELINE MEAN  =",
        f"{baseline_best:.6f}",
    )

    print(
        "V2 BEST EPOCH     =",
        best_epoch,
    )

    print(
        "V2 BEST VAL MEAN  =",
        f"{best:.6f}",
    )

    print(
        "DELTA             =",
        f"{best - baseline_best:+.6f}",
    )

    print(
        "checkpoint        =",
        run_dir / "best.pt",
    )
    print("=" * 68)

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
