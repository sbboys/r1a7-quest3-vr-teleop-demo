#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from train_hole_board_unet_v1 import (
    CAMERAS,
    CLASS_NAMES,
    IGNORE_LABEL,
    UNet,
    HoleCropDataset,
    build_samples,
    foreground_dice_loss,
    validate,
    seed_everything,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--old-root",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/hole_tube_001"
        ),
    )

    parser.add_argument(
        "--old-session",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/hole_tube_001/"
            "images_raw/"
            "session_20261105_105208"
        ),
    )

    parser.add_argument(
        "--old-split",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/hole_tube_001/"
            "splits_hole_board_grouped_v1.json"
        ),
    )

    parser.add_argument(
        "--new-root",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/insertion_joint_001/"
            "training_runtime"
        ),
    )

    parser.add_argument(
        "--new-split",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/insertion_joint_001/"
            "training_runtime/"
            "splits_insertion_joint_grouped_seed44_v1.json"
        ),
    )

    parser.add_argument(
        "--resume",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/hole_tube_001/"
            "runs/hole_board_unet_v1/"
            "best.pt"
        ),
    )

    parser.add_argument(
        "--run-dir",
        default=(
            "/home/robot/r1a7_datasets/"
            "wire_unet/insertion_joint_001/"
            "training_runtime/runs/"
            "hole_v2_insertion_replay_seed44"
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
        "--old-repeat",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--new-repeat",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--old-score-weight",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--new-score-weight",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--old-max-drop",
        type=float,
        default=0.030,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=44,
    )

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    old_root = Path(
        args.old_root
    )

    old_session = Path(
        args.old_session
    )

    old_split_path = Path(
        args.old_split
    )

    new_root = Path(
        args.new_root
    )

    new_session = (
        new_root
        / "images"
    )

    new_split_path = Path(
        args.new_split
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

    old_split = json.loads(
        old_split_path.read_text()
    )

    new_split = json.loads(
        new_split_path.read_text()
    )

    old_train = build_samples(
        old_root,
        old_session,
        old_split["train"],
    )

    old_val = build_samples(
        old_root,
        old_session,
        old_split["val"],
    )

    new_train = build_samples(
        new_root,
        new_session,
        new_split["train"],
    )

    new_val = build_samples(
        new_root,
        new_session,
        new_split["val"],
    )

    mixed_samples = (
        old_train
        * int(
            args.old_repeat
        )
        +
        new_train
        * int(
            args.new_repeat
        )
    )

    train_dataset = HoleCropDataset(
        mixed_samples,
        crop_size=args.crop_size,
        repeats=1,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
        drop_last=True,
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

    if "model_state_dict" in ckpt:

        state = ckpt[
            "model_state_dict"
        ]

    elif "model" in ckpt:

        state = ckpt[
            "model"
        ]

    else:

        raise RuntimeError(
            "checkpoint has no "
            "model_state_dict/model"
        )

    model.load_state_dict(
        state
    )

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

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=(
                args.weight_decay
            ),
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )
    )

    use_amp = (
        device.type
        == "cuda"
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=use_amp
        )
    )

    print("=" * 76)
    print(
        "R1-A7 HOLE BOARD U-NET V2 "
        "INSERTION MIXED REPLAY"
    )
    print("=" * 76)

    print("device =", device)

    print(
        "old train unique =",
        len(old_train),
    )

    print(
        "new train unique =",
        len(new_train),
    )

    print(
        "old repeat =",
        args.old_repeat,
    )

    print(
        "new repeat =",
        args.new_repeat,
    )

    print(
        "mixed train items =",
        len(mixed_samples),
    )

    print(
        "new val images =",
        len(new_val),
    )

    print(
        "new test RESERVED =",
        len(
            new_split[
                "test"
            ]
        )
        * len(CAMERAS),
    )

    print(
        "resume =",
        resume_path,
    )

    print()

    print(
        "[BASELINE] old hole val"
    )

    old_metrics = validate(
        model,
        old_val,
        device,
    )

    print(
        "[BASELINE] insertion val"
    )

    new_metrics = validate(
        model,
        new_val,
        device,
    )

    old_base = float(
        old_metrics[
            "score"
        ]
    )

    new_base = float(
        new_metrics[
            "score"
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
            0,
        )
    )

    best_epoch = start_epoch

    baseline = {
        "source_checkpoint":
            str(resume_path),

        "source_epoch":
            start_epoch,

        "old_val":
            old_metrics,

        "new_val":
            new_metrics,

        "combined_score":
            best_score,

        "old_guard_floor":
            old_floor,

        "new_test_used":
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
            "architecture":
                "UNet_GroupNorm",

            "class_names":
                CLASS_NAMES,

            "epoch":
                start_epoch,

            "ignore_label":
                IGNORE_LABEL,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "metrics":
                baseline,

            "source_checkpoint":
                str(resume_path),
        },
        run_dir / "best.pt",
    )

    print(
        "BASE old score =",
        f"{old_base:.6f}",
    )

    print(
        "BASE new score =",
        f"{new_base:.6f}",
    )

    print(
        "BASE combined  =",
        f"{best_score:.6f}",
    )

    print(
        "OLD guard floor =",
        f"{old_floor:.6f}",
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

        for image, target in loader:

            image = image.to(
                device,
                non_blocking=True,
            )

            target = target.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with (
                torch.cuda.amp.autocast(
                    enabled=use_amp
                )
            ):

                logits = model(
                    image
                )

                ce = ce_loss(
                    logits,
                    target,
                )

                dice = (
                    foreground_dice_loss(
                        logits,
                        target,
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

        scheduler.step()

        train_loss = (
            running_loss
            / max(
                seen,
                1,
            )
        )

        old_metrics = validate(
            model,
            old_val,
            device,
        )

        new_metrics = validate(
            model,
            new_val,
            device,
        )

        old_score = float(
            old_metrics[
                "score"
            ]
        )

        new_score = float(
            new_metrics[
                "score"
            ]
        )

        score = (
            args.old_score_weight
            * old_score
            +
            args.new_score_weight
            * new_score
        )

        old_guard_ok = (
            old_score
            >= old_floor
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
            f"old={old_score:.4f} "
            f"new={new_score:.4f} "
            f"combined={score:.4f} "
            f"old_guard="
            f"{int(old_guard_ok)}"
            f"{flag}"
        )

        row = {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "learning_rate":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ],

            "old_val":
                old_metrics,

            "new_val":
                new_metrics,

            "combined_score":
                score,

            "old_guard_ok":
                old_guard_ok,
        }

        history.append(
            row
        )

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

            metrics = {
                "old_val":
                    old_metrics,

                "new_val":
                    new_metrics,

                "combined_score":
                    score,

                "old_guard_ok":
                    old_guard_ok,

                "new_test_used":
                    0,
            }

            torch.save(
                {
                    "architecture":
                        "UNet_GroupNorm",

                    "class_names":
                        CLASS_NAMES,

                    "epoch":
                        epoch,

                    "ignore_label":
                        IGNORE_LABEL,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "metrics":
                        metrics,

                    "source_checkpoint":
                        str(
                            resume_path
                        ),
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

                        **metrics,
                    },
                    indent=2,
                )
                + "\n"
            )

    print()
    print("=" * 76)
    print(
        "V2 BEST EPOCH =",
        best_epoch,
    )
    print(
        "V2 BEST SCORE =",
        f"{best_score:.6f}",
    )
    print(
        "CHECKPOINT =",
        run_dir / "best.pt",
    )
    print(
        "NEW TEST USED = 0"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
