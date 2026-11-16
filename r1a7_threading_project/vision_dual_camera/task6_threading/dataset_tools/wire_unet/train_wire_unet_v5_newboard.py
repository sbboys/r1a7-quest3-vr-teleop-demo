#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_wire_unet_v4_insertion_joint import (
    CAMERAS,
    UNet,
    MixedWireDataset,
    make_items,
    validate_views,
    mean_fg,
    foreground_dice_loss,
    seed_all,
)


def load_state(ckpt):
    if "model" in ckpt:
        return ckpt["model"]
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    raise RuntimeError("checkpoint has no model/model_state_dict")


def save_ckpt(
    path,
    model,
    optimizer,
    epoch,
    metrics,
    source_checkpoint,
    source,
):
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "source_checkpoint": str(source_checkpoint),
            "source": source,
            "training_version": "wire_v5_newboard",
        },
        path,
    )


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--pilot-root",
        default="/home/robot/r1a7_datasets/wire_unet/pilot_001",
    )

    p.add_argument(
        "--open-root",
        default="/home/robot/r1a7_datasets/wire_unet/right_wrist_open_001",
    )

    p.add_argument(
        "--insertion-root",
        default=(
            "/home/robot/r1a7_datasets/wire_unet/"
            "insertion_joint_001/training_runtime"
        ),
    )

    p.add_argument("--newboard-root", required=True)
    p.add_argument("--newboard-split", required=True)
    p.add_argument("--resume", required=True)
    p.add_argument("--run-dir", required=True)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--val-stride", type=int, default=384)

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--pilot-repeat", type=int, default=1)
    p.add_argument("--open-repeat", type=int, default=1)
    p.add_argument("--insertion-repeat", type=int, default=3)
    p.add_argument("--newboard-repeat", type=int, default=3)

    p.add_argument("--pilot-score-weight", type=float, default=0.15)
    p.add_argument("--open-score-weight", type=float, default=0.20)
    p.add_argument("--insertion-score-weight", type=float, default=0.30)
    p.add_argument("--newboard-score-weight", type=float, default=0.35)

    p.add_argument("--pilot-max-drop", type=float, default=0.030)
    p.add_argument("--open-max-drop", type=float, default=0.030)
    p.add_argument("--insertion-max-drop", type=float, default=0.030)
    p.add_argument("--newboard-max-drop", type=float, default=0.000)

    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--preflight-only", action="store_true")

    args = p.parse_args()

    seed_all(args.seed)

    pilot_root = Path(args.pilot_root)
    open_root = Path(args.open_root)
    insertion_root = Path(args.insertion_root)
    newboard_root = Path(args.newboard_root)

    split_path = Path(args.newboard_split)
    resume_path = Path(args.resume)
    run_dir = Path(args.run_dir)

    pilot_split = json.loads(
        (pilot_root / "splits_sample_level_seed42.json").read_text()
    )

    open_split = json.loads(
        (open_root / "splits_right_wrist_open_grouped_v1.json").read_text()
    )

    insertion_split = json.loads(
        (
            insertion_root
            / "splits_insertion_joint_grouped_seed44_v1.json"
        ).read_text()
    )

    new_split = json.loads(
        split_path.read_text(encoding="utf-8")
    )

    train_ids = set(new_split["train"])
    val_ids = set(new_split["val"])
    test_ids = set(new_split["test"])

    if (
        train_ids & val_ids
        or train_ids & test_ids
        or val_ids & test_ids
    ):
        raise RuntimeError("NEWBOARD_SPLIT_OVERLAP")

    # Verify every new-board view before any training starts.
    for split_name in ("train", "val", "test"):
        for sid in new_split[split_name]:
            for cam in CAMERAS:
                image = (
                    newboard_root
                    / "images"
                    / cam
                    / f"{sid}.png"
                )
                mask = (
                    newboard_root
                    / "masks"
                    / cam
                    / f"{sid}.png"
                )

                if not image.is_file():
                    raise FileNotFoundError(image)

                if not mask.is_file():
                    raise FileNotFoundError(mask)

    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)

    print("=" * 78)
    print("R1-A7 WIRE U-NET V5 NEW-BOARD MIXED REPLAY")
    print("=" * 78)

    print("pilot repeat =", args.pilot_repeat)
    print("open repeat =", args.open_repeat)
    print("insertion repeat =", args.insertion_repeat)
    print("newboard repeat =", args.newboard_repeat)

    print(
        "newboard train/val/test groups =",
        len(new_split["train"]),
        len(new_split["val"]),
        len(new_split["test"]),
    )

    print(
        "newboard train/val/test images =",
        len(new_split["train"]) * len(CAMERAS),
        len(new_split["val"]) * len(CAMERAS),
        len(new_split["test"]) * len(CAMERAS),
    )

    print("resume =", resume_path)

    if args.preflight_only:
        print("PREFLIGHT_RESULT = PASS")
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    stats = json.loads(
        (
            pilot_root
            / "train_class_statistics.json"
        ).read_text()
    )

    weights = stats[
        "candidate_weights"
    ][
        "sqrt_inverse_clipped_normalized"
    ]

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

    items += make_items(
        newboard_root,
        new_split["train"],
        CAMERAS,
        "newboard",
        args.newboard_repeat,
    )

    train_dataset = MixedWireDataset(
        items,
        crop_size=args.crop_size,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
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
        load_state(ckpt),
        strict=True,
    )

    start_epoch = int(
        ckpt.get("epoch", 0)
    )

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

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda")
    )

    print("device =", device)
    print("source epoch =", start_epoch)
    print("total train items =", len(train_dataset))
    print("batches/epoch =", len(loader))
    print("NEWBOARD TEST USED FOR TRAINING = 0")
    print()

    print("[BASELINE] pilot")
    pilot_val = validate_views(
        model,
        pilot_root,
        pilot_split["val"],
        CAMERAS,
        device,
        tile=args.crop_size,
        stride=args.val_stride,
    )

    print("[BASELINE] open")
    open_val = validate_views(
        model,
        open_root,
        open_split["val"],
        ("right_wrist",),
        device,
        tile=args.crop_size,
        stride=args.val_stride,
    )

    print("[BASELINE] insertion")
    insertion_val = validate_views(
        model,
        insertion_root,
        insertion_split["val"],
        CAMERAS,
        device,
        tile=args.crop_size,
        stride=args.val_stride,
    )

    print("[BASELINE] NEW_BOARD_01")
    newboard_val = validate_views(
        model,
        newboard_root,
        new_split["val"],
        CAMERAS,
        device,
        tile=args.crop_size,
        stride=args.val_stride,
    )

    pilot_base = mean_fg(pilot_val)
    open_base = mean_fg(open_val)
    insertion_base = mean_fg(insertion_val)
    newboard_base = mean_fg(newboard_val)

    pilot_floor = pilot_base - args.pilot_max_drop
    open_floor = open_base - args.open_max_drop
    insertion_floor = (
        insertion_base
        - args.insertion_max_drop
    )
    newboard_floor = (
        newboard_base
        - args.newboard_max_drop
    )

    weight_sum = (
        args.pilot_score_weight
        + args.open_score_weight
        + args.insertion_score_weight
        + args.newboard_score_weight
    )

    if weight_sum <= 0:
        raise RuntimeError("INVALID_SCORE_WEIGHT_SUM")

    def combined_score(
        pilot_score,
        open_score,
        insertion_score,
        newboard_score,
    ):
        return (
            args.pilot_score_weight * pilot_score
            + args.open_score_weight * open_score
            + args.insertion_score_weight * insertion_score
            + args.newboard_score_weight * newboard_score
        ) / weight_sum

    best_score = combined_score(
        pilot_base,
        open_base,
        insertion_base,
        newboard_base,
    )

    best_epoch = start_epoch
    best_source = "wire_v4_initialization"

    baseline = {
        "source_checkpoint": str(resume_path),
        "source_epoch": start_epoch,
        "pilot_val": pilot_val,
        "open_val": open_val,
        "insertion_val": insertion_val,
        "newboard_val": newboard_val,
        "pilot_floor": pilot_floor,
        "open_floor": open_floor,
        "insertion_floor": insertion_floor,
        "newboard_floor": newboard_floor,
        "combined_score": best_score,
        "newboard_test_used": 0,
    }

    (
        run_dir / "baseline_metrics.json"
    ).write_text(
        json.dumps(
            baseline,
            indent=2,
        ) + "\n"
    )

    save_ckpt(
        run_dir / "best.pt",
        model,
        optimizer,
        start_epoch,
        baseline,
        resume_path,
        "wire_v4_initialization",
    )

    (
        run_dir / "best_metrics.json"
    ).write_text(
        json.dumps(
            {
                "epoch": start_epoch,
                "source": best_source,
                **baseline,
            },
            indent=2,
        ) + "\n"
    )

    print()
    print("BASE pilot     =", f"{pilot_base:.6f}")
    print("BASE open      =", f"{open_base:.6f}")
    print("BASE insertion =", f"{insertion_base:.6f}")
    print("BASE newboard  =", f"{newboard_base:.6f}")
    print("BASE combined  =", f"{best_score:.6f}")

    print("pilot floor     =", f"{pilot_floor:.6f}")
    print("open floor      =", f"{open_floor:.6f}")
    print("insertion floor =", f"{insertion_floor:.6f}")
    print("newboard floor  =", f"{newboard_floor:.6f}")
    print()

    history = []

    for local_epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch = start_epoch + local_epoch

        model.train()

        running_loss = 0.0
        seen = 0

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
                enabled=(device.type == "cuda")
            ):

                logits = model(image)

                ce = F.cross_entropy(
                    logits,
                    target,
                    weight=class_weights,
                    ignore_index=255,
                )

                dice_loss = foreground_dice_loss(
                    logits,
                    target,
                )

                loss = (
                    0.5 * ce
                    + 0.5 * dice_loss
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"NON_FINITE_LOSS epoch={epoch}"
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = image.shape[0]

            running_loss += (
                float(loss.item()) * bs
            )
            seen += bs

        train_loss = (
            running_loss
            / max(seen, 1)
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

        newboard_val = validate_views(
            model,
            newboard_root,
            new_split["val"],
            CAMERAS,
            device,
            tile=args.crop_size,
            stride=args.val_stride,
        )

        pilot_mean = mean_fg(pilot_val)
        open_mean = mean_fg(open_val)
        insertion_mean = mean_fg(insertion_val)
        newboard_mean = mean_fg(newboard_val)

        score = combined_score(
            pilot_mean,
            open_mean,
            insertion_mean,
            newboard_mean,
        )

        pilot_ok = pilot_mean >= pilot_floor
        open_ok = open_mean >= open_floor
        insertion_ok = (
            insertion_mean >= insertion_floor
        )
        newboard_ok = (
            newboard_mean >= newboard_floor
        )

        improved = (
            pilot_ok
            and open_ok
            and insertion_ok
            and newboard_ok
            and score > best_score + 1e-8
        )

        if improved:
            best_score = score
            best_epoch = epoch
            best_source = "wire_v5_training"

        row = {
            "epoch": epoch,
            "local_epoch": local_epoch,
            "train_loss": train_loss,
            "pilot_val": pilot_val,
            "open_val": open_val,
            "insertion_val": insertion_val,
            "newboard_val": newboard_val,
            "combined_score": score,
            "pilot_guard_ok": pilot_ok,
            "open_guard_ok": open_ok,
            "insertion_guard_ok": insertion_ok,
            "newboard_guard_ok": newboard_ok,
            "improved": improved,
        }

        history.append(row)

        (
            run_dir / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            ) + "\n"
        )

        save_ckpt(
            run_dir / "last.pt",
            model,
            optimizer,
            epoch,
            row,
            resume_path,
            "last_epoch",
        )

        if improved:

            save_ckpt(
                run_dir / "best.pt",
                model,
                optimizer,
                epoch,
                row,
                resume_path,
                "wire_v5_best",
            )

            (
                run_dir / "best_metrics.json"
            ).write_text(
                json.dumps(
                    row,
                    indent=2,
                ) + "\n"
            )

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.6f} "
            f"pilot={pilot_mean:.4f} "
            f"open={open_mean:.4f} "
            f"insert={insertion_mean:.4f} "
            f"newboard={newboard_mean:.4f} "
            f"score={score:.4f} "
            f"guards="
            f"{int(pilot_ok)}"
            f"{int(open_ok)}"
            f"{int(insertion_ok)}"
            f"{int(newboard_ok)}"
            f"{' *BEST*' if improved else ''}",
            flush=True,
        )

    # Reserved new-board test: evaluated exactly once
    # after training and model selection.
    best_ckpt = torch.load(
        run_dir / "best.pt",
        map_location=device,
    )

    model.load_state_dict(
        load_state(best_ckpt),
        strict=True,
    )

    print()
    print("[FINAL] NEW_BOARD_01 WIRE RESERVED TEST")

    newboard_test = validate_views(
        model,
        newboard_root,
        new_split["test"],
        CAMERAS,
        device,
        tile=args.crop_size,
        stride=args.val_stride,
    )

    test_result = {
        "dataset": "NEW_BOARD_01",
        "task": "wire_tweezer",
        "best_epoch": best_epoch,
        "best_source": best_source,
        "checkpoint": str(run_dir / "best.pt"),
        "newboard_test": newboard_test,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
    }

    (
        run_dir / "newboard_test_metrics.json"
    ).write_text(
        json.dumps(
            test_result,
            indent=2,
        ) + "\n"
    )

    summary = {
        "model": "Wire V5 NEW_BOARD_01",
        "source_checkpoint": str(resume_path),
        "source_epoch": start_epoch,
        "best_epoch": best_epoch,
        "best_source": best_source,
        "best_combined_score": best_score,
        "epochs_requested": args.epochs,
        "newboard_train_groups":
            len(new_split["train"]),
        "newboard_val_groups":
            len(new_split["val"]),
        "newboard_test_groups":
            len(new_split["test"]),
        "newboard_test_score":
            mean_fg(newboard_test),
        "newboard_test_used_for_training":
            False,
        "newboard_test_used_for_model_selection":
            False,
    }

    (
        run_dir / "training_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ) + "\n"
    )

    print()
    print("=" * 78)
    print("WIRE V5 TRAINING FINISHED")
    print("BEST EPOCH =", best_epoch)
    print("BEST SOURCE =", best_source)
    print("BEST SCORE =", f"{best_score:.6f}")
    print(
        "NEWBOARD TEST =",
        f"{mean_fg(newboard_test):.6f}",
    )
    print(
        "CHECKPOINT =",
        run_dir / "best.pt",
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
