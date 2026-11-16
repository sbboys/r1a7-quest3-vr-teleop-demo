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


def load_model_state(checkpoint):
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "model" in checkpoint:
        return checkpoint["model"]
    raise RuntimeError("Checkpoint has no model_state_dict/model")


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
    source_checkpoint,
    status,
):
    torch.save(
        {
            "architecture": "UNet_GroupNorm",
            "class_names": CLASS_NAMES,
            "epoch": int(epoch),
            "ignore_label": IGNORE_LABEL,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "source_checkpoint": str(source_checkpoint),
            "training_version": "hole_v3_newboard",
            "status": status,
        },
        path,
    )


def main():

    parser = argparse.ArgumentParser(
        description="Hole V3 training for NEW_BOARD_01 only."
    )

    parser.add_argument("--new-root", required=True)
    parser.add_argument("--new-split", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--run-dir", required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=512)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--new-repeat", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=44)

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    new_root = Path(args.new_root)
    new_session = new_root / "images"

    split_path = Path(args.new_split)
    resume_path = Path(args.resume)
    run_dir = Path(args.run_dir)

    if not new_session.is_dir():
        raise FileNotFoundError(new_session)

    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)

    split = json.loads(
        split_path.read_text(encoding="utf-8")
    )

    train_ids = split["train"]
    val_ids = split["val"]
    test_ids = split["test"]

    if (
        set(train_ids) & set(val_ids)
        or set(train_ids) & set(test_ids)
        or set(val_ids) & set(test_ids)
    ):
        raise RuntimeError("DATA_SPLIT_OVERLAP")

    train_samples = build_samples(
        new_root,
        new_session,
        train_ids,
    )

    val_samples = build_samples(
        new_root,
        new_session,
        val_ids,
    )

    test_samples = build_samples(
        new_root,
        new_session,
        test_ids,
    )

    print("=" * 76)
    print("R1-A7 HOLE V3 NEW BOARD TRAINING")
    print("=" * 76)

    print("new root =", new_root)
    print("new session =", new_session)
    print("train groups =", len(train_ids))
    print("train images =", len(train_samples))
    print("val groups =", len(val_ids))
    print("val images =", len(val_samples))
    print("test groups RESERVED =", len(test_ids))
    print("test images RESERVED =", len(test_samples))
    print("old board replay = DISABLED")

    if args.preflight_only:
        print("PREFLIGHT_RESULT = PASS")
        return

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = UNet(
        in_channels=3,
        num_classes=3,
        base=32,
    ).to(device)

    checkpoint = torch.load(
        resume_path,
        map_location=device,
    )

    model.load_state_dict(
        load_model_state(checkpoint),
        strict=True,
    )

    source_epoch = int(
        checkpoint.get("epoch", 0)
    )

    class_weights = torch.tensor(
        [0.20, 1.00, 5.00],
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    use_amp = device.type == "cuda"

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    # Only NEW_BOARD_01 train samples enter the training loader.
    repeated_samples = (
        train_samples * args.new_repeat
    )

    train_dataset = HoleCropDataset(
        repeated_samples,
        crop_size=args.crop_size,
        repeats=1,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_amp,
        drop_last=True,
    )

    if len(loader) == 0:
        raise RuntimeError("EMPTY_TRAIN_LOADER")

    print("device =", device)
    print("source checkpoint =", resume_path)
    print("source epoch =", source_epoch)
    print("new repeat =", args.new_repeat)
    print("training items =", len(train_dataset))
    print("batches per epoch =", len(loader))
    print("old data replay = 0")
    print("test used for training = 0")
    print()

    # Baseline: Hole V2 on NEW_BOARD_01 validation set.
    print("[BASELINE] Hole V2 on new-board validation")

    baseline_val = validate(
        model,
        val_samples,
        device,
    )

    best_score = float(baseline_val["score"])
    best_epoch = source_epoch
    best_source = "hole_v2_initialization"

    baseline = {
        "source_checkpoint": str(resume_path),
        "source_epoch": source_epoch,
        "new_board_val": baseline_val,
        "new_test_used": 0,
        "old_board_replay": False,
    }

    (run_dir / "baseline_metrics.json").write_text(
        json.dumps(baseline, indent=2) + "\n",
        encoding="utf-8",
    )

    # Keep the V2 initialization as a fallback candidate.
    save_checkpoint(
        run_dir / "best.pt",
        model,
        optimizer,
        source_epoch,
        baseline,
        resume_path,
        best_source,
    )

    history = []

    for local_epoch in range(1, args.epochs + 1):

        epoch = source_epoch + local_epoch

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

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):

                logits = model(image)

                ce = ce_loss(
                    logits,
                    target,
                )

                dice = foreground_dice_loss(
                    logits,
                    target,
                )

                loss = 0.5 * ce + 0.5 * dice

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"NON_FINITE_LOSS epoch={epoch}"
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = image.shape[0]

            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        scheduler.step()

        train_loss = running_loss / max(seen, 1)

        val_metrics = validate(
            model,
            val_samples,
            device,
        )

        val_score = float(val_metrics["score"])

        improved = val_score > best_score + 1e-8

        if improved:
            best_score = val_score
            best_epoch = epoch
            best_source = "new_board_training"

        row = {
            "epoch": epoch,
            "local_epoch": local_epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "new_board_val": val_metrics,
            "best_score": best_score,
            "improved": improved,
        }

        history.append(row)

        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

        # Save the latest model every epoch for recovery.
        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            epoch,
            row,
            resume_path,
            "last_epoch",
        )

        if improved:

            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                epoch,
                row,
                resume_path,
                "best_new_board_validation",
            )

            (run_dir / "best_metrics.json").write_text(
                json.dumps(row, indent=2) + "\n",
                encoding="utf-8",
            )

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.6f} "
            f"new_val={val_score:.6f} "
            f"best={best_score:.6f} "
            f"{'*BEST*' if improved else ''}",
            flush=True,
        )

    # Final evaluation: load selected checkpoint and evaluate test once.
    best_checkpoint = torch.load(
        run_dir / "best.pt",
        map_location=device,
    )

    model.load_state_dict(
        load_model_state(best_checkpoint),
        strict=True,
    )

    print()
    print("[FINAL] NEW_BOARD_01 reserved test evaluation")

    test_metrics = validate(
        model,
        test_samples,
        device,
    )

    test_result = {
        "dataset": "NEW_BOARD_01",
        "checkpoint": str(run_dir / "best.pt"),
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
        "same_scene_warning": split.get(
            "same_scene_warning",
            "Check split metadata.",
        ),
    }

    (run_dir / "test_metrics.json").write_text(
        json.dumps(test_result, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "model": "Hole V3 NEW_BOARD_01",
        "source_checkpoint": str(resume_path),
        "source_epoch": source_epoch,
        "best_epoch": best_epoch,
        "best_source": best_source,
        "best_validation_score": best_score,
        "final_test_score": test_metrics["score"],
        "epochs_requested": args.epochs,
        "train_groups": len(train_ids),
        "train_views": len(train_samples),
        "validation_groups": len(val_ids),
        "validation_views": len(val_samples),
        "test_groups": len(test_ids),
        "test_views": len(test_samples),
        "old_data_replay": False,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
    }

    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("HOLE V3 TRAINING FINISHED")
    print("BEST EPOCH =", best_epoch)
    print("BEST SOURCE =", best_source)
    print("BEST VAL SCORE =", f"{best_score:.6f}")
    print("TEST SCORE =", f"{test_metrics['score']:.6f}")
    print("CHECKPOINT =", run_dir / "best.pt")
    print("=" * 76)


if __name__ == "__main__":
    main()
