from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from golden_ring_unet import (
    BCEDiceLoss,
    BinaryMetrics,
    GoldenRingDataset,
    UNet,
    discover_samples,
    samples_for_subjects,
    save_split,
    split_subjects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net for golden-ring segmentation")
    parser.add_argument("--data-root", default=r"D:\AAA_DataRepo\golden_ring\ring")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="", help="Path to last.pt or another training checkpoint")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, threshold, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    metrics = BinaryMetrics()
    total_loss = 0.0
    progress = tqdm(loader, desc="train" if training else "valid", leave=False, ncols=100)

    for images, masks, _ in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        metrics.update(torch.sigmoid(logits.detach()), masks, threshold)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    result = metrics.compute()
    result["loss"] = total_loss / max(1, len(loader.dataset))
    return result


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = discover_samples(args.data_root)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_config = resume_checkpoint.get("config", {})
        args.height = int(saved_config.get("height", args.height))
        args.width = int(saved_config.get("width", args.width))
        args.base_channels = int(saved_config.get("base_channels", args.base_channels))
    split = (
        resume_checkpoint["split"]
        if resume_checkpoint is not None and "split" in resume_checkpoint
        else split_subjects(samples, args.seed, args.val_ratio, args.test_ratio)
    )
    save_split(output_dir / "split_subjects.json", split)
    train_samples = samples_for_subjects(samples, split["train"])
    val_samples = samples_for_subjects(samples, split["val"])

    train_dataset = GoldenRingDataset(train_samples, (args.height, args.width), augment=True)
    val_dataset = GoldenRingDataset(val_samples, (args.height, args.width), augment=False)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = UNet(args.base_channels).to(device)
    criterion = BCEDiceLoss(args.pos_weight, args.bce_weight).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_score = 1, -1.0

    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", checkpoint.get("best_dice", -1.0)))

    config = vars(args).copy()
    print(f"Device: {device}")
    print(
        f"Images: {len(samples)} (with mask: {sum(s.mask is not None for s in samples)}, "
        f"negative: {sum(s.mask is None for s in samples)})"
    )
    print(
        f"Subjects train/val/test: {len(split['train'])}/{len(split['val'])}/{len(split['test'])}; "
        f"images train/val: {len(train_samples)}/{len(val_samples)}"
    )

    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        train_result = run_epoch(
            model, train_loader, criterion, device, args.threshold, optimizer=optimizer, scaler=scaler
        )
        with torch.no_grad():
            val_result = run_epoch(model, val_loader, criterion, device, args.threshold)
        selection_score = val_result["mean_image_dice"]
        scheduler.step(selection_score)

        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_result,
            "val": val_result,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": max(best_score, selection_score),
            "threshold": args.threshold,
            "split": split,
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if selection_score > best_score:
            best_score = selection_score
            checkpoint["best_score"] = best_score
            torch.save(checkpoint, output_dir / "best.pt")

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_result['loss']:.4f} dice {train_result['dice']:.4f} | "
            f"val loss {val_result['loss']:.4f} dice {val_result['dice']:.4f} "
            f"mean-image Dice {selection_score:.4f} neg-acc {val_result['negative_image_accuracy']:.4f} "
            f"| best {best_score:.4f}"
        )


if __name__ == "__main__":
    main()
