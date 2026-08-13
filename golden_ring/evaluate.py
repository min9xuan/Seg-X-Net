from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from golden_ring_unet import (
    BinaryMetrics,
    GoldenRingDataset,
    discover_samples,
    load_checkpoint,
    samples_for_subjects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate golden-ring segmentation")
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    parser.add_argument("--data-root", default=r"D:\AAA_DataRepo\golden_ring\ring")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if "split" not in checkpoint:
        raise ValueError("Checkpoint does not contain the subject split")
    config = checkpoint.get("config", {})
    height = int(config.get("height", 288))
    width = int(config.get("width", 512))
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", 0.5))

    samples = discover_samples(args.data_root)
    selected = samples_for_subjects(samples, checkpoint["split"][args.split])
    dataset = GoldenRingDataset(selected, (height, width), augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    metrics = BinaryMetrics()
    with torch.inference_mode():
        for images, masks, _ in tqdm(loader, desc=f"evaluate {args.split}", ncols=100):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            metrics.update(torch.sigmoid(model(images)), masks, threshold)

    result = metrics.compute()
    print(f"Split: {args.split} ({len(dataset)} images), threshold: {threshold:.3f}")
    for name, value in result.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()

