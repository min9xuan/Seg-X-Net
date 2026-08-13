from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from golden_ring_unet import IMAGE_SUFFIXES, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict golden-ring masks with a trained U-Net")
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    parser.add_argument(
        "--input",
        default="D:/BaiduNetdiskDownload/all_pics/healthy",
        help="An image file or a directory",
    )
    parser.add_argument("--output-dir", default="predict_results/healthy")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--no-post-filter",
        action="store_true",
        help="Disable thickness and multi-view filtering",
    )
    return parser.parse_args()


def find_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def prepare(image: Image.Image, height: int, width: int) -> torch.Tensor:
    resized = TF.resize(image, [height, width], InterpolationMode.BILINEAR, antialias=True)
    tensor = TF.to_tensor(resized)
    return TF.normalize(tensor, mean=[0.5] * 3, std=[0.5] * 3)


def subject_key(stem: str) -> str:
    """Group gaze directions and calibrated copies belonging to one subject."""
    stem = re.sub(r"^(LDown|LLeft|LN|LRight|LUp|RDown|RLeft|RN|RRight|RUp)_", "", stem)
    return stem.removeprefix("calibrated_")


def largest_component_metrics(binary: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return largest component, area ratio, and scale-normalized thickness."""
    binary = (binary > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(binary), 0.0, 0.0

    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == label).astype(np.uint8)
    area = float(component.sum())
    contours, _ = cv2.findContours(component, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    area_ratio = area / float(component.size)
    equivalent_thickness = 2.0 * area / max(perimeter, 1.0)
    normalized_thickness = equivalent_thickness * (512.0 / component.shape[1])
    return component, area_ratio, normalized_thickness


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint.get("config", {})
    height = int(config.get("height", 288))
    width = int(config.get("width", 512))
    checkpoint_threshold = float(checkpoint.get("threshold", 0.5))
    default_threshold = checkpoint_threshold if args.no_post_filter else max(0.8, checkpoint_threshold)
    threshold = args.threshold if args.threshold is not None else default_threshold
    image_paths = find_images(Path(args.input))

    # Data-derived defaults from 630 labelled positive images, 140 unlabelled
    # ring-set negatives, and 336 available healthy predictions.
    strong_min_thickness = 10.0
    strong_min_area_ratio = 0.010
    strong_view_fraction = 0.34
    final_min_thickness = 4.0

    records = []
    with torch.inference_mode():
        for image_path in tqdm(image_paths, desc="predict", ncols=100):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            tensor = prepare(image, height, width).unsqueeze(0).to(device)
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
            binary = (probability >= threshold).astype(np.uint8)
            component, area_ratio, thickness = largest_component_metrics(binary)
            records.append(
                {
                    "path": image_path,
                    "component": component,
                    "raw_binary": binary if args.no_post_filter else None,
                    "area_ratio": area_ratio,
                    "thickness": thickness,
                    "subject": subject_key(image_path.stem),
                    "strong": thickness >= strong_min_thickness
                    and area_ratio >= strong_min_area_ratio,
                }
            )

    strong_views: dict[str, list[bool]] = {}
    for record in records:
        strong_views.setdefault(record["subject"], []).append(record["strong"])
    accepted_subjects = {
        subject
        for subject, views in strong_views.items()
        if float(np.mean(views)) >= strong_view_fraction
    }

    kept = 0
    for record in tqdm(records, desc="save", ncols=100):
        image_path = record["path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        accepted = args.no_post_filter or (
            record["subject"] in accepted_subjects
            and record["thickness"] >= final_min_thickness
        )
        if args.no_post_filter:
            final_binary = record["raw_binary"]
        else:
            final_binary = record["component"] if accepted else np.zeros_like(record["component"])
        kept += int(final_binary.any())
        mask = Image.fromarray(final_binary * 255).resize(image.size, Image.Resampling.NEAREST)
        mask.save(mask_dir / f"{image_path.stem}_golden_ring.png")

        image_array = np.asarray(image, dtype=np.float32)
        mask_array = np.asarray(mask) > 0
        color = np.zeros_like(image_array)
        color[..., 0] = 255
        overlay = image_array.copy()
        overlay[mask_array] = 0.65 * image_array[mask_array] + 0.35 * color[mask_array]
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
            overlay_dir / f"{image_path.stem}_overlay.jpg", quality=95
        )

    print(f"Saved {len(image_paths)} masks to {mask_dir}")
    print(f"Saved overlays to {overlay_dir}")
    print(f"Post-filter retained golden-ring masks for {kept}/{len(records)} images")


if __name__ == "__main__":
    main()
