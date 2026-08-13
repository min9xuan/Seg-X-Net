from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path | None
    subject: str


def subject_from_stem(stem: str) -> str:
    match = re.match(r"^[^_]+_(.+)$", stem)
    return match.group(1) if match else stem


def discover_samples(data_root: str | Path) -> list[Sample]:
    root = Path(data_root)
    image_dir = root / "images"
    mask_dir = root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected {image_dir} and {mask_dir}")

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    masks = {
        p.stem: p
        for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    samples = [Sample(p, masks.get(p.stem), subject_from_stem(p.stem)) for p in images]
    if not samples:
        raise RuntimeError(f"No images found in {image_dir}")

    image_stems = {p.stem for p in images}
    extra_masks = sorted(set(masks) - image_stems)
    if extra_masks:
        raise RuntimeError(f"Masks without matching images: {extra_masks[:10]}")
    return samples


def split_subjects(
    samples: Sequence[Sample], seed: int = 42, val_ratio: float = 0.15, test_ratio: float = 0.15
) -> dict[str, list[str]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")
    subjects = sorted({sample.subject for sample in samples})
    random.Random(seed).shuffle(subjects)
    n_test = round(len(subjects) * test_ratio)
    n_val = round(len(subjects) * val_ratio)
    return {
        "train": sorted(subjects[n_test + n_val :]),
        "val": sorted(subjects[n_test : n_test + n_val]),
        "test": sorted(subjects[:n_test]),
    }


def samples_for_subjects(samples: Sequence[Sample], subjects: Iterable[str]) -> list[Sample]:
    selected = set(subjects)
    return [sample for sample in samples if sample.subject in selected]


def save_split(path: str | Path, split: dict[str, list[str]]) -> None:
    Path(path).write_text(json.dumps(split, indent=2, ensure_ascii=True), encoding="utf-8")


class GoldenRingDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: tuple[int, int] = (288, 512),
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.height, self.width = image_size
        self.augment = augment
        self.color_jitter = ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pair(self, sample: Sample) -> tuple[Image.Image, Image.Image]:
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        if sample.mask is None:
            mask = Image.new("L", image.size, 0)
        else:
            with Image.open(sample.mask) as source:
                mask = source.convert("L")
            if mask.size != image.size:
                raise ValueError(f"Size mismatch: {sample.image.name} and {sample.mask.name}")

        image = TF.resize(image, [self.height, self.width], InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, [self.height, self.width], InterpolationMode.NEAREST)
        return image, mask

    def _augment_pair(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        angle = random.uniform(-7.0, 7.0)
        translate = (
            int(random.uniform(-0.03, 0.03) * self.width),
            int(random.uniform(-0.03, 0.03) * self.height),
        )
        scale = random.uniform(0.94, 1.06)
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        return self.color_jitter(image), mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]
        image, mask = self._load_pair(sample)
        if self.augment:
            image, mask = self._augment_pair(image, mask)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=[0.5] * 3, std=[0.5] * 3)
        mask_tensor = (TF.pil_to_tensor(mask).float() > 0).float()
        return image_tensor, mask_tensor, sample.image.name


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels * (2**i) for i in range(5)]
        self.enc1 = DoubleConv(3, channels[0])
        self.enc2 = DoubleConv(channels[0], channels[1])
        self.enc3 = DoubleConv(channels[1], channels[2])
        self.enc4 = DoubleConv(channels[2], channels[3])
        self.bottleneck = DoubleConv(channels[3], channels[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.dec4 = DoubleConv(channels[4], channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3 = DoubleConv(channels[3], channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2 = DoubleConv(channels[2], channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec1 = DoubleConv(channels[1], channels[0])
        self.output = nn.Conv2d(channels[0], 1, 1)

    @staticmethod
    def _join(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, up], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        center = self.bottleneck(self.pool(e4))
        d4 = self.dec4(self._join(self.up4(center), e4))
        d3 = self.dec3(self._join(self.up3(d4), e3))
        d2 = self.dec2(self._join(self.up2(d3), e2))
        d1 = self.dec1(self._join(self.up1(d2), e1))
        return self.output(d1)


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 5.0, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        probability = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probability * target).sum(dims)
        denominator = probability.sum(dims) + target.sum(dims)
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


class BinaryMetrics:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0.0
        self.image_dice: list[float] = []
        self.negative_images = 0
        self.clean_negative_images = 0

    def update(self, probability: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> None:
        prediction = probability >= threshold
        truth = target > 0.5
        dims = (1, 2, 3)
        tp = (prediction & truth).sum(dims).double()
        fp = (prediction & ~truth).sum(dims).double()
        fn = (~prediction & truth).sum(dims).double()
        self.tp += tp.sum().item()
        self.fp += fp.sum().item()
        self.fn += fn.sum().item()
        dice = (2.0 * tp + 1.0) / (2.0 * tp + fp + fn + 1.0)
        self.image_dice.extend(dice.cpu().tolist())
        negative = truth.sum(dims) == 0
        clean_negative = negative & (prediction.sum(dims) == 0)
        self.negative_images += int(negative.sum().item())
        self.clean_negative_images += int(clean_negative.sum().item())

    def compute(self) -> dict[str, float]:
        eps = 1e-7
        return {
            "dice": 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn + eps),
            "mean_image_dice": float(np.mean(self.image_dice)) if self.image_dice else 0.0,
            "iou": self.tp / (self.tp + self.fp + self.fn + eps),
            "precision": self.tp / (self.tp + self.fp + eps),
            "recall": self.tp / (self.tp + self.fn + eps),
            "negative_image_accuracy": self.clean_negative_images / max(1, self.negative_images),
        }


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[UNet, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Expected a checkpoint created by golden_ring/train.py")
    config = checkpoint.get("config", {})
    model = UNet(base_channels=int(config.get("base_channels", 32))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
