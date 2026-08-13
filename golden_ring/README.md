# Golden-ring U-Net segmentation

This folder trains a binary U-Net for corneal arcus (golden-ring) segmentation.

The expected dataset layout is:

```text
D:\AAA_DataRepo\golden_ring\ring
|-- images
|   |-- LDown_T1.jpg
|   `-- ...
`-- masks
    |-- LDown_T1.png
    `-- ...
```

An image without a same-stem file in `masks` is treated as a valid negative sample with an all-zero mask. The split is performed by subject (`T1`, `TJ1`, and so on), so views of one subject never occur in different splits.

## Train

Run from this folder:

```powershell
cd D:\AAA_CodeRepo\Seg-X-Net\golden_ring
python train.py
```

The default input size is `512x288`, preserving the source 16:9 aspect ratio. Checkpoints and logs are written to `outputs`:

- `outputs/best.pt`: checkpoint with the best validation Dice.
- `outputs/last.pt`: latest checkpoint for resuming.
- `outputs/history.jsonl`: per-epoch training and validation metrics.
- `outputs/split_subjects.json`: reproducible subject split.

Resume training:

```powershell
python train.py --resume outputs\last.pt
```

If GPU memory is insufficient, lower the batch size:

```powershell
python train.py --batch-size 2
```

## Evaluate

```powershell
python evaluate.py --checkpoint outputs\best.pt --split test
```

This reports Dice, mean per-image Dice, IoU, precision, and recall. Empty ground-truth images are included, so false-positive golden-ring predictions are penalized.
`negative_image_accuracy` is the fraction of no-golden-ring images for which the predicted mask is also completely empty. The best checkpoint is selected by mean per-image Dice rather than foreground-only global Dice.

## Predict

Predict a directory:

```powershell
python predict.py --checkpoint outputs\best.pt --input D:\path\to\images
```

Predict one image:

```powershell
python predict.py --checkpoint outputs\best.pt --input D:\path\to\image.jpg
```

Binary masks are written to `predict_results\masks`; red overlays are written to `predict_results\overlays`.

Prediction includes a post-filter calibrated from the current healthy and ring results. It removes small components, rejects thin arcs, and uses agreement between different gaze views of the same subject. This suppresses the thin false golden rings commonly predicted on healthy eyes. To save the unfiltered model output, use:

```powershell
python predict.py --checkpoint outputs\best.pt --input D:\path\to\images --no-post-filter
```
