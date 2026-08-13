import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_binary_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    return (mask > 127).astype(np.uint8)


def largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return mask.astype(np.uint8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


def fill_and_smooth(mask, close_kernel=9, open_kernel=5):
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    smoothed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, close_k)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, open_k)

    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return smoothed

    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return filled


def mask_center(mask):
    moments = cv2.moments(mask.astype(np.uint8))
    if abs(moments["m00"]) < 1e-6:
        return None
    return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)


def choose_center(iris_mask, pupil_mask):
    if pupil_mask is not None and pupil_mask.sum() > 20:
        center = mask_center(largest_component(pupil_mask))
        if center is not None:
            return center

    center = mask_center(iris_mask)
    if center is not None:
        return center

    ys, xs = np.where(iris_mask > 0)
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def odd_kernel_size(value, min_size=3, max_size=99):
    size = int(round(value))
    size = max(min_size, min(max_size, size))
    if size % 2 == 0:
        size += 1
    return size


def equivalent_radius(mask):
    area = float(mask.sum())
    if area <= 0:
        return 1.0
    return float(np.sqrt(area / np.pi))


def limit_addition_by_distance(base, candidate, max_addition_ratio):
    base_area = int(base.sum())
    addition = (candidate & (1 - base)).astype(np.uint8)
    max_addition = max(0, int(base_area * max_addition_ratio))
    if int(addition.sum()) <= max_addition:
        return np.maximum(base, addition).astype(np.uint8)

    distance_to_base = cv2.distanceTransform((1 - base).astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.where(addition > 0)
    order = np.argsort(distance_to_base[ys, xs])[:max_addition]
    limited = np.zeros_like(base, dtype=np.uint8)
    limited[ys[order], xs[order]] = 1
    return np.maximum(base, limited).astype(np.uint8)


def build_eye_region(sclera_mask, iris_mask, pupil_mask):
    eye = ((sclera_mask > 0) | (iris_mask > 0) | (pupil_mask > 0)).astype(np.uint8)
    if eye.sum() < 30:
        return eye

    # Keep this conservative: it is only a feasible eye boundary, not a source of
    # pixels to assign to iris.
    iris_radius = equivalent_radius(iris_mask)
    bridge_size = odd_kernel_size(iris_radius * 0.14, min_size=15, max_size=81)
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_size, bridge_size))
    eye = cv2.morphologyEx(eye, cv2.MORPH_CLOSE, bridge_kernel)
    eye = fill_and_smooth(eye, close_kernel=odd_kernel_size(iris_radius * 0.05, min_size=7, max_size=31), open_kernel=3)
    return largest_component(eye)


def assign_eye_gap_to_iris(iris_mask, sclera_mask, pupil_mask, eye_region):
    iris = (iris_mask > 0).astype(np.uint8)
    sclera = (sclera_mask > 0).astype(np.uint8)
    pupil = (pupil_mask > 0).astype(np.uint8)
    labeled = ((iris > 0) | (sclera > 0) | (pupil > 0)).astype(np.uint8)
    gap = ((eye_region > 0) & (labeled == 0)).astype(np.uint8)
    if gap.sum() == 0:
        return iris

    radius = equivalent_radius(iris)
    adjacency_size = odd_kernel_size(radius * 0.10, min_size=11, max_size=51)
    adjacency_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (adjacency_size, adjacency_size))
    iris_neighborhood = cv2.dilate(iris, adjacency_kernel)
    sclera_neighborhood = cv2.dilate(sclera, adjacency_kernel)

    h, w = gap.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gap, 8)
    iris_gap = np.zeros_like(gap, dtype=np.uint8)
    max_gap_area = max(100, int(iris.sum() * 0.35))
    bridge_band = ((gap > 0) & (iris_neighborhood > 0) & (sclera_neighborhood > 0)).astype(np.uint8)

    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > max_gap_area:
            # Large components are usually connected to background; only keep
            # their local iris-sclera bridge band.
            iris_gap[component & (bridge_band > 0)] = 1
            continue

        touches_iris = np.any(component & (iris_neighborhood > 0))
        touches_sclera = np.any(component & (sclera_neighborhood > 0))
        if touches_iris and touches_sclera:
            iris_gap[component] = 1

    return np.maximum(iris, iris_gap).astype(np.uint8)


def remove_edge_connected_additions(original, repaired, edge_margin_px=3):
    original = (original > 0).astype(np.uint8)
    repaired = (repaired > 0).astype(np.uint8)
    addition = ((repaired > 0) & (original == 0)).astype(np.uint8)
    if addition.sum() == 0:
        return repaired

    h, w = addition.shape
    edge_seed = np.zeros_like(addition, dtype=np.uint8)
    edge_seed[:edge_margin_px, :] = addition[:edge_margin_px, :]
    edge_seed[h - edge_margin_px :, :] = addition[h - edge_margin_px :, :]
    edge_seed[:, :edge_margin_px] = addition[:, :edge_margin_px]
    edge_seed[:, w - edge_margin_px :] = addition[:, w - edge_margin_px :]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(addition, 8)
    safe_addition = np.zeros_like(addition, dtype=np.uint8)
    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        edge_pixels = int(edge_seed[component].sum())
        if edge_pixels == 0:
            safe_addition[component] = 1
            continue

        # A tiny edge contact can happen when a valid repaired iris touches the
        # crop boundary. Remove the whole component only when the new region is
        # materially stuck to the image edge.
        if edge_pixels / max(1, area) <= 0.03:
            safe_addition[component & (edge_seed == 0)] = 1

    return np.maximum(original, safe_addition).astype(np.uint8)


def main_contour(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def fit_ellipse_mask(mask):
    contour = main_contour(mask)
    if contour is None or len(contour) < 5:
        return np.zeros_like(mask, dtype=np.uint8)

    ellipse = cv2.fitEllipse(contour)
    ellipse_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse, 1, -1)
    return ellipse_mask


def convex_hull_mask(mask):
    contour = main_contour(mask)
    if contour is None:
        return np.zeros_like(mask, dtype=np.uint8)

    hull = cv2.convexHull(contour)
    result = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(result, [hull], -1, 1, -1)
    return result


def smooth_contour_mask(mask, window_ratio=0.07):
    contour = main_contour(mask)
    if contour is None or len(contour) < 12:
        return mask.astype(np.uint8)

    points = contour[:, 0, :].astype(np.float32)
    window = int(round(len(points) * window_ratio))
    window = max(9, min(121, window))
    if window % 2 == 0:
        window += 1
    half = window // 2

    smoothed = np.zeros_like(points)
    for offset in range(-half, half + 1):
        smoothed += np.roll(points, offset, axis=0)
    smoothed /= float(window)

    h, w = mask.shape
    smoothed[:, 0] = np.clip(smoothed[:, 0], 0, w - 1)
    smoothed[:, 1] = np.clip(smoothed[:, 1], 0, h - 1)
    smoothed = np.round(smoothed).astype(np.int32)

    result = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(result, [smoothed], 1)
    return largest_component(result)


def boundary_defect_ratio(mask):
    contour = main_contour(mask)
    if contour is None:
        return 0.0

    area = float(cv2.contourArea(contour))
    if area <= 1.0:
        return 0.0

    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    if hull_area <= 1.0:
        return 0.0

    return max(0.0, min(1.0, (hull_area - area) / hull_area))


def circular_morph_close(values, window):
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    half = window // 2

    dilated = np.copy(values)
    for offset in range(-half, half + 1):
        dilated = np.maximum(dilated, np.roll(values, offset))

    closed = np.copy(dilated)
    for offset in range(-half, half + 1):
        closed = np.minimum(closed, np.roll(dilated, offset))
    return closed


def fill_missing_circular_values(values, valid):
    if valid.all():
        return values
    if not valid.any():
        return values

    x = np.arange(len(values))
    valid_x = x[valid]
    valid_y = values[valid]
    x_ext = np.concatenate([valid_x - len(values), valid_x, valid_x + len(values)])
    y_ext = np.concatenate([valid_y, valid_y, valid_y])
    return np.interp(x, x_ext, y_ext).astype(np.float32)


def polar_boundary_smooth_mask(mask, center, angle_bins=720):
    ys, xs = np.where(mask > 0)
    if len(xs) < 5 or center is None:
        return np.zeros_like(mask, dtype=np.uint8)

    cx, cy = float(center[0]), float(center[1])
    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy
    radii = np.sqrt(dx * dx + dy * dy)
    angles = (np.arctan2(dy, dx) + 2.0 * np.pi) % (2.0 * np.pi)
    bins = np.floor(angles / (2.0 * np.pi) * angle_bins).astype(np.int32)
    bins = np.clip(bins, 0, angle_bins - 1)

    outer_radius = np.zeros(angle_bins, dtype=np.float32)
    np.maximum.at(outer_radius, bins, radii)
    valid = outer_radius > 0
    outer_radius = fill_missing_circular_values(outer_radius, valid)

    contour = main_contour(mask)
    arc_len = cv2.arcLength(contour, True) if contour is not None else 0.0
    smooth_degrees = max(9, min(35, int(round(arc_len / max(1.0, equivalent_radius(mask)) * 3.0))))
    window = max(5, int(round(angle_bins * smooth_degrees / 360.0)))
    outer_radius = circular_morph_close(outer_radius, window)

    points = []
    for idx, radius in enumerate(outer_radius):
        theta = 2.0 * np.pi * idx / angle_bins
        x = int(round(cx + radius * np.cos(theta)))
        y = int(round(cy + radius * np.sin(theta)))
        points.append([x, y])

    h, w = mask.shape
    points = np.array(points, dtype=np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)

    repaired = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(repaired, [points], 1)
    return repaired


def repair_iris_boundary(iris_mask, sclera_mask, pupil_mask):
    base = largest_component((iris_mask > 0).astype(np.uint8))
    if base.sum() < 30:
        return base

    eye_region = build_eye_region(sclera_mask, base, pupil_mask)
    if eye_region.sum() < 30:
        return base

    radius = equivalent_radius(base)
    smooth_size = odd_kernel_size(radius * 0.045, min_size=5, max_size=25)
    base_smooth = fill_and_smooth(base, close_kernel=smooth_size, open_kernel=3)

    center = choose_center(base_smooth, pupil_mask)
    polar_prior = polar_boundary_smooth_mask(base_smooth, center)
    ellipse_prior = fit_ellipse_mask(base_smooth)
    hull_prior = convex_hull_mask(base_smooth)
    contour_prior = smooth_contour_mask(base_smooth)

    band_size = odd_kernel_size(radius * 0.34, min_size=31, max_size=131)
    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_size, band_size))
    near_original = cv2.dilate(base, band_kernel)

    shape_prior = np.maximum(np.maximum(np.maximum(polar_prior, ellipse_prior), hull_prior), contour_prior).astype(np.uint8)
    shape_candidate = (shape_prior & near_original & eye_region).astype(np.uint8)
    candidate = np.maximum(base_smooth, shape_candidate).astype(np.uint8)

    defect_ratio = boundary_defect_ratio(base_smooth)
    max_addition_ratio = min(0.65, max(0.35, 0.18 + defect_ratio * 2.6))
    candidate_addition_ratio = float((candidate & (1 - base)).sum()) / max(1.0, float(base.sum()))
    if candidate_addition_ratio <= max_addition_ratio:
        regularized = np.maximum(base, candidate).astype(np.uint8)
    else:
        regularized = limit_addition_by_distance(base, candidate, max_addition_ratio)

    final_close = odd_kernel_size(radius * 0.028, min_size=5, max_size=17)
    smoothed_regularized = fill_and_smooth(regularized, close_kernel=final_close, open_kernel=3)
    regularized = np.maximum(regularized, smoothed_regularized).astype(np.uint8)
    regularized = (regularized & eye_region).astype(np.uint8)
    regularized = assign_eye_gap_to_iris(regularized, sclera_mask, pupil_mask, eye_region)
    regularized = np.maximum(regularized, smooth_contour_mask(regularized, window_ratio=0.05)).astype(np.uint8)
    regularized = (regularized & eye_region).astype(np.uint8)
    regularized = remove_edge_connected_additions(base, regularized)
    return largest_component(regularized)


def constrained_regularize(
    iris_mask,
    sclera_mask,
    pupil_mask=None,
):
    iris = largest_component((iris_mask > 0).astype(np.uint8))
    if iris.sum() < 30:
        return iris

    return repair_iris_boundary(iris, sclera_mask, pupil_mask)


def find_mask(mask_dir, stem, suffix):
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        path = mask_dir / f"{stem}_{suffix}{ext}"
        if path.exists():
            return path
    return None


def clean_binary(mask, min_area=30, close_kernel=5):
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() < min_area:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned


def make_predict_style_overlay(image, sclera, iris_regularized, pupil):
    color_mask = np.zeros_like(image)

    pupil = (pupil > 0).astype(np.uint8)
    iris_regularized = ((iris_regularized > 0) & (pupil == 0)).astype(np.uint8)
    sclera = ((sclera > 0) & (iris_regularized == 0) & (pupil == 0)).astype(np.uint8)

    # Predict.py-style single combined overlay, but with exclusive class priority:
    # pupil > regularized iris > sclera. This avoids mixed colors at boundaries.
    color_mask[:, :, 0] = sclera * 255
    color_mask[:, :, 1] = pupil * 255
    color_mask[:, :, 2] = iris_regularized * 255

    return cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)


def select_image_files(image_dir, image):
    image_files = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
    )
    if image is None:
        return image_files

    requested = Path(image)
    direct_path = requested if requested.is_absolute() else image_dir / requested
    if direct_path.is_file():
        if direct_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {direct_path.suffix}")
        return [direct_path]

    requested_stem = requested.stem if requested.suffix else requested.name
    matches = [path for path in image_files if path.stem == requested_stem]
    if not matches:
        raise ValueError(f"Image not found: {image}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Multiple images match '{image}': {names}")
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Create predict.py-style combined visualization with regularized iris."
    )
    parser.add_argument("--image-dir", default="D:/BaiduNetdiskDownload/all_pics/ring")
    parser.add_argument("--sclera-dir", default="D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/sclera")
    parser.add_argument("--iris-dir", default="D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/iris")
    parser.add_argument("--pupil-dir", default="D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/pupil")
    parser.add_argument("--out-dir", default="D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/combine_regularized")
    parser.add_argument(
        "--image",
        default=None,
        help="Process one image by stem, file name, or full path.",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    sclera_dir = Path(args.sclera_dir)
    iris_dir = Path(args.iris_dir)
    pupil_dir = Path(args.pupil_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        image_files = select_image_files(image_dir, args.image)
    except ValueError as error:
        parser.error(str(error))
    for image_path in tqdm(image_files, desc="Combining predict-style masks"):
        stem = image_path.stem
        sclera_path = find_mask(sclera_dir, stem, "sclera")
        iris_path = find_mask(iris_dir, stem, "iris")
        pupil_path = find_mask(pupil_dir, stem, "pupil")

        if sclera_path is None or iris_path is None or pupil_path is None:
            print(f"Skip {stem}: missing mask")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip {stem}: cannot read image")
            continue

        sclera = clean_binary(read_binary_mask(sclera_path), min_area=50, close_kernel=5)
        # Keep the model's pupil prediction unchanged. It is only used as a
        # reference while repairing the iris and has display priority later.
        pupil = read_binary_mask(pupil_path)
        iris = read_binary_mask(iris_path)

        iris_regularized = constrained_regularize(
            iris,
            sclera_mask=sclera,
            pupil_mask=pupil,
        )
        combined_vis = make_predict_style_overlay(image, sclera, iris_regularized, pupil)
        cv2.imwrite(str(out_dir / f"{stem}_combined.png"), combined_vis)


if __name__ == "__main__":
    main()
