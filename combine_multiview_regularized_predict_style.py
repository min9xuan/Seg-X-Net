import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from combine_regularized_predict_style import (
    IMAGE_EXTS,
    boundary_defect_ratio,
    build_eye_region,
    clean_binary,
    constrained_regularize,
    find_mask,
    fit_ellipse_mask,
    largest_component,
    make_predict_style_overlay,
    mask_center,
    read_binary_mask,
)


VIEW_PREFIX = re.compile(r"^(LDown|LLeft|LN|LRight|LUp|RDown|RLeft|RN|RRight|RUp)_")


def subject_key(stem):
    return VIEW_PREFIX.sub("", stem)


def load_masks(stem, sclera_dir, iris_dir, pupil_dir):
    sclera_path = find_mask(sclera_dir, stem, "sclera")
    iris_path = find_mask(iris_dir, stem, "iris")
    pupil_path = find_mask(pupil_dir, stem, "pupil")
    if sclera_path is None or iris_path is None or pupil_path is None:
        return None
    sclera = clean_binary(read_binary_mask(sclera_path), min_area=50, close_kernel=5)
    iris = read_binary_mask(iris_path)
    pupil = read_binary_mask(pupil_path)
    return sclera, iris, pupil


def dice_score(first, second):
    first = first > 0
    second = second > 0
    denominator = float(first.sum() + second.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float((first & second).sum()) / denominator


def outer_radius(mask, center):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20 or center is None:
        return 0.0
    distances = np.hypot(xs.astype(np.float32) - center[0], ys.astype(np.float32) - center[1])
    return float(np.percentile(distances, 95))


def supported_outer_radius(mask, center, angle_bins=360, support_degrees=45):
    ys, xs = np.nonzero(largest_component((mask > 0).astype(np.uint8)))
    if len(xs) < 20 or center is None:
        return 0.0
    dx = xs.astype(np.float32) - center[0]
    dy = ys.astype(np.float32) - center[1]
    distances = np.hypot(dx, dy)
    bins = np.floor(
        (np.arctan2(dy, dx) + np.pi) * angle_bins / (2.0 * np.pi)
    ).astype(np.int32) % angle_bins
    radial_profile = np.zeros(angle_bins, dtype=np.float32)
    np.maximum.at(radial_profile, bins, distances)

    window = max(3, int(round(support_degrees * angle_bins / 360.0)))
    if window % 2 == 0:
        window += 1
    half = window // 2
    supported = []
    for index in range(angle_bins):
        values = np.take(
            radial_profile,
            np.arange(index - half, index + half + 1),
            mode="wrap",
        )
        values = values[values > 0]
        if len(values) >= window // 2:
            supported.append(float(np.median(values)))
    return max(supported) if supported else float(np.percentile(distances, 95))


def geometry_center(pupil, iris):
    pupil_main = largest_component((pupil > 0).astype(np.uint8))
    if pupil_main.sum() > 20:
        return mask_center(pupil_main)
    return mask_center(iris)


def pupil_projection(pupil):
    pupil_main = largest_component((pupil > 0).astype(np.uint8))
    contours, _ = cv2.findContours(
        pupil_main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5 or cv2.contourArea(contour) < 30:
        return None

    ellipse = cv2.fitEllipse(contour)
    width, height = ellipse[1]
    if width <= 1 or height <= 1:
        return None
    fitted = np.zeros_like(pupil_main, dtype=np.uint8)
    cv2.ellipse(fitted, ellipse, 1, -1)
    if dice_score(pupil_main, fitted) < 0.80:
        return None

    ratio = float(np.clip(min(width, height) / max(width, height), 0.45, 1.0))
    major_angle = float(ellipse[2] if width >= height else ellipse[2] + 90.0)
    return {
        "ratio": ratio,
        "major_angle": major_angle % 180.0,
    }


def view_constrained_major_angle(stem, projection):
    if not stem:
        return 0.0 if projection is None else projection["major_angle"]
    view = stem.split("_", 1)[0]
    if view in {"LUp", "LDown", "RUp", "RDown"}:
        return 0.0
    if view in {"LLeft", "LRight", "RLeft", "RRight"}:
        return 90.0
    return 0.0 if projection is None else projection["major_angle"]


def radial_cumulative(mask, center):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or center is None:
        return np.zeros(1, dtype=np.int64)
    radii = np.floor(
        np.hypot(xs.astype(np.float32) - center[0], ys.astype(np.float32) - center[1])
    ).astype(np.int32)
    return np.cumsum(np.bincount(radii), dtype=np.int64)


def circle_missing_ratio(record, radius):
    iris_counts = record["iris_radial_counts"]
    eye_counts = record["eye_radial_counts"]
    index = max(0, int(round(radius)))
    iris_inside = iris_counts[min(index, len(iris_counts) - 1)]
    eye_inside = eye_counts[min(index, len(eye_counts) - 1)]
    return 1.0 - float(iris_inside) / max(float(eye_inside), 1.0)


def iris_sclera_arc_points(iris, sclera, center):
    iris = (iris > 0).astype(np.uint8)
    boundary = iris - cv2.erode(iris, np.ones((3, 3), dtype=np.uint8))
    radius = outer_radius(iris, center)
    max_gap = int(np.clip(radius * 0.065, 12, 45))
    distance_to_sclera = cv2.distanceTransform(
        (sclera == 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    interface = (boundary > 0) & (distance_to_sclera <= max_gap)
    ys, xs = np.nonzero(interface)
    if len(xs) < 30:
        return np.empty((0, 2), dtype=np.float64)
    step = max(1, len(xs) // 4000)
    return np.column_stack((xs[::step], ys[::step])).astype(np.float64)


def robust_circle_fit(points):
    if len(points) < 30:
        return None
    keep = np.ones(len(points), dtype=bool)
    center = None
    radius = 0.0
    residuals = None
    for _ in range(5):
        selected = points[keep]
        matrix = np.column_stack(
            (2.0 * selected[:, 0], 2.0 * selected[:, 1], np.ones(len(selected)))
        )
        target = selected[:, 0] ** 2 + selected[:, 1] ** 2
        cx, cy, constant = np.linalg.lstsq(matrix, target, rcond=None)[0]
        center = np.asarray([cx, cy], dtype=np.float64)
        radius = float(np.sqrt(max(1.0, constant + cx * cx + cy * cy)))
        residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        median = float(np.median(residuals[keep]))
        mad = float(np.median(np.abs(residuals[keep] - median)))
        new_keep = residuals <= max(3.0, median + 2.5 * max(mad, 1.0))
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
    error = float(np.median(residuals[keep]) / max(radius, 1.0))
    return center, radius, error, int(keep.sum())


def describe_view(stem, iris, sclera, pupil, regularized, eye_region):
    center = geometry_center(pupil, regularized)
    projection = pupil_projection(pupil)
    arc_points = iris_sclera_arc_points(regularized, sclera, center)
    arc_fit = robust_circle_fit(arc_points)
    fitted = fit_ellipse_mask(regularized)
    fitted = ((fitted > 0) & (eye_region > 0) & (pupil == 0)).astype(np.uint8)
    visible_iris = ((regularized > 0) & (pupil == 0)).astype(np.uint8)
    record = {
        "stem": stem,
        "subject": subject_key(stem),
        "area": float(visible_iris.sum()),
        "radius": outer_radius(regularized, center),
        "supported_radius": supported_outer_radius(regularized, center),
        "fit_dice": dice_score(visible_iris, fitted),
        "defect": float(boundary_defect_ratio(iris)),
        "arc_radius": 0.0 if arc_fit is None else arc_fit[1],
        "arc_error": 1.0 if arc_fit is None else arc_fit[2],
        "arc_points": len(arc_points),
        "projection_ratio": 1.0 if projection is None else projection["ratio"],
        "projection_valid": projection is not None,
        "iris_radial_counts": radial_cumulative(visible_iris, center),
        "eye_radial_counts": radial_cumulative(
            ((eye_region > 0) & (pupil == 0)).astype(np.uint8), center
        ),
    }
    record["self_circle_missing"] = circle_missing_ratio(
        record, record["supported_radius"]
    )
    return record


def robust_consensus(records):
    valid = [record for record in records if record["radius"] > 5 and record["area"] > 30]
    if len(valid) < 4:
        return None

    radii = np.asarray([record["radius"] for record in valid], dtype=np.float64)
    areas = np.asarray([record["area"] for record in valid], dtype=np.float64)
    fit_scores = np.asarray([record["fit_dice"] for record in valid], dtype=np.float64)
    median_radius = float(np.median(radii))
    median_area = float(np.median(areas))
    fit_floor = max(0.58, float(np.percentile(fit_scores, 20)) - 0.04)

    reliable = [
        record
        for record in valid
        if 0.68 * median_radius <= record["radius"] <= 1.40 * median_radius
        and 0.42 * median_area <= record["area"] <= 1.90 * median_area
        and record["fit_dice"] >= fit_floor
    ]
    if len(reliable) < 3:
        reliable = sorted(valid, key=lambda item: item["fit_dice"], reverse=True)[: max(3, len(valid) // 2)]

    radius_scales = [
        record["supported_radius"] / record["radius"]
        for record in reliable
        if record["radius"] > 5
        and 0.80 <= record["supported_radius"] / record["radius"] <= 1.35
    ]
    radius_scale = float(np.median(radius_scales)) if radius_scales else 1.0
    projection_ratios = [
        record["projection_ratio"]
        for record in reliable
        if record["projection_valid"]
    ]
    projection_ratio = (
        float(np.median(projection_ratios)) if projection_ratios else 1.0
    )

    return {
        "radius": float(np.median([record["radius"] for record in reliable])),
        "area": float(np.median([record["area"] for record in reliable])),
        "fit_dice": float(np.median([record["fit_dice"] for record in reliable])),
        "defect": float(np.median([record["defect"] for record in reliable])),
        "circle_missing": float(
            np.median([record["self_circle_missing"] for record in reliable])
        ),
        "radius_scale": radius_scale,
        "projection_ratio": projection_ratio,
        "reliable_count": len(reliable),
    }


def failure_reason(record, consensus):
    radius_ratio = record["radius"] / max(consensus["radius"], 1.0)
    area_ratio = record["area"] / max(consensus["area"], 1.0)
    reasons = []
    if radius_ratio < 0.58:
        reasons.append("radius_too_small")
    elif radius_ratio > 1.58:
        reasons.append("radius_too_large")
    if area_ratio < 0.30:
        reasons.append("area_too_small")
    elif area_ratio > 2.40:
        reasons.append("area_too_large")
    severe_shape_failure = (
        record["fit_dice"] < min(0.62, consensus["fit_dice"] - 0.16)
        and record["defect"] > max(0.20, consensus["defect"] * 1.55)
    )
    if severe_shape_failure:
        reasons.append("severe_boundary_defect")
    if record["arc_points"] >= 80 and record["arc_error"] < 0.12:
        arc_ratio = record["arc_radius"] / max(consensus["radius"], 1.0)
        if arc_ratio > 1.30:
            reasons.append("arc_too_flat")
        elif arc_ratio < 0.70:
            reasons.append("arc_too_tight")
    expected_radius = (
        record["radius"] * consensus["radius_scale"]
        if record["radius"] >= 0.50 * consensus["radius"]
        else consensus["radius"] * consensus["radius_scale"]
    )
    missing_ratio = circle_missing_ratio(record, expected_radius)
    missing_limit = max(0.25, consensus["circle_missing"] + 0.12)
    if missing_ratio > missing_limit:
        reasons.append("circle_incomplete")
    return reasons


def consensus_circle(shape, center, radius):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(
        mask,
        (int(round(center[0])), int(round(center[1]))),
        max(1, int(round(radius))),
        1,
        -1,
        lineType=cv2.LINE_8,
    )
    return mask


def projected_iris_ellipse(shape, center, major_radius, projection):
    if projection is None:
        return consensus_circle(shape, center, major_radius)

    minor_radius = major_radius * projection["ratio"]
    axes = (int(round(major_radius)), int(round(minor_radius)))
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(center[0])), int(round(center[1]))),
        (max(1, axes[0]), max(1, axes[1])),
        projection["major_angle"],
        0,
        360,
        1,
        -1,
        lineType=cv2.LINE_8,
    )
    return mask


def repair_from_other_views(
    iris, sclera, pupil, regularized, consensus, reasons, stem=None
):
    eye_region = build_eye_region(sclera, iris, pupil)
    center = geometry_center(pupil, regularized)
    if center is None:
        return regularized
    current_radius = outer_radius(regularized, center)
    radius = current_radius * consensus["radius_scale"]
    if current_radius < 0.50 * consensus["radius"]:
        radius = consensus["radius"] * consensus["radius_scale"]

    projection = pupil_projection(pupil)
    if projection is None:
        return regularized
    projection = projection.copy()
    projection["ratio"] = float(
        np.sqrt(projection["ratio"] * consensus["projection_ratio"])
    )
    projection["major_angle"] = view_constrained_major_angle(stem, projection)
    candidate = projected_iris_ellipse(iris.shape, center, radius, projection)
    candidate = (candidate & eye_region).astype(np.uint8)
    if candidate.sum() < 30:
        return regularized
    return candidate


def main():
    parser = argparse.ArgumentParser(
        description="Regularize iris masks and repair extreme failures using other views of the same subject."
    )
    parser.add_argument("--image-dir", default="D:/BaiduNetdiskDownload/all_pics/ring")
    parser.add_argument(
        "--sclera-dir",
        default="D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/sclera",
    )
    parser.add_argument(
        "--iris-dir",
        default="D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/iris",
    )
    parser.add_argument(
        "--pupil-dir",
        default="D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/pupil",
    )
    parser.add_argument(
        "--out-dir",
        default="D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/combine_multiview_regularized",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Only process one subject key, for example T18.",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    sclera_dir = Path(args.sclera_dir)
    iris_dir = Path(args.iris_dir)
    pupil_dir = Path(args.pupil_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    grouped_images = {}
    for image_path in image_files:
        grouped_images.setdefault(subject_key(image_path.stem), []).append(image_path)
    if args.subject is not None:
        grouped_images = {
            subject: paths
            for subject, paths in grouped_images.items()
            if subject.lower() == args.subject.lower()
        }

    saved_count = 0
    repair_count = 0
    for subject, subject_images in tqdm(
        sorted(grouped_images.items()), desc="Processing subjects"
    ):
        records = []
        valid_images = {}

        # Finish and repair each subject before moving to the next one. This
        # prevents an interrupted full run from leaving every result at pass one.
        for image_path in subject_images:
            stem = image_path.stem
            masks = load_masks(stem, sclera_dir, iris_dir, pupil_dir)
            if masks is None:
                print(f"Skip {stem}: missing mask")
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skip {stem}: cannot read image")
                continue

            sclera, iris, pupil = masks
            regularized = constrained_regularize(iris, sclera_mask=sclera, pupil_mask=pupil)
            eye_region = build_eye_region(sclera, iris, pupil)
            records.append(
                describe_view(stem, iris, sclera, pupil, regularized, eye_region)
            )
            valid_images[stem] = image_path
            combined = make_predict_style_overlay(image, sclera, regularized, pupil)
            cv2.imwrite(str(out_dir / f"{stem}_combined.png"), combined)
            saved_count += 1

        group_consensus = robust_consensus(records)
        if group_consensus is None:
            continue
        repairs = []
        for record in records:
            peer_records = [item for item in records if item["stem"] != record["stem"]]
            consensus = robust_consensus(peer_records) or group_consensus
            reasons = failure_reason(record, consensus)
            if reasons:
                repairs.append((record, consensus, reasons))

        for record, consensus, reasons in repairs:
            stem = record["stem"]
            if not record["projection_valid"]:
                print(f"Keep original {stem}: no valid pupil projection")
                continue
            masks = load_masks(stem, sclera_dir, iris_dir, pupil_dir)
            image = cv2.imread(str(valid_images[stem]))
            if masks is None or image is None:
                continue
            sclera, iris, pupil = masks
            regularized = constrained_regularize(iris, sclera_mask=sclera, pupil_mask=pupil)
            repaired = repair_from_other_views(
                iris, sclera, pupil, regularized, consensus, reasons, stem=stem
            )
            combined = make_predict_style_overlay(image, sclera, repaired, pupil)
            cv2.imwrite(str(out_dir / f"{stem}_combined.png"), combined)
            repair_count += 1
            replacement_radius = (
                record["radius"] * consensus["radius_scale"]
                if record["radius"] >= 0.50 * consensus["radius"]
                else consensus["radius"] * consensus["radius_scale"]
            )
            projection_ratio = float(
                np.sqrt(
                    record["projection_ratio"]
                    * consensus["projection_ratio"]
                )
            )
            print(
                f"Multi-view repair {stem}: {', '.join(reasons)}; "
                f"peer-scaled radius {replacement_radius:.1f}; "
                f"multi-view reference {consensus['radius']:.1f}; "
                f"scale {consensus['radius_scale']:.3f}; "
                f"projection ratio {projection_ratio:.3f}; "
                f"from {consensus['reliable_count']} reliable views"
            )

    print(f"Saved {saved_count} results to {out_dir}")
    print(f"Multi-view repair applied to {repair_count} extreme views")


if __name__ == "__main__":
    main()
