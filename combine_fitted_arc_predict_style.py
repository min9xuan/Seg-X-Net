import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from combine_regularized_predict_style import (
    IMAGE_EXTS,
    build_eye_region,
    clean_binary,
    find_mask,
    largest_component,
    make_predict_style_overlay,
    mask_center,
    read_binary_mask,
)


def resize_for_fit(mask, max_side=640):
    height, width = mask.shape
    scale = min(1.0, max_side / float(max(height, width)))
    work_width = max(1, int(round(width * scale)))
    work_height = max(1, int(round(height * scale)))
    resized = cv2.resize(mask.astype(np.uint8), (work_width, work_height), interpolation=cv2.INTER_NEAREST)
    return resized, scale


def ellipse_mask(shape, parameters):
    cx, cy, axis_x, axis_y, angle = parameters
    result = np.zeros(shape, dtype=np.uint8)
    center = (int(round(cx)), int(round(cy)))
    axes = (max(1, int(round(axis_x))), max(1, int(round(axis_y))))
    cv2.ellipse(result, center, axes, float(angle), 0, 360, 1, -1, lineType=cv2.LINE_8)
    return result


def fit_score(parameters, target, eye_region, pupil):
    candidate = ellipse_mask(target.shape, parameters)
    candidate = (candidate & eye_region).astype(np.uint8)
    candidate_area = float(candidate.sum())
    target_area = float(target.sum())
    if candidate_area < 10 or target_area < 10:
        return -1.0

    if pupil.sum() > 20:
        coverage = float((candidate & pupil).sum()) / float(pupil.sum())
        if coverage < 0.90:
            return -1.0 + coverage * 0.1

    intersection = float((candidate & target).sum())
    return (2.0 * intersection) / (candidate_area + target_area + 1e-6)


def clamp_parameters(parameters, bounds):
    cx, cy, axis_x, axis_y, angle = parameters
    cx = float(np.clip(cx, bounds[0][0], bounds[0][1]))
    cy = float(np.clip(cy, bounds[1][0], bounds[1][1]))
    axis_x = float(np.clip(axis_x, bounds[2][0], bounds[2][1]))
    axis_y = float(np.clip(axis_y, bounds[3][0], bounds[3][1]))
    ratio = axis_x / max(axis_y, 1e-6)
    if ratio > 2.2:
        axis_x = axis_y * 2.2
    elif ratio < 1.0 / 2.2:
        axis_y = axis_x * 2.2
    return np.array([cx, cy, axis_x, axis_y, angle % 180.0], dtype=np.float64)


def coordinate_descent(initial, target, eye_region, pupil, bounds, span):
    best = clamp_parameters(initial, bounds)
    best_score = fit_score(best, target, eye_region, pupil)
    steps = np.array(
        [max(2.0, span * 0.10), max(2.0, span * 0.10), max(2.0, span * 0.12), max(2.0, span * 0.12), 12.0],
        dtype=np.float64,
    )

    for _ in range(6):
        for _ in range(4):
            improved = False
            for index in range(5):
                for direction in (-1.0, 1.0):
                    trial = best.copy()
                    trial[index] += direction * steps[index]
                    trial = clamp_parameters(trial, bounds)
                    score = fit_score(trial, target, eye_region, pupil)
                    if score > best_score + 1e-6:
                        best, best_score = trial, score
                        improved = True
            if not improved:
                break
        steps *= 0.5
    return best, best_score


def initial_candidates(target, pupil):
    ys, xs = np.nonzero(target)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    width = x_max - x_min + 1.0
    height = y_max - y_min + 1.0
    target_center = np.array([xs.mean(), ys.mean()], dtype=np.float64)
    pupil_center = mask_center(pupil) if pupil.sum() > 20 else None
    center = np.asarray(pupil_center, dtype=np.float64) if pupil_center is not None else target_center
    radius = max(np.sqrt(float(target.sum()) / np.pi), 0.42 * max(width, height))

    candidates = [
        np.array([center[0], center[1], radius, radius, 0.0]),
        np.array([center[0], center[1], width * 0.50, height * 0.50, 0.0]),
        np.array([target_center[0], target_center[1], width * 0.50, height * 0.50, 0.0]),
    ]

    contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if len(contour) >= 5:
            (cx, cy), (diameter_x, diameter_y), angle = cv2.fitEllipse(contour)
            candidates.append(np.array([cx, cy, diameter_x * 0.5, diameter_y * 0.5, angle]))
    return candidates, (x_min, x_max, y_min, y_max, width, height, center)


def closest_regular_arc(iris_mask, sclera_mask, pupil_mask):
    iris = largest_component((iris_mask > 0).astype(np.uint8))
    pupil = (pupil_mask > 0).astype(np.uint8)
    if iris.sum() < 30:
        return iris, None, 0.0

    eye_region = build_eye_region(sclera_mask, iris, pupil)
    target = ((iris > 0) | (pupil > 0)).astype(np.uint8)
    target = (target & eye_region).astype(np.uint8)

    work_eye, scale = resize_for_fit(eye_region)
    work_iris = cv2.resize(iris, (work_eye.shape[1], work_eye.shape[0]), interpolation=cv2.INTER_NEAREST)
    work_pupil = cv2.resize(pupil, (work_eye.shape[1], work_eye.shape[0]), interpolation=cv2.INTER_NEAREST)
    work_target = ((work_iris > 0) | (work_pupil > 0)).astype(np.uint8)
    work_target = (work_target & work_eye).astype(np.uint8)
    if work_target.sum() < 10:
        return iris, None, 0.0

    candidates, geometry = initial_candidates(work_target, work_pupil)
    x_min, x_max, y_min, y_max, width, height, center = geometry
    span = max(width, height)
    center_margin = max(6.0, span * 0.35)
    min_axis = max(3.0, span * 0.15)
    max_axis = max(8.0, span * 1.20)
    bounds = (
        (max(0.0, center[0] - center_margin), min(work_eye.shape[1] - 1.0, center[0] + center_margin)),
        (max(0.0, center[1] - center_margin), min(work_eye.shape[0] - 1.0, center[1] + center_margin)),
        (min_axis, max_axis),
        (min_axis, max_axis),
    )

    best_parameters, best_score = None, -1.0
    for candidate in candidates:
        parameters, score = coordinate_descent(
            candidate, work_target, work_eye, work_pupil, bounds, span
        )
        if score > best_score:
            best_parameters, best_score = parameters, score

    if best_parameters is None:
        return iris, None, 0.0

    full_parameters = best_parameters.copy()
    full_parameters[:4] /= scale
    fitted = ellipse_mask(iris.shape, full_parameters)
    fitted = (fitted & eye_region).astype(np.uint8)
    return fitted, full_parameters, best_score


def main():
    parser = argparse.ArgumentParser(
        description="Fit the closest circle/ellipse to the iris inside the predicted eye region."
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
        default="D:/AAA_CodeRepo/Seg-X-Net/result/seg_sclera_iris_pupil/predict_result/combine_fitted_arc",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    sclera_dir = Path(args.sclera_dir)
    iris_dir = Path(args.iris_dir)
    pupil_dir = Path(args.pupil_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    for image_path in tqdm(image_files, desc="Fitting iris circle/ellipse"):
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
        iris = read_binary_mask(iris_path)
        pupil = read_binary_mask(pupil_path)
        fitted_iris, _, _ = closest_regular_arc(iris, sclera, pupil)

        # Pupil has the highest display priority. The fitted iris may replace
        # original sclera pixels; only the unoccupied original sclera remains.
        combined = make_predict_style_overlay(image, sclera, fitted_iris, pupil)
        cv2.imwrite(str(out_dir / f"{stem}_combined.png"), combined)


if __name__ == "__main__":
    main()
