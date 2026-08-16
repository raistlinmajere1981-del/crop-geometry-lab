"""Image- and circle-level error measurements for sixfold rotational symmetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import median
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

from .feature_table import CircleFeature

Point = tuple[float, float]
SIXFOLD_ROTATIONS = (60, 120, 180, 240, 300)


@dataclass(frozen=True, slots=True)
class ImageSymmetryError:
    angle_degrees: int
    normalized_mae: float
    normalized_rmse: float
    correlation: float
    overlap_fraction: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GeometricSymmetryError:
    angle_degrees: int
    mean_position_error_pixels: float
    normalized_position_error: float
    mean_radius_relative_error: float
    matched_fraction: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _gray_float(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unsupported image shape")
    return gray.astype(np.float32) / 255.0


def image_symmetry_errors(
    image: NDArray[np.uint8],
    center: Point,
    *,
    max_radius: float | None = None,
) -> list[ImageSymmetryError]:
    """Compare a circular image region with its 60° rotational copies."""

    gray = _gray_float(image)
    height, width = gray.shape
    center_x, center_y = map(float, center)
    if not (0 <= center_x < width and 0 <= center_y < height):
        raise ValueError("Symmetry center must lie inside the image")
    available_radius = min(
        center_x,
        center_y,
        width - 1.0 - center_x,
        height - 1.0 - center_y,
    )
    radius = available_radius if max_radius is None else min(available_radius, max_radius)
    if radius < 2.0:
        raise ValueError("Not enough image area around the selected center")

    y_grid, x_grid = np.ogrid[:height, :width]
    base_mask = (
        (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius**2
    ).astype(np.uint8)
    region_pixels = max(1, int(base_mask.sum()))
    results: list[ImageSymmetryError] = []
    for angle in SIXFOLD_ROTATIONS:
        matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        rotated = cv2.warpAffine(
            gray,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        rotated_mask = cv2.warpAffine(
            base_mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        overlap = (base_mask > 0) & (rotated_mask > 0)
        original_values = gray[overlap].astype(np.float64)
        rotated_values = rotated[overlap].astype(np.float64)
        difference = original_values - rotated_values
        mae = float(np.mean(np.abs(difference))) if difference.size else 1.0
        rmse = (
            float(math.sqrt(float(np.mean(difference**2))))
            if difference.size
            else 1.0
        )
        if difference.size < 2:
            correlation = 0.0
        else:
            original_std = float(original_values.std())
            rotated_std = float(rotated_values.std())
            if original_std < 1e-12 or rotated_std < 1e-12:
                correlation = 1.0 if rmse < 1e-12 else 0.0
            else:
                correlation = float(
                    np.corrcoef(original_values, rotated_values)[0, 1]
                )
        results.append(
            ImageSymmetryError(
                angle_degrees=angle,
                normalized_mae=mae,
                normalized_rmse=rmse,
                correlation=max(-1.0, min(1.0, correlation)),
                overlap_fraction=float(overlap.sum()) / region_pixels,
            )
        )
    return results


def _rotate_point(point: Point, center: Point, angle_degrees: float) -> Point:
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    # Positive angles follow the conventional CCW system despite image y-down.
    return (
        center[0] + cosine * dx + sine * dy,
        center[1] - sine * dx + cosine * dy,
    )


def geometric_symmetry_errors(
    circles: Iterable[CircleFeature], center: Point
) -> list[GeometricSymmetryError]:
    """Match rotated circles one-to-one and report scale-independent errors."""

    items = list(circles)
    if not items:
        return [
            GeometricSymmetryError(angle, 0.0, 0.0, 0.0, 0.0)
            for angle in SIXFOLD_ROTATIONS
        ]
    characteristic_radius = max(1.0, float(median(circle.radius for circle in items)))
    results: list[GeometricSymmetryError] = []
    for angle in SIXFOLD_ROTATIONS:
        pair_costs: list[tuple[float, int, int, float, float]] = []
        for source_index, source in enumerate(items):
            target_x, target_y = _rotate_point((source.x, source.y), center, angle)
            for target_index, target in enumerate(items):
                position_error = math.hypot(target_x - target.x, target_y - target.y)
                radius_error = abs(source.radius - target.radius) / max(
                    source.radius, target.radius, 1e-6
                )
                type_penalty = 2.0 * characteristic_radius if (
                    source.circle_type != target.circle_type
                ) else 0.0
                cost = position_error + characteristic_radius * radius_error + type_penalty
                pair_costs.append(
                    (cost, source_index, target_index, position_error, radius_error)
                )
        used_sources: set[int] = set()
        used_targets: set[int] = set()
        position_errors: list[float] = []
        radius_errors: list[float] = []
        for _, source_index, target_index, position_error, radius_error in sorted(
            pair_costs
        ):
            if source_index in used_sources or target_index in used_targets:
                continue
            used_sources.add(source_index)
            used_targets.add(target_index)
            position_errors.append(position_error)
            radius_errors.append(radius_error)
            if len(used_sources) == len(items):
                break
        mean_position = float(np.mean(position_errors)) if position_errors else 0.0
        results.append(
            GeometricSymmetryError(
                angle_degrees=angle,
                mean_position_error_pixels=mean_position,
                normalized_position_error=mean_position / characteristic_radius,
                mean_radius_relative_error=(
                    float(np.mean(radius_errors)) if radius_errors else 0.0
                ),
                matched_fraction=len(position_errors) / len(items),
            )
        )
    return results


def sixfold_summary(
    image_errors: Iterable[ImageSymmetryError],
    geometric_errors: Iterable[GeometricSymmetryError],
) -> dict[str, float]:
    image_items = list(image_errors)
    geometry_items = list(geometric_errors)
    return {
        "mean_image_mae": (
            float(np.mean([item.normalized_mae for item in image_items]))
            if image_items
            else 0.0
        ),
        "mean_image_correlation": (
            float(np.mean([item.correlation for item in image_items]))
            if image_items
            else 0.0
        ),
        "mean_geometric_error": (
            float(np.mean([item.normalized_position_error for item in geometry_items]))
            if geometry_items
            else 0.0
        ),
    }

