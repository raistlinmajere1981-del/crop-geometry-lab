"""Role-aware, tiled, multi-scale classical circle detection.

The three sampled size classes describe shared satellite-circle standards:
small circles use Difference-of-Gaussians extrema, medium circles use contour
analysis (optionally watershed-separated), and large circles use Hough plus
ellipse fitting. A separate continuous-radius pass finds variable-size
main-arm circles without widening those three standards. Images are never
upscaled for detection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Iterator, Literal, Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .circle_detector import _edge_confidence
from .feature_table import CircleFeature
from .preprocessing import (
    PreprocessProfile,
    PreprocessResult,
    transform_point,
    transform_radius,
)

SizeClass = Literal["small", "medium", "large"]


@dataclass(slots=True)
class DetectionEvidence:
    x: float
    y: float
    radius: float
    size_class: SizeClass | None
    detector: str
    circle_type: Literal["main", "satellite"] = "main"
    edge_score: float = 0.0
    color_score: float = 0.5
    circularity_score: float = 0.0
    symmetry_support: float = 0.0
    raw_score: float = 0.0


def _tiles(
    width: int, height: int, tile_size: int, overlap: float
) -> Iterator[tuple[int, int, int, int]]:
    tile = max(64, min(int(tile_size), max(width, height)))
    stride = max(1, int(round(tile * (1.0 - overlap))))

    def positions(length: int) -> list[int]:
        if length <= tile:
            return [0]
        values = list(range(0, max(1, length - tile + 1), stride))
        final = length - tile
        if values[-1] != final:
            values.append(final)
        return values

    for y in positions(height):
        for x in positions(width):
            yield x, y, min(width, x + tile), min(height, y + tile)


def effective_radius_ranges(
    profile: PreprocessProfile,
    source_to_processed: NDArray[np.float64] | None = None,
) -> dict[SizeClass, tuple[float, float]]:
    result: dict[SizeClass, tuple[float, float]] = {}
    for size_name in ("small", "medium", "large"):
        settings = profile.size_classes[size_name]
        minimum = float(settings["min_radius"])
        maximum = float(settings["max_radius"])
        samples = settings.get("samples", [])
        radii: list[float] = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            radius = float(sample.get("radius", 0))
            if radius <= 0:
                continue
            if source_to_processed is not None:
                center = (
                    float(sample.get("source_x", 0)),
                    float(sample.get("source_y", 0)),
                )
                processed_center = transform_point(
                    center, source_to_processed
                )
                processed_edge = transform_point(
                    (center[0] + radius, center[1]),
                    source_to_processed,
                )
                radius = math.dist(processed_center, processed_edge)
            radii.append(radius)
        if radii:
            minimum = max(1.0, min(radii) * 0.60)
            maximum = max(minimum + 1.0, max(radii) * 1.55)
        result[size_name] = (minimum, maximum)  # type: ignore[index]
    return result


def effective_main_arm_radius_range(
    profile: PreprocessProfile,
) -> tuple[float, float]:
    """Return the independent processed-pixel range for main-arm circles."""

    settings = profile.detection.get("main_arm", {})
    if not isinstance(settings, Mapping):
        raise ValueError("main_arm detection settings must be an object")
    minimum = float(settings.get("min_radius", 10.0))
    maximum = float(settings.get("max_radius", 36.0))
    if minimum <= 0 or maximum <= minimum:
        raise ValueError("Invalid variable main-arm radius interval")
    return minimum, maximum


def _valid_center(
    x: float,
    y: float,
    radius: float,
    exclusion_mask: NDArray[np.uint8],
) -> bool:
    height, width = exclusion_mask.shape
    if not (0 <= x < width and 0 <= y < height):
        return False
    # A sub-pixel Hough center such as 511.6 is inside a 512-pixel tile,
    # although round(511.6) is 512. Clamp the nearest-pixel lookup so valid
    # border candidates cannot escape the mask bounds.
    center_x = min(width - 1, max(0, int(round(x))))
    center_y = min(height - 1, max(0, int(round(y))))
    if exclusion_mask[center_y, center_x] > 0:
        return False
    samples = 24
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    xs = np.rint(x + radius * 0.7 * np.cos(angles)).astype(np.int32)
    ys = np.rint(y + radius * 0.7 * np.sin(angles)).astype(np.int32)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if int(valid.sum()) < samples // 2:
        return False
    return float(np.mean(exclusion_mask[ys[valid], xs[valid]] > 0)) < 0.35


def _dog_candidates(
    gray: NDArray[np.uint8],
    probability: NDArray[np.float32],
    exclusion: NDArray[np.uint8],
    radius_range: tuple[float, float],
    offset: tuple[int, int],
    *,
    max_candidates: int = 60,
) -> list[DetectionEvidence]:
    minimum, maximum = radius_range
    if maximum <= minimum:
        return []
    radii = np.geomspace(max(1.4, minimum), max(minimum + 0.5, maximum), 5)
    source = gray.astype(np.float32) / 255.0
    has_probability = (
        probability.size > 0 and float(np.max(probability)) > 0.05
    )
    candidates: list[DetectionEvidence] = []
    for radius in radii:
        sigma = max(0.8, float(radius) / math.sqrt(2.0))
        inner = cv2.GaussianBlur(source, (0, 0), sigma)
        outer = cv2.GaussianBlur(source, (0, 0), sigma * 1.6)
        response = np.abs(inner - outer)
        if has_probability:
            response *= 0.25 + 0.75 * probability
        response[exclusion > 0] = 0.0
        kernel_size = max(3, int(round(radius * 1.45)) | 1)
        local_maximum = cv2.dilate(
            response,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        )
        valid_values = response[exclusion == 0]
        if valid_values.size == 0:
            continue
        threshold = max(
            0.012,
            float(np.mean(valid_values) + 2.2 * np.std(valid_values)),
            float(np.quantile(valid_values, 0.995)),
        )
        maxima = (response >= local_maximum - 1e-7) & (response >= threshold)
        ys, xs = np.nonzero(maxima)
        if len(xs) == 0:
            continue
        ranked = sorted(
            zip(xs.tolist(), ys.tolist()),
            key=lambda point: float(response[point[1], point[0]]),
            reverse=True,
        )[:max_candidates]
        normalization = max(
            threshold,
            float(np.quantile(valid_values, 0.998)),
            1e-6,
        )
        for x, y in ranked:
            if has_probability and float(probability[y, x]) < 0.08:
                continue
            if not _valid_center(x, y, radius, exclusion):
                continue
            ring_angles = np.linspace(
                0.0, 2.0 * math.pi, 24, endpoint=False
            )
            ring_x = np.rint(
                x + max(2.0, radius * 1.15) * np.cos(ring_angles)
            ).astype(np.int32)
            ring_y = np.rint(
                y + max(2.0, radius * 1.15) * np.sin(ring_angles)
            ).astype(np.int32)
            valid_ring = (
                (ring_x >= 0)
                & (ring_x < response.shape[1])
                & (ring_y >= 0)
                & (ring_y < response.shape[0])
            )
            if int(valid_ring.sum()) < 16:
                continue
            center_response = float(response[y, x])
            ring_response = response[
                ring_y[valid_ring], ring_x[valid_ring]
            ]
            radial_drop = max(
                0.0,
                (center_response - float(np.median(ring_response)))
                / max(center_response, 1e-6),
            )
            ring_uniformity = 1.0 - min(
                1.0,
                float(np.std(ring_response))
                / max(center_response, 1e-6),
            )
            blob_shape = min(
                1.0, max(0.0, 0.65 * radial_drop + 0.35 * ring_uniformity)
            )
            if blob_shape < 0.24:
                continue
            raw_score = min(1.0, float(response[y, x]) / normalization)
            candidates.append(
                DetectionEvidence(
                    x=float(x + offset[0]),
                    y=float(y + offset[1]),
                    radius=float(radius),
                    size_class="small",
                    detector="dog",
                    circle_type="satellite",
                    circularity_score=blob_shape,
                    raw_score=raw_score,
                )
            )
    return sorted(
        candidates, key=lambda candidate: candidate.raw_score, reverse=True
    )[:max_candidates]


def _candidate_binary_masks(
    image: NDArray[np.uint8],
    probability: NDArray[np.float32],
    exclusion: NDArray[np.uint8],
) -> list[NDArray[np.uint8]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks: list[NDArray[np.uint8]] = []
    if float(np.max(probability)) > 0.05:
        for threshold in (0.38, 0.55):
            masks.append(
                np.where(probability >= threshold, 255, 0).astype(np.uint8)
            )
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.8)
    _, otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    masks.append(otsu)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -3,
    )
    masks.append(adaptive)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    prepared: list[NDArray[np.uint8]] = []
    for mask in masks:
        item = mask.copy()
        item[exclusion > 0] = 0
        item = cv2.morphologyEx(item, cv2.MORPH_OPEN, kernel)
        item = cv2.morphologyEx(item, cv2.MORPH_CLOSE, kernel)
        prepared.append(item)
    return prepared


def _watershed_contours(
    binary: NDArray[np.uint8], image: NDArray[np.uint8]
) -> list[NDArray[np.int32]]:
    if int(np.count_nonzero(binary)) == 0:
        return []
    kernel = np.ones((3, 3), dtype=np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_background = cv2.dilate(opening, kernel, iterations=2)
    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if float(distance.max()) <= 0:
        return []
    sure_foreground = np.where(
        distance >= 0.38 * float(distance.max()), 255, 0
    ).astype(np.uint8)
    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers.astype(np.int32) + 1
    unknown = cv2.subtract(sure_background, sure_foreground)
    markers[unknown > 0] = 0
    cv2.watershed(image.copy(), markers)
    contours: list[NDArray[np.int32]] = []
    for label in range(2, int(markers.max()) + 1):
        region = np.where(markers == label, 255, 0).astype(np.uint8)
        found, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours.extend(found)
    return contours


def _contour_candidates(
    image: NDArray[np.uint8],
    probability: NDArray[np.float32],
    exclusion: NDArray[np.uint8],
    radius_range: tuple[float, float],
    offset: tuple[int, int],
    *,
    watershed: bool,
    size_class: SizeClass | None = "medium",
    detector_prefix: str = "",
    circle_type: Literal["main", "satellite"] = "satellite",
    max_candidates: int = 180,
) -> list[DetectionEvidence]:
    minimum, maximum = radius_range
    candidates: list[DetectionEvidence] = []
    for binary in _candidate_binary_masks(image, probability, exclusion):
        if watershed:
            contours = _watershed_contours(binary, image)
        else:
            contours, _ = cv2.findContours(
                binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if area <= 0 or perimeter <= 0:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            radius = float(radius)
            if not minimum * 0.72 <= radius <= maximum * 1.25:
                continue
            circularity = min(1.0, 4.0 * math.pi * area / (perimeter**2))
            fill_ratio = min(1.0, area / max(math.pi * radius**2, 1e-6))
            _, _, box_width, box_height = cv2.boundingRect(contour)
            inertia = min(box_width, box_height) / max(box_width, box_height, 1)
            shape_score = (
                0.45 * circularity + 0.35 * fill_ratio + 0.20 * inertia
            )
            if shape_score < 0.30:
                continue
            if not _valid_center(x, y, radius, exclusion):
                continue
            candidates.append(
                DetectionEvidence(
                    x=float(x + offset[0]),
                    y=float(y + offset[1]),
                    radius=radius,
                    size_class=size_class,
                    detector=(
                        f"{detector_prefix}contour_watershed"
                        if watershed
                        else f"{detector_prefix}contour"
                    ),
                    circle_type=circle_type,
                    circularity_score=shape_score,
                    raw_score=shape_score,
                )
            )
    return sorted(
        candidates, key=lambda candidate: candidate.raw_score, reverse=True
    )[:max_candidates]


def _large_candidates(
    image: NDArray[np.uint8],
    probability: NDArray[np.float32],
    exclusion: NDArray[np.uint8],
    radius_range: tuple[float, float],
    offset: tuple[int, int],
    *,
    size_class: SizeClass | None = "large",
    detector_prefix: str = "",
    circle_type: Literal["main", "satellite"] = "satellite",
    hough_accumulator_threshold: float = 32.0,
    max_hough_candidates: int = 180,
    max_ellipse_candidates: int = 120,
) -> list[DetectionEvidence]:
    minimum, maximum = radius_range
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    enhanced[exclusion > 0] = int(np.median(enhanced[exclusion == 0])) if np.any(
        exclusion == 0
    ) else 0
    hough_candidates: list[DetectionEvidence] = []
    raw = cv2.HoughCircles(
        cv2.GaussianBlur(enhanced, (7, 7), 1.4),
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=max(8.0, minimum * 0.8),
        param1=115,
        param2=max(10.0, float(hough_accumulator_threshold)),
        minRadius=max(2, int(round(minimum * 0.75))),
        maxRadius=max(3, int(round(maximum * 1.15))),
    )
    if raw is not None:
        for x, y, radius in raw[0]:
            if not _valid_center(float(x), float(y), float(radius), exclusion):
                continue
            hough_candidates.append(
                DetectionEvidence(
                    x=float(x + offset[0]),
                    y=float(y + offset[1]),
                    radius=float(radius),
                    size_class=size_class,
                    detector=f"{detector_prefix}hough",
                    circle_type=circle_type,
                    circularity_score=0.88,
                    raw_score=0.72,
                )
            )
            if len(hough_candidates) >= max_hough_candidates:
                break
    edges = cv2.Canny(enhanced, 45, 135, L2gradient=True)
    edges[exclusion > 0] = 0
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    ellipse_candidates: list[DetectionEvidence] = []
    for contour in contours:
        if len(contour) < 12:
            continue
        ellipse = cv2.fitEllipse(contour)
        (x, y), (diameter_a, diameter_b), _ = ellipse
        if diameter_a <= 0 or diameter_b <= 0:
            continue
        radius = math.sqrt(float(diameter_a) * float(diameter_b)) / 2.0
        if not minimum * 0.72 <= radius <= maximum * 1.25:
            continue
        axis_ratio = min(diameter_a, diameter_b) / max(diameter_a, diameter_b)
        if axis_ratio < 0.48:
            continue
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        contour_circularity = (
            min(1.0, 4 * math.pi * area / max(perimeter**2, 1e-6))
            if area > 0
            else 0.0
        )
        shape_score = 0.65 * float(axis_ratio) + 0.35 * contour_circularity
        if not _valid_center(float(x), float(y), radius, exclusion):
            continue
        ellipse_candidates.append(
            DetectionEvidence(
                x=float(x + offset[0]),
                y=float(y + offset[1]),
                radius=radius,
                size_class=size_class,
                detector=f"{detector_prefix}ellipse",
                circle_type=circle_type,
                circularity_score=shape_score,
                raw_score=shape_score,
            )
        )
    ellipse_candidates.sort(
        key=lambda candidate: candidate.raw_score, reverse=True
    )
    return hough_candidates + ellipse_candidates[:max_ellipse_candidates]


def _ring_symmetry_support(
    gray: NDArray[np.uint8], x: float, y: float, radius: float
) -> float:
    """Measure local circular evidence and explicit 60-degree repeatability."""

    samples = 96
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    height, width = gray.shape
    inner_radius = max(1.0, radius * 0.75)
    outer_radius = radius * 1.18
    inner_x = np.rint(x + inner_radius * np.cos(angles)).astype(np.int32)
    inner_y = np.rint(y + inner_radius * np.sin(angles)).astype(np.int32)
    outer_x = np.rint(x + outer_radius * np.cos(angles)).astype(np.int32)
    outer_y = np.rint(y + outer_radius * np.sin(angles)).astype(np.int32)
    valid = (
        (inner_x >= 0)
        & (inner_x < width)
        & (inner_y >= 0)
        & (inner_y < height)
        & (outer_x >= 0)
        & (outer_x < width)
        & (outer_y >= 0)
        & (outer_y < height)
    )
    if int(valid.sum()) < samples // 2:
        return 0.0
    contrast = np.zeros(samples, dtype=np.float32)
    contrast[valid] = np.abs(
        gray[inner_y[valid], inner_x[valid]].astype(np.float32)
        - gray[outer_y[valid], outer_x[valid]].astype(np.float32)
    )
    valid_contrast = contrast[valid]
    mean = float(np.mean(valid_contrast))
    deviation = float(np.std(valid_contrast))
    uniformity = mean / max(mean + deviation, 1e-6)
    coverage = float(np.mean(valid_contrast >= 8.0))
    shift = samples // 6
    paired = valid & np.roll(valid, shift)
    if np.any(paired):
        difference = np.abs(contrast - np.roll(contrast, shift))
        normalization = max(mean + deviation, 8.0)
        sixfold_similarity = 1.0 - min(
            1.0, float(np.mean(difference[paired])) / normalization
        )
    else:
        sixfold_similarity = 0.0
    return min(
        1.0,
        max(
            0.0,
            0.35 * uniformity
            + 0.25 * coverage
            + 0.40 * sixfold_similarity,
        ),
    )


def _color_score(
    probability: NDArray[np.float32],
    x: float,
    y: float,
    radius: float,
) -> float:
    if probability.size == 0 or float(np.max(probability)) <= 0.01:
        return 0.5
    height, width = probability.shape
    left = max(0, int(math.floor(x - radius * 1.4)))
    right = min(width, int(math.ceil(x + radius * 1.4 + 1)))
    top = max(0, int(math.floor(y - radius * 1.4)))
    bottom = min(height, int(math.ceil(y + radius * 1.4 + 1)))
    if left >= right or top >= bottom:
        return 0.0
    yy, xx = np.ogrid[top:bottom, left:right]
    distance_squared = (xx - x) ** 2 + (yy - y) ** 2
    inner = distance_squared <= (radius * 0.78) ** 2
    annulus = (distance_squared >= (radius * 1.05) ** 2) & (
        distance_squared <= (radius * 1.35) ** 2
    )
    patch = probability[top:bottom, left:right]
    if not np.any(inner):
        return 0.0
    inner_mean = float(np.mean(patch[inner]))
    annulus_mean = float(np.mean(patch[annulus])) if np.any(annulus) else 0.0
    return min(
        1.0,
        max(0.0, 0.72 * inner_mean + 0.28 * max(0.0, inner_mean - annulus_mean)),
    )


def _score_candidates(
    candidates: Iterable[DetectionEvidence],
    image: NDArray[np.uint8],
    probability: NDArray[np.float32],
) -> list[DetectionEvidence]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    height, width = gray.shape
    scored: list[DetectionEvidence] = []
    for candidate in candidates:
        if not (
            0 <= candidate.x < width
            and 0 <= candidate.y < height
            and candidate.radius > 0
        ):
            continue
        candidate.edge_score = _edge_confidence(
            gray,
            gradient_x,
            gradient_y,
            candidate.x,
            candidate.y,
            candidate.radius,
        )
        candidate.color_score = _color_score(
            probability,
            candidate.x,
            candidate.y,
            candidate.radius,
        )
        candidate.symmetry_support = _ring_symmetry_support(
            gray,
            candidate.x,
            candidate.y,
            candidate.radius,
        )
        scored.append(candidate)
    return scored


def _same_circle(left: DetectionEvidence, right: DetectionEvidence) -> bool:
    center_distance = math.hypot(left.x - right.x, left.y - right.y)
    radius_difference = abs(left.radius - right.radius) / max(
        left.radius, right.radius, 1e-6
    )
    return (
        center_distance <= max(3.0, 0.42 * min(left.radius, right.radius))
        and radius_difference <= 0.42
    )


def _preliminary_score(candidate: DetectionEvidence) -> float:
    return (
        0.28 * candidate.edge_score
        + 0.20 * candidate.color_score
        + 0.24 * candidate.circularity_score
        + 0.18 * candidate.symmetry_support
        + 0.10 * candidate.raw_score
    )


def _detector_family(detector: str) -> str:
    """Collapse role prefixes while retaining the underlying algorithm."""

    parts = detector.split("_")
    if parts and parts[0] == "main" and len(parts) > 1:
        return parts[1]
    return parts[0] if parts else detector


def _spatial_clusters(
    candidates: Sequence[DetectionEvidence],
) -> list[list[DetectionEvidence]]:
    """Cluster nearby evidence with a small spatial hash instead of O(n²) scans."""

    if not candidates:
        return []
    cell_size = 12.0
    parents = list(range(len(candidates)))
    ranks = [0] * len(candidates)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if ranks[left_root] < ranks[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        if ranks[left_root] == ranks[right_root]:
            ranks[left_root] += 1

    buckets: dict[tuple[int, int], list[int]] = {}
    for index, candidate in enumerate(candidates):
        cell_x = int(math.floor(candidate.x / cell_size))
        cell_y = int(math.floor(candidate.y / cell_size))
        search_radius = max(3.0, 0.42 * candidate.radius)
        cell_reach = max(1, int(math.ceil(search_radius / cell_size)))
        for candidate_cell_y in range(
            cell_y - cell_reach, cell_y + cell_reach + 1
        ):
            for candidate_cell_x in range(
                cell_x - cell_reach, cell_x + cell_reach + 1
            ):
                for other_index in buckets.get(
                    (candidate_cell_x, candidate_cell_y), ()
                ):
                    if _same_circle(candidate, candidates[other_index]):
                        union(index, other_index)
        buckets.setdefault((cell_x, cell_y), []).append(index)

    grouped: dict[int, list[DetectionEvidence]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(find(index), []).append(candidate)
    return list(grouped.values())


def _reduce_evidence_for_scoring(
    candidates: Sequence[DetectionEvidence],
) -> list[DetectionEvidence]:
    """Keep the strongest observation from each detector in a raw cluster."""

    reduced: list[DetectionEvidence] = []
    for cluster in _spatial_clusters(candidates):
        strongest_by_detector: dict[str, DetectionEvidence] = {}
        for candidate in cluster:
            previous = strongest_by_detector.get(candidate.detector)
            raw_quality = candidate.raw_score + candidate.circularity_score
            previous_quality = (
                -1.0
                if previous is None
                else previous.raw_score + previous.circularity_score
            )
            if raw_quality > previous_quality:
                strongest_by_detector[candidate.detector] = candidate
        reduced.extend(strongest_by_detector.values())
    return reduced


def fuse_candidates(
    candidates: Sequence[DetectionEvidence],
    *,
    min_final_confidence: float,
    main_arm_min_final_confidence: float | None = None,
    processed_to_source: NDArray[np.float64],
    max_candidates: int = 700,
) -> list[CircleFeature]:
    clusters = _spatial_clusters(candidates)
    circles: list[CircleFeature] = []
    for cluster in clusters:
        main_evidence = [
            item for item in cluster if item.circle_type == "main"
        ]
        satellite_evidence = [
            item for item in cluster if item.circle_type == "satellite"
        ]
        main_families = {
            _detector_family(item.detector) for item in main_evidence
        }
        satellite_families = {
            _detector_family(item.detector)
            for item in satellite_evidence
        }
        circle_type: Literal["main", "satellite"] = (
            "main"
            if main_evidence
            and (
                not satellite_evidence
                or len(main_families) >= len(satellite_families)
            )
            else "satellite"
        )
        weights = np.asarray(
            [max(0.05, _preliminary_score(item)) for item in cluster],
            dtype=np.float64,
        )
        weights /= float(np.sum(weights))
        x = float(sum(item.x * weight for item, weight in zip(cluster, weights)))
        y = float(sum(item.y * weight for item, weight in zip(cluster, weights)))
        radius = float(
            sum(item.radius * weight for item, weight in zip(cluster, weights))
        )
        edge_score = float(
            sum(item.edge_score * weight for item, weight in zip(cluster, weights))
        )
        color_score = float(
            sum(item.color_score * weight for item, weight in zip(cluster, weights))
        )
        circularity_score = float(
            sum(
                item.circularity_score * weight
                for item, weight in zip(cluster, weights)
            )
        )
        symmetry_support = float(
            sum(
                item.symmetry_support * weight
                for item, weight in zip(cluster, weights)
            )
        )
        detector_families = {
            _detector_family(item.detector) for item in cluster
        }
        detector_agreement = min(
            1.0, 0.35 + 0.325 * (len(detector_families) - 1)
        )
        final_confidence = min(
            1.0,
            max(
                0.0,
                0.25 * edge_score
                + 0.20 * color_score
                + 0.20 * circularity_score
                + 0.15 * detector_agreement
                + 0.20 * symmetry_support,
            ),
        )
        required_confidence = min_final_confidence
        if (
            main_arm_min_final_confidence is not None
            and circle_type == "main"
        ):
            required_confidence = max(
                required_confidence,
                float(main_arm_min_final_confidence),
            )
        if final_confidence < required_confidence:
            continue
        role_evidence = [
            item for item in cluster if item.circle_type == circle_type
        ]
        best = max(role_evidence, key=_preliminary_score)
        source_x, source_y = transform_point(
            (x, y), processed_to_source
        )
        source_radius = transform_radius(
            (x, y), radius, processed_to_source
        )
        circles.append(
            CircleFeature(
                id="pending",
                x=x,
                y=y,
                processed_x=x,
                processed_y=y,
                source_x=source_x,
                source_y=source_y,
                source_radius=source_radius,
                corrected_x=x,
                corrected_y=y,
                corrected_radius=radius,
                corrected_major_axis=radius * 2.0,
                corrected_minor_axis=radius * 2.0,
                source_major_axis=source_radius * 2.0,
                source_minor_axis=source_radius * 2.0,
                radius=radius,
                confidence=final_confidence,
                final_confidence=final_confidence,
                edge_score=edge_score,
                color_score=color_score,
                circularity_score=circularity_score,
                detector_agreement=detector_agreement,
                symmetry_support=symmetry_support,
                size_class=(
                    None if circle_type == "main" else best.size_class
                ),
                detector="+".join(sorted({item.detector for item in cluster})),
                circle_type=circle_type,
                source="automatic",
            )
        )
    ranked = sorted(
        circles,
        key=lambda circle: (
            -circle.final_confidence,
            -float(circle.detector_agreement or 0.0),
            -float(circle.edge_score or 0.0),
            round(circle.y, 2),
            round(circle.x, 2),
        ),
    )[:max_candidates]
    ranked.sort(
        key=lambda circle: (
            round(circle.y, 2),
            round(circle.x, 2),
            circle.radius,
        )
    )
    for index, circle in enumerate(ranked, start=1):
        circle.id = f"auto-{index:04d}"
    return ranked


def _overlaps_locked(candidate: CircleFeature, locked: CircleFeature) -> bool:
    distance = math.hypot(candidate.x - locked.x, candidate.y - locked.y)
    radius_difference = abs(candidate.radius - locked.radius) / max(
        candidate.radius, locked.radius, 1e-6
    )
    return (
        distance <= max(3.0, 0.5 * min(candidate.radius, locked.radius))
        and radius_difference <= 0.5
    )


def detect_multiscale_circles(
    result: PreprocessResult,
    *,
    locked_circles: Iterable[CircleFeature] = (),
    max_candidates: int = 700,
) -> list[CircleFeature]:
    """Run native-resolution, overlapping-tile detectors and preserve locks."""

    profile = result.profile
    image = result.measurement_image
    probability = result.foreground_probability
    exclusion = result.exclusion_mask
    radius_ranges = effective_radius_ranges(
        profile, result.source_to_processed
    )
    tile_size = int(profile.detection.get("tile_size", 512))
    large_tile_size = int(
        profile.detection.get("large_tile_size", max(1024, tile_size))
    )
    overlap = float(profile.detection.get("tile_overlap", 0.2))
    watershed = bool(profile.detection.get("watershed", False))
    hough_accumulator_threshold = float(
        profile.detection.get("hough_accumulator_threshold", 32.0)
    )
    raw_main_arm = profile.detection.get("main_arm", {})
    main_arm_settings = (
        dict(raw_main_arm) if isinstance(raw_main_arm, Mapping) else {}
    )
    main_arm_enabled = bool(main_arm_settings.get("enabled", True))
    main_arm_min_confidence = float(
        main_arm_settings.get("min_final_confidence", 0.50)
    )
    all_evidence: list[DetectionEvidence] = []
    # Downsampling is allowed for large structures; no scale exceeds 1.0.
    scale_plan: dict[SizeClass, tuple[float, ...]] = {
        "small": (1.0,),
        "medium": (1.0,),
        "large": (1.0, 0.75),
    }
    for size_class in ("small", "medium", "large"):
        if not bool(profile.size_classes[size_class].get("enabled", True)):
            continue
        base_minimum, base_maximum = radius_ranges[size_class]
        for scale in scale_plan[size_class]:
            if scale > 1.0:
                raise AssertionError("Detection must never upscale source pixels")
            if scale == 1.0:
                scaled_image = image
                scaled_probability = probability
                scaled_exclusion = exclusion
            else:
                scaled_image = cv2.resize(
                    image,
                    (
                        max(2, int(round(image.shape[1] * scale))),
                        max(2, int(round(image.shape[0] * scale))),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                scaled_probability = cv2.resize(
                    probability,
                    (scaled_image.shape[1], scaled_image.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                scaled_exclusion = cv2.resize(
                    exclusion,
                    (scaled_image.shape[1], scaled_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            scaled_range = (base_minimum * scale, base_maximum * scale)
            class_tile_size = (
                max(tile_size, large_tile_size)
                if size_class == "large"
                else tile_size
            )
            scaled_tile = max(128, int(round(class_tile_size * scale)))
            for left, top, right, bottom in _tiles(
                scaled_image.shape[1],
                scaled_image.shape[0],
                scaled_tile,
                overlap,
            ):
                tile_image = scaled_image[top:bottom, left:right]
                tile_probability = scaled_probability[top:bottom, left:right]
                tile_exclusion = scaled_exclusion[top:bottom, left:right]
                if size_class == "small":
                    evidence = _dog_candidates(
                        cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY),
                        tile_probability,
                        tile_exclusion,
                        scaled_range,
                        (left, top),
                    )
                elif size_class == "medium":
                    evidence = _contour_candidates(
                        tile_image,
                        tile_probability,
                        tile_exclusion,
                        scaled_range,
                        (left, top),
                        watershed=watershed,
                    )
                else:
                    evidence = _large_candidates(
                        tile_image,
                        tile_probability,
                        tile_exclusion,
                        scaled_range,
                        (left, top),
                        hough_accumulator_threshold=(
                            hough_accumulator_threshold
                        ),
                    )
                if scale != 1.0:
                    for item in evidence:
                        item.x /= scale
                        item.y /= scale
                        item.radius /= scale
                all_evidence.extend(evidence)

    # Main-arm circles intentionally do not inherit the three sampled side
    # standards. They are searched over one continuous processed-pixel range.
    # When side exemplars exist, their calibrated bands remain authoritative
    # and are omitted from this pass to avoid duplicate role lanes.
    if main_arm_enabled:
        main_minimum, main_maximum = effective_main_arm_radius_range(profile)
        sampled_side_ranges = [
            radius_ranges[size_class]
            for size_class in ("small", "medium", "large")
            if profile.size_classes[size_class].get("samples")
        ]
        exclude_sampled_side_ranges = bool(
            main_arm_settings.get("exclude_sampled_side_ranges", True)
        )
        main_hough_threshold = float(
            main_arm_settings.get("hough_accumulator_threshold", 48.0)
        )
        for left, top, right, bottom in _tiles(
            image.shape[1],
            image.shape[0],
            max(tile_size, large_tile_size),
            overlap,
        ):
            tile_image = image[top:bottom, left:right]
            tile_probability = probability[top:bottom, left:right]
            tile_exclusion = exclusion[top:bottom, left:right]
            main_evidence = _contour_candidates(
                tile_image,
                tile_probability,
                tile_exclusion,
                (main_minimum, main_maximum),
                (left, top),
                watershed=watershed,
                size_class=None,
                detector_prefix="main_",
                circle_type="main",
            )
            main_evidence.extend(
                _large_candidates(
                    tile_image,
                    tile_probability,
                    tile_exclusion,
                    (main_minimum, main_maximum),
                    (left, top),
                    size_class=None,
                    detector_prefix="main_",
                    circle_type="main",
                    hough_accumulator_threshold=main_hough_threshold,
                )
            )
            for item in main_evidence:
                if not main_minimum <= item.radius <= main_maximum:
                    continue
                if exclude_sampled_side_ranges and any(
                    minimum <= item.radius <= maximum
                    for minimum, maximum in sampled_side_ranges
                ):
                    continue
                all_evidence.append(item)
    reduced_evidence = _reduce_evidence_for_scoring(all_evidence)
    scored = _score_candidates(reduced_evidence, image, probability)
    automatic = fuse_candidates(
        scored,
        min_final_confidence=float(
            profile.detection.get("min_final_confidence", 0.2)
        ),
        main_arm_min_final_confidence=(
            main_arm_min_confidence if main_arm_enabled else None
        ),
        processed_to_source=result.processed_to_source,
        max_candidates=max_candidates,
    )
    locked_items: list[CircleFeature] = []
    for circle in locked_circles:
        source_point = (
            float(circle.source_x if circle.source_x is not None else circle.x),
            float(circle.source_y if circle.source_y is not None else circle.y),
        )
        processed_x, processed_y = transform_point(
            source_point, result.source_to_processed
        )
        source_radius = float(
            circle.source_radius
            if circle.source_radius is not None
            else circle.radius
        )
        processed_radius = transform_radius(
            source_point,
            source_radius,
            result.source_to_processed,
        )
        locked_items.append(
            replace(
                circle,
                x=processed_x,
                y=processed_y,
                radius=processed_radius,
                processed_x=processed_x,
                processed_y=processed_y,
                source_x=source_point[0],
                source_y=source_point[1],
                source_radius=source_radius,
                corrected_x=processed_x,
                corrected_y=processed_y,
                corrected_radius=processed_radius,
                locked=True,
            )
        )
    automatic = [
        candidate
        for candidate in automatic
        if not any(
            _overlaps_locked(candidate, locked) for locked in locked_items
        )
    ]
    occupied_ids = {circle.id for circle in locked_items}
    next_id = 1
    for candidate in automatic:
        while f"auto-{next_id:04d}" in occupied_ids:
            next_id += 1
        candidate.id = f"auto-{next_id:04d}"
        occupied_ids.add(candidate.id)
        next_id += 1
    return locked_items + automatic
