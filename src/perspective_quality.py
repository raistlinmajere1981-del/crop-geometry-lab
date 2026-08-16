"""Non-destructive validation of perspective correction quality.

The routines in this module compare source and rectified geometry.  They never
mutate circle annotations, never relabel structure, and never apply a second
geometric correction.  In particular, no residual/S-drift correction is
performed here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .feature_table import CircleFeature

Point = tuple[float, float]

DEFAULT_QUALITY_WEIGHTS: dict[str, float] = {
    "tramline": 0.30,
    "roundness": 0.30,
    "arm_symmetry": 0.25,
    "k5k6": 0.10,
    "warp_penalty": 0.05,
}


@dataclass(slots=True)
class PerspectiveQualityResult:
    """A JSON report, tabular rows, and review-only raster previews."""

    report: dict[str, Any]
    circle_metrics: list[dict[str, Any]]
    arm_metrics: list[dict[str, Any]]
    tramline_metrics: list[dict[str, Any]]
    images: dict[str, NDArray[np.uint8]]


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return min(upper, max(lower, float(value)))


def _rounded(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _point(value: Any) -> Point:
    if isinstance(value, Mapping):
        x, y = value.get("x"), value.get("y")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("A point requires two values")
        x, y = value
    else:
        raise ValueError("A point must be an object or a pair")
    result = (float(x), float(y))
    if not all(math.isfinite(item) for item in result):
        raise ValueError("Point coordinates must be finite")
    return result


def _matrix(value: Any) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("Perspective quality requires a finite 3x3 homography")
    if abs(float(np.linalg.det(result))) < 1e-12:
        raise ValueError("Perspective homography is singular")
    return result


def _transform_points(
    points: Sequence[Point],
    matrix: NDArray[np.float64],
) -> list[Point]:
    if not points:
        return []
    source = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(source, matrix).reshape(-1, 2)
    return [(float(item[0]), float(item[1])) for item in transformed]


def _transform_point(point: Point, matrix: NDArray[np.float64]) -> Point:
    return _transform_points([point], matrix)[0]


def _orientation_difference(left: float, right: float) -> float:
    difference = abs(float(left) - float(right)) % 180.0
    return min(difference, 180.0 - difference)


def _orientation_mean(angles: Sequence[float]) -> float:
    if not angles:
        return 0.0
    radians = np.radians(np.asarray(angles, dtype=np.float64) * 2.0)
    return float(
        (math.degrees(math.atan2(float(np.sin(radians).mean()), float(np.cos(radians).mean()))) / 2.0)
        % 180.0
    )


def _orientation_std(angles: Sequence[float]) -> float | None:
    if not angles:
        return None
    mean = _orientation_mean(angles)
    residuals = np.asarray(
        [
            (
                ((float(angle) - mean + 90.0) % 180.0)
                - 90.0
            )
            for angle in angles
        ],
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(residuals**2)))


def _line_geometry(p1: Point, p2: Point) -> dict[str, Any]:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        raise ValueError("Tramline endpoints must be different")
    a, b = -dy / length, dx / length
    c = -(a * p1[0] + b * p1[1])
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    return {
        "p1": {"x": p1[0], "y": p1[1]},
        "p2": {"x": p2[0], "y": p2[1]},
        "angle_degrees": angle,
        "length": length,
        "line_a": a,
        "line_b": b,
        "line_c": c,
        "midpoint": ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0),
    }


def _tramline_space_summary(
    geometries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not geometries:
        return {
            "line_count": 0,
            "dominant_angle_degrees": None,
            "angle_std": None,
            "mean_spacing": None,
            "spacing_cv": None,
            "parallelism_score": None,
        }
    angles = [float(item["angle_degrees"]) for item in geometries]
    dominant = _orientation_mean(angles)
    angle_std = _orientation_std(angles) or 0.0
    normal = np.asarray(
        [
            -math.sin(math.radians(dominant)),
            math.cos(math.radians(dominant)),
        ],
        dtype=np.float64,
    )
    offsets = sorted(
        float(np.dot(np.asarray(item["midpoint"], dtype=np.float64), normal))
        for item in geometries
    )
    differences = np.diff(np.asarray(offsets, dtype=np.float64))
    positive = differences[differences > 1e-6]
    mean_spacing: float | None = None
    spacing_cv: float | None = None
    if positive.size:
        mean_spacing = float(np.mean(positive))
        spacing_cv = (
            0.0
            if positive.size == 1
            else float(np.std(positive) / max(mean_spacing, 1e-6))
        )
    angle_score = 100.0 * math.exp(-angle_std / 7.5)
    if spacing_cv is None:
        spacing_score = 50.0
    elif len(geometries) < 3:
        spacing_score = 50.0
    else:
        spacing_score = 100.0 * math.exp(-spacing_cv / 0.30)
    score = _clamp(0.72 * angle_score + 0.28 * spacing_score)
    return {
        "line_count": len(geometries),
        "dominant_angle_degrees": _rounded(dominant),
        "angle_std": _rounded(angle_std),
        "mean_spacing": _rounded(mean_spacing),
        "spacing_cv": _rounded(spacing_cv),
        "parallelism_score": _rounded(score),
    }


def measure_tramline_quality(
    lines: Sequence[Mapping[str, Any]],
    source_to_corrected: Any,
) -> dict[str, Any]:
    """Measure an approved/manual line family before and after homography."""

    matrix = _matrix(source_to_corrected)
    rows: list[dict[str, Any]] = []
    before_geometries: list[dict[str, Any]] = []
    after_geometries: list[dict[str, Any]] = []
    for index, raw in enumerate(lines, start=1):
        try:
            source_p1 = _point(raw.get("source_p1", raw.get("p1")))
            source_p2 = _point(raw.get("source_p2", raw.get("p2")))
            before = _line_geometry(source_p1, source_p2)
            corrected_p1, corrected_p2 = _transform_points(
                [source_p1, source_p2], matrix
            )
            after = _line_geometry(corrected_p1, corrected_p2)
        except (KeyError, TypeError, ValueError):
            continue
        before_geometries.append(before)
        after_geometries.append(after)
        rows.append(
            {
                "id": str(raw.get("id") or f"tramline-{index:02d}"),
                "manual": bool(raw.get("manual", False)),
                "support": _rounded(float(raw.get("support", 1.0))),
                "source_p1_x": _rounded(source_p1[0]),
                "source_p1_y": _rounded(source_p1[1]),
                "source_p2_x": _rounded(source_p2[0]),
                "source_p2_y": _rounded(source_p2[1]),
                "corrected_p1_x": _rounded(corrected_p1[0]),
                "corrected_p1_y": _rounded(corrected_p1[1]),
                "corrected_p2_x": _rounded(corrected_p2[0]),
                "corrected_p2_y": _rounded(corrected_p2[1]),
                "angle_before": _rounded(float(before["angle_degrees"])),
                "angle_after": _rounded(float(after["angle_degrees"])),
                "line_a_before": _rounded(float(before["line_a"])),
                "line_b_before": _rounded(float(before["line_b"])),
                "line_c_before": _rounded(float(before["line_c"])),
                "line_a_after": _rounded(float(after["line_a"])),
                "line_b_after": _rounded(float(after["line_b"])),
                "line_c_after": _rounded(float(after["line_c"])),
                "length_before": _rounded(float(before["length"])),
                "length_after": _rounded(float(after["length"])),
            }
        )
    before = _tramline_space_summary(before_geometries)
    after = _tramline_space_summary(after_geometries)
    gain = (
        None
        if before["parallelism_score"] is None
        or after["parallelism_score"] is None
        else float(after["parallelism_score"]) - float(before["parallelism_score"])
    )
    comment = "Tarla izi ailesi ölçülemedi."
    if gain is not None:
        if gain > 4.0:
            comment = "Açı ve aralık tutarlılığı düzeltme sonrasında arttı."
        elif gain < -4.0:
            comment = "Tarla izi tutarlılığı düştü; referans seçimini yeniden kontrol edin."
        else:
            comment = "Tarla izi geometrisindeki değişim sınırlı."
    return {
        "before": before,
        "after": after,
        "gain": _rounded(gain),
        "comment": comment,
        "lines": rows,
    }


def _gradient_image(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.1)
    dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(dx, dy)


def _ellipse_payload(
    center: Point,
    major_axis: float,
    minor_axis: float,
    angle_degrees: float,
    residual: float | None,
    *,
    fit_method: str,
    edge_samples: int,
) -> dict[str, Any]:
    ratio = minor_axis / max(major_axis, 1e-9)
    score = (
        None
        if residual is None
        else _clamp(
            70.0 * ratio
            + 30.0 * math.exp(-5.0 * max(0.0, residual))
        )
    )
    return {
        "center_x": _rounded(center[0]),
        "center_y": _rounded(center[1]),
        "major_axis": _rounded(major_axis),
        "minor_axis": _rounded(minor_axis),
        "angle_degrees": _rounded(angle_degrees % 180.0),
        "axis_ratio": _rounded(ratio),
        "ellipticity": _rounded(1.0 - ratio),
        "residual": _rounded(residual),
        "circularity_score": _rounded(score),
        "fit_method": fit_method,
        "edge_sample_count": int(edge_samples),
    }


def _standardize_cv_ellipse(
    fitted: tuple[tuple[float, float], tuple[float, float], float],
    points: NDArray[np.float32],
) -> dict[str, Any]:
    (cx, cy), (width, height), raw_angle = fitted
    if width >= height:
        major, minor, angle = float(width), float(height), float(raw_angle)
    else:
        major, minor, angle = float(height), float(width), float(raw_angle + 90.0)
    theta = math.radians(angle)
    cosine, sine = math.cos(theta), math.sin(theta)
    dx = points[:, 0] - float(cx)
    dy = points[:, 1] - float(cy)
    local_x = dx * cosine + dy * sine
    local_y = -dx * sine + dy * cosine
    normalized_radius = np.sqrt(
        (local_x / max(major / 2.0, 1e-6)) ** 2
        + (local_y / max(minor / 2.0, 1e-6)) ** 2
    )
    residual = float(np.median(np.abs(normalized_radius - 1.0)))
    return _ellipse_payload(
        (float(cx), float(cy)),
        major,
        minor,
        angle,
        residual,
        fit_method="edge_fit",
        edge_samples=len(points),
    )


def _fit_edge_ellipse(
    gradient: NDArray[np.float32],
    center: Point,
    radius: float,
) -> dict[str, Any] | None:
    if radius < 3.0 or not all(math.isfinite(value) for value in (*center, radius)):
        return None
    height, width = gradient.shape[:2]
    if center[0] < -radius or center[1] < -radius:
        return None
    if center[0] > width + radius or center[1] > height + radius:
        return None
    angle_count = int(np.clip(round(radius * 3.0), 72, 180))
    radial_count = int(np.clip(round(radius * 1.4), 24, 80))
    angles = np.linspace(0.0, math.tau, angle_count, endpoint=False)
    radii = np.linspace(radius * 0.55, radius * 1.45, radial_count)
    map_x = (
        center[0]
        + np.cos(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    map_y = (
        center[1]
        + np.sin(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    sampled = cv2.remap(
        gradient,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    indices = np.argmax(sampled, axis=1)
    peaks = sampled[np.arange(angle_count), indices]
    nonzero = peaks[peaks > 1e-6]
    if nonzero.size < max(18, angle_count // 3):
        return None
    threshold = max(1.0, float(np.percentile(nonzero, 22)) * 0.45)
    keep = peaks >= threshold
    selected_angles = angles[keep]
    selected_radii = radii[indices[keep]]
    points = np.column_stack(
        [
            center[0] + np.cos(selected_angles) * selected_radii,
            center[1] + np.sin(selected_angles) * selected_radii,
        ]
    ).astype(np.float32)
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    points = points[inside]
    if len(points) < max(24, angle_count // 3):
        return None
    fitted = cv2.fitEllipse(points.reshape(-1, 1, 2))
    payload = _standardize_cv_ellipse(fitted, points)
    fitted_center = (
        float(payload["center_x"]),
        float(payload["center_y"]),
    )
    center_offset = math.dist(center, fitted_center)
    major = float(payload["major_axis"])
    minor = float(payload["minor_axis"])
    if (
        center_offset > radius * 0.65
        or major < radius * 0.75
        or major > radius * 3.3
        or minor < radius * 0.45
    ):
        return None
    return payload


def _project_circle(
    center: Point,
    radius: float,
    matrix: NDArray[np.float64],
) -> dict[str, Any]:
    angles = np.linspace(0.0, math.tau, 96, endpoint=False)
    source = [
        (
            center[0] + math.cos(float(angle)) * radius,
            center[1] + math.sin(float(angle)) * radius,
        )
        for angle in angles
    ]
    transformed = np.asarray(_transform_points(source, matrix), dtype=np.float32)
    fitted = cv2.fitEllipse(transformed.reshape(-1, 1, 2))
    payload = _standardize_cv_ellipse(fitted, transformed)
    payload["fit_method"] = "projected_annotation"
    payload["residual"] = None
    payload["circularity_score"] = None
    return payload


def _equivalent_radius(
    center: Point,
    radius: float,
    matrix: NDArray[np.float64],
) -> float:
    ellipse = _project_circle(center, radius, matrix)
    return math.sqrt(
        float(ellipse["major_axis"]) * float(ellipse["minor_axis"])
    ) / 2.0


def _circle_source_geometry(
    circle: CircleFeature,
    corrected_to_source: NDArray[np.float64],
) -> tuple[Point, float]:
    source_center = (
        float(circle.source_x if circle.source_x is not None else circle.x),
        float(circle.source_y if circle.source_y is not None else circle.y),
    )
    source_radius = getattr(circle, "source_radius", None)
    if source_radius is not None and float(source_radius) > 0:
        return source_center, float(source_radius)
    corrected_x = getattr(circle, "corrected_x", None)
    corrected_y = getattr(circle, "corrected_y", None)
    corrected_radius = getattr(circle, "corrected_radius", None)
    corrected_center = (
        float(circle.x if corrected_x is None else corrected_x),
        float(circle.y if corrected_y is None else corrected_y),
    )
    return source_center, _equivalent_radius(
        corrected_center,
        float(circle.radius if corrected_radius is None else corrected_radius),
        corrected_to_source,
    )


def _stats(values: Iterable[float | None]) -> dict[str, Any]:
    array = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "std": None}
    return {
        "count": int(array.size),
        "mean": _rounded(float(np.mean(array))),
        "median": _rounded(float(np.median(array))),
        "std": _rounded(float(np.std(array))),
    }


def _roundness_group(
    circle_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paired = [
        item
        for item in circle_metrics
        if item.get("circularity_score_before") is not None
        and item.get("circularity_score_after") is not None
    ]
    before_scores = _stats(
        item.get("circularity_score_before") for item in paired
    )
    after_scores = _stats(
        item.get("circularity_score_after") for item in paired
    )
    return {
        "circle_count": len(circle_metrics),
        "paired_edge_fit_count": len(paired),
        "before": {
            "ellipse_ratio": _stats(
                item.get("ellipse_ratio_before") for item in paired
            ),
            "ellipticity": _stats(
                item.get("ellipticity_before") for item in paired
            ),
            "circularity_residual": _stats(
                item.get("circularity_residual_before") for item in paired
            ),
            "circularity_score": before_scores,
        },
        "after": {
            "ellipse_ratio": _stats(
                item.get("ellipse_ratio_after") for item in paired
            ),
            "ellipticity": _stats(
                item.get("ellipticity_after") for item in paired
            ),
            "circularity_residual": _stats(
                item.get("circularity_residual_after") for item in paired
            ),
            "circularity_score": after_scores,
        },
        "score_before": before_scores["mean"],
        "score_after": after_scores["mean"],
        "gain": (
            None
            if before_scores["mean"] is None or after_scores["mean"] is None
            else _rounded(
                float(after_scores["mean"]) - float(before_scores["mean"])
            )
        ),
    }


def _measure_circles(
    source_image: NDArray[np.uint8],
    corrected_image: NDArray[np.uint8],
    circles: Sequence[CircleFeature],
    source_to_corrected: NDArray[np.float64],
    selected_circle_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corrected_to_source = np.linalg.inv(source_to_corrected)
    source_gradient = _gradient_image(source_image)
    corrected_gradient = _gradient_image(corrected_image)
    rows: list[dict[str, Any]] = []
    for circle in circles:
        source_center, source_radius = _circle_source_geometry(
            circle, corrected_to_source
        )
        projected = _project_circle(
            source_center, source_radius, source_to_corrected
        )
        corrected_center = (
            float(projected["center_x"]),
            float(projected["center_y"]),
        )
        corrected_radius = math.sqrt(
            float(projected["major_axis"]) * float(projected["minor_axis"])
        ) / 2.0
        source_fit = _fit_edge_ellipse(
            source_gradient, source_center, source_radius
        )
        corrected_fit = _fit_edge_ellipse(
            corrected_gradient, corrected_center, corrected_radius
        )
        source_axes = source_fit or _ellipse_payload(
            source_center,
            source_radius * 2.0,
            source_radius * 2.0,
            0.0,
            None,
            fit_method="annotation_fallback",
            edge_samples=0,
        )
        corrected_axes = corrected_fit or projected
        before_score = (
            None
            if source_fit is None
            else source_fit["circularity_score"]
        )
        after_score = (
            None
            if corrected_fit is None
            else corrected_fit["circularity_score"]
        )
        benefit = (
            None
            if before_score is None or after_score is None
            else float(after_score) - float(before_score)
        )
        rows.append(
            {
                "circle_id": circle.id,
                "circle_type": circle.circle_type,
                "arm_id": circle.arm_id,
                "sequence_index": circle.sequence_index,
                "selected": circle.id in selected_circle_ids,
                "source_x": _rounded(source_center[0]),
                "source_y": _rounded(source_center[1]),
                "source_radius": _rounded(source_radius),
                "corrected_x": corrected_axes["center_x"],
                "corrected_y": corrected_axes["center_y"],
                "corrected_radius": _rounded(corrected_radius),
                "source_major_axis": source_axes["major_axis"],
                "source_minor_axis": source_axes["minor_axis"],
                "corrected_major_axis": corrected_axes["major_axis"],
                "corrected_minor_axis": corrected_axes["minor_axis"],
                "source_ellipse_angle": source_axes["angle_degrees"],
                "corrected_ellipse_angle": corrected_axes["angle_degrees"],
                "ellipse_ratio_before": (
                    None if source_fit is None else source_fit["axis_ratio"]
                ),
                "ellipse_ratio_after": (
                    None if corrected_fit is None else corrected_fit["axis_ratio"]
                ),
                "ellipticity_before": (
                    None if source_fit is None else source_fit["ellipticity"]
                ),
                "ellipticity_after": (
                    None if corrected_fit is None else corrected_fit["ellipticity"]
                ),
                "circularity_residual_before": (
                    None if source_fit is None else source_fit["residual"]
                ),
                "circularity_residual_after": (
                    None if corrected_fit is None else corrected_fit["residual"]
                ),
                "circularity_score_before": _rounded(
                    None if before_score is None else float(before_score)
                ),
                "circularity_score_after": _rounded(
                    None if after_score is None else float(after_score)
                ),
                "circularity_gain": _rounded(benefit),
                "source_fit_method": source_axes["fit_method"],
                "corrected_fit_method": corrected_axes["fit_method"],
                "source_edge_sample_count": source_axes["edge_sample_count"],
                "corrected_edge_sample_count": corrected_axes[
                    "edge_sample_count"
                ],
            }
        )
    main_rows = [
        row for row in rows if str(row["circle_type"]) == "main"
    ]
    largest = sorted(
        main_rows,
        key=lambda item: float(item["source_radius"] or 0.0),
        reverse=True,
    )[:18]
    selected = [row for row in rows if bool(row["selected"])]
    groups = {
        "all_main": _roundness_group(main_rows),
        "largest_18_main": _roundness_group(largest),
        "selected": _roundness_group(selected),
    }
    active = groups["all_main"]
    gain = active["gain"]
    if active["paired_edge_fit_count"] == 0:
        comment = "Ana çemberlerde güvenilir çift taraflı kenar/elips uyumu bulunamadı."
    elif gain is not None and float(gain) > 3.0:
        comment = "Ana çemberler düzeltme sonrasında daha yuvarlak ve daha düşük artık hatalı."
    elif gain is not None and float(gain) < -3.0:
        comment = "Düzeltme ana çemberlerin daireselliğini azalttı."
    else:
        comment = "Dairesellik değişimi sınırlı."
    return rows, {"groups": groups, "active_group": "all_main", "comment": comment}


def _circular_phase_std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    radians = np.radians(np.asarray(values, dtype=np.float64))
    mean = math.atan2(float(np.sin(radians).mean()), float(np.cos(radians).mean()))
    residuals = np.angle(np.exp(1j * (radians - mean)))
    return float(np.degrees(np.sqrt(np.mean(residuals**2))))


def _coefficient_of_variation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = float(np.mean(values))
    if abs(mean) < 1e-9:
        return None
    return float(np.std(values) / abs(mean))


def _main_chain_records(
    circles: Sequence[CircleFeature],
    structure_state: Mapping[str, Any] | None,
) -> list[tuple[int, int, CircleFeature]]:
    by_id = {circle.id: circle for circle in circles}
    raw_chains = (
        structure_state.get("main_chains", {})
        if isinstance(structure_state, Mapping)
        else {}
    )
    records: list[tuple[int, int, CircleFeature]] = []
    seen: set[str] = set()
    if isinstance(raw_chains, Mapping):
        for arm in range(1, 7):
            raw_ids = raw_chains.get(str(arm), raw_chains.get(arm, []))
            if not isinstance(raw_ids, Sequence) or isinstance(
                raw_ids, (str, bytes)
            ):
                continue
            for index, raw_id in enumerate(raw_ids[:13], start=1):
                circle = by_id.get(str(raw_id))
                if circle is not None and circle.id not in seen:
                    records.append((arm, index, circle))
                    seen.add(circle.id)
    for circle in circles:
        if (
            circle.id not in seen
            and circle.circle_type == "main"
            and circle.arm_id in range(1, 7)
            and circle.sequence_index is not None
            and 1 <= int(circle.sequence_index) <= 13
        ):
            records.append(
                (int(circle.arm_id), int(circle.sequence_index), circle)
            )
    return records


def _resolve_center_source(
    circles: Sequence[CircleFeature],
    explicit: Point | None,
) -> Point | None:
    if explicit is not None:
        return explicit
    center_circle = next(
        (circle for circle in circles if circle.circle_type == "center"),
        None,
    )
    if center_circle is not None:
        return (
            float(
                center_circle.source_x
                if center_circle.source_x is not None
                else center_circle.x
            ),
            float(
                center_circle.source_y
                if center_circle.source_y is not None
                else center_circle.y
            ),
        )
    main = [circle for circle in circles if circle.circle_type == "main"]
    if not main:
        return None
    return (
        float(
            np.median(
                [
                    circle.source_x
                    if circle.source_x is not None
                    else circle.x
                    for circle in main
                ]
            )
        ),
        float(
            np.median(
                [
                    circle.source_y
                    if circle.source_y is not None
                    else circle.y
                    for circle in main
                ]
            )
        ),
    )


def _arm_space_summary(
    rows: Sequence[Mapping[str, Any]],
    prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sequence_rows: list[dict[str, Any]] = []
    radius_cvs: list[float] = []
    distance_cvs: list[float] = []
    angular_stds: list[float] = []
    for sequence_index in range(1, 14):
        sequence = [
            row
            for row in rows
            if int(row["sequence_index"]) == sequence_index
        ]
        radii = [float(row[f"{prefix}_radius"]) for row in sequence]
        distances = [float(row[f"{prefix}_center_distance"]) for row in sequence]
        phases = [
            (
                float(row[f"{prefix}_center_angle"])
                - (int(row["arm_id"]) - 1) * 60.0
            )
            % 360.0
            for row in sequence
        ]
        radius_cv = _coefficient_of_variation(radii)
        distance_cv = _coefficient_of_variation(distances)
        angular_std = _circular_phase_std(phases) if len(phases) >= 2 else None
        if radius_cv is not None:
            radius_cvs.append(radius_cv)
        if distance_cv is not None:
            distance_cvs.append(distance_cv)
        if angular_std is not None:
            angular_stds.append(angular_std)
        sequence_rows.append(
            {
                "sequence_index": sequence_index,
                "sample_count": len(sequence),
                f"{prefix}_radius_cv": _rounded(radius_cv),
                f"{prefix}_distance_cv": _rounded(distance_cv),
                f"{prefix}_angular_std": _rounded(angular_std),
            }
        )
    radius_mean = float(np.mean(radius_cvs)) if radius_cvs else None
    distance_mean = float(np.mean(distance_cvs)) if distance_cvs else None
    angular_mean = float(np.mean(angular_stds)) if angular_stds else None
    radius_score = (
        None if radius_mean is None else 100.0 * math.exp(-4.0 * radius_mean)
    )
    distance_score = (
        None
        if distance_mean is None
        else 100.0 * math.exp(-4.0 * distance_mean)
    )
    angular_score = (
        None
        if angular_mean is None
        else 100.0 * math.exp(-angular_mean / 12.0)
    )
    components = [
        value
        for value in (radius_score, distance_score, angular_score)
        if value is not None
    ]
    return (
        {
            "radius_cv_mean": _rounded(radius_mean),
            "distance_cv_mean": _rounded(distance_mean),
            "angular_std_mean": _rounded(angular_mean),
            "radius_consistency": _rounded(radius_score),
            "distance_consistency": _rounded(distance_score),
            "angular_consistency": _rounded(angular_score),
            "symmetry_score": (
                None
                if not components
                else _rounded(float(np.mean(components)))
            ),
        },
        sequence_rows,
    )


def _opposite_scale_score(
    rows: Sequence[Mapping[str, Any]],
    prefix: str,
) -> tuple[float | None, float | None]:
    by_key = {
        (int(row["arm_id"]), int(row["sequence_index"])): row
        for row in rows
    }
    differences: list[float] = []
    for first, opposite in ((1, 4), (2, 5), (3, 6)):
        for sequence_index in range(1, 14):
            left = by_key.get((first, sequence_index))
            right = by_key.get((opposite, sequence_index))
            if left is None or right is None:
                continue
            for field in ("radius", "center_distance"):
                left_value = float(left[f"{prefix}_{field}"])
                right_value = float(right[f"{prefix}_{field}"])
                denominator = max((abs(left_value) + abs(right_value)) / 2.0, 1e-6)
                differences.append(abs(left_value - right_value) / denominator)
    if not differences:
        return None, None
    mean_difference = float(np.mean(differences))
    return 100.0 * math.exp(-3.5 * mean_difference), mean_difference


def _spiral_parameters(
    rows: Sequence[Mapping[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in range(1, 7):
        items = sorted(
            [row for row in rows if int(row["arm_id"]) == arm],
            key=lambda row: int(row["sequence_index"]),
        )
        if len(items) < 3:
            result[str(arm)] = {
                "sample_count": len(items),
                "twist_degrees_per_100px": None,
                "rmse_degrees": None,
            }
            continue
        distances = np.asarray(
            [float(row[f"{prefix}_center_distance"]) for row in items],
            dtype=np.float64,
        )
        angles = np.unwrap(
            np.radians(
                [float(row[f"{prefix}_center_angle"]) for row in items]
            )
        )
        design = np.column_stack(
            [np.ones(len(items)), distances / 100.0]
        )
        coefficients, *_ = np.linalg.lstsq(design, angles, rcond=None)
        prediction = design @ coefficients
        result[str(arm)] = {
            "sample_count": len(items),
            "phase_degrees": _rounded(math.degrees(float(coefficients[0])) % 360.0),
            "twist_degrees_per_100px": _rounded(
                math.degrees(float(coefficients[1]))
            ),
            "rmse_degrees": _rounded(
                math.degrees(float(np.sqrt(np.mean((angles - prediction) ** 2))))
            ),
        }
    return result


def _measure_arm_symmetry(
    circles: Sequence[CircleFeature],
    source_to_corrected: NDArray[np.float64],
    center_source: Point | None,
    structure_state: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = _main_chain_records(circles, structure_state)
    center = _resolve_center_source(circles, center_source)
    if center is None or not records:
        return [], {
            "available": False,
            "before": {},
            "after": {},
            "sequence_metrics": [],
            "opposite_scale_score_before": None,
            "opposite_scale_score_after": None,
            "score_before": None,
            "score_after": None,
            "gain": None,
            "comment": "Onaylı ana zincirler ve merkez olmadan kol simetrisi ölçülemez.",
        }
    corrected_to_source = np.linalg.inv(source_to_corrected)
    center_after = _transform_point(center, source_to_corrected)
    rows: list[dict[str, Any]] = []
    for arm, sequence_index, circle in records:
        source_circle, source_radius = _circle_source_geometry(
            circle, corrected_to_source
        )
        projected = _project_circle(
            source_circle, source_radius, source_to_corrected
        )
        corrected_circle = (
            float(projected["center_x"]),
            float(projected["center_y"]),
        )
        corrected_radius = math.sqrt(
            float(projected["major_axis"]) * float(projected["minor_axis"])
        ) / 2.0
        before_dx = source_circle[0] - center[0]
        before_dy = source_circle[1] - center[1]
        after_dx = corrected_circle[0] - center_after[0]
        after_dy = corrected_circle[1] - center_after[1]
        rows.append(
            {
                "circle_id": circle.id,
                "arm_id": arm,
                "sequence_index": sequence_index,
                "source_x": _rounded(source_circle[0]),
                "source_y": _rounded(source_circle[1]),
                "source_radius": _rounded(source_radius),
                "source_center_distance": _rounded(
                    math.hypot(before_dx, before_dy)
                ),
                "source_center_angle": _rounded(
                    math.degrees(math.atan2(-before_dy, before_dx)) % 360.0
                ),
                "corrected_x": _rounded(corrected_circle[0]),
                "corrected_y": _rounded(corrected_circle[1]),
                "corrected_radius": _rounded(corrected_radius),
                "corrected_center_distance": _rounded(
                    math.hypot(after_dx, after_dy)
                ),
                "corrected_center_angle": _rounded(
                    math.degrees(math.atan2(-after_dy, after_dx)) % 360.0
                ),
            }
        )
    before, before_sequences = _arm_space_summary(rows, "source")
    after, after_sequences = _arm_space_summary(rows, "corrected")
    sequence_metrics: list[dict[str, Any]] = []
    for before_row, after_row in zip(before_sequences, after_sequences):
        merged = {**before_row, **after_row}
        for metric in ("radius_cv", "distance_cv", "angular_std"):
            before_value = merged.get(f"source_{metric}")
            after_value = merged.get(f"corrected_{metric}")
            merged[f"{metric}_improvement"] = (
                None
                if before_value is None or after_value is None
                else _rounded(float(before_value) - float(after_value))
            )
        sequence_metrics.append(merged)
    sequence_by_index = {
        int(row["sequence_index"]): row for row in sequence_metrics
    }
    spiral_before = _spiral_parameters(rows, "source")
    spiral_after = _spiral_parameters(rows, "corrected")
    for row in rows:
        row.update(
            {
                key: value
                for key, value in sequence_by_index[
                    int(row["sequence_index"])
                ].items()
                if key not in {"sequence_index", "sample_count"}
            }
        )
        arm_key = str(int(row["arm_id"]))
        row["spiral_twist_before"] = spiral_before[arm_key][
            "twist_degrees_per_100px"
        ]
        row["spiral_twist_after"] = spiral_after[arm_key][
            "twist_degrees_per_100px"
        ]
        row["spiral_rmse_before"] = spiral_before[arm_key]["rmse_degrees"]
        row["spiral_rmse_after"] = spiral_after[arm_key]["rmse_degrees"]
    opposite_before, opposite_error_before = _opposite_scale_score(
        rows, "source"
    )
    opposite_after, opposite_error_after = _opposite_scale_score(
        rows, "corrected"
    )
    before_score = before.get("symmetry_score")
    after_score = after.get("symmetry_score")
    gain = (
        None
        if before_score is None or after_score is None
        else float(after_score) - float(before_score)
    )
    sequence_seven = next(
        (row for row in sequence_metrics if row["sequence_index"] == 7),
        None,
    )
    if gain is None:
        comment = "Altı kol için yeterli eşleşmiş ana sıra bulunamadı."
    elif gain > 3.0:
        comment = "Kol ölçeği, uzaklığı ve açısal fazı daha tutarlı hale geldi."
    elif gain < -3.0:
        comment = "Kol tutarlılığı azaldı; perspektif referanslarını kontrol edin."
    else:
        comment = "Kol simetrisindeki değişim sınırlı."
    if sequence_seven and sequence_seven.get("radius_cv_improvement") is not None:
        comment += (
            " 7. sıra yarıçap CV değişimi "
            f"{float(sequence_seven['radius_cv_improvement']):+.3f}."
        )
    return rows, {
        "available": before_score is not None and after_score is not None,
        "center_source": {"x": _rounded(center[0]), "y": _rounded(center[1])},
        "center_corrected": {
            "x": _rounded(center_after[0]),
            "y": _rounded(center_after[1]),
        },
        "before": before,
        "after": after,
        "sequence_metrics": sequence_metrics,
        "spiral_parameters_before": spiral_before,
        "spiral_parameters_after": spiral_after,
        "opposite_scale_score_before": _rounded(opposite_before),
        "opposite_scale_score_after": _rounded(opposite_after),
        "opposite_scale_error_before": _rounded(opposite_error_before),
        "opposite_scale_error_after": _rounded(opposite_error_after),
        "score_before": before_score,
        "score_after": after_score,
        "gain": _rounded(gain),
        "comment": comment,
    }


def _measure_k5k6(
    arm_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (int(row["arm_id"]), int(row["sequence_index"])): row
        for row in arm_rows
    }
    rows: list[dict[str, Any]] = []
    for sequence_index in range(1, 14):
        fifth = by_key.get((5, sequence_index))
        sixth = by_key.get((6, sequence_index))
        if fifth is None or sixth is None:
            continue
        row: dict[str, Any] = {"sequence_index": sequence_index}
        for prefix in ("source", "corrected"):
            radius_fifth = float(fifth[f"{prefix}_radius"])
            radius_sixth = float(sixth[f"{prefix}_radius"])
            distance_fifth = float(fifth[f"{prefix}_center_distance"])
            distance_sixth = float(sixth[f"{prefix}_center_distance"])
            row[f"{prefix}_radius_difference"] = _rounded(
                abs(radius_fifth - radius_sixth)
            )
            row[f"{prefix}_distance_difference"] = _rounded(
                abs(distance_fifth - distance_sixth)
            )
            row[f"{prefix}_normalized_diameter_difference"] = _rounded(
                abs(2.0 * radius_fifth - 2.0 * radius_sixth)
                / max(radius_fifth + radius_sixth, 1e-6)
            )
        rows.append(row)
    if not rows:
        return {
            "available": False,
            "matched_sequence_count": 0,
            "rows": [],
            "k5k6_radius_mae_before": None,
            "k5k6_radius_mae_after": None,
            "k5k6_distance_mae_before": None,
            "k5k6_distance_mae_after": None,
            "normalized_diameter_mae_before": None,
            "normalized_diameter_mae_after": None,
            "score_before": None,
            "score_after": None,
            "improvement_score": None,
            "gain": None,
            "comment": "K5 ve K6 için eşleşmiş ana sıra bulunamadı.",
        }
    radius_before = float(
        np.mean([row["source_radius_difference"] for row in rows])
    )
    radius_after = float(
        np.mean([row["corrected_radius_difference"] for row in rows])
    )
    distance_before = float(
        np.mean([row["source_distance_difference"] for row in rows])
    )
    distance_after = float(
        np.mean([row["corrected_distance_difference"] for row in rows])
    )
    normalized_before = float(
        np.mean([row["source_normalized_diameter_difference"] for row in rows])
    )
    normalized_after = float(
        np.mean([row["corrected_normalized_diameter_difference"] for row in rows])
    )
    mean_distance = float(
        np.mean(
            [
                (
                    float(by_key[(5, int(row["sequence_index"]))]["source_center_distance"])
                    + float(by_key[(6, int(row["sequence_index"]))]["source_center_distance"])
                )
                / 2.0
                for row in rows
            ]
        )
    )
    before_error = 0.65 * normalized_before + 0.35 * (
        distance_before / max(mean_distance, 1e-6)
    )
    after_error = 0.65 * normalized_after + 0.35 * (
        distance_after / max(mean_distance, 1e-6)
    )
    score_before = 100.0 * math.exp(-3.0 * before_error)
    score_after = 100.0 * math.exp(-3.0 * after_error)
    gain = score_after - score_before
    comment = (
        "K5–K6 ölçek ve uzaklık farkları azaldı."
        if gain > 3.0
        else "K5–K6 farkları büyüdü; düzeltme seçimini kontrol edin."
        if gain < -3.0
        else "K5–K6 farkındaki değişim sınırlı."
    )
    return {
        "available": True,
        "matched_sequence_count": len(rows),
        "rows": rows,
        "k5k6_radius_mae_before": _rounded(radius_before),
        "k5k6_radius_mae_after": _rounded(radius_after),
        "k5k6_distance_mae_before": _rounded(distance_before),
        "k5k6_distance_mae_after": _rounded(distance_after),
        "normalized_diameter_mae_before": _rounded(normalized_before),
        "normalized_diameter_mae_after": _rounded(normalized_after),
        "score_before": _rounded(score_before),
        "score_after": _rounded(score_after),
        "improvement_score": _rounded(score_after),
        "gain": _rounded(gain),
        "comment": comment,
    }


def _local_jacobian(
    matrix: NDArray[np.float64],
    x: float,
    y: float,
) -> NDArray[np.float64] | None:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    denominator = g * x + h * y + i
    if abs(float(denominator)) < 1e-9:
        return None
    numerator_x = a * x + b * y + c
    numerator_y = d * x + e * y + f
    denominator_squared = denominator * denominator
    return np.asarray(
        [
            [
                (a * denominator - g * numerator_x) / denominator_squared,
                (b * denominator - h * numerator_x) / denominator_squared,
            ],
            [
                (d * denominator - g * numerator_y) / denominator_squared,
                (e * denominator - h * numerator_y) / denominator_squared,
            ],
        ],
        dtype=np.float64,
    )


def measure_warp_penalty(
    source_to_corrected: Any,
    source_size: tuple[int, int],
) -> dict[str, Any]:
    """Return local scale/shear/anisotropy diagnostics and a 0–100 penalty."""

    matrix = _matrix(source_to_corrected)
    width, height = int(source_size[0]), int(source_size[1])
    corners = [
        (0.0, 0.0),
        (float(max(0, width - 1)), 0.0),
        (float(max(0, width - 1)), float(max(0, height - 1))),
        (0.0, float(max(0, height - 1))),
    ]
    transformed_corners = _transform_points(corners, matrix)
    source_area = max(float(max(1, width - 1) * max(1, height - 1)), 1.0)
    polygon = np.asarray(transformed_corners, dtype=np.float64)
    transformed_area = 0.5 * abs(
        float(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )
    area_ratio = transformed_area / source_area
    samples: list[dict[str, float]] = []
    for y in np.linspace(0.05 * height, 0.95 * height, 7):
        for x in np.linspace(0.05 * width, 0.95 * width, 7):
            jacobian = _local_jacobian(matrix, float(x), float(y))
            if jacobian is None:
                continue
            singular = np.linalg.svd(jacobian, compute_uv=False)
            if singular[-1] <= 1e-12:
                continue
            column_a, column_b = jacobian[:, 0], jacobian[:, 1]
            shear = abs(
                float(np.dot(column_a, column_b))
                / max(
                    float(np.linalg.norm(column_a) * np.linalg.norm(column_b)),
                    1e-12,
                )
            )
            samples.append(
                {
                    "area_scale": abs(float(np.linalg.det(jacobian))),
                    "anisotropy": float(singular[0] / singular[-1]),
                    "shear": shear,
                }
            )
    if not samples:
        return {
            "warp_area_ratio": _rounded(area_ratio),
            "corner_scale_ratios": [],
            "warp_shear_score": 100.0,
            "warp_anisotropy_score": 100.0,
            "local_distortion_indicator": 100.0,
            "warp_penalty": 100.0,
            "comment": "Homografi yerel olarak değerlendirilemedi.",
        }
    area_scales = np.asarray(
        [item["area_scale"] for item in samples], dtype=np.float64
    )
    anisotropies = np.asarray(
        [item["anisotropy"] for item in samples], dtype=np.float64
    )
    shears = np.asarray([item["shear"] for item in samples], dtype=np.float64)
    area_severity = 100.0 * (
        1.0 - math.exp(-abs(math.log(max(area_ratio, 1e-9))) / 1.2)
    )
    shear_severity = 100.0 * (
        1.0 - math.exp(-float(np.mean(shears)) / 0.22)
    )
    anisotropy_severity = 100.0 * (
        1.0
        - math.exp(
            -float(np.mean(np.abs(np.log(np.maximum(anisotropies, 1.0)))))
            / 0.48
        )
    )
    area_cv = float(np.std(area_scales) / max(float(np.mean(area_scales)), 1e-9))
    local_severity = 100.0 * (1.0 - math.exp(-area_cv / 0.45))
    penalty = _clamp(
        0.24 * area_severity
        + 0.28 * shear_severity
        + 0.32 * anisotropy_severity
        + 0.16 * local_severity
    )
    corner_scales: list[float] = []
    for x, y in corners:
        jacobian = _local_jacobian(matrix, x, y)
        if jacobian is not None:
            corner_scales.append(
                math.sqrt(abs(float(np.linalg.det(jacobian))))
            )
    comment = (
        "Dönüşüm yerel olarak dengeli."
        if penalty < 20.0
        else "Dönüşüm orta düzeyde ölçek/shear değişimi içeriyor."
        if penalty < 45.0
        else "Dönüşüm aşırı yerel germe riski taşıyor."
    )
    return {
        "warp_area_ratio": _rounded(area_ratio),
        "corner_scale_ratios": [_rounded(value) for value in corner_scales],
        "warp_shear_score": _rounded(shear_severity),
        "warp_anisotropy_score": _rounded(anisotropy_severity),
        "local_distortion_indicator": _rounded(local_severity),
        "warp_penalty": _rounded(penalty),
        "comment": comment,
    }


def _normalise_weights(
    supplied: Mapping[str, Any] | None,
) -> dict[str, float]:
    values = dict(DEFAULT_QUALITY_WEIGHTS)
    if supplied is not None:
        for key in values:
            if key in supplied:
                number = float(supplied[key])
                if not math.isfinite(number) or number < 0:
                    raise ValueError("Quality weights must be finite and non-negative")
                values[key] = number
    total = sum(values.values())
    if total <= 1e-12:
        raise ValueError("At least one perspective quality weight must be positive")
    return {key: value / total for key, value in values.items()}


def _component_comment(
    before: float | None,
    after: float | None,
    *,
    higher_is_better: bool = True,
) -> str:
    if before is None or after is None:
        return "Ölçüm için yeterli onaylı veri yok."
    gain = (after - before) * (1.0 if higher_is_better else -1.0)
    if gain > 4.0:
        return "Belirgin iyileşme."
    if gain < -4.0:
        return "Belirgin kötüleşme."
    return "Değişim sınırlı."


def _overall_score(
    values: Mapping[str, float | None],
    weights: Mapping[str, float],
) -> float | None:
    available = [
        key
        for key, value in values.items()
        if value is not None and weights.get(key, 0.0) > 0.0
    ]
    denominator = sum(weights[key] for key in available)
    if denominator <= 1e-12:
        return None
    return _clamp(
        sum(float(values[key]) * weights[key] for key in available)
        / denominator
    )


def _interpretation(score: float | None) -> tuple[str, str]:
    if score is None:
        return "not_measurable", "Kalite puanı için yeterli ölçüm yok."
    if score >= 80.0:
        return "strong_correction", "Güçlü perspektif düzeltmesi."
    if score >= 60.0:
        return "usable_but_verify", "Kullanılabilir; dikkatli yorumlanmalı."
    if score >= 40.0:
        return "limited_benefit", "Sınırlı fayda; ek doğrulama gerekli."
    return "weak_correction", "Zayıf düzeltme; referans seçimini yenileyin."


def _confidence(
    tramline_count: int,
    paired_circle_count: int,
    arm_row_count: int,
) -> tuple[str, float]:
    tramline_coverage = min(1.0, tramline_count / 6.0)
    circle_coverage = min(1.0, paired_circle_count / 18.0)
    arm_coverage = min(1.0, arm_row_count / 78.0)
    coverages = (tramline_coverage, circle_coverage, arm_coverage)
    # A dense single source is useful but cannot by itself establish high
    # confidence. Reward both the strongest evidence family and independent
    # cross-check coverage.
    numeric = (
        0.45 * max(coverages)
        + 0.55 * float(np.mean(coverages))
    )
    label = "high" if numeric >= 0.75 else "medium" if numeric >= 0.42 else "low"
    return label, numeric


def _method_guidance(method: str) -> str:
    if method == "four_point":
        return (
            "Görüntü köşelerini değil, zemindeki aynı düzleme ait dört anlamlı "
            "referans noktayı seçin. Seçilen dört nokta, yerdeki bir "
            "dikdörtgen/paralelkenarın köşelerini temsil etmelidir."
        )
    if method == "tramline_guided":
        return (
            "Paralel tarla izleri güçlü referans sağlar. Ancak tek doğrultudaki "
            "paralellik, tek başına tam metrik düzeltme garantisi vermez. "
            "Sonucu dairesellik ve simetri skorlarıyla doğrulayın."
        )
    return "Aktif bir perspektif düzeltmesi bulunmuyor."


def _draw_label(
    image: NDArray[np.uint8],
    text: str,
    origin: tuple[int, int] = (18, 32),
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (235, 245, 255),
        1,
        cv2.LINE_AA,
    )


def _resize_height(
    image: NDArray[np.uint8],
    height: int,
) -> NDArray[np.uint8]:
    scale = height / max(image.shape[0], 1)
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _side_by_side(
    source: NDArray[np.uint8],
    corrected: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    target_height = min(900, max(source.shape[0], corrected.shape[0]))
    left = _resize_height(source, target_height)
    right = _resize_height(corrected, target_height)
    _draw_label(left, "ONCE / SOURCE")
    _draw_label(right, "SONRA / CORRECTED")
    separator = np.full((target_height, 6, 3), 30, dtype=np.uint8)
    return np.hstack([left, separator, right])


def _draw_ellipse(
    image: NDArray[np.uint8],
    row: Mapping[str, Any],
    prefix: str,
    color: tuple[int, int, int],
) -> None:
    major = row.get(f"{prefix}_major_axis")
    minor = row.get(f"{prefix}_minor_axis")
    if major is None or minor is None:
        return
    center_key = "source" if prefix == "source" else "corrected"
    angle_key = (
        "source_ellipse_angle"
        if prefix == "source"
        else "corrected_ellipse_angle"
    )
    center = (
        int(round(float(row[f"{center_key}_x"]))),
        int(round(float(row[f"{center_key}_y"]))),
    )
    axes = (
        max(1, int(round(float(major) / 2.0))),
        max(1, int(round(float(minor) / 2.0))),
    )
    cv2.ellipse(
        image,
        center,
        axes,
        float(row.get(angle_key) or 0.0),
        0,
        360,
        color,
        2,
        cv2.LINE_AA,
    )


def _roundness_overlay(
    source: NDArray[np.uint8],
    corrected: NDArray[np.uint8],
    rows: Sequence[Mapping[str, Any]],
) -> NDArray[np.uint8]:
    source_overlay = source.copy()
    corrected_overlay = corrected.copy()
    ranked = sorted(
        [row for row in rows if row.get("circle_type") == "main"],
        key=lambda row: float(row.get("source_radius") or 0.0),
        reverse=True,
    )[:18]
    for row in ranked:
        _draw_ellipse(source_overlay, row, "source", (255, 210, 40))
        benefit = row.get("circularity_gain")
        color = (
            (70, 220, 80)
            if benefit is not None and float(benefit) > 2.0
            else (70, 90, 245)
            if benefit is not None and float(benefit) < -2.0
            else (60, 205, 245)
        )
        _draw_ellipse(corrected_overlay, row, "corrected", color)
    return _side_by_side(source_overlay, corrected_overlay)


def _tramline_overlay(
    corrected: NDArray[np.uint8],
    rows: Sequence[Mapping[str, Any]],
) -> NDArray[np.uint8]:
    overlay = corrected.copy()
    for row in rows:
        p1 = (
            int(round(float(row["corrected_p1_x"]))),
            int(round(float(row["corrected_p1_y"]))),
        )
        p2 = (
            int(round(float(row["corrected_p2_x"]))),
            int(round(float(row["corrected_p2_y"]))),
        )
        cv2.line(overlay, p1, p2, (70, 180, 255), 2, cv2.LINE_AA)
    _draw_label(overlay, "DUZELTILMIS TRAMLINE AILESI")
    return overlay


def _arm_heatmap(
    corrected: NDArray[np.uint8],
    arm_rows: Sequence[Mapping[str, Any]],
    sequence_rows: Sequence[Mapping[str, Any]],
) -> NDArray[np.uint8]:
    overlay = corrected.copy()
    improvement = {
        int(row["sequence_index"]): row.get("radius_cv_improvement")
        for row in sequence_rows
    }
    for row in arm_rows:
        value = improvement.get(int(row["sequence_index"]))
        if value is None:
            color = (150, 150, 150)
        else:
            normalized = max(-1.0, min(1.0, float(value) / 0.12))
            color = (
                int(60 + 180 * max(0.0, -normalized)),
                int(80 + 165 * max(0.0, normalized)),
                70,
            )
        center = (
            int(round(float(row["corrected_x"]))),
            int(round(float(row["corrected_y"]))),
        )
        radius = max(2, int(round(float(row["corrected_radius"]))))
        cv2.circle(overlay, center, radius, color, 2, cv2.LINE_AA)
    _draw_label(overlay, "KOL OLCEK FARKI: YESIL IYILESME / KIRMIZI KOTULESME")
    return overlay


def analyze_perspective_quality(
    source_image: NDArray[np.uint8],
    corrected_image: NDArray[np.uint8],
    source_to_corrected: Any,
    *,
    circles: Sequence[CircleFeature] = (),
    tramlines: Sequence[Mapping[str, Any]] = (),
    center_source: Point | None = None,
    structure_state: Mapping[str, Any] | None = None,
    method: str = "none",
    status: str = "not_applied",
    weights: Mapping[str, Any] | None = None,
    selected_circle_ids: Iterable[str] = (),
) -> PerspectiveQualityResult:
    """Build a before/after quality report without changing input objects."""

    if source_image.size == 0 or corrected_image.size == 0:
        raise ValueError("Perspective quality requires source and corrected images")
    matrix = _matrix(source_to_corrected)
    selected = {str(item) for item in selected_circle_ids}
    normalized_weights = _normalise_weights(weights)
    tramline = measure_tramline_quality(tramlines, matrix)
    circle_rows, roundness = _measure_circles(
        source_image,
        corrected_image,
        circles,
        matrix,
        selected,
    )
    arm_rows, arm = _measure_arm_symmetry(
        circles,
        matrix,
        center_source,
        structure_state,
    )
    k5k6 = _measure_k5k6(arm_rows)
    warp = measure_warp_penalty(
        matrix,
        (source_image.shape[1], source_image.shape[0]),
    )
    roundness_active = roundness["groups"][roundness["active_group"]]
    tramline_before = tramline["before"]["parallelism_score"]
    tramline_after = tramline["after"]["parallelism_score"]
    roundness_before = roundness_active["score_before"]
    roundness_after = roundness_active["score_after"]
    arm_before = arm.get("score_before")
    arm_after = arm.get("score_after")
    k5k6_before = k5k6.get("score_before")
    k5k6_after = k5k6.get("score_after")
    warp_quality_after = 100.0 - float(warp["warp_penalty"])
    before_values = {
        "tramline": tramline_before,
        "roundness": roundness_before,
        "arm_symmetry": arm_before,
        "k5k6": k5k6_before,
        "warp_penalty": 100.0,
    }
    after_values = {
        "tramline": tramline_after,
        "roundness": roundness_after,
        "arm_symmetry": arm_after,
        "k5k6": k5k6_after,
        "warp_penalty": warp_quality_after,
    }
    overall_before = _overall_score(before_values, normalized_weights)
    overall_after = _overall_score(after_values, normalized_weights)
    has_validation_evidence = any(
        after_values[key] is not None
        for key in ("tramline", "roundness", "arm_symmetry", "k5k6")
    )
    if not has_validation_evidence:
        overall_before = None
        overall_after = None
    elif method == "none" or status == "not_applied":
        overall_after = None
    overall_gain = (
        None
        if overall_before is None or overall_after is None
        else overall_after - overall_before
    )
    interpretation, interpretation_message = _interpretation(overall_after)
    if method == "none" or status == "not_applied":
        interpretation = "not_applied"
        interpretation_message = (
            "Aktif veya değerlendirilebilir bir perspektif düzeltmesi yok."
        )
    paired_count = int(roundness_active["paired_edge_fit_count"])
    confidence_label, confidence_value = _confidence(
        int(tramline["before"]["line_count"]),
        paired_count,
        len(arm_rows),
    )
    if (
        status != "not_applied"
        and overall_after is not None
        and overall_gain is not None
        and overall_gain <= 0.0
    ):
        interpretation_message += (
            " Kaynak referansa göre net kalite kazancı yok."
        )
    if status != "not_applied" and confidence_label == "low":
        interpretation_message += (
            " Ölçüm kapsamı düşük; kararı ek tramline, dairesellik veya "
            "kol verisiyle doğrulayın."
        )
    components = [
        {
            "id": "tramline_parallelism",
            "label": "Tramline paralellik",
            "before": tramline_before,
            "after": tramline_after,
            "delta": (
                None
                if tramline_before is None or tramline_after is None
                else _rounded(float(tramline_after) - float(tramline_before))
            ),
            "comment": tramline["comment"],
        },
        {
            "id": "roundness",
            "label": "Dairesellik iyileşmesi",
            "before": roundness_before,
            "after": roundness_after,
            "delta": roundness_active["gain"],
            "comment": roundness["comment"],
        },
        {
            "id": "arm_symmetry",
            "label": "Kol simetrisi",
            "before": arm_before,
            "after": arm_after,
            "delta": arm.get("gain"),
            "comment": arm.get("comment"),
        },
        {
            "id": "opposite_arm_scale",
            "label": "Karşılıklı kol ölçek tutarlılığı",
            "before": arm.get("opposite_scale_score_before"),
            "after": arm.get("opposite_scale_score_after"),
            "delta": (
                None
                if arm.get("opposite_scale_score_before") is None
                or arm.get("opposite_scale_score_after") is None
                else _rounded(
                    float(arm["opposite_scale_score_after"])
                    - float(arm["opposite_scale_score_before"])
                )
            ),
            "comment": _component_comment(
                arm.get("opposite_scale_score_before"),
                arm.get("opposite_scale_score_after"),
            ),
        },
        {
            "id": "k5k6",
            "label": "K5–K6 fark iyileşmesi",
            "before": k5k6_before,
            "after": k5k6_after,
            "delta": k5k6.get("gain"),
            "comment": k5k6.get("comment"),
        },
        {
            "id": "warp_stability",
            "label": "Perspektif stabilitesi",
            "before": 100.0,
            "after": _rounded(warp_quality_after),
            "delta": _rounded(-float(warp["warp_penalty"])),
            "comment": warp["comment"],
        },
    ]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "method": method,
        "status": status,
        "coordinate_systems": {
            "source": "source_image_pixels",
            "corrected": "corrected_image_pixels",
            "active_analysis": "corrected",
        },
        "non_destructive": True,
        "annotation_data_modified": False,
        "s_drift_correction_applied": False,
        "weights": {key: _rounded(value) for key, value in normalized_weights.items()},
        "confidence": {
            "level": confidence_label,
            "value": _rounded(confidence_value),
            "tramline_count": int(tramline["before"]["line_count"]),
            "paired_circle_count": paired_count,
            "arm_main_circle_count": len(arm_rows),
        },
        "scores": {
            "overall_before": _rounded(overall_before),
            "overall_after": _rounded(overall_after),
            "gain": _rounded(overall_gain),
            "tramline_parallelism_before": tramline_before,
            "tramline_parallelism_after": tramline_after,
            "tramline_parallelism": tramline_after,
            "roundness_before": roundness_before,
            "roundness_after": roundness_after,
            "roundness_improvement": roundness_after,
            "arm_symmetry_before": arm_before,
            "arm_symmetry_after": arm_after,
            "arm_symmetry": arm_after,
            "k5k6_before": k5k6_before,
            "k5k6_after": k5k6_after,
            "k5k6_improvement": k5k6_after,
            "warp_penalty": warp["warp_penalty"],
        },
        "metrics": {
            "tramline_angle_std_before": tramline["before"]["angle_std"],
            "tramline_angle_std_after": tramline["after"]["angle_std"],
            "tramline_spacing_cv_before": tramline["before"]["spacing_cv"],
            "tramline_spacing_cv_after": tramline["after"]["spacing_cv"],
            "tramline_parallelism_score_before": tramline_before,
            "tramline_parallelism_score_after": tramline_after,
            "ellipse_ratio_mean_before": roundness_active["before"][
                "ellipse_ratio"
            ]["mean"],
            "ellipse_ratio_mean_after": roundness_active["after"][
                "ellipse_ratio"
            ]["mean"],
            "circularity_residual_mean_before": roundness_active["before"][
                "circularity_residual"
            ]["mean"],
            "circularity_residual_mean_after": roundness_active["after"][
                "circularity_residual"
            ]["mean"],
            "arm_radius_consistency_before": arm.get("before", {}).get(
                "radius_consistency"
            ),
            "arm_radius_consistency_after": arm.get("after", {}).get(
                "radius_consistency"
            ),
            "arm_distance_consistency_before": arm.get("before", {}).get(
                "distance_consistency"
            ),
            "arm_distance_consistency_after": arm.get("after", {}).get(
                "distance_consistency"
            ),
            "arm_angular_consistency_before": arm.get("before", {}).get(
                "angular_consistency"
            ),
            "arm_angular_consistency_after": arm.get("after", {}).get(
                "angular_consistency"
            ),
            "k5k6_radius_mae_before": k5k6.get("k5k6_radius_mae_before"),
            "k5k6_radius_mae_after": k5k6.get("k5k6_radius_mae_after"),
            "k5k6_distance_mae_before": k5k6.get("k5k6_distance_mae_before"),
            "k5k6_distance_mae_after": k5k6.get("k5k6_distance_mae_after"),
            **warp,
        },
        "components": components,
        "tramline": {key: value for key, value in tramline.items() if key != "lines"},
        "roundness": roundness,
        "arm_symmetry": arm,
        "k5k6": k5k6,
        "warp": warp,
        "circle_metrics": circle_rows,
        "arm_metrics": arm_rows,
        "tramline_metrics": tramline["lines"],
        "interpretation": interpretation,
        "interpretation_message": interpretation_message,
        "method_guidance": _method_guidance(method),
    }
    images = {
        "source_view": source_image.copy(),
        "corrected_view": corrected_image.copy(),
        "perspective_before_after": _side_by_side(
            source_image, corrected_image
        ),
        "roundness_overlay": _roundness_overlay(
            source_image, corrected_image, circle_rows
        ),
        "tramline_overlay_corrected": _tramline_overlay(
            corrected_image, tramline["lines"]
        ),
        "arm_scale_heatmap": _arm_heatmap(
            corrected_image,
            arm_rows,
            arm.get("sequence_metrics", []),
        ),
    }
    return PerspectiveQualityResult(
        report=report,
        circle_metrics=circle_rows,
        arm_metrics=arm_rows,
        tramline_metrics=tramline["lines"],
        images=images,
    )


def _write_csv_rows(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["status"]
        rows = [{"status": "no_measurements"}]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def write_perspective_quality_exports(
    result: PerspectiveQualityResult,
    destination_dir: str | Path,
) -> dict[str, Path]:
    """Write the stable JSON/CSV/PNG perspective-quality export bundle."""

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "perspective_quality_report": destination
        / "perspective_quality_report.json",
        "perspective_circle_metrics": destination
        / "perspective_circle_metrics.csv",
        "perspective_arm_metrics": destination
        / "perspective_arm_metrics.csv",
        "tramline_metrics": destination / "tramline_metrics.csv",
        "perspective_before_after": destination
        / "perspective_before_after.png",
        "roundness_overlay": destination / "roundness_overlay.png",
        "tramline_overlay_corrected": destination
        / "tramline_overlay_corrected.png",
        "arm_scale_heatmap": destination / "arm_scale_heatmap.png",
        "source_view": destination / "perspective_source.png",
        "corrected_view": destination / "perspective_corrected.png",
    }
    files["perspective_quality_report"].write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv_rows(
        result.circle_metrics, files["perspective_circle_metrics"]
    )
    _write_csv_rows(result.arm_metrics, files["perspective_arm_metrics"])
    _write_csv_rows(result.tramline_metrics, files["tramline_metrics"])
    for key, image in result.images.items():
        path = files.get(key)
        if path is not None and not cv2.imwrite(str(path), image):
            raise OSError(f"Perspective quality image could not be written: {path.name}")
    return files
