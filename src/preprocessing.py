"""Non-destructive, source-coordinate preprocessing for aerial field imagery.

The module deliberately limits itself to geometric rectification, classical
illumination/color operations, masks, and user-provided samples. It never
inpaints tramlines and never performs image or AI super-resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .perspective import correct_perspective

Point = tuple[float, float]
SampleClass = Literal[
    "flattened_crop", "standing_crop", "tramline", "shadow"
]
SAMPLE_CLASSES: tuple[SampleClass, ...] = (
    "flattened_crop",
    "standing_crop",
    "tramline",
    "shadow",
)
SIZE_CLASSES = ("small", "medium", "large")


def identity_transform() -> NDArray[np.float64]:
    return np.eye(3, dtype=np.float64)


def transform_points(
    points: Sequence[Sequence[float]], matrix: Sequence[Sequence[float]] | NDArray[np.float64]
) -> list[Point]:
    if not points:
        return []
    source = np.asarray(points, dtype=np.float64).reshape(1, -1, 2)
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (3, 3):
        raise ValueError("Coordinate transform must be a 3x3 matrix")
    mapped = cv2.perspectiveTransform(source, transform).reshape(-1, 2)
    return [(float(point[0]), float(point[1])) for point in mapped]


def transform_point(
    point: Sequence[float], matrix: Sequence[Sequence[float]] | NDArray[np.float64]
) -> Point:
    return transform_points([point], matrix)[0]


def transform_radius(
    center: Sequence[float],
    radius: float,
    matrix: Sequence[Sequence[float]] | NDArray[np.float64],
) -> float:
    """Map a circular radius to an area-equivalent radius under a homography."""

    value = float(radius)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Radius must be positive and finite")
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (3, 3):
        raise ValueError("Coordinate transform must be a 3x3 matrix")
    if np.allclose(transform, identity_transform(), rtol=0.0, atol=1e-12):
        return value
    center_point = (float(center[0]), float(center[1]))
    angles = np.linspace(0.0, math.tau, 48, endpoint=False)
    boundary = [
        (
            center_point[0] + value * math.cos(float(angle)),
            center_point[1] + value * math.sin(float(angle)),
        )
        for angle in angles
    ]
    transformed = np.asarray(
        transform_points(boundary, transform), dtype=np.float32
    )
    _, (width, height), _ = cv2.fitEllipse(
        transformed.reshape(-1, 1, 2)
    )
    return max(0.1, math.sqrt(float(width) * float(height)) / 2.0)


@dataclass(slots=True)
class PreprocessOperation:
    id: str
    type: Literal["perspective", "illumination", "clahe", "foreground_probability"]
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PreprocessOperation":
        operation_type = str(data.get("type") or "")
        if operation_type not in {
            "perspective",
            "illumination",
            "clahe",
            "foreground_probability",
        }:
            raise ValueError(f"Unsupported preprocessing operation: {operation_type}")
        params = data.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError("Operation params must be an object")
        return cls(
            id=str(data.get("id") or operation_type),
            type=operation_type,  # type: ignore[arg-type]
            enabled=bool(data.get("enabled", True)),
            params=dict(params),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "enabled": self.enabled,
            "params": self.params,
        }


@dataclass(slots=True)
class BrushStroke:
    points: list[Point]
    radius: float
    mode: Literal["paint", "erase"] = "paint"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BrushStroke":
        raw_points = data.get("points", [])
        if not isinstance(raw_points, list):
            raise ValueError("Brush stroke points must be a list")
        points = [_parse_point(point) for point in raw_points]
        radius = float(data.get("radius", 1.0))
        if radius <= 0 or not math.isfinite(radius):
            raise ValueError("Brush radius must be positive")
        mode = str(data.get("mode") or "paint")
        if mode not in {"paint", "erase"}:
            raise ValueError("Brush mode must be paint or erase")
        return cls(points, radius, mode)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [{"x": point[0], "y": point[1]} for point in self.points],
            "radius": self.radius,
            "mode": self.mode,
        }


@dataclass(slots=True)
class PreprocessProfile:
    """Serializable preprocessing recipe; all drawn geometry is source-based."""

    source_width: int
    source_height: int
    operations: list[PreprocessOperation]
    analysis_polygon: list[Point] = field(default_factory=list)
    exclusion_strokes: list[BrushStroke] = field(default_factory=list)
    color_samples: dict[SampleClass, list[BrushStroke]] = field(
        default_factory=lambda: {sample_class: [] for sample_class in SAMPLE_CLASSES}
    )
    tramline: dict[str, Any] = field(default_factory=dict)
    size_classes: dict[str, dict[str, Any]] = field(default_factory=dict)
    detection: dict[str, Any] = field(default_factory=dict)
    display: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.2"
    coordinate_space: str = "source_image_pixels"

    @classmethod
    def default(cls, width: int, height: int) -> "PreprocessProfile":
        maximum = max(width, height)
        return cls(
            source_width=int(width),
            source_height=int(height),
            operations=[
                PreprocessOperation(
                    id="perspective",
                    type="perspective",
                    enabled=False,
                    params={
                        "mode": "reference_quad",
                        "points": [],
                        "preserve_full_image": True,
                    },
                ),
                PreprocessOperation(
                    id="illumination",
                    type="illumination",
                    enabled=False,
                    params={
                        "estimator": "gaussian",
                        "scale": max(31, int(round(maximum * 0.08)) | 1),
                        "method": "subtract",
                        "strength": 1.0,
                    },
                ),
                PreprocessOperation(
                    id="clahe",
                    type="clahe",
                    enabled=False,
                    params={"clip_limit": 2.0, "tile_grid_size": 8},
                ),
                PreprocessOperation(
                    id="foreground_probability",
                    type="foreground_probability",
                    enabled=True,
                    params={"variance_floor": 36.0, "temperature": 1.0},
                ),
            ],
            tramline={
                "enabled": False,
                "lines": [],
                "mask_width": max(3.0, maximum * 0.006),
                "visible": True,
                "detection": {
                    "min_length_ratio": 0.28,
                    "angle_tolerance_degrees": 9.0,
                    "expected_spacing_pixels": 0.0,
                    "regularity": 0.72,
                },
            },
            size_classes={
                "small": {
                    "min_radius": 2.0,
                    "max_radius": max(5.0, maximum * 0.012),
                    "samples": [],
                    "enabled": True,
                },
                "medium": {
                    "min_radius": max(5.0, maximum * 0.010),
                    "max_radius": max(14.0, maximum * 0.035),
                    "samples": [],
                    "enabled": True,
                },
                "large": {
                    "min_radius": max(14.0, maximum * 0.030),
                    "max_radius": max(42.0, maximum * 0.075),
                    "samples": [],
                    "enabled": True,
                },
            },
            detection={
                "tile_size": 512,
                "large_tile_size": 1024,
                "tile_overlap": 0.20,
                "watershed": False,
                "min_final_confidence": 0.50,
                "hough_accumulator_threshold": 32.0,
                "main_arm": {
                    "enabled": True,
                    "min_radius": max(10.0, maximum * 0.014),
                    "max_radius": max(36.0, maximum * 0.060),
                    "min_final_confidence": 0.50,
                    "hough_accumulator_threshold": 48.0,
                    "exclude_sampled_side_ranges": True,
                },
                "native_resolution_only": True,
            },
            display={
                "show_exclusion_mask": True,
                "show_analysis_polygon": True,
                "mask_opacity": 0.45,
                "show_tramline_suggestion": True,
            },
        )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        fallback_size: tuple[int, int] | None = None,
    ) -> "PreprocessProfile":
        if fallback_size is None:
            fallback_size = (
                int(data.get("source_width", 0)),
                int(data.get("source_height", 0)),
            )
        width = int(data.get("source_width") or fallback_size[0])
        height = int(data.get("source_height") or fallback_size[1])
        if width <= 0 or height <= 0:
            raise ValueError("Profile source dimensions must be positive")
        defaults = cls.default(width, height)
        raw_operations = data.get("operations", [])
        operations = (
            [
                PreprocessOperation.from_mapping(item)
                for item in raw_operations
                if isinstance(item, Mapping)
            ]
            if isinstance(raw_operations, list) and raw_operations
            else defaults.operations
        )
        if sum(operation.type == "perspective" for operation in operations) > 1:
            raise ValueError("Only one perspective operation is supported")
        raw_polygon = data.get("analysis_polygon", [])
        if not isinstance(raw_polygon, list):
            raise ValueError("analysis_polygon must be a list")
        polygon = [_parse_point(point) for point in raw_polygon]
        raw_strokes = data.get("exclusion_strokes", [])
        if not isinstance(raw_strokes, list):
            raise ValueError("exclusion_strokes must be a list")
        exclusion_strokes = [
            BrushStroke.from_mapping(stroke)
            for stroke in raw_strokes
            if isinstance(stroke, Mapping)
        ]
        raw_samples = data.get("color_samples", {})
        color_samples: dict[SampleClass, list[BrushStroke]] = {
            sample_class: [] for sample_class in SAMPLE_CLASSES
        }
        if isinstance(raw_samples, Mapping):
            for sample_class in SAMPLE_CLASSES:
                class_strokes = raw_samples.get(sample_class, [])
                if isinstance(class_strokes, list):
                    color_samples[sample_class] = [
                        BrushStroke.from_mapping(stroke)
                        for stroke in class_strokes
                        if isinstance(stroke, Mapping)
                    ]
        tramline = dict(defaults.tramline)
        if isinstance(data.get("tramline"), Mapping):
            tramline.update(dict(data["tramline"]))
        tramline["mask_width"] = max(1.0, float(tramline.get("mask_width", 5.0)))
        tramline["enabled"] = bool(tramline.get("enabled", False))
        raw_lines = tramline.get("lines", [])
        tramline["lines"] = _normalize_lines(raw_lines)
        detection_settings = dict(defaults.tramline["detection"])
        if isinstance(tramline.get("detection"), Mapping):
            detection_settings.update(dict(tramline["detection"]))
        detection_settings["min_length_ratio"] = min(
            0.9,
            max(0.1, float(detection_settings.get("min_length_ratio", 0.28))),
        )
        detection_settings["angle_tolerance_degrees"] = min(
            30.0,
            max(
                2.0,
                float(
                    detection_settings.get("angle_tolerance_degrees", 9.0)
                ),
            ),
        )
        detection_settings["expected_spacing_pixels"] = max(
            0.0,
            float(detection_settings.get("expected_spacing_pixels", 0.0)),
        )
        detection_settings["regularity"] = min(
            1.0,
            max(0.0, float(detection_settings.get("regularity", 0.72))),
        )
        tramline["detection"] = detection_settings
        size_classes: dict[str, dict[str, Any]] = {}
        raw_sizes = data.get("size_classes", {})
        for size_name in SIZE_CLASSES:
            settings = dict(defaults.size_classes[size_name])
            if isinstance(raw_sizes, Mapping) and isinstance(
                raw_sizes.get(size_name), Mapping
            ):
                settings.update(dict(raw_sizes[size_name]))
            settings["min_radius"] = float(settings["min_radius"])
            settings["max_radius"] = float(settings["max_radius"])
            if settings["min_radius"] <= 0 or (
                settings["max_radius"] <= settings["min_radius"]
            ):
                raise ValueError(f"Invalid radius interval for {size_name}")
            settings["enabled"] = bool(settings.get("enabled", True))
            samples = settings.get("samples", [])
            if not isinstance(samples, list):
                samples = []
            settings["samples"] = [
                _normalize_circle_sample(sample)
                for sample in samples
                if isinstance(sample, Mapping)
            ]
            size_classes[size_name] = settings
        detection = dict(defaults.detection)
        if isinstance(data.get("detection"), Mapping):
            detection.update(dict(data["detection"]))
        detection["tile_size"] = max(128, int(detection.get("tile_size", 512)))
        detection["large_tile_size"] = max(
            detection["tile_size"],
            int(detection.get("large_tile_size", 1024)),
        )
        detection["tile_overlap"] = min(
            0.45, max(0.0, float(detection.get("tile_overlap", 0.2)))
        )
        detection["hough_accumulator_threshold"] = max(
            10.0,
            float(detection.get("hough_accumulator_threshold", 32.0)),
        )
        main_arm = dict(defaults.detection["main_arm"])
        if isinstance(detection.get("main_arm"), Mapping):
            main_arm.update(dict(detection["main_arm"]))
        main_arm["enabled"] = bool(main_arm.get("enabled", True))
        main_arm["min_radius"] = float(main_arm.get("min_radius", 10.0))
        main_arm["max_radius"] = float(main_arm.get("max_radius", 36.0))
        if (
            main_arm["min_radius"] <= 0
            or main_arm["max_radius"] <= main_arm["min_radius"]
        ):
            raise ValueError("Invalid radius interval for variable main-arm circles")
        main_arm["min_final_confidence"] = min(
            1.0,
            max(
                0.0,
                float(main_arm.get("min_final_confidence", 0.50)),
            ),
        )
        main_arm["hough_accumulator_threshold"] = max(
            10.0,
            float(main_arm.get("hough_accumulator_threshold", 48.0)),
        )
        main_arm["exclude_sampled_side_ranges"] = bool(
            main_arm.get("exclude_sampled_side_ranges", True)
        )
        detection["main_arm"] = main_arm
        # This invariant makes the no-super-resolution promise executable.
        detection["native_resolution_only"] = True
        display = dict(defaults.display)
        if isinstance(data.get("display"), Mapping):
            display.update(dict(data["display"]))
        display["show_exclusion_mask"] = bool(
            display.get("show_exclusion_mask", True)
        )
        display["show_analysis_polygon"] = bool(
            display.get("show_analysis_polygon", True)
        )
        display["show_tramline_suggestion"] = bool(
            display.get("show_tramline_suggestion", True)
        )
        display["mask_opacity"] = min(
            1.0, max(0.0, float(display.get("mask_opacity", 0.45)))
        )
        return cls(
            source_width=width,
            source_height=height,
            operations=operations,
            analysis_polygon=polygon,
            exclusion_strokes=exclusion_strokes,
            color_samples=color_samples,
            tramline=tramline,
            size_classes=size_classes,
            detection=detection,
            display=display,
            schema_version=str(data.get("schema_version") or "1.2"),
            coordinate_space="source_image_pixels",
        )

    def operation(self, operation_type: str) -> PreprocessOperation | None:
        return next(
            (
                operation
                for operation in self.operations
                if operation.type == operation_type
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "coordinate_space": self.coordinate_space,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "operations": [operation.to_dict() for operation in self.operations],
            "analysis_polygon": [
                {"x": point[0], "y": point[1]}
                for point in self.analysis_polygon
            ],
            "exclusion_strokes": [
                stroke.to_dict() for stroke in self.exclusion_strokes
            ],
            "color_samples": {
                sample_class: [
                    stroke.to_dict()
                    for stroke in self.color_samples.get(sample_class, [])
                ]
                for sample_class in SAMPLE_CLASSES
            },
            "tramline": self.tramline,
            "size_classes": self.size_classes,
            "detection": self.detection,
            "display": self.display,
            "constraints": {
                "super_resolution": False,
                "tramline_inpainting": False,
                "image_enlargement_for_detection": False,
            },
        }


@dataclass(slots=True)
class OperationSnapshot:
    id: str
    type: str
    enabled: bool
    before: NDArray[np.uint8]
    after: NDArray[np.uint8]


@dataclass(slots=True)
class PreprocessResult:
    source_image: NDArray[np.uint8]
    rectified: NDArray[np.uint8]
    illumination_corrected: NDArray[np.uint8]
    measurement_image: NDArray[np.uint8]
    foreground_probability: NDArray[np.float32]
    analysis_mask: NDArray[np.uint8]
    exclusion_mask: NDArray[np.uint8]
    source_to_processed: NDArray[np.float64]
    processed_to_source: NDArray[np.float64]
    snapshots: list[OperationSnapshot]
    profile: PreprocessProfile
    background_estimate: NDArray[np.uint8] | None = None

    @property
    def width(self) -> int:
        return int(self.measurement_image.shape[1])

    @property
    def height(self) -> int:
        return int(self.measurement_image.shape[0])


@dataclass(frozen=True, slots=True)
class DetectedLine:
    p1: Point
    p2: Point
    angle_degrees: float
    length: float
    source_p1: Point | None = None
    source_p2: Point | None = None
    support: float = 0.0
    id: str = ""
    manual: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "p1": {"x": self.p1[0], "y": self.p1[1]},
            "p2": {"x": self.p2[0], "y": self.p2[1]},
            "source_p1": (
                None
                if self.source_p1 is None
                else {"x": self.source_p1[0], "y": self.source_p1[1]}
            ),
            "source_p2": (
                None
                if self.source_p2 is None
                else {"x": self.source_p2[0], "y": self.source_p2[1]}
            ),
            "angle_degrees": self.angle_degrees,
            "length": self.length,
            "support": self.support,
            "manual": self.manual,
        }


@dataclass(frozen=True, slots=True)
class TramlineSuggestion:
    lines: tuple[DetectedLine, ...]
    dominant_angle_degrees: float
    angle_dispersion_degrees: float
    confidence: float
    spacing_pixels: float = 0.0
    spacing_consistency: float = 0.0
    orientation_confidence: float = 0.0
    perspective_points_processed: tuple[Point, ...] = ()
    perspective_points_source: tuple[Point, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "lines": [line.to_dict() for line in self.lines],
            "dominant_angle_degrees": self.dominant_angle_degrees,
            "angle_dispersion_degrees": self.angle_dispersion_degrees,
            "confidence": self.confidence,
            "spacing_pixels": self.spacing_pixels,
            "spacing_consistency": self.spacing_consistency,
            "orientation_confidence": self.orientation_confidence,
            "method": "oriented_periodic_profile_v2",
            "perspective_points_processed": [
                {"x": point[0], "y": point[1]}
                for point in self.perspective_points_processed
            ],
            "perspective_points_source": [
                {"x": point[0], "y": point[1]}
                for point in self.perspective_points_source
            ],
            "requires_user_approval": True,
        }


def _parse_point(data: Any) -> Point:
    if isinstance(data, Mapping):
        point = (float(data["x"]), float(data["y"]))
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        point = (float(data[0]), float(data[1]))
    else:
        raise ValueError("A point must contain x and y")
    if not all(math.isfinite(value) for value in point):
        raise ValueError("Point coordinates must be finite")
    return point


def _normalize_lines(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        return []
    lines: list[dict[str, Any]] = []
    for raw_line in data:
        if not isinstance(raw_line, Mapping):
            continue
        p1_data = raw_line.get("source_p1", raw_line.get("p1"))
        p2_data = raw_line.get("source_p2", raw_line.get("p2"))
        if p1_data is None or p2_data is None:
            continue
        p1 = _parse_point(p1_data)
        p2 = _parse_point(p2_data)
        lines.append(
            {
                "p1": {"x": p1[0], "y": p1[1]},
                "p2": {"x": p2[0], "y": p2[1]},
            }
        )
    return lines


def _normalize_circle_sample(data: Mapping[str, Any]) -> dict[str, float]:
    center_data = data.get("source_center", data.get("center"))
    if center_data is None and (
        data.get("source_x") is not None
        and data.get("source_y") is not None
    ):
        center_data = {
            "x": data["source_x"],
            "y": data["source_y"],
        }
    if center_data is None and (
        data.get("x") is not None and data.get("y") is not None
    ):
        center_data = {"x": data["x"], "y": data["y"]}
    center = _parse_point(center_data)
    radius = float(data.get("radius", 0.0))
    if radius <= 0:
        raise ValueError("Size sample radius must be positive")
    return {
        "source_x": center[0],
        "source_y": center[1],
        "radius": radius,
    }


def rasterize_strokes(
    shape: tuple[int, int], strokes: Iterable[BrushStroke]
) -> NDArray[np.uint8]:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for stroke in strokes:
        if not stroke.points:
            continue
        value = 255 if stroke.mode == "paint" else 0
        thickness = max(1, int(round(stroke.radius * 2.0)))
        integer_points = [
            (
                int(round(min(width - 1, max(0.0, point[0])))),
                int(round(min(height - 1, max(0.0, point[1])))),
            )
            for point in stroke.points
        ]
        if len(integer_points) == 1:
            cv2.circle(
                mask,
                integer_points[0],
                max(1, int(round(stroke.radius))),
                value,
                -1,
                cv2.LINE_AA,
            )
        else:
            cv2.polylines(
                mask,
                [np.asarray(integer_points, dtype=np.int32)],
                False,
                value,
                thickness,
                cv2.LINE_AA,
            )
            for endpoint in (integer_points[0], integer_points[-1]):
                cv2.circle(
                    mask,
                    endpoint,
                    max(1, int(round(stroke.radius))),
                    value,
                    -1,
                    cv2.LINE_AA,
                )
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def polygon_mask(
    shape: tuple[int, int], polygon: Sequence[Point]
) -> NDArray[np.uint8]:
    height, width = shape
    if len(polygon) < 3:
        return np.full((height, width), 255, dtype=np.uint8)
    points = np.asarray(
        [
            (
                int(round(min(width - 1, max(0.0, point[0])))),
                int(round(min(height - 1, max(0.0, point[1])))),
            )
            for point in polygon
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255, cv2.LINE_AA)
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def _warp_mask(
    mask: NDArray[np.uint8],
    matrix: NDArray[np.float64],
    output_size: tuple[int, int],
) -> NDArray[np.uint8]:
    if np.allclose(matrix, identity_transform()) and (
        mask.shape[1],
        mask.shape[0],
    ) == output_size:
        return mask.copy()
    warped = cv2.warpPerspective(
        mask,
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.where(warped >= 128, 255, 0).astype(np.uint8)


def estimate_illumination(
    lightness: NDArray[np.uint8],
    *,
    scale: int,
    estimator: str = "gaussian",
) -> NDArray[np.uint8]:
    size = max(3, int(scale) | 1)
    if estimator == "morph_open":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.morphologyEx(lightness, cv2.MORPH_OPEN, kernel)
    sigma = max(1.0, size / 3.0)
    return cv2.GaussianBlur(
        lightness,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )


def correct_illumination_lab(
    image: NDArray[np.uint8],
    *,
    scale: int = 61,
    method: Literal["subtract", "divide"] = "subtract",
    estimator: str = "gaussian",
    strength: float = 1.0,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    background = estimate_illumination(
        lightness, scale=scale, estimator=estimator
    )
    source = lightness.astype(np.float32)
    background_float = background.astype(np.float32)
    target_level = float(np.median(background_float))
    if method == "divide":
        corrected = source / np.maximum(background_float, 1.0) * target_level
    elif method == "subtract":
        corrected = source - background_float + target_level
    else:
        raise ValueError("Illumination method must be subtract or divide")
    blend = min(1.0, max(0.0, float(strength)))
    corrected = source * (1.0 - blend) + corrected * blend
    output_lab = lab.copy()
    output_lab[:, :, 0] = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR), background


def apply_clahe_lab(
    image: NDArray[np.uint8],
    *,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> NDArray[np.uint8]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    grid = max(2, int(tile_grid_size))
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(clip_limit)),
        tileGridSize=(grid, grid),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def foreground_probability_from_samples(
    image: NDArray[np.uint8],
    sample_masks: Mapping[str, NDArray[np.uint8]],
    *,
    variance_floor: float = 36.0,
    temperature: float = 1.0,
) -> NDArray[np.float32]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    models: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}
    for sample_class in SAMPLE_CLASSES:
        mask = sample_masks.get(sample_class)
        if mask is None:
            continue
        values = lab[mask > 0]
        if len(values) < 5:
            continue
        mean = np.mean(values, axis=0)
        variance = np.maximum(np.var(values, axis=0), float(variance_floor))
        models[sample_class] = (
            mean.astype(np.float32),
            variance.astype(np.float32),
        )
    flattened = models.get("flattened_crop")
    if flattened is None:
        return np.zeros(lab.shape[:2], dtype=np.float32)

    def likelihood(
        model: tuple[NDArray[np.float32], NDArray[np.float32]]
    ) -> NDArray[np.float32]:
        mean, variance = model
        squared_distance = np.sum((lab - mean) ** 2 / variance, axis=2)
        divisor = max(0.1, float(temperature)) * 3.0
        return np.exp(-0.5 * squared_distance / divisor).astype(np.float32)

    foreground = likelihood(flattened)
    negative_likelihoods = [
        likelihood(model)
        for sample_class, model in models.items()
        if sample_class != "flattened_crop"
    ]
    if not negative_likelihoods:
        return np.clip(foreground, 0.0, 1.0)
    negative = np.sum(np.stack(negative_likelihoods), axis=0)
    probability = foreground / np.maximum(foreground + negative, 1e-6)
    return np.clip(probability, 0.0, 1.0).astype(np.float32)


def _draw_source_lines(
    shape: tuple[int, int],
    lines: Sequence[Mapping[str, Any]],
    width: float,
) -> NDArray[np.uint8]:
    mask = np.zeros(shape, dtype=np.uint8)
    for line in lines:
        try:
            p1 = _parse_point(line.get("p1"))
            p2 = _parse_point(line.get("p2"))
        except (KeyError, TypeError, ValueError):
            continue
        cv2.line(
            mask,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            255,
            max(1, int(round(width))),
            cv2.LINE_AA,
        )
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def apply_preprocess_stack(
    source_image: NDArray[np.uint8],
    profile: PreprocessProfile,
) -> PreprocessResult:
    if source_image.size == 0:
        raise ValueError("Source image cannot be empty")
    source_height, source_width = source_image.shape[:2]
    if (
        profile.source_width != source_width
        or profile.source_height != source_height
    ):
        raise ValueError(
            "Profile source dimensions do not match the current source image"
        )
    current = source_image.copy()
    rectified = current.copy()
    source_to_processed = identity_transform()
    processed_to_source = identity_transform()
    snapshots: list[OperationSnapshot] = []
    background_estimate: NDArray[np.uint8] | None = None

    for operation in profile.operations:
        before = current.copy()
        if operation.enabled and operation.type == "perspective":
            raw_points = operation.params.get("points", [])
            if not isinstance(raw_points, list) or len(raw_points) != 4:
                raise ValueError("Enabled perspective operation requires four points")
            points = [_parse_point(point) for point in raw_points]
            mode = str(operation.params.get("mode") or "reference_quad")
            perspective_result = correct_perspective(
                source_image,
                points,
                mode=mode,  # type: ignore[arg-type]
                preserve_full_image=bool(
                    operation.params.get("preserve_full_image", True)
                ),
            )
            current = perspective_result.image
            rectified = current.copy()
            source_to_processed = perspective_result.matrix
            processed_to_source = np.linalg.inv(source_to_processed)
        elif operation.enabled and operation.type == "illumination":
            current, background_estimate = correct_illumination_lab(
                current,
                scale=int(operation.params.get("scale", 61)),
                method=str(
                    operation.params.get("method") or "subtract"
                ),  # type: ignore[arg-type]
                estimator=str(operation.params.get("estimator") or "gaussian"),
                strength=float(operation.params.get("strength", 1.0)),
            )
        elif operation.enabled and operation.type == "clahe":
            current = apply_clahe_lab(
                current,
                clip_limit=float(operation.params.get("clip_limit", 2.0)),
                tile_grid_size=int(operation.params.get("tile_grid_size", 8)),
            )
        # Probability is computed after geometric masks are available below.
        snapshots.append(
            OperationSnapshot(
                id=operation.id,
                type=operation.type,
                enabled=operation.enabled,
                before=before,
                after=current.copy(),
            )
        )

    output_size = (current.shape[1], current.shape[0])
    source_shape = (source_height, source_width)
    source_analysis = polygon_mask(source_shape, profile.analysis_polygon)
    analysis_mask = _warp_mask(
        source_analysis, source_to_processed, output_size
    )
    source_exclusion = rasterize_strokes(
        source_shape, profile.exclusion_strokes
    )
    tramline_lines = profile.tramline.get("lines", [])
    if profile.tramline.get("enabled") and isinstance(tramline_lines, list):
        source_tramlines = _draw_source_lines(
            source_shape,
            tramline_lines,
            float(profile.tramline.get("mask_width", 5.0)),
        )
        source_exclusion = cv2.bitwise_or(source_exclusion, source_tramlines)
    exclusion_mask = _warp_mask(
        source_exclusion, source_to_processed, output_size
    )
    exclusion_mask = cv2.bitwise_or(
        exclusion_mask, cv2.bitwise_not(analysis_mask)
    )
    sample_masks: dict[str, NDArray[np.uint8]] = {}
    for sample_class in SAMPLE_CLASSES:
        source_sample_mask = rasterize_strokes(
            source_shape, profile.color_samples.get(sample_class, [])
        )
        sample_masks[sample_class] = _warp_mask(
            source_sample_mask, source_to_processed, output_size
        )
    foreground_operation = profile.operation("foreground_probability")
    if foreground_operation is not None and foreground_operation.enabled:
        foreground_probability = foreground_probability_from_samples(
            current,
            sample_masks,
            variance_floor=float(
                foreground_operation.params.get("variance_floor", 36.0)
            ),
            temperature=float(
                foreground_operation.params.get("temperature", 1.0)
            ),
        )
    else:
        foreground_probability = np.zeros(current.shape[:2], dtype=np.float32)
    foreground_probability[exclusion_mask > 0] = 0.0
    illumination_corrected = current.copy()
    # Add a meaningful probability preview to the operation stack.
    if foreground_operation is not None:
        probability_preview = cv2.applyColorMap(
            np.clip(foreground_probability * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS,
        )
        for index, snapshot in enumerate(snapshots):
            if snapshot.id == foreground_operation.id:
                snapshots[index] = OperationSnapshot(
                    id=snapshot.id,
                    type=snapshot.type,
                    enabled=snapshot.enabled,
                    before=current.copy(),
                    after=(
                        probability_preview
                        if foreground_operation.enabled
                        else current.copy()
                    ),
                )
                break
    return PreprocessResult(
        source_image=source_image.copy(),
        rectified=rectified,
        illumination_corrected=illumination_corrected,
        measurement_image=current.copy(),
        foreground_probability=foreground_probability,
        analysis_mask=analysis_mask,
        exclusion_mask=exclusion_mask,
        source_to_processed=source_to_processed,
        processed_to_source=processed_to_source,
        snapshots=snapshots,
        profile=profile,
        background_estimate=background_estimate,
    )


def _orientation_distance(left: float, right: float) -> float:
    difference = abs(left - right) % 180.0
    return min(difference, 180.0 - difference)


def _extend_line_to_bounds(
    midpoint: Point,
    angle_degrees: float,
    width: int,
    height: int,
) -> tuple[Point, Point] | None:
    angle = math.radians(angle_degrees)
    direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    origin = np.asarray(midpoint, dtype=np.float64)
    intersections: list[Point] = []
    for axis, boundary in ((0, 0.0), (0, width - 1.0), (1, 0.0), (1, height - 1.0)):
        if abs(direction[axis]) < 1e-9:
            continue
        parameter = (boundary - origin[axis]) / direction[axis]
        point = origin + parameter * direction
        if -1e-6 <= point[0] <= width - 1 + 1e-6 and (
            -1e-6 <= point[1] <= height - 1 + 1e-6
        ):
            intersections.append((float(point[0]), float(point[1])))
    unique: list[Point] = []
    for point in intersections:
        if not any(math.dist(point, existing) < 1.0 for existing in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    pair = max(
        (
            (first, second)
            for index, first in enumerate(unique)
            for second in unique[index + 1 :]
        ),
        key=lambda item: math.dist(item[0], item[1]),
    )
    return pair


def _orientation_field(
    lightness: NDArray[np.uint8],
    valid_mask: NDArray[np.bool_],
    *,
    angle_tolerance_degrees: float,
    gradient_quantile: float,
) -> tuple[
    float,
    float,
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """Estimate line direction from an axial gradient-orientation histogram."""

    gradient_x = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    tangent = (
        np.degrees(np.arctan2(gradient_y, gradient_x)) + 90.0
    ) % 180.0
    valid_values = magnitude[valid_mask]
    if valid_values.size < 32 or float(np.max(valid_values)) <= 1e-6:
        return 0.0, 0.0, magnitude, tangent.astype(np.float32)
    lower = float(np.quantile(valid_values, gradient_quantile))
    upper = max(lower + 1e-6, float(np.quantile(valid_values, 0.995)))
    weights = np.where(
        valid_mask & (magnitude >= lower),
        np.minimum(magnitude, upper),
        0.0,
    ).astype(np.float32)
    bin_count = 360
    bins = np.floor(tangent * (bin_count / 180.0)).astype(np.int32)
    bins %= bin_count
    histogram = np.bincount(
        bins.ravel(),
        weights=weights.ravel(),
        minlength=bin_count,
    ).astype(np.float32)
    tiled = np.tile(histogram, 3).reshape(1, -1)
    smoothed = cv2.GaussianBlur(tiled, (0, 0), 4.0).ravel()[
        bin_count : bin_count * 2
    ]
    peak_bin = int(np.argmax(smoothed))
    peak_angle = (peak_bin + 0.5) * 180.0 / bin_count
    distance = np.abs(
        ((tangent - peak_angle + 90.0) % 180.0) - 90.0
    )
    refinement_weights = np.where(
        distance <= max(4.0, angle_tolerance_degrees),
        weights,
        0.0,
    )
    doubled = np.radians(tangent * 2.0)
    sine = float(np.sum(refinement_weights * np.sin(doubled)))
    cosine = float(np.sum(refinement_weights * np.cos(doubled)))
    if abs(sine) + abs(cosine) > 1e-9:
        dominant = (
            0.5 * math.degrees(math.atan2(sine, cosine))
        ) % 180.0
    else:
        dominant = peak_angle
    peak_ratio = float(np.max(smoothed)) / max(
        float(np.mean(smoothed)), 1e-6
    )
    orientation_confidence = min(
        1.0, max(0.0, (peak_ratio - 1.0) / 4.0)
    )
    return (
        dominant,
        orientation_confidence,
        magnitude,
        tangent.astype(np.float32),
    )


def _line_evidence_profile(
    lightness: NDArray[np.uint8],
    valid_mask: NDArray[np.bool_],
    magnitude: NDArray[np.float32],
    tangent: NDArray[np.float32],
    *,
    dominant_angle_degrees: float,
    angle_tolerance_degrees: float,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.bool_],
    int,
]:
    """Project dark, directionally aligned evidence onto the line normal."""

    height, width = lightness.shape
    angle = math.radians(dominant_angle_degrees)
    normal = np.asarray(
        [-math.sin(angle), math.cos(angle)], dtype=np.float64
    )
    y_grid, x_grid = np.mgrid[:height, :width]
    rho = x_grid * normal[0] + y_grid * normal[1]
    rho_min = math.floor(float(np.min(rho))) - 2
    rho_index = np.rint(rho - rho_min).astype(np.int32)
    profile_size = int(np.max(rho_index)) + 3
    counts = np.bincount(
        rho_index[valid_mask], minlength=profile_size
    ).astype(np.float32)

    distance = np.abs(
        ((tangent - dominant_angle_degrees + 90.0) % 180.0) - 90.0
    )
    alignment_sigma = max(2.0, angle_tolerance_degrees * 0.55)
    alignment = np.exp(
        -0.5 * np.square(distance / alignment_sigma)
    ).astype(np.float32)
    valid_magnitudes = magnitude[valid_mask]
    gradient_floor = float(np.quantile(valid_magnitudes, 0.55))
    gradient_ceiling = max(
        gradient_floor + 1e-6,
        float(np.quantile(valid_magnitudes, 0.995)),
    )
    oriented_gradient = np.where(
        valid_mask & (magnitude >= gradient_floor),
        np.minimum(magnitude, gradient_ceiling) * alignment,
        0.0,
    )
    background = cv2.GaussianBlur(
        lightness,
        (0, 0),
        max(2.0, min(height, width) * 0.006),
    )
    dark_ridge = np.maximum(
        background.astype(np.float32) - lightness.astype(np.float32),
        0.0,
    )
    evidence = np.where(
        valid_mask,
        oriented_gradient + dark_ridge * 0.18,
        0.0,
    )
    projected = np.bincount(
        rho_index.ravel(),
        weights=evidence.ravel(),
        minlength=profile_size,
    ).astype(np.float32)
    # Partial normalization keeps border tracks eligible without allowing very
    # short, high-contrast fragments to outrank field-spanning tracks.
    profile = projected / np.power(np.maximum(counts, 1.0), 0.62)
    profile = cv2.GaussianBlur(
        profile.reshape(1, -1), (0, 0), 1.7
    ).ravel()
    return profile, counts, valid_mask, rho_min


def _periodic_spacing(
    profile: NDArray[np.float32],
    valid_bins: NDArray[np.bool_],
    *,
    minimum_dimension: int,
    expected_spacing_pixels: float,
) -> tuple[float, float]:
    minimum_spacing = max(10, int(round(minimum_dimension * 0.025)))
    maximum_spacing = min(
        max(minimum_spacing + 1, int(round(minimum_dimension * 0.24))),
        max(minimum_spacing + 1, profile.size // 3),
    )
    if expected_spacing_pixels > 0:
        return (
            float(
                min(
                    maximum_spacing,
                    max(minimum_spacing, expected_spacing_pixels),
                )
            ),
            1.0,
        )
    if np.count_nonzero(valid_bins) < minimum_spacing * 2:
        return float(minimum_spacing), 0.0
    median = float(np.median(profile[valid_bins]))
    positive = np.where(valid_bins, np.maximum(profile - median, 0.0), 0.0)
    positive -= float(np.mean(positive[valid_bins]))
    autocorrelation = np.correlate(positive, positive, mode="full")[
        profile.size - 1 :
    ]
    if float(autocorrelation[0]) <= 1e-9:
        return float(minimum_spacing), 0.0
    autocorrelation /= float(autocorrelation[0])
    candidates: list[tuple[float, int]] = []
    valid_span = max(1, int(np.count_nonzero(valid_bins)))
    for lag in range(minimum_spacing, maximum_spacing + 1):
        if (
            autocorrelation[lag] >= autocorrelation[lag - 1]
            and autocorrelation[lag] > autocorrelation[lag + 1]
        ):
            repetitions = valid_span / max(lag, 1)
            repetition_weight = min(
                1.0, math.sqrt(max(repetitions, 1.0) / 5.0)
            )
            candidates.append(
                (float(autocorrelation[lag]) * repetition_weight, lag)
            )
    if not candidates:
        return float(max(minimum_spacing, round(minimum_dimension * 0.08))), 0.0
    score, spacing = max(candidates)
    return float(spacing), min(1.0, max(0.0, score))


def _regular_line_offsets(
    profile: NDArray[np.float32],
    valid_bins: NDArray[np.bool_],
    *,
    rho_min: int,
    spacing: float,
    regularity: float,
    max_lines: int,
) -> list[tuple[float, float]]:
    valid_indices = np.flatnonzero(valid_bins)
    if valid_indices.size < 2 or spacing <= 0:
        return []
    first = int(valid_indices[0])
    last = int(valid_indices[-1])
    median = float(np.median(profile[valid_bins]))
    deviation = float(
        np.median(np.abs(profile[valid_bins] - median))
    )
    evidence = np.maximum(profile - median, 0.0)
    phase_steps = max(1, int(round(spacing)))
    phase_scores: list[float] = []
    for phase in range(phase_steps):
        start = first + ((phase - first) % phase_steps)
        positions = np.arange(start, last + 1, spacing)
        indices = np.clip(np.rint(positions).astype(np.int32), first, last)
        phase_scores.append(
            float(np.mean(evidence[indices])) if indices.size >= 2 else 0.0
        )
    best_phase = int(np.argmax(phase_scores))
    start = first + ((best_phase - first) % phase_steps)
    predicted = np.arange(float(start), last + 0.5, spacing)
    search_ratio = 0.07 + 0.19 * (1.0 - regularity)
    search_radius = max(2, int(round(spacing * search_ratio)))
    strength_floor = median + deviation * 0.20
    candidates: list[tuple[float, float]] = []
    for prediction in predicted:
        center = int(round(prediction))
        lower = max(first, center - search_radius)
        upper = min(last, center + search_radius)
        if upper <= lower:
            continue
        peak_index = lower + int(np.argmax(profile[lower : upper + 1]))
        strength = float(profile[peak_index])
        if strength < strength_floor:
            continue
        adjusted_index = prediction + (
            peak_index - prediction
        ) * (1.0 - regularity)
        offset = float(adjusted_index + rho_min)
        if candidates and (
            offset - candidates[-1][0] < max(3.0, spacing * 0.45)
        ):
            if strength > candidates[-1][1]:
                candidates[-1] = (offset, strength)
            continue
        candidates.append((offset, strength))
    if len(candidates) > max_lines:
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: item[1][1],
            reverse=True,
        )[:max_lines]
        keep = {index for index, _ in ranked}
        candidates = [
            candidate
            for index, candidate in enumerate(candidates)
            if index in keep
        ]
    return candidates


def detect_parallel_tramlines(
    image: NDArray[np.uint8],
    *,
    processed_to_source: NDArray[np.float64] | None = None,
    detection_mask: NDArray[np.uint8] | None = None,
    min_length_ratio: float = 0.28,
    angle_tolerance_degrees: float = 9.0,
    hough_threshold: int = 55,
    expected_spacing_pixels: float = 0.0,
    regularity: float = 0.72,
    max_lines: int = 32,
) -> TramlineSuggestion:
    """Propose a supported, regularly spaced parallel track family.

    The detector uses a dense orientation field and a periodic normal-axis
    profile instead of selecting the longest probabilistic-Hough fragments.
    This prevents one high-contrast strip from consuming the whole result.
    It only returns editable suggestions and never mutates the image/profile.
    """

    height, width = image.shape[:2]
    if height < 8 or width < 8:
        return TramlineSuggestion((), 0.0, 180.0, 0.0)
    min_length_ratio = min(0.9, max(0.1, float(min_length_ratio)))
    angle_tolerance_degrees = min(
        30.0, max(2.0, float(angle_tolerance_degrees))
    )
    regularity = min(1.0, max(0.0, float(regularity)))
    max_lines = max(2, int(max_lines))
    if detection_mask is None:
        valid_mask = np.ones((height, width), dtype=np.bool_)
    else:
        if detection_mask.shape != (height, width):
            raise ValueError("Tramline detection mask shape must match the image")
        valid_mask = detection_mask > 0
    valid_mask = valid_mask.copy()
    valid_mask[:3, :] = False
    valid_mask[-3:, :] = False
    valid_mask[:, :3] = False
    valid_mask[:, -3:] = False
    if np.count_nonzero(valid_mask) < 64:
        return TramlineSuggestion((), 0.0, 180.0, 0.0)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
    gradient_quantile = min(
        0.78, max(0.55, 0.50 + max(10, int(hough_threshold)) / 500.0)
    )
    (
        dominant_angle,
        orientation_confidence,
        magnitude,
        tangent,
    ) = _orientation_field(
        lightness,
        valid_mask,
        angle_tolerance_degrees=angle_tolerance_degrees,
        gradient_quantile=gradient_quantile,
    )
    profile, counts, _, rho_min = _line_evidence_profile(
        lightness,
        valid_mask,
        magnitude,
        tangent,
        dominant_angle_degrees=dominant_angle,
        angle_tolerance_degrees=angle_tolerance_degrees,
    )
    minimum_supported_length = (
        math.hypot(width, height) * min_length_ratio
    )
    valid_bins = counts >= minimum_supported_length
    if np.count_nonzero(valid_bins) < 2:
        return TramlineSuggestion(
            (),
            dominant_angle,
            180.0,
            0.0,
            orientation_confidence=orientation_confidence,
        )
    spacing, periodicity = _periodic_spacing(
        profile,
        valid_bins,
        minimum_dimension=min(width, height),
        expected_spacing_pixels=float(expected_spacing_pixels),
    )
    offset_strengths = _regular_line_offsets(
        profile,
        valid_bins,
        rho_min=rho_min,
        spacing=spacing,
        regularity=regularity,
        max_lines=max_lines,
    )
    if len(offset_strengths) < 2:
        return TramlineSuggestion(
            (),
            dominant_angle,
            180.0,
            0.0,
            spacing_pixels=spacing,
            orientation_confidence=orientation_confidence,
        )

    normal = np.asarray(
        [
            -math.sin(math.radians(dominant_angle)),
            math.cos(math.radians(dominant_angle)),
        ],
        dtype=np.float64,
    )
    transform = (
        identity_transform()
        if processed_to_source is None
        else np.asarray(processed_to_source, dtype=np.float64)
    )
    strengths = np.asarray(
        [strength for _, strength in offset_strengths], dtype=np.float64
    )
    strength_floor = float(np.min(strengths))
    strength_ceiling = max(
        strength_floor + 1e-6, float(np.quantile(strengths, 0.90))
    )
    detected: list[DetectedLine] = []
    for index, (offset, strength) in enumerate(offset_strengths, start=1):
        midpoint_array = normal * offset
        midpoint = (float(midpoint_array[0]), float(midpoint_array[1]))
        extended = _extend_line_to_bounds(
            midpoint, dominant_angle, width, height
        )
        if extended is None:
            continue
        p1, p2 = extended
        source_p1, source_p2 = transform_points([p1, p2], transform)
        support = min(
            1.0,
            max(
                0.0,
                (strength - strength_floor)
                / (strength_ceiling - strength_floor),
            ),
        )
        detected.append(
            DetectedLine(
                p1=p1,
                p2=p2,
                angle_degrees=dominant_angle,
                length=math.dist(p1, p2),
                source_p1=source_p1,
                source_p2=source_p2,
                support=support,
                id=f"tramline-auto-{index:02d}",
            )
        )
    if len(detected) < 2:
        return TramlineSuggestion(
            tuple(detected),
            dominant_angle,
            180.0,
            0.0,
            spacing_pixels=spacing,
            orientation_confidence=orientation_confidence,
        )
    detected_offsets = np.asarray(
        [
            float(
                np.dot(
                    np.asarray(
                        [
                            (line.p1[0] + line.p2[0]) / 2.0,
                            (line.p1[1] + line.p2[1]) / 2.0,
                        ]
                    ),
                    normal,
                )
            )
            for line in detected
        ],
        dtype=np.float64,
    )
    differences = np.diff(detected_offsets)
    spacing_consistency = (
        1.0
        if differences.size == 0
        else min(
            1.0,
            max(
                0.0,
                1.0
                - float(np.median(np.abs(differences - spacing)))
                / max(spacing, 1.0),
            ),
        )
    )
    support_confidence = float(
        np.mean([line.support for line in detected])
    )
    count_confidence = min(1.0, len(detected) / 6.0)
    confidence = min(
        1.0,
        max(
            0.0,
            count_confidence
            * (
                0.28 * orientation_confidence
                + 0.34 * periodicity
                + 0.22 * spacing_consistency
                + 0.16 * support_confidence
            ),
        ),
    )
    extreme_a, extreme_b = detected[0], detected[-1]
    a1, a2 = extreme_a.p1, extreme_a.p2
    b1, b2 = extreme_b.p1, extreme_b.p2
    if math.dist(a1, b1) + math.dist(a2, b2) > (
        math.dist(a1, b2) + math.dist(a2, b1)
    ):
        b1, b2 = b2, b1
    perspective_processed = (a1, a2, b1, b2)
    perspective_source = tuple(
        transform_points(perspective_processed, transform)
    )
    return TramlineSuggestion(
        lines=tuple(detected),
        dominant_angle_degrees=dominant_angle,
        angle_dispersion_degrees=0.0,
        confidence=confidence,
        spacing_pixels=spacing,
        spacing_consistency=spacing_consistency,
        orientation_confidence=orientation_confidence,
        perspective_points_processed=perspective_processed,
        perspective_points_source=perspective_source,
    )


def write_preprocess_profile(
    profile: PreprocessProfile, path: str | Path
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(profile.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def read_preprocess_profile(
    path: str | Path,
    *,
    fallback_size: tuple[int, int] | None = None,
) -> PreprocessProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Preprocess profile JSON must contain an object")
    return PreprocessProfile.from_mapping(data, fallback_size=fallback_size)
