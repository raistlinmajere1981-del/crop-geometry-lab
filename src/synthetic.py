"""Deterministic six-arm and advanced preprocessing fixtures with ground truth."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .arm_classifier import assign_arms
from .feature_table import AnnotationSet, CircleFeature, write_json
from .preprocessing import transform_point


@dataclass(frozen=True, slots=True)
class PreprocessingFixture:
    image: NDArray[np.uint8]
    circles: list[CircleFeature]
    perspective_quad: tuple[tuple[float, float], ...]
    source_to_aerial: NDArray[np.float64]
    class_points: dict[str, tuple[float, float]]


def generate_synthetic_image(
    *,
    size: int = 720,
    rings: int = 3,
    seed: int = 2001,
    add_texture: bool = True,
    add_field_tracks: bool = True,
) -> tuple[NDArray[np.uint8], list[CircleFeature], tuple[float, float]]:
    if size < 240:
        raise ValueError("Synthetic image size must be at least 240 pixels")
    if rings < 1:
        raise ValueError("At least one ring is required")
    rng = np.random.default_rng(seed)
    center = (size / 2.0, size / 2.0)
    base = np.full((size, size, 3), (92, 132, 112), dtype=np.float32)
    y_grid, x_grid = np.mgrid[:size, :size]
    broad_light = 9.0 * np.sin((x_grid + 0.4 * y_grid) / max(12.0, size / 12.0))
    base += broad_light[..., None]
    if add_texture:
        texture = rng.normal(0.0, 5.0, (size, size, 1))
        base += texture
    image = np.clip(base, 0, 255).astype(np.uint8)

    if add_field_tracks:
        track_color = (75, 112, 91)
        for offset in range(-size, size * 2, max(32, size // 14)):
            cv2.line(
                image,
                (offset, 0),
                (offset + size // 4, size - 1),
                track_color,
                max(1, size // 240),
                cv2.LINE_AA,
            )

    circles: list[CircleFeature] = []
    central_radius = max(8.0, size * 0.023)
    circles.append(
        CircleFeature(
            id="truth-center",
            x=center[0],
            y=center[1],
            radius=central_radius,
            confidence=1.0,
            circle_type="main",
            source="manual",
        )
    )
    radial_step = size * 0.105
    for arm_zero in range(6):
        for ring in range(1, rings + 1):
            angle_degrees = arm_zero * 60.0 + ring * 7.0
            angle = math.radians(angle_degrees)
            distance = radial_step * ring
            x = center[0] + distance * math.cos(angle)
            y = center[1] - distance * math.sin(angle)
            radius = size * (0.026 - 0.0025 * (ring - 1))
            circles.append(
                CircleFeature(
                    id=f"truth-main-{arm_zero + 1}-{ring}",
                    x=x,
                    y=y,
                    radius=radius,
                    confidence=1.0,
                    circle_type="main",
                    arm_id=arm_zero + 1,
                    sequence_index=ring,
                    side="left",
                    source="manual",
                )
            )
            if ring >= 2:
                side_angle = math.radians(angle_degrees - 11.0)
                side_distance = distance + radius * 0.15
                side_radius = radius * 0.47
                circles.append(
                    CircleFeature(
                        id=f"truth-side-{arm_zero + 1}-{ring}",
                        x=center[0] + side_distance * math.cos(side_angle),
                        y=center[1] - side_distance * math.sin(side_angle),
                        radius=side_radius,
                        confidence=1.0,
                        circle_type="side",
                        parent_id=f"truth-main-{arm_zero + 1}-{ring}",
                        arm_id=arm_zero + 1,
                        sequence_index=ring,
                        side="right",
                        source="manual",
                    )
                )

    for circle in circles:
        fill = (168, 194, 159) if circle.circle_type == "main" else (151, 184, 147)
        edge = (47, 76, 61)
        center_point = (int(round(circle.x)), int(round(circle.y)))
        radius = int(round(circle.radius))
        cv2.circle(image, center_point, radius, fill, -1, cv2.LINE_AA)
        cv2.circle(
            image,
            center_point,
            radius,
            edge,
            max(1, size // 300),
            cv2.LINE_AA,
        )
        cv2.circle(
            image,
            center_point,
            max(1, int(radius * 0.72)),
            (178, 201, 169),
            1,
            cv2.LINE_AA,
        )
    circles = assign_arms(circles, center, preserve_manual_fields=True)
    return image, circles, center


def write_synthetic_dataset(
    image_path: str | Path,
    annotation_path: str | Path | None = None,
    *,
    size: int = 720,
    rings: int = 3,
) -> tuple[Path, Path]:
    image, circles, center = generate_synthetic_image(size=size, rings=rings)
    destination = Path(image_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not write synthetic image to {destination}")
    truth_path = (
        Path(annotation_path)
        if annotation_path is not None
        else destination.with_suffix(".truth.json")
    )
    write_json(
        AnnotationSet(
            circles=circles,
            image_id="synthetic-six-arm",
            image_name=destination.name,
            general_center=center,
            arm_phase_degrees=0.0,
        ),
        truth_path,
    )
    return destination, truth_path


def generate_preprocessing_fixture(
    *,
    width: int = 840,
    height: int = 640,
    seed: int = 2026,
    apply_perspective: bool = True,
) -> PreprocessingFixture:
    """Create small/medium/large circles under tracks and uneven illumination."""

    if width < 420 or height < 320:
        raise ValueError("Advanced fixture must be at least 420×320")
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), (82, 122, 94), dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    illumination = (
        0.68
        + 0.46 * (xx / max(1, width - 1))
        + 0.12 * np.sin(yy / max(18.0, height / 7.0))
    )
    base *= illumination[..., None]
    base += rng.normal(0.0, 3.0, base.shape[:2])[..., None]
    rectified = np.clip(base, 0, 255).astype(np.uint8)

    tramline_color = (42, 72, 52)
    for x in range(-80, width + 100, max(58, width // 11)):
        cv2.line(
            rectified,
            (x, 0),
            (x + width // 9, height - 1),
            tramline_color,
            max(2, width // 280),
            cv2.LINE_AA,
        )
        cv2.line(
            rectified,
            (x + 8, 0),
            (x + width // 9 + 8, height - 1),
            (58, 88, 67),
            1,
            cv2.LINE_AA,
        )

    definitions: list[tuple[str, float, float, float]] = [
        ("small", 0.20, 0.22, 5.0),
        ("small", 0.29, 0.31, 7.0),
        ("small", 0.72, 0.21, 6.0),
        ("small", 0.80, 0.35, 8.0),
        ("medium", 0.19, 0.56, 14.0),
        ("medium", 0.39, 0.42, 18.0),
        ("medium", 0.43, 0.46, 17.0),  # touching pair
        ("medium", 0.70, 0.57, 21.0),
        ("large", 0.28, 0.77, 34.0),
        ("large", 0.57, 0.72, 43.0),
        ("large", 0.79, 0.76, 52.0),
    ]
    truth_rectified: list[CircleFeature] = []
    flattened_color = (170, 196, 150)
    for index, (size_class, x_fraction, y_fraction, radius) in enumerate(
        definitions, start=1
    ):
        x = float(width * x_fraction)
        y = float(height * y_fraction)
        cv2.circle(
            rectified,
            (int(round(x)), int(round(y))),
            int(round(radius)),
            flattened_color,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            rectified,
            (int(round(x)), int(round(y))),
            int(round(radius)),
            (40, 68, 49),
            max(1, int(round(radius * 0.10))),
            cv2.LINE_AA,
        )
        truth_rectified.append(
            CircleFeature(
                id=f"advanced-{size_class}-{index:02d}",
                x=x,
                y=y,
                radius=radius,
                size_class=size_class,  # type: ignore[arg-type]
                confidence=1.0,
                final_confidence=1.0,
                source="manual",
                locked=True,
            )
        )
    # A shadow region provides an explicit negative sample class.
    cv2.ellipse(
        rectified,
        (int(width * 0.58), int(height * 0.30)),
        (int(width * 0.10), int(height * 0.055)),
        -12,
        0,
        360,
        (39, 57, 48),
        -1,
        cv2.LINE_AA,
    )

    if apply_perspective:
        source_corners = np.asarray(
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ],
            dtype=np.float32,
        )
        aerial_quad = np.asarray(
            [
                [width * 0.17, height * 0.06],
                [width * 0.83, height * 0.015],
                [width * 0.98, height * 0.96],
                [width * 0.02, height * 0.995],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source_corners, aerial_quad)
        aerial = cv2.warpPerspective(
            rectified,
            matrix,
            (width, height),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_REFLECT,
        )
    else:
        aerial_quad = np.asarray(
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ],
            dtype=np.float32,
        )
        matrix = np.eye(3, dtype=np.float64)
        aerial = rectified

    transformed_truth: list[CircleFeature] = []
    for circle in truth_rectified:
        aerial_x, aerial_y = transform_point((circle.x, circle.y), matrix)
        radius_point = transform_point((circle.x + circle.radius, circle.y), matrix)
        aerial_radius = math.dist((aerial_x, aerial_y), radius_point)
        transformed_truth.append(
            CircleFeature(
                id=circle.id,
                x=aerial_x,
                y=aerial_y,
                processed_x=aerial_x,
                processed_y=aerial_y,
                source_x=aerial_x,
                source_y=aerial_y,
                radius=max(1.0, aerial_radius),
                size_class=circle.size_class,
                confidence=1.0,
                final_confidence=1.0,
                source="manual",
                locked=True,
            )
        )
    class_points_rectified = {
        "flattened_crop": (
            truth_rectified[5].x,
            truth_rectified[5].y,
        ),
        "standing_crop": (width * 0.50, height * 0.88),
        "tramline": (width * 0.50 + width / 22, height * 0.50),
        "shadow": (width * 0.58, height * 0.30),
    }
    class_points = {
        name: transform_point(point, matrix)
        for name, point in class_points_rectified.items()
    }
    return PreprocessingFixture(
        image=aerial,
        circles=transformed_truth,
        perspective_quad=tuple(
            (float(point[0]), float(point[1])) for point in aerial_quad
        ),
        source_to_aerial=np.asarray(matrix, dtype=np.float64),
        class_points=class_points,
    )


def write_preprocessing_dataset(
    image_path: str | Path,
    annotation_path: str | Path | None = None,
    *,
    width: int = 840,
    height: int = 640,
) -> tuple[Path, Path]:
    fixture = generate_preprocessing_fixture(width=width, height=height)
    destination = Path(image_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), fixture.image):
        raise OSError(f"Could not write preprocessing fixture to {destination}")
    truth_path = (
        Path(annotation_path)
        if annotation_path is not None
        else destination.with_suffix(".truth.json")
    )
    payload = {
        "schema_version": "1.0",
        "image_name": destination.name,
        "perspective_quad": [
            {"x": point[0], "y": point[1]}
            for point in fixture.perspective_quad
        ],
        "source_to_aerial": fixture.source_to_aerial.tolist(),
        "class_points": {
            name: {"x": point[0], "y": point[1]}
            for name, point in fixture.class_points.items()
        },
        "circles": [circle.to_dict() for circle in fixture.circles],
    }
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination, truth_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/synthetic_six_arm.png"),
    )
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--rings", type=int, default=3)
    parser.add_argument(
        "--preprocessing",
        action="store_true",
        help="Generate the perspective/illumination/size-class fixture",
    )
    args = parser.parse_args()
    if args.preprocessing:
        image_path, truth_path = write_preprocessing_dataset(args.output)
    else:
        image_path, truth_path = write_synthetic_dataset(
            args.output, size=args.size, rings=args.rings
        )
    print(f"Synthetic image: {image_path}")
    print(f"Ground truth: {truth_path}")


if __name__ == "__main__":
    main()
