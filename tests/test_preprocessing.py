from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from src.preprocessing import (
    BrushStroke,
    PreprocessProfile,
    apply_preprocess_stack,
    detect_parallel_tramlines,
    rasterize_strokes,
    read_preprocess_profile,
    transform_point,
    write_preprocess_profile,
)
from src.multiscale_detector import effective_radius_ranges
from src.synthetic import generate_preprocessing_fixture


def stroke_at(point: tuple[float, float], radius: float = 7) -> BrushStroke:
    return BrushStroke(points=[point], radius=radius, mode="paint")


def test_paint_erase_and_analysis_polygon_are_non_destructive() -> None:
    fixture = generate_preprocessing_fixture(
        width=520,
        height=380,
        apply_perspective=False,
    )
    original = fixture.image.copy()
    profile = PreprocessProfile.default(520, 380)
    profile.analysis_polygon = [(40, 30), (480, 30), (480, 350), (40, 350)]
    profile.exclusion_strokes = [
        BrushStroke([(100, 100), (180, 100)], 12, "paint"),
        BrushStroke([(140, 100)], 5, "erase"),
    ]

    result = apply_preprocess_stack(fixture.image, profile)
    mask = rasterize_strokes(
        (380, 520), profile.exclusion_strokes
    )

    assert np.array_equal(fixture.image, original)
    assert mask[100, 100] == 255
    assert mask[100, 140] == 0
    assert result.exclusion_mask[10, 10] == 255
    assert result.exclusion_mask[200, 260] == 0


def test_perspective_transform_round_trip_and_operation_toggle() -> None:
    fixture = generate_preprocessing_fixture(width=600, height=450)
    profile = PreprocessProfile.default(600, 450)
    perspective = profile.operation("perspective")
    assert perspective is not None
    perspective.enabled = True
    perspective.params = {
        "mode": "reference_quad",
        "points": [
            {"x": point[0], "y": point[1]}
            for point in fixture.perspective_quad
        ],
    }

    corrected = apply_preprocess_stack(fixture.image, profile)
    source_point = fixture.class_points["standing_crop"]
    processed = transform_point(source_point, corrected.source_to_processed)
    restored = transform_point(processed, corrected.processed_to_source)

    assert math.dist(source_point, restored) < 1e-4
    assert corrected.rectified.shape == fixture.image.shape
    corners = [
        (0, 0),
        (fixture.image.shape[1] - 1, 0),
        (fixture.image.shape[1] - 1, fixture.image.shape[0] - 1),
        (0, fixture.image.shape[0] - 1),
    ]
    mapped_corners = [
        transform_point(point, corrected.source_to_processed)
        for point in corners
    ]
    assert all(
        -1e-4 <= x <= fixture.image.shape[1] - 1 + 1e-4
        and -1e-4 <= y <= fixture.image.shape[0] - 1 + 1e-4
        for x, y in mapped_corners
    )
    perspective.enabled = False
    disabled = apply_preprocess_stack(fixture.image, profile)
    assert np.array_equal(disabled.rectified, fixture.image)
    assert not np.array_equal(corrected.rectified, disabled.rectified)


def test_lab_illumination_correction_reduces_low_frequency_variation() -> None:
    fixture = generate_preprocessing_fixture(
        width=620,
        height=460,
        apply_perspective=False,
    )
    profile = PreprocessProfile.default(620, 460)
    illumination = profile.operation("illumination")
    assert illumination is not None
    illumination.enabled = True
    illumination.params.update(
        {"scale": 91, "method": "divide", "strength": 1.0}
    )
    result = apply_preprocess_stack(fixture.image, profile)

    before_l = cv2.cvtColor(fixture.image, cv2.COLOR_BGR2LAB)[:, :, 0]
    after_l = cv2.cvtColor(
        result.illumination_corrected, cv2.COLOR_BGR2LAB
    )[:, :, 0]
    before_background = cv2.GaussianBlur(before_l, (0, 0), 35)
    after_background = cv2.GaussianBlur(after_l, (0, 0), 35)

    assert float(after_background.std()) < float(before_background.std())


def test_painted_color_classes_create_foreground_probability() -> None:
    fixture = generate_preprocessing_fixture(
        width=620,
        height=460,
        apply_perspective=False,
    )
    profile = PreprocessProfile.default(620, 460)
    for sample_class, point in fixture.class_points.items():
        profile.color_samples[sample_class] = [stroke_at(point, 6)]

    result = apply_preprocess_stack(fixture.image, profile)
    probability = result.foreground_probability
    flat_point = fixture.class_points["flattened_crop"]
    negative_points = [
        fixture.class_points[name]
        for name in ("standing_crop", "tramline", "shadow")
    ]
    flat_value = float(
        probability[int(round(flat_point[1])), int(round(flat_point[0]))]
    )
    negative_values = [
        float(probability[int(round(point[1])), int(round(point[0]))])
        for point in negative_points
    ]

    assert flat_value > 0.55
    assert flat_value > max(negative_values)


def test_parallel_tramlines_return_unapplied_suggestion() -> None:
    fixture = generate_preprocessing_fixture(
        width=620,
        height=460,
        apply_perspective=True,
    )
    suggestion = detect_parallel_tramlines(
        fixture.image,
        min_length_ratio=0.20,
        hough_threshold=35,
    )

    assert len(suggestion.lines) >= 2
    assert len(suggestion.perspective_points_source) == 4
    assert suggestion.to_dict()["requires_user_approval"] is True


def test_parallel_tramlines_use_a_regular_supported_track_family() -> None:
    fixture = generate_preprocessing_fixture(
        width=620,
        height=460,
        apply_perspective=False,
    )
    suggestion = detect_parallel_tramlines(fixture.image)

    assert 7 <= len(suggestion.lines) <= 14
    assert suggestion.spacing_pixels == pytest.approx(58, abs=3)
    assert suggestion.spacing_consistency > 0.95
    assert suggestion.orientation_confidence > 0.8
    expected_angle = math.degrees(math.atan2(459, 620 // 9)) % 180
    assert min(
        abs(suggestion.dominant_angle_degrees - expected_angle),
        180 - abs(suggestion.dominant_angle_degrees - expected_angle),
    ) < 3
    assert all(
        line.angle_degrees == pytest.approx(
            suggestion.dominant_angle_degrees
        )
        for line in suggestion.lines
    )


def test_parallel_tramline_spacing_can_be_locked_without_losing_regularity() -> None:
    fixture = generate_preprocessing_fixture(
        width=620,
        height=460,
        apply_perspective=False,
    )
    suggestion = detect_parallel_tramlines(
        fixture.image,
        expected_spacing_pixels=60,
        regularity=1.0,
    )
    angle = math.radians(suggestion.dominant_angle_degrees)
    normal = (-math.sin(angle), math.cos(angle))
    offsets = [
        (
            (line.p1[0] + line.p2[0]) * 0.5 * normal[0]
            + (line.p1[1] + line.p2[1]) * 0.5 * normal[1]
        )
        for line in suggestion.lines
    ]

    assert suggestion.spacing_pixels == 60
    assert len(offsets) >= 6
    assert np.diff(offsets) == pytest.approx(60, abs=1e-5)
    assert suggestion.to_dict()["method"] == "oriented_periodic_profile_v2"


def test_preprocess_profile_json_round_trip(tmp_path) -> None:
    profile = PreprocessProfile.default(500, 400)
    profile.analysis_polygon = [(10, 10), (490, 10), (490, 390)]
    profile.exclusion_strokes = [stroke_at((100, 120))]
    profile.display["show_exclusion_mask"] = False
    profile.display["mask_opacity"] = 0.27
    profile.size_classes["small"]["samples"] = [
        {"source_x": 135.5, "source_y": 214.25, "radius": 8.75}
    ]
    path = write_preprocess_profile(profile, tmp_path / "preprocess_profile.json")
    loaded = read_preprocess_profile(path)

    assert loaded.to_dict() == profile.to_dict()


def test_size_sample_radius_maps_from_source_to_rectified_pixels() -> None:
    fixture = generate_preprocessing_fixture(width=600, height=450)
    profile = PreprocessProfile.default(600, 450)
    perspective = profile.operation("perspective")
    assert perspective is not None
    perspective.enabled = True
    perspective.params = {
        "mode": "reference_quad",
        "points": [
            {"x": point[0], "y": point[1]}
            for point in fixture.perspective_quad
        ],
    }
    center = fixture.class_points["flattened_crop"]
    source_radius = 18.0
    profile.size_classes["small"]["samples"] = [
        {
            "source_center": {"x": center[0], "y": center[1]},
            "source_x": center[0],
            "source_y": center[1],
            "radius": source_radius,
        }
    ]

    result = apply_preprocess_stack(fixture.image, profile)
    mapped_center = transform_point(center, result.source_to_processed)
    mapped_edge = transform_point(
        (center[0] + source_radius, center[1]),
        result.source_to_processed,
    )
    mapped_radius = math.dist(mapped_center, mapped_edge)
    minimum, maximum = effective_radius_ranges(
        profile, result.source_to_processed
    )["small"]

    assert minimum == pytest.approx(mapped_radius * 0.60)
    assert maximum == pytest.approx(mapped_radius * 1.55)
