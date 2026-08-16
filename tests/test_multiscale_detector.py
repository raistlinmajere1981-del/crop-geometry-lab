from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from src.multiscale_detector import (
    DetectionEvidence,
    _valid_center,
    detect_multiscale_circles,
    effective_radius_ranges,
    fuse_candidates,
)
from src.preprocessing import BrushStroke, PreprocessProfile, apply_preprocess_stack
from src.synthetic import generate_preprocessing_fixture


def configured_result():
    fixture = generate_preprocessing_fixture(
        width=640,
        height=480,
        apply_perspective=False,
    )
    profile = PreprocessProfile.default(640, 480)
    for sample_class, point in fixture.class_points.items():
        profile.color_samples[sample_class] = [
            BrushStroke([point], 7, "paint")
        ]
    profile.detection["tile_size"] = 320
    profile.detection["tile_overlap"] = 0.25
    profile.detection["min_final_confidence"] = 0.14
    return fixture, apply_preprocess_stack(fixture.image, profile)


def test_size_specific_detectors_produce_all_three_classes() -> None:
    _, result = configured_result()
    circles = detect_multiscale_circles(result, max_candidates=300)
    classes = {circle.size_class for circle in circles}

    assert {"small", "medium", "large"} <= classes
    for circle in circles:
        assert circle.processed_x == circle.x
        assert circle.processed_y == circle.y
        assert circle.source_x is not None
        assert circle.source_y is not None
        assert circle.edge_score is not None
        assert circle.color_score is not None
        assert circle.circularity_score is not None
        assert circle.detector_agreement is not None
        assert circle.symmetry_support is not None
        assert circle.final_confidence == circle.confidence


def test_locked_circle_is_preserved_across_detection_round() -> None:
    fixture, result = configured_result()
    locked = replace(
        fixture.circles[0],
        id="approved-small",
        locked=True,
        source="manual_override",
    )
    circles = detect_multiscale_circles(
        result,
        locked_circles=[locked],
        max_candidates=300,
    )
    approved = next(circle for circle in circles if circle.id == "approved-small")

    assert approved.locked is True
    assert approved.source_x == locked.source_x
    assert approved.source_y == locked.source_y
    assert approved.radius == locked.radius


def test_watershed_mode_runs_on_touching_medium_regions() -> None:
    _, result = configured_result()
    result.profile.detection["watershed"] = True
    circles = detect_multiscale_circles(result, max_candidates=300)

    assert any(
        circle.size_class == "medium"
        and circle.detector
        and "watershed" in circle.detector
        for circle in circles
    )


def test_variable_main_arm_pass_is_independent_from_side_size_standards() -> None:
    image = np.full((420, 420, 3), 60, dtype=np.uint8)
    side_truth = [
        (55, 60, 4, "small"),
        (110, 65, 7, "medium"),
        (175, 70, 10, "large"),
    ]
    main_truth = [(80, 220, 18), (200, 220, 26), (330, 225, 34)]
    for x, y, radius, *_ in [*side_truth, *main_truth]:
        cv2.circle(
            image,
            (x, y),
            radius,
            (215, 215, 215),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            image,
            (x, y),
            radius,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )

    profile = PreprocessProfile.default(420, 420)
    for x, y, radius, size_class in side_truth:
        profile.size_classes[size_class]["samples"] = [
            {
                "source_x": x,
                "source_y": y,
                "radius": radius,
            }
        ]
    profile.detection["min_final_confidence"] = 0.14
    profile.detection["main_arm"].update(
        {
            "min_radius": 16,
            "max_radius": 38,
            "min_final_confidence": 0.25,
            "hough_accumulator_threshold": 14,
        }
    )
    result = apply_preprocess_stack(image, profile)
    side_ranges = effective_radius_ranges(
        profile, result.source_to_processed
    )
    circles = detect_multiscale_circles(result, max_candidates=200)

    assert max(maximum for _, maximum in side_ranges.values()) == 15.5
    for truth_x, truth_y, truth_radius in main_truth:
        assert any(
            circle.circle_type == "main"
            and circle.size_class is None
            and circle.detector is not None
            and "main_" in circle.detector
            and np.hypot(circle.x - truth_x, circle.y - truth_y) < 8
            and abs(circle.radius - truth_radius) < 8
            for circle in circles
        )
    for truth_x, truth_y, _truth_radius, _size_class in side_truth:
        assert any(
            circle.circle_type == "satellite"
            and circle.size_class in {"small", "medium", "large"}
            and np.hypot(circle.x - truth_x, circle.y - truth_y) < 8
            for circle in circles
        )


def test_subpixel_center_at_tile_edge_does_not_overflow_mask() -> None:
    exclusion = np.zeros((512, 512), dtype=np.uint8)

    result = _valid_center(300.0, 511.6, 8.0, exclusion)

    assert isinstance(result, bool)


def test_candidate_limit_keeps_highest_confidence_not_topmost_geometry() -> None:
    top_low_quality = DetectionEvidence(
        x=20,
        y=10,
        radius=8,
        size_class="medium",
        detector="contour",
        edge_score=0.45,
        color_score=0.5,
        circularity_score=0.45,
        symmetry_support=0.4,
        raw_score=0.4,
    )
    bottom_high_quality = DetectionEvidence(
        x=300,
        y=450,
        radius=8,
        size_class="medium",
        detector="contour",
        edge_score=1.0,
        color_score=1.0,
        circularity_score=1.0,
        symmetry_support=1.0,
        raw_score=1.0,
    )

    selected = fuse_candidates(
        [top_low_quality, bottom_high_quality],
        min_final_confidence=0.0,
        processed_to_source=np.eye(3, dtype=np.float64),
        max_candidates=1,
    )

    assert len(selected) == 1
    assert selected[0].x == 300
    assert selected[0].y == 450
