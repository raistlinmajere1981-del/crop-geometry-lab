from __future__ import annotations

import math

import pytest

from src.arm_classifier import (
    assign_arms,
    estimate_spiral_parameters,
    polar_coordinates,
)
from src.feature_table import CircleFeature


def circle_at(circle_id: str, angle: float, distance: float = 100) -> CircleFeature:
    radians = math.radians(angle)
    return CircleFeature(
        id=circle_id,
        x=200 + distance * math.cos(radians),
        y=200 - distance * math.sin(radians),
        radius=8,
        source="automatic",
    )


def test_polar_coordinates_use_east_zero_and_counter_clockwise() -> None:
    distance, angle = polar_coordinates((200, 100), (200, 200))
    assert distance == pytest.approx(100)
    assert angle == pytest.approx(90)


def test_six_spokes_receive_distinct_arm_ids_and_radial_sequence() -> None:
    circles = [
        circle_at(f"arm-{index + 1}", index * 60)
        for index in range(6)
    ]
    circles.append(circle_at("arm-1-inner", 0, distance=50))
    result = assign_arms(circles, (200, 200), phase_degrees=0)
    by_id = {circle.id: circle for circle in result}

    assert [by_id[f"arm-{index + 1}"].arm_id for index in range(6)] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert by_id["arm-1-inner"].sequence_index == 1
    assert by_id["arm-1"].sequence_index == 2
    assert by_id["arm-1"].side == "center"


def test_manual_arm_and_side_are_not_overwritten() -> None:
    circle = circle_at("manual", 118)
    circle.arm_id = 5
    circle.side = "right"
    classified = assign_arms([circle], (200, 200), preserve_manual_fields=True)[0]

    assert classified.arm_id == 5
    assert classified.side == "right"
    assert classified.center_angle == pytest.approx(118)


def test_api_style_field_locks_distinguish_manual_and_derived_arm_values() -> None:
    circle = circle_at("editable", 120)
    circle.arm_id = 5
    circle.side = "right"

    recomputed = assign_arms(
        [circle],
        (200, 200),
        locked_fields={},
    )[0]
    preserved = assign_arms(
        [circle],
        (200, 200),
        locked_fields={"editable": {"arm_id", "side"}},
    )[0]

    assert recomputed.arm_id == 3
    assert recomputed.side == "center"
    assert preserved.arm_id == 5
    assert preserved.side == "right"


def test_stale_sequence_is_recomputed_after_manual_arm_change() -> None:
    inner = circle_at("inner", 60, distance=50)
    outer = circle_at("outer", 60, distance=100)
    inner.arm_id = 2
    inner.sequence_index = 91
    outer.arm_id = 2
    outer.sequence_index = 92

    classified = assign_arms(
        [outer, inner],
        (200, 200),
        locked_fields={
            "inner": {"arm_id"},
            "outer": {"arm_id"},
        },
    )
    by_id = {circle.id: circle for circle in classified}

    assert by_id["inner"].sequence_index == 1
    assert by_id["outer"].sequence_index == 2


def test_circle_at_general_center_is_not_forced_into_arm_one() -> None:
    center_circle = CircleFeature(
        id="hub",
        x=200,
        y=200,
        radius=12,
        source="automatic",
    )
    classified = assign_arms([center_circle], (200, 200))[0]

    assert classified.arm_id is None
    assert classified.sequence_index is None
    assert classified.center_distance == 0


def test_spiral_model_follows_radius_dependent_arm_phase() -> None:
    center = (200.0, 200.0)
    phase = 18.0
    twist = -24.0
    circles: list[CircleFeature] = []
    for arm_index in range(6):
        for sequence, distance in enumerate((55.0, 105.0, 155.0), start=1):
            angle = phase + arm_index * 60.0 + twist * distance / 100.0
            circle = circle_at(
                f"spiral-{arm_index + 1}-{sequence}",
                angle,
                distance,
            )
            circles.append(circle)

    classified = assign_arms(
        circles,
        center,
        phase_degrees=phase,
        model="spiral",
        twist_degrees_per_100px=twist,
    )

    for circle in classified:
        expected_arm = int(circle.id.split("-")[1])
        assert circle.arm_id == expected_arm
        assert circle.side == "center"


def test_spiral_fit_recovers_phase_and_twist() -> None:
    center = (200.0, 200.0)
    phase = 22.0
    twist = -27.0
    circles: list[CircleFeature] = []
    for arm_index in range(6):
        for sequence, distance in enumerate((50.0, 90.0, 140.0, 190.0)):
            angle = phase + arm_index * 60.0 + twist * distance / 100.0
            circle = circle_at(
                f"fit-{arm_index}-{sequence}",
                angle,
                distance,
            )
            circle.radius = 15 - sequence
            circles.append(circle)

    suggestion = estimate_spiral_parameters(
        circles,
        center,
        phase_hint_degrees=phase,
    )

    assert suggestion.phase_degrees == pytest.approx(phase, abs=0.15)
    assert suggestion.twist_degrees_per_100px == pytest.approx(
        twist,
        abs=0.15,
    )
    assert suggestion.weighted_rms_error_degrees < 0.1
    assert suggestion.improvement_fraction > 0.9
