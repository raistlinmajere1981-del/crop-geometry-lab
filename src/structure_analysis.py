"""Human-reviewable structural analysis for six-arm circle annotations.

The functions in this module never delete circles and never mutate confirmed
classification fields.  They return measurements and ``suggested_*`` values
that a caller may present for explicit user approval.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Collection, Iterable, Mapping, Sequence

from .arm_classifier import (
    ArmModel,
    arm_phase_at_distance,
    polar_coordinates,
    signed_angle_difference,
)
from .feature_table import CircleFeature, write_csv

EXPECTED_ARM_TOTAL = 68
EXPECTED_MAIN_PER_ARM = 13
EXPECTED_SATELLITE_PER_ARM = 55
EXPECTED_CENTER_COUNT = 1
EXPECTED_TOTAL = 409
BASE_SATELLITE_SEQUENCE_CANDIDATE = [1, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 1]


def canonical_circle_type(value: str | None) -> str:
    """Map the schema-1.3 ``side`` alias to the structural name."""

    if value == "side":
        return "satellite"
    if value in {"center", "main", "satellite", "unclassified"}:
        return str(value)
    return "unclassified"


def _locked(
    circle: CircleFeature,
    field_name: str,
    manual_locks: Mapping[str, Collection[str]] | None,
) -> bool:
    if circle.locked:
        return True
    if field_name in circle.locked_fields:
        return True
    return (
        manual_locks is not None
        and field_name in manual_locks.get(circle.id, ())
    )


def structure_summary(circles: Iterable[CircleFeature]) -> dict[str, Any]:
    """Return actual counts and non-binding Milk Hill validation targets."""

    circle_list = list(circles)
    arms: dict[str, dict[str, int]] = {}
    for arm_id in range(1, 7):
        arm_circles = [
            circle for circle in circle_list if circle.arm_id == arm_id
        ]
        kinds = [canonical_circle_type(circle.circle_type) for circle in arm_circles]
        arms[str(arm_id)] = {
            "total": len(arm_circles),
            "main": kinds.count("main"),
            "satellite": kinds.count("satellite"),
            "unclassified": kinds.count("unclassified"),
            "target_difference": len(arm_circles) - EXPECTED_ARM_TOTAL,
        }
    center_count = sum(
        canonical_circle_type(circle.circle_type) == "center"
        for circle in circle_list
    )
    total = len(circle_list)
    validation = {
        "total_matches": total == EXPECTED_TOTAL,
        "center_matches": center_count == EXPECTED_CENTER_COUNT,
        "all_arms_have_68": all(
            arm["total"] == EXPECTED_ARM_TOTAL for arm in arms.values()
        ),
        "all_arms_have_13_main": all(
            arm["main"] == EXPECTED_MAIN_PER_ARM for arm in arms.values()
        ),
        "all_arms_have_55_satellite": all(
            arm["satellite"] == EXPECTED_SATELLITE_PER_ARM
            for arm in arms.values()
        ),
    }
    return {
        "total_circles": total,
        "center_count": center_count,
        "arms": arms,
        "expected": {
            "total_circles": EXPECTED_TOTAL,
            "center_count": EXPECTED_CENTER_COUNT,
            "arm_total": EXPECTED_ARM_TOTAL,
            "main_per_arm": EXPECTED_MAIN_PER_ARM,
            "satellite_per_arm": EXPECTED_SATELLITE_PER_ARM,
        },
        "total_difference": total - EXPECTED_TOTAL,
        "validation": validation,
    }


def center_candidate(
    circles: Iterable[CircleFeature],
    center: tuple[float, float],
    manual_locks: Mapping[str, Collection[str]] | None = None,
) -> dict[str, Any] | None:
    """Suggest the circle whose centre is nearest the chosen general centre."""

    ranked = sorted(
        (
            (
                math.hypot(circle.x - center[0], circle.y - center[1]),
                circle,
            )
            for circle in circles
        ),
        key=lambda item: (item[0], item[1].id),
    )
    if not ranked:
        return None
    distance, circle = ranked[0]
    second_distance = ranked[1][0] if len(ranked) > 1 else None
    proximity = 1.0 / (1.0 + distance / max(circle.radius, 1.0))
    separation = (
        1.0
        if second_distance is None
        else min(
            1.0,
            max(0.0, (second_distance - distance) / max(second_distance, 1.0)),
        )
    )
    confidence = min(1.0, 0.75 * proximity + 0.25 * separation)
    return {
        "id": circle.id,
        "distance_to_general_center": round(distance, 6),
        "second_candidate_distance": (
            None if second_distance is None else round(second_distance, 6)
        ),
        "radius": circle.radius,
        "current_circle_type": circle.circle_type,
        "suggested_circle_type": "center",
        "suggested_arm_id": None,
        "suggested_sequence_index": None,
        "suggested_parent_id": None,
        "suggested_side": "center",
        "confidence": round(confidence, 6),
        "locked": any(
            _locked(circle, field_name, manual_locks)
            and current_value != desired_value
            for field_name, current_value, desired_value in (
                (
                    "circle_type",
                    canonical_circle_type(circle.circle_type),
                    "center",
                ),
                ("arm_id", circle.arm_id, None),
                ("sequence_index", circle.sequence_index, None),
                ("parent_id", circle.parent_id, None),
                ("side", circle.side, "center"),
            )
        ),
    }


def _arm_distances(
    circle: CircleFeature,
    center: tuple[float, float],
    phase_degrees: float,
    model: ArmModel,
    twist_degrees_per_100px: float,
    arm_id: int,
) -> tuple[float, float]:
    radius, angle = polar_coordinates((circle.x, circle.y), center)
    phase = arm_phase_at_distance(
        phase_degrees,
        radius,
        model=model,
        twist_degrees_per_100px=twist_degrees_per_100px,
    )
    nominal = phase + (arm_id - 1) * 60.0
    angular_distance = abs(signed_angle_difference(angle, nominal))
    chord_distance = (
        2.0
        * radius
        * math.sin(math.radians(min(180.0, angular_distance)) / 2.0)
    )
    return chord_distance, angular_distance


def boundary_candidates(
    circles: Iterable[CircleFeature],
    center: tuple[float, float],
    phase_degrees: float,
    *,
    model: ArmModel = "radial",
    twist_degrees_per_100px: float = 0.0,
    manual_locks: Mapping[str, Collection[str]] | None = None,
    boundary_band_degrees: float = 9.0,
) -> list[dict[str, Any]]:
    """Rank K1/K6 records near their shared geometric boundary.

    A positive difference score means the other arm is geometrically closer.
    Records near the equidistant boundary are also retained for inspection,
    even when their current arm remains the better fit.
    """

    candidates: list[dict[str, Any]] = []
    for circle in circles:
        if circle.arm_id not in {1, 6}:
            continue
        d1, a1 = _arm_distances(
            circle,
            center,
            phase_degrees,
            model,
            twist_degrees_per_100px,
            1,
        )
        d6, a6 = _arm_distances(
            circle,
            center,
            phase_degrees,
            model,
            twist_degrees_per_100px,
            6,
        )
        current_distance = d1 if circle.arm_id == 1 else d6
        proposed_distance = d6 if circle.arm_id == 1 else d1
        current_angle_distance = a1 if circle.arm_id == 1 else a6
        proposed_angle_distance = a6 if circle.arm_id == 1 else a1
        proposed_arm = 6 if circle.arm_id == 1 else 1
        score = current_distance - proposed_distance
        angular_balance = abs(a1 - a6)
        near_boundary = angular_balance <= boundary_band_degrees * 2.0
        if score <= 0 and not near_boundary:
            continue
        confidence = 1.0 / (
            1.0
            + math.exp(
                -score / max(1.0, circle.radius * 0.75)
            )
        )
        candidates.append(
            {
                "id": circle.id,
                "current_arm_id": circle.arm_id,
                "suggested_arm_id": proposed_arm,
                "current_arm_distance": round(current_distance, 6),
                "suggested_arm_distance": round(proposed_distance, 6),
                "current_angular_distance": round(
                    current_angle_distance, 6
                ),
                "suggested_angular_distance": round(
                    proposed_angle_distance, 6
                ),
                "difference_score": round(score, 6),
                "near_boundary": near_boundary,
                "manual_locked": _locked(
                    circle, "arm_id", manual_locks
                ),
                "confidence": round(confidence, 6),
                "most_likely_four": False,
            }
        )
    candidates.sort(
        key=lambda item: (
            -float(item["difference_score"]),
            abs(
                float(item["current_angular_distance"])
                - float(item["suggested_angular_distance"])
            ),
            str(item["id"]),
        )
    )
    eligible = [
        item
        for item in candidates
        if item["difference_score"] > 0
    ][:4]
    eligible_ids = {item["id"] for item in eligible}
    for item in candidates:
        item["most_likely_four"] = item["id"] in eligible_ids
    return candidates


def _circle_intersection_area(
    first: CircleFeature, second: CircleFeature
) -> float:
    distance = math.hypot(first.x - second.x, first.y - second.y)
    r1 = first.radius
    r2 = second.radius
    if distance >= r1 + r2:
        return 0.0
    if distance <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    first_angle = math.acos(
        max(-1.0, min(1.0, (distance * distance + r1 * r1 - r2 * r2) / (2 * distance * r1)))
    )
    second_angle = math.acos(
        max(-1.0, min(1.0, (distance * distance + r2 * r2 - r1 * r1) / (2 * distance * r2)))
    )
    triangle = 0.5 * math.sqrt(
        max(
            0.0,
            (-distance + r1 + r2)
            * (distance + r1 - r2)
            * (distance - r1 + r2)
            * (distance + r1 + r2),
        )
    )
    return r1 * r1 * first_angle + r2 * r2 * second_angle - triangle


def _record_priority(circle: CircleFeature) -> tuple[int, int, float]:
    source_priority = {
        "manual_override": 3,
        "manual": 2,
        "automatic": 1,
    }.get(circle.source, 0)
    return (
        1 if circle.locked or circle.locked_fields else 0,
        source_priority,
        circle.confidence,
    )


def duplicate_candidates(
    circles: Iterable[CircleFeature],
    *,
    center_ratio: float = 0.45,
    radius_difference_ratio: float = 0.25,
    overlap_ratio_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    """Find likely duplicate records without merging or deleting anything."""

    circle_list = list(circles)
    result: list[dict[str, Any]] = []
    for first_index, first in enumerate(circle_list):
        for second in circle_list[first_index + 1 :]:
            center_distance = math.hypot(
                first.x - second.x, first.y - second.y
            )
            smaller_radius = min(first.radius, second.radius)
            if center_distance >= center_ratio * smaller_radius:
                continue
            radius_difference = abs(first.radius - second.radius)
            radius_ratio = radius_difference / max(first.radius, second.radius)
            if radius_ratio > radius_difference_ratio:
                continue
            intersection = _circle_intersection_area(first, second)
            overlap_ratio = intersection / (
                math.pi * smaller_radius * smaller_radius
            )
            if overlap_ratio < overlap_ratio_threshold:
                continue
            preferred, other = (
                (first, second)
                if _record_priority(first) >= _record_priority(second)
                else (second, first)
            )
            result.append(
                {
                    "id_a": first.id,
                    "id_b": second.id,
                    "center_distance": round(center_distance, 6),
                    "radius_difference": round(radius_difference, 6),
                    "radius_difference_ratio": round(radius_ratio, 6),
                    "overlap_ratio": round(overlap_ratio, 6),
                    "source_a": first.source,
                    "source_b": second.source,
                    "locked_a": bool(first.locked or first.locked_fields),
                    "locked_b": bool(second.locked or second.locked_fields),
                    "preferred_id": preferred.id,
                    "merge_candidate_id": other.id,
                }
            )
    result.sort(
        key=lambda item: (
            -float(item["overlap_ratio"]),
            float(item["center_distance"]),
            str(item["id_a"]),
            str(item["id_b"]),
        )
    )
    return result


def _main_chains(
    circles: Sequence[CircleFeature],
    structure_state: Mapping[str, Any] | None,
) -> dict[int, list[CircleFeature]]:
    by_id = {circle.id: circle for circle in circles}
    raw_chains: Any = (
        structure_state.get("main_chains", {})
        if isinstance(structure_state, Mapping)
        else {}
    )
    result: dict[int, list[CircleFeature]] = {arm: [] for arm in range(1, 7)}
    if isinstance(raw_chains, Mapping):
        for arm in range(1, 7):
            raw_ids = raw_chains.get(str(arm), raw_chains.get(arm, []))
            if not isinstance(raw_ids, Sequence) or isinstance(
                raw_ids, (str, bytes)
            ):
                continue
            seen: set[str] = set()
            for raw_id in raw_ids:
                circle_id = str(raw_id)
                circle = by_id.get(circle_id)
                if circle is None or circle_id in seen:
                    continue
                seen.add(circle_id)
                result[arm].append(circle)
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _project_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return the nearest point, unit tangent, and distance to a segment."""

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        projection = start
        tangent = (1.0, 0.0)
    else:
        fraction = _clamp(
            (
                (point[0] - start[0]) * dx
                + (point[1] - start[1]) * dy
            )
            / length_squared,
            0.0,
            1.0,
        )
        projection = (
            start[0] + fraction * dx,
            start[1] + fraction * dy,
        )
        length = math.sqrt(length_squared)
        tangent = (dx / length, dy / length)
    distance = math.hypot(
        point[0] - projection[0],
        point[1] - projection[1],
    )
    return projection, tangent, distance


def _project_to_chain(
    circle: CircleFeature,
    chain: Sequence[CircleFeature],
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """Project a circle centre onto an approved main-chain polyline."""

    if len(chain) < 2:
        return None
    point = (circle.x, circle.y)
    ranked = [
        _project_to_segment(
            point,
            (first.x, first.y),
            (second.x, second.y),
        )
        for first, second in zip(chain, chain[1:])
    ]
    return min(ranked, key=lambda item: item[2])


def _spiral_point_and_tangent(
    radius: float,
    center: tuple[float, float],
    phase_degrees: float,
    arm_id: int,
    *,
    model: ArmModel,
    twist_degrees_per_100px: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a six-arm model point and its screen-coordinate unit tangent."""

    angle_degrees = arm_phase_at_distance(
        phase_degrees,
        radius,
        model=model,
        twist_degrees_per_100px=twist_degrees_per_100px,
    ) + (arm_id - 1) * 60.0
    angle = math.radians(angle_degrees)
    point = (
        center[0] + radius * math.cos(angle),
        center[1] - radius * math.sin(angle),
    )
    angular_rate = (
        0.0
        if model == "radial"
        else math.radians(twist_degrees_per_100px / 100.0)
    )
    tangent = (
        math.cos(angle) - radius * math.sin(angle) * angular_rate,
        -math.sin(angle) - radius * math.cos(angle) * angular_rate,
    )
    magnitude = math.hypot(*tangent)
    if magnitude <= 1e-12:
        return point, (1.0, 0.0)
    return point, (tangent[0] / magnitude, tangent[1] / magnitude)


def _project_to_spiral(
    circle: CircleFeature,
    center: tuple[float, float],
    phase_degrees: float,
    arm_id: int,
    *,
    model: ArmModel,
    twist_degrees_per_100px: float,
    max_radius: float,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Numerically project a circle centre onto its assigned arm model."""

    point = (circle.x, circle.y)

    def distance_squared(radius: float) -> float:
        curve_point, _ = _spiral_point_and_tangent(
            radius,
            center,
            phase_degrees,
            arm_id,
            model=model,
            twist_degrees_per_100px=twist_degrees_per_100px,
        )
        return (
            (point[0] - curve_point[0]) ** 2
            + (point[1] - curve_point[1]) ** 2
        )

    # A full-radius scan avoids choosing a wrong turn when a fitted spiral curls
    # through more than one revolution.  Golden-section refinement then gives a
    # stable sub-pixel projection without mutating annotation geometry.
    sample_count = 160
    step = max_radius / sample_count
    samples = [index * step for index in range(sample_count + 1)]
    best_index = min(
        range(len(samples)),
        key=lambda index: distance_squared(samples[index]),
    )
    lower = samples[max(0, best_index - 1)]
    upper = samples[min(sample_count, best_index + 1)]
    golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - golden_ratio * (upper - lower)
    right = lower + golden_ratio * (upper - lower)
    left_value = distance_squared(left)
    right_value = distance_squared(right)
    for _ in range(32):
        if left_value <= right_value:
            upper = right
            right = left
            right_value = left_value
            left = upper - golden_ratio * (upper - lower)
            left_value = distance_squared(left)
        else:
            lower = left
            left = right
            left_value = right_value
            right = lower + golden_ratio * (upper - lower)
            right_value = distance_squared(right)
    projected_radius = (lower + upper) / 2.0
    projection, tangent = _spiral_point_and_tangent(
        projected_radius,
        center,
        phase_degrees,
        arm_id,
        model=model,
        twist_degrees_per_100px=twist_degrees_per_100px,
    )
    distance = math.hypot(
        point[0] - projection[0],
        point[1] - projection[1],
    )
    return projection, tangent, distance


def local_side_candidates(
    circles: Iterable[CircleFeature],
    center: tuple[float, float],
    phase_degrees: float,
    *,
    model: ArmModel = "radial",
    twist_degrees_per_100px: float = 0.0,
    manual_locks: Mapping[str, Collection[str]] | None = None,
    structure_state: Mapping[str, Any] | None = None,
    center_candidate_id: str | None = None,
) -> list[dict[str, Any]]:
    """Suggest local ``inner/outer/centerline`` sides for every circle.

    Approved main chains are preferred as the arm spine.  Until a chain exists,
    the current radial/spiral model is used.  This function is deliberately
    non-mutating; callers must ask for explicit user approval before storing a
    suggestion in ``side``.
    """

    circle_list = list(circles)
    chains = _main_chains(circle_list, structure_state)
    maximum_circle_distance = max(
        (
            math.hypot(circle.x - center[0], circle.y - center[1])
            + 3.0 * max(circle.radius, 1.0)
            for circle in circle_list
        ),
        default=1.0,
    )
    max_radius = max(10.0, maximum_circle_distance)
    radius_by_id = {
        circle.id: math.hypot(
            circle.x - center[0],
            circle.y - center[1],
        )
        for circle in circle_list
    }
    result: list[dict[str, Any]] = []

    for circle in circle_list:
        current_side = circle.side or "unknown"
        is_center = (
            circle.id == center_candidate_id
            or canonical_circle_type(circle.circle_type) == "center"
        )
        if is_center:
            suggested_side = "center"
            method = "general_center"
            centerline_distance = math.hypot(
                circle.x - center[0], circle.y - center[1]
            )
            signed_radial_offset = 0.0
            tangent_angle = None
            confidence = 1.0 / (
                1.0 + centerline_distance / max(circle.radius, 1.0)
            )
        elif circle.arm_id not in range(1, 7):
            suggested_side = "unknown"
            method = "missing_arm"
            centerline_distance = None
            signed_radial_offset = None
            tangent_angle = None
            confidence = 0.0
        else:
            chain_projection = _project_to_chain(
                circle,
                chains[int(circle.arm_id)],
            )
            if chain_projection is not None:
                projection, tangent, distance = chain_projection
                method = "main_chain"
            else:
                projection, tangent, distance = _project_to_spiral(
                    circle,
                    center,
                    phase_degrees,
                    int(circle.arm_id),
                    model=model,
                    twist_degrees_per_100px=twist_degrees_per_100px,
                    max_radius=max_radius,
                )
                method = "spiral_model" if model == "spiral" else "radial_model"

            projection_from_center = (
                projection[0] - center[0],
                projection[1] - center[1],
            )
            projection_radius = math.hypot(*projection_from_center)
            displacement = (
                circle.x - projection[0],
                circle.y - projection[1],
            )
            if projection_radius <= 1e-9:
                radial_unit = (0.0, 0.0)
            else:
                radial_unit = (
                    projection_from_center[0] / projection_radius,
                    projection_from_center[1] / projection_radius,
                )
            # The local tangent divides the image into a centre-facing and an
            # outward-facing half-plane.  This radial projection chooses that
            # side even when a large angular displacement leaves the two
            # points at nearly the same absolute centre radius.
            radial_offset = (
                displacement[0] * radial_unit[0]
                + displacement[1] * radial_unit[1]
            )
            centerline_tolerance = max(1.5, circle.radius * 0.22)
            radial_alignment = (
                0.0
                if distance <= 1e-9
                else abs(radial_offset) / distance
            )
            if distance <= centerline_tolerance:
                suggested_side = "centerline"
                confidence = 0.55 + 0.45 * (
                    1.0 - distance / centerline_tolerance
                )
            elif (
                method == "radial_model"
                or radial_alignment < 0.15
                or abs(radial_offset) < max(0.75, distance * 0.12)
            ):
                # A straight radial arm has no geometrically meaningful
                # inward/outward normal.  Keep these cases explicit for review.
                suggested_side = "unknown"
                confidence = min(0.49, 0.2 + radial_alignment)
            else:
                suggested_side = "inner" if radial_offset < 0 else "outer"
                separation = min(
                    1.0,
                    distance / max(centerline_tolerance * 2.0, 1e-9),
                )
                confidence = (
                    0.35 + 0.35 * min(1.0, radial_alignment) + 0.30 * separation
                )
            centerline_distance = distance
            signed_radial_offset = radial_offset
            tangent_angle = (
                math.degrees(math.atan2(-tangent[1], tangent[0])) % 360.0
            )

        side_locked = _locked(circle, "side", manual_locks)
        result.append(
            {
                "id": circle.id,
                "arm_id": circle.arm_id,
                "current_side": current_side,
                "legacy_side": circle.legacy_side,
                "suggested_side": suggested_side,
                "centerline_distance": (
                    None
                    if centerline_distance is None
                    else round(centerline_distance, 6)
                ),
                "signed_radial_offset": (
                    None
                    if signed_radial_offset is None
                    else round(signed_radial_offset, 6)
                ),
                "local_tangent_angle": (
                    None if tangent_angle is None else round(tangent_angle, 6)
                ),
                "method": method,
                "confidence": round(_clamp(confidence, 0.0, 1.0), 6),
                "side_locked": side_locked,
                "manual_locked": (
                    side_locked and current_side != suggested_side
                ),
            }
        )

    result.sort(
        key=lambda item: (
            0 if item["arm_id"] is None else int(item["arm_id"]),
            radius_by_id[str(item["id"])],
            str(item["id"]),
        )
    )
    return result


def _unit_tangent(
    chain: Sequence[CircleFeature],
    index: int,
    center: tuple[float, float],
) -> tuple[float, float]:
    if len(chain) >= 2:
        if index == 0:
            start, end = chain[0], chain[1]
        elif index == len(chain) - 1:
            start, end = chain[-2], chain[-1]
        else:
            start, end = chain[index - 1], chain[index + 1]
        dx = end.x - start.x
        dy = end.y - start.y
    else:
        main = chain[index]
        dx = main.x - center[0]
        dy = main.y - center[1]
    magnitude = math.hypot(dx, dy)
    if magnitude <= 1e-9:
        return 1.0, 0.0
    return dx / magnitude, dy / magnitude


def _relationship_geometry(
    satellite: CircleFeature,
    parent: CircleFeature,
    tangent: tuple[float, float],
    center: tuple[float, float],
) -> tuple[float, float, str]:
    dx = satellite.x - parent.x
    dy = satellite.y - parent.y
    distance = math.hypot(dx, dy)
    normalized = distance / max(parent.radius, 1e-9)
    # Convert the screen-coordinate vectors to y-up before reporting an angle.
    tangent_angle = math.atan2(-tangent[1], tangent[0])
    relative_angle = math.atan2(-dy, dx)
    local_angle = math.degrees(
        math.atan2(
            math.sin(relative_angle - tangent_angle),
            math.cos(relative_angle - tangent_angle),
        )
    )
    perpendicular_distance = abs(tangent[0] * dy - tangent[1] * dx)
    centerline_tolerance = max(1.5, parent.radius * 0.18)
    if perpendicular_distance <= centerline_tolerance:
        side = "centerline"
    else:
        parent_radius = math.hypot(
            parent.x - center[0], parent.y - center[1]
        )
        satellite_radius = math.hypot(
            satellite.x - center[0], satellite.y - center[1]
        )
        side = "inner" if satellite_radius < parent_radius else "outer"
    return local_angle, normalized, side


def satellite_candidates(
    circles: Iterable[CircleFeature],
    center: tuple[float, float],
    *,
    structure_state: Mapping[str, Any] | None = None,
    center_candidate_id: str | None = None,
    manual_locks: Mapping[str, Collection[str]] | None = None,
) -> list[dict[str, Any]]:
    """Suggest nearest and second-nearest confirmed main-circle parents."""

    circle_list = list(circles)
    chains = _main_chains(circle_list, structure_state)
    main_ids = {
        circle.id for chain in chains.values() for circle in chain
    }
    result: list[dict[str, Any]] = []
    for arm_id, chain in chains.items():
        if not chain:
            continue
        tangents = [
            _unit_tangent(chain, index, center)
            for index in range(len(chain))
        ]
        for circle in circle_list:
            if (
                circle.id in main_ids
                or circle.id == center_candidate_id
                or canonical_circle_type(circle.circle_type) == "center"
                or circle.arm_id != arm_id
            ):
                continue
            ranked = sorted(
                (
                    (
                        math.hypot(circle.x - main.x, circle.y - main.y),
                        index,
                        main,
                    )
                    for index, main in enumerate(chain)
                ),
                key=lambda item: (item[0], item[2].id),
            )
            nearest_distance, nearest_index, nearest = ranked[0]
            second_distance: float | None = None
            second: CircleFeature | None = None
            if len(ranked) > 1:
                second_distance, _, second = ranked[1]
            ratio = (
                None
                if second_distance is None or second_distance <= 1e-9
                else nearest_distance / second_distance
            )
            local_angle, normalized, side = _relationship_geometry(
                circle,
                nearest,
                tangents[nearest_index],
                center,
            )
            distinctiveness = 0.5 if ratio is None else max(0.0, 1.0 - ratio)
            proximity = 1.0 / (1.0 + max(0.0, normalized - 1.0))
            confidence = min(
                1.0, max(0.0, 0.65 * distinctiveness + 0.35 * proximity)
            )
            result.append(
                {
                    "id": circle.id,
                    "arm_id": arm_id,
                    "current_circle_type": circle.circle_type,
                    "nearest_main_id": nearest.id,
                    "nearest_main_sequence": nearest_index + 1,
                    "nearest_distance": round(nearest_distance, 6),
                    "second_nearest_main_id": (
                        None if second is None else second.id
                    ),
                    "second_nearest_distance": (
                        None
                        if second_distance is None
                        else round(second_distance, 6)
                    ),
                    "distance_ratio": (
                        None if ratio is None else round(ratio, 6)
                    ),
                    "normalized_parent_distance": round(normalized, 6),
                    "local_angle": round(local_angle, 6),
                    "suggested_circle_type": "satellite",
                    "suggested_parent_id": nearest.id,
                    "suggested_side": side,
                    "suggested_arm_id": arm_id,
                    "confidence": round(confidence, 6),
                    "manual_locked": any(
                        _locked(circle, field_name, manual_locks)
                        and current_value != desired_value
                        for field_name, current_value, desired_value in (
                            (
                                "circle_type",
                                canonical_circle_type(circle.circle_type),
                                "satellite",
                            ),
                            ("parent_id", circle.parent_id, nearest.id),
                            ("side", circle.side, side),
                            ("arm_id", circle.arm_id, arm_id),
                            ("sequence_index", circle.sequence_index, None),
                        )
                    ),
                }
            )
    result.sort(
        key=lambda item: (
            int(item["arm_id"]),
            -float(item["confidence"]),
            str(item["id"]),
        )
    )
    return result


def satellite_status_tables(
    circles: Iterable[CircleFeature],
    center: tuple[float, float],
    *,
    structure_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the 13-row relationship audit table for every arm."""

    circle_list = list(circles)
    chains = _main_chains(circle_list, structure_state)
    by_parent: dict[str, list[CircleFeature]] = {}
    for circle in circle_list:
        if (
            canonical_circle_type(circle.circle_type) == "satellite"
            and circle.parent_id
        ):
            by_parent.setdefault(circle.parent_id, []).append(circle)
    arms: dict[str, list[dict[str, Any]]] = {}
    for arm_id in range(1, 7):
        chain = chains[arm_id]
        rows: list[dict[str, Any]] = []
        for sequence in range(1, EXPECTED_MAIN_PER_ARM + 1):
            main = chain[sequence - 1] if sequence <= len(chain) else None
            satellites = by_parent.get(main.id, []) if main is not None else []
            inner = [item for item in satellites if item.side == "inner"]
            outer = [item for item in satellites if item.side == "outer"]
            inner_angles = [
                float(item.local_angle)
                for item in inner
                if item.local_angle is not None
            ]
            outer_angles = [
                float(item.local_angle)
                for item in outer
                if item.local_angle is not None
            ]
            rows.append(
                {
                    "sequence_index": sequence,
                    "main_id": None if main is None else main.id,
                    "main_radius": None if main is None else main.radius,
                    "inner_count": len(inner),
                    "outer_count": len(outer),
                    "total_satellite": len(satellites),
                    "inner_angle": (
                        None if not inner_angles else round(fmean(inner_angles), 6)
                    ),
                    "outer_angle": (
                        None if not outer_angles else round(fmean(outer_angles), 6)
                    ),
                    "highlight": sequence in {6, 7},
                }
            )
        arms[str(arm_id)] = rows
    return {
        "arms": arms,
        "hypothesis_warning": {
            "is_enforced": False,
            "base_satellite_sequence_candidate": (
                BASE_SATELLITE_SEQUENCE_CANDIDATE
            ),
            "position_6_note": (
                "6. konumda bir taraftaki üçlü grubun eksik olabileceği "
                "yalnızca inceleme hipotezidir."
            ),
        },
    }


def structural_report(
    circles: Iterable[CircleFeature],
    center: tuple[float, float],
    phase_degrees: float,
    *,
    model: ArmModel = "radial",
    twist_degrees_per_100px: float = 0.0,
    manual_locks: Mapping[str, Collection[str]] | None = None,
    structure_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a complete, non-mutating structural review report."""

    circle_list = list(circles)
    center_item = center_candidate(circle_list, center, manual_locks)
    boundary = boundary_candidates(
        circle_list,
        center,
        phase_degrees,
        model=model,
        twist_degrees_per_100px=twist_degrees_per_100px,
        manual_locks=manual_locks,
    )
    satellites = satellite_candidates(
        circle_list,
        center,
        structure_state=structure_state,
        center_candidate_id=(
            None if center_item is None else str(center_item["id"])
        ),
        manual_locks=manual_locks,
    )
    local_sides = local_side_candidates(
        circle_list,
        center,
        phase_degrees,
        model=model,
        twist_degrees_per_100px=twist_degrees_per_100px,
        manual_locks=manual_locks,
        structure_state=structure_state,
        center_candidate_id=(
            None if center_item is None else str(center_item["id"])
        ),
    )
    suggestions: dict[str, dict[str, Any]] = {}
    for candidate in local_sides:
        suggestions.setdefault(str(candidate["id"]), {}).update(
            {
                "suggested_side": candidate["suggested_side"],
                "suggestion_confidence": candidate["confidence"],
            }
        )
    if center_item is not None:
        suggestions[str(center_item["id"])] = {
            "suggested_circle_type": "center",
            "suggested_arm_id": None,
            "suggested_sequence_index": None,
            "suggested_parent_id": None,
            "suggested_side": "center",
            "suggestion_confidence": center_item["confidence"],
        }
    for candidate in boundary:
        if float(candidate["difference_score"]) <= 0:
            continue
        suggestions.setdefault(str(candidate["id"]), {}).update(
            {
                "suggested_arm_id": candidate["suggested_arm_id"],
                "suggestion_confidence": candidate["confidence"],
            }
        )
    for candidate in satellites:
        suggestions.setdefault(str(candidate["id"]), {}).update(
            {
                "suggested_circle_type": "satellite",
                "suggested_arm_id": candidate["suggested_arm_id"],
                "suggested_parent_id": candidate["suggested_parent_id"],
                "suggested_side": candidate["suggested_side"],
                "suggestion_confidence": candidate["confidence"],
                "local_angle": candidate["local_angle"],
                "normalized_parent_distance": candidate[
                    "normalized_parent_distance"
                ],
            }
        )
    return {
        "summary": structure_summary(circle_list),
        "center_candidate": center_item,
        "boundary_candidates": boundary,
        "local_side_candidates": local_sides,
        "duplicate_candidates": duplicate_candidates(circle_list),
        "satellite_candidates": satellites,
        "satellite_tables": satellite_status_tables(
            circle_list,
            center,
            structure_state=structure_state,
        ),
        "suggestions": suggestions,
        "notice": (
            "Bütün sonuçlar öneridir; gerçek sınıflandırma alanları kullanıcı "
            "onayı olmadan değiştirilmez."
        ),
    }


def _write_rows(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field_name: row.get(field_name)
                    for field_name in fieldnames
                }
            )
    return path


@dataclass(frozen=True, slots=True)
class StructureExport:
    files: dict[str, Path]
    summary: dict[str, Any]


def write_structure_exports(
    circles: Sequence[CircleFeature],
    destination_dir: str | Path,
    *,
    center: tuple[float, float],
    phase_degrees: float,
    model: ArmModel = "radial",
    twist_degrees_per_100px: float = 0.0,
    manual_locks: Mapping[str, Collection[str]] | None = None,
    structure_state: Mapping[str, Any] | None = None,
) -> StructureExport:
    """Write the five requested CSV files and ``structure_summary.json``."""

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = structural_report(
        circles,
        center,
        phase_degrees,
        model=model,
        twist_degrees_per_100px=twist_degrees_per_100px,
        manual_locks=manual_locks,
        structure_state=structure_state,
    )
    summary_path = destination / "structure_summary.json"
    summary_path.write_text(
        json.dumps(report["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    classified_path = write_csv(
        circles, destination / "circles_classified.csv"
    )
    arm_rows = [
        {"arm_id": int(arm_id), **counts}
        for arm_id, counts in report["summary"]["arms"].items()
    ]
    arm_summary_path = _write_rows(
        destination / "arm_summary.csv",
        (
            "arm_id",
            "total",
            "main",
            "satellite",
            "unclassified",
            "target_difference",
        ),
        arm_rows,
    )
    chains = _main_chains(circles, structure_state)
    main_rows = [
        {
            "arm_id": arm_id,
            "sequence_index": sequence,
            "id": circle.id,
            "x": circle.x,
            "y": circle.y,
            "radius": circle.radius,
            "classification_source": circle.classification_source,
            "classification_confidence": circle.classification_confidence,
        }
        for arm_id, chain in chains.items()
        for sequence, circle in enumerate(chain, start=1)
    ]
    main_sequence_path = _write_rows(
        destination / "main_sequence.csv",
        (
            "arm_id",
            "sequence_index",
            "id",
            "x",
            "y",
            "radius",
            "classification_source",
            "classification_confidence",
        ),
        main_rows,
    )
    satellite_rows = [
        {
            "id": circle.id,
            "arm_id": circle.arm_id,
            "parent_id": circle.parent_id,
            "side": circle.side,
            "legacy_side": circle.legacy_side,
            "local_angle": circle.local_angle,
            "normalized_parent_distance": (
                circle.normalized_parent_distance
            ),
            "classification_source": circle.classification_source,
            "classification_confidence": circle.classification_confidence,
        }
        for circle in circles
        if canonical_circle_type(circle.circle_type) == "satellite"
    ]
    satellite_path = _write_rows(
        destination / "satellite_relationships.csv",
        (
            "id",
            "arm_id",
            "parent_id",
            "side",
            "legacy_side",
            "local_angle",
            "normalized_parent_distance",
            "classification_source",
            "classification_confidence",
        ),
        satellite_rows,
    )
    boundary_path = _write_rows(
        destination / "boundary_candidates.csv",
        (
            "id",
            "current_arm_id",
            "suggested_arm_id",
            "current_arm_distance",
            "suggested_arm_distance",
            "difference_score",
            "manual_locked",
            "most_likely_four",
            "confidence",
        ),
        report["boundary_candidates"],
    )
    return StructureExport(
        files={
            "structure_summary": summary_path,
            "circles_classified": classified_path,
            "arm_summary": arm_summary_path,
            "main_sequence": main_sequence_path,
            "satellite_relationships": satellite_path,
            "boundary_candidates": boundary_path,
        },
        summary=report["summary"],
    )
