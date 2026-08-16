"""Polar coordinates and deterministic six-arm classification."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from statistics import median
from typing import Collection, Iterable, Literal, Mapping

from .feature_table import CircleFeature

Point = tuple[float, float]
ArmModel = Literal["radial", "spiral"]


@dataclass(frozen=True, slots=True)
class ArmFitSuggestion:
    """Human-reviewable Archimedean six-arm fit."""

    phase_degrees: float
    twist_degrees_per_100px: float
    weighted_rms_error_degrees: float
    radial_rms_error_degrees: float
    sample_count: int

    @property
    def improvement_fraction(self) -> float:
        if self.radial_rms_error_degrees <= 1e-9:
            return 0.0
        return max(
            0.0,
            1.0
            - self.weighted_rms_error_degrees
            / self.radial_rms_error_degrees,
        )

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "model": "spiral",
            "phase_degrees": self.phase_degrees,
            "twist_degrees_per_100px": self.twist_degrees_per_100px,
            "weighted_rms_error_degrees": self.weighted_rms_error_degrees,
            "radial_rms_error_degrees": self.radial_rms_error_degrees,
            "improvement_fraction": self.improvement_fraction,
            "sample_count": self.sample_count,
        }


def normalize_angle(angle_degrees: float) -> float:
    return angle_degrees % 360.0


def signed_angle_difference(angle: float, reference: float) -> float:
    return (angle - reference + 180.0) % 360.0 - 180.0


def _periodic_residual(value: float, period: float = 60.0) -> float:
    half_period = period / 2.0
    return (value + half_period) % period - half_period


def polar_coordinates(
    point: Point, center: Point
) -> tuple[float, float]:
    """Distance and conventional angle: 0° east, increasing counter-clockwise."""

    dx = float(point[0]) - float(center[0])
    dy = float(point[1]) - float(center[1])
    distance = math.hypot(dx, dy)
    angle = normalize_angle(math.degrees(math.atan2(-dy, dx)))
    return distance, angle


def _nearest_arm(angle: float, phase_degrees: float) -> tuple[int, float]:
    relative = normalize_angle(angle - phase_degrees)
    zero_based = int(math.floor(relative / 60.0 + 0.5)) % 6
    nominal = normalize_angle(phase_degrees + zero_based * 60.0)
    return zero_based + 1, signed_angle_difference(angle, nominal)


def arm_phase_at_distance(
    phase_degrees: float,
    distance: float,
    *,
    model: ArmModel = "radial",
    twist_degrees_per_100px: float = 0.0,
) -> float:
    """Return the arm-one tangent angle at a given radius."""

    if model == "radial":
        return normalize_angle(phase_degrees)
    if model != "spiral":
        raise ValueError("arm model must be 'radial' or 'spiral'")
    return normalize_angle(
        phase_degrees + twist_degrees_per_100px * float(distance) / 100.0
    )


def _fit_phase_for_twist(
    samples: list[tuple[float, float, float]],
    twist_degrees_per_100px: float,
) -> tuple[float, float, float]:
    adjusted = [
        (
            normalize_angle(
                angle
                - twist_degrees_per_100px * distance / 100.0
            )
            % 60.0,
            weight,
        )
        for distance, angle, weight in samples
    ]
    sine = sum(
        weight * math.sin(math.radians(phase * 6.0))
        for phase, weight in adjusted
    )
    cosine = sum(
        weight * math.cos(math.radians(phase * 6.0))
        for phase, weight in adjusted
    )
    if math.hypot(sine, cosine) <= 1e-12:
        phase = median(item[0] for item in adjusted) % 60.0
    else:
        phase = (
            math.degrees(math.atan2(sine, cosine)) / 6.0
        ) % 60.0
    residuals = [
        (abs(_periodic_residual(item_phase - phase)), weight)
        for item_phase, weight in adjusted
    ]
    total_weight = sum(weight for _, weight in residuals)
    rms = math.sqrt(
        sum(residual * residual * weight for residual, weight in residuals)
        / max(total_weight, 1e-12)
    )
    # Robust objective: side circles may legitimately sit away from an arm spine.
    huber_limit = 12.0
    robust_loss = sum(
        weight
        * (
            residual * residual
            if residual <= huber_limit
            else 2.0 * huber_limit * residual - huber_limit * huber_limit
        )
        for residual, weight in residuals
    ) / max(total_weight, 1e-12)
    return phase, rms, robust_loss


def estimate_spiral_parameters(
    circles: Iterable[CircleFeature],
    center: Point,
    *,
    phase_hint_degrees: float = 0.0,
    max_abs_twist_degrees_per_100px: float = 90.0,
) -> ArmFitSuggestion:
    """Fit a sixfold Archimedean arm model to the strongest circle centers."""

    candidates: list[tuple[float, float, float, float]] = []
    for circle in circles:
        distance, angle = polar_coordinates((circle.x, circle.y), center)
        if distance <= max(8.0, circle.radius * 0.75):
            continue
        weight = (
            max(1.0, circle.radius) ** 1.5
            * max(0.2, circle.confidence)
        )
        candidates.append((distance, angle, weight, circle.radius))
    if len(candidates) < 6:
        raise ValueError("Spiral önerisi için merkez dışında en az 6 daire gerekli")

    candidates.sort(key=lambda item: (item[3], item[2]), reverse=True)
    sample_limit = min(
        len(candidates),
        max(24, round(math.sqrt(len(candidates)) * 3.0)),
    )
    samples = [
        (distance, angle, weight)
        for distance, angle, weight, _ in candidates[:sample_limit]
    ]
    twist_limit = abs(float(max_abs_twist_degrees_per_100px))
    coarse_twists = [
        -twist_limit + index
        for index in range(int(round(twist_limit * 2.0)) + 1)
    ]
    best_twist = 0.0
    best_phase = 0.0
    best_rms = float("inf")
    best_loss = float("inf")
    for twist in coarse_twists:
        phase, rms, loss = _fit_phase_for_twist(samples, twist)
        if loss < best_loss:
            best_twist, best_phase, best_rms, best_loss = (
                twist,
                phase,
                rms,
                loss,
            )
    refinement_start = max(-twist_limit, best_twist - 1.0)
    refinement_end = min(twist_limit, best_twist + 1.0)
    refinement_steps = int(
        round((refinement_end - refinement_start) / 0.05)
    )
    for index in range(refinement_steps + 1):
        twist = refinement_start + index * 0.05
        phase, rms, loss = _fit_phase_for_twist(samples, twist)
        if loss < best_loss:
            best_twist, best_phase, best_rms, best_loss = (
                twist,
                phase,
                rms,
                loss,
            )
    _, radial_rms, _ = _fit_phase_for_twist(samples, 0.0)
    phase_near_hint = best_phase + 60.0 * round(
        (float(phase_hint_degrees) - best_phase) / 60.0
    )
    return ArmFitSuggestion(
        phase_degrees=round(phase_near_hint, 3),
        twist_degrees_per_100px=round(best_twist, 3),
        weighted_rms_error_degrees=round(best_rms, 3),
        radial_rms_error_degrees=round(radial_rms, 3),
        sample_count=len(samples),
    )


def assign_arms(
    circles: Iterable[CircleFeature],
    center: Point,
    phase_degrees: float = 0.0,
    *,
    preserve_manual_fields: bool = True,
    locked_fields: Mapping[str, Collection[str]] | None = None,
    center_tolerance_degrees: float = 2.5,
    infer_parents: bool = True,
    model: ArmModel = "radial",
    twist_degrees_per_100px: float = 0.0,
) -> list[CircleFeature]:
    """Add polar data, six-arm ids, radial order, side, and optional parent links."""

    phase = normalize_angle(float(phase_degrees))

    def is_locked(circle: CircleFeature, field_name: str) -> bool:
        if not preserve_manual_fields:
            return False
        if locked_fields is None:
            return getattr(circle, field_name) is not None
        return field_name in locked_fields.get(circle.id, ())

    classified: list[CircleFeature] = []
    for circle in circles:
        distance, angle = polar_coordinates((circle.x, circle.y), center)
        if distance <= max(1.0, circle.radius * 0.25):
            suggested_arm: int | None = None
            residual = 0.0
        else:
            local_phase = arm_phase_at_distance(
                phase,
                distance,
                model=model,
                twist_degrees_per_100px=twist_degrees_per_100px,
            )
            suggested_arm, residual = _nearest_arm(angle, local_phase)
        arm_id = circle.arm_id if is_locked(circle, "arm_id") else suggested_arm
        suggested_side = (
            "center"
            if abs(residual) <= center_tolerance_degrees
            else ("left" if residual > 0 else "right")
        )
        side = circle.side if is_locked(circle, "side") else suggested_side
        classified.append(
            replace(
                circle,
                arm_id=arm_id,
                side=side,
                center_distance=distance,
                center_angle=angle,
            )
        )

    by_arm: dict[int, list[CircleFeature]] = {arm: [] for arm in range(1, 7)}
    for circle in classified:
        if circle.arm_id is not None:
            by_arm[circle.arm_id].append(circle)
    indexed: dict[str, CircleFeature] = {circle.id: circle for circle in classified}
    for arm_circles in by_arm.values():
        arm_circles.sort(
            key=lambda circle: (
                float(circle.center_distance or 0.0),
                circle.id,
            )
        )
        occupied = {
            circle.sequence_index
            for circle in arm_circles
            if is_locked(circle, "sequence_index")
            and circle.sequence_index is not None
        }
        next_index = 1
        for circle in arm_circles:
            if is_locked(circle, "sequence_index"):
                continue
            while next_index in occupied:
                next_index += 1
            indexed[circle.id] = replace(circle, sequence_index=next_index)
            occupied.add(next_index)
            next_index += 1

    if infer_parents:
        main_by_arm: dict[int, list[CircleFeature]] = {arm: [] for arm in range(1, 7)}
        for circle in indexed.values():
            if circle.circle_type == "main" and circle.arm_id is not None:
                main_by_arm[circle.arm_id].append(circle)
        for circle_id, circle in tuple(indexed.items()):
            if (
                circle.circle_type not in {"side", "satellite"}
                or circle.arm_id is None
                or not main_by_arm[circle.arm_id]
                or is_locked(circle, "parent_id")
            ):
                continue
            parent = min(
                main_by_arm[circle.arm_id],
                key=lambda main: abs(
                    float(main.center_distance or 0.0)
                    - float(circle.center_distance or 0.0)
                ),
            )
            indexed[circle_id] = replace(circle, parent_id=parent.id)
    return [indexed[circle.id] for circle in classified]


def estimate_arm_phase(
    circles: Iterable[CircleFeature], center: Point
) -> float:
    """Estimate a robust spoke phase in [0, 60) from main-circle angles."""

    angles = [
        polar_coordinates((circle.x, circle.y), center)[1] % 60.0
        for circle in circles
        if circle.circle_type == "main"
        and polar_coordinates((circle.x, circle.y), center)[0]
        > max(1.0, circle.radius * 0.25)
    ]
    if not angles:
        return 0.0
    # Circular mean on a 60-degree period.
    radians = [math.radians(angle * 6.0) for angle in angles]
    mean_sine = sum(math.sin(value) for value in radians) / len(radians)
    mean_cosine = sum(math.cos(value) for value in radians) / len(radians)
    if math.hypot(mean_sine, mean_cosine) < 1e-8:
        return median(angles) % 60.0
    return normalize_angle(math.degrees(math.atan2(mean_sine, mean_cosine)) / 6.0) % 60.0
