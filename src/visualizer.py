"""OpenCV rendering for arm layers and export images."""

from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

from .arm_classifier import ArmModel, arm_phase_at_distance
from .feature_table import CircleFeature

Point = tuple[float, float]

# BGR values chosen for separation on both green fields and grayscale imagery.
ARM_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    1: (68, 68, 255),
    2: (40, 180, 255),
    3: (70, 220, 120),
    4: (230, 190, 55),
    5: (255, 90, 170),
    6: (180, 75, 255),
}


def _safe_point(x: float, y: float) -> tuple[int, int]:
    return int(round(x)), int(round(y))


def _hex_to_bgr(
    value: str | None,
    fallback: tuple[int, int, int],
) -> tuple[int, int, int]:
    if value is None:
        return fallback
    color = value.lstrip("#")
    if len(color) != 6:
        return fallback
    try:
        red, green, blue = (
            int(color[0:2], 16),
            int(color[2:4], 16),
            int(color[4:6], 16),
        )
    except ValueError:
        return fallback
    return blue, green, red


def _fill_circle_with_alpha(
    canvas: NDArray[np.uint8],
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    opacity: float,
) -> None:
    """Alpha-fill only the affected ROI to keep large exports inexpensive."""

    if opacity <= 0:
        return
    height, width = canvas.shape[:2]
    left = max(0, center[0] - radius - 2)
    right = min(width, center[0] + radius + 3)
    top = max(0, center[1] - radius - 2)
    bottom = min(height, center[1] + radius + 3)
    if left >= right or top >= bottom:
        return
    region = canvas[top:bottom, left:right]
    local_center = center[0] - left, center[1] - top
    if opacity >= 1:
        cv2.circle(region, local_center, radius, color, -1, cv2.LINE_AA)
        return
    fill_layer = region.copy()
    cv2.circle(fill_layer, local_center, radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(
        fill_layer,
        opacity,
        region,
        1.0 - opacity,
        0.0,
        dst=region,
    )


def draw_overlay(
    image: NDArray[np.uint8],
    circles: Iterable[CircleFeature],
    *,
    center: Point | None = None,
    phase_degrees: float = 0.0,
    arm_model: ArmModel = "radial",
    arm_twist_degrees_per_100px: float = 0.0,
    draw_arm_guides: bool = True,
    draw_labels: bool = True,
) -> NDArray[np.uint8]:
    """Render a non-destructive, color-coded analysis layer."""

    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        canvas = image.copy()
    overlay = canvas.copy()
    height, width = canvas.shape[:2]

    if center is not None and draw_arm_guides:
        guide_length = math.hypot(width, height)
        center_point = _safe_point(*center)
        for arm_index in range(6):
            if arm_model == "radial":
                angle = math.radians(phase_degrees + arm_index * 60.0)
                endpoint = _safe_point(
                    center[0] + guide_length * math.cos(angle),
                    center[1] - guide_length * math.sin(angle),
                )
                cv2.line(
                    overlay,
                    center_point,
                    endpoint,
                    ARM_COLORS_BGR[arm_index + 1],
                    2,
                    cv2.LINE_AA,
                )
            else:
                distances = np.linspace(0.0, guide_length, 320)
                points = []
                for distance in distances:
                    local_phase = arm_phase_at_distance(
                        phase_degrees + arm_index * 60.0,
                        float(distance),
                        model=arm_model,
                        twist_degrees_per_100px=(
                            arm_twist_degrees_per_100px
                        ),
                    )
                    angle = math.radians(local_phase)
                    points.append(
                        _safe_point(
                            center[0] + float(distance) * math.cos(angle),
                            center[1] - float(distance) * math.sin(angle),
                        )
                    )
                cv2.polylines(
                    overlay,
                    [np.asarray(points, dtype=np.int32)],
                    False,
                    ARM_COLORS_BGR[arm_index + 1],
                    2,
                    cv2.LINE_AA,
                )
        canvas = cv2.addWeighted(overlay, 0.38, canvas, 0.62, 0.0)

    for circle in circles:
        if circle.circle_type == "center":
            fallback_color = (255, 255, 255)
        elif circle.circle_type == "unclassified":
            fallback_color = (170, 160, 150)
        elif circle.circle_type in {"satellite", "side"}:
            fallback_color = ARM_COLORS_BGR.get(
                circle.arm_id or 0, (240, 216, 73)
            )
        else:
            fallback_color = ARM_COLORS_BGR.get(
                circle.arm_id or 0, (70, 220, 255)
            )
        color = _hex_to_bgr(circle.stroke_color, fallback_color)
        fill_color = _hex_to_bgr(circle.fill_color, color)
        circle_center = _safe_point(circle.x, circle.y)
        circle_radius = max(1, int(round(circle.radius)))
        default_thickness = (
            4
            if circle.circle_type == "center"
            else 3 if circle.circle_type == "main" else 2
        )
        thickness = max(
            1,
            int(
                round(
                    circle.stroke_width
                    if circle.stroke_width is not None
                    else default_thickness
                )
            ),
        )
        if circle.fill_enabled:
            _fill_circle_with_alpha(
                canvas,
                circle_center,
                circle_radius,
                fill_color,
                min(1.0, max(0.0, circle.fill_opacity)),
            )
        cv2.circle(canvas, circle_center, circle_radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, circle_center, 2, color, -1, cv2.LINE_AA)
        if draw_labels:
            label = circle.id
            if circle.arm_id is not None:
                label += f" · K{circle.arm_id}"
            cv2.putText(
                canvas,
                label,
                (circle_center[0] + 5, circle_center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
    if center is not None:
        center_point = _safe_point(*center)
        cv2.drawMarker(
            canvas,
            center_point,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=3,
            line_type=cv2.LINE_AA,
        )
        cv2.drawMarker(
            canvas,
            center_point,
            (15, 15, 15),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=1,
            line_type=cv2.LINE_AA,
        )
    return canvas


def save_overlay_png(
    image: NDArray[np.uint8],
    circles: Iterable[CircleFeature],
    destination: str,
    *,
    center: Point | None = None,
    phase_degrees: float = 0.0,
    arm_model: ArmModel = "radial",
    arm_twist_degrees_per_100px: float = 0.0,
) -> None:
    rendered = draw_overlay(
        image,
        circles,
        center=center,
        phase_degrees=phase_degrees,
        arm_model=arm_model,
        arm_twist_degrees_per_100px=arm_twist_degrees_per_100px,
    )
    if not cv2.imwrite(destination, rendered):
        raise OSError(f"Could not write PNG to {destination}")
