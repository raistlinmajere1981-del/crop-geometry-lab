"""Perspective correction from a reference quadrilateral or two field tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

PerspectiveMode = Literal["reference_quad", "field_tracks"]
Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class PerspectiveResult:
    image: NDArray[np.uint8]
    matrix: NDArray[np.float64]
    source_points: NDArray[np.float32]
    destination_points: NDArray[np.float32]
    mode: PerspectiveMode
    preserves_full_image: bool = True


def _points_array(points: Sequence[Sequence[float]]) -> NDArray[np.float32]:
    array = np.asarray(points, dtype=np.float32)
    if array.shape != (4, 2):
        raise ValueError("Exactly four two-dimensional points are required")
    if not np.isfinite(array).all():
        raise ValueError("Perspective points must be finite")
    if min(
        float(np.linalg.norm(array[index] - array[other]))
        for index in range(4)
        for other in range(index)
    ) < 1.0:
        raise ValueError("Perspective points must be distinct")
    return array


def order_quad_points(
    points: Sequence[Sequence[float]],
) -> NDArray[np.float32]:
    """Order arbitrary quadrilateral points as TL, TR, BR, BL."""

    array = _points_array(points)
    sums = array.sum(axis=1)
    differences = np.diff(array, axis=1).reshape(-1)
    ordered = np.empty((4, 2), dtype=np.float32)
    ordered[0] = array[int(np.argmin(sums))]
    ordered[2] = array[int(np.argmax(sums))]
    ordered[1] = array[int(np.argmin(differences))]
    ordered[3] = array[int(np.argmax(differences))]
    if len(np.unique(ordered, axis=0)) != 4:
        # The sum/difference method can be ambiguous for a perfect diamond.
        center = array.mean(axis=0)
        angles = np.arctan2(array[:, 1] - center[1], array[:, 0] - center[0])
        cycle = array[np.argsort(angles)]
        start = int(np.argmin(cycle.sum(axis=1)))
        cycle = np.roll(cycle, -start, axis=0)
        if cycle[1, 0] < cycle[-1, 0]:
            cycle = cycle[[0, 3, 2, 1]]
        ordered = cycle.astype(np.float32)
    area = abs(float(cv2.contourArea(ordered)))
    if area < 4.0:
        raise ValueError("Reference quadrilateral is degenerate")
    return ordered


def _rectangle_destination(
    ordered: NDArray[np.float32],
) -> NDArray[np.float32]:
    top_left, top_right, bottom_right, bottom_left = ordered
    width = max(
        float(np.linalg.norm(top_right - top_left)),
        float(np.linalg.norm(bottom_right - bottom_left)),
    )
    height = max(
        float(np.linalg.norm(bottom_left - top_left)),
        float(np.linalg.norm(bottom_right - top_right)),
    )
    output_width = max(2, int(round(width)))
    output_height = max(2, int(round(height)))
    return np.asarray(
        [
            [0.0, 0.0],
            [output_width - 1.0, 0.0],
            [output_width - 1.0, output_height - 1.0],
            [0.0, output_height - 1.0],
        ],
        dtype=np.float32,
    )


def _fit_transform_to_source_canvas(
    image: NDArray[np.uint8],
    matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit every transformed source corner into the original-size canvas."""

    height, width = image.shape[:2]
    source_corners = np.asarray(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack(
        [source_corners, np.ones(4, dtype=np.float64)]
    )
    mapped_homogeneous = (matrix @ homogeneous.T).T
    denominators = mapped_homogeneous[:, 2]
    if (
        np.any(np.abs(denominators) < 1e-7)
        or float(np.min(denominators)) * float(np.max(denominators)) <= 0
    ):
        raise ValueError(
            "Seçilen perspektif kaynak görüntünün içinden bir ufuk çizgisi "
            "geçiriyor; dört noktayı daha geniş ve tutarlı seçin"
        )
    mapped = mapped_homogeneous[:, :2] / denominators[:, None]
    if not np.isfinite(mapped).all():
        raise ValueError("Perspective transform produced non-finite bounds")
    minimum = mapped.min(axis=0)
    maximum = mapped.max(axis=0)
    span = maximum - minimum
    if float(np.min(span)) < 1e-6:
        raise ValueError("Perspective transform collapses the source frame")

    padding = 1.0 if min(width, height) > 4 else 0.0
    available_width = max(1.0, width - 1.0 - padding * 2.0)
    available_height = max(1.0, height - 1.0 - padding * 2.0)
    scale = min(
        1.0,
        available_width / float(span[0]),
        available_height / float(span[1]),
    )
    center = (minimum + maximum) * 0.5
    translation_x = (width - 1.0) * 0.5 - scale * float(center[0])
    translation_y = (height - 1.0) * 0.5 - scale * float(center[1])
    fit = np.asarray(
        [
            [scale, 0.0, translation_x],
            [0.0, scale, translation_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return fit @ matrix


def _finish_perspective(
    image: NDArray[np.uint8],
    ordered: NDArray[np.float32],
    destination: NDArray[np.float32],
    mode: PerspectiveMode,
    *,
    preserve_full_image: bool,
) -> PerspectiveResult:
    matrix = cv2.getPerspectiveTransform(ordered, destination).astype(np.float64)
    if preserve_full_image:
        matrix = _fit_transform_to_source_canvas(image, matrix)
        height, width = image.shape[:2]
        output_size = (width, height)
        final_destination = cv2.perspectiveTransform(
            ordered.reshape(1, 4, 2),
            matrix,
        ).reshape(4, 2)
        border_mode = cv2.BORDER_CONSTANT
    else:
        output_size = (
            int(destination[1, 0]) + 1,
            int(destination[2, 1]) + 1,
        )
        final_destination = destination
        border_mode = cv2.BORDER_REPLICATE
    corrected = cv2.warpPerspective(
        image,
        matrix,
        output_size,
        flags=cv2.INTER_CUBIC,
        borderMode=border_mode,
        borderValue=0,
    )
    return PerspectiveResult(
        image=corrected,
        matrix=matrix,
        source_points=ordered,
        destination_points=np.asarray(final_destination, dtype=np.float32),
        mode=mode,
        preserves_full_image=preserve_full_image,
    )


def correct_reference_quad(
    image: NDArray[np.uint8],
    points: Sequence[Sequence[float]],
    *,
    preserve_full_image: bool = True,
) -> PerspectiveResult:
    """Rectify a quadrilateral while keeping the whole source frame by default."""

    ordered = order_quad_points(points)
    destination = _rectangle_destination(ordered)
    return _finish_perspective(
        image,
        ordered,
        destination,
        "reference_quad",
        preserve_full_image=preserve_full_image,
    )


def _ordered_track_quad(
    points: Sequence[Sequence[float]],
) -> NDArray[np.float32]:
    """Convert A0, A1, B0, B1 trace clicks into A0, B0, B1, A1."""

    array = _points_array(points)
    a0, a1, b0, b1 = array
    aligned_cost = np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1)
    crossed_cost = np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0)
    if crossed_cost < aligned_cost:
        b0, b1 = b1.copy(), b0.copy()
    ordered = np.asarray([a0, b0, b1, a1], dtype=np.float32)
    if abs(float(cv2.contourArea(ordered))) < 4.0:
        raise ValueError("Selected field tracks do not enclose a usable strip")
    return ordered


def _orientation_preserving_track_destination(
    ordered: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Build a parallel target strip without rotating it to the image axes."""

    a0, b0, b1, a1 = ordered.astype(np.float64)
    vector_a = a1 - a0
    vector_b = b1 - b0
    length_a = float(np.linalg.norm(vector_a))
    length_b = float(np.linalg.norm(vector_b))
    direction_a = vector_a / max(length_a, 1e-9)
    direction_b = vector_b / max(length_b, 1e-9)
    if float(np.dot(direction_a, direction_b)) < 0:
        direction_b = -direction_b
    direction = direction_a + direction_b
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        direction = direction_a
    else:
        direction /= norm
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    separation_start = float(np.dot(b0 - a0, normal))
    separation_end = float(np.dot(b1 - a1, normal))
    if separation_start + separation_end < 0:
        normal = -normal
        separation_start = -separation_start
        separation_end = -separation_end
    separation = max(
        2.0,
        (abs(separation_start) + abs(separation_end)) / 2.0,
    )
    length = max(2.0, (length_a + length_b) / 2.0)
    center = ordered.astype(np.float64).mean(axis=0)
    center_a = center - normal * separation / 2.0
    center_b = center + normal * separation / 2.0
    return np.asarray(
        [
            center_a - direction * length / 2.0,
            center_b - direction * length / 2.0,
            center_b + direction * length / 2.0,
            center_a + direction * length / 2.0,
        ],
        dtype=np.float32,
    )


def correct_field_tracks(
    image: NDArray[np.uint8],
    points: Sequence[Sequence[float]],
    *,
    preserve_full_image: bool = True,
) -> PerspectiveResult:
    """Make two selected, physically parallel field tracks parallel in the image.

    Point order is two endpoints on track A followed by two endpoints on track B.
    Endpoint direction may be reversed; the closest cross-track pairing is used.
    The selected strip defines the rectification, but the complete source
    frame remains visible unless ``preserve_full_image`` is disabled. This is
    a practical local rectification, not an assertion of absolute ground scale.
    """

    ordered = _ordered_track_quad(points)
    a0, b0, b1, a1 = ordered
    width = max(
        2,
        int(
            round(
                (float(np.linalg.norm(a0 - b0)) + float(np.linalg.norm(a1 - b1)))
                / 2.0
            )
        ),
    )
    height = max(
        2,
        int(
            round(
                (float(np.linalg.norm(a0 - a1)) + float(np.linalg.norm(b0 - b1)))
                / 2.0
            )
        ),
    )
    destination = (
        _orientation_preserving_track_destination(ordered)
        if preserve_full_image
        else np.asarray(
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ],
            dtype=np.float32,
        )
    )
    return _finish_perspective(
        image,
        ordered,
        destination,
        "field_tracks",
        preserve_full_image=preserve_full_image,
    )


def correct_perspective(
    image: NDArray[np.uint8],
    points: Sequence[Sequence[float]],
    mode: PerspectiveMode = "reference_quad",
    *,
    preserve_full_image: bool = True,
) -> PerspectiveResult:
    if image.size == 0:
        raise ValueError("Image cannot be empty")
    if mode == "reference_quad":
        return correct_reference_quad(
            image,
            points,
            preserve_full_image=preserve_full_image,
        )
    if mode == "field_tracks":
        return correct_field_tracks(
            image,
            points,
            preserve_full_image=preserve_full_image,
        )
    raise ValueError(f"Unsupported perspective mode: {mode}")
