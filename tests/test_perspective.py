from __future__ import annotations

import cv2
import numpy as np

from src.perspective import (
    correct_field_tracks,
    correct_reference_quad,
    order_quad_points,
)


def assert_source_frame_fits(result, width: int, height: int) -> None:
    corners = np.asarray(
        [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]],
        dtype=np.float32,
    )
    mapped = cv2.perspectiveTransform(corners, result.matrix).reshape(4, 2)

    assert np.isfinite(mapped).all()
    assert float(mapped[:, 0].min()) >= -1e-3
    assert float(mapped[:, 1].min()) >= -1e-3
    assert float(mapped[:, 0].max()) <= width - 1 + 1e-3
    assert float(mapped[:, 1].max()) <= height - 1 + 1e-3


def assert_track_edges_are_parallel(mapped: np.ndarray) -> None:
    vector_a = mapped[3] - mapped[0]
    vector_b = mapped[2] - mapped[1]
    cross = float(
        vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
    )
    normalized_cross = abs(cross) / max(
        float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b)),
        1e-9,
    )

    assert normalized_cross < 1e-5
    assert float(np.dot(vector_a, vector_b)) > 0


def test_quad_points_are_ordered_and_mapped_to_output_corners() -> None:
    image = np.zeros((120, 140, 3), dtype=np.uint8)
    points = [(112, 102), (18, 15), (126, 20), (8, 96)]
    ordered = order_quad_points(points)
    result = correct_reference_quad(image, points)
    mapped = cv2.perspectiveTransform(
        ordered.reshape(1, 4, 2), result.matrix
    ).reshape(4, 2)

    assert result.image.shape[0] > 70
    assert result.image.shape[1] > 90
    assert result.image.shape[:2] == image.shape[:2]
    assert result.preserves_full_image is True
    assert np.allclose(mapped, result.destination_points, atol=1e-3)
    assert_source_frame_fits(result, 140, 120)


def test_two_field_tracks_become_parallel_rectangle_edges() -> None:
    image = np.zeros((130, 150, 3), dtype=np.uint8)
    # Two converging tracks; B is intentionally clicked in reverse.
    points = [(35, 12), (16, 112), (124, 117), (104, 20)]
    result = correct_field_tracks(image, points)
    mapped = cv2.perspectiveTransform(
        result.source_points.reshape(1, 4, 2), result.matrix
    ).reshape(4, 2)

    assert result.mode == "field_tracks"
    assert np.allclose(mapped, result.destination_points, atol=1e-3)
    assert result.image.shape[:2] == image.shape[:2]
    assert_track_edges_are_parallel(mapped)
    assert_source_frame_fits(result, 150, 130)


def test_field_track_rectification_preserves_user_like_full_frame() -> None:
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    points = [(2, 244), (270, 4), (795, 559), (650, 795)]

    result = correct_field_tracks(image, points)
    mapped_tracks = cv2.perspectiveTransform(
        result.source_points.reshape(1, 4, 2),
        result.matrix,
    ).reshape(4, 2)

    assert result.image.shape[:2] == (800, 800)
    assert_track_edges_are_parallel(mapped_tracks)
    assert_source_frame_fits(result, 800, 800)


def test_selection_crop_remains_available_as_explicit_opt_out() -> None:
    image = np.zeros((130, 150, 3), dtype=np.uint8)
    points = [(35, 12), (16, 112), (124, 117), (104, 20)]

    result = correct_field_tracks(
        image,
        points,
        preserve_full_image=False,
    )

    assert result.preserves_full_image is False
    assert result.image.shape[:2] != image.shape[:2]
