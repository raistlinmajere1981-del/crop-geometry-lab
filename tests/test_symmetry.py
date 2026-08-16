from __future__ import annotations

import math

import cv2
import numpy as np

from src.feature_table import CircleFeature
from src.symmetry_tests import geometric_symmetry_errors, image_symmetry_errors


def exact_sixfold_fixture() -> tuple[np.ndarray, list[CircleFeature], tuple[float, float]]:
    image = np.full((401, 401, 3), 80, dtype=np.uint8)
    center = (200.0, 200.0)
    circles: list[CircleFeature] = []
    cv2.circle(image, (200, 200), 12, (210, 210, 210), -1, cv2.LINE_AA)
    circles.append(CircleFeature(id="center", x=200, y=200, radius=12))
    for arm in range(6):
        angle = math.radians(arm * 60)
        x = center[0] + 100 * math.cos(angle)
        y = center[1] - 100 * math.sin(angle)
        cv2.circle(
            image,
            (int(round(x)), int(round(y))),
            15,
            (220, 220, 220),
            -1,
            cv2.LINE_AA,
        )
        circles.append(
            CircleFeature(
                id=f"arm-{arm + 1}",
                x=x,
                y=y,
                radius=15,
                arm_id=arm + 1,
            )
        )
    return image, circles, center


def test_exact_fixture_has_low_sixfold_image_and_geometry_error() -> None:
    image, circles, center = exact_sixfold_fixture()
    image_errors = image_symmetry_errors(image, center, max_radius=175)
    geometry_errors = geometric_symmetry_errors(circles, center)

    assert max(item.normalized_mae for item in image_errors) < 0.004
    assert max(item.normalized_position_error for item in geometry_errors) < 0.01
    assert min(item.matched_fraction for item in geometry_errors) == 1.0


def test_asymmetric_patch_increases_image_error() -> None:
    image, _, center = exact_sixfold_fixture()
    baseline = np.mean(
        [item.normalized_mae for item in image_symmetry_errors(image, center)]
    )
    cv2.rectangle(image, (220, 220), (295, 280), (255, 10, 10), -1)
    changed = np.mean(
        [item.normalized_mae for item in image_symmetry_errors(image, center)]
    )

    assert changed > baseline + 0.005
