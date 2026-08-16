from __future__ import annotations

import cv2
import numpy as np

from src.feature_table import CircleFeature
from src.quality_control import rank_weak_visual_support


def test_uniform_false_annotation_ranks_below_visible_circle() -> None:
    image = np.full((180, 180, 3), 70, dtype=np.uint8)
    cv2.circle(image, (55, 90), 18, (210, 210, 210), -1, cv2.LINE_AA)
    circles = [
        CircleFeature(id="visible", x=55, y=90, radius=18),
        CircleFeature(id="unsupported", x=135, y=90, radius=12),
    ]

    ranked = rank_weak_visual_support(image, circles, limit=2)

    assert [item.id for item in ranked] == ["unsupported", "visible"]
    assert ranked[0].score < ranked[1].score
