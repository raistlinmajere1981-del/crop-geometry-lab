from __future__ import annotations

import numpy as np

from src.feature_table import CircleFeature
from src.visualizer import draw_overlay


def test_overlay_uses_custom_stroke_and_fill_appearance() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    circle = CircleFeature(
        id="styled",
        x=50,
        y=50,
        radius=20,
        stroke_color="#ff0000",
        stroke_width=5,
        fill_enabled=True,
        fill_color="#00ff00",
        fill_opacity=1,
    )

    rendered = draw_overlay(
        image,
        [circle],
        draw_arm_guides=False,
        draw_labels=False,
    )

    # OpenCV pixels are BGR. Sample away from the center marker and AA edge.
    interior = rendered[50, 58]
    boundary = rendered[50, 70]
    assert interior[1] > 240
    assert interior[0] < 15 and interior[2] < 15
    assert boundary[2] > 240
    assert boundary[0] < 15 and boundary[1] < 15


def test_overlay_fill_opacity_blends_with_source() -> None:
    image = np.full((80, 80, 3), 100, dtype=np.uint8)
    circle = CircleFeature(
        id="blend",
        x=40,
        y=40,
        radius=15,
        fill_enabled=True,
        fill_color="#ffffff",
        fill_opacity=0.5,
    )

    rendered = draw_overlay(
        image,
        [circle],
        draw_arm_guides=False,
        draw_labels=False,
    )

    interior = rendered[40, 47]
    assert np.all((interior >= 176) & (interior <= 179))
