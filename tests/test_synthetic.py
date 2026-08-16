from __future__ import annotations

import cv2

from src.feature_table import read_json
from src.synthetic import generate_synthetic_image, write_synthetic_dataset


def test_synthetic_fixture_requires_no_external_sample(tmp_path) -> None:
    image, circles, center = generate_synthetic_image(
        size=480,
        rings=2,
        add_texture=False,
        add_field_tracks=False,
    )

    assert image.shape == (480, 480, 3)
    assert len(circles) == 1 + 6 * 3
    assert center == (240.0, 240.0)
    assert {circle.arm_id for circle in circles if circle.id != "truth-center"} == {
        1,
        2,
        3,
        4,
        5,
        6,
    }

    image_path, truth_path = write_synthetic_dataset(
        tmp_path / "synthetic.png",
        size=480,
        rings=2,
    )
    assert cv2.imread(str(image_path)) is not None
    assert len(read_json(truth_path).circles) == len(circles)

