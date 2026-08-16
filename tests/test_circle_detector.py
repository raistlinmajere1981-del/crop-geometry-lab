from __future__ import annotations

import cv2
import numpy as np

from src.circle_detector import DetectorConfig, detect_circles


def test_opencv_detector_finds_synthetic_circle_candidates() -> None:
    image = np.full((420, 420, 3), 65, dtype=np.uint8)
    truths = [(90, 90, 18), (300, 95, 27), (120, 285, 22), (305, 300, 34)]
    for x, y, radius in truths:
        cv2.circle(image, (x, y), radius, (215, 215, 215), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), radius, (15, 15, 15), 2, cv2.LINE_AA)

    candidates = detect_circles(
        image,
        DetectorConfig(
            min_radius=10,
            max_radius=42,
            min_distance=35,
            hough_threshold=14,
            min_confidence=0.08,
        ),
    )

    matched = 0
    for truth_x, truth_y, truth_radius in truths:
        if any(
            np.hypot(candidate.x - truth_x, candidate.y - truth_y) < 6
            and abs(candidate.radius - truth_radius) < 6
            for candidate in candidates
        ):
            matched += 1
    assert matched >= 3
    assert all(candidate.source == "automatic" for candidate in candidates)
    assert all(0 <= candidate.confidence <= 1 for candidate in candidates)
