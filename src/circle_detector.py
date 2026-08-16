"""OpenCV circular-blob detection with a Hough fallback and confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

from .feature_table import CircleFeature


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    min_radius: float = 4.0
    max_radius: float = 80.0
    min_distance: float = 10.0
    min_circularity: float = 0.38
    hough_threshold: float = 34.0
    canny_threshold: float = 110.0
    min_confidence: float = 0.12
    max_candidates: int = 350
    max_working_dimension: int = 1600

    def __post_init__(self) -> None:
        if self.min_radius <= 0:
            raise ValueError("min_radius must be positive")
        if self.max_radius <= self.min_radius:
            raise ValueError("max_radius must exceed min_radius")
        if self.min_distance <= 0:
            raise ValueError("min_distance must be positive")
        if not 0.05 <= self.min_circularity <= 1.0:
            raise ValueError("min_circularity must be between 0.05 and 1")
        if not 1 <= self.hough_threshold <= 100:
            raise ValueError("hough_threshold must be between 1 and 100")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")


def _to_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("Expected a grayscale, BGR, or BGRA image")


def _preprocess(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.GaussianBlur(enhanced, (5, 5), 1.1)


def _edge_confidence(
    gray: NDArray[np.uint8],
    gradient_x: NDArray[np.float32],
    gradient_y: NDArray[np.float32],
    x: float,
    y: float,
    radius: float,
) -> float:
    samples = max(72, min(360, int(round(2.0 * math.pi * radius))))
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    height, width = gray.shape
    oriented_responses: list[NDArray[np.float32]] = []
    cosine = np.cos(angles).astype(np.float32)
    sine = np.sin(angles).astype(np.float32)
    for radius_offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
        sample_radius = max(1.0, radius + radius_offset)
        xs = np.rint(x + sample_radius * cosine).astype(np.int32)
        ys = np.rint(y + sample_radius * sine).astype(np.int32)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        response = np.zeros(samples, dtype=np.float32)
        if int(valid.sum()) >= samples // 2:
            gx = gradient_x[ys[valid], xs[valid]]
            gy = gradient_y[ys[valid], xs[valid]]
            magnitude = np.hypot(gx, gy)
            radial_alignment = np.abs(
                (gx * cosine[valid] + gy * sine[valid])
                / np.maximum(magnitude, 1e-6)
            )
            strength = np.minimum(1.0, magnitude / 110.0)
            response[valid] = radial_alignment * strength
        oriented_responses.append(response)
    if oriented_responses:
        best_response = np.max(np.stack(oriented_responses), axis=0)
        edge_support = 0.55 * float(np.mean(best_response)) + 0.45 * float(
            np.mean(best_response >= 0.42)
        )
    else:
        edge_support = 0.0

    inner_radius = max(1.0, radius * 0.72)
    outer_radius = radius * 1.28
    inner_x = np.rint(x + inner_radius * np.cos(angles)).astype(np.int32)
    inner_y = np.rint(y + inner_radius * np.sin(angles)).astype(np.int32)
    outer_x = np.rint(x + outer_radius * np.cos(angles)).astype(np.int32)
    outer_y = np.rint(y + outer_radius * np.sin(angles)).astype(np.int32)
    valid = (
        (inner_x >= 0)
        & (inner_x < width)
        & (inner_y >= 0)
        & (inner_y < height)
        & (outer_x >= 0)
        & (outer_x < width)
        & (outer_y >= 0)
        & (outer_y < height)
    )
    photometric_support = 0.0
    if int(valid.sum()) >= samples // 3:
        inner_values = gray[inner_y[valid], inner_x[valid]].astype(np.float32)
        outer_values = gray[outer_y[valid], outer_x[valid]].astype(np.float32)
        signed_difference = inner_values - outer_values
        mean_signed_contrast = abs(float(np.mean(signed_difference)))
        mean_absolute_contrast = float(np.mean(np.abs(signed_difference)))
        # A real disk boundary tends to have the same contrast direction around
        # most of its circumference. Dense field texture can have high absolute
        # contrast while cancelling in sign, so coherence suppresses those
        # visually plausible but geometrically false Hough circles.
        coherence = mean_signed_contrast / max(1e-6, mean_absolute_contrast)
        contrast_strength = min(1.0, mean_signed_contrast / 42.0)
        photometric_support = contrast_strength * math.sqrt(coherence)
    return max(
        0.0,
        min(1.0, 0.42 * edge_support + 0.58 * photometric_support),
    )


def _non_maximum_suppression(
    candidates: Iterable[tuple[float, float, float, float]],
    max_candidates: int,
) -> list[tuple[float, float, float, float]]:
    kept: list[tuple[float, float, float, float]] = []
    for candidate in sorted(candidates, key=lambda item: item[3], reverse=True):
        x, y, radius, _ = candidate
        duplicate = False
        for kept_x, kept_y, kept_radius, _ in kept:
            center_distance = math.hypot(x - kept_x, y - kept_y)
            radius_similarity = abs(radius - kept_radius) / max(radius, kept_radius)
            if (
                center_distance < 0.45 * min(radius, kept_radius)
                and radius_similarity < 0.35
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
        if len(kept) >= max_candidates:
            break
    return kept


def _blob_candidates(
    gray: NDArray[np.uint8],
    gradient_x: NDArray[np.float32],
    gradient_y: NDArray[np.float32],
    settings: DetectorConfig,
    scale: float,
) -> list[tuple[float, float, float, float]]:
    """Detect light, approximately circular regions over many thresholds."""

    parameters = cv2.SimpleBlobDetector_Params()
    parameters.minThreshold = 5.0
    parameters.maxThreshold = 250.0
    parameters.thresholdStep = 5.0
    parameters.minRepeatability = 2
    parameters.minDistBetweenBlobs = max(2.0, settings.min_distance * scale * 0.3)
    parameters.filterByColor = True
    parameters.blobColor = 255
    parameters.filterByArea = True
    parameters.minArea = max(
        4.0, math.pi * (settings.min_radius * scale) ** 2 * 0.48
    )
    parameters.maxArea = (
        math.pi * (settings.max_radius * scale) ** 2 * 1.25
    )
    parameters.filterByCircularity = True
    parameters.minCircularity = float(settings.min_circularity)
    parameters.filterByConvexity = True
    parameters.minConvexity = 0.40
    parameters.filterByInertia = True
    parameters.minInertiaRatio = 0.30
    detector = cv2.SimpleBlobDetector_create(parameters)
    keypoints = detector.detect(gray)
    candidates: list[tuple[float, float, float, float]] = []
    for keypoint in keypoints:
        radius = float(keypoint.size) / 2.0
        x, y = map(float, keypoint.pt)
        if not (
            settings.min_radius * scale * 0.75
            <= radius
            <= settings.max_radius * scale * 1.15
        ):
            continue
        edge_score = _edge_confidence(
            gray, gradient_x, gradient_y, x, y, radius
        )
        # Passing repeatability, circularity, convexity, and inertia filters is
        # itself evidence. Edge support then differentiates clean from weak
        # candidates without discarding softly bounded crop marks.
        confidence = min(1.0, 0.38 + 0.62 * edge_score)
        if confidence >= settings.min_confidence:
            candidates.append(
                (x / scale, y / scale, radius / scale, confidence)
            )
    return candidates


def _hough_fallback_candidates(
    gray: NDArray[np.uint8],
    gradient_x: NDArray[np.float32],
    gradient_y: NDArray[np.float32],
    settings: DetectorConfig,
    scale: float,
) -> list[tuple[float, float, float, float]]:
    """Recover candidates when thresholded blobs are too weak or outline-only."""

    raw = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(2.0, settings.min_distance * scale),
        param1=settings.canny_threshold,
        param2=settings.hough_threshold,
        minRadius=max(1, int(round(settings.min_radius * scale))),
        maxRadius=max(2, int(round(settings.max_radius * scale))),
    )
    if raw is None:
        return []
    candidates: list[tuple[float, float, float, float]] = []
    for x, y, radius in raw[0]:
        confidence = _edge_confidence(
            gray,
            gradient_x,
            gradient_y,
            float(x),
            float(y),
            float(radius),
        )
        if confidence >= max(settings.min_confidence, 0.34):
            candidates.append(
                (
                    float(x) / scale,
                    float(y) / scale,
                    float(radius) / scale,
                    confidence,
                )
            )
    return candidates


def detect_circles(
    image: NDArray[np.uint8], config: DetectorConfig | None = None
) -> list[CircleFeature]:
    """Return editable candidates; Hough output is deliberately not ground truth."""

    settings = config or DetectorConfig()
    if image.size == 0:
        raise ValueError("Image cannot be empty")
    height, width = image.shape[:2]
    scale = min(
        1.0, float(settings.max_working_dimension) / float(max(height, width))
    )
    working = image
    if scale < 1.0:
        working = cv2.resize(
            image,
            (max(2, int(round(width * scale))), max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    blob_gray = cv2.GaussianBlur(_to_gray(working), (3, 3), 0.7)
    blob_gradient_x = cv2.Sobel(blob_gray, cv2.CV_32F, 1, 0, ksize=3)
    blob_gradient_y = cv2.Sobel(blob_gray, cv2.CV_32F, 0, 1, ksize=3)
    scaled_candidates = _blob_candidates(
        blob_gray,
        blob_gradient_x,
        blob_gradient_y,
        settings,
        scale,
    )
    # Hough is valuable for outline drawings and very low-contrast imagery, but
    # on dense formations it readily hallucinates large circles around clusters.
    # Use it only when multi-threshold blob evidence is genuinely sparse.
    if len(scaled_candidates) < 3:
        hough_gray = _preprocess(working)
        hough_gradient_x = cv2.Sobel(hough_gray, cv2.CV_32F, 1, 0, ksize=3)
        hough_gradient_y = cv2.Sobel(hough_gray, cv2.CV_32F, 0, 1, ksize=3)
        scaled_candidates.extend(
            _hough_fallback_candidates(
                hough_gray,
                hough_gradient_x,
                hough_gradient_y,
                settings,
                scale,
            )
        )
    candidates = _non_maximum_suppression(
        scaled_candidates, settings.max_candidates
    )
    candidates.sort(key=lambda item: (round(item[1], 1), round(item[0], 1), item[2]))
    return [
        CircleFeature(
            id=f"auto-{index:04d}",
            x=x,
            y=y,
            radius=radius,
            confidence=confidence,
            circle_type="main",
            source="automatic",
        )
        for index, (x, y, radius, confidence) in enumerate(candidates, start=1)
    ]
