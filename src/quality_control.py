"""Human-review helpers for finding weakly supported circle annotations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

from .feature_table import CircleFeature


@dataclass(frozen=True, slots=True)
class VisualSupport:
    """Image evidence for one annotated circle; lower scores are weaker."""

    id: str
    score: float
    contrast_score: float
    edge_score: float
    positive_edge_fraction: float
    x: float
    y: float
    radius: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def _circle_visual_support(
    luminance: NDArray[np.float32],
    circle: CircleFeature,
) -> VisualSupport:
    half_size = max(5.0, circle.radius * 2.0)
    patch_size = max(12, int(round(half_size * 2.0 + 1.0)))
    patch = cv2.getRectSubPix(
        luminance,
        (patch_size, patch_size),
        (float(circle.x), float(circle.y)),
    )
    normalized = cv2.resize(
        patch,
        (96, 96),
        interpolation=cv2.INTER_CUBIC,
    )
    rows, columns = np.mgrid[:96, :96]
    radial_distance = np.hypot(columns - 47.5, rows - 47.5)
    normalized_radius = 24.0
    inside = normalized[radial_distance <= normalized_radius * 0.72]
    annulus = normalized[
        (radial_distance >= normalized_radius * 1.15)
        & (radial_distance <= normalized_radius * 1.65)
    ]
    local = normalized[radial_distance <= normalized_radius * 1.75]
    local_scale = float(np.std(local)) + 1e-6
    contrast = (float(np.mean(inside)) - float(np.mean(annulus))) / local_scale

    angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    inner_x = (
        47.5 + normalized_radius * 0.72 * np.cos(angles)
    ).astype(np.float32)[None, :]
    inner_y = (
        47.5 + normalized_radius * 0.72 * np.sin(angles)
    ).astype(np.float32)[None, :]
    outer_x = (
        47.5 + normalized_radius * 1.25 * np.cos(angles)
    ).astype(np.float32)[None, :]
    outer_y = (
        47.5 + normalized_radius * 1.25 * np.sin(angles)
    ).astype(np.float32)[None, :]
    inner_values = cv2.remap(
        normalized,
        inner_x,
        inner_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    outer_values = cv2.remap(
        normalized,
        outer_x,
        outer_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    radial_edges = inner_values - outer_values
    edge_score = float(np.mean(np.abs(radial_edges))) / local_scale
    positive_fraction = float(np.mean(radial_edges > 0))
    score = max(0.0, contrast) * 0.65 + edge_score * 0.35
    return VisualSupport(
        id=circle.id,
        score=round(score, 6),
        contrast_score=round(contrast, 6),
        edge_score=round(edge_score, 6),
        positive_edge_fraction=round(positive_fraction, 6),
        x=circle.x,
        y=circle.y,
        radius=circle.radius,
    )


def rank_weak_visual_support(
    image: NDArray[np.uint8],
    circles: Iterable[CircleFeature],
    *,
    limit: int = 5,
) -> list[VisualSupport]:
    """Rank review suggestions without deleting or mutating annotations."""

    if image.ndim == 2:
        luminance = image.astype(np.float32)
    else:
        bgr = (
            cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            if image.shape[2] == 4
            else image
        )
        luminance = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[
            :, :, 0
        ].astype(np.float32)
    ranked = sorted(
        (
            _circle_visual_support(luminance, circle)
            for circle in circles
        ),
        key=lambda item: (item.score, item.id),
    )
    return ranked[: max(1, int(limit))]
