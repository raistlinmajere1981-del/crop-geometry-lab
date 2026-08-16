"""Typed circle annotations and portable CSV/JSON serialization."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

CircleType = Literal[
    "center",
    "main",
    "satellite",
    "unclassified",
    "side",
]
CircleSide = Literal[
    "inner",
    "outer",
    "centerline",
    "unknown",
    "center",
    "left",
    "right",
]
CircleSource = Literal["automatic", "manual", "manual_override"]
CircleSizeClass = Literal["small", "medium", "large"]

STRUCTURAL_LOCK_FIELDS = frozenset(
    {
        "circle_type",
        "arm_id",
        "sequence_index",
        "parent_id",
        "side",
    }
)

REQUIRED_EXPORT_FIELDS = (
    "id",
    "x",
    "y",
    "radius",
    "confidence",
    "circle_type",
    "parent_id",
    "arm_id",
    "sequence_index",
    "side",
)
COORDINATE_EXPORT_FIELDS = (
    "processed_x",
    "processed_y",
    "source_x",
    "source_y",
    "source_radius",
    "corrected_x",
    "corrected_y",
    "corrected_radius",
    "corrected_major_axis",
    "corrected_minor_axis",
    "source_major_axis",
    "source_minor_axis",
)
CONFIDENCE_EXPORT_FIELDS = (
    "edge_score",
    "color_score",
    "circularity_score",
    "detector_agreement",
    "symmetry_support",
    "final_confidence",
)
DERIVED_EXPORT_FIELDS = ("center_distance", "center_angle")
STRUCTURE_EXPORT_FIELDS = (
    "legacy_side",
    "local_angle",
    "normalized_parent_distance",
    "classification_source",
    "classification_confidence",
    "locked_fields",
    "suggested_circle_type",
    "suggested_arm_id",
    "suggested_sequence_index",
    "suggested_parent_id",
    "suggested_side",
    "suggestion_confidence",
)
APPEARANCE_EXPORT_FIELDS = (
    "stroke_color",
    "stroke_width",
    "fill_enabled",
    "fill_color",
    "fill_opacity",
)
CSV_EXPORT_FIELDS = (
    REQUIRED_EXPORT_FIELDS
    + COORDINATE_EXPORT_FIELDS
    + CONFIDENCE_EXPORT_FIELDS
    + DERIVED_EXPORT_FIELDS
    + STRUCTURE_EXPORT_FIELDS
    + APPEARANCE_EXPORT_FIELDS
    + ("size_class", "detector", "locked", "source")
)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Floating-point values must be finite")
    return number


def _optional_score(value: Any) -> float | None:
    number = _optional_float(value)
    if number is None:
        return None
    return min(1.0, max(0.0, number))


def _optional_color(value: Any) -> str | None:
    if value is None or value == "":
        return None
    color = str(value).strip().lower()
    if (
        len(color) != 7
        or not color.startswith("#")
        or any(character not in "0123456789abcdef" for character in color[1:])
    ):
        raise ValueError("Colors must use #RRGGBB format")
    return color


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in {None, "", 0, "0", "false", "False", "no", "No"}:
        return False
    return True


def _string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = [
                item.strip()
                for item in value.strip("[]").split(",")
                if item.strip()
            ]
        value = decoded
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("locked_fields must be a list of field names")
    return list(dict.fromkeys(str(item) for item in value))


@dataclass(slots=True)
class CircleFeature:
    """One editable circle with processed and original-source coordinates."""

    id: str
    x: float
    y: float
    radius: float
    confidence: float = 1.0
    circle_type: CircleType = "unclassified"
    parent_id: str | None = None
    arm_id: int | None = None
    sequence_index: int | None = None
    side: CircleSide | None = None
    center_distance: float | None = None
    center_angle: float | None = None
    legacy_side: str | None = None
    local_angle: float | None = None
    normalized_parent_distance: float | None = None
    classification_source: str = "legacy"
    classification_confidence: float = 0.0
    locked_fields: list[str] = field(default_factory=list)
    suggested_circle_type: CircleType | None = None
    suggested_arm_id: int | None = None
    suggested_sequence_index: int | None = None
    suggested_parent_id: str | None = None
    suggested_side: CircleSide | None = None
    suggestion_confidence: float | None = None
    processed_x: float | None = None
    processed_y: float | None = None
    source_x: float | None = None
    source_y: float | None = None
    source_radius: float | None = None
    corrected_x: float | None = None
    corrected_y: float | None = None
    corrected_radius: float | None = None
    corrected_major_axis: float | None = None
    corrected_minor_axis: float | None = None
    source_major_axis: float | None = None
    source_minor_axis: float | None = None
    edge_score: float | None = None
    color_score: float | None = None
    circularity_score: float | None = None
    detector_agreement: float | None = None
    symmetry_support: float | None = None
    final_confidence: float | None = None
    stroke_color: str | None = None
    stroke_width: float | None = None
    fill_enabled: bool = False
    fill_color: str | None = None
    fill_opacity: float = 0.2
    size_class: CircleSizeClass | None = None
    detector: str | None = None
    locked: bool = False
    source: CircleSource = "manual"

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        self.x = float(self.x)
        self.y = float(self.y)
        self.radius = float(self.radius)
        self.confidence = float(self.confidence)
        self.processed_x = self.x if self.processed_x is None else float(self.processed_x)
        self.processed_y = self.y if self.processed_y is None else float(self.processed_y)
        self.source_x = self.x if self.source_x is None else float(self.source_x)
        self.source_y = self.y if self.source_y is None else float(self.source_y)
        self.source_radius = (
            None
            if self.source_radius is None
            else float(self.source_radius)
        )
        self.corrected_x = (
            self.processed_x
            if self.corrected_x is None
            else float(self.corrected_x)
        )
        self.corrected_y = (
            self.processed_y
            if self.corrected_y is None
            else float(self.corrected_y)
        )
        self.corrected_radius = (
            self.radius
            if self.corrected_radius is None
            else float(self.corrected_radius)
        )
        for field_name in (
            "corrected_major_axis",
            "corrected_minor_axis",
            "source_major_axis",
            "source_minor_axis",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, float(value))
        if not self.id:
            raise ValueError("Circle id cannot be empty")
        if not all(
            math.isfinite(value)
            for value in (
                self.x,
                self.y,
                self.radius,
                self.processed_x,
                self.processed_y,
                self.source_x,
                self.source_y,
                self.corrected_x,
                self.corrected_y,
                self.corrected_radius,
            )
        ):
            raise ValueError("Circle geometry must contain finite values")
        if (
            self.radius <= 0
            or self.corrected_radius <= 0
            or (
                self.source_radius is not None
                and self.source_radius <= 0
            )
        ):
            raise ValueError("Circle radii must be positive")
        for field_name in (
            "corrected_major_axis",
            "corrected_minor_axis",
            "source_major_axis",
            "source_minor_axis",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{field_name} must be positive and finite")
        if not math.isfinite(self.confidence):
            raise ValueError("Circle confidence must be finite")
        self.confidence = min(1.0, max(0.0, self.confidence))
        score_fields = (
            "edge_score",
            "color_score",
            "circularity_score",
            "detector_agreement",
            "symmetry_support",
        )
        for field_name in score_fields:
            value = getattr(self, field_name)
            if value is not None:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"{field_name} must be finite")
                setattr(self, field_name, min(1.0, max(0.0, number)))
        if self.final_confidence is None:
            self.final_confidence = self.confidence
        else:
            self.final_confidence = min(
                1.0, max(0.0, float(self.final_confidence))
            )
            self.confidence = self.final_confidence
        self.stroke_color = _optional_color(self.stroke_color)
        self.fill_color = _optional_color(self.fill_color)
        if self.stroke_width is not None:
            self.stroke_width = float(self.stroke_width)
            if not math.isfinite(self.stroke_width) or self.stroke_width <= 0:
                raise ValueError("stroke_width must be a positive finite value")
        self.fill_enabled = _boolean(self.fill_enabled)
        self.fill_opacity = float(self.fill_opacity)
        if not math.isfinite(self.fill_opacity):
            raise ValueError("fill_opacity must be finite")
        self.fill_opacity = min(1.0, max(0.0, self.fill_opacity))
        if self.circle_type not in {
            "center",
            "main",
            "satellite",
            "unclassified",
            "side",
        }:
            raise ValueError(
                "circle_type must be center, main, satellite, unclassified, "
                "or the legacy side value"
            )
        if self.arm_id is not None and not 1 <= int(self.arm_id) <= 6:
            raise ValueError("arm_id must be between 1 and 6")
        if self.sequence_index is not None and int(self.sequence_index) < 0:
            raise ValueError("sequence_index cannot be negative")
        allowed_sides = {
            None,
            "inner",
            "outer",
            "centerline",
            "unknown",
            "center",
            "left",
            "right",
        }
        if self.side not in allowed_sides:
            raise ValueError(
                "side must be inner, outer, centerline, unknown, center, "
                "a legacy left/right value, or null"
            )
        if self.legacy_side is not None:
            self.legacy_side = str(self.legacy_side)
        elif self.side in {"left", "right"}:
            self.legacy_side = self.side
        self.local_angle = _optional_float(self.local_angle)
        self.normalized_parent_distance = _optional_float(
            self.normalized_parent_distance
        )
        if (
            self.normalized_parent_distance is not None
            and self.normalized_parent_distance < 0
        ):
            raise ValueError("normalized_parent_distance cannot be negative")
        self.classification_source = str(
            self.classification_source or "legacy"
        )
        self.classification_confidence = min(
            1.0, max(0.0, float(self.classification_confidence))
        )
        self.locked_fields = _string_list(self.locked_fields)
        if any(
            field_name not in STRUCTURAL_LOCK_FIELDS
            for field_name in self.locked_fields
        ):
            raise ValueError("locked_fields contains an unsupported field")
        if self.suggested_circle_type not in {
            None,
            "center",
            "main",
            "satellite",
            "unclassified",
            "side",
        }:
            raise ValueError("suggested_circle_type is invalid")
        if (
            self.suggested_arm_id is not None
            and not 1 <= int(self.suggested_arm_id) <= 6
        ):
            raise ValueError("suggested_arm_id must be between 1 and 6")
        if (
            self.suggested_sequence_index is not None
            and int(self.suggested_sequence_index) < 0
        ):
            raise ValueError("suggested_sequence_index cannot be negative")
        if self.suggested_side not in allowed_sides:
            raise ValueError("suggested_side is invalid")
        self.suggestion_confidence = _optional_score(
            self.suggestion_confidence
        )
        if self.circle_type == "center":
            self.arm_id = None
            self.sequence_index = None
            self.parent_id = None
            self.side = "center"
        if self.size_class not in {None, "small", "medium", "large"}:
            raise ValueError("size_class must be small, medium, large, or null")
        if self.source not in {"automatic", "manual", "manual_override"}:
            raise ValueError("Unknown circle source")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "CircleFeature":
        """Build a validated feature from API, JSON, or CSV data."""

        return cls(
            id=str(row["id"]),
            x=float(row["x"]),
            y=float(row["y"]),
            radius=float(row["radius"]),
            confidence=float(row.get("confidence", 1.0)),
            circle_type=str(row.get("circle_type") or "main"),  # type: ignore[arg-type]
            parent_id=(
                None
                if row.get("parent_id") in {None, ""}
                else str(row.get("parent_id"))
            ),
            arm_id=_optional_int(row.get("arm_id")),
            sequence_index=_optional_int(row.get("sequence_index")),
            side=(
                None if row.get("side") in {None, ""} else str(row.get("side"))
            ),  # type: ignore[arg-type]
            center_distance=_optional_float(row.get("center_distance")),
            center_angle=_optional_float(row.get("center_angle")),
            legacy_side=(
                None
                if row.get("legacy_side") in {None, ""}
                else str(row.get("legacy_side"))
            ),
            local_angle=_optional_float(row.get("local_angle")),
            normalized_parent_distance=_optional_float(
                row.get("normalized_parent_distance")
            ),
            classification_source=str(
                row.get("classification_source") or "legacy"
            ),
            classification_confidence=(
                0.0
                if row.get("classification_confidence") in {None, ""}
                else float(row["classification_confidence"])
            ),
            locked_fields=_string_list(row.get("locked_fields")),
            suggested_circle_type=(
                None
                if row.get("suggested_circle_type") in {None, ""}
                else str(row.get("suggested_circle_type"))
            ),  # type: ignore[arg-type]
            suggested_arm_id=_optional_int(row.get("suggested_arm_id")),
            suggested_sequence_index=_optional_int(
                row.get("suggested_sequence_index")
            ),
            suggested_parent_id=(
                None
                if row.get("suggested_parent_id") in {None, ""}
                else str(row.get("suggested_parent_id"))
            ),
            suggested_side=(
                None
                if row.get("suggested_side") in {None, ""}
                else str(row.get("suggested_side"))
            ),  # type: ignore[arg-type]
            suggestion_confidence=_optional_score(
                row.get("suggestion_confidence")
            ),
            processed_x=_optional_float(row.get("processed_x")),
            processed_y=_optional_float(row.get("processed_y")),
            source_x=_optional_float(row.get("source_x")),
            source_y=_optional_float(row.get("source_y")),
            source_radius=_optional_float(row.get("source_radius")),
            corrected_x=_optional_float(row.get("corrected_x")),
            corrected_y=_optional_float(row.get("corrected_y")),
            corrected_radius=_optional_float(row.get("corrected_radius")),
            corrected_major_axis=_optional_float(
                row.get("corrected_major_axis")
            ),
            corrected_minor_axis=_optional_float(
                row.get("corrected_minor_axis")
            ),
            source_major_axis=_optional_float(row.get("source_major_axis")),
            source_minor_axis=_optional_float(row.get("source_minor_axis")),
            edge_score=_optional_score(row.get("edge_score")),
            color_score=_optional_score(row.get("color_score")),
            circularity_score=_optional_score(row.get("circularity_score")),
            detector_agreement=_optional_score(row.get("detector_agreement")),
            symmetry_support=_optional_score(row.get("symmetry_support")),
            final_confidence=_optional_score(row.get("final_confidence")),
            stroke_color=_optional_color(row.get("stroke_color")),
            stroke_width=_optional_float(row.get("stroke_width")),
            fill_enabled=_boolean(row.get("fill_enabled")),
            fill_color=_optional_color(row.get("fill_color")),
            fill_opacity=(
                0.2
                if row.get("fill_opacity") in {None, ""}
                else float(row["fill_opacity"])
            ),
            size_class=(
                None
                if row.get("size_class") in {None, ""}
                else str(row.get("size_class"))
            ),  # type: ignore[arg-type]
            detector=(
                None
                if row.get("detector") in {None, ""}
                else str(row.get("detector"))
            ),
            locked=_boolean(row.get("locked")),
            source=str(row.get("source") or "manual"),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        """Return stable, JSON-safe field ordering."""

        raw = asdict(self)
        return {name: raw[name] for name in CSV_EXPORT_FIELDS}


@dataclass(slots=True)
class AnnotationSet:
    """An exported annotation document with analysis context."""

    circles: list[CircleFeature] = field(default_factory=list)
    image_id: str | None = None
    image_name: str | None = None
    coordinate_system: str = "source_and_corrected_image_pixels"
    general_center: tuple[float, float] | None = None
    arm_phase_degrees: float = 0.0
    arm_model: str = "radial"
    arm_twist_degrees_per_100px: float = 0.0
    manual_locks: dict[str, list[str]] = field(default_factory=dict)
    deleted_automatic_ids: list[str] = field(default_factory=list)
    suppressed_automatic_geometry: list[dict[str, Any]] = field(
        default_factory=list
    )
    image_symmetry: list[dict[str, Any]] = field(default_factory=list)
    geometric_symmetry: list[dict[str, Any]] = field(default_factory=list)
    source_to_processed: list[list[float]] | None = None
    processed_to_source: list[list[float]] | None = None
    preprocess_profile: dict[str, Any] | None = None
    edit_history: list[dict[str, Any]] = field(default_factory=list)
    structure_state: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.4"

    def validate(self) -> None:
        ids = [circle.id for circle in self.circles]
        if len(ids) != len(set(ids)):
            raise ValueError("Circle ids must be unique")
        valid_ids = set(ids)
        for circle in self.circles:
            if circle.parent_id is not None and circle.parent_id not in valid_ids:
                raise ValueError(
                    f"Circle {circle.id!r} refers to unknown parent {circle.parent_id!r}"
                )
            if circle.parent_id == circle.id:
                raise ValueError(f"Circle {circle.id!r} cannot be its own parent")
        if self.general_center is not None:
            if len(self.general_center) != 2 or not all(
                math.isfinite(float(value)) for value in self.general_center
            ):
                raise ValueError("general_center must contain two finite values")
        if self.arm_model not in {"radial", "spiral"}:
            raise ValueError("arm_model must be radial or spiral")
        if not math.isfinite(float(self.arm_phase_degrees)):
            raise ValueError("arm_phase_degrees must be finite")
        if not math.isfinite(float(self.arm_twist_degrees_per_100px)):
            raise ValueError("arm twist must be finite")
        allowed_locks = STRUCTURAL_LOCK_FIELDS
        for circle_id, fields in self.manual_locks.items():
            if circle_id not in valid_ids:
                raise ValueError(
                    f"manual_locks refers to unknown circle {circle_id!r}"
                )
            if any(field_name not in allowed_locks for field_name in fields):
                raise ValueError("manual_locks contains an unsupported field")
        for circle in self.circles:
            combined = list(
                dict.fromkeys(
                    [
                        *circle.locked_fields,
                        *self.manual_locks.get(circle.id, []),
                    ]
                )
            )
            circle.locked_fields = combined
            if combined:
                self.manual_locks[circle.id] = combined
        if not isinstance(self.edit_history, list) or not all(
            isinstance(item, Mapping) for item in self.edit_history
        ):
            raise ValueError("edit_history must contain objects")
        if not isinstance(self.structure_state, Mapping):
            raise ValueError("structure_state must be an object")
        raw_chains = self.structure_state.get("main_chains", {})
        if raw_chains is not None and not isinstance(raw_chains, Mapping):
            raise ValueError("structure_state.main_chains must be an object")
        claimed_main_ids: set[str] = set()
        if isinstance(raw_chains, Mapping):
            for raw_arm_id, raw_ids in raw_chains.items():
                try:
                    arm_id = int(raw_arm_id)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "main chain arm ids must be between 1 and 6"
                    ) from exc
                if not 1 <= arm_id <= 6:
                    raise ValueError(
                        "main chain arm ids must be between 1 and 6"
                    )
                if not isinstance(raw_ids, Sequence) or isinstance(
                    raw_ids, (str, bytes)
                ):
                    raise ValueError("each main chain must be a list")
                chain_ids = [str(circle_id) for circle_id in raw_ids]
                if len(chain_ids) > 13:
                    raise ValueError("a main chain cannot contain more than 13 circles")
                if len(chain_ids) != len(set(chain_ids)):
                    raise ValueError("a main chain cannot repeat a circle")
                if any(circle_id not in valid_ids for circle_id in chain_ids):
                    raise ValueError("a main chain refers to an unknown circle")
                if claimed_main_ids.intersection(chain_ids):
                    raise ValueError(
                        "a circle cannot belong to two different main chains"
                    )
                claimed_main_ids.update(chain_ids)
        raw_draft = self.structure_state.get("main_chain_draft")
        if raw_draft is not None:
            if not isinstance(raw_draft, Mapping):
                raise ValueError(
                    "structure_state.main_chain_draft must be an object or null"
                )
            try:
                draft_arm = int(raw_draft.get("arm"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "main chain draft arm must be between 1 and 6"
                ) from exc
            if not 1 <= draft_arm <= 6:
                raise ValueError(
                    "main chain draft arm must be between 1 and 6"
                )
            raw_draft_ids = raw_draft.get("ids", [])
            if not isinstance(raw_draft_ids, Sequence) or isinstance(
                raw_draft_ids, (str, bytes)
            ):
                raise ValueError("main chain draft ids must be a list")
            draft_ids = [str(circle_id) for circle_id in raw_draft_ids]
            if len(draft_ids) > 13:
                raise ValueError(
                    "a main chain draft cannot contain more than 13 circles"
                )
            if len(draft_ids) != len(set(draft_ids)):
                raise ValueError("a main chain draft cannot repeat a circle")
            if any(circle_id not in valid_ids for circle_id in draft_ids):
                raise ValueError(
                    "a main chain draft refers to an unknown circle"
                )
        self.deleted_automatic_ids = [
            str(circle_id) for circle_id in self.deleted_automatic_ids
        ]
        for geometry in self.suppressed_automatic_geometry:
            if not isinstance(geometry, Mapping):
                raise ValueError(
                    "suppressed automatic geometry must contain objects"
                )
            for field_name in ("x", "y", "radius"):
                if field_name not in geometry:
                    raise ValueError(
                        "suppressed automatic geometry requires x, y, radius"
                    )
                value = float(geometry[field_name])
                if not math.isfinite(value):
                    raise ValueError(
                        "suppressed automatic geometry must be finite"
                    )
                if field_name == "radius" and value <= 0:
                    raise ValueError(
                        "suppressed automatic radius must be positive"
                    )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "image_id": self.image_id,
            "image_name": self.image_name,
            "coordinate_system": self.coordinate_system,
            "general_center": (
                None
                if self.general_center is None
                else {
                    "x": float(self.general_center[0]),
                    "y": float(self.general_center[1]),
                }
            ),
            "arm_phase_degrees": float(self.arm_phase_degrees),
            "arm_model": self.arm_model,
            "arm_twist_degrees_per_100px": float(
                self.arm_twist_degrees_per_100px
            ),
            "manual_locks": self.manual_locks,
            "editor_state": {
                "deleted_automatic_ids": self.deleted_automatic_ids,
                "suppressed_automatic_geometry": (
                    self.suppressed_automatic_geometry
                ),
            },
            "image_symmetry": self.image_symmetry,
            "geometric_symmetry": self.geometric_symmetry,
            "coordinate_transform": {
                "source_to_processed": self.source_to_processed,
                "processed_to_source": self.processed_to_source,
            },
            "preprocess_profile": self.preprocess_profile,
            "edit_history": self.edit_history,
            "structure_state": dict(self.structure_state),
            "circles": [circle.to_dict() for circle in self.circles],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnnotationSet":
        center_data = data.get("general_center")
        center = None
        if isinstance(center_data, Mapping):
            center = (float(center_data["x"]), float(center_data["y"]))
        transform_data = data.get("coordinate_transform")
        source_to_processed = None
        processed_to_source = None
        if isinstance(transform_data, Mapping):
            source_to_processed = transform_data.get("source_to_processed")
            processed_to_source = transform_data.get("processed_to_source")
        editor_state = data.get("editor_state")
        if not isinstance(editor_state, Mapping):
            editor_state = {}
        raw_manual_locks = data.get("manual_locks")
        manual_locks = (
            {
                str(circle_id): [str(field_name) for field_name in fields]
                for circle_id, fields in raw_manual_locks.items()
                if isinstance(fields, Sequence)
                and not isinstance(fields, (str, bytes))
            }
            if isinstance(raw_manual_locks, Mapping)
            else {}
        )
        document = cls(
            circles=[
                CircleFeature.from_mapping(item)
                for item in data.get("circles", [])
            ],
            image_id=data.get("image_id"),
            image_name=data.get("image_name"),
            coordinate_system=str(
                data.get("coordinate_system")
                or "source_and_corrected_image_pixels"
            ),
            general_center=center,
            arm_phase_degrees=float(data.get("arm_phase_degrees", 0.0)),
            arm_model=str(data.get("arm_model") or "radial"),
            arm_twist_degrees_per_100px=float(
                data.get("arm_twist_degrees_per_100px", 0.0)
            ),
            manual_locks=manual_locks,
            deleted_automatic_ids=[
                str(circle_id)
                for circle_id in editor_state.get(
                    "deleted_automatic_ids", []
                )
            ],
            suppressed_automatic_geometry=[
                dict(geometry)
                for geometry in editor_state.get(
                    "suppressed_automatic_geometry", []
                )
                if isinstance(geometry, Mapping)
            ],
            image_symmetry=list(data.get("image_symmetry", [])),
            geometric_symmetry=list(data.get("geometric_symmetry", [])),
            source_to_processed=source_to_processed,
            processed_to_source=processed_to_source,
            preprocess_profile=(
                dict(data["preprocess_profile"])
                if isinstance(data.get("preprocess_profile"), Mapping)
                else None
            ),
            edit_history=[
                dict(item)
                for item in data.get("edit_history", [])
                if isinstance(item, Mapping)
            ],
            structure_state=(
                dict(data["structure_state"])
                if isinstance(data.get("structure_state"), Mapping)
                else {}
            ),
            schema_version=str(data.get("schema_version") or "1.1"),
        )
        document.validate()
        return document


def circles_from_mappings(rows: Iterable[Mapping[str, Any]]) -> list[CircleFeature]:
    return [CircleFeature.from_mapping(row) for row in rows]


def write_json(document: AnnotationSet, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def read_json(path: str | Path) -> AnnotationSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Annotation JSON must contain an object")
    return AnnotationSet.from_dict(data)


def write_csv(circles: Sequence[CircleFeature], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_EXPORT_FIELDS)
        writer.writeheader()
        for circle in circles:
            row = circle.to_dict()
            row["locked_fields"] = json.dumps(
                row["locked_fields"], ensure_ascii=False
            )
            writer.writerow(row)
    return destination


def read_csv(path: str | Path) -> list[CircleFeature]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [CircleFeature.from_mapping(row) for row in csv.DictReader(handle)]
