from __future__ import annotations

from dataclasses import replace

import pytest

from src.feature_table import (
    AnnotationSet,
    CircleFeature,
    read_csv,
    read_json,
    write_csv,
    write_json,
)
from src.manual_editor import ManualOverrideLedger


def test_json_and_csv_round_trip(tmp_path) -> None:
    circles = [
        CircleFeature(
            id="main-1",
            x=12.5,
            y=31.25,
            radius=7.0,
            confidence=0.8,
            circle_type="main",
            arm_id=1,
            sequence_index=1,
            side="center",
            center_distance=25.0,
            center_angle=60.0,
            stroke_color="#ff7043",
            stroke_width=4.5,
            fill_enabled=True,
            fill_color="#8bc34a",
            fill_opacity=0.35,
            source="automatic",
        ),
        CircleFeature(
            id="side-1",
            x=18.0,
            y=35.0,
            radius=3.0,
            circle_type="side",
            parent_id="main-1",
            arm_id=1,
            sequence_index=2,
            side="left",
            source="manual",
        ),
    ]
    document = AnnotationSet(
        circles=circles,
        image_id="image-1",
        image_name="test.webp",
        general_center=(10.0, 10.0),
        arm_model="spiral",
        arm_twist_degrees_per_100px=-28.5,
        manual_locks={"side-1": ["arm_id", "side"]},
        deleted_automatic_ids=["auto-deleted"],
        suppressed_automatic_geometry=[
            {"id": "auto-deleted", "x": 42.0, "y": 43.0, "radius": 5.0}
        ],
    )
    json_path = write_json(document, tmp_path / "annotations.json")
    csv_path = write_csv(circles, tmp_path / "circles.csv")

    loaded_document = read_json(json_path)
    loaded_circles = read_csv(csv_path)

    assert loaded_document.to_dict() == document.to_dict()
    assert [circle.to_dict() for circle in loaded_circles] == [
        circle.to_dict() for circle in circles
    ]
    assert loaded_document.schema_version == "1.4"
    assert loaded_document.arm_model == "spiral"
    assert loaded_document.arm_twist_degrees_per_100px == -28.5
    assert loaded_document.manual_locks == {
        "side-1": ["arm_id", "side"]
    }
    assert loaded_document.deleted_automatic_ids == ["auto-deleted"]
    assert loaded_circles[0].stroke_color == "#ff7043"
    assert loaded_circles[0].stroke_width == 4.5
    assert loaded_circles[0].fill_enabled is True
    assert loaded_circles[0].fill_color == "#8bc34a"
    assert loaded_circles[0].fill_opacity == 0.35
    assert loaded_circles[0].corrected_x == 12.5
    assert loaded_circles[0].corrected_y == 31.25
    assert loaded_circles[0].corrected_radius == 7.0


def test_circle_appearance_validation_and_defaults() -> None:
    circle = CircleFeature(id="default", x=10, y=10, radius=2)
    assert circle.stroke_color is None
    assert circle.stroke_width is None
    assert circle.fill_enabled is False
    assert circle.fill_color is None
    assert circle.fill_opacity == 0.2

    with pytest.raises(ValueError, match="#RRGGBB"):
        CircleFeature(
            id="bad-color",
            x=10,
            y=10,
            radius=2,
            stroke_color="red",
        )


def test_annotation_rejects_missing_parent() -> None:
    document = AnnotationSet(
        circles=[
            CircleFeature(
                id="side",
                x=10,
                y=10,
                radius=2,
                circle_type="side",
                parent_id="missing",
            )
        ]
    )
    with pytest.raises(ValueError, match="unknown parent"):
        document.validate()


def test_manual_decisions_survive_automatic_refresh() -> None:
    original = CircleFeature(
        id="auto-0001",
        x=20,
        y=20,
        radius=6,
        source="automatic",
    )
    deleted = CircleFeature(
        id="auto-0002",
        x=50,
        y=50,
        radius=5,
        source="automatic",
    )
    ledger = ManualOverrideLedger([original, deleted])
    corrected = replace(original, x=24, radius=8)
    ledger.update(corrected)
    ledger.delete(deleted.id)
    ledger.add(
        CircleFeature(id="manual-1", x=70, y=80, radius=4, source="manual")
    )

    # A later detector run disagrees with every user decision.
    ledger.replace_automatic(
        [
            replace(original, x=18, radius=4),
            replace(deleted, confidence=0.95),
            CircleFeature(
                id="auto-0003",
                x=90,
                y=90,
                radius=9,
                source="automatic",
            ),
        ]
    )
    resolved = {circle.id: circle for circle in ledger.resolve()}

    assert resolved["auto-0001"].x == 24
    assert resolved["auto-0001"].radius == 8
    assert resolved["auto-0001"].source == "manual_override"
    assert "auto-0002" not in resolved
    assert resolved["manual-1"].source == "manual"
    assert "auto-0003" in resolved
