from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np

from src.feature_table import AnnotationSet, CircleFeature
from src.structure_analysis import (
    boundary_candidates,
    center_candidate,
    duplicate_candidates,
    local_side_candidates,
    satellite_candidates,
    structural_report,
    structure_summary,
    write_structure_exports,
)
from src.webapp import create_app


def circle(
    circle_id: str,
    x: float,
    y: float,
    radius: float = 5,
    **values: object,
) -> CircleFeature:
    return CircleFeature(
        id=circle_id,
        x=x,
        y=y,
        radius=radius,
        **values,
    )


def test_summary_keeps_actual_counts_and_only_reports_targets() -> None:
    circles: list[CircleFeature] = [
        circle("hub", 0, 0, 10, circle_type="main")
    ]
    for arm_id, count in {1: 72, 2: 68, 3: 68, 4: 68, 5: 68, 6: 64}.items():
        circles.extend(
            circle(
                f"k{arm_id}-{index}",
                arm_id * 100 + index,
                index,
                arm_id=arm_id,
                circle_type="main",
            )
            for index in range(count)
        )

    summary = structure_summary(circles)

    assert summary["total_circles"] == 409
    assert summary["arms"]["1"]["total"] == 72
    assert summary["arms"]["1"]["target_difference"] == 4
    assert summary["arms"]["6"]["total"] == 64
    assert summary["arms"]["6"]["target_difference"] == -4
    assert summary["validation"]["total_matches"] is True
    assert summary["validation"]["all_arms_have_68"] is False
    # Counting does not rewrite the inaccurate legacy labels.
    assert all(item.circle_type == "main" for item in circles)


def test_center_and_boundary_results_are_suggestions_and_respect_locks() -> None:
    hub = circle("hub", 0.4, -0.2, 12, circle_type="main")
    near_k6 = circle(
        "near-k6",
        100 * math.cos(math.radians(-50)),
        -100 * math.sin(math.radians(-50)),
        arm_id=1,
        circle_type="unclassified",
    )
    current_k1 = circle(
        "current-k1",
        100,
        0,
        arm_id=1,
        circle_type="unclassified",
    )
    circles = [hub, near_k6, current_k1]

    center_result = center_candidate(circles, (0, 0))
    boundaries = boundary_candidates(
        circles,
        (0, 0),
        0,
        manual_locks={"near-k6": ["arm_id"]},
    )

    assert center_result is not None
    assert center_result["id"] == "hub"
    assert hub.circle_type == "main"
    candidate = next(item for item in boundaries if item["id"] == "near-k6")
    assert candidate["suggested_arm_id"] == 6
    assert candidate["difference_score"] > 0
    assert candidate["manual_locked"] is True
    assert near_k6.arm_id == 1


def test_local_side_candidates_classify_spiral_normal_without_mutation() -> None:
    center = (0.0, 0.0)
    radius = 100.0
    twist = 30.0
    angle = math.radians(twist)
    angular_rate = math.radians(twist / 100.0)
    tangent = (
        math.cos(angle) - radius * math.sin(angle) * angular_rate,
        -math.sin(angle) - radius * math.cos(angle) * angular_rate,
    )
    tangent_length = math.hypot(*tangent)
    tangent = (tangent[0] / tangent_length, tangent[1] / tangent_length)
    outward_normal = (-tangent[1], tangent[0])
    radial_unit = (math.cos(angle), -math.sin(angle))
    if (
        outward_normal[0] * radial_unit[0]
        + outward_normal[1] * radial_unit[1]
    ) < 0:
        outward_normal = (-outward_normal[0], -outward_normal[1])
    spine = (radius * math.cos(angle), -radius * math.sin(angle))
    centerline = circle(
        "centerline",
        *spine,
        radius=5,
        arm_id=1,
        side="left",
    )
    outer = circle(
        "outer",
        spine[0] + 18 * outward_normal[0],
        spine[1] + 18 * outward_normal[1],
        radius=5,
        arm_id=1,
        side="right",
    )
    inner = circle(
        "inner",
        spine[0] - 18 * outward_normal[0],
        spine[1] - 18 * outward_normal[1],
        radius=5,
        arm_id=1,
        side="left",
    )

    candidates = local_side_candidates(
        [centerline, outer, inner],
        center,
        phase_degrees=0,
        model="spiral",
        twist_degrees_per_100px=twist,
    )
    by_id = {item["id"]: item for item in candidates}

    assert by_id["centerline"]["suggested_side"] == "centerline"
    assert by_id["outer"]["suggested_side"] == "outer"
    assert by_id["inner"]["suggested_side"] == "inner"
    assert by_id["outer"]["signed_radial_offset"] > 0
    assert by_id["inner"]["signed_radial_offset"] < 0
    assert by_id["outer"]["confidence"] >= 0.6
    assert by_id["inner"]["confidence"] >= 0.6
    assert all(item["method"] == "spiral_model" for item in candidates)
    # Suggestions do not rewrite the confirmed/legacy value.
    assert outer.side == "right"
    assert inner.side == "left"


def test_local_side_candidates_prefer_main_chain_and_respect_side_lock() -> None:
    chain = [
        circle("main-1", 20, 20, arm_id=1),
        circle("main-2", 60, 30, arm_id=1),
        circle("main-3", 100, 55, arm_id=1),
    ]
    candidate = circle(
        "candidate",
        62,
        48,
        arm_id=1,
        side="left",
        locked_fields=["side"],
    )

    rows = local_side_candidates(
        [*chain, candidate],
        (0, 0),
        phase_degrees=0,
        model="radial",
        structure_state={
            "main_chains": {"1": [item.id for item in chain]}
        },
    )
    item = next(row for row in rows if row["id"] == "candidate")

    assert item["method"] == "main_chain"
    assert item["suggested_side"] in {
        "inner",
        "outer",
        "centerline",
        "unknown",
    }
    assert item["side_locked"] is True
    assert item["manual_locked"] is True
    assert candidate.side == "left"


def test_radial_arm_does_not_invent_inner_outer_side() -> None:
    off_axis = circle(
        "off-axis",
        100,
        20,
        radius=4,
        arm_id=1,
        side="left",
    )

    item = local_side_candidates(
        [off_axis],
        (0, 0),
        phase_degrees=0,
        model="radial",
    )[0]

    assert item["method"] == "radial_model"
    assert item["suggested_side"] == "unknown"
    assert item["confidence"] < 0.5
    assert off_axis.side == "left"


def test_local_side_report_covers_every_circle_and_marks_center() -> None:
    hub = circle("hub", 0.2, 0.1, radius=10, circle_type="main")
    assigned = circle("assigned", 50, 0, arm_id=1, side="right")
    unassigned = circle("unassigned", 20, 20, arm_id=None, side=None)

    report = structural_report(
        [hub, assigned, unassigned],
        (0, 0),
        phase_degrees=0,
        model="spiral",
        twist_degrees_per_100px=20,
    )
    by_id = {
        item["id"]: item for item in report["local_side_candidates"]
    }

    assert len(by_id) == 3
    assert by_id["hub"]["suggested_side"] == "center"
    assert by_id["unassigned"]["suggested_side"] == "unknown"
    assert report["suggestions"]["hub"]["suggested_side"] == "center"
    assert report["suggestions"]["assigned"]["suggested_side"] in {
        "inner",
        "outer",
        "centerline",
        "unknown",
    }


def test_local_side_round_trip_keeps_legacy_left_right_value() -> None:
    item = circle("legacy", 10, 10, side="left")
    assert item.legacy_side == "left"
    item.side = "inner"
    item.locked_fields = ["side"]
    item.classification_source = "user_confirmed"

    restored = AnnotationSet.from_dict(
        AnnotationSet(circles=[item]).to_dict()
    ).circles[0]

    assert restored.side == "inner"
    assert restored.legacy_side == "left"
    assert restored.locked_fields == ["side"]


def test_parent_geometry_uses_only_explicit_main_chain() -> None:
    center = (0.0, 0.0)
    mains = [
        circle(
            f"main-{index}",
            20.0 * index,
            0,
            radius=10,
            circle_type="main",
            arm_id=1,
            sequence_index=index,
            classification_source="user_confirmed",
        )
        for index in range(1, 14)
    ]
    satellite = circle(
        "satellite",
        121,
        12,
        radius=3,
        circle_type="unclassified",
        arm_id=1,
    )
    # This legacy-main record must not silently become part of the chain.
    legacy = circle(
        "legacy-main",
        121,
        11,
        radius=6,
        circle_type="main",
        arm_id=1,
        sequence_index=6,
    )
    structure_state = {
        "main_chains": {"1": [item.id for item in mains]}
    }

    candidates = satellite_candidates(
        [*mains, satellite, legacy],
        center,
        structure_state=structure_state,
    )
    by_id = {item["id"]: item for item in candidates}

    assert by_id["satellite"]["nearest_main_id"] == "main-6"
    assert by_id["satellite"]["second_nearest_main_id"] in {
        "main-5",
        "main-7",
    }
    assert by_id["satellite"]["normalized_parent_distance"] > 1
    assert by_id["satellite"]["suggested_side"] in {
        "inner",
        "outer",
        "centerline",
    }
    assert by_id["legacy-main"]["suggested_circle_type"] == "satellite"


def test_duplicate_priority_prefers_locked_manual_override() -> None:
    automatic = circle(
        "auto",
        10,
        10,
        radius=10,
        source="automatic",
    )
    manual = circle(
        "manual",
        10.5,
        10.2,
        radius=10.2,
        source="manual_override",
        locked=True,
    )

    candidates = duplicate_candidates([automatic, manual])

    assert len(candidates) == 1
    assert candidates[0]["overlap_ratio"] > 0.9
    assert candidates[0]["preferred_id"] == "manual"
    assert candidates[0]["merge_candidate_id"] == "auto"


def test_structure_exports_have_requested_names_and_columns(tmp_path) -> None:
    center = circle("center", 0, 0, 12, circle_type="center")
    mains = [
        circle(
            f"main-{index}",
            index * 20,
            0,
            radius=8,
            circle_type="main",
            arm_id=1,
            sequence_index=index,
            classification_source="user_confirmed",
            locked_fields=["circle_type", "arm_id", "sequence_index"],
        )
        for index in range(1, 14)
    ]
    satellite = circle(
        "satellite",
        22,
        8,
        circle_type="satellite",
        arm_id=1,
        parent_id="main-1",
        side="outer",
        local_angle=15,
        normalized_parent_distance=1.1,
    )
    circles = [center, *mains, satellite]
    exported = write_structure_exports(
        circles,
        tmp_path,
        center=(0, 0),
        phase_degrees=0,
        structure_state={
            "main_chains": {"1": [item.id for item in mains]}
        },
    )

    assert {path.name for path in exported.files.values()} == {
        "structure_summary.json",
        "circles_classified.csv",
        "arm_summary.csv",
        "main_sequence.csv",
        "satellite_relationships.csv",
        "boundary_candidates.csv",
    }
    summary = json.loads((tmp_path / "structure_summary.json").read_text())
    assert summary["total_circles"] == len(circles)
    with (tmp_path / "circles_classified.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        header = next(csv.reader(handle))
    for field in (
        "circle_type",
        "arm_id",
        "sequence_index",
        "parent_id",
        "side",
        "legacy_side",
        "local_angle",
        "normalized_parent_distance",
        "classification_source",
        "classification_confidence",
        "locked_fields",
    ):
        assert field in header


def test_annotation_preserves_chains_history_and_rejects_cross_arm_repeat() -> None:
    circles = [
        circle(
            "main",
            10,
            10,
            circle_type="main",
            arm_id=1,
            sequence_index=1,
            locked_fields=["circle_type"],
        )
    ]
    document = AnnotationSet(
        circles=circles,
        structure_state={
            "main_chains": {"1": ["main"]},
            "main_chain_draft": {"arm": 2, "ids": ["main"]},
        },
        edit_history=[
            {
                "circle_id": "main",
                "field": "circle_type",
                "previous_value": "unclassified",
                "new_value": "main",
                "actor": "user",
                "timestamp": "2026-07-25T20:00:00Z",
                "operation_type": "confirm_main_chain",
            }
        ],
    )

    restored = AnnotationSet.from_dict(document.to_dict())

    assert restored.structure_state["main_chains"]["1"] == ["main"]
    assert restored.structure_state["main_chain_draft"] == {
        "arm": 2,
        "ids": ["main"],
    }
    assert restored.edit_history[0]["operation_type"] == "confirm_main_chain"
    assert restored.circles[0].locked_fields == ["circle_type"]


def _encoded_image() -> bytes:
    image = np.full((80, 80, 3), 80, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def test_index_exposes_structural_review_controls(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "DATA_ROOT": tmp_path / "data",
            "OUTPUT_ROOT": tmp_path / "outputs",
        }
    )
    html = app.test_client().get("/").get_data(as_text=True)

    assert "Kol Denetimi" in html
    for control_id in (
        "refreshStructureButton",
        "structureSummaryBody",
        "clearAllCircleLocks",
        "structureLockSummary",
        "approveCenterCandidate",
        "boundaryTopFourOnly",
        "boundaryCandidatesBody",
        "localSideCandidatesBody",
        "localSideOnlyLowConfidence",
        "selectSafeLocalSides",
        "applySelectedLocalSides",
        "mainChainArm",
        "confirmMainChain",
        "satelliteCandidatesBody",
        "satelliteStatusBody",
        "duplicateCandidatesBody",
        "undoStructureButton",
        "labelMode",
    ):
        assert f'id="{control_id}"' in html
    for value in ("center", "main", "satellite", "unclassified"):
        assert f'<option value="{value}">' in html


def test_structure_api_does_not_change_confirmed_fields(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "DATA_ROOT": tmp_path / "data",
            "OUTPUT_ROOT": tmp_path / "outputs",
        }
    )
    client = app.test_client()
    record = client.post(
        "/api/upload",
        data={"image": (io.BytesIO(_encoded_image()), "fixture.png")},
        content_type="multipart/form-data",
    ).get_json()
    raw_circles = [
        {
            "id": "hub",
            "x": 40,
            "y": 40,
            "radius": 8,
            "circle_type": "main",
            "arm_id": None,
            "source": "manual_override",
        },
        {
            "id": "k1",
            "x": 70,
            "y": 40,
            "radius": 5,
            "circle_type": "main",
            "arm_id": 1,
            "source": "manual_override",
        },
    ]

    response = client.post(
        "/api/structure/analyze",
        json={
            "image_id": record["image_id"],
            "circles": raw_circles,
            "center": {"x": 40, "y": 40},
            "phase_degrees": 0,
            "manual_locks": {"k1": ["arm_id"]},
            "structure_state": {"main_chains": {}},
        },
    )

    assert response.status_code == 200
    report = response.get_json()
    assert report["center_candidate"]["id"] == "hub"
    assert report["suggestions"]["hub"]["suggested_circle_type"] == "center"
    # API output contains suggestions, not a rewritten circles array.
    assert "circles" not in report


def test_export_api_writes_all_structural_artifacts_without_reassigning_arm(
    tmp_path,
) -> None:
    app = create_app(
        {
            "TESTING": True,
            "DATA_ROOT": tmp_path / "data",
            "OUTPUT_ROOT": tmp_path / "outputs",
        }
    )
    client = app.test_client()
    record = client.post(
        "/api/upload",
        data={"image": (io.BytesIO(_encoded_image()), "fixture.png")},
        content_type="multipart/form-data",
    ).get_json()
    response = client.post(
        "/api/export",
        json={
            "image_id": record["image_id"],
            "center": {"x": 40, "y": 40},
            "phase_degrees": 0,
            "circles": [
                {
                    "id": "kept",
                    "x": 70,
                    "y": 40,
                    "radius": 5,
                    "circle_type": "unclassified",
                    "arm_id": 4,
                    "source": "manual_override",
                }
            ],
            "structure_state": {"main_chains": {}},
            "edit_history": [],
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["circles"][0]["arm_id"] == 4
    expected = {
        "structure_summary": "structure_summary.json",
        "circles_classified": "circles_classified.csv",
        "arm_summary": "arm_summary.csv",
        "main_sequence": "main_sequence.csv",
        "satellite_relationships": "satellite_relationships.csv",
        "boundary_candidates": "boundary_candidates.csv",
    }
    for key, filename in expected.items():
        assert payload["files"][key]["name"] == filename
        download = client.get(payload["files"][key]["url"])
        assert download.status_code == 200
        assert download.data


def test_real_milk_hill_work_file_keeps_409_and_expected_deviation() -> None:
    document = AnnotationSet.from_dict(
        json.loads(
            (
                Path(__file__).parents[1]
                / "data"
                / "annotations"
                / "Milk_Hill_Galaxy_Spiral_2001-20260725-220610-685899.work.json"
            ).read_text(encoding="utf-8")
        )
    )
    report = structural_report(
        document.circles,
        document.general_center or (0, 0),
        document.arm_phase_degrees,
        model=document.arm_model,
        twist_degrees_per_100px=document.arm_twist_degrees_per_100px,
        manual_locks=document.manual_locks,
        structure_state=document.structure_state,
    )

    assert report["summary"]["total_circles"] == 409
    assert report["summary"]["arms"]["1"]["total"] == 72
    assert report["summary"]["arms"]["6"]["total"] == 64
    assert report["center_candidate"]["id"] == "auto-0132"
    assert report["boundary_candidates"]
