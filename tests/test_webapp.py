from __future__ import annotations

import csv
import io
import json
import math

import cv2
import numpy as np
import pytest

from src.webapp import create_app


def encoded_fixture(suffix: str = ".png") -> bytes:
    image = np.full((240, 240, 3), 72, dtype=np.uint8)
    for x, y in [(70, 70), (170, 70), (70, 170), (170, 170)]:
        cv2.circle(image, (x, y), 18, (220, 220, 220), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 18, (20, 20, 20), 2, cv2.LINE_AA)
    ok, buffer = cv2.imencode(suffix, image)
    assert ok
    return buffer.tobytes()


def test_index_exposes_fine_confidence_and_appearance_controls(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "DATA_ROOT": tmp_path / "data",
            "OUTPUT_ROOT": tmp_path / "outputs",
        }
    )
    page = app.test_client().get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert 'id="minFinalConfidence"' in html
    assert 'step="0.01"' in html
    assert 'id="maxCandidates" type="number" value="500"' in html
    for field in (
        "stroke_color",
        "stroke_width",
        "fill_enabled",
        "fill_color",
        "fill_opacity",
    ):
        assert f'data-field="{field}"' in html
    for control_id in (
        "armModel",
        "armTwist",
        "suggestArmModel",
        "reindexArmsButton",
        "inspectWeakCircle",
        "saveProjectButton",
        "projectInput",
        "tramlineExpectedSpacing",
        "tramlineRegularity",
        "mainArmEnabled",
        "mainArmMinRadius",
        "mainArmMaxRadius",
        "mainArmMinConfidence",
        "mainArmHoughThreshold",
        "detectionRangeReadout",
        "undoTramlineEdit",
            "parallelizeTramlines",
            "preservePerspectiveFrame",
            "perspectiveQualityPanel",
            "calculatePerspectiveQuality",
            "exportPerspectiveQuality",
            "qualityCoordinateSpace",
            "qualityViewSideBySide",
            "qualityViewRoundness",
            "qualityViewTramlines",
            "qualityViewHeatmap",
        ):
        assert f'id="{control_id}"' in html
    for tool in ("tramlineEdit", "tramlineAdd", "tramlineDelete"):
        assert f'data-tool="{tool}"' in html
    assert "/static/app.js?v=0.6.0" in html


def test_low_visual_support_api_only_suggests_and_does_not_delete(tmp_path) -> None:
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
        data={"image": (io.BytesIO(encoded_fixture()), "fixture.png")},
        content_type="multipart/form-data",
    ).get_json()
    circles = [
        {
            "id": "visible",
            "x": 70,
            "y": 70,
            "radius": 18,
            "circle_type": "main",
        },
        {
            "id": "unsupported",
            "x": 120,
            "y": 120,
            "radius": 10,
            "circle_type": "main",
        },
    ]

    response = client.post(
        "/api/circles/suspects",
        json={
            "image_id": record["image_id"],
            "circles": circles,
            "limit": 2,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["suggestions"][0]["id"] == "unsupported"
    assert {item["id"] for item in payload["suggestions"]} == {
        "visible",
        "unsupported",
    }
    assert "otomatik silinmez" in payload["notice"]


def test_local_api_upload_analyze_and_export(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "DATA_ROOT": tmp_path / "data",
            "OUTPUT_ROOT": tmp_path / "outputs",
            "SAMPLE_IMAGE": tmp_path / "does-not-exist.jpg",
        }
    )
    client = app.test_client()
    upload_response = client.post(
        "/api/upload",
        data={"image": (io.BytesIO(encoded_fixture()), "fixture.png")},
        content_type="multipart/form-data",
    )
    assert upload_response.status_code == 200
    record = upload_response.get_json()
    assert record["width"] == 240
    assert client.get(record["image_url"]).status_code == 200

    circles = [
        {
            "id": "manual-1",
            "x": 170,
            "y": 120,
            "radius": 18,
            "confidence": 1,
            "circle_type": "main",
            "parent_id": None,
            "arm_id": None,
            "sequence_index": None,
            "side": None,
            "stroke_color": "#ff0000",
            "stroke_width": 4,
            "fill_enabled": True,
            "fill_color": "#00ff00",
            "fill_opacity": 0.25,
            "source": "manual",
        }
    ]
    analysis_response = client.post(
        "/api/analyze",
        json={
            "image_id": record["image_id"],
            "center": {"x": 120, "y": 120},
            "phase_degrees": 0,
            "circles": circles,
        },
    )
    assert analysis_response.status_code == 200
    analysis = analysis_response.get_json()
    assert len(analysis["image_symmetry"]) == 5
    assert analysis["circles"][0]["center_distance"] == 50
    assert analysis["circles"][0]["arm_id"] == 1
    assert analysis["circles"][0]["stroke_color"] == "#ff0000"
    assert analysis["circles"][0]["stroke_width"] == 4
    assert analysis["circles"][0]["fill_enabled"] is True
    assert analysis["circles"][0]["fill_color"] == "#00ff00"
    assert analysis["circles"][0]["fill_opacity"] == 0.25

    export_response = client.post(
        "/api/export",
        json={
            "image_id": record["image_id"],
            "center": {"x": 120, "y": 120},
            "phase_degrees": 0,
            "circles": circles,
        },
    )
    assert export_response.status_code == 200
    exported = export_response.get_json()
    for kind in ("png", "csv", "json"):
        response = client.get(exported["files"][kind]["url"])
        assert response.status_code == 200
        assert response.data
    csv_path = tmp_path / "outputs" / exported["files"]["csv"]["name"]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header[:10] == [
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
    ]
    for field in (
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
        "edge_score",
        "color_score",
        "circularity_score",
        "detector_agreement",
        "symmetry_support",
        "final_confidence",
        "stroke_color",
        "stroke_width",
        "fill_enabled",
        "fill_color",
        "fill_opacity",
        "locked",
    ):
        assert field in header


def test_project_save_and_load_preserves_editor_and_arm_state(tmp_path) -> None:
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
        data={"image": (io.BytesIO(encoded_fixture()), "fixture.png")},
        content_type="multipart/form-data",
    ).get_json()
    circle = {
        "id": "manual-1",
        "x": 170,
        "y": 120,
        "radius": 18,
        "confidence": 1,
        "circle_type": "main",
        "arm_id": 2,
        "side": "center",
        "source": "manual_override",
    }
    response = client.post(
        "/api/project/save",
        json={
            "image_id": record["image_id"],
            "circles": [circle],
            "center": {"x": 120, "y": 120},
            "phase_degrees": 82,
            "arm_model": "spiral",
            "arm_twist_degrees_per_100px": -28.5,
            "manual_locks": {"manual-1": ["arm_id", "side"]},
            "editor_state": {
                "deleted_automatic_ids": ["auto-removed"],
                "suppressed_automatic_geometry": [
                    {
                        "id": "auto-removed",
                        "x": 80,
                        "y": 90,
                        "radius": 7,
                    }
                ],
            },
        },
    )

    assert response.status_code == 200
    saved = response.get_json()
    download = client.get(saved["file"]["url"])
    document = json.loads(download.data)
    assert document["schema_version"] == "1.4"
    assert document["arm_model"] == "spiral"
    assert document["arm_twist_degrees_per_100px"] == -28.5
    assert document["manual_locks"]["manual-1"] == ["arm_id", "side"]
    assert document["editor_state"]["deleted_automatic_ids"] == [
        "auto-removed"
    ]

    loaded_response = client.post(
        "/api/project/load",
        data={
            "image_id": record["image_id"],
            "project": (io.BytesIO(download.data), "saved.work.json"),
        },
        content_type="multipart/form-data",
    )
    assert loaded_response.status_code == 200
    loaded = loaded_response.get_json()
    assert loaded["warning"] is None
    assert loaded["document"]["circles"][0]["id"] == "manual-1"
    assert loaded["document"]["general_center"] == {"x": 120.0, "y": 120.0}
    assert loaded["document"]["manual_locks"]["manual-1"] == [
        "arm_id",
        "side",
    ]


def test_spiral_arm_suggestion_api(tmp_path) -> None:
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
        data={"image": (io.BytesIO(encoded_fixture()), "fixture.png")},
        content_type="multipart/form-data",
    ).get_json()
    center = (120.0, 120.0)
    phase = 18.0
    twist = -26.0
    circles = []
    for arm_index in range(6):
        for sequence, distance in enumerate((30.0, 50.0, 75.0)):
            angle = math.radians(
                phase + arm_index * 60 + twist * distance / 100
            )
            circles.append(
                {
                    "id": f"arm-{arm_index}-{sequence}",
                    "x": center[0] + distance * math.cos(angle),
                    "y": center[1] - distance * math.sin(angle),
                    "radius": 12 - sequence,
                    "confidence": 0.9,
                    "circle_type": "main",
                    "source": "automatic",
                }
            )

    response = client.post(
        "/api/arms/suggest",
        json={
            "image_id": record["image_id"],
            "circles": circles,
            "center": {"x": center[0], "y": center[1]},
            "phase_degrees": phase,
        },
    )

    assert response.status_code == 200
    suggestion = response.get_json()
    assert suggestion["model"] == "spiral"
    assert suggestion["phase_degrees"] == pytest.approx(phase, abs=0.2)
    assert suggestion["twist_degrees_per_100px"] == pytest.approx(
        twist,
        abs=0.2,
    )
    assert suggestion["improvement_fraction"] > 0.8


def test_upload_rejects_unknown_extension(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "DATA_ROOT": tmp_path / "data",
            "OUTPUT_ROOT": tmp_path / "outputs",
        }
    )
    response = app.test_client().post(
        "/api/upload",
        data={"image": (io.BytesIO(encoded_fixture()), "fixture.gif")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "JPG" in response.get_json()["error"]


@pytest.mark.parametrize(
    ("suffix", "filename"),
    [
        (".jpg", "fixture.jpg"),
        (".png", "fixture.png"),
        (".webp", "fixture.webp"),
    ],
)
def test_all_documented_image_formats_are_accepted(
    tmp_path, suffix: str, filename: str
) -> None:
    app = create_app(
        {
            "TESTING": True,
            "DATA_ROOT": tmp_path / suffix.removeprefix(".") / "data",
            "OUTPUT_ROOT": tmp_path / suffix.removeprefix(".") / "outputs",
        }
    )
    response = app.test_client().post(
        "/api/upload",
        data={"image": (io.BytesIO(encoded_fixture(suffix)), filename)},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["width"] == 240
