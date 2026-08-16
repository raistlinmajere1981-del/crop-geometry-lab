from __future__ import annotations

import io
import json

import cv2
import numpy as np

from src.feature_table import CircleFeature
from src.perspective_quality import (
    analyze_perspective_quality,
    measure_tramline_quality,
    measure_warp_penalty,
    write_perspective_quality_exports,
)
from src.preprocessing import transform_points
from src.webapp import create_app


def ideal_six_arm_fixture() -> tuple[np.ndarray, list[CircleFeature], dict]:
    image = np.zeros((720, 720, 3), dtype=np.uint8)
    center = (360.0, 360.0)
    circles: list[CircleFeature] = []
    chains: dict[str, list[str]] = {}
    for arm in range(1, 7):
        angle = np.deg2rad((arm - 1) * 60.0)
        chains[str(arm)] = []
        for sequence in range(1, 14):
            distance = 42.0 + sequence * 19.0
            radius = 7.0 + sequence * 0.18
            x = center[0] + np.cos(angle) * distance
            y = center[1] - np.sin(angle) * distance
            circle_id = f"k{arm}-{sequence:02d}"
            circles.append(
                CircleFeature(
                    id=circle_id,
                    x=x,
                    y=y,
                    radius=radius,
                    source_x=x,
                    source_y=y,
                    source_radius=radius,
                    corrected_x=x,
                    corrected_y=y,
                    corrected_radius=radius,
                    circle_type="main",
                    arm_id=arm,
                    sequence_index=sequence,
                )
            )
            chains[str(arm)].append(circle_id)
            cv2.circle(
                image,
                (int(round(x)), int(round(y))),
                int(round(radius)),
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
    return image, circles, {"main_chains": chains}


def test_tramline_metrics_improve_for_rectified_trapezoid() -> None:
    source_quad = np.float32(
        [[130, 30], [270, 30], [350, 370], [50, 370]]
    )
    destination_quad = np.float32(
        [[70, 30], [330, 30], [330, 370], [70, 370]]
    )
    source_to_corrected = cv2.getPerspectiveTransform(
        source_quad, destination_quad
    )
    corrected_to_source = np.linalg.inv(source_to_corrected)
    lines = []
    for index, x in enumerate((90.0, 140.0, 190.0, 240.0, 290.0), start=1):
        p1, p2 = transform_points(
            [(x, 30.0), (x, 370.0)], corrected_to_source
        )
        lines.append(
            {
                "id": f"line-{index}",
                "source_p1": {"x": p1[0], "y": p1[1]},
                "source_p2": {"x": p2[0], "y": p2[1]},
            }
        )

    result = measure_tramline_quality(lines, source_to_corrected)

    assert result["before"]["angle_std"] > result["after"]["angle_std"]
    assert result["after"]["angle_std"] < 0.01
    assert result["after"]["spacing_cv"] < 0.01
    assert (
        result["after"]["parallelism_score"]
        > result["before"]["parallelism_score"]
    )
    assert len(result["lines"]) == 5
    assert all("line_a_after" in row for row in result["lines"])


def test_warp_penalty_detects_anisotropic_shear() -> None:
    identity = measure_warp_penalty(np.eye(3), (800, 600))
    distorted = measure_warp_penalty(
        np.asarray(
            [
                [1.8, 0.55, 0.0],
                [0.05, 0.48, 0.0],
                [0.0008, -0.0004, 1.0],
            ]
        ),
        (800, 600),
    )

    assert identity["warp_penalty"] < 0.01
    assert distorted["warp_penalty"] > identity["warp_penalty"] + 20
    assert distorted["warp_anisotropy_score"] > 0
    assert distorted["local_distortion_indicator"] > 0


def test_warp_penalty_alone_is_not_reported_as_quality_evidence() -> None:
    image = np.zeros((180, 180, 3), dtype=np.uint8)
    result = analyze_perspective_quality(
        image,
        image.copy(),
        np.eye(3),
        method="four_point",
        status="proposed",
    )

    assert result.report["scores"]["overall_before"] is None
    assert result.report["scores"]["overall_after"] is None
    assert result.report["interpretation"] == "not_measurable"
    assert "Ölçüm kapsamı düşük" in result.report["interpretation_message"]


def test_single_dense_evidence_family_has_medium_confidence() -> None:
    image = np.zeros((180, 180, 3), dtype=np.uint8)
    tramlines = [
        {
            "id": f"line-{index}",
            "source_p1": {"x": x, "y": 10},
            "source_p2": {"x": x, "y": 170},
        }
        for index, x in enumerate((20, 45, 70, 95, 120, 145), start=1)
    ]
    result = analyze_perspective_quality(
        image,
        image.copy(),
        np.eye(3),
        tramlines=tramlines,
        method="tramline_guided",
        status="proposed",
    )

    assert result.report["scores"]["overall_after"] is not None
    assert result.report["confidence"]["level"] == "medium"


def test_quality_report_is_non_destructive_and_exports_bundle(tmp_path) -> None:
    image, circles, structure_state = ideal_six_arm_fixture()
    original = [circle.to_dict() for circle in circles]
    tramlines = [
        {
            "id": f"line-{index}",
            "source_p1": {"x": x, "y": 20},
            "source_p2": {"x": x, "y": 700},
            "manual": True,
        }
        for index, x in enumerate((100, 180, 260, 340, 420, 500, 580), start=1)
    ]

    result = analyze_perspective_quality(
        image,
        image.copy(),
        np.eye(3),
        circles=circles,
        tramlines=tramlines,
        center_source=(360.0, 360.0),
        structure_state=structure_state,
        method="tramline_guided",
        status="approved",
    )

    assert [circle.to_dict() for circle in circles] == original
    assert result.report["non_destructive"] is True
    assert result.report["annotation_data_modified"] is False
    assert result.report["s_drift_correction_applied"] is False
    assert result.report["confidence"]["level"] == "high"
    assert result.report["scores"]["overall_after"] is not None
    assert result.report["arm_symmetry"]["available"] is True
    assert len(result.report["arm_metrics"]) == 78
    assert result.report["k5k6"]["matched_sequence_count"] == 13
    assert len(result.report["k5k6"]["rows"]) == 13
    metric = result.report["circle_metrics"][0]
    for field in (
        "source_x",
        "source_y",
        "source_radius",
        "corrected_x",
        "corrected_y",
        "corrected_radius",
        "source_major_axis",
        "source_minor_axis",
        "corrected_major_axis",
        "corrected_minor_axis",
    ):
        assert metric[field] is not None

    files = write_perspective_quality_exports(result, tmp_path)
    expected_names = {
        "perspective_quality_report.json",
        "perspective_circle_metrics.csv",
        "perspective_arm_metrics.csv",
        "tramline_metrics.csv",
        "perspective_before_after.png",
        "roundness_overlay.png",
        "tramline_overlay_corrected.png",
    }
    assert expected_names <= {path.name for path in files.values()}
    for path in files.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    report = json.loads(
        (tmp_path / "perspective_quality_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["method"] == "tramline_guided"
    assert report["status"] == "approved"


def test_quality_api_and_export_are_available(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "DATA_ROOT": tmp_path / "data",
            "OUTPUT_ROOT": tmp_path / "outputs",
        }
    )
    client = app.test_client()
    image = np.zeros((240, 240, 3), dtype=np.uint8)
    cv2.circle(image, (120, 120), 34, (255, 255, 255), 2)
    success, encoded = cv2.imencode(".png", image)
    assert success
    record = client.post(
        "/api/upload",
        data={"image": (io.BytesIO(encoded.tobytes()), "quality.png")},
        content_type="multipart/form-data",
    ).get_json()
    profile = client.get(
        f"/api/preprocess/{record['image_id']}"
    ).get_json()["profile"]
    applied = client.post(
        "/api/preprocess/apply",
        json={"image_id": record["image_id"], "profile": profile},
    )
    assert applied.status_code == 200
    applied_payload = applied.get_json()
    version_before_quality = applied_payload["version"]
    request = {
        "image_id": record["image_id"],
        "method": "four_point",
        "status": "proposed",
        "perspective_points": [
            {"x": 10, "y": 10},
            {"x": 229, "y": 10},
            {"x": 229, "y": 229},
            {"x": 10, "y": 229},
        ],
        "center_source": {"x": 120, "y": 120},
        "circles": [
            {
                "id": "main-1",
                "x": 120,
                "y": 120,
                "radius": 34,
                "source_x": 120,
                "source_y": 120,
                "source_radius": 34,
                "circle_type": "main",
                "arm_id": 1,
                "sequence_index": 1,
            }
        ],
        "tramlines": [
            {
                "id": f"line-{index}",
                "source_p1": {"x": x, "y": 0},
                "source_p2": {"x": x, "y": 239},
            }
            for index, x in enumerate((50, 100, 150), start=1)
        ],
    }

    response = client.post("/api/perspective/quality", json=request)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["image_version"] == version_before_quality
    assert payload["report"]["method"] == "four_point"
    assert payload["report"]["status"] == "proposed"
    assert payload["report"]["scores"]["overall_after"] is not None
    persisted = client.get(
        f"/api/preprocess/{record['image_id']}"
    ).get_json()
    assert persisted["version"] == version_before_quality
    assert persisted["profile"] == profile
    for key in (
        "perspective_before_after",
        "roundness_overlay",
        "tramline_overlay_corrected",
    ):
        artifact = client.get(payload["artifacts"][key])
        assert artifact.status_code == 200
        assert artifact.data

    exported = client.post(
        "/api/perspective/quality/export", json=request
    )
    assert exported.status_code == 200
    export_payload = exported.get_json()
    for key in (
        "perspective_quality_report",
        "perspective_circle_metrics",
        "perspective_arm_metrics",
        "tramline_metrics",
    ):
        download = client.get(export_payload["files"][key]["url"])
        assert download.status_code == 200
        assert download.data

    reference_request = dict(request)
    reference_request.pop("perspective_points")
    reference_request["method"] = "none"
    reference_request["status"] = "not_applied"
    reference = client.post(
        "/api/perspective/quality", json=reference_request
    )
    assert reference.status_code == 200
    reference_report = reference.get_json()["report"]
    assert reference_report["method"] == "none"
    assert reference_report["status"] == "not_applied"
    assert reference_report["scores"]["overall_after"] is None
