from __future__ import annotations

import io

import cv2

from src.synthetic import generate_preprocessing_fixture
from src.webapp import create_app


def upload_fixture(client, *, width: int = 520, height: int = 380) -> dict:
    fixture = generate_preprocessing_fixture(
        width=width,
        height=height,
        apply_perspective=False,
    )
    ok, encoded = cv2.imencode(".png", fixture.image)
    assert ok
    response = client.post(
        "/api/upload",
        data={
            "image": (
                io.BytesIO(encoded.tobytes()),
                "preprocessing-fixture.png",
            )
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    return response.get_json()


def configured_app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "DATA_ROOT": tmp_path / "data",
            "OUTPUT_ROOT": tmp_path / "outputs",
            "SAMPLE_IMAGE": tmp_path / "missing.jpg",
        }
    )


def test_preprocess_apply_profile_and_intermediate_exports(tmp_path) -> None:
    app = configured_app(tmp_path)
    client = app.test_client()
    record = upload_fixture(client)

    profile_response = client.get(f"/api/preprocess/{record['image_id']}")
    assert profile_response.status_code == 200
    profile = profile_response.get_json()["profile"]
    profile["analysis_polygon"] = [
        {"x": 20, "y": 20},
        {"x": 500, "y": 20},
        {"x": 500, "y": 360},
        {"x": 20, "y": 360},
    ]
    profile["exclusion_strokes"] = [
        {
            "points": [{"x": 60, "y": 70}, {"x": 140, "y": 70}],
            "radius": 8,
            "mode": "paint",
        }
    ]
    illumination = next(
        operation
        for operation in profile["operations"]
        if operation["type"] == "illumination"
    )
    illumination["enabled"] = True
    illumination["params"].update(
        {"method": "divide", "scale": 61, "strength": 0.8}
    )
    apply_response = client.post(
        "/api/preprocess/apply",
        json={"image_id": record["image_id"], "profile": profile},
    )
    assert apply_response.status_code == 200
    applied = apply_response.get_json()
    assert applied["version"] > record["version"]
    assert len(applied["steps"]) == 4
    cached_response = client.post(
        "/api/preprocess/apply",
        json={"image_id": record["image_id"], "profile": profile},
    )
    assert cached_response.status_code == 200
    assert cached_response.get_json()["version"] == applied["version"]
    for key in (
        "rectified",
        "illumination_corrected",
        "foreground_probability",
        "exclusion_mask",
        "measurement",
    ):
        assert client.get(applied["artifacts"][key]).status_code == 200

    save_response = client.post(
        "/api/preprocess/profile/save",
        json={"image_id": record["image_id"], "profile": profile},
    )
    assert save_response.status_code == 200
    saved = save_response.get_json()
    assert saved["name"] == "preprocess_profile.json"
    profile_bytes = client.get(saved["url"]).data
    load_response = client.post(
        "/api/preprocess/profile/load",
        data={
            "image_id": record["image_id"],
            "profile": (
                io.BytesIO(profile_bytes),
                "preprocess_profile.json",
            ),
        },
        content_type="multipart/form-data",
    )
    assert load_response.status_code == 200
    assert (
        load_response.get_json()["profile"]["analysis_polygon"]
        == profile["analysis_polygon"]
    )

    export_response = client.post(
        "/api/preprocess/export",
        json={
            "image_id": record["image_id"],
            "circles": [],
            "center": None,
        },
    )
    assert export_response.status_code == 200
    exported = export_response.get_json()["files"]
    assert set(exported) == {
        "rectified",
        "illumination_corrected",
        "foreground_probability",
        "exclusion_mask",
        "detection_overlay",
    }
    assert {item["name"] for item in exported.values()} == {
        "rectified.png",
        "illumination_corrected.png",
        "foreground_probability.png",
        "exclusion_mask.png",
        "detection_overlay.png",
    }
    for item in exported.values():
        assert client.get(item["url"]).status_code == 200


def test_tramline_suggestion_requires_explicit_followup(tmp_path) -> None:
    app = configured_app(tmp_path)
    client = app.test_client()
    record = upload_fixture(client, width=620, height=460)
    # Ensure a current, identity preprocessing result exists.
    profile = client.get(
        f"/api/preprocess/{record['image_id']}"
    ).get_json()["profile"]
    client.post(
        "/api/preprocess/apply",
        json={"image_id": record["image_id"], "profile": profile},
    )
    response = client.post(
        "/api/preprocess/tramlines/suggest",
        json={
            "image_id": record["image_id"],
            "min_length_ratio": 0.20,
            "hough_threshold": 35,
            "expected_spacing_pixels": 58,
            "regularity": 1.0,
        },
    )

    assert response.status_code == 200
    suggestion = response.get_json()
    assert suggestion["requires_user_approval"] is True
    assert len(suggestion["lines"]) >= 2
    assert suggestion["spacing_pixels"] == 58
    assert suggestion["spacing_consistency"] > 0.95
    assert suggestion["method"] == "oriented_periodic_profile_v2"
    # Suggestion is read-only: profile remains unchanged until another endpoint
    # receives the user's explicit, edited profile.
    current = client.get(
        f"/api/preprocess/{record['image_id']}"
    ).get_json()["profile"]
    assert current["tramline"]["enabled"] is False
    assert current["tramline"]["lines"] == []


def test_advanced_detection_preserves_supplied_lock(tmp_path) -> None:
    app = configured_app(tmp_path)
    client = app.test_client()
    record = upload_fixture(client, width=480, height=360)
    profile = client.get(
        f"/api/preprocess/{record['image_id']}"
    ).get_json()["profile"]
    profile["detection"].update(
        {
            "tile_size": 512,
            "tile_overlap": 0.20,
            "min_final_confidence": 0.22,
        }
    )
    client.post(
        "/api/preprocess/apply",
        json={"image_id": record["image_id"], "profile": profile},
    )
    locked = {
        "id": "approved-1",
        "x": 100,
        "y": 100,
        "processed_x": 100,
        "processed_y": 100,
        "source_x": 100,
        "source_y": 100,
        "radius": 9,
        "confidence": 1,
        "final_confidence": 1,
        "circle_type": "main",
        "parent_id": None,
        "arm_id": None,
        "sequence_index": None,
        "side": None,
        "locked": True,
        "source": "manual_override",
    }
    response = client.post(
        "/api/detect/advanced",
        json={
            "image_id": record["image_id"],
            "circles": [locked],
            "max_candidates": 120,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["elapsed_seconds"] >= 0
    assert set(payload["radius_ranges"]) == {
        "small",
        "medium",
        "large",
        "main_variable",
    }
    assert payload["radius_ranges"]["small"]["role"] == "satellite_standard"
    assert (
        payload["radius_ranges"]["main_variable"]["role"]
        == "main_arm_variable"
    )
    assert set(payload["detection_counts"]) == {
        "main_variable",
        "side_standard",
    }
    assert sum(payload["detection_counts"].values()) == payload["count"] - 1
    approved = next(
        circle
        for circle in payload["circles"]
        if circle["id"] == "approved-1"
    )
    assert approved["locked"] is True
    assert approved["source_x"] == 100
    assert approved["source_y"] == 100
    automatic = [
        circle
        for circle in payload["circles"]
        if circle["id"] != "approved-1"
    ]
    if automatic:
        candidate = automatic[0]
        for field in (
            "edge_score",
            "color_score",
            "circularity_score",
            "detector_agreement",
            "symmetry_support",
            "final_confidence",
        ):
            assert candidate[field] is not None
