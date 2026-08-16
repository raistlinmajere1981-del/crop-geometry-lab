"""Local Flask application for the Crop Geometry Lab browser editor."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Collection, Mapping
from uuid import uuid4

import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
import numpy as np
from numpy.typing import NDArray
from werkzeug.utils import secure_filename

from .arm_classifier import (
    ArmModel,
    assign_arms,
    estimate_spiral_parameters,
    polar_coordinates,
)
from .circle_detector import DetectorConfig, detect_circles
from .feature_table import (
    AnnotationSet,
    CircleFeature,
    circles_from_mappings,
    write_csv,
    write_json,
)
from .perspective import correct_perspective
from .perspective_quality import (
    PerspectiveQualityResult,
    analyze_perspective_quality,
    write_perspective_quality_exports,
)
from .quality_control import rank_weak_visual_support
from .multiscale_detector import (
    detect_multiscale_circles,
    effective_main_arm_radius_range,
    effective_radius_ranges,
)
from .preprocessing import (
    PreprocessProfile,
    PreprocessResult,
    apply_preprocess_stack,
    detect_parallel_tramlines,
    identity_transform,
    transform_point,
    transform_radius,
    write_preprocess_profile,
)
from .symmetry_tests import (
    geometric_symmetry_errors,
    image_symmetry_errors,
    sixfold_summary,
)
from .structure_analysis import structural_report, write_structure_exports
from .visualizer import draw_overlay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
APP_VERSION = "0.6.0"


@dataclass(slots=True)
class ImageRecord:
    id: str
    display_name: str
    original_path: Path
    working_path: Path
    width: int
    height: int
    version: int = 1
    correction_history: list[dict[str, Any]] = field(default_factory=list)
    preprocess_result: PreprocessResult | None = None
    preprocess_artifacts: dict[str, Path] = field(default_factory=dict)
    preprocess_snapshots: list[dict[str, Any]] = field(default_factory=list)
    perspective_quality_result: PerspectiveQualityResult | None = None
    perspective_quality_artifacts: dict[str, Path] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        payload = {
            "image_id": self.id,
            "name": self.display_name,
            "width": self.width,
            "height": self.height,
            "version": self.version,
            "image_url": url_for(
                "image_content", image_id=self.id, v=self.version
            ),
        }
        if self.preprocess_result is not None:
            payload["source_to_processed"] = (
                self.preprocess_result.source_to_processed.tolist()
            )
            payload["processed_to_source"] = (
                self.preprocess_result.processed_to_source.tolist()
            )
            payload["source_width"] = int(
                self.preprocess_result.source_image.shape[1]
            )
            payload["source_height"] = int(
                self.preprocess_result.source_image.shape[0]
            )
        return payload


class LocalImageStore:
    """Thread-safe, process-local registry backed by project data directories."""

    def __init__(self, raw_dir: Path, corrected_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.corrected_dir = corrected_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.corrected_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, ImageRecord] = {}
        self._lock = threading.RLock()

    @staticmethod
    def decode(data: bytes) -> NDArray[np.uint8]:
        if not data:
            raise ValueError("Görüntü dosyası boş")
        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("Dosya geçerli bir JPG, PNG veya WebP görüntüsü değil")
        return image

    def add_upload(self, filename: str, data: bytes) -> ImageRecord:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise ValueError("Yalnızca JPG, PNG ve WebP görüntüleri desteklenir")
        image = self.decode(data)
        image_id = uuid4().hex[:16]
        safe_name = secure_filename(filename) or f"image{suffix}"
        destination = self.raw_dir / f"{image_id}_{safe_name}"
        destination.write_bytes(data)
        height, width = image.shape[:2]
        record = ImageRecord(
            id=image_id,
            display_name=filename,
            original_path=destination,
            working_path=destination,
            width=width,
            height=height,
        )
        with self._lock:
            self._records[image_id] = record
        return record

    def add_existing(self, path: Path, display_name: str | None = None) -> ImageRecord:
        if not path.is_file():
            raise FileNotFoundError(f"Örnek görüntü bulunamadı: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Örnek görüntü okunamadı: {path}")
        image_id = uuid4().hex[:16]
        height, width = image.shape[:2]
        record = ImageRecord(
            id=image_id,
            display_name=display_name or path.name,
            original_path=path.resolve(),
            working_path=path.resolve(),
            width=width,
            height=height,
        )
        with self._lock:
            self._records[image_id] = record
        return record

    def get(self, image_id: str) -> ImageRecord:
        with self._lock:
            record = self._records.get(image_id)
        if record is None:
            raise ValueError("Görüntü oturumu bulunamadı; dosyayı yeniden açın")
        return record

    def read(self, image_id: str) -> tuple[ImageRecord, NDArray[np.uint8]]:
        record = self.get(image_id)
        image = cv2.imread(str(record.working_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError("Çalışma görüntüsü diskten okunamadı")
        return record, image

    def read_source(
        self, image_id: str
    ) -> tuple[ImageRecord, NDArray[np.uint8]]:
        record = self.get(image_id)
        image = cv2.imread(str(record.original_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError("Kaynak görüntü diskten okunamadı")
        return record, image

    def apply_preprocess(
        self, image_id: str, profile: PreprocessProfile
    ) -> ImageRecord:
        with self._lock:
            record, source = self.read_source(image_id)
            if (
                record.preprocess_result is not None
                and record.preprocess_result.profile.to_dict()
                == profile.to_dict()
            ):
                return record
            result = apply_preprocess_stack(source, profile)
            next_version = record.version + 1
            destination_dir = (
                self.corrected_dir / record.id / f"preprocess_v{next_version}"
            )
            destination_dir.mkdir(parents=True, exist_ok=True)
            artifacts: dict[str, Path] = {
                "rectified": destination_dir / "rectified.png",
                "illumination_corrected": (
                    destination_dir / "illumination_corrected.png"
                ),
                "measurement": destination_dir / "measurement.png",
                "foreground_probability": (
                    destination_dir / "foreground_probability.png"
                ),
                "exclusion_mask": destination_dir / "exclusion_mask.png",
                "analysis_mask": destination_dir / "analysis_mask.png",
            }
            foreground_png = np.clip(
                result.foreground_probability * 255.0, 0, 255
            ).astype(np.uint8)
            images_to_write: dict[str, NDArray[np.uint8]] = {
                "rectified": result.rectified,
                "illumination_corrected": result.illumination_corrected,
                "measurement": result.measurement_image,
                "foreground_probability": foreground_png,
                "exclusion_mask": result.exclusion_mask,
                "analysis_mask": result.analysis_mask,
            }
            for key, image in images_to_write.items():
                if not cv2.imwrite(str(artifacts[key]), image):
                    raise OSError(f"Ön işleme ara çıktısı yazılamadı: {key}")
            snapshots: list[dict[str, Any]] = []
            for index, snapshot in enumerate(result.snapshots):
                safe_id = secure_filename(snapshot.id) or f"step-{index + 1}"
                before_key = f"step_{index}_before"
                after_key = f"step_{index}_after"
                before_path = destination_dir / f"{safe_id}_before.png"
                after_path = destination_dir / f"{safe_id}_after.png"
                if not cv2.imwrite(str(before_path), snapshot.before):
                    raise OSError("İşlem öncesi görünüm yazılamadı")
                if not cv2.imwrite(str(after_path), snapshot.after):
                    raise OSError("İşlem sonrası görünüm yazılamadı")
                artifacts[before_key] = before_path
                artifacts[after_key] = after_path
                snapshots.append(
                    {
                        "id": snapshot.id,
                        "type": snapshot.type,
                        "enabled": snapshot.enabled,
                        "before_key": before_key,
                        "after_key": after_key,
                        "before_width": int(snapshot.before.shape[1]),
                        "before_height": int(snapshot.before.shape[0]),
                        "after_width": int(snapshot.after.shape[1]),
                        "after_height": int(snapshot.after.shape[0]),
                    }
                )
            profile_path = destination_dir / "preprocess_profile.json"
            write_preprocess_profile(profile, profile_path)
            artifacts["profile"] = profile_path
            record.preprocess_result = result
            record.preprocess_artifacts = artifacts
            record.preprocess_snapshots = snapshots
            record.working_path = artifacts["measurement"]
            record.width = result.width
            record.height = result.height
            record.version = next_version
            record.perspective_quality_result = None
            record.perspective_quality_artifacts = {}
            return record

    def ensure_preprocess(self, image_id: str) -> ImageRecord:
        record, source = self.read_source(image_id)
        if record.preprocess_result is not None:
            return record
        profile = PreprocessProfile.default(source.shape[1], source.shape[0])
        return self.apply_preprocess(image_id, profile)

    def update_corrected(
        self,
        image_id: str,
        image: NDArray[np.uint8],
        metadata: Mapping[str, Any],
    ) -> ImageRecord:
        with self._lock:
            record = self.get(image_id)
            next_version = record.version + 1
            destination = self.corrected_dir / (
                f"{record.id}_corrected_v{next_version}.png"
            )
            if not cv2.imwrite(str(destination), image):
                raise OSError("Düzeltilmiş görüntü kaydedilemedi")
            record.working_path = destination
            record.height, record.width = image.shape[:2]
            record.version = next_version
            record.correction_history.append(dict(metadata))
            record.preprocess_result = None
            record.preprocess_artifacts = {}
            record.preprocess_snapshots = []
            record.perspective_quality_result = None
            record.perspective_quality_artifacts = {}
            return record


def _json_body() -> Mapping[str, Any]:
    data = request.get_json(silent=False)
    if not isinstance(data, Mapping):
        raise ValueError("JSON gövdesi bir nesne olmalı")
    return data


def _point(data: Any, label: str = "center") -> tuple[float, float]:
    if isinstance(data, Mapping):
        try:
            return float(data["x"]), float(data["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} x ve y değerleri içermeli") from exc
    if isinstance(data, (list, tuple)) and len(data) == 2:
        return float(data[0]), float(data[1])
    raise ValueError(f"{label} geçerli bir iki boyutlu nokta olmalı")


def _circles(data: Any) -> list[CircleFeature]:
    if not isinstance(data, list):
        raise ValueError("circles bir liste olmalı")
    if not all(isinstance(item, Mapping) for item in data):
        raise ValueError("Her daire bir JSON nesnesi olmalı")
    circles = circles_from_mappings(data)
    ids = [circle.id for circle in circles]
    if len(ids) != len(set(ids)):
        raise ValueError("Daire kimlikleri benzersiz olmalı")
    return circles


def _manual_locks(data: Any) -> dict[str, Collection[str]]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("manual_locks bir nesne olmalı")
    allowed = {
        "circle_type",
        "arm_id",
        "sequence_index",
        "side",
        "parent_id",
    }
    result: dict[str, Collection[str]] = {}
    for circle_id, fields in data.items():
        if not isinstance(fields, list) or not all(
            isinstance(field, str) for field in fields
        ):
            raise ValueError("Her manual_locks değeri alan adlarından oluşan liste olmalı")
        result[str(circle_id)] = tuple(field for field in fields if field in allowed)
    return result


def _edit_history(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if not isinstance(data, list) or not all(
        isinstance(item, Mapping) for item in data
    ):
        raise ValueError("edit_history bir nesne listesi olmalı")
    # Round-tripping through JSON rejects browser-only/non-serializable values
    # while retaining previous/new scalar, list, and object values verbatim.
    return json.loads(json.dumps(data, ensure_ascii=False))


def _structure_state(data: Any) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("structure_state bir nesne olmalı")
    return json.loads(json.dumps(data, ensure_ascii=False))


def _arm_settings(data: Mapping[str, Any]) -> tuple[ArmModel, float]:
    model = str(data.get("arm_model") or "radial")
    if model not in {"radial", "spiral"}:
        raise ValueError("Kol modeli radial veya spiral olmalı")
    twist = float(data.get("arm_twist_degrees_per_100px", 0.0))
    if not math.isfinite(twist) or abs(twist) > 360.0:
        raise ValueError("Spiral dönüşü -360 ile 360 derece/100 px arasında olmalı")
    return model, twist  # type: ignore[return-value]


def _editor_state(data: Any) -> tuple[list[str], list[dict[str, Any]]]:
    if data is None:
        return [], []
    if not isinstance(data, Mapping):
        raise ValueError("editor_state bir nesne olmalı")
    raw_deleted = data.get("deleted_automatic_ids", [])
    raw_suppressed = data.get("suppressed_automatic_geometry", [])
    if not isinstance(raw_deleted, list) or not all(
        isinstance(circle_id, str) for circle_id in raw_deleted
    ):
        raise ValueError("Silinen otomatik kimlikleri bir metin listesi olmalı")
    if not isinstance(raw_suppressed, list):
        raise ValueError("Bastırılmış otomatik geometri bir liste olmalı")
    suppressed: list[dict[str, Any]] = []
    for item in raw_suppressed:
        if not isinstance(item, Mapping):
            raise ValueError("Bastırılmış geometri nesnelerden oluşmalı")
        try:
            geometry = {
                "id": str(item.get("id") or ""),
                "x": float(item["x"]),
                "y": float(item["y"]),
                "radius": float(item["radius"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Bastırılmış geometri x, y ve radius içermeli") from exc
        if (
            not all(
                math.isfinite(geometry[field_name])
                for field_name in ("x", "y", "radius")
            )
            or geometry["radius"] <= 0
        ):
            raise ValueError("Bastırılmış geometri sonlu ve pozitif olmalı")
        suppressed.append(geometry)
    return list(dict.fromkeys(raw_deleted)), suppressed


def _analysis(
    image: NDArray[np.uint8],
    circles: list[CircleFeature],
    center: tuple[float, float],
    phase: float,
    manual_locks: Mapping[str, Collection[str]] | None = None,
    arm_model: ArmModel = "radial",
    arm_twist_degrees_per_100px: float = 0.0,
) -> tuple[
    list[CircleFeature],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, float],
]:
    classified = assign_arms(
        circles,
        center,
        phase,
        preserve_manual_fields=True,
        locked_fields=manual_locks,
        infer_parents=True,
        model=arm_model,
        twist_degrees_per_100px=arm_twist_degrees_per_100px,
    )
    image_errors = image_symmetry_errors(image, center)
    geometry_errors = geometric_symmetry_errors(classified, center)
    image_data = [item.to_dict() for item in image_errors]
    geometry_data = [item.to_dict() for item in geometry_errors]
    return classified, image_data, geometry_data, sixfold_summary(
        image_errors, geometry_errors
    )


def _measure_without_reclassification(
    image: NDArray[np.uint8],
    circles: list[CircleFeature],
    center: tuple[float, float],
) -> tuple[
    list[CircleFeature],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, float],
]:
    """Refresh derived measurements without changing approved structure."""

    measured: list[CircleFeature] = []
    for circle in circles:
        distance, angle = polar_coordinates((circle.x, circle.y), center)
        measured.append(
            replace(
                circle,
                center_distance=distance,
                center_angle=angle,
            )
        )
    image_errors = image_symmetry_errors(image, center)
    geometry_errors = geometric_symmetry_errors(measured, center)
    image_data = [item.to_dict() for item in image_errors]
    geometry_data = [item.to_dict() for item in geometry_errors]
    return measured, image_data, geometry_data, sixfold_summary(
        image_errors, geometry_errors
    )


def _preprocess_profile_from_data(
    data: Any,
    source_image: NDArray[np.uint8],
) -> PreprocessProfile:
    if not isinstance(data, Mapping):
        raise ValueError("preprocess_profile bir JSON nesnesi olmalı")
    return PreprocessProfile.from_mapping(
        data,
        fallback_size=(source_image.shape[1], source_image.shape[0]),
    )


def _annotation_document(
    record: ImageRecord,
    circles: list[CircleFeature],
    *,
    center: tuple[float, float] | None,
    phase: float,
    arm_model: ArmModel,
    arm_twist: float,
    manual_locks: Mapping[str, Collection[str]] | None = None,
    deleted_automatic_ids: list[str] | None = None,
    suppressed_geometry: list[dict[str, Any]] | None = None,
    image_errors: list[dict[str, Any]] | None = None,
    geometry_errors: list[dict[str, Any]] | None = None,
    edit_history: list[dict[str, Any]] | None = None,
    structure_state: Mapping[str, Any] | None = None,
) -> AnnotationSet:
    return AnnotationSet(
        circles=circles,
        image_id=record.id,
        image_name=record.display_name,
        general_center=center,
        arm_phase_degrees=phase,
        arm_model=arm_model,
        arm_twist_degrees_per_100px=arm_twist,
        manual_locks={
            circle_id: list(fields)
            for circle_id, fields in (manual_locks or {}).items()
        },
        deleted_automatic_ids=deleted_automatic_ids or [],
        suppressed_automatic_geometry=suppressed_geometry or [],
        image_symmetry=image_errors or [],
        geometric_symmetry=geometry_errors or [],
        source_to_processed=(
            record.preprocess_result.source_to_processed.tolist()
            if record.preprocess_result is not None
            else identity_transform().tolist()
        ),
        processed_to_source=(
            record.preprocess_result.processed_to_source.tolist()
            if record.preprocess_result is not None
            else identity_transform().tolist()
        ),
        preprocess_profile=(
            record.preprocess_result.profile.to_dict()
            if record.preprocess_result is not None
            else None
        ),
        edit_history=edit_history or [],
        structure_state=dict(structure_state or {}),
    )


def create_app(test_config: Mapping[str, Any] | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "web" / "templates"),
        static_folder=str(PROJECT_ROOT / "web" / "static"),
        static_url_path="/static",
    )
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=40 * 1024 * 1024,
        DATA_ROOT=PROJECT_ROOT / "data",
        OUTPUT_ROOT=PROJECT_ROOT / "outputs",
        SAMPLE_IMAGE=PROJECT_ROOT / "raw" / "milk_hill_01.jpg",
        PREPROCESS_SAMPLE_IMAGE=(
            PROJECT_ROOT / "data" / "raw" / "synthetic_preprocessing.png"
        ),
        JSON_SORT_KEYS=False,
    )
    if test_config:
        app.config.update(test_config)
    data_root = Path(app.config["DATA_ROOT"])
    output_root = Path(app.config["OUTPUT_ROOT"])
    annotation_root = data_root / "annotations"
    output_root.mkdir(parents=True, exist_ok=True)
    annotation_root.mkdir(parents=True, exist_ok=True)
    store = LocalImageStore(data_root / "raw", data_root / "corrected")
    app.extensions["crop_image_store"] = store

    @app.get("/")
    def index() -> str:
        sample_available = Path(app.config["SAMPLE_IMAGE"]).is_file()
        preprocess_sample_available = Path(
            app.config["PREPROCESS_SAMPLE_IMAGE"]
        ).is_file()
        return render_template(
            "index.html",
            sample_available=sample_available,
            preprocess_sample_available=preprocess_sample_available,
            static_version=APP_VERSION,
        )

    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"status": "ok", "version": APP_VERSION})

    @app.post("/api/upload")
    def upload() -> Response:
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            raise ValueError("Bir görüntü dosyası seçin")
        record = store.add_upload(uploaded.filename, uploaded.read())
        return jsonify(record.public_dict())

    @app.post("/api/sample")
    def load_sample() -> Response:
        record = store.add_existing(
            Path(app.config["SAMPLE_IMAGE"]),
            "Milk Hill Galaxy Spiral 2001",
        )
        return jsonify(record.public_dict())

    @app.post("/api/sample/preprocessing")
    def load_preprocessing_sample() -> Response:
        record = store.add_existing(
            Path(app.config["PREPROCESS_SAMPLE_IMAGE"]),
            "Sentetik Ön İşleme Doğrulama Görüntüsü",
        )
        return jsonify(record.public_dict())

    @app.get("/api/image/<image_id>")
    def image_content(image_id: str) -> Response:
        record = store.get(image_id)
        return send_file(record.working_path, conditional=True)

    def preprocess_payload(record: ImageRecord) -> dict[str, Any]:
        if record.preprocess_result is None:
            raise ValueError("Ön işleme sonucu henüz oluşturulmadı")
        artifacts = {
            key: url_for(
                "preprocess_artifact",
                image_id=record.id,
                artifact_key=key,
                v=record.version,
            )
            for key in record.preprocess_artifacts
        }
        steps = []
        for step in record.preprocess_snapshots:
            item = dict(step)
            item["before_url"] = artifacts[step["before_key"]]
            item["after_url"] = artifacts[step["after_key"]]
            steps.append(item)
        payload = record.public_dict()
        payload.update(
            {
                "profile": record.preprocess_result.profile.to_dict(),
                "artifacts": artifacts,
                "steps": steps,
                "source_to_processed": (
                    record.preprocess_result.source_to_processed.tolist()
                ),
                "processed_to_source": (
                    record.preprocess_result.processed_to_source.tolist()
                ),
            }
        )
        return payload

    @app.get("/api/preprocess/<image_id>")
    def get_preprocess(image_id: str) -> Response:
        record, source = store.read_source(image_id)
        if record.preprocess_result is None:
            profile = PreprocessProfile.default(
                source.shape[1], source.shape[0]
            )
            return jsonify(
                {
                    "profile": profile.to_dict(),
                    "source_to_processed": identity_transform().tolist(),
                    "processed_to_source": identity_transform().tolist(),
                    "source_width": source.shape[1],
                    "source_height": source.shape[0],
                    "applied": False,
                }
            )
        payload = preprocess_payload(record)
        payload["applied"] = True
        return jsonify(payload)

    @app.post("/api/preprocess/apply")
    def apply_preprocess() -> Response:
        data = _json_body()
        image_id = str(data.get("image_id") or "")
        _, source = store.read_source(image_id)
        profile = _preprocess_profile_from_data(
            data.get("profile"), source
        )
        record = store.apply_preprocess(image_id, profile)
        return jsonify(preprocess_payload(record))

    @app.get("/api/preprocess/artifact/<image_id>/<artifact_key>")
    def preprocess_artifact(
        image_id: str, artifact_key: str
    ) -> Response:
        record = store.get(image_id)
        path = record.preprocess_artifacts.get(artifact_key)
        if path is None or not path.is_file():
            raise FileNotFoundError("Ön işleme ara çıktısı bulunamadı")
        return send_file(path, conditional=True)

    @app.post("/api/preprocess/tramlines/suggest")
    def suggest_tramlines() -> Response:
        data = _json_body()
        image_id = str(data.get("image_id") or "")
        record = store.ensure_preprocess(image_id)
        result = record.preprocess_result
        if result is None:
            raise ValueError("Ön işleme sonucu bulunamadı")
        suggestion = detect_parallel_tramlines(
            result.measurement_image,
            processed_to_source=result.processed_to_source,
            detection_mask=result.analysis_mask,
            min_length_ratio=float(data.get("min_length_ratio", 0.28)),
            angle_tolerance_degrees=float(
                data.get("angle_tolerance_degrees", 9.0)
            ),
            hough_threshold=int(data.get("hough_threshold", 55)),
            expected_spacing_pixels=float(
                data.get("expected_spacing_pixels", 0.0)
            ),
            regularity=float(data.get("regularity", 0.72)),
        )
        return jsonify(suggestion.to_dict())

    def build_perspective_quality(
        data: Mapping[str, Any],
    ) -> tuple[ImageRecord, PerspectiveQualityResult]:
        image_id = str(data.get("image_id") or "")
        record = store.ensure_preprocess(image_id)
        preprocess = record.preprocess_result
        if preprocess is None:
            raise ValueError("Ön işleme sonucu bulunamadı")
        circles = _circles(data.get("circles", []))
        raw_tramlines = data.get("tramlines")
        if raw_tramlines is None:
            profile_lines = preprocess.profile.tramline.get("lines", [])
            raw_tramlines = [
                {
                    "id": f"tramline-profile-{index + 1}",
                    "source_p1": line.get("p1"),
                    "source_p2": line.get("p2"),
                    "manual": True,
                    "support": 1.0,
                }
                for index, line in enumerate(profile_lines)
                if isinstance(line, Mapping)
            ]
        if not isinstance(raw_tramlines, list) or not all(
            isinstance(line, Mapping) for line in raw_tramlines
        ):
            raise ValueError("tramlines bir çizgi nesneleri listesi olmalı")
        center_source: tuple[float, float] | None = None
        if data.get("center_source") is not None:
            center_source = _point(data.get("center_source"), "center_source")
        elif data.get("center") is not None:
            center_processed = _point(data.get("center"), "center")
            center_source = transform_point(
                center_processed,
                preprocess.processed_to_source,
            )
        raw_structure = data.get("structure_state", {})
        if not isinstance(raw_structure, Mapping):
            raise ValueError("structure_state bir JSON nesnesi olmalı")
        raw_weights = data.get("weights")
        if raw_weights is not None and not isinstance(raw_weights, Mapping):
            raise ValueError("weights bir JSON nesnesi olmalı")
        raw_selected = data.get("selected_circle_ids", [])
        if not isinstance(raw_selected, list):
            raise ValueError("selected_circle_ids bir liste olmalı")
        perspective_operation = preprocess.profile.operation("perspective")
        operation_enabled = bool(
            perspective_operation is not None
            and perspective_operation.enabled
            and len(perspective_operation.params.get("points", [])) == 4
        )
        operation_mode = (
            str(perspective_operation.params.get("mode") or "reference_quad")
            if perspective_operation is not None
            else "reference_quad"
        )
        requested_method = str(data.get("method") or "")
        if operation_enabled:
            method = (
                "tramline_guided"
                if operation_mode == "field_tracks"
                else "four_point"
            )
        elif requested_method in {"four_point", "tramline_guided"}:
            method = requested_method
        else:
            method = "none"
        requested_status = str(data.get("status") or "")
        if operation_enabled:
            status = "approved"
        elif requested_status == "proposed":
            status = "proposed"
        else:
            status = "not_applied"
        quality_corrected = preprocess.rectified
        quality_matrix = preprocess.source_to_processed
        if not operation_enabled and status == "proposed":
            raw_points = data.get("perspective_points")
            if not isinstance(raw_points, list) or len(raw_points) != 4:
                method = "none"
                status = "not_applied"
            else:
                points = [
                    _point(item, f"perspective_points[{index}]")
                    for index, item in enumerate(raw_points)
                ]
                proposed = correct_perspective(
                    preprocess.source_image,
                    points,
                    mode=(
                        "field_tracks"
                        if method == "tramline_guided"
                        else "reference_quad"
                    ),
                    preserve_full_image=bool(
                        data.get("preserve_full_image", True)
                    ),
                )
                quality_corrected = proposed.image
                quality_matrix = proposed.matrix
        quality = analyze_perspective_quality(
            preprocess.source_image,
            quality_corrected,
            quality_matrix,
            circles=circles,
            tramlines=[dict(line) for line in raw_tramlines],
            center_source=center_source,
            structure_state=raw_structure,
            method=method,
            status=status,
            weights=raw_weights,
            selected_circle_ids=[str(item) for item in raw_selected],
        )
        return record, quality

    def cache_perspective_quality(
        record: ImageRecord,
        quality: PerspectiveQualityResult,
    ) -> dict[str, Any]:
        destination = (
            data_root
            / "corrected"
            / record.id
            / f"perspective_quality_v{record.version}"
        )
        files = write_perspective_quality_exports(quality, destination)
        record.perspective_quality_result = quality
        record.perspective_quality_artifacts = files
        return {
            key: url_for(
                "perspective_quality_artifact",
                image_id=record.id,
                artifact_key=key,
                v=record.version,
            )
            for key in files
        }

    @app.post("/api/perspective/quality")
    def perspective_quality() -> Response:
        data = _json_body()
        record, quality = build_perspective_quality(data)
        artifacts = cache_perspective_quality(record, quality)
        return jsonify(
            {
                "image_id": record.id,
                "image_version": record.version,
                "report": quality.report,
                "artifacts": artifacts,
            }
        )

    @app.get(
        "/api/perspective/quality/artifact/<image_id>/<artifact_key>"
    )
    def perspective_quality_artifact(
        image_id: str,
        artifact_key: str,
    ) -> Response:
        record = store.get(image_id)
        path = record.perspective_quality_artifacts.get(artifact_key)
        if path is None or not path.is_file():
            raise FileNotFoundError(
                "Perspektif kalite ara çıktısı bulunamadı"
            )
        return send_file(path, conditional=True)

    @app.post("/api/perspective/quality/export")
    def export_perspective_quality() -> Response:
        data = _json_body()
        record, quality = build_perspective_quality(data)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        stem = secure_filename(Path(record.display_name).stem) or "analysis"
        relative_dir = Path(f"{stem}-{stamp}-perspective-quality")
        files = write_perspective_quality_exports(
            quality,
            output_root / relative_dir,
        )
        cache_perspective_quality(record, quality)
        return jsonify(
            {
                "report": quality.report,
                "files": {
                    key: {
                        "name": path.name,
                        "url": url_for(
                            "download_output",
                            filename=str(relative_dir / path.name),
                        ),
                    }
                    for key, path in files.items()
                },
            }
        )

    @app.post("/api/preprocess/profile/save")
    def save_preprocess_profile() -> Response:
        data = _json_body()
        image_id = str(data.get("image_id") or "")
        record, source = store.read_source(image_id)
        profile = _preprocess_profile_from_data(
            data.get("profile"), source
        )
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        stem = secure_filename(Path(record.display_name).stem) or "analysis"
        relative_dir = Path(f"{stem}-{stamp}-preprocess")
        destination_dir = output_root / relative_dir
        destination = destination_dir / "preprocess_profile.json"
        write_preprocess_profile(profile, destination)
        annotation_destination = (
            annotation_root / f"{record.id}_preprocess_profile.json"
        )
        write_preprocess_profile(profile, annotation_destination)
        relative_name = str(relative_dir / destination.name)
        return jsonify(
            {
                "name": "preprocess_profile.json",
                "url": url_for(
                    "download_output", filename=relative_name
                ),
                "profile": profile.to_dict(),
            }
        )

    @app.post("/api/preprocess/profile/load")
    def load_preprocess_profile() -> Response:
        image_id = str(request.form.get("image_id") or "")
        uploaded = request.files.get("profile")
        if uploaded is None:
            raise ValueError("Bir preprocess_profile.json dosyası seçin")
        record, source = store.read_source(image_id)
        try:
            raw = json.loads(uploaded.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Profil geçerli bir UTF-8 JSON dosyası değil") from exc
        profile = _preprocess_profile_from_data(raw, source)
        if (
            profile.source_width != source.shape[1]
            or profile.source_height != source.shape[0]
        ):
            raise ValueError(
                "Profil boyutları açık kaynak görüntüyle eşleşmiyor"
            )
        return jsonify(
            {
                "image_id": record.id,
                "profile": profile.to_dict(),
            }
        )

    @app.post("/api/perspective")
    def perspective() -> Response:
        data = _json_body()
        image_id = str(data.get("image_id") or "")
        mode = str(data.get("mode") or "reference_quad")
        raw_points = data.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("Dört perspektif noktası gerekli")
        points = [_point(item, f"point {index + 1}") for index, item in enumerate(raw_points)]
        _, image = store.read(image_id)
        result = correct_perspective(
            image,
            points,
            mode=mode,  # type: ignore[arg-type]
            preserve_full_image=bool(data.get("preserve_full_image", True)),
        )
        metadata = {
            "mode": result.mode,
            "preserves_full_image": result.preserves_full_image,
            "source_points": result.source_points.tolist(),
            "destination_points": result.destination_points.tolist(),
            "matrix": result.matrix.tolist(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        record = store.update_corrected(image_id, result.image, metadata)
        response = record.public_dict()
        response["perspective"] = metadata
        return jsonify(response)

    @app.post("/api/detect")
    def detect() -> Response:
        data = _json_body()
        image_id = str(data.get("image_id") or "")
        _, image = store.read(image_id)
        config_data = data.get("config", {})
        if not isinstance(config_data, Mapping):
            raise ValueError("config bir nesne olmalı")
        config = DetectorConfig(
            min_radius=float(config_data.get("min_radius", 4.0)),
            max_radius=float(config_data.get("max_radius", 80.0)),
            min_distance=float(config_data.get("min_distance", 10.0)),
            min_circularity=float(config_data.get("min_circularity", 0.38)),
            hough_threshold=float(config_data.get("hough_threshold", 34.0)),
            canny_threshold=float(config_data.get("canny_threshold", 110.0)),
            min_confidence=float(config_data.get("min_confidence", 0.12)),
            max_candidates=int(config_data.get("max_candidates", 350)),
        )
        candidates = detect_circles(image, config)
        record = store.get(image_id)
        inverse = (
            record.preprocess_result.processed_to_source
            if record.preprocess_result is not None
            else identity_transform()
        )
        mapped_candidates: list[CircleFeature] = []
        for candidate in candidates:
            source_x, source_y = transform_point(
                (candidate.x, candidate.y), inverse
            )
            source_radius = transform_radius(
                (candidate.x, candidate.y),
                candidate.radius,
                inverse,
            )
            mapped_candidates.append(
                replace(
                    candidate,
                    processed_x=candidate.x,
                    processed_y=candidate.y,
                    source_x=source_x,
                    source_y=source_y,
                    source_radius=source_radius,
                    corrected_x=candidate.x,
                    corrected_y=candidate.y,
                    corrected_radius=candidate.radius,
                    corrected_major_axis=candidate.radius * 2.0,
                    corrected_minor_axis=candidate.radius * 2.0,
                    source_major_axis=source_radius * 2.0,
                    source_minor_axis=source_radius * 2.0,
                    edge_score=candidate.confidence,
                    color_score=0.5,
                    circularity_score=0.7,
                    detector_agreement=0.35,
                    symmetry_support=0.5,
                    final_confidence=candidate.confidence,
                    detector="opencv_blob",
                )
            )
        return jsonify(
            {
                "circles": [
                    circle.to_dict() for circle in mapped_candidates
                ],
                "count": len(mapped_candidates),
                "notice": (
                    "Adaylar otomatik tahmindir; manuel düzeltmeler değişmeden korunur."
                ),
            }
        )

    @app.post("/api/detect/advanced")
    def detect_advanced() -> Response:
        data = _json_body()
        image_id = str(data.get("image_id") or "")
        record = store.ensure_preprocess(image_id)
        result = record.preprocess_result
        if result is None:
            raise ValueError("Ön işleme sonucu bulunamadı")
        supplied = _circles(data.get("circles", []))
        locked = [circle for circle in supplied if circle.locked]
        detection_started = time.perf_counter()
        max_candidates = max(
            1,
            min(1500, int(data.get("max_candidates", 500))),
        )
        candidates = detect_multiscale_circles(
            result,
            locked_circles=locked,
            max_candidates=max_candidates,
        )
        elapsed_seconds = time.perf_counter() - detection_started
        side_radius_ranges = effective_radius_ranges(
            result.profile,
            result.source_to_processed,
        )
        main_minimum, main_maximum = effective_main_arm_radius_range(
            result.profile
        )
        automatic_candidates = [
            circle for circle in candidates if not circle.locked
        ]
        main_variable_count = sum(
            circle.circle_type == "main"
            for circle in automatic_candidates
        )
        return jsonify(
            {
                "circles": [circle.to_dict() for circle in candidates],
                "count": len(candidates),
                "locked_count": len(locked),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "radius_ranges": {
                    name: {
                        "min_radius": side_radius_ranges[name][0],
                        "max_radius": side_radius_ranges[name][1],
                        "role": "satellite_standard",
                    }
                    for name in ("small", "medium", "large")
                }
                | {
                    "main_variable": {
                        "min_radius": main_minimum,
                        "max_radius": main_maximum,
                        "role": "main_arm_variable",
                        "enabled": bool(
                            result.profile.detection.get("main_arm", {}).get(
                                "enabled", True
                            )
                        ),
                    }
                },
                "detection_counts": {
                    "main_variable": main_variable_count,
                    "side_standard": (
                        len(automatic_candidates) - main_variable_count
                    ),
                },
                "confidence_components": [
                    "edge_score",
                    "color_score",
                    "circularity_score",
                    "detector_agreement",
                    "symmetry_support",
                    "final_confidence",
                ],
                "notice": (
                    "Üç ortak yan-daire standardı ile değişken çaplı ana-kol "
                    "geçişi bağımsız çalıştırıldı; kilitli daireler değiştirilmedi."
                ),
            }
        )

    @app.post("/api/analyze")
    def analyze() -> Response:
        data = _json_body()
        image_id = str(data.get("image_id") or "")
        center = _point(data.get("center"), "center")
        phase = float(data.get("phase_degrees", 0.0))
        arm_model, arm_twist = _arm_settings(data)
        circles = _circles(data.get("circles", []))
        manual_locks = (
            _manual_locks(data.get("manual_locks"))
            if "manual_locks" in data
            else None
        )
        _, image = store.read(image_id)
        classified, image_errors, geometry_errors, summary = _analysis(
            image,
            circles,
            center,
            phase,
            manual_locks,
            arm_model,
            arm_twist,
        )
        return jsonify(
            {
                "circles": [circle.to_dict() for circle in classified],
                "image_symmetry": image_errors,
                "geometric_symmetry": geometry_errors,
                "summary": summary,
                "arm_model": arm_model,
                "arm_twist_degrees_per_100px": arm_twist,
            }
        )

    @app.post("/api/arms/suggest")
    def suggest_arm_model() -> Response:
        data = _json_body()
        image_id = str(data.get("image_id") or "")
        store.get(image_id)
        center = _point(data.get("center"), "center")
        circles = _circles(data.get("circles", []))
        phase_hint = float(data.get("phase_degrees", 0.0))
        if not math.isfinite(phase_hint):
            raise ValueError("Kol fazı sonlu olmalı")
        suggestion = estimate_spiral_parameters(
            circles,
            center,
            phase_hint_degrees=phase_hint,
        )
        return jsonify(suggestion.to_dict())

    @app.post("/api/structure/analyze")
    def analyze_structure() -> Response:
        """Return structural suggestions without mutating confirmed fields."""

        data = _json_body()
        image_id = str(data.get("image_id") or "")
        store.get(image_id)
        center = _point(data.get("center"), "center")
        phase = float(data.get("phase_degrees", 0.0))
        if not math.isfinite(phase):
            raise ValueError("Kol fazı sonlu olmalı")
        arm_model, arm_twist = _arm_settings(data)
        circles = _circles(data.get("circles", []))
        manual_locks = _manual_locks(data.get("manual_locks"))
        structure_state = _structure_state(data.get("structure_state"))
        return jsonify(
            structural_report(
                circles,
                center,
                phase,
                model=arm_model,
                twist_degrees_per_100px=arm_twist,
                manual_locks=manual_locks,
                structure_state=structure_state,
            )
        )

    @app.post("/api/circles/suspects")
    def suggest_weak_circles() -> Response:
        data = _json_body()
        image_id = str(data.get("image_id") or "")
        circles = _circles(data.get("circles", []))
        if not circles:
            raise ValueError("İncelemek için en az bir daire gerekli")
        _, image = store.read(image_id)
        suggestions = rank_weak_visual_support(
            image,
            circles,
            limit=min(20, max(1, int(data.get("limit", 5)))),
        )
        return jsonify(
            {
                "suggestions": [
                    suggestion.to_dict() for suggestion in suggestions
                ],
                "notice": (
                    "Bunlar yalnızca düşük görsel destek önerileridir; "
                    "hiçbir daire otomatik silinmez."
                ),
            }
        )

    @app.post("/api/preprocess/export")
    def export_preprocess_artifacts() -> Response:
        data = _json_body()
        image_id = str(data.get("image_id") or "")
        record = store.ensure_preprocess(image_id)
        result = record.preprocess_result
        if result is None:
            raise ValueError("Ön işleme sonucu bulunamadı")
        circles = _circles(data.get("circles", []))
        center_data = data.get("center")
        center = (
            None if center_data is None else _point(center_data, "center")
        )
        phase = float(data.get("phase_degrees", 0.0))
        arm_model, arm_twist = _arm_settings(data)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        stem = secure_filename(Path(record.display_name).stem) or "analysis"
        relative_dir = Path(f"{stem}-{stamp}-preprocess")
        destination_dir = output_root / relative_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        rendered = draw_overlay(
            result.measurement_image,
            circles,
            center=center,
            phase_degrees=phase,
            arm_model=arm_model,
            arm_twist_degrees_per_100px=arm_twist,
            draw_labels=False,
        )
        images: dict[str, NDArray[np.uint8]] = {
            "rectified.png": result.rectified,
            "illumination_corrected.png": (
                result.illumination_corrected
            ),
            "foreground_probability.png": np.clip(
                result.foreground_probability * 255.0, 0, 255
            ).astype(np.uint8),
            "exclusion_mask.png": result.exclusion_mask,
            "detection_overlay.png": rendered,
        }
        files: dict[str, dict[str, str]] = {}
        for filename, image in images.items():
            destination = destination_dir / filename
            if not cv2.imwrite(str(destination), image):
                raise OSError(f"Ara çıktı yazılamadı: {filename}")
            relative_name = str(relative_dir / filename)
            key = Path(filename).stem
            files[key] = {
                "name": filename,
                "url": url_for(
                    "download_output", filename=relative_name
                ),
            }
        profile_destination = destination_dir / "preprocess_profile.json"
        write_preprocess_profile(result.profile, profile_destination)
        return jsonify({"files": files})

    @app.post("/api/project/save")
    def save_project() -> Response:
        data = _json_body()
        image_id = str(data.get("image_id") or "")
        record = store.get(image_id)
        circles = _circles(data.get("circles", []))
        manual_locks = _manual_locks(data.get("manual_locks"))
        deleted_ids, suppressed_geometry = _editor_state(
            data.get("editor_state")
        )
        edit_history = _edit_history(data.get("edit_history"))
        structure_state = _structure_state(data.get("structure_state"))
        center_data = data.get("center")
        center = (
            None if center_data is None else _point(center_data, "center")
        )
        phase = float(data.get("phase_degrees", 0.0))
        arm_model, arm_twist = _arm_settings(data)
        document = _annotation_document(
            record,
            circles,
            center=center,
            phase=phase,
            arm_model=arm_model,
            arm_twist=arm_twist,
            manual_locks=manual_locks,
            deleted_automatic_ids=deleted_ids,
            suppressed_geometry=suppressed_geometry,
            edit_history=edit_history,
            structure_state=structure_state,
        )
        document.validate()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        stem = secure_filename(Path(record.display_name).stem) or "analysis"
        filename = f"{stem}-{stamp}.work.json"
        destination = output_root / filename
        annotation_destination = annotation_root / filename
        write_json(document, destination)
        write_json(document, annotation_destination)
        return jsonify(
            {
                "file": {
                    "name": filename,
                    "url": url_for(
                        "download_output",
                        filename=filename,
                    ),
                },
                "circle_count": len(circles),
                "schema_version": document.schema_version,
            }
        )

    @app.post("/api/project/load")
    def load_project() -> Response:
        image_id = str(request.form.get("image_id") or "")
        uploaded = request.files.get("project")
        if uploaded is None:
            raise ValueError("Bir çalışma JSON dosyası seçin")
        record, source_image = store.read_source(image_id)
        uploaded_bytes = uploaded.read()
        backup_dir = annotation_root / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup_stem = (
            secure_filename(Path(uploaded.filename or "project").stem)
            or "project"
        )
        backup_path = backup_dir / (
            f"{backup_stem}-{backup_stamp}.pre-load.backup.json"
        )
        backup_path.write_bytes(uploaded_bytes)
        try:
            raw = json.loads(uploaded_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Çalışma dosyası geçerli bir UTF-8 JSON değil"
            ) from exc
        if not isinstance(raw, Mapping):
            raise ValueError("Çalışma JSON kökü bir nesne olmalı")
        try:
            document = AnnotationSet.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Çalışma JSON'u geçerli daire/anotasyon alanları içermiyor"
            ) from exc
        source_schema_version = document.schema_version
        if document.preprocess_profile is not None:
            profile = _preprocess_profile_from_data(
                document.preprocess_profile,
                source_image,
            )
            if (
                profile.source_width != source_image.shape[1]
                or profile.source_height != source_image.shape[0]
            ):
                raise ValueError(
                    "Çalışma profili açık kaynak görüntü boyutlarıyla eşleşmiyor"
                )
            document.preprocess_profile = profile.to_dict()
        warning = None
        if (
            document.image_name
            and document.image_name != record.display_name
        ):
            warning = (
                f"Dosya {document.image_name!r} için kaydedilmiş; "
                f"açık görüntü {record.display_name!r}."
            )
        document.schema_version = "1.4"
        return jsonify(
            {
                "document": document.to_dict(),
                "warning": warning,
                "source_schema_version": source_schema_version,
                "backup": {
                    "name": backup_path.name,
                    "relative_path": str(
                        backup_path.relative_to(data_root)
                    ),
                },
            }
        )

    @app.post("/api/export")
    def export() -> Response:
        data = _json_body()
        image_id = str(data.get("image_id") or "")
        record, image = store.read(image_id)
        circles = _circles(data.get("circles", []))
        manual_locks = (
            _manual_locks(data.get("manual_locks"))
            if "manual_locks" in data
            else None
        )
        phase = float(data.get("phase_degrees", 0.0))
        arm_model, arm_twist = _arm_settings(data)
        deleted_automatic_ids, suppressed_geometry = _editor_state(
            data.get("editor_state")
        )
        edit_history = _edit_history(data.get("edit_history"))
        structure_state = _structure_state(data.get("structure_state"))
        center_data = data.get("center")
        center: tuple[float, float] | None = None
        image_errors: list[dict[str, Any]] = []
        geometry_errors: list[dict[str, Any]] = []
        summary: dict[str, float] = {}
        if center_data is not None:
            center = _point(center_data, "center")
            circles, image_errors, geometry_errors, summary = (
                _measure_without_reclassification(
                    image,
                    circles,
                    center,
                )
            )
        document = _annotation_document(
            record,
            circles,
            center=center,
            phase=phase,
            arm_model=arm_model,
            arm_twist=arm_twist,
            manual_locks=manual_locks,
            deleted_automatic_ids=deleted_automatic_ids,
            suppressed_geometry=suppressed_geometry,
            image_errors=image_errors,
            geometry_errors=geometry_errors,
            edit_history=edit_history,
            structure_state=structure_state,
        )
        document.validate()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        stem = secure_filename(Path(record.display_name).stem) or "analysis"
        base_name = f"{stem}-{stamp}"
        png_path = output_root / f"{base_name}.png"
        csv_path = output_root / f"{base_name}.csv"
        json_path = output_root / f"{base_name}.json"
        annotation_path = annotation_root / f"{base_name}.json"
        rendered = draw_overlay(
            image,
            circles,
            center=center,
            phase_degrees=phase,
            arm_model=arm_model,
            arm_twist_degrees_per_100px=arm_twist,
            draw_labels=False,
        )
        if not cv2.imwrite(str(png_path), rendered):
            raise OSError("Sonuç PNG dosyası yazılamadı")
        write_csv(circles, csv_path)
        write_json(document, json_path)
        write_json(document, annotation_path)
        if center is None:
            if circles:
                structure_center = (
                    sum(circle.x for circle in circles) / len(circles),
                    sum(circle.y for circle in circles) / len(circles),
                )
            else:
                structure_center = (
                    image.shape[1] / 2.0,
                    image.shape[0] / 2.0,
                )
        else:
            structure_center = center
        structure_relative_dir = Path(f"{base_name}-structure")
        structure_export = write_structure_exports(
            circles,
            output_root / structure_relative_dir,
            center=structure_center,
            phase_degrees=phase,
            model=arm_model,
            twist_degrees_per_100px=arm_twist,
            manual_locks=manual_locks,
            structure_state=structure_state,
        )
        structure_files = {
            key: {
                "name": path.name,
                "url": url_for(
                    "download_output",
                    filename=str(structure_relative_dir / path.name),
                ),
            }
            for key, path in structure_export.files.items()
        }
        return jsonify(
            {
                "files": {
                    "png": {
                        "name": png_path.name,
                        "url": url_for("download_output", filename=png_path.name),
                    },
                    "csv": {
                        "name": csv_path.name,
                        "url": url_for("download_output", filename=csv_path.name),
                    },
                    "json": {
                        "name": json_path.name,
                        "url": url_for("download_output", filename=json_path.name),
                    },
                    **structure_files,
                },
                "summary": summary,
                "structure_summary": structure_export.summary,
                "circles": [circle.to_dict() for circle in circles],
            }
        )

    @app.get("/downloads/<path:filename>")
    def download_output(filename: str) -> Response:
        return send_from_directory(output_root, filename, as_attachment=True)

    @app.errorhandler(ValueError)
    def value_error(error: ValueError) -> tuple[Response, int]:
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(FileNotFoundError)
    def not_found_error(error: FileNotFoundError) -> tuple[Response, int]:
        return jsonify({"error": str(error)}), 404

    @app.errorhandler(413)
    def too_large(_error: Exception) -> tuple[Response, int]:
        return jsonify({"error": "Dosya 40 MB sınırını aşıyor"}), 413

    return app


def main() -> None:
    app = create_app()
    print("Crop Geometry Lab: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
