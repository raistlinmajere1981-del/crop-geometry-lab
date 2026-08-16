"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const canvas = $("#analysisCanvas");
const context = canvas.getContext("2d");
const canvasWrap = $("#canvasWrap");

const ARM_COLORS = {
  1: "#ff4444",
  2: "#ffb428",
  3: "#78dc46",
  4: "#37bee6",
  5: "#aa5aff",
  6: "#ff4baa",
};

const LOCAL_SIDE_LABELS = {
  inner: "İç",
  outer: "Dış",
  centerline: "Kol ekseni",
  unknown: "Bilinmiyor",
  center: "Merkez",
  left: "Sol (eski)",
  right: "Sağ (eski)",
};

const LOCAL_SIDE_METHOD_LABELS = {
  main_chain: "Ana Zincir",
  spiral_model: "Spiral model",
  radial_model: "Işınsal model",
  general_center: "Genel merkez",
  missing_arm: "Kol yok",
};

const TOOL_HINTS = {
  select: "Seçili daireyi merkezinden taşıyın veya çemberinden sürükleyerek yarıçapını değiştirin.",
  pan: "Görüntüyü sürükleyin. Fare tekerleği imleç çevresinde yakınlaştırır.",
  add: "Merkezden dışarı sürükleyerek yeni bir manuel daire ekleyin.",
  delete: "Silmek istediğiniz daireye tıklayın. Bu karar otomatik tespitte korunur.",
  center: "Altılı analiz için genel merkezi görüntü üzerinde seçin.",
  perspective: "Perspektif için dört noktayı sırayla görüntü üzerinde işaretleyin.",
  roiPolygon: "Analize alınacak çokgenin köşelerini kaynak görüntü üzerinde işaretleyin.",
  maskPaint: "Algılamanın yok sayacağı alanları fırçayla boyayın.",
  maskErase: "Dışlama maskesini fırçayla geri silin ve düzeltin.",
  samplePaint: "Seçili renk sınıfına ait temsilî alanları boyayın.",
  sizeSample: "Seçili ortak yan-daire boyu için merkezden dışarı bir örnek daire çizin.",
  mainChain: "Seçili kolun ana zinciri için merkezden dışa doğru 13 daireyi sırayla seçin.",
  tramlineEdit: "Bir izi seçin; gövdesini sürükleyerek taşıyın, uç tutamaçlarını sürükleyerek yönünü ve uzunluğunu düzeltin.",
  tramlineAdd: "Yeni izin ilk ucundan ikinci ucuna sürükleyin.",
  tramlineDelete: "Silmek istediğiniz tarla izine tıklayın.",
};

const IDENTITY_MATRIX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const SAMPLE_COLORS = {
  flattened_crop: "#d9f06d",
  standing_crop: "#53c880",
  tramline: "#ffad4e",
  shadow: "#996ce8",
};

const state = {
  workspace: "preprocess",
  imageId: null,
  imageName: null,
  imageWidth: 0,
  imageHeight: 0,
  imageVersion: 0,
  sourceWidth: 0,
  sourceHeight: 0,
  sourceToProcessed: IDENTITY_MATRIX.map((row) => [...row]),
  processedToSource: IDENTITY_MATRIX.map((row) => [...row]),
  imageElement: null,
  circles: [],
  selectedId: null,
  deletedAutoIds: new Set(),
  suppressedAutomaticGeometry: [],
  center: null,
  centerSource: null,
  phaseDegrees: 0,
  armModel: "radial",
  armTwistDegreesPer100px: 0,
  armSuggestion: null,
  perspectivePoints: [],
  tool: "select",
  view: { scale: 1, tx: 0, ty: 0 },
  viewport: { width: 1, height: 1, dpr: 1 },
  pointer: null,
  previewCircle: null,
  spaceDown: false,
  manualCounter: 0,
  showArms: true,
  showLabels: false,
  labelMode: "selected",
  structureReport: null,
  structureUndo: [],
  editHistory: [],
  mainChains: { 1: [], 2: [], 3: [], 4: [], 5: [], 6: [] },
  mainChainDraft: null,
  boundarySelection: new Set(),
  localSideSelection: new Set(),
  satelliteSelection: new Set(),
  preserveLoadedCircleSet: false,
  preprocessProfile: null,
  defaultPreprocessProfile: null,
  preprocessHistory: [],
  preprocessSteps: [],
  preprocessArtifacts: {},
  preprocessApplied: false,
  appliedPreprocessSignature: null,
  showExclusionMask: true,
  showRoi: true,
  maskOpacity: 0.45,
  tramlineSuggestion: null,
  selectedTramlineId: null,
  tramlineHistory: [],
  tramlineCounter: 0,
  stepPreviewOpen: false,
  perspectiveQuality: null,
  perspectiveQualityArtifacts: {},
  perspectiveQualityStale: false,
  showQualityEllipses: false,
  showQualityBenefits: false,
};

function setStatus(message, kind = "") {
  const element = $("#status");
  element.textContent = message;
  element.className = `status ${kind}`.trim();
}

function setBusy(isBusy, message = "İşleniyor…") {
  $("#busyText").textContent = message;
  $("#busyOverlay").classList.toggle("hidden", !isBusy);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(payload.error || `İstek başarısız (${response.status})`);
  }
  return payload;
}

function selectedCircle() {
  return state.circles.find((circle) => circle.id === state.selectedId) || null;
}

function normalizeCircle(raw) {
  const lockedFields = [
    ...(Array.isArray(raw.locked_fields) ? raw.locked_fields : []),
    ...(Array.isArray(raw._locked_fields) ? raw._locked_fields : []),
  ].filter((field, index, values) => values.indexOf(field) === index);
  const suggestedNumber = (value) =>
    value === null || value === "" || value === undefined ? null : Number(value);
  const circle = {
    id: String(raw.id),
    x: Number(raw.x),
    y: Number(raw.y),
    radius: Number(raw.radius),
    confidence: Number(raw.confidence ?? 1),
    circle_type: raw.circle_type || "main",
    parent_id: raw.parent_id || null,
    arm_id: raw.arm_id === null || raw.arm_id === "" || raw.arm_id === undefined ? null : Number(raw.arm_id),
    sequence_index:
      raw.sequence_index === null || raw.sequence_index === "" || raw.sequence_index === undefined
        ? null
        : Number(raw.sequence_index),
    side: raw.side || null,
    legacy_side: raw.legacy_side || (["left", "right"].includes(raw.side) ? raw.side : null),
    local_angle:
      raw.local_angle === null || raw.local_angle === "" || raw.local_angle === undefined
        ? null
        : Number(raw.local_angle),
    normalized_parent_distance:
      raw.normalized_parent_distance === null ||
      raw.normalized_parent_distance === "" ||
      raw.normalized_parent_distance === undefined
        ? null
        : Number(raw.normalized_parent_distance),
    classification_source: raw.classification_source || "legacy",
    classification_confidence: Number(raw.classification_confidence ?? 0),
    locked_fields: [...lockedFields],
    suggested_circle_type: raw.suggested_circle_type || null,
    suggested_arm_id: suggestedNumber(raw.suggested_arm_id),
    suggested_sequence_index: suggestedNumber(raw.suggested_sequence_index),
    suggested_parent_id: raw.suggested_parent_id || null,
    suggested_side: raw.suggested_side || null,
    suggestion_confidence: suggestedNumber(raw.suggestion_confidence),
    center_distance:
      raw.center_distance === null || raw.center_distance === "" || raw.center_distance === undefined
        ? null
        : Number(raw.center_distance),
    center_angle:
      raw.center_angle === null || raw.center_angle === "" || raw.center_angle === undefined
        ? null
        : Number(raw.center_angle),
    processed_x: Number(raw.processed_x ?? raw.x),
    processed_y: Number(raw.processed_y ?? raw.y),
    source_x: Number(raw.source_x ?? raw.x),
    source_y: Number(raw.source_y ?? raw.y),
    source_radius:
      raw.source_radius === null || raw.source_radius === "" || raw.source_radius === undefined
        ? null
        : Number(raw.source_radius),
    corrected_x: Number(raw.corrected_x ?? raw.processed_x ?? raw.x),
    corrected_y: Number(raw.corrected_y ?? raw.processed_y ?? raw.y),
    corrected_radius: Number(raw.corrected_radius ?? raw.radius),
    corrected_major_axis:
      raw.corrected_major_axis === null || raw.corrected_major_axis === undefined
        ? null
        : Number(raw.corrected_major_axis),
    corrected_minor_axis:
      raw.corrected_minor_axis === null || raw.corrected_minor_axis === undefined
        ? null
        : Number(raw.corrected_minor_axis),
    source_major_axis:
      raw.source_major_axis === null || raw.source_major_axis === undefined
        ? null
        : Number(raw.source_major_axis),
    source_minor_axis:
      raw.source_minor_axis === null || raw.source_minor_axis === undefined
        ? null
        : Number(raw.source_minor_axis),
    edge_score: raw.edge_score === null || raw.edge_score === undefined ? null : Number(raw.edge_score),
    color_score: raw.color_score === null || raw.color_score === undefined ? null : Number(raw.color_score),
    circularity_score: raw.circularity_score === null || raw.circularity_score === undefined ? null : Number(raw.circularity_score),
    detector_agreement: raw.detector_agreement === null || raw.detector_agreement === undefined ? null : Number(raw.detector_agreement),
    symmetry_support: raw.symmetry_support === null || raw.symmetry_support === undefined ? null : Number(raw.symmetry_support),
    final_confidence: Number(raw.final_confidence ?? raw.confidence ?? 1),
    stroke_color: raw.stroke_color || null,
    stroke_width:
      raw.stroke_width === null || raw.stroke_width === "" || raw.stroke_width === undefined
        ? null
        : Number(raw.stroke_width),
    fill_enabled: Boolean(raw.fill_enabled),
    fill_color: raw.fill_color || null,
    fill_opacity:
      raw.fill_opacity === null || raw.fill_opacity === "" || raw.fill_opacity === undefined
        ? 0.2
        : Math.max(0, Math.min(1, Number(raw.fill_opacity))),
    size_class: raw.size_class || null,
    detector: raw.detector || null,
    locked: Boolean(raw.locked),
    source: raw.source || "manual",
    _locked_fields: [...lockedFields],
  };
  if (circle.source === "automatic") {
    circle._automatic_origin = raw._automatic_origin || {
      id: circle.id,
      x: circle.x,
      y: circle.y,
      radius: circle.radius,
    };
  } else if (raw._automatic_origin) {
    circle._automatic_origin = { ...raw._automatic_origin };
  }
  return circle;
}

function resetAnnotations() {
  state.circles = [];
  state.selectedId = null;
  state.deletedAutoIds.clear();
  state.suppressedAutomaticGeometry = [];
  state.center = null;
  state.centerSource = null;
  state.phaseDegrees = 0;
  state.armModel = "radial";
  state.armTwistDegreesPer100px = 0;
  state.armSuggestion = null;
  state.structureReport = null;
  state.structureUndo = [];
  state.editHistory = [];
  state.mainChains = { 1: [], 2: [], 3: [], 4: [], 5: [], 6: [] };
  state.mainChainDraft = null;
  state.boundarySelection = new Set();
  state.localSideSelection = new Set();
  state.satelliteSelection = new Set();
  state.preserveLoadedCircleSet = false;
  state.labelMode = "selected";
  state.perspectivePoints = [];
  state.previewCircle = null;
  state.sourceToProcessed = IDENTITY_MATRIX.map((row) => [...row]);
  state.processedToSource = IDENTITY_MATRIX.map((row) => [...row]);
  state.preprocessProfile = null;
  state.defaultPreprocessProfile = null;
  state.preprocessHistory = [];
  state.preprocessSteps = [];
  state.preprocessArtifacts = {};
  state.preprocessApplied = false;
  state.appliedPreprocessSignature = null;
  state.tramlineSuggestion = null;
  state.selectedTramlineId = null;
  state.tramlineHistory = [];
  state.tramlineCounter = 0;
  state.perspectiveQuality = null;
  state.perspectiveQualityArtifacts = {};
  state.perspectiveQualityStale = false;
  state.showQualityEllipses = false;
  state.showQualityBenefits = false;
  $("#phaseDegrees").value = "0";
  $("#armModel").value = "radial";
  $("#armTwist").value = "0";
  $("#applyArmSuggestion").classList.add("hidden");
  $("#armFitReadout").textContent = "Öneri uygulanmadan önce kullanıcı onayı gerekir.";
  if ($("#showQualityEllipses")) $("#showQualityEllipses").checked = false;
  if ($("#showQualityBenefits")) $("#showQualityBenefits").checked = false;
  if ($("#labelMode")) $("#labelMode").value = "selected";
  updateArmModelUi();
  clearSymmetry();
  updateUi();
}

async function useImageRecord(record, { reset = true } = {}) {
  state.imageId = record.image_id;
  state.imageName = record.name;
  state.imageWidth = Number(record.width);
  state.imageHeight = Number(record.height);
  state.imageVersion = Number(record.version);
  if (reset) {
    resetAnnotations();
  }
  state.sourceWidth = Number(record.source_width || record.width);
  state.sourceHeight = Number(record.source_height || record.height);
  if (record.source_to_processed) {
    state.sourceToProcessed = record.source_to_processed.map((row) => row.map(Number));
  }
  if (record.processed_to_source) {
    state.processedToSource = record.processed_to_source.map((row) => row.map(Number));
  }
  const image = new Image();
  const loadPromise = new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error("Görüntü tarayıcıya yüklenemedi"));
  });
  image.src = `${record.image_url}${record.image_url.includes("?") ? "&" : "?"}cache=${Date.now()}`;
  await loadPromise;
  state.imageElement = image;
  state.imageWidth = image.naturalWidth;
  state.imageHeight = image.naturalHeight;
  if (!reset) {
    reprojectCircles();
    reprojectTramlineSuggestion();
    if (state.centerSource) {
      state.center = transformCoordinate(state.centerSource, state.sourceToProcessed);
    }
  }
  $("#emptyState").classList.add("hidden");
  $("#imageInfo").innerHTML = `<strong>${escapeHtml(state.imageName)}</strong><br>${state.imageWidth} × ${state.imageHeight} px · çalışma sürümü ${state.imageVersion}`;
  fitView();
  updateUi();
}

function escapeHtml(text) {
  const wrapper = document.createElement("span");
  wrapper.textContent = String(text);
  return wrapper.innerHTML;
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function transformCoordinate(point, matrix) {
  const x = Number(point.x);
  const y = Number(point.y);
  const denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
  if (Math.abs(denominator) < 1e-12) return { x, y };
  return {
    x: (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
    y: (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator,
  };
}

function processedToSourceCoordinate(point) {
  return transformCoordinate(point, state.processedToSource);
}

function sourceToProcessedCoordinate(point) {
  return transformCoordinate(point, state.sourceToProcessed);
}

function processedRadiusToSource(center, radius) {
  const sourceCenter = processedToSourceCoordinate(center);
  const sourceEdgeX = processedToSourceCoordinate({ x: center.x + radius, y: center.y });
  const sourceEdgeY = processedToSourceCoordinate({ x: center.x, y: center.y + radius });
  const radiusX = Math.hypot(sourceEdgeX.x - sourceCenter.x, sourceEdgeX.y - sourceCenter.y);
  const radiusY = Math.hypot(sourceEdgeY.x - sourceCenter.x, sourceEdgeY.y - sourceCenter.y);
  return Math.max(0.1, Math.sqrt(Math.max(0.01, radiusX * radiusY)));
}

function sourceRadiusToProcessed(center, radius) {
  const processedCenter = sourceToProcessedCoordinate(center);
  const processedEdgeX = sourceToProcessedCoordinate({ x: center.x + radius, y: center.y });
  const processedEdgeY = sourceToProcessedCoordinate({ x: center.x, y: center.y + radius });
  const radiusX = Math.hypot(processedEdgeX.x - processedCenter.x, processedEdgeX.y - processedCenter.y);
  const radiusY = Math.hypot(processedEdgeY.x - processedCenter.x, processedEdgeY.y - processedCenter.y);
  return Math.max(0.1, Math.sqrt(Math.max(0.01, radiusX * radiusY)));
}

function updateCircleSourceCoordinates(circle) {
  const source = processedToSourceCoordinate({ x: circle.x, y: circle.y });
  circle.processed_x = circle.x;
  circle.processed_y = circle.y;
  circle.source_x = source.x;
  circle.source_y = source.y;
  circle.source_radius = processedRadiusToSource(
    { x: circle.x, y: circle.y },
    circle.radius,
  );
  circle.corrected_x = circle.x;
  circle.corrected_y = circle.y;
  circle.corrected_radius = circle.radius;
  circle.source_major_axis = circle.source_radius * 2;
  circle.source_minor_axis = circle.source_radius * 2;
  circle.corrected_major_axis = circle.radius * 2;
  circle.corrected_minor_axis = circle.radius * 2;
  if (state.perspectiveQuality) state.perspectiveQualityStale = true;
}

function reprojectCircles() {
  for (const circle of state.circles) {
    const source = {
      x: Number(circle.source_x ?? circle.x),
      y: Number(circle.source_y ?? circle.y),
    };
    const processed = sourceToProcessedCoordinate(source);
    circle.x = processed.x;
    circle.y = processed.y;
    circle.processed_x = processed.x;
    circle.processed_y = processed.y;
    const sourceRadius = Number(
      circle.source_radius
      ?? processedRadiusToSource(
        {
          x: Number(circle.corrected_x ?? circle.x),
          y: Number(circle.corrected_y ?? circle.y),
        },
        Number(circle.corrected_radius ?? circle.radius),
      ),
    );
    circle.source_radius = sourceRadius;
    circle.radius = sourceRadiusToProcessed(source, sourceRadius);
    circle.corrected_x = processed.x;
    circle.corrected_y = processed.y;
    circle.corrected_radius = circle.radius;
  }
}

function tramlineAngle(line) {
  return (
    Math.atan2(
      Number(line.p2.y) - Number(line.p1.y),
      Number(line.p2.x) - Number(line.p1.x),
    ) * 180 / Math.PI + 180
  ) % 180;
}

function tramlineLength(line) {
  return Math.hypot(
    Number(line.p2.x) - Number(line.p1.x),
    Number(line.p2.y) - Number(line.p1.y),
  );
}

function selectedTramline() {
  return state.tramlineSuggestion?.lines?.find(
    (line) => line.id === state.selectedTramlineId,
  ) || null;
}

function syncTramlineSourceGeometry(line) {
  line.p1 = { x: Number(line.p1.x), y: Number(line.p1.y) };
  line.p2 = { x: Number(line.p2.x), y: Number(line.p2.y) };
  line.source_p1 = processedToSourceCoordinate(line.p1);
  line.source_p2 = processedToSourceCoordinate(line.p2);
  line.angle_degrees = tramlineAngle(line);
  line.length = tramlineLength(line);
}

function pushTramlineHistory() {
  state.tramlineHistory.push(
    state.tramlineSuggestion ? deepClone(state.tramlineSuggestion) : null,
  );
  if (state.tramlineHistory.length > 50) state.tramlineHistory.shift();
}

function normalizeTramlineSuggestion(raw) {
  const suggestion = {
    ...(raw || {}),
    lines: [],
    dominant_angle_degrees: Number(raw?.dominant_angle_degrees || 0),
    angle_dispersion_degrees: Number(raw?.angle_dispersion_degrees || 0),
    confidence: Number(raw?.confidence || 0),
    spacing_pixels: Number(raw?.spacing_pixels || 0),
    spacing_consistency: Number(raw?.spacing_consistency || 0),
    orientation_confidence: Number(raw?.orientation_confidence || 0),
    perspective_points_processed: raw?.perspective_points_processed || [],
    perspective_points_source: raw?.perspective_points_source || [],
    requires_user_approval: true,
  };
  suggestion.lines = (raw?.lines || []).map((rawLine, index) => {
    const line = {
      ...rawLine,
      id: rawLine.id || `tramline-${Date.now().toString(36)}-${index + 1}`,
      p1: {
        x: Number(rawLine.p1?.x ?? rawLine.source_p1?.x ?? 0),
        y: Number(rawLine.p1?.y ?? rawLine.source_p1?.y ?? 0),
      },
      p2: {
        x: Number(rawLine.p2?.x ?? rawLine.source_p2?.x ?? 0),
        y: Number(rawLine.p2?.y ?? rawLine.source_p2?.y ?? 0),
      },
      source_p1: rawLine.source_p1
        ? { x: Number(rawLine.source_p1.x), y: Number(rawLine.source_p1.y) }
        : null,
      source_p2: rawLine.source_p2
        ? { x: Number(rawLine.source_p2.x), y: Number(rawLine.source_p2.y) }
        : null,
      support: Math.max(0, Math.min(1, Number(rawLine.support ?? 1))),
      manual: Boolean(rawLine.manual),
    };
    if (!line.source_p1 || !line.source_p2) syncTramlineSourceGeometry(line);
    return line;
  });
  return suggestion;
}

function orientationDistance(left, right) {
  const difference = Math.abs(left - right) % 180;
  return Math.min(difference, 180 - difference);
}

function recomputeTramlineSuggestion({ edited = true, refresh = true } = {}) {
  const suggestion = state.tramlineSuggestion;
  if (!suggestion) {
    updateTramlineUi();
    return;
  }
  suggestion.lines = (suggestion.lines || []).filter(
    (line) => line?.p1 && line?.p2 && tramlineLength(line) >= 1,
  );
  if (!suggestion.lines.length) {
    suggestion.dominant_angle_degrees = 0;
    suggestion.angle_dispersion_degrees = 180;
    suggestion.spacing_pixels = 0;
    suggestion.spacing_consistency = 0;
    suggestion.confidence = 0;
    suggestion.perspective_points_processed = [];
    suggestion.perspective_points_source = [];
    if (edited) suggestion.edited = true;
    updateTramlineUi();
    if (refresh) render();
    return;
  }
  let axialX = 0;
  let axialY = 0;
  for (const line of suggestion.lines) {
    syncTramlineSourceGeometry(line);
    const angle = tramlineAngle(line) * Math.PI / 180;
    const weight = Math.max(1, tramlineLength(line));
    axialX += Math.cos(angle * 2) * weight;
    axialY += Math.sin(angle * 2) * weight;
  }
  const dominant = (
    Math.atan2(axialY, axialX) * 90 / Math.PI + 180
  ) % 180;
  const radians = dominant * Math.PI / 180;
  const normal = { x: -Math.sin(radians), y: Math.cos(radians) };
  const offsetOf = (line) => (
    (Number(line.p1.x) + Number(line.p2.x)) * 0.5 * normal.x
    + (Number(line.p1.y) + Number(line.p2.y)) * 0.5 * normal.y
  );
  suggestion.lines.sort((left, right) => offsetOf(left) - offsetOf(right));
  const offsets = suggestion.lines.map(offsetOf);
  const differences = offsets.slice(1).map((value, index) => value - offsets[index]);
  const sortedDifferences = [...differences].sort((left, right) => left - right);
  const spacing = sortedDifferences.length
    ? sortedDifferences[Math.floor(sortedDifferences.length / 2)]
    : 0;
  const signedAngles = suggestion.lines.map((line) => {
    const angle = tramlineAngle(line);
    return ((((angle - dominant + 90) % 180) + 180) % 180) - 90;
  });
  const meanAngle = signedAngles.reduce((sum, value) => sum + value, 0) / signedAngles.length;
  const dispersion = Math.sqrt(
    signedAngles.reduce((sum, value) => sum + (value - meanAngle) ** 2, 0)
      / signedAngles.length,
  );
  const spacingConsistency = differences.length
    ? Math.max(
        0,
        Math.min(
          1,
          1 - (
            [...differences]
              .map((value) => Math.abs(value - spacing))
              .sort((left, right) => left - right)[Math.floor(differences.length / 2)]
            / Math.max(spacing, 1)
          ),
        ),
      )
    : 0;
  suggestion.dominant_angle_degrees = dominant;
  suggestion.angle_dispersion_degrees = dispersion;
  suggestion.spacing_pixels = spacing;
  suggestion.spacing_consistency = spacingConsistency;
  suggestion.confidence = Math.max(
    0,
    Math.min(
      1,
      Math.min(1, suggestion.lines.length / 6)
        * Math.max(0, 1 - dispersion / 20)
        * (0.65 + spacingConsistency * 0.35),
    ),
  );
  if (suggestion.lines.length >= 2) {
    const first = suggestion.lines[0];
    const last = suggestion.lines[suggestion.lines.length - 1];
    let lastP1 = last.p1;
    let lastP2 = last.p2;
    const aligned = Math.hypot(first.p1.x - lastP1.x, first.p1.y - lastP1.y)
      + Math.hypot(first.p2.x - lastP2.x, first.p2.y - lastP2.y);
    const crossed = Math.hypot(first.p1.x - lastP2.x, first.p1.y - lastP2.y)
      + Math.hypot(first.p2.x - lastP1.x, first.p2.y - lastP1.y);
    if (crossed < aligned) [lastP1, lastP2] = [lastP2, lastP1];
    suggestion.perspective_points_processed = [
      { ...first.p1 },
      { ...first.p2 },
      { ...lastP1 },
      { ...lastP2 },
    ];
    suggestion.perspective_points_source = suggestion.perspective_points_processed
      .map(processedToSourceCoordinate);
  } else {
    suggestion.perspective_points_processed = [];
    suggestion.perspective_points_source = [];
  }
  if (edited) suggestion.edited = true;
  if (edited && state.perspectiveQuality) {
    state.perspectiveQualityStale = true;
  }
  updateTramlineUi();
  if (refresh) render();
}

function updateTramlineUi() {
  const suggestion = state.tramlineSuggestion;
  const count = suggestion?.lines?.length || 0;
  if (suggestion) {
    const spacing = Number(suggestion.spacing_pixels || 0);
    const regularity = Number(suggestion.spacing_consistency || 0);
    $("#tramlineSuggestionInfo").textContent = count
      ? `${count} iz · yön ${Number(suggestion.dominant_angle_degrees || 0).toFixed(1)}° · aralık ${spacing.toFixed(1)} px · düzen ${Math.round(regularity * 100)}% · güven ${Number(suggestion.confidence || 0).toFixed(2)}${suggestion.edited ? " · manuel düzenlendi" : ""}`
      : "Öneride iz kalmadı; yeni iz çizebilir veya öneriyi yeniden üretebilirsiniz.";
  } else {
    $("#tramlineSuggestionInfo").textContent = "Henüz öneri üretilmedi.";
  }
  const selected = selectedTramline();
  $("#selectedTramlineReadout").textContent = selected
    ? `${selected.id} · ${tramlineLength(selected).toFixed(1)} px · ${tramlineAngle(selected).toFixed(1)}°`
    : count
      ? "Bir çizgiyi seçin; gövde taşıma, uçlar yön ve uzunluk düzeltme içindir."
      : "Bir öneri üretin; sonra çizgiyi veya uç tutamaçlarını sürükleyin.";
  $("#approvePerspectiveSuggestion").disabled =
    suggestion?.perspective_points_source?.length !== 4;
  $("#acceptTramlineMask").disabled = count === 0;
  $("#parallelizeTramlines").disabled = count < 2;
  $("#undoTramlineEdit").disabled = state.tramlineHistory.length === 0;
}

function reprojectTramlineSuggestion() {
  if (!state.tramlineSuggestion?.lines) return;
  for (const line of state.tramlineSuggestion.lines) {
    if (line.source_p1 && line.source_p2) {
      line.p1 = sourceToProcessedCoordinate(line.source_p1);
      line.p2 = sourceToProcessedCoordinate(line.source_p2);
    }
  }
  recomputeTramlineSuggestion({ edited: false, refresh: false });
}

function hydrateTramlineSuggestionFromProfile() {
  const lines = state.preprocessProfile?.tramline?.lines;
  if (state.tramlineSuggestion || !Array.isArray(lines) || !lines.length) return;
  state.tramlineSuggestion = normalizeTramlineSuggestion({
    lines: lines.map((line, index) => ({
      id: `tramline-profile-${index + 1}`,
      p1: sourceToProcessedCoordinate(line.p1),
      p2: sourceToProcessedCoordinate(line.p2),
      source_p1: line.p1,
      source_p2: line.p2,
      support: 1,
      manual: true,
    })),
    requires_user_approval: true,
    edited: false,
  });
  recomputeTramlineSuggestion({ edited: false, refresh: false });
}

function pointToSegmentMeasurement(point, line) {
  const dx = line.p2.x - line.p1.x;
  const dy = line.p2.y - line.p1.y;
  const squaredLength = dx * dx + dy * dy;
  const projection = squaredLength > 0
    ? Math.max(
        0,
        Math.min(
          1,
          ((point.x - line.p1.x) * dx + (point.y - line.p1.y) * dy)
            / squaredLength,
        ),
      )
    : 0;
  const closest = {
    x: line.p1.x + dx * projection,
    y: line.p1.y + dy * projection,
  };
  return {
    distance: Math.hypot(point.x - closest.x, point.y - closest.y),
    projection,
  };
}

function hitTestTramline(point) {
  const lines = state.tramlineSuggestion?.lines || [];
  const endpointTolerance = 11 / state.view.scale;
  const lineTolerance = 8 / state.view.scale;
  const endpointHits = [];
  for (const line of lines) {
    for (const endpoint of ["p1", "p2"]) {
      endpointHits.push({
        line,
        endpoint,
        distance: Math.hypot(
          point.x - line[endpoint].x,
          point.y - line[endpoint].y,
        ),
      });
    }
  }
  const endpointHit = endpointHits
    .filter((hit) => hit.distance <= endpointTolerance)
    .sort((left, right) => left.distance - right.distance)[0];
  if (endpointHit) return { ...endpointHit, mode: "endpoint" };
  const lineHit = lines
    .map((line) => ({ line, ...pointToSegmentMeasurement(point, line) }))
    .filter((hit) => hit.distance <= lineTolerance)
    .sort((left, right) => left.distance - right.distance)[0];
  return lineHit ? { ...lineHit, mode: "line" } : null;
}

function clampImagePoint(point) {
  return {
    x: Math.max(0, Math.min(state.imageWidth - 1, point.x)),
    y: Math.max(0, Math.min(state.imageHeight - 1, point.y)),
  };
}

function extendLineToImageBounds(midpoint, angleDegrees) {
  const angle = angleDegrees * Math.PI / 180;
  const direction = { x: Math.cos(angle), y: Math.sin(angle) };
  const intersections = [];
  for (const [axis, boundary] of [
    ["x", 0],
    ["x", state.imageWidth - 1],
    ["y", 0],
    ["y", state.imageHeight - 1],
  ]) {
    if (Math.abs(direction[axis]) < 1e-9) continue;
    const parameter = (boundary - midpoint[axis]) / direction[axis];
    const point = {
      x: midpoint.x + parameter * direction.x,
      y: midpoint.y + parameter * direction.y,
    };
    if (
      point.x >= -1e-6 && point.x <= state.imageWidth - 1 + 1e-6
      && point.y >= -1e-6 && point.y <= state.imageHeight - 1 + 1e-6
    ) intersections.push(clampImagePoint(point));
  }
  const unique = intersections.filter((point, index) =>
    intersections.findIndex((other) =>
      Math.hypot(point.x - other.x, point.y - other.y) < 0.5
    ) === index
  );
  if (unique.length < 2) return null;
  let best = null;
  for (let first = 0; first < unique.length; first += 1) {
    for (let second = first + 1; second < unique.length; second += 1) {
      const distance = Math.hypot(
        unique[first].x - unique[second].x,
        unique[first].y - unique[second].y,
      );
      if (!best || distance > best.distance) {
        best = { p1: unique[first], p2: unique[second], distance };
      }
    }
  }
  return best;
}

function operationByType(type) {
  return state.preprocessProfile?.operations?.find((operation) => operation.type === type) || null;
}

function pushPreprocessHistory() {
  if (!state.preprocessProfile) return;
  state.preprocessHistory.push(deepClone(state.preprocessProfile));
  if (state.preprocessHistory.length > 50) state.preprocessHistory.shift();
}

function sourceBrushRadius(processedPoint) {
  return processedRadiusToSource(processedPoint, Number($("#brushRadius").value || 16));
}

function strokePoint(point) {
  return { x: point.x, y: point.y };
}

function resizeCanvas() {
  const rect = canvasWrap.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  state.viewport = { width: rect.width, height: rect.height, dpr };
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  render();
}

function fitView() {
  if (!state.imageElement) return;
  const padding = 28;
  const availableWidth = Math.max(1, state.viewport.width - padding * 2);
  const availableHeight = Math.max(1, state.viewport.height - padding * 2);
  const scale = Math.min(availableWidth / state.imageWidth, availableHeight / state.imageHeight);
  state.view.scale = Math.max(0.02, Math.min(20, scale));
  state.view.tx = (state.viewport.width - state.imageWidth * state.view.scale) / 2;
  state.view.ty = (state.viewport.height - state.imageHeight * state.view.scale) / 2;
  render();
}

function imageToScreen(point) {
  return {
    x: point.x * state.view.scale + state.view.tx,
    y: point.y * state.view.scale + state.view.ty,
  };
}

function screenToImage(point) {
  return {
    x: (point.x - state.view.tx) / state.view.scale,
    y: (point.y - state.view.ty) / state.view.scale,
  };
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function insideImage(point) {
  return point.x >= 0 && point.y >= 0 && point.x <= state.imageWidth && point.y <= state.imageHeight;
}

function circleFallbackColor(circle) {
  if (circle.circle_type === "center") return "#ffffff";
  if (circle.circle_type === "unclassified") return "#9aa5b1";
  if (state.showArms && circle.arm_id && ARM_COLORS[circle.arm_id]) {
    return ARM_COLORS[circle.arm_id];
  }
  return ["side", "satellite"].includes(circle.circle_type) ? "#49d8f0" : "#eaf676";
}

function circleColor(circle) {
  return circle.stroke_color || circleFallbackColor(circle);
}

function circleFillColor(circle) {
  return circle.fill_color || circleColor(circle);
}

function canonicalCircleType(circle) {
  if (circle.circle_type === "side") return "satellite";
  return ["center", "main", "satellite", "unclassified"].includes(circle.circle_type)
    ? circle.circle_type
    : "unclassified";
}

function reportBoundaryIds() {
  return new Set(
    (state.structureReport?.boundary_candidates || []).map((item) => String(item.id)),
  );
}

function reportDuplicateIds() {
  return new Set(
    (state.structureReport?.duplicate_candidates || []).flatMap((item) => [
      String(item.id_a),
      String(item.id_b),
    ]),
  );
}

function circleVisible(circle) {
  const selectedTypes = [];
  if ($("#filterUnclassified")?.checked) selectedTypes.push("unclassified");
  if ($("#filterMain")?.checked) selectedTypes.push("main");
  if ($("#filterSatellite")?.checked) selectedTypes.push("satellite");
  if ($("#filterCenter")?.checked) selectedTypes.push("center");
  if (selectedTypes.length && !selectedTypes.includes(canonicalCircleType(circle))) return false;
  if (
    $("#filterSelectedArm")?.checked &&
    circle.arm_id !== Number($("#mainChainArm")?.value || 1)
  ) return false;
  if ($("#filterBoundary")?.checked && !reportBoundaryIds().has(circle.id)) return false;
  if (
    $("#filterMissingParent")?.checked &&
    !(canonicalCircleType(circle) === "satellite" && !circle.parent_id)
  ) return false;
  if ($("#filterDuplicates")?.checked && !reportDuplicateIds().has(circle.id)) return false;
  if (
    $("#filterLowConfidence")?.checked &&
    Number(circle.final_confidence ?? circle.confidence ?? 1) >=
      Number($("#minFinalConfidence")?.value || 0.5)
  ) return false;
  if ($("#filterManual")?.checked && circle.source === "automatic") return false;
  return true;
}

function approvedMainIds() {
  return new Set(
    Object.values(state.mainChains).flatMap((ids) => ids.map(String)),
  );
}

function drawChain(ids, color, inverseScale, { dashed = false } = {}) {
  const points = ids
    .map((id) => state.circles.find((circle) => circle.id === id))
    .filter(Boolean);
  if (points.length < 2) return;
  context.save();
  context.strokeStyle = color;
  context.globalAlpha = dashed ? 0.92 : 0.78;
  context.lineWidth = (dashed ? 2.8 : 2.2) * inverseScale;
  if (dashed) context.setLineDash([7 * inverseScale, 4 * inverseScale]);
  context.beginPath();
  points.forEach((point, index) => {
    if (index === 0) context.moveTo(point.x, point.y);
    else context.lineTo(point.x, point.y);
  });
  context.stroke();
  context.restore();
}

function drawStrokeMask(strokes, color, opacity) {
  if (!strokes?.length || !state.imageWidth || !state.imageHeight) return;
  const layer = document.createElement("canvas");
  layer.width = Math.max(1, Math.round(state.imageWidth));
  layer.height = Math.max(1, Math.round(state.imageHeight));
  const layerContext = layer.getContext("2d");
  layerContext.lineCap = "round";
  layerContext.lineJoin = "round";
  for (const stroke of strokes) {
    if (!stroke.points?.length) continue;
    const points = stroke.points.map(sourceToProcessedCoordinate);
    const sourceCenter = stroke.points[0];
    const lineWidth = sourceRadiusToProcessed(sourceCenter, Number(stroke.radius || 1)) * 2;
    layerContext.globalCompositeOperation = stroke.mode === "erase" ? "destination-out" : "source-over";
    layerContext.strokeStyle = color;
    layerContext.fillStyle = color;
    layerContext.lineWidth = Math.max(1, lineWidth);
    layerContext.beginPath();
    layerContext.moveTo(points[0].x, points[0].y);
    for (const point of points.slice(1)) layerContext.lineTo(point.x, point.y);
    if (points.length === 1) {
      layerContext.arc(points[0].x, points[0].y, Math.max(1, lineWidth / 2), 0, Math.PI * 2);
      layerContext.fill();
    } else {
      layerContext.stroke();
    }
  }
  layerContext.globalCompositeOperation = "source-over";
  context.save();
  context.globalAlpha = opacity;
  context.drawImage(layer, 0, 0);
  context.restore();
}

function drawPreprocessOverlays(inverseScale) {
  const profile = state.preprocessProfile;
  if (!profile || state.workspace !== "preprocess") return;
  if (state.showRoi && profile.analysis_polygon?.length >= 3) {
    const polygon = profile.analysis_polygon.map(sourceToProcessedCoordinate);
    context.save();
    context.fillStyle = "rgba(255, 73, 87, .28)";
    context.beginPath();
    context.rect(0, 0, state.imageWidth, state.imageHeight);
    context.moveTo(polygon[0].x, polygon[0].y);
    for (const point of polygon.slice(1)) context.lineTo(point.x, point.y);
    context.closePath();
    context.fill("evenodd");
    context.strokeStyle = "#c8f06b";
    context.lineWidth = 2 * inverseScale;
    context.setLineDash([7 * inverseScale, 5 * inverseScale]);
    context.beginPath();
    context.moveTo(polygon[0].x, polygon[0].y);
    for (const point of polygon.slice(1)) context.lineTo(point.x, point.y);
    context.closePath();
    context.stroke();
    context.restore();
  } else if (state.showRoi && profile.analysis_polygon?.length) {
    const polygon = profile.analysis_polygon.map(sourceToProcessedCoordinate);
    context.save();
    context.strokeStyle = "#c8f06b";
    context.lineWidth = 2 * inverseScale;
    context.beginPath();
    context.moveTo(polygon[0].x, polygon[0].y);
    for (const point of polygon.slice(1)) context.lineTo(point.x, point.y);
    context.stroke();
    context.restore();
  }
  if (state.showExclusionMask) {
    const pendingMaskStroke =
      state.pointer?.kind === "brush" && ["maskPaint", "maskErase"].includes(state.pointer.tool)
        ? [state.pointer.stroke]
        : [];
    drawStrokeMask(
      [...(profile.exclusion_strokes || []), ...pendingMaskStroke],
      "#ff4055",
      state.maskOpacity,
    );
  }
  for (const [sampleClass, color] of Object.entries(SAMPLE_COLORS)) {
    const pending =
      state.pointer?.kind === "brush" &&
      state.pointer.tool === "samplePaint" &&
      state.pointer.sampleClass === sampleClass
        ? [state.pointer.stroke]
        : [];
    drawStrokeMask(
      [...(profile.color_samples?.[sampleClass] || []), ...pending],
      color,
      0.55,
    );
  }
  if (
    $("#showTramlines")?.checked
    && (state.tramlineSuggestion?.lines || state.pointer?.kind === "tramlineAdd")
  ) {
    context.save();
    const maskWidth = Math.max(1, Number($("#tramlineMaskWidth")?.value || 7));
    context.lineCap = "round";
    for (const line of state.tramlineSuggestion?.lines || []) {
      const selected = line.id === state.selectedTramlineId;
      if (selected) {
        context.strokeStyle = "rgba(255,255,255,.92)";
        context.lineWidth = maskWidth + 4 * inverseScale;
        context.globalAlpha = 0.95;
        context.beginPath();
        context.moveTo(line.p1.x, line.p1.y);
        context.lineTo(line.p2.x, line.p2.y);
        context.stroke();
      }
      context.strokeStyle = line.manual ? "#ffd06f" : "#ffad4e";
      context.lineWidth = maskWidth;
      context.globalAlpha = selected
        ? 0.92
        : 0.48 + Math.max(0, Math.min(1, Number(line.support ?? 0.5))) * 0.34;
      context.beginPath();
      context.moveTo(line.p1.x, line.p1.y);
      context.lineTo(line.p2.x, line.p2.y);
      context.stroke();
      if (selected) {
        for (const endpoint of [line.p1, line.p2]) {
          context.globalAlpha = 1;
          context.fillStyle = "#ffffff";
          context.strokeStyle = "#11160b";
          context.lineWidth = 1.5 * inverseScale;
          context.beginPath();
          context.arc(endpoint.x, endpoint.y, 6 * inverseScale, 0, Math.PI * 2);
          context.fill();
          context.stroke();
        }
      }
    }
    if (state.pointer?.kind === "tramlineAdd") {
      context.globalAlpha = 0.95;
      context.strokeStyle = "#ffffff";
      context.lineWidth = Math.max(2 * inverseScale, maskWidth);
      context.setLineDash([8 * inverseScale, 5 * inverseScale]);
      context.beginPath();
      context.moveTo(state.pointer.startImage.x, state.pointer.startImage.y);
      context.lineTo(state.pointer.currentImage.x, state.pointer.currentImage.y);
      context.stroke();
    }
    context.restore();
  }
  for (const [sizeClass, settings] of Object.entries(profile.size_classes || {})) {
    const color = { small: "#58d7ff", medium: "#f1dd59", large: "#ff7ca8" }[sizeClass] || "#fff";
    for (const sample of settings.samples || []) {
      const sourceCenter = { x: sample.source_x, y: sample.source_y };
      const center = sourceToProcessedCoordinate(sourceCenter);
      const displayRadius = sourceRadiusToProcessed(sourceCenter, Number(sample.radius));
      context.save();
      context.strokeStyle = color;
      context.lineWidth = 2 * inverseScale;
      context.setLineDash([5 * inverseScale, 3 * inverseScale]);
      context.beginPath();
      context.arc(center.x, center.y, displayRadius, 0, Math.PI * 2);
      context.stroke();
      context.restore();
    }
  }
}

function render() {
  const { width, height, dpr } = state.viewport;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!state.imageElement) {
    $("#zoomReadout").textContent = "—";
    return;
  }

  context.save();
  context.translate(state.view.tx, state.view.ty);
  context.scale(state.view.scale, state.view.scale);
  context.drawImage(state.imageElement, 0, 0);

  const inverseScale = 1 / state.view.scale;
  drawPreprocessOverlays(inverseScale);
  if (state.workspace === "measure" && state.center && state.showArms) {
    context.save();
    context.globalAlpha = 0.58;
    context.setLineDash([7 * inverseScale, 7 * inverseScale]);
    context.lineWidth = 1.5 * inverseScale;
    const length = Math.hypot(state.imageWidth, state.imageHeight) * 1.4;
    for (let arm = 1; arm <= 6; arm += 1) {
      context.strokeStyle = ARM_COLORS[arm];
      context.beginPath();
      const steps = state.armModel === "spiral" ? 320 : 1;
      for (let step = 0; step <= steps; step += 1) {
        const distance = length * step / steps;
        const twist =
          state.armModel === "spiral"
            ? state.armTwistDegreesPer100px * distance / 100
            : 0;
        const angle = (
          (state.phaseDegrees + (arm - 1) * 60 + twist)
          * Math.PI
          / 180
        );
        const x = state.center.x + distance * Math.cos(angle);
        const y = state.center.y - distance * Math.sin(angle);
        if (step === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
    }
    context.restore();
  }

  if (state.workspace === "measure") {
    for (let arm = 1; arm <= 6; arm += 1) {
      drawChain(state.mainChains[arm] || [], ARM_COLORS[arm], inverseScale);
    }
    if (state.mainChainDraft) {
      drawChain(state.mainChainDraft.ids, "#ffffff", inverseScale, { dashed: true });
    }
  }

  const visibleCircles = state.circles.filter(circleVisible);
  const visibleIds = new Set(visibleCircles.map((circle) => circle.id));
  if (state.workspace === "measure") {
    context.save();
    context.lineWidth = 1.1 * inverseScale;
    context.globalAlpha = 0.34;
    for (const circle of visibleCircles) {
      if (canonicalCircleType(circle) !== "satellite" || !circle.parent_id) continue;
      const parent = state.circles.find((item) => item.id === circle.parent_id);
      if (!parent || !visibleIds.has(parent.id)) continue;
      context.strokeStyle = circleColor(parent);
      context.beginPath();
      context.moveTo(parent.x, parent.y);
      context.lineTo(circle.x, circle.y);
      context.stroke();
    }
    context.restore();
  }
  const satelliteCounts = new Map();
  const approvedIds = approvedMainIds();
  for (const circle of state.circles) {
    if (canonicalCircleType(circle) !== "satellite" || !circle.parent_id) continue;
    satelliteCounts.set(circle.parent_id, (satelliteCounts.get(circle.parent_id) || 0) + 1);
  }

  for (const circle of visibleCircles) {
    const selected = circle.id === state.selectedId;
    const color = circleColor(circle);
    const circleType = canonicalCircleType(circle);
    const baseAlpha = circle.source === "automatic" ? Math.max(0.48, circle.confidence) : 1;
    context.save();
    context.beginPath();
    context.arc(circle.x, circle.y, Math.max(0.1, circle.radius), 0, Math.PI * 2);
    if (circle.fill_enabled) {
      context.globalAlpha =
        baseAlpha * Math.max(0, Math.min(1, Number(circle.fill_opacity ?? 0.2)));
      context.fillStyle = circleFillColor(circle);
      context.fill();
    }
    context.globalAlpha = baseAlpha;
    context.strokeStyle = color;
    const strokeWidth = Number(circle.stroke_width);
    context.lineWidth = (
      Number.isFinite(strokeWidth) && strokeWidth > 0
        ? Math.max(0.5, Math.min(20, strokeWidth))
        : circleType === "center" ? 3.2 : circleType === "main" ? 2.4 : 1.7
    ) * inverseScale;
    if (circleType === "satellite") {
      context.setLineDash([5 * inverseScale, 3 * inverseScale]);
    } else if (circleType === "unclassified") {
      context.setLineDash([2 * inverseScale, 4 * inverseScale]);
    }
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = color;
    context.beginPath();
    context.arc(circle.x, circle.y, 2.2 * inverseScale, 0, Math.PI * 2);
    context.fill();
    if (selected) {
      context.strokeStyle = "#ffffff";
      context.lineWidth = 1.1 * inverseScale;
      context.setLineDash([3 * inverseScale, 3 * inverseScale]);
      context.beginPath();
      context.arc(circle.x, circle.y, circle.radius + 5 * inverseScale, 0, Math.PI * 2);
      context.stroke();
      context.setLineDash([]);
      context.fillStyle = "#ffffff";
      context.beginPath();
      context.arc(circle.x + circle.radius, circle.y, 4.2 * inverseScale, 0, Math.PI * 2);
      context.fill();
    }
    const showLabel =
      state.labelMode === "all" ||
      (state.labelMode === "selected" && selected) ||
      (state.showLabels && state.labelMode !== "none");
    if (showLabel) {
      const label = `${circle.id}${circle.arm_id ? ` · K${circle.arm_id}` : ""}`;
      context.font = `${11 * inverseScale}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      context.lineWidth = 3 * inverseScale;
      context.strokeStyle = "rgba(5,8,10,.85)";
      context.strokeText(label, circle.x + 5 * inverseScale, circle.y - 5 * inverseScale);
      context.fillStyle = color;
      context.fillText(label, circle.x + 5 * inverseScale, circle.y - 5 * inverseScale);
    }
    const satelliteCount = satelliteCounts.get(circle.id) || 0;
    if (satelliteCount > 0 && approvedIds.has(circle.id)) {
      const badgeX = circle.x - circle.radius * 0.72;
      const badgeY = circle.y - circle.radius * 0.72;
      const badgeRadius = 7 * inverseScale;
      context.fillStyle = "#c8f06b";
      context.beginPath();
      context.arc(badgeX, badgeY, badgeRadius, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#11160b";
      context.font = `bold ${8 * inverseScale}px ui-sans-serif, system-ui`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(String(satelliteCount), badgeX, badgeY + 0.3 * inverseScale);
    }
    context.restore();
  }

  drawPerspectiveQualityOverlays(inverseScale);

  if (state.previewCircle) {
    context.save();
    context.strokeStyle = "#ffffff";
    context.lineWidth = 1.5 * inverseScale;
    context.setLineDash([5 * inverseScale, 3 * inverseScale]);
    context.beginPath();
    context.arc(
      state.previewCircle.x,
      state.previewCircle.y,
      state.previewCircle.radius,
      0,
      Math.PI * 2,
    );
    context.stroke();
    context.restore();
  }

  if (state.center) {
    const marker = 12 * inverseScale;
    context.save();
    context.strokeStyle = "#ffffff";
    context.lineWidth = 2.4 * inverseScale;
    context.beginPath();
    context.moveTo(state.center.x - marker, state.center.y);
    context.lineTo(state.center.x + marker, state.center.y);
    context.moveTo(state.center.x, state.center.y - marker);
    context.lineTo(state.center.x, state.center.y + marker);
    context.stroke();
    context.strokeStyle = "#111111";
    context.lineWidth = 0.8 * inverseScale;
    context.beginPath();
    context.arc(state.center.x, state.center.y, 4 * inverseScale, 0, Math.PI * 2);
    context.stroke();
    context.restore();
  }

  if (state.perspectivePoints.length) {
    const perspectiveDisplayPoints = state.perspectivePoints.map(sourceToProcessedCoordinate);
    context.save();
    context.lineWidth = 2 * inverseScale;
    context.strokeStyle = "#ffffff";
    context.fillStyle = "#c8f06b";
    context.beginPath();
    perspectiveDisplayPoints.forEach((point, index) => {
      if (index === 0) context.moveTo(point.x, point.y);
      else context.lineTo(point.x, point.y);
    });
    if (state.perspectivePoints.length === 4 && $("#perspectiveMode").value === "reference_quad") {
      context.closePath();
    }
    context.stroke();
    perspectiveDisplayPoints.forEach((point, index) => {
      context.beginPath();
      context.arc(point.x, point.y, 6 * inverseScale, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#11160b";
      context.font = `bold ${9 * inverseScale}px ui-sans-serif, system-ui`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(String(index + 1), point.x, point.y);
      context.fillStyle = "#c8f06b";
    });
    context.restore();
  }
  context.restore();
  $("#zoomReadout").textContent = `${Math.round(state.view.scale * 100)}%`;
}

function hitTest(point) {
  const tolerance = 9 / state.view.scale;
  const measured = state.circles.filter(circleVisible).map((circle) => {
    const distance = Math.hypot(point.x - circle.x, point.y - circle.y);
    return { circle, distance, boundaryDistance: Math.abs(distance - circle.radius) };
  });
  const boundaryHit = measured
    .filter((item) => item.boundaryDistance <= tolerance)
    .sort((left, right) => left.boundaryDistance - right.boundaryDistance)[0];
  if (boundaryHit) return { circle: boundaryHit.circle, mode: "resize" };
  const centerHit = measured
    .filter((item) => item.distance <= Math.max(item.circle.radius, tolerance * 1.4))
    .sort((left, right) => left.distance - right.distance || left.circle.radius - right.circle.radius)[0];
  if (centerHit) {
    return { circle: centerHit.circle, mode: "move" };
  }
  return null;
}

function setTool(tool) {
  state.tool = tool;
  if (tool.startsWith("tramline") && $("#showTramlines")) {
    $("#showTramlines").checked = true;
  }
  $$(".tool[data-tool]").forEach((button) => {
    button.classList.toggle("active", button.dataset.tool === tool);
  });
  $("#toolHint").textContent = TOOL_HINTS[tool] || "";
  canvas.style.cursor =
    tool === "pan" ? "grab" :
    ["add", "roiPolygon", "maskPaint", "maskErase", "samplePaint", "sizeSample", "mainChain", "tramlineAdd"].includes(tool) ? "crosshair" :
    ["delete", "tramlineDelete"].includes(tool) ? "not-allowed" :
    tool === "center" || tool === "perspective" ? "crosshair" :
    "default";
  render();
}

function suppressAutomatic(circle) {
  if (!circle || !String(circle.id || "").startsWith("auto-")) return;
  const origin = circle._automatic_origin || {
    id: circle.id,
    x: circle.x,
    y: circle.y,
    radius: circle.radius,
  };
  if (!state.suppressedAutomaticGeometry.some((item) => item.id === origin.id)) {
    state.suppressedAutomaticGeometry.push({ ...origin });
  }
}

function markManual(circle, field = null) {
  if (circle.source === "automatic") suppressAutomatic(circle);
  circle.source = circle.source === "automatic" ? "manual_override" : circle.source || "manual";
  if (field && ["circle_type", "arm_id", "sequence_index", "side", "parent_id"].includes(field)) {
    const locks = new Set(circle._locked_fields || []);
    locks.add(field);
    circle._locked_fields = [...locks];
    circle.locked_fields = [...locks];
    circle.classification_source = "user_confirmed";
    circle.classification_confidence = 1;
  }
  updateCircleSourceCoordinates(circle);
  circle.processed_x = circle.x;
  circle.processed_y = circle.y;
  if (field === "confidence") {
    circle.final_confidence = circle.confidence;
  }
  updatePolar(circle);
}

function structuralValuesEqual(left, right) {
  return JSON.stringify(left ?? null) === JSON.stringify(right ?? null);
}

function recordStructuralEdit(
  circleId,
  field,
  previousValue,
  newValue,
  actor,
  operationType,
  batchId = null,
) {
  if (structuralValuesEqual(previousValue, newValue)) return;
  state.editHistory.push({
    id: `edit-${Date.now().toString(36)}-${state.editHistory.length + 1}`,
    circle_id: circleId,
    field,
    previous_value: deepClone(previousValue ?? null),
    new_value: deepClone(newValue ?? null),
    actor,
    timestamp: new Date().toISOString(),
    operation_type: operationType,
    batch_id: batchId,
  });
  if (state.editHistory.length > 10000) state.editHistory.shift();
  if (state.perspectiveQuality) state.perspectiveQualityStale = true;
}

function structureSnapshot(operationType) {
  return {
    operationType,
    circles: state.circles.map((circle) => deepClone(circle)),
    mainChains: deepClone(state.mainChains),
    selectedId: state.selectedId,
    deletedAutoIds: [...state.deletedAutoIds],
    suppressedAutomaticGeometry: deepClone(state.suppressedAutomaticGeometry),
    center: state.center ? { ...state.center } : null,
    centerSource: state.centerSource ? { ...state.centerSource } : null,
    phaseDegrees: state.phaseDegrees,
    armModel: state.armModel,
    armTwistDegreesPer100px: state.armTwistDegreesPer100px,
  };
}

function pushStructureUndo(operationType) {
  state.structureUndo.push(structureSnapshot(operationType));
  if (state.structureUndo.length > 50) state.structureUndo.shift();
  $("#undoStructureButton").disabled = false;
}

function circleFieldLocked(circle, field) {
  return Boolean(circle.locked || (circle._locked_fields || []).includes(field));
}

function circleLockSummary() {
  let recordLocks = 0;
  let fieldLocks = 0;
  let affectedCircles = 0;
  for (const circle of state.circles) {
    const fields = new Set([
      ...(circle._locked_fields || []),
      ...(circle.locked_fields || []),
    ]);
    const hasRecordLock = Boolean(circle.locked);
    if (hasRecordLock) recordLocks += 1;
    fieldLocks += fields.size;
    if (hasRecordLock || fields.size) affectedCircles += 1;
  }
  return {
    recordLocks,
    fieldLocks,
    affectedCircles,
    totalLocks: recordLocks + fieldLocks,
  };
}

function updateStructureLockUi() {
  const summary = circleLockSummary();
  const button = $("#clearAllCircleLocks");
  const readout = $("#structureLockSummary");
  if (!button || !readout) return;
  button.disabled = summary.totalLocks === 0;
  button.textContent = summary.totalLocks
    ? `Tüm kilitleri aç (${summary.affectedCircles})`
    : "Tüm kilitler açık";
  readout.textContent =
    `${summary.affectedCircles} kilitli çember · ` +
    `${summary.fieldLocks} alan kilidi · ${summary.recordLocks} genel kilit`;
}

function lockCircleFields(circle, fields, operationType, batchId) {
  const previous = [...(circle._locked_fields || [])];
  const next = [...new Set([...previous, ...fields])];
  circle._locked_fields = next;
  circle.locked_fields = [...next];
  recordStructuralEdit(
    circle.id,
    "locked_fields",
    previous,
    next,
    "user",
    operationType,
    batchId,
  );
}

function applyCircleChanges(
  circle,
  changes,
  {
    actor = "user",
    operationType = "structural_update",
    batchId = null,
    respectLocks = false,
    lockFields = [],
  } = {},
) {
  let changed = false;
  for (const [field, value] of Object.entries(changes)) {
    if (respectLocks && circleFieldLocked(circle, field)) continue;
    const previous = circle[field] ?? null;
    if (structuralValuesEqual(previous, value)) continue;
    circle[field] = deepClone(value);
    recordStructuralEdit(
      circle.id,
      field,
      previous,
      value,
      actor,
      operationType,
      batchId,
    );
    changed = true;
  }
  if (changed && actor === "user") markManual(circle);
  if (lockFields.length) lockCircleFields(circle, lockFields, operationType, batchId);
  circle.locked_fields = [...(circle._locked_fields || [])];
  return changed;
}

function undoLastStructureChange() {
  const snapshot = state.structureUndo.pop();
  if (!snapshot) {
    setStatus("Geri alınacak yapısal değişiklik yok.", "error");
    return;
  }
  const currentById = new Map(state.circles.map((circle) => [circle.id, circle]));
  const previousById = new Map(snapshot.circles.map((circle) => [circle.id, circle]));
  const batchId = `undo-${Date.now().toString(36)}`;
  const fields = [
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
    "locked",
    "locked_fields",
  ];
  for (const circleId of new Set([...currentById.keys(), ...previousById.keys()])) {
    const current = currentById.get(circleId);
    const previous = previousById.get(circleId);
    if (!current || !previous) {
      recordStructuralEdit(
        circleId,
        "circle_record",
        current || null,
        previous || null,
        "user",
        `undo:${snapshot.operationType}`,
        batchId,
      );
      continue;
    }
    for (const field of fields) {
      recordStructuralEdit(
        circleId,
        field,
        current[field] ?? null,
        previous[field] ?? null,
        "user",
        `undo:${snapshot.operationType}`,
        batchId,
      );
    }
  }
  recordStructuralEdit(
    null,
    "general_center",
    state.center,
    snapshot.center,
    "user",
    `undo:${snapshot.operationType}`,
    batchId,
  );
  recordStructuralEdit(
    null,
    "arm_model_settings",
    {
      arm_model: state.armModel,
      phase_degrees: state.phaseDegrees,
      arm_twist_degrees_per_100px: state.armTwistDegreesPer100px,
    },
    {
      arm_model: snapshot.armModel,
      phase_degrees: snapshot.phaseDegrees,
      arm_twist_degrees_per_100px: snapshot.armTwistDegreesPer100px,
    },
    "user",
    `undo:${snapshot.operationType}`,
    batchId,
  );
  state.circles = snapshot.circles.map(normalizeCircle);
  state.mainChains = deepClone(snapshot.mainChains);
  state.selectedId = snapshot.selectedId;
  state.deletedAutoIds = new Set(snapshot.deletedAutoIds);
  state.suppressedAutomaticGeometry = deepClone(snapshot.suppressedAutomaticGeometry);
  state.center = snapshot.center ? { ...snapshot.center } : null;
  state.centerSource = snapshot.centerSource ? { ...snapshot.centerSource } : null;
  state.phaseDegrees = snapshot.phaseDegrees;
  state.armModel = snapshot.armModel;
  state.armTwistDegreesPer100px = snapshot.armTwistDegreesPer100px;
  $("#phaseDegrees").value = String(state.phaseDegrees);
  $("#armModel").value = state.armModel;
  $("#armTwist").value = String(state.armTwistDegreesPer100px);
  updateArmModelUi();
  state.mainChainDraft = null;
  state.structureReport = null;
  state.boundarySelection.clear();
  state.localSideSelection.clear();
  state.satelliteSelection.clear();
  $("#undoStructureButton").disabled = state.structureUndo.length === 0;
  setTool("select");
  updateUi();
  void refreshStructureAnalysis(false);
  setStatus(`${snapshot.operationType} işlemi tek adımda geri alındı.`, "success");
}

function updatePolar(circle) {
  if (!state.center) {
    circle.center_distance = null;
    circle.center_angle = null;
    return;
  }
  const dx = circle.x - state.center.x;
  const dy = circle.y - state.center.y;
  circle.center_distance = Math.hypot(dx, dy);
  circle.center_angle = ((Math.atan2(-dy, dx) * 180) / Math.PI + 360) % 360;
}

function updateAllPolar() {
  state.circles.forEach(updatePolar);
}

function selectCircle(circleId) {
  state.selectedId = circleId;
  updateInspector();
  renderSelectedCircleQuality();
  render();
}

function focusCircle(circle) {
  if (!circle) return;
  state.selectedId = circle.id;
  const contextDiameter = Math.max(100, circle.radius * 12);
  const targetScale = Math.max(
    state.view.scale,
    Math.min(
      12,
      Math.min(state.viewport.width, state.viewport.height) / contextDiameter,
    ),
  );
  state.view.scale = targetScale;
  state.view.tx = state.viewport.width / 2 - circle.x * targetScale;
  state.view.ty = state.viewport.height / 2 - circle.y * targetScale;
  updateUi();
}

function removeCircle(circleId, { trackUndo = true, operationType = "delete_circle" } = {}) {
  const circle = state.circles.find((item) => item.id === circleId);
  if (!circle) return;
  const lockedChild = state.circles.find(
    (item) => item.parent_id === circleId && circleFieldLocked(item, "parent_id"),
  );
  if (lockedChild) {
    setStatus(`${lockedChild.id} parent_id alanı kilitli; silme işlemi durduruldu.`, "error");
    return;
  }
  if (trackUndo) pushStructureUndo(operationType);
  const batchId = `${operationType}-${Date.now().toString(36)}`;
  recordStructuralEdit(
    circleId,
    "circle_record",
    circle,
    null,
    "user",
    operationType,
    batchId,
  );
  if (circle.id.startsWith("auto-")) {
    state.deletedAutoIds.add(circle.id);
    suppressAutomatic(circle);
  }
  state.circles = state.circles.filter((item) => item.id !== circleId);
  for (const item of state.circles) {
    if (item.parent_id === circleId) {
      applyCircleChanges(
        item,
        {
          parent_id: null,
          local_angle: null,
          normalized_parent_distance: null,
        },
        {
          operationType,
          batchId,
          lockFields: ["parent_id"],
        },
      );
    }
  }
  for (let arm = 1; arm <= 6; arm += 1) {
    state.mainChains[arm] = (state.mainChains[arm] || []).filter(
      (id) => id !== circleId,
    );
  }
  if (state.selectedId === circleId) state.selectedId = null;
  clearSymmetry();
  updateUi();
  void refreshStructureAnalysis(false);
  setStatus(`${circleId} silindi; manuel silme kaydedildi.`, "success");
}

function addManualCircle(x, y, radius) {
  pushStructureUndo("add_circle");
  state.manualCounter += 1;
  const id = `manual-${Date.now().toString(36)}-${state.manualCounter}`;
  const circle = normalizeCircle({
    id,
    x,
    y,
    radius: Math.max(1, radius),
    confidence: 1,
    circle_type: "unclassified",
    processed_x: x,
    processed_y: y,
    source: "manual",
  });
  updateCircleSourceCoordinates(circle);
  updatePolar(circle);
  state.circles.push(circle);
  recordStructuralEdit(
    id,
    "circle_record",
    null,
    circle,
    "user",
    "add_circle",
  );
  state.selectedId = id;
  clearSymmetry();
  updateUi();
  void refreshStructureAnalysis(false);
  setTool("select");
  setStatus("Manuel daire eklendi.", "success");
}

function updatePerspectiveUi() {
  const count = state.perspectivePoints.length;
  $("#perspectiveCount").textContent = String(count);
  $("#applyPerspective").disabled = !state.imageId || count !== 4;
  const mode = $("#perspectiveMode").value;
  $("#perspectiveMethodGuidance").textContent = mode === "field_tracks"
    ? "Paralel tarla izleri güçlü referans sağlar. Ancak tek doğrultudaki paralellik, tek başına tam metrik düzeltme garantisi vermez. Sonucu dairesellik ve simetri skorlarıyla doğrulayın."
    : "Görüntü köşelerini değil, zemindeki aynı düzleme ait dört anlamlı referans noktayı seçin. Seçilen dört nokta, yerdeki bir dikdörtgen/paralelkenarın köşelerini temsil etmelidir.";
}

function perspectiveQualityWeights() {
  return {
    tramline: Math.max(0, Number($("#qualityWeightTramline").value || 0)),
    roundness: Math.max(0, Number($("#qualityWeightRoundness").value || 0)),
    arm_symmetry: Math.max(0, Number($("#qualityWeightArm").value || 0)),
    k5k6: Math.max(0, Number($("#qualityWeightK5K6").value || 0)),
    warp_penalty: Math.max(0, Number($("#qualityWeightWarp").value || 0)),
  };
}

function currentPerspectiveQualityMethod() {
  const perspective = operationByType("perspective");
  if (perspective?.enabled) {
    return perspective.params.mode === "field_tracks"
      ? "tramline_guided"
      : "four_point";
  }
  if (state.tramlineSuggestion?.lines?.length) return "tramline_guided";
  if (state.perspectivePoints.length === 4) {
    return $("#perspectiveMode").value === "field_tracks"
      ? "tramline_guided"
      : "four_point";
  }
  return "none";
}

function currentPerspectiveQualityStatus() {
  const perspective = operationByType("perspective");
  if (perspective?.enabled && (perspective.params.points || []).length === 4) {
    return "approved";
  }
  if (
    state.perspectivePoints.length === 4
    || state.tramlineSuggestion?.perspective_points_source?.length === 4
  ) {
    return "proposed";
  }
  return "not_applied";
}

function perspectiveQualityRequestPayload() {
  const tramlines = (state.tramlineSuggestion?.lines || []).map((line) => ({
    id: line.id,
    source_p1: line.source_p1,
    source_p2: line.source_p2,
    manual: Boolean(line.manual || state.tramlineSuggestion?.edited),
    support: Number(line.support ?? 1),
  })).filter((line) => line.source_p1 && line.source_p2);
  const proposedPoints = state.perspectivePoints.length === 4
    ? state.perspectivePoints
    : state.tramlineSuggestion?.perspective_points_source || [];
  return {
    image_id: state.imageId,
    circles: state.circles,
    tramlines,
    center: state.center,
    center_source: state.centerSource,
    structure_state: {
      main_chains: state.mainChains,
    },
    selected_circle_ids: state.selectedId ? [state.selectedId] : [],
    method: currentPerspectiveQualityMethod(),
    status: currentPerspectiveQualityStatus(),
    perspective_points: proposedPoints.map(strokePoint),
    preserve_full_image: $("#preservePerspectiveFrame").checked,
    weights: perspectiveQualityWeights(),
  };
}

function qualityValue(value, digits = 1, suffix = "") {
  const number = Number(value);
  return value === null || value === undefined || !Number.isFinite(number)
    ? "—"
    : `${number.toFixed(digits)}${suffix}`;
}

function qualityMethodLabel(method) {
  return {
    none: "none",
    four_point: "four_point",
    tramline_guided: "tramline_guided",
  }[method] || method || "none";
}

function qualityStatusLabel(status) {
  return {
    not_applied: "uygulanmadı",
    proposed: "önerildi",
    approved: "onaylandı",
  }[status] || status || "uygulanmadı";
}

function qualityConfidenceLabel(level) {
  return { low: "düşük", medium: "orta", high: "yüksek" }[level] || "düşük";
}

function renderPerspectiveQualityMetrics(report) {
  const metrics = report?.metrics || {};
  const scope = $("#qualityCircleScope")?.value || "all_main";
  const roundnessGroup = report?.roundness?.groups?.[scope];
  const pair = (before, after, unit = "") =>
    `${qualityValue(metrics[before], 2, unit)} → ${qualityValue(metrics[after], 2, unit)}`;
  const directPair = (before, after, unit = "") =>
    `${qualityValue(before, 2, unit)} → ${qualityValue(after, 2, unit)}`;
  const rows = [
    ["Tramline açı std.", pair("tramline_angle_std_before", "tramline_angle_std_after", "°")],
    ["Tramline aralık CV", pair("tramline_spacing_cv_before", "tramline_spacing_cv_after")],
    ["Dairesellik kümesi", `${roundnessGroup?.paired_edge_fit_count ?? 0} / ${roundnessGroup?.circle_count ?? 0} çift uyum`],
    ["Elips oranı ort.", directPair(
      roundnessGroup?.before?.ellipse_ratio?.mean,
      roundnessGroup?.after?.ellipse_ratio?.mean,
    )],
    ["Dairesellik artığı", directPair(
      roundnessGroup?.before?.circularity_residual?.mean,
      roundnessGroup?.after?.circularity_residual?.mean,
    )],
    ["Kol yarıçap tutarlılığı", pair("arm_radius_consistency_before", "arm_radius_consistency_after")],
    ["Kol uzaklık tutarlılığı", pair("arm_distance_consistency_before", "arm_distance_consistency_after")],
    ["Kol açı tutarlılığı", pair("arm_angular_consistency_before", "arm_angular_consistency_after")],
    ["K5–K6 yarıçap MAE", pair("k5k6_radius_mae_before", "k5k6_radius_mae_after", " px")],
    ["K5–K6 uzaklık MAE", pair("k5k6_distance_mae_before", "k5k6_distance_mae_after", " px")],
    ["Warp alan oranı", qualityValue(metrics.warp_area_ratio, 3)],
    ["Warp shear cezası", qualityValue(metrics.warp_shear_score, 1)],
    ["Warp anisotropy cezası", qualityValue(metrics.warp_anisotropy_score, 1)],
    ["Yerel deformasyon", qualityValue(metrics.local_distortion_indicator, 1)],
    ["Warp penalty", qualityValue(metrics.warp_penalty, 1)],
  ];
  $("#perspectiveQualityMetrics").innerHTML = rows
    .map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

function drawK5K6QualityChart(report) {
  const chart = $("#k5k6QualityChart");
  if (!chart) return;
  const chartContext = chart.getContext("2d");
  const width = chart.width;
  const height = chart.height;
  chartContext.clearRect(0, 0, width, height);
  chartContext.fillStyle = "#090d12";
  chartContext.fillRect(0, 0, width, height);
  const rows = report?.k5k6?.rows || [];
  if (!rows.length) {
    chartContext.fillStyle = "#83909f";
    chartContext.font = "11px system-ui";
    chartContext.fillText("K5–K6 eşleşmiş sırası yok", 12, 24);
    return;
  }
  const before = rows.map((row) => Number(row.source_normalized_diameter_difference));
  const after = rows.map((row) => Number(row.corrected_normalized_diameter_difference));
  const maximum = Math.max(0.05, ...before, ...after);
  const margin = { left: 28, right: 10, top: 20, bottom: 23 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  chartContext.strokeStyle = "#29323d";
  chartContext.lineWidth = 1;
  chartContext.beginPath();
  chartContext.moveTo(margin.left, margin.top);
  chartContext.lineTo(margin.left, margin.top + plotHeight);
  chartContext.lineTo(margin.left + plotWidth, margin.top + plotHeight);
  chartContext.stroke();
  const drawSeries = (values, color) => {
    chartContext.strokeStyle = color;
    chartContext.lineWidth = 2;
    chartContext.beginPath();
    values.forEach((value, index) => {
      const x = margin.left + (rows.length === 1 ? plotWidth / 2 : index * plotWidth / (rows.length - 1));
      const y = margin.top + plotHeight - value / maximum * plotHeight;
      if (index === 0) chartContext.moveTo(x, y);
      else chartContext.lineTo(x, y);
    });
    chartContext.stroke();
  };
  drawSeries(before, "#ffad4e");
  drawSeries(after, "#72da7f");
  chartContext.font = "9px system-ui";
  chartContext.fillStyle = "#ffad4e";
  chartContext.fillText("önce", margin.left, 12);
  chartContext.fillStyle = "#72da7f";
  chartContext.fillText("sonra", margin.left + 42, 12);
  chartContext.fillStyle = "#83909f";
  rows.forEach((row, index) => {
    if (index % 2 !== 0 && rows.length > 8) return;
    const x = margin.left + (rows.length === 1 ? plotWidth / 2 : index * plotWidth / (rows.length - 1));
    chartContext.fillText(String(row.sequence_index), x - 2, height - 7);
  });
}

function renderSelectedCircleQuality() {
  const box = $("#selectedCircleQuality");
  if (!box) return;
  if (!state.selectedId) {
    box.textContent = "Bir ana çembere tıklayın; source/corrected elips oranı ve yerel artık burada gösterilir.";
    return;
  }
  const row = state.perspectiveQuality?.circle_metrics?.find(
    (item) => item.circle_id === state.selectedId,
  );
  if (!row) {
    box.innerHTML = `<strong>${escapeHtml(state.selectedId)}</strong><br>Bu daire için kalite ölçümü yok; kaliteyi yeniden hesaplayın.`;
    return;
  }
  box.innerHTML = `
    <strong>${escapeHtml(row.circle_id)}</strong> · ${escapeHtml(row.circle_type || "—")}<br>
    <span>source</span>: ${qualityValue(row.source_x, 1)}, ${qualityValue(row.source_y, 1)} ·
    oran ${qualityValue(row.ellipse_ratio_before, 3)} · artık ${qualityValue(row.circularity_residual_before, 3)}<br>
    <span>corrected</span>: ${qualityValue(row.corrected_x, 1)}, ${qualityValue(row.corrected_y, 1)} ·
    oran ${qualityValue(row.ellipse_ratio_after, 3)} · artık ${qualityValue(row.circularity_residual_after, 3)}<br>
    dairesellik farkı ${qualityValue(row.circularity_gain, 1)}
    <div class="button-row">
      <button class="button ghost" data-selected-quality-view="source">source view</button>
      <button class="button ghost" data-selected-quality-view="corrected">corrected view</button>
    </div>
  `;
}

function renderPerspectiveQuality() {
  const report = state.perspectiveQuality;
  const artifacts = state.perspectiveQualityArtifacts;
  if (!report) {
    $("#perspectiveQualityScore").textContent = "—";
    $("#qualityMethod").textContent = qualityMethodLabel(currentPerspectiveQualityMethod());
    $("#qualityStatus").textContent = qualityStatusLabel(currentPerspectiveQualityStatus());
    $("#qualityGain").textContent = "—";
    $("#qualityConfidence").textContent = "düşük";
    $("#qualityInterpretation").textContent = "Kalite hesabı henüz çalıştırılmadı.";
    $("#qualityInterpretation").classList.remove("metric-warning");
    $("#perspectiveQualityComponents").innerHTML =
      '<tr><td colspan="5" class="muted">Kalite hesabı bekleniyor.</td></tr>';
    $("#exportPerspectiveQuality").disabled = true;
    ["Before", "After", "SideBySide", "Roundness", "Tramlines", "Heatmap"].forEach((name) => {
      $(`#qualityView${name}`).disabled = true;
    });
    renderSelectedCircleQuality();
    drawK5K6QualityChart(null);
    return;
  }
  $("#perspectiveQualityScore").textContent = qualityValue(report.scores?.overall_after, 1);
  $("#qualityMethod").textContent = qualityMethodLabel(report.method);
  $("#qualityStatus").textContent = qualityStatusLabel(report.status);
  $("#qualityGain").textContent = qualityValue(report.scores?.gain, 1);
  $("#qualityConfidence").textContent = qualityConfidenceLabel(report.confidence?.level);
  $("#qualityInterpretation").textContent =
    `${report.interpretation_message || ""} · Dairesellik kümesi: ${$("#qualityCircleScope").selectedOptions[0]?.textContent || "tüm ana çemberler"}.` +
    `${state.perspectiveQualityStale ? " · Geometri değişti; puanı yeniden hesaplayın." : ""}`;
  $("#qualityInterpretation").classList.toggle("metric-warning", state.perspectiveQualityStale);
  $("#perspectiveMethodGuidance").textContent = report.method_guidance || "";
  $("#perspectiveQualityComponents").innerHTML = (report.components || []).map((component) => {
    const delta = Number(component.delta);
    const deltaClass = !Number.isFinite(delta)
      ? ""
      : delta > 0.05
        ? "quality-delta-positive"
        : delta < -0.05
          ? "quality-delta-negative"
          : "";
    return `<tr>
      <td>${escapeHtml(component.label)}</td>
      <td>${qualityValue(component.before, 1)}</td>
      <td>${qualityValue(component.after, 1)}</td>
      <td class="${deltaClass}">${qualityValue(component.delta, 1)}</td>
      <td>${escapeHtml(component.comment || "—")}</td>
    </tr>`;
  }).join("");
  renderPerspectiveQualityMetrics(report);
  drawK5K6QualityChart(report);
  renderSelectedCircleQuality();
  $("#exportPerspectiveQuality").disabled = state.perspectiveQualityStale;
  $("#qualityViewBefore").disabled = !artifacts.source_view;
  $("#qualityViewAfter").disabled = !artifacts.corrected_view;
  $("#qualityViewSideBySide").disabled = !artifacts.perspective_before_after;
  $("#qualityViewRoundness").disabled = !artifacts.roundness_overlay;
  $("#qualityViewTramlines").disabled = !artifacts.tramline_overlay_corrected;
  $("#qualityViewHeatmap").disabled = !artifacts.arm_scale_heatmap;
}

function invalidatePerspectiveQuality() {
  if (!state.perspectiveQuality) return;
  state.perspectiveQualityStale = true;
  renderPerspectiveQuality();
}

async function calculatePerspectiveQuality({ showBusy = true } = {}) {
  if (!state.imageId) {
    setStatus("Önce bir görüntü açın.", "error");
    return null;
  }
  if (showBusy) setBusy(true, "Perspektif önce/sonra geometrisi ölçülüyor…");
  try {
    await ensurePreprocessCurrent(false);
    const payload = await api("/api/perspective/quality", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(perspectiveQualityRequestPayload()),
    });
    state.perspectiveQuality = payload.report;
    state.perspectiveQualityArtifacts = payload.artifacts || {};
    state.perspectiveQualityStale = false;
    renderPerspectiveQuality();
    render();
    const scoreValue = payload.report?.scores?.overall_after;
    const score = qualityValue(scoreValue, 1);
    setStatus(
      scoreValue === null || scoreValue === undefined
        ? "Aktif veya değerlendirilebilir perspektif düzeltmesi yok; kaynak referans raporu hazırlandı."
        : `Perspektif kalite puanı ${score}/100 olarak hesaplandı; anotasyonlar değiştirilmedi.`,
      scoreValue === null || scoreValue === undefined ? "warning" : "success",
    );
    return payload;
  } catch (error) {
    setStatus(error.message, "error");
    return null;
  } finally {
    if (showBusy) setBusy(false);
  }
}

async function exportPerspectiveQualityReports() {
  if (!state.imageId || !state.perspectiveQuality || state.perspectiveQualityStale) {
    setStatus("Önce güncel perspektif kalite hesabını çalıştırın.", "warning");
    return;
  }
  setBusy(true, "Perspektif kalite JSON, CSV ve PNG raporları hazırlanıyor…");
  try {
    const payload = await api("/api/perspective/quality/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(perspectiveQualityRequestPayload()),
    });
    state.perspectiveQuality = payload.report;
    state.perspectiveQualityStale = false;
    const order = [
      "perspective_quality_report",
      "perspective_circle_metrics",
      "perspective_arm_metrics",
      "tramline_metrics",
      "perspective_before_after",
      "roundness_overlay",
      "tramline_overlay_corrected",
      "arm_scale_heatmap",
      "source_view",
      "corrected_view",
    ];
    $("#perspectiveQualityDownloads").innerHTML = order
      .filter((key) => payload.files?.[key])
      .map((key) => {
        const file = payload.files[key];
        return `<a href="${file.url}" download="${escapeHtml(file.name)}">${escapeHtml(file.name)}</a>`;
      })
      .join("");
    $("#perspectiveQualityDownloads").classList.remove("hidden");
    renderPerspectiveQuality();
    setStatus("Perspektif kalite raporları outputs/ klasörüne kaydedildi.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
}

function showPerspectiveQualityArtifact(key, label) {
  const url = state.perspectiveQualityArtifacts[key];
  if (url) showStepPreview(url, label);
}

async function showSelectedCircleQualityView(space) {
  const row = state.perspectiveQuality?.circle_metrics?.find(
    (item) => item.circle_id === state.selectedId,
  );
  const key = space === "source" ? "source_view" : "corrected_view";
  const url = state.perspectiveQualityArtifacts[key];
  if (!row || !url) return;
  const image = new Image();
  const loaded = new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error("Seçili daire görünümü yüklenemedi"));
  });
  image.src = `${url}${url.includes("?") ? "&" : "?"}circle=${Date.now()}`;
  try {
    await loaded;
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }
  const centerX = Number(row[`${space}_x`]);
  const centerY = Number(row[`${space}_y`]);
  const major = Number(row[`${space}_major_axis`] || row[`${space}_radius`] * 2 || 20);
  const minor = Number(row[`${space}_minor_axis`] || row[`${space}_radius`] * 2 || 20);
  const requestedExtent = Math.max(48, major, minor) * 2.6;
  const cropSize = Math.min(
    requestedExtent,
    image.naturalWidth,
    image.naturalHeight,
  );
  const sourceX = Math.max(0, Math.min(image.naturalWidth - cropSize, centerX - cropSize / 2));
  const sourceY = Math.max(0, Math.min(image.naturalHeight - cropSize, centerY - cropSize / 2));
  const previewCanvas = document.createElement("canvas");
  previewCanvas.width = 520;
  previewCanvas.height = 520;
  const previewContext = previewCanvas.getContext("2d");
  previewContext.fillStyle = "#090d12";
  previewContext.fillRect(0, 0, 520, 520);
  previewContext.drawImage(
    image,
    sourceX,
    sourceY,
    cropSize,
    cropSize,
    0,
    0,
    520,
    520,
  );
  const scale = 520 / cropSize;
  previewContext.strokeStyle = "#c8f06b";
  previewContext.lineWidth = 3;
  previewContext.setLineDash([9, 5]);
  previewContext.beginPath();
  previewContext.ellipse(
    (centerX - sourceX) * scale,
    (centerY - sourceY) * scale,
    major * scale / 2,
    minor * scale / 2,
    Number(row[`${space}_ellipse_angle`] || 0) * Math.PI / 180,
    0,
    Math.PI * 2,
  );
  previewContext.stroke();
  state.stepPreviewOpen = true;
  $("#stepPreviewImage").src = previewCanvas.toDataURL("image/png");
  $("#stepPreviewLabel").textContent =
    `${row.circle_id} · ${space} · oran ${qualityValue(row[space === "source" ? "ellipse_ratio_before" : "ellipse_ratio_after"], 3)}`;
  $("#stepPreview").classList.remove("hidden");
}

function drawPerspectiveQualityOverlays(inverseScale) {
  const rows = state.perspectiveQuality?.circle_metrics || [];
  if (!rows.length || (!state.showQualityEllipses && !state.showQualityBenefits)) return;
  const benefited = new Set(
    [...rows]
      .filter((row) => Number.isFinite(Number(row.circularity_gain)))
      .sort((left, right) => Number(right.circularity_gain) - Number(left.circularity_gain))
      .slice(0, 18)
      .map((row) => row.circle_id),
  );
  context.save();
  for (const row of rows) {
    const x = Number(row.corrected_x);
    const y = Number(row.corrected_y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (state.showQualityEllipses) {
      const major = Number(row.corrected_major_axis);
      const minor = Number(row.corrected_minor_axis);
      if (major > 0 && minor > 0) {
        const gain = Number(row.circularity_gain);
        context.strokeStyle = Number.isFinite(gain) && gain > 2
          ? "#72da7f"
          : Number.isFinite(gain) && gain < -2
            ? "#ff6e72"
            : "#f2bf61";
        context.lineWidth = 1.8 * inverseScale;
        context.setLineDash([5 * inverseScale, 3 * inverseScale]);
        context.beginPath();
        context.ellipse(
          x,
          y,
          major / 2,
          minor / 2,
          Number(row.corrected_ellipse_angle || 0) * Math.PI / 180,
          0,
          Math.PI * 2,
        );
        context.stroke();
      }
    }
    if (state.showQualityBenefits && benefited.has(row.circle_id)) {
      context.setLineDash([]);
      context.fillStyle = "#72da7f";
      context.beginPath();
      context.moveTo(x, y - 9 * inverseScale);
      context.lineTo(x - 5 * inverseScale, y - 18 * inverseScale);
      context.lineTo(x + 5 * inverseScale, y - 18 * inverseScale);
      context.closePath();
      context.fill();
    }
  }
  context.restore();
}

function updatePreprocessStats() {
  const profile = state.preprocessProfile;
  if (!profile) {
    $("#maskStats").textContent = "ROI yok · 0 fırça darbesi";
    return;
  }
  const polygonCount = profile.analysis_polygon?.length || 0;
  const strokeCount = profile.exclusion_strokes?.length || 0;
  $("#maskStats").textContent = `${polygonCount >= 3 ? `${polygonCount} köşeli ROI` : "ROI yok"} · ${strokeCount} fırça darbesi`;
  const labels = {
    flattened_crop: ".sample-stats .sample-flat",
    standing_crop: ".sample-stats .sample-standing",
    tramline: ".sample-stats .sample-tramline",
    shadow: ".sample-stats .sample-shadow",
  };
  for (const [sampleClass, selector] of Object.entries(labels)) {
    const count = profile.color_samples?.[sampleClass]?.length || 0;
    const prefix = { flattened_crop: "F", standing_crop: "S", tramline: "T", shadow: "G" }[sampleClass];
    $(selector).textContent = `${prefix} ${count}`;
  }
}

function setWorkspace(workspace) {
  state.workspace = workspace;
  if (workspace === "measure") closeStepPreview();
  $$(".workspace-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.workspace === workspace);
  });
  $$(".preprocess-section").forEach((element) => element.classList.toggle("hidden", workspace !== "preprocess"));
  $$(".measure-section").forEach((element) => element.classList.toggle("hidden", workspace !== "measure"));
  $$(".preprocess-tool").forEach((element) => element.classList.toggle("hidden", workspace !== "preprocess"));
  $$(".measure-tool").forEach((element) => element.classList.toggle("hidden", workspace !== "measure"));
  $("#preprocessInspector").classList.toggle("hidden", workspace !== "preprocess");
  $("#circleInspector").classList.toggle("hidden", workspace !== "measure");
  $("#preprocessLegend").classList.toggle("hidden", workspace !== "preprocess");
  if (workspace === "measure") setTool("select");
  else setTool("pan");
  updateUi();
}

function updateInspector() {
  const circle = selectedCircle();
  const form = $("#circleForm");
  form.classList.toggle("disabled-form", !circle);
  $("#selectionState").textContent = circle ? circle.id : "Bir daire seçin";
  for (const input of $$("[data-field]")) {
    const field = input.dataset.field;
    input.disabled =
      !circle ||
      (
        ["circle_type", "arm_id", "sequence_index", "parent_id", "side"].includes(field) &&
        circleFieldLocked(circle, field)
      );
    if (input.type === "checkbox") {
      input.checked = Boolean(circle?.[field]);
    } else if (input.type === "color") {
      input.value = circle
        ? field === "fill_color" ? circleFillColor(circle) : circleColor(circle)
        : "#eaf676";
    } else {
      input.value = circle && circle[field] !== null && circle[field] !== undefined ? String(circle[field]) : "";
    }
  }
  for (const input of $$("[data-fill-control]")) {
    input.disabled = !circle || !circle.fill_enabled;
  }
  for (const input of $$("[data-lock-field]")) {
    input.disabled = !circle;
    input.checked = Boolean(
      circle && (circle._locked_fields || []).includes(input.dataset.lockField),
    );
  }
  $("#resetCircleAppearance").disabled = !circle;
  const distance = $("[data-derived='center_distance']");
  const angle = $("[data-derived='center_angle']");
  const source = $("[data-derived='source']");
  distance.textContent = circle?.center_distance === null || circle?.center_distance === undefined
    ? "—"
    : `${circle.center_distance.toFixed(2)} px`;
  angle.textContent = circle?.center_angle === null || circle?.center_angle === undefined
    ? "—"
    : `${circle.center_angle.toFixed(2)}°`;
  source.textContent = circle
    ? ({ automatic: "otomatik", manual: "manuel", manual_override: "manuel düzeltme" }[circle.source] || circle.source)
    : "—";
  const setDerived = (name, value) => {
    const element = $(`[data-derived='${name}']`);
    if (element) element.textContent = value;
  };
  setDerived(
    "processed_coordinates",
    circle ? `${Number(circle.processed_x ?? circle.x).toFixed(2)}, ${Number(circle.processed_y ?? circle.y).toFixed(2)}` : "—",
  );
  setDerived(
    "source_coordinates",
    circle ? `${Number(circle.source_x ?? circle.x).toFixed(2)}, ${Number(circle.source_y ?? circle.y).toFixed(2)}` : "—",
  );
  const qualityCircle = circle
    ? state.perspectiveQuality?.circle_metrics?.find((item) => item.circle_id === circle.id)
    : null;
  setDerived("coordinate_space", $("#qualityCoordinateSpace")?.value || "corrected");
  setDerived(
    "source_radius",
    circle ? `${Number(circle.source_radius ?? circle.radius).toFixed(2)} px` : "—",
  );
  setDerived(
    "corrected_radius",
    circle ? `${Number(circle.corrected_radius ?? circle.radius).toFixed(2)} px` : "—",
  );
  setDerived(
    "source_axes",
    circle
      ? `${qualityValue(qualityCircle?.source_major_axis ?? circle.source_major_axis, 2)} / ${qualityValue(qualityCircle?.source_minor_axis ?? circle.source_minor_axis, 2)}`
      : "—",
  );
  setDerived(
    "corrected_axes",
    circle
      ? `${qualityValue(qualityCircle?.corrected_major_axis ?? circle.corrected_major_axis, 2)} / ${qualityValue(qualityCircle?.corrected_minor_axis ?? circle.corrected_minor_axis, 2)}`
      : "—",
  );
  setDerived(
    "detector",
    circle
      ? `${circle.size_class || (circle.detector?.includes("main_") ? "ana değişken" : "—")} · ${circle.detector || "manuel"}`
      : "—",
  );
  for (const field of [
    "edge_score",
    "color_score",
    "circularity_score",
    "detector_agreement",
    "symmetry_support",
    "final_confidence",
  ]) {
    const value = circle?.[field];
    setDerived(field, value === null || value === undefined ? "—" : Number(value).toFixed(3));
  }
  setDerived(
    "suggested_classification",
    circle
      ? `${circle.suggested_circle_type || "—"} / ${circle.suggested_arm_id ? `K${circle.suggested_arm_id}` : "—"}`
      : "—",
  );
  setDerived(
    "suggested_relationship",
    circle
      ? `${circle.suggested_parent_id || "—"} / ${circle.suggested_side || "—"}`
      : "—",
  );
  setDerived(
    "local_angle",
    circle?.local_angle === null || circle?.local_angle === undefined
      ? "—"
      : `${Number(circle.local_angle).toFixed(2)}°`,
  );
  setDerived(
    "normalized_parent_distance",
    circle?.normalized_parent_distance === null ||
      circle?.normalized_parent_distance === undefined
      ? "—"
      : Number(circle.normalized_parent_distance).toFixed(3),
  );
  setDerived(
    "classification_source",
    circle
      ? `${circle.classification_source || "—"} · ${Number(circle.classification_confidence || 0).toFixed(2)}`
      : "—",
  );
  const dataList = $("#circleIds");
  const mainIds = approvedMainIds();
  dataList.innerHTML = state.circles
    .filter(
      (item) =>
        item.id !== state.selectedId &&
        (mainIds.size ? mainIds.has(item.id) : canonicalCircleType(item) === "main"),
    )
    .map((item) => `<option value="${escapeHtml(item.id)}"></option>`)
    .join("");
}

function updateArmModelUi() {
  $("#armModel").value = state.armModel;
  $("#armTwist").value = String(state.armTwistDegreesPer100px);
  $("#armTwist").disabled = state.armModel !== "spiral";
  $("#suggestArmModel").disabled =
    !state.imageId || !state.center || state.circles.length < 6;
}

function clearArmSuggestion() {
  state.armSuggestion = null;
  $("#applyArmSuggestion").classList.add("hidden");
  $("#armFitReadout").textContent =
    "Öneri uygulanmadan önce kullanıcı onayı gerekir.";
}

function updateUi() {
  $("#circleCount").textContent = `${state.circles.length} daire`;
  $("#centerReadout").textContent = state.center
    ? `${state.center.x.toFixed(1)}, ${state.center.y.toFixed(1)}`
    : "—";
  $("#armLegend").classList.toggle("hidden", !state.showArms || state.workspace !== "measure");
  $("#inspectWeakCircle").disabled = !state.imageId || !state.circles.length;
  $("#detectButton").disabled = !state.imageId || state.preserveLoadedCircleSet;
  $("#reindexArmsButton").disabled =
    !state.imageId || !state.center || !state.circles.length;
  $("#preprocessLegend").classList.toggle("hidden", state.workspace !== "preprocess");
  updatePerspectiveUi();
  updatePreprocessStats();
  updateTramlineUi();
  updateArmModelUi();
  updateInspector();
  renderPerspectiveQuality();
  if ($("#structureSummaryBody")) {
    renderStructureSummary();
    renderMainChainUi();
    renderSelectedRelationships();
    renderEditHistory();
    updateStructureLockUi();
    $("#undoStructureButton").disabled = state.structureUndo.length === 0;
  }
  render();
}

function clearSymmetry() {
  $("#symmetrySummary").classList.add("hidden");
  $("#symmetryDetails").classList.add("hidden");
  $("#symmetryRows").innerHTML = "";
}

function showSymmetry(payload) {
  const summary = payload.summary || {};
  $("#symmetrySummary").innerHTML = `
    <div><strong>${Number(summary.mean_image_mae || 0).toFixed(4)}</strong><span>ort. görüntü MAE</span></div>
    <div><strong>${Number(summary.mean_image_correlation || 0).toFixed(3)}</strong><span>ort. korelasyon</span></div>
    <div><strong>${Number(summary.mean_geometric_error || 0).toFixed(3)}</strong><span>ort. geometri hatası</span></div>
  `;
  $("#symmetrySummary").classList.remove("hidden");
  const imageByAngle = new Map((payload.image_symmetry || []).map((item) => [item.angle_degrees, item]));
  const geometryByAngle = new Map((payload.geometric_symmetry || []).map((item) => [item.angle_degrees, item]));
  $("#symmetryRows").innerHTML = [60, 120, 180, 240, 300].map((angle) => {
    const image = imageByAngle.get(angle) || {};
    const geometry = geometryByAngle.get(angle) || {};
    return `<tr>
      <td>${angle}°</td>
      <td>${Number(image.normalized_mae || 0).toFixed(4)}</td>
      <td>${Number(image.correlation || 0).toFixed(3)}</td>
      <td>${Number(geometry.normalized_position_error || 0).toFixed(3)}</td>
    </tr>`;
  }).join("");
  $("#symmetryDetails").classList.remove("hidden");
}

function localStructureSummary() {
  const arms = {};
  for (let arm = 1; arm <= 6; arm += 1) {
    const circles = state.circles.filter((circle) => circle.arm_id === arm);
    const types = circles.map(canonicalCircleType);
    arms[arm] = {
      total: circles.length,
      main: types.filter((value) => value === "main").length,
      satellite: types.filter((value) => value === "satellite").length,
      unclassified: types.filter((value) => value === "unclassified").length,
      target_difference: circles.length - 68,
    };
  }
  return {
    total_circles: state.circles.length,
    center_count: state.circles.filter(
      (circle) => canonicalCircleType(circle) === "center",
    ).length,
    total_difference: state.circles.length - 409,
    arms,
  };
}

function metricClass(value, target, warningTolerance = 4) {
  if (value === target) return "metric-ok";
  return Math.abs(value - target) <= warningTolerance ? "metric-warning" : "metric-error";
}

function renderStructureSummary() {
  const summary = localStructureSummary();
  $("#structureSummaryBody").innerHTML = Array.from({ length: 6 }, (_, index) => {
    const arm = index + 1;
    const item = summary.arms[arm];
    const rowClass =
      item.total === 68 &&
      item.main === 13 &&
      item.satellite === 55 &&
      item.unclassified === 0
        ? "metric-ok"
        : Math.abs(item.total - 68) <= 4
          ? "metric-warning"
          : "metric-error";
    const difference = item.target_difference > 0
      ? `+${item.target_difference}`
      : String(item.target_difference);
    return `<tr class="${rowClass}">
      <td>K${arm}</td>
      <td class="${metricClass(item.total, 68)}">${item.total}</td>
      <td class="${metricClass(item.main, 13, 2)}">${item.main}</td>
      <td class="${metricClass(item.satellite, 55, 3)}">${item.satellite}</td>
      <td class="${metricClass(item.unclassified, 0, 3)}">${item.unclassified}</td>
      <td class="${metricClass(item.target_difference, 0)}">${difference}</td>
    </tr>`;
  }).join("");
  const centerElement = $("#structureCenterCount");
  centerElement.textContent = `${summary.center_count} / 1`;
  centerElement.className = metricClass(summary.center_count, 1, 0);
  const totalElement = $("#structureTotalCount");
  totalElement.textContent = String(summary.total_circles);
  totalElement.className = metricClass(summary.total_circles, 409, 4);
  const differenceElement = $("#structureTotalDifference");
  differenceElement.textContent =
    summary.total_difference > 0
      ? `+${summary.total_difference}`
      : String(summary.total_difference);
  differenceElement.className = metricClass(summary.total_difference, 0, 4);
}

function renderCenterCandidate() {
  const candidate = state.structureReport?.center_candidate;
  if (!candidate) {
    $("#centerCandidateBox").textContent = "Merkez adayı üretilemedi.";
    $("#approveCenterCandidate").disabled = true;
    return;
  }
  $("#centerCandidateBox").innerHTML =
    `<strong>${escapeHtml(candidate.id)}</strong> · genel merkeze ` +
    `${Number(candidate.distance_to_general_center).toFixed(2)} px · ` +
    `r ${Number(candidate.radius).toFixed(2)} · güven ${Number(candidate.confidence).toFixed(3)}` +
    `${candidate.locked ? " · alan kilitli" : ""}`;
  const circle = state.circles.find((item) => item.id === candidate.id);
  $("#approveCenterCandidate").disabled =
    !circle ||
    canonicalCircleType(circle) === "center" ||
    Boolean(candidate.locked);
}

function renderBoundaryCandidates() {
  const topFourOnly = $("#boundaryTopFourOnly").checked;
  const rows = (state.structureReport?.boundary_candidates || []).filter(
    (item) => !topFourOnly || item.most_likely_four,
  );
  const validIds = new Set(rows.map((item) => String(item.id)));
  state.boundarySelection = new Set(
    [...state.boundarySelection].filter((id) => validIds.has(id)),
  );
  $("#boundaryCandidatesBody").innerHTML = rows.length
    ? rows.map((item) => {
      const checked = state.boundarySelection.has(String(item.id));
      return `<tr class="${item.most_likely_four ? "sequence-focus" : ""}">
        <td><input type="checkbox" data-boundary-id="${escapeHtml(item.id)}" ${checked ? "checked" : ""} ${item.manual_locked || Number(item.difference_score) <= 0 ? "disabled" : ""}></td>
        <td><button class="button ghost" data-focus-id="${escapeHtml(item.id)}">${escapeHtml(item.id)}</button></td>
        <td>K${item.current_arm_id}</td>
        <td>K${item.suggested_arm_id}</td>
        <td>${Number(item.current_arm_distance).toFixed(2)}</td>
        <td>${Number(item.suggested_arm_distance).toFixed(2)}</td>
        <td class="${Number(item.difference_score) > 0 ? "metric-warning" : ""}">${Number(item.difference_score).toFixed(2)}</td>
        <td>${item.manual_locked ? "kilitli" : "açık"}</td>
      </tr>`;
    }).join("")
    : '<tr><td colspan="8" class="muted">Bu filtre için aday yok.</td></tr>';
  $("#applySelectedBoundary").disabled = state.boundarySelection.size === 0;
  const selectableRows = rows.filter(
    (item) => !item.manual_locked && Number(item.difference_score) > 0,
  );
  $("#boundarySelectAll").checked =
    selectableRows.length > 0 &&
    selectableRows.every(
      (item) => state.boundarySelection.has(String(item.id)),
    );
}

function localSideLabel(value) {
  return LOCAL_SIDE_LABELS[value] || value || "Bilinmiyor";
}

function localSideMethodLabel(value) {
  return LOCAL_SIDE_METHOD_LABELS[value] || value || "—";
}

function visibleLocalSideCandidates() {
  const armFilter = $("#localSideArmFilter").value;
  const typeFilter = $("#localSideTypeFilter").value;
  const onlyChanged = $("#localSideOnlyChanged").checked;
  const onlyLowConfidence = $("#localSideOnlyLowConfidence").checked;
  const confidenceThreshold = Math.max(
    0,
    Math.min(1, Number($("#localSideMinConfidence").value || 0.6)),
  );
  return (state.structureReport?.local_side_candidates || []).filter((item) => {
    if (
      armFilter !== "all" &&
      (
        armFilter === "center"
          ? item.arm_id !== null
          : Number(item.arm_id) !== Number(armFilter)
      )
    ) return false;
    if (typeFilter !== "all" && item.suggested_side !== typeFilter) return false;
    if (onlyChanged && item.current_side === item.suggested_side) return false;
    if (
      onlyLowConfidence &&
      Number(item.confidence) >= confidenceThreshold &&
      item.suggested_side !== "unknown"
    ) return false;
    return true;
  });
}

function renderLocalSideCandidates() {
  const allRows = state.structureReport?.local_side_candidates || [];
  const allIds = new Set(
    allRows
      .filter((item) => !item.side_locked)
      .map((item) => String(item.id)),
  );
  state.localSideSelection = new Set(
    [...state.localSideSelection].filter((id) => allIds.has(id)),
  );
  const rows = visibleLocalSideCandidates();
  const confidenceThreshold = Math.max(
    0,
    Math.min(1, Number($("#localSideMinConfidence").value || 0.6)),
  );
  const counts = {
    inner: 0,
    outer: 0,
    centerline: 0,
    unknown: 0,
    center: 0,
  };
  const methods = {};
  for (const item of allRows) {
    counts[item.suggested_side] = (counts[item.suggested_side] || 0) + 1;
    methods[item.method] = (methods[item.method] || 0) + 1;
  }
  $("#localSideCoverage").textContent =
    `${allRows.length} / ${state.circles.length}`;
  $("#localSideSelectionCount").textContent =
    String(state.localSideSelection.size);
  $("#localSideLockedCount").textContent =
    String(allRows.filter((item) => item.side_locked).length);
  $("#localSideLowConfidenceCount").textContent = String(
    allRows.filter(
      (item) =>
        Number(item.confidence) < confidenceThreshold ||
        item.suggested_side === "unknown",
    ).length,
  );
  $("#localSideDistribution").textContent =
    `İç ${counts.inner} · Dış ${counts.outer} · Kol ekseni ${counts.centerline} · ` +
    `Merkez ${counts.center} · Bilinmiyor ${counts.unknown}`;
  const methodText = Object.entries(methods)
    .map(([method, count]) => `${localSideMethodLabel(method)} ${count}`)
    .join(" · ");
  $("#localSideMethodNote").textContent = allRows.length
    ? `${methodText}. Onaylı Ana Zincir varsa o geometri kullanılır; ` +
      "radyal farkta eksi İç, artı Dış yönüdür. " +
      "`left/right` değeri `legacy_side` içinde korunur."
    : "Onaylı Ana Zincir varsa o geometri, yoksa mevcut kol modeli kullanılır.";

  $("#localSideCandidatesBody").innerHTML = rows.length
    ? rows.map((item) => {
      const checked = state.localSideSelection.has(String(item.id));
      const confidence = Number(item.confidence || 0);
      const confidenceClass =
        confidence >= confidenceThreshold ? "metric-ok" : "metric-warning";
      const radialOffset = item.signed_radial_offset === null
        ? "—"
        : `${Number(item.signed_radial_offset) > 0 ? "+" : ""}${Number(item.signed_radial_offset).toFixed(2)}`;
      return `<tr>
        <td><input type="checkbox" data-local-side-id="${escapeHtml(item.id)}" ${checked ? "checked" : ""} ${item.side_locked ? "disabled" : ""}></td>
        <td><button class="button ghost" data-focus-id="${escapeHtml(item.id)}">${escapeHtml(item.id)}</button></td>
        <td>${item.arm_id === null ? "—" : `K${Number(item.arm_id)}`}</td>
        <td>${escapeHtml(localSideLabel(item.current_side))}</td>
        <td>${escapeHtml(localSideLabel(item.suggested_side))}</td>
        <td>${item.centerline_distance === null ? "—" : Number(item.centerline_distance).toFixed(2)}</td>
        <td>${radialOffset}</td>
        <td>${item.local_tangent_angle === null ? "—" : `${Number(item.local_tangent_angle).toFixed(1)}°`}</td>
        <td>${escapeHtml(localSideMethodLabel(item.method))}</td>
        <td class="${confidenceClass}">${confidence.toFixed(3)}</td>
        <td>${item.side_locked ? "kilitli" : "açık"}</td>
      </tr>`;
    }).join("")
    : '<tr><td colspan="11" class="muted">Bu filtre için yerel taraf adayı yok.</td></tr>';

  const safeVisibleRows = rows.filter(
    (item) =>
      !item.side_locked &&
      item.suggested_side !== "unknown" &&
      Number(item.confidence) >= confidenceThreshold,
  );
  $("#selectSafeLocalSides").disabled = safeVisibleRows.length === 0;
  $("#clearLocalSideSelection").disabled = state.localSideSelection.size === 0;
  $("#applySelectedLocalSides").disabled = state.localSideSelection.size === 0;
}

function renderSatelliteCandidates() {
  const selectedArm = Number($("#mainChainArm").value || 1);
  const rows = (state.structureReport?.satellite_candidates || []).filter(
    (item) => Number(item.arm_id) === selectedArm,
  );
  const validIds = new Set(rows.map((item) => String(item.id)));
  state.satelliteSelection = new Set(
    [...state.satelliteSelection].filter((id) => validIds.has(id)),
  );
  $("#satelliteCandidatesBody").innerHTML = rows.length
    ? rows.map((item) => `<tr>
      <td><input type="checkbox" data-satellite-id="${escapeHtml(item.id)}" ${state.satelliteSelection.has(String(item.id)) ? "checked" : ""} ${item.manual_locked ? "disabled" : ""}></td>
      <td><button class="button ghost" data-focus-id="${escapeHtml(item.id)}">${escapeHtml(item.id)}</button></td>
      <td>${escapeHtml(item.nearest_main_id)} · ${item.nearest_main_sequence}</td>
      <td>${item.second_nearest_main_id ? escapeHtml(item.second_nearest_main_id) : "—"}</td>
      <td>${item.distance_ratio === null ? "—" : Number(item.distance_ratio).toFixed(3)}</td>
      <td>${Number(item.normalized_parent_distance).toFixed(3)}</td>
      <td>${Number(item.local_angle).toFixed(1)}°</td>
      <td>${escapeHtml(item.suggested_side)}</td>
      <td>${Number(item.confidence).toFixed(3)}</td>
    </tr>`).join("")
    : '<tr><td colspan="9" class="muted">Bu kol için onaylı ana zincir veya yan çember adayı yok.</td></tr>';
  $("#applySelectedSatellites").disabled = state.satelliteSelection.size === 0;
}

function renderDuplicateCandidates() {
  const rows = state.structureReport?.duplicate_candidates || [];
  $("#duplicateCandidatesBody").innerHTML = rows.length
    ? rows.map((item, index) => {
      const remove = state.circles.find((circle) => circle.id === item.merge_candidate_id);
      const removalLocked = remove && (
        remove.locked || (remove._locked_fields || []).length > 0
      );
      return `<tr>
        <td><button class="button ghost" data-focus-id="${escapeHtml(item.id_a)}">${escapeHtml(item.id_a)}</button></td>
        <td><button class="button ghost" data-focus-id="${escapeHtml(item.id_b)}">${escapeHtml(item.id_b)}</button></td>
        <td>${Number(item.center_distance).toFixed(2)}</td>
        <td>${Number(item.radius_difference).toFixed(2)}</td>
        <td>${(Number(item.overlap_ratio) * 100).toFixed(1)}%</td>
        <td>${escapeHtml(item.source_a)} / ${escapeHtml(item.source_b)}</td>
        <td>${escapeHtml(item.preferred_id)}</td>
        <td><button class="button danger" data-merge-duplicate="${index}" ${removalLocked ? "disabled" : ""}>Birleştir</button></td>
      </tr>`;
    }).join("")
    : '<tr><td colspan="8" class="muted">Mükerrer ölçütlerini karşılayan çift yok.</td></tr>';
}

function renderSatelliteStatus() {
  const arm = String($("#satelliteTableArm").value || 1);
  const rows = state.structureReport?.satellite_tables?.arms?.[arm] || [];
  $("#satelliteStatusBody").innerHTML = rows.length
    ? rows.map((item) => `<tr class="${item.highlight ? "sequence-focus" : ""}">
      <td>${item.sequence_index}</td>
      <td>${item.main_radius === null ? "—" : Number(item.main_radius).toFixed(2)}</td>
      <td>${item.inner_count}</td>
      <td>${item.outer_count}</td>
      <td>${item.total_satellite}</td>
      <td>${item.inner_angle === null ? "—" : `${Number(item.inner_angle).toFixed(1)}°`}</td>
      <td>${item.outer_angle === null ? "—" : `${Number(item.outer_angle).toFixed(1)}°`}</td>
    </tr>`).join("")
    : '<tr><td colspan="7" class="muted">Denetimi yenileyin.</td></tr>';
}

function renderMainChainUi() {
  const arm = Number($("#mainChainArm").value || 1);
  const ids = state.mainChainDraft?.arm === arm
    ? state.mainChainDraft.ids
    : state.mainChains[arm] || [];
  $("#mainChainReadout").textContent = `${ids.length} / 13`;
  $("#mainChainReadout").className =
    ids.length === 13 ? "metric-ok" : ids.length > 13 ? "metric-error" : "metric-warning";
  $("#mainChainList").innerHTML = ids.length
    ? ids.map((id, index) => `<li draggable="${Boolean(state.mainChainDraft)}" data-chain-index="${index}">
      <span>${index + 1}</span>
      <button class="button ghost" data-focus-id="${escapeHtml(id)}">${escapeHtml(id)}</button>
      <button type="button" data-chain-move="-1" data-chain-index="${index}" ${index === 0 || !state.mainChainDraft ? "disabled" : ""}>↑</button>
      <button type="button" data-chain-move="1" data-chain-index="${index}" ${index === ids.length - 1 || !state.mainChainDraft ? "disabled" : ""}>↓</button>
      <button type="button" data-chain-remove="${index}" ${!state.mainChainDraft ? "disabled" : ""}>×</button>
    </li>`).join("")
    : '<li class="muted">Bir kol seçip taslağı açın; merkezden dışa dairelere tıklayın.</li>';
  $("#startMainChain").textContent = state.mainChainDraft ? "Taslak açık" : "Zincir taslağını aç";
  $("#startMainChain").disabled = Boolean(state.mainChainDraft);
  $("#cancelMainChain").disabled = !state.mainChainDraft;
  $("#confirmMainChain").disabled = !state.mainChainDraft || ids.length !== 13;
  $("#mainChainArm").disabled = Boolean(state.mainChainDraft);
}

function renderSelectedRelationships() {
  const circle = selectedCircle();
  const box = $("#selectedRelationships");
  const unlink = $("#unlinkSelectedParent");
  if (!circle) {
    box.textContent = "Bir ana veya yan çember seçildiğinde bağlantıları burada görünür.";
    unlink.disabled = true;
    return;
  }
  const children = state.circles.filter(
    (item) => canonicalCircleType(item) === "satellite" && item.parent_id === circle.id,
  );
  if (canonicalCircleType(circle) === "satellite") {
    box.innerHTML = `<strong>${escapeHtml(circle.id)}</strong> → ${circle.parent_id ? escapeHtml(circle.parent_id) : "parent yok"} · ${circle.side || "unknown"} · ${circle.local_angle === null ? "açı —" : `${Number(circle.local_angle).toFixed(1)}°`}`;
    unlink.disabled = !circle.parent_id || circleFieldLocked(circle, "parent_id");
  } else {
    box.innerHTML = `<strong>${escapeHtml(circle.id)}</strong> · bağlı yan çember: ${children.length}` +
      (children.length ? `<br>${children.map((item) => escapeHtml(item.id)).join(", ")}` : "");
    unlink.disabled = true;
  }
}

function renderEditHistory() {
  const rows = state.editHistory.slice(-50).reverse();
  $("#editHistoryList").innerHTML = rows.length
    ? rows.map((item) => `<div class="edit-history-item">
      <strong>${escapeHtml(item.operation_type || "değişiklik")}</strong> · ${escapeHtml(item.circle_id || "toplu")} · ${escapeHtml(item.field || "—")}<br>
      ${escapeHtml(item.actor || "user")} · ${escapeHtml(item.timestamp || "")}
    </div>`).join("")
    : '<p class="muted">Henüz yapısal değişiklik yok.</p>';
}

function renderStructureReport() {
  renderCenterCandidate();
  renderBoundaryCandidates();
  renderLocalSideCandidates();
  renderSatelliteCandidates();
  renderDuplicateCandidates();
  renderSatelliteStatus();
}

function applyReportSuggestions(report) {
  const suggestionMap = report?.suggestions || {};
  const suggestedFields = [
    "suggested_circle_type",
    "suggested_arm_id",
    "suggested_sequence_index",
    "suggested_parent_id",
    "suggested_side",
    "suggestion_confidence",
  ];
  const changes = [];
  for (const circle of state.circles) {
    const suggestion = suggestionMap[circle.id] || {};
    for (const field of suggestedFields) {
      const next = Object.prototype.hasOwnProperty.call(suggestion, field)
        ? suggestion[field]
        : null;
      if (!structuralValuesEqual(circle[field] ?? null, next)) {
        changes.push({ circle, field, previous: circle[field] ?? null, next });
      }
    }
    circle._suggested_local_angle = suggestion.local_angle ?? null;
    circle._suggested_normalized_parent_distance =
      suggestion.normalized_parent_distance ?? null;
  }
  if (!changes.length) return;
  const batchId = `suggest-${Date.now().toString(36)}`;
  for (const change of changes) {
    change.circle[change.field] = change.next;
    recordStructuralEdit(
      change.circle.id,
      change.field,
      change.previous,
      change.next,
      "automatic_suggestion",
      "refresh_structure_suggestions",
      batchId,
    );
  }
}

async function refreshStructureAnalysis(showBusy = true) {
  renderStructureSummary();
  if (!state.imageId || !state.center) {
    if (showBusy) setStatus("Kol denetimi için önce görüntüyü açıp genel merkezi seçin.", "error");
    return null;
  }
  if (showBusy) setBusy(true, "Yapısal öneriler hesaplanıyor…");
  try {
    const report = await api("/api/structure/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        circles: state.circles,
        manual_locks: manualLocksPayload(),
        structure_state: { main_chains: state.mainChains },
        center: state.center,
        phase_degrees: state.phaseDegrees,
        ...armSettingsPayload(),
      }),
    });
    applyReportSuggestions(report);
    state.structureReport = report;
    renderStructureReport();
    updateInspector();
    renderEditHistory();
    render();
    if (showBusy) {
      setStatus(
        `${report.summary.total_circles} çember korundu; öneriler onay bekliyor.`,
        "success",
      );
    }
    return report;
  } catch (error) {
    if (showBusy) setStatus(error.message, "error");
    return null;
  } finally {
    if (showBusy) setBusy(false);
  }
}

function approveCenterCandidate() {
  const candidate = state.structureReport?.center_candidate;
  const circle = state.circles.find((item) => item.id === candidate?.id);
  if (!candidate || !circle || candidate.locked) return;
  pushStructureUndo("approve_center");
  const batchId = `center-${Date.now().toString(36)}`;
  applyCircleChanges(
    circle,
    {
      circle_type: "center",
      arm_id: null,
      sequence_index: null,
      parent_id: null,
      side: "center",
      classification_source: "user_confirmed",
      classification_confidence: Number(candidate.confidence || 1),
    },
    {
      operationType: "approve_center",
      batchId,
      lockFields: ["circle_type", "arm_id", "sequence_index", "parent_id", "side"],
    },
  );
  for (let arm = 1; arm <= 6; arm += 1) {
    state.mainChains[arm] = (state.mainChains[arm] || []).filter(
      (id) => id !== circle.id,
    );
  }
  updateUi();
  void refreshStructureAnalysis(false);
  setStatus(`${circle.id} kullanıcı onayıyla merkez olarak kaydedildi.`, "success");
}

function clearAllCircleLocks() {
  const summary = circleLockSummary();
  if (!summary.totalLocks) {
    setStatus("Bütün daire ve alan kilitleri zaten açık.");
    return;
  }
  const approved = window.confirm(
    `${summary.affectedCircles} çemberdeki ${summary.fieldLocks} alan kilidi ` +
    `ve ${summary.recordLocks} genel kilit kaldırılacak. ` +
    "Geometri ve sınıflandırma değerleri değişmeyecek. Devam edilsin mi?",
  );
  if (!approved) return;

  pushStructureUndo("clear_all_circle_locks");
  const batchId = `unlock-all-${Date.now().toString(36)}`;
  let changedCircles = 0;
  for (const circle of state.circles) {
    const previousFields = [...new Set([
      ...(circle._locked_fields || []),
      ...(circle.locked_fields || []),
    ])];
    const hadRecordLock = Boolean(circle.locked);
    if (!hadRecordLock && !previousFields.length) continue;
    if (hadRecordLock) {
      recordStructuralEdit(
        circle.id,
        "locked",
        true,
        false,
        "user",
        "clear_all_circle_locks",
        batchId,
      );
    }
    if (previousFields.length) {
      recordStructuralEdit(
        circle.id,
        "locked_fields",
        previousFields,
        [],
        "user",
        "clear_all_circle_locks",
        batchId,
      );
    }
    circle.locked = false;
    circle._locked_fields = [];
    circle.locked_fields = [];
    changedCircles += 1;
  }

  updateUi();
  void refreshStructureAnalysis(false);
  setStatus(
    `${changedCircles} çemberin kilitleri tek işlemde açıldı; ` +
    "geometri ve sınıflandırmalar korunuyor.",
    "success",
  );
}

function addMainChainDraftCircle(circleId) {
  const draft = state.mainChainDraft;
  const circle = state.circles.find((item) => item.id === circleId);
  if (!draft || !circle) {
    setStatus("Önce Ana Zincir taslağını açın.", "error");
    return;
  }
  if (canonicalCircleType(circle) === "center") {
    setStatus("Merkez çemberi ana zincire eklenemez.", "error");
    return;
  }
  const otherArm = Object.entries(state.mainChains).find(
    ([rawArm, ids]) => Number(rawArm) !== draft.arm && ids.includes(circleId),
  );
  if (otherArm) {
    setStatus(`${circleId} zaten K${otherArm[0]} ana zincirinde.`, "error");
    return;
  }
  if (draft.ids.includes(circleId)) {
    setStatus(`${circleId} taslakta zaten seçili.`, "warning");
    return;
  }
  if (draft.ids.length >= 13) {
    setStatus("Ana zincir 13 çembere ulaştı; sırayı düzenleyin veya onaylayın.", "warning");
    return;
  }
  draft.ids.push(circleId);
  state.selectedId = circleId;
  updateUi();
}

function startMainChainDraft() {
  if (state.mainChainDraft) return;
  const arm = Number($("#mainChainArm").value || 1);
  state.mainChainDraft = {
    arm,
    ids: [...(state.mainChains[arm] || [])],
  };
  setTool("mainChain");
  updateUi();
  setStatus(`K${arm} Ana Zincir taslağı açık; merkezden dışa daireleri seçin.`, "success");
}

function cancelMainChainDraft() {
  if (!state.mainChainDraft) return;
  state.mainChainDraft = null;
  setTool("select");
  updateUi();
  setStatus("Ana Zincir taslağı değişiklik yapmadan kapatıldı.");
}

function moveMainChainItem(index, direction) {
  const draft = state.mainChainDraft;
  if (!draft) return;
  const target = index + direction;
  if (index < 0 || index >= draft.ids.length || target < 0 || target >= draft.ids.length) return;
  [draft.ids[index], draft.ids[target]] = [draft.ids[target], draft.ids[index]];
  updateUi();
}

function removeMainChainDraftItem(index) {
  const draft = state.mainChainDraft;
  if (!draft || index < 0 || index >= draft.ids.length) return;
  draft.ids.splice(index, 1);
  updateUi();
}

function confirmMainChainDraft() {
  const draft = state.mainChainDraft;
  if (!draft || draft.ids.length !== 13) return;
  const circles = draft.ids.map((id) => state.circles.find((item) => item.id === id));
  if (circles.some((circle) => !circle)) {
    setStatus("Taslakta artık bulunmayan bir daire var; taslağı yeniden oluşturun.", "error");
    return;
  }
  for (const [index, circle] of circles.entries()) {
    const desired = { circle_type: "main", arm_id: draft.arm, sequence_index: index + 1, parent_id: null };
    const conflict = Object.entries(desired).find(
      ([field, value]) => circleFieldLocked(circle, field) && !structuralValuesEqual(circle[field], value),
    );
    if (conflict) {
      setStatus(`${circle.id} için ${conflict[0]} alanı kilitli; önce kilidi kaldırın.`, "error");
      return;
    }
  }
  const previousIds = [...(state.mainChains[draft.arm] || [])];
  const removedIds = previousIds.filter((id) => !draft.ids.includes(id));
  for (const removedId of removedIds) {
    const circle = state.circles.find((item) => item.id === removedId);
    if (
      circle &&
      (circleFieldLocked(circle, "circle_type") || circleFieldLocked(circle, "sequence_index"))
    ) {
      setStatus(`${removedId} zincirden çıkarılmak için kilitli; önce alan kilidini kaldırın.`, "error");
      return;
    }
  }
  pushStructureUndo("confirm_main_chain");
  const batchId = `chain-${draft.arm}-${Date.now().toString(36)}`;
  for (const removedId of removedIds) {
    const circle = state.circles.find((item) => item.id === removedId);
    if (!circle) continue;
    applyCircleChanges(
      circle,
      {
        circle_type: "unclassified",
        sequence_index: null,
        parent_id: null,
        classification_source: "user_confirmed",
        classification_confidence: 1,
      },
      {
        operationType: "confirm_main_chain",
        batchId,
        lockFields: ["circle_type", "sequence_index"],
      },
    );
  }
  circles.forEach((circle, index) => {
    applyCircleChanges(
      circle,
      {
        circle_type: "main",
        arm_id: draft.arm,
        sequence_index: index + 1,
        parent_id: null,
        classification_source: "user_confirmed",
        classification_confidence: 1,
      },
      {
        operationType: "confirm_main_chain",
        batchId,
        lockFields: ["circle_type", "arm_id", "sequence_index", "parent_id"],
      },
    );
  });
  state.mainChains[draft.arm] = [...draft.ids];
  state.mainChainDraft = null;
  setTool("select");
  updateUi();
  void refreshStructureAnalysis(false);
  setStatus(`K${draft.arm} Ana Zinciri 13 çemberle tek işlemde onaylandı.`, "success");
}

function applyBoundarySelection() {
  const selected = (state.structureReport?.boundary_candidates || []).filter(
    (item) =>
      state.boundarySelection.has(String(item.id)) &&
      !item.manual_locked &&
      Number(item.difference_score) > 0,
  );
  if (!selected.length) return;
  pushStructureUndo("transfer_k1_k6");
  const batchId = `boundary-${Date.now().toString(36)}`;
  let changed = 0;
  for (const candidate of selected) {
    const circle = state.circles.find((item) => item.id === candidate.id);
    if (!circle || circleFieldLocked(circle, "arm_id")) continue;
    if (applyCircleChanges(
      circle,
      {
        arm_id: Number(candidate.suggested_arm_id),
        classification_source: "user_confirmed",
        classification_confidence: Number(candidate.confidence || 0),
      },
      {
        operationType: "transfer_k1_k6",
        batchId,
        lockFields: ["arm_id"],
      },
    )) changed += 1;
  }
  if (!changed) state.structureUndo.pop();
  state.boundarySelection.clear();
  updateUi();
  void refreshStructureAnalysis(false);
  setStatus(`${changed} kol ataması kullanıcı onayıyla tek adımda değiştirildi.`, "success");
}

function applyLocalSideSelection() {
  const selected = (
    state.structureReport?.local_side_candidates || []
  ).filter(
    (item) =>
      state.localSideSelection.has(String(item.id)) &&
      !item.side_locked,
  );
  if (!selected.length) return;
  pushStructureUndo("approve_local_sides");
  const batchId = `local-sides-${Date.now().toString(36)}`;
  let approved = 0;
  for (const candidate of selected) {
    const circle = state.circles.find((item) => item.id === candidate.id);
    if (!circle || circleFieldLocked(circle, "side")) continue;
    const changes = {
      side: candidate.suggested_side,
      classification_source: "user_confirmed",
      classification_confidence: Number(candidate.confidence || 0),
    };
    if (
      ["left", "right"].includes(circle.side) &&
      !circle.legacy_side
    ) {
      changes.legacy_side = circle.side;
    }
    applyCircleChanges(
      circle,
      changes,
      {
        operationType: "approve_local_sides",
        batchId,
        lockFields: ["side"],
      },
    );
    approved += 1;
  }
  if (!approved) state.structureUndo.pop();
  state.localSideSelection.clear();
  updateUi();
  renderLocalSideCandidates();
  void refreshStructureAnalysis(false);
  setStatus(
    `${approved} çemberin yerel tarafı kullanıcı onayıyla atandı ve tek adımda kilitlendi.`,
    "success",
  );
}

function applySatelliteSelection() {
  const selected = (state.structureReport?.satellite_candidates || []).filter(
    (item) => state.satelliteSelection.has(String(item.id)) && !item.manual_locked,
  );
  if (!selected.length) return;
  pushStructureUndo("approve_satellite_relationships");
  const batchId = `satellites-${Date.now().toString(36)}`;
  let changed = 0;
  for (const candidate of selected) {
    const circle = state.circles.find((item) => item.id === candidate.id);
    if (!circle) continue;
    const fields = ["circle_type", "arm_id", "sequence_index", "parent_id", "side"];
    const desired = {
      circle_type: "satellite",
      arm_id: Number(candidate.suggested_arm_id),
      sequence_index: null,
      parent_id: candidate.suggested_parent_id,
      side: candidate.suggested_side,
    };
    if (fields.some(
      (field) =>
        circleFieldLocked(circle, field) &&
        !structuralValuesEqual(
          field === "circle_type" ? canonicalCircleType(circle) : circle[field],
          desired[field],
        ),
    )) continue;
    if (applyCircleChanges(
      circle,
      {
        ...desired,
        local_angle: Number(candidate.local_angle),
        normalized_parent_distance: Number(candidate.normalized_parent_distance),
        classification_source: "user_confirmed",
        classification_confidence: Number(candidate.confidence || 0),
      },
      {
        operationType: "approve_satellite_relationships",
        batchId,
        lockFields: fields,
      },
    )) changed += 1;
  }
  if (!changed) state.structureUndo.pop();
  state.satelliteSelection.clear();
  updateUi();
  void refreshStructureAnalysis(false);
  setStatus(`${changed} yan çember eşlemesi tek adımda onaylandı.`, "success");
}

function relationshipMetrics(circle, parent) {
  const arm = Number(parent.arm_id);
  const chainIds = state.mainChains[arm] || [];
  const index = chainIds.indexOf(parent.id);
  const chain = chainIds
    .map((id) => state.circles.find((item) => item.id === id))
    .filter(Boolean);
  let start = parent;
  let end = parent;
  if (chain.length >= 2 && index >= 0) {
    if (index === 0) [start, end] = [chain[0], chain[1]];
    else if (index === chain.length - 1) [start, end] = [chain[index - 1], chain[index]];
    else [start, end] = [chain[index - 1], chain[index + 1]];
  } else if (state.center) {
    start = { x: state.center.x, y: state.center.y };
  }
  const tangentAngle = Math.atan2(-(end.y - start.y), end.x - start.x);
  const relativeAngle = Math.atan2(-(circle.y - parent.y), circle.x - parent.x);
  const localAngle = Math.atan2(
    Math.sin(relativeAngle - tangentAngle),
    Math.cos(relativeAngle - tangentAngle),
  ) * 180 / Math.PI;
  return {
    local_angle: localAngle,
    normalized_parent_distance:
      Math.hypot(circle.x - parent.x, circle.y - parent.y) / Math.max(parent.radius, 0.1),
  };
}

function unlinkSelectedParent() {
  const circle = selectedCircle();
  if (!circle || !circle.parent_id || circleFieldLocked(circle, "parent_id")) return;
  pushStructureUndo("unlink_satellite_parent");
  applyCircleChanges(
    circle,
    {
      parent_id: null,
      local_angle: null,
      normalized_parent_distance: null,
      classification_source: "user_confirmed",
    },
    {
      operationType: "unlink_satellite_parent",
      lockFields: ["parent_id"],
    },
  );
  updateUi();
  void refreshStructureAnalysis(false);
  setStatus(`${circle.id} parent bağlantısı kullanıcı onayıyla kaldırıldı.`, "success");
}

function mergeDuplicateCandidate(index) {
  const candidate = state.structureReport?.duplicate_candidates?.[index];
  if (!candidate) return;
  const preferred = state.circles.find((circle) => circle.id === candidate.preferred_id);
  const removed = state.circles.find((circle) => circle.id === candidate.merge_candidate_id);
  if (!preferred || !removed || removed.locked || (removed._locked_fields || []).length) return;
  const lockedChild = state.circles.find(
    (circle) =>
      circle.parent_id === removed.id &&
      circleFieldLocked(circle, "parent_id") &&
      circle.parent_id !== preferred.id,
  );
  if (lockedChild) {
    setStatus(`${lockedChild.id} parent_id alanı kilitli; mükerrer birleştirme durduruldu.`, "error");
    return;
  }
  const preferredArm = Object.entries(state.mainChains).find(([, ids]) =>
    ids.includes(preferred.id)
  )?.[0];
  const removedArm = Object.entries(state.mainChains).find(([, ids]) =>
    ids.includes(removed.id)
  )?.[0];
  if (preferredArm && removedArm && preferredArm !== removedArm) {
    setStatus("İki kayıt farklı Ana Zincirlerde; zincirlerden biri düzenlenmeden birleştirilemez.", "error");
    return;
  }
  const approved = window.confirm(
    `${removed.id} kaydı ${preferred.id} ile birleştirilecek ve ${preferred.id} korunacak. Devam edilsin mi?`,
  );
  if (!approved) return;
  pushStructureUndo("merge_duplicate");
  const batchId = `duplicate-${Date.now().toString(36)}`;
  for (const circle of state.circles) {
    if (circle.parent_id !== removed.id) continue;
    applyCircleChanges(
      circle,
      { parent_id: canonicalCircleType(preferred) === "main" ? preferred.id : null },
      { operationType: "merge_duplicate", batchId },
    );
  }
  for (let arm = 1; arm <= 6; arm += 1) {
    const ids = state.mainChains[arm] || [];
    state.mainChains[arm] = ids
      .map((id) => id === removed.id ? preferred.id : id)
      .filter((id, itemIndex, values) => values.indexOf(id) === itemIndex);
  }
  recordStructuralEdit(
    removed.id,
    "circle_record",
    removed,
    null,
    "user",
    "merge_duplicate",
    batchId,
  );
  if (removed.id.startsWith("auto-")) {
    state.deletedAutoIds.add(removed.id);
    suppressAutomatic(removed);
  }
  state.circles = state.circles.filter((circle) => circle.id !== removed.id);
  state.selectedId = preferred.id;
  updateUi();
  void refreshStructureAnalysis(false);
  setStatus(`${removed.id}, ${preferred.id} içinde kullanıcı onayıyla birleştirildi.`, "success");
}

function syncControlsFromProfile() {
  const profile = state.preprocessProfile;
  if (!profile) return;
  const perspective = operationByType("perspective");
  const illumination = operationByType("illumination");
  const clahe = operationByType("clahe");
  if (perspective) {
    $("#perspectiveMode").value = perspective.params.mode || "reference_quad";
    $("#preservePerspectiveFrame").checked =
      perspective.params.preserve_full_image !== false;
    state.perspectivePoints = (perspective.params.points || []).map((point) => ({ x: Number(point.x), y: Number(point.y) }));
  }
  if (illumination) {
    $("#illuminationEnabled").checked = Boolean(illumination.enabled);
    $("#illuminationMethod").value = illumination.params.method || "subtract";
    $("#illuminationEstimator").value = illumination.params.estimator || "gaussian";
    $("#illuminationScale").value = String(illumination.params.scale || 61);
    $("#illuminationStrength").value = String(illumination.params.strength ?? 1);
  }
  if (clahe) {
    $("#claheEnabled").checked = Boolean(clahe.enabled);
    $("#claheClipLimit").value = String(clahe.params.clip_limit ?? 2);
    $("#claheGridSize").value = String(clahe.params.tile_grid_size ?? 8);
  }
  for (const sizeClass of ["small", "medium", "large"]) {
    const settings = profile.size_classes[sizeClass];
    $(`#${sizeClass}MinRadius`).value = String(settings.min_radius);
    $(`#${sizeClass}MaxRadius`).value = String(settings.max_radius);
  }
  const mainArm = profile.detection.main_arm || {};
  $("#mainArmEnabled").checked = mainArm.enabled !== false;
  $("#mainArmMinRadius").value = String(mainArm.min_radius ?? 10);
  $("#mainArmMaxRadius").value = String(mainArm.max_radius ?? 48);
  $("#mainArmMinConfidence").value = Number(
    mainArm.min_final_confidence ?? 0.5,
  ).toFixed(2);
  $("#mainArmHoughThreshold").value = String(
    mainArm.hough_accumulator_threshold ?? 48,
  );
  $("#watershedEnabled").checked = Boolean(profile.detection.watershed);
  $("#tileSize").value = String(profile.detection.tile_size || 512);
  $("#tileOverlap").value = String(profile.detection.tile_overlap ?? 0.2);
  $("#minFinalConfidence").value = Number(
    profile.detection.min_final_confidence ?? 0.5,
  ).toFixed(2);
  $("#tramlineMaskWidth").value = String(profile.tramline.mask_width || 7);
  $("#tramlineMaskEnabled").checked = Boolean(profile.tramline.enabled);
  const tramlineDetection = profile.tramline.detection || {};
  $("#tramlineMinLength").value = String(
    tramlineDetection.min_length_ratio ?? 0.28,
  );
  $("#tramlineAngleTolerance").value = String(
    tramlineDetection.angle_tolerance_degrees ?? 9,
  );
  $("#tramlineExpectedSpacing").value = String(
    tramlineDetection.expected_spacing_pixels ?? 0,
  );
  const regularityPercent = Math.round(
    Math.max(0, Math.min(1, Number(tramlineDetection.regularity ?? 0.72))) * 100,
  );
  $("#tramlineRegularity").value = String(regularityPercent);
  $("#tramlineRegularityReadout").textContent = `${regularityPercent}%`;
  const display = profile.display || {};
  state.showExclusionMask = display.show_exclusion_mask !== false;
  state.showRoi = display.show_analysis_polygon !== false;
  state.maskOpacity = Math.max(0, Math.min(1, Number(display.mask_opacity ?? 0.45)));
  $("#showExclusionMask").checked = state.showExclusionMask;
  $("#showRoi").checked = state.showRoi;
  $("#maskOpacity").value = String(Math.round(state.maskOpacity * 100));
  $("#maskOpacityReadout").textContent = `${Math.round(state.maskOpacity * 100)}%`;
  $("#showTramlines").checked = display.show_tramline_suggestion !== false;
  hydrateTramlineSuggestionFromProfile();
  updatePerspectiveUi();
  updateUi();
}

function syncProfileFromControls() {
  const profile = state.preprocessProfile;
  if (!profile) return;
  const perspective = operationByType("perspective");
  if (perspective) {
    perspective.params.mode = $("#perspectiveMode").value;
    perspective.params.points = state.perspectivePoints.map(strokePoint);
    perspective.params.preserve_full_image =
      $("#preservePerspectiveFrame").checked;
  }
  const illumination = operationByType("illumination");
  if (illumination) {
    illumination.enabled = $("#illuminationEnabled").checked;
    illumination.params.method = $("#illuminationMethod").value;
    illumination.params.estimator = $("#illuminationEstimator").value;
    illumination.params.scale = Math.max(3, Number($("#illuminationScale").value || 61));
    illumination.params.strength = Math.max(0, Math.min(1, Number($("#illuminationStrength").value || 1)));
  }
  const clahe = operationByType("clahe");
  if (clahe) {
    clahe.enabled = $("#claheEnabled").checked;
    clahe.params.clip_limit = Math.max(0.1, Number($("#claheClipLimit").value || 2));
    clahe.params.tile_grid_size = Math.max(2, Number($("#claheGridSize").value || 8));
  }
  for (const sizeClass of ["small", "medium", "large"]) {
    profile.size_classes[sizeClass].min_radius = Math.max(1, Number($(`#${sizeClass}MinRadius`).value));
    profile.size_classes[sizeClass].max_radius = Math.max(
      profile.size_classes[sizeClass].min_radius + 0.1,
      Number($(`#${sizeClass}MaxRadius`).value),
    );
  }
  const mainMinimum = Math.max(
    1,
    Number($("#mainArmMinRadius").value || 10),
  );
  profile.detection.main_arm = {
    ...(profile.detection.main_arm || {}),
    enabled: $("#mainArmEnabled").checked,
    min_radius: mainMinimum,
    max_radius: Math.max(
      mainMinimum + 0.1,
      Number($("#mainArmMaxRadius").value || 48),
    ),
    min_final_confidence: Math.max(
      0,
      Math.min(1, Number($("#mainArmMinConfidence").value || 0.5)),
    ),
    hough_accumulator_threshold: Math.max(
      10,
      Number($("#mainArmHoughThreshold").value || 48),
    ),
    exclude_sampled_side_ranges: true,
  };
  profile.detection.watershed = $("#watershedEnabled").checked;
  profile.detection.tile_size = Math.max(128, Number($("#tileSize").value || 512));
  profile.detection.tile_overlap = Math.max(0, Math.min(0.45, Number($("#tileOverlap").value || 0.2)));
  profile.detection.min_final_confidence = Math.max(
    0,
    Math.min(1, Number($("#minFinalConfidence").value || 0.5)),
  );
  profile.detection.native_resolution_only = true;
  profile.tramline.mask_width = Math.max(1, Number($("#tramlineMaskWidth").value || 7));
  profile.tramline.enabled = $("#tramlineMaskEnabled").checked;
  profile.tramline.detection = {
    min_length_ratio: Math.max(
      0.1,
      Math.min(0.9, Number($("#tramlineMinLength").value || 0.28)),
    ),
    angle_tolerance_degrees: Math.max(
      2,
      Math.min(30, Number($("#tramlineAngleTolerance").value || 9)),
    ),
    expected_spacing_pixels: Math.max(
      0,
      Number($("#tramlineExpectedSpacing").value || 0),
    ),
    regularity: Math.max(
      0,
      Math.min(1, Number($("#tramlineRegularity").value || 72) / 100),
    ),
  };
  profile.display = {
    show_exclusion_mask: state.showExclusionMask,
    show_analysis_polygon: state.showRoi,
    mask_opacity: state.maskOpacity,
    show_tramline_suggestion: $("#showTramlines").checked,
  };
}

function operationSummary(operation) {
  if (operation.type === "perspective") {
    return operation.enabled
      ? `${operation.params.mode} · ${(operation.params.points || []).length}/4 nokta · ${operation.params.preserve_full_image === false ? "seçimi kırp" : "tam kare"}`
      : "Kapalı · kaynak geometri";
  }
  if (operation.type === "illumination") {
    return operation.enabled ? `${operation.params.method} · ${operation.params.scale}px` : "Kapalı";
  }
  if (operation.type === "clahe") {
    return operation.enabled ? `clip ${operation.params.clip_limit} · grid ${operation.params.tile_grid_size}` : "Kapalı";
  }
  return operation.enabled ? "LAB örneklerinden olasılık" : "Kapalı";
}

function renderOperationStack() {
  const profile = state.preprocessProfile;
  if (!profile) return;
  const stepsById = new Map(state.preprocessSteps.map((step) => [step.id, step]));
  $("#operationStack").innerHTML = profile.operations.map((operation) => {
    const step = stepsById.get(operation.id);
    return `<div class="operation-item">
      <div class="operation-item-header">
        <label><input type="checkbox" data-operation-toggle="${escapeHtml(operation.id)}" ${operation.enabled ? "checked" : ""}>${escapeHtml(operation.type)}</label>
        <span class="muted">${operation.enabled ? "açık" : "kapalı"}</span>
      </div>
      <p>${escapeHtml(operationSummary(operation))}</p>
      <div class="operation-actions">
        <button class="button ghost" data-step-preview="before" data-step-id="${escapeHtml(operation.id)}" ${step ? "" : "disabled"}>Önce</button>
        <button class="button ghost" data-step-preview="after" data-step-id="${escapeHtml(operation.id)}" ${step ? "" : "disabled"}>Sonra</button>
      </div>
    </div>`;
  }).join("");
  $$("[data-operation-toggle]").forEach((input) => {
    input.addEventListener("change", async (event) => {
      const operation = state.preprocessProfile.operations.find((item) => item.id === event.target.dataset.operationToggle);
      if (!operation) return;
      pushPreprocessHistory();
      operation.enabled = event.target.checked;
      syncControlsFromProfile();
      await applyPreprocessStack();
    });
  });
  $$("[data-step-preview]").forEach((button) => {
    button.addEventListener("click", () => {
      const step = state.preprocessSteps.find((item) => item.id === button.dataset.stepId);
      if (!step) return;
      const side = button.dataset.stepPreview;
      showStepPreview(step[`${side}_url`], `${step.type} · ${side === "before" ? "önce" : "sonra"}`);
    });
  });
}

function renderArtifactButtons() {
  const labels = {
    rectified: "Rectified",
    illumination_corrected: "Aydınlatma",
    foreground_probability: "Olasılık",
    exclusion_mask: "Dışlama",
    measurement: "Ölçüm",
  };
  const available = Object.entries(labels).filter(([key]) => state.preprocessArtifacts[key]);
  $("#artifactPreviewButtons").classList.toggle("hidden", !available.length);
  $("#artifactPreviewButtons").innerHTML = available.map(([key, label]) =>
    `<button class="button ghost" data-artifact-preview="${key}">${label}</button>`
  ).join("");
  $$("[data-artifact-preview]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.artifactPreview;
      showStepPreview(state.preprocessArtifacts[key], labels[key]);
    });
  });
}

function showStepPreview(url, label) {
  if (!url) return;
  state.stepPreviewOpen = true;
  $("#stepPreviewImage").src = `${url}${url.includes("?") ? "&" : "?"}preview=${Date.now()}`;
  $("#stepPreviewLabel").textContent = label;
  $("#stepPreview").classList.remove("hidden");
}

function closeStepPreview() {
  state.stepPreviewOpen = false;
  $("#stepPreview").classList.add("hidden");
  $("#stepPreviewImage").removeAttribute("src");
}

async function initializePreprocessState() {
  if (!state.imageId) return;
  const payload = await api(`/api/preprocess/${state.imageId}`);
  state.preprocessProfile = payload.profile;
  state.defaultPreprocessProfile = deepClone(payload.profile);
  state.preprocessHistory = [];
  state.preprocessApplied = Boolean(payload.applied);
  state.appliedPreprocessSignature = state.preprocessApplied
    ? JSON.stringify(payload.profile)
    : null;
  state.preprocessSteps = payload.steps || [];
  state.preprocessArtifacts = payload.artifacts || {};
  state.sourceToProcessed = (payload.source_to_processed || IDENTITY_MATRIX).map((row) => row.map(Number));
  state.processedToSource = (payload.processed_to_source || IDENTITY_MATRIX).map((row) => row.map(Number));
  state.sourceWidth = Number(payload.source_width || state.sourceWidth);
  state.sourceHeight = Number(payload.source_height || state.sourceHeight);
  syncControlsFromProfile();
  renderOperationStack();
  renderArtifactButtons();
}

async function applyPreprocessStack(showBusy = true) {
  if (!state.imageId || !state.preprocessProfile) return null;
  syncProfileFromControls();
  if (showBusy) setBusy(true, "Non-destructive ön işleme yığını uygulanıyor…");
  try {
    for (const circle of state.circles) {
      if (circle.source_x === null || circle.source_x === undefined) updateCircleSourceCoordinates(circle);
    }
    const payload = await api("/api/preprocess/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        profile: state.preprocessProfile,
      }),
    });
    state.preprocessProfile = payload.profile;
    state.appliedPreprocessSignature = JSON.stringify(payload.profile);
    state.preprocessSteps = payload.steps || [];
    state.preprocessArtifacts = payload.artifacts || {};
    state.preprocessApplied = true;
    await useImageRecord(payload, { reset: false });
    syncControlsFromProfile();
    renderOperationStack();
    renderArtifactButtons();
    invalidatePerspectiveQuality();
    $("#transformState").textContent = operationByType("perspective")?.enabled ? "homografi" : "özdeşlik";
    setStatus("Ön işleme kaynak görüntü değiştirilmeden uygulandı.", "success");
    return payload;
  } catch (error) {
    setStatus(error.message, "error");
    throw error;
  } finally {
    if (showBusy) setBusy(false);
  }
}

async function ensurePreprocessCurrent(showBusy = false) {
  if (!state.imageId || !state.preprocessProfile) return null;
  syncProfileFromControls();
  const currentSignature = JSON.stringify(state.preprocessProfile);
  if (
    state.preprocessApplied &&
    state.appliedPreprocessSignature === currentSignature
  ) {
    return null;
  }
  return applyPreprocessStack(showBusy);
}

canvas.addEventListener("pointerdown", (event) => {
  if (!state.imageElement) return;
  event.preventDefault();
  canvas.setPointerCapture(event.pointerId);
  const screen = pointerPosition(event);
  const imagePoint = screenToImage(screen);
  const temporaryPan = state.spaceDown || event.button === 1 || event.button === 2;
  if (state.tool === "pan" || temporaryPan) {
    state.pointer = {
      kind: "pan",
      startScreen: screen,
      startView: { ...state.view },
    };
    canvas.style.cursor = "grabbing";
    return;
  }
  if (!insideImage(imagePoint)) return;

  if (state.tool === "tramlineAdd") {
    state.pointer = {
      kind: "tramlineAdd",
      startImage: clampImagePoint(imagePoint),
      currentImage: clampImagePoint(imagePoint),
    };
    render();
    return;
  }
  if (state.tool === "tramlineEdit" || state.tool === "tramlineDelete") {
    const tramlineHit = hitTestTramline(imagePoint);
    if (!tramlineHit) {
      state.selectedTramlineId = null;
      updateTramlineUi();
      render();
      return;
    }
    state.selectedTramlineId = tramlineHit.line.id;
    if (state.tool === "tramlineDelete") {
      pushTramlineHistory();
      state.tramlineSuggestion.lines = state.tramlineSuggestion.lines.filter(
        (line) => line.id !== tramlineHit.line.id,
      );
      state.selectedTramlineId = null;
      recomputeTramlineSuggestion();
      setStatus("Tarla izi öneriden silindi; onaylanmış maske henüz değiştirilmedi.", "success");
      return;
    }
    state.pointer = {
      kind: tramlineHit.mode === "endpoint"
        ? "tramlineEndpoint"
        : "tramlineMove",
      endpoint: tramlineHit.endpoint || null,
      lineId: tramlineHit.line.id,
      startImage: clampImagePoint(imagePoint),
      startLine: deepClone(tramlineHit.line),
      historyPushed: false,
      moved: false,
    };
    updateTramlineUi();
    render();
    return;
  }
  if (state.tool === "roiPolygon") {
    if (!state.preprocessProfile) return;
    pushPreprocessHistory();
    state.preprocessProfile.analysis_polygon.push(
      strokePoint(processedToSourceCoordinate(imagePoint)),
    );
    updateUi();
    return;
  }
  if (["maskPaint", "maskErase", "samplePaint"].includes(state.tool)) {
    if (!state.preprocessProfile) return;
    pushPreprocessHistory();
    const sourcePoint = processedToSourceCoordinate(imagePoint);
    state.pointer = {
      kind: "brush",
      tool: state.tool,
      sampleClass: $("#sampleClass").value,
      stroke: {
        points: [strokePoint(sourcePoint)],
        radius: sourceBrushRadius(imagePoint),
        mode: state.tool === "maskErase" ? "erase" : "paint",
      },
    };
    render();
    return;
  }
  if (state.tool === "sizeSample") {
    state.pointer = {
      kind: "sizeSample",
      startImage: imagePoint,
      sourceCenter: processedToSourceCoordinate(imagePoint),
      sizeClass: $("#sizeSampleClass").value,
    };
    state.previewCircle = { x: imagePoint.x, y: imagePoint.y, radius: 1 };
    render();
    return;
  }
  if (state.tool === "add") {
    state.pointer = { kind: "add", startImage: imagePoint };
    state.previewCircle = { x: imagePoint.x, y: imagePoint.y, radius: 1 };
    render();
    return;
  }
  if (state.tool === "center") {
    pushStructureUndo("set_general_center");
    const previousCenter = state.center ? { ...state.center } : null;
    state.center = imagePoint;
    state.centerSource = processedToSourceCoordinate(imagePoint);
    recordStructuralEdit(
      null,
      "general_center",
      previousCenter,
      state.center,
      "user",
      "set_general_center",
    );
    updateAllPolar();
    clearSymmetry();
    updateUi();
    setStatus("Genel merkez seçildi. Simetriyi hesaplayabilirsiniz.", "success");
    setTool("select");
    void refreshStructureAnalysis(false);
    return;
  }
  if (state.tool === "perspective") {
    if (state.perspectivePoints.length >= 4) state.perspectivePoints = [];
    state.perspectivePoints.push(
      processedToSourceCoordinate(imagePoint),
    );
    updatePerspectiveUi();
    render();
    if (state.perspectivePoints.length === 4) {
      setStatus("Dört perspektif noktası hazır.", "success");
    }
    return;
  }
  const hit = hitTest(imagePoint);
  if (state.tool === "mainChain") {
    if (hit) addMainChainDraftCircle(hit.circle.id);
    return;
  }
  if (state.tool === "delete") {
    if (hit) removeCircle(hit.circle.id);
    return;
  }
  if (state.tool === "select") {
    if (!hit) {
      selectCircle(null);
      return;
    }
    selectCircle(hit.circle.id);
    state.pointer = {
      kind: hit.mode,
      startImage: imagePoint,
      startCircle: { ...hit.circle },
      circleId: hit.circle.id,
    };
  }
});

canvas.addEventListener("pointermove", (event) => {
  const screen = pointerPosition(event);
  const imagePoint = screenToImage(screen);
  $("#cursorReadout").textContent = state.imageElement && insideImage(imagePoint)
    ? `x ${imagePoint.x.toFixed(1)} · y ${imagePoint.y.toFixed(1)}`
    : "x — · y —";
  if (!state.pointer) return;
  event.preventDefault();
  if (state.pointer.kind === "pan") {
    state.view.tx = state.pointer.startView.tx + screen.x - state.pointer.startScreen.x;
    state.view.ty = state.pointer.startView.ty + screen.y - state.pointer.startScreen.y;
    render();
    return;
  }
  if (state.pointer.kind === "tramlineAdd") {
    state.pointer.currentImage = clampImagePoint(imagePoint);
    render();
    return;
  }
  if (
    state.pointer.kind === "tramlineMove"
    || state.pointer.kind === "tramlineEndpoint"
  ) {
    const line = state.tramlineSuggestion?.lines?.find(
      (item) => item.id === state.pointer.lineId,
    );
    if (!line) return;
    if (!state.pointer.historyPushed) {
      pushTramlineHistory();
      state.pointer.historyPushed = true;
    }
    const current = clampImagePoint(imagePoint);
    if (state.pointer.kind === "tramlineEndpoint") {
      line[state.pointer.endpoint] = current;
    } else {
      let dx = current.x - state.pointer.startImage.x;
      let dy = current.y - state.pointer.startImage.y;
      const onBoundary = (point) =>
        point.x <= 1
        || point.y <= 1
        || point.x >= state.imageWidth - 2
        || point.y >= state.imageHeight - 2;
      const spansImage = (
        onBoundary(state.pointer.startLine.p1)
        && onBoundary(state.pointer.startLine.p2)
      );
      if (spansImage) {
        const angle = tramlineAngle(state.pointer.startLine) * Math.PI / 180;
        const normal = { x: -Math.sin(angle), y: Math.cos(angle) };
        const normalShift = dx * normal.x + dy * normal.y;
        const midpoint = {
          x: (
            state.pointer.startLine.p1.x
            + state.pointer.startLine.p2.x
          ) / 2 + normalShift * normal.x,
          y: (
            state.pointer.startLine.p1.y
            + state.pointer.startLine.p2.y
          ) / 2 + normalShift * normal.y,
        };
        const extended = extendLineToImageBounds(
          midpoint,
          tramlineAngle(state.pointer.startLine),
        );
        if (extended) {
          line.p1 = extended.p1;
          line.p2 = extended.p2;
        }
      } else {
        dx = Math.max(
          -Math.min(state.pointer.startLine.p1.x, state.pointer.startLine.p2.x),
          Math.min(
            state.imageWidth - 1 - Math.max(
              state.pointer.startLine.p1.x,
              state.pointer.startLine.p2.x,
            ),
            dx,
          ),
        );
        dy = Math.max(
          -Math.min(state.pointer.startLine.p1.y, state.pointer.startLine.p2.y),
          Math.min(
            state.imageHeight - 1 - Math.max(
              state.pointer.startLine.p1.y,
              state.pointer.startLine.p2.y,
            ),
            dy,
          ),
        );
        line.p1 = {
          x: state.pointer.startLine.p1.x + dx,
          y: state.pointer.startLine.p1.y + dy,
        };
        line.p2 = {
          x: state.pointer.startLine.p2.x + dx,
          y: state.pointer.startLine.p2.y + dy,
        };
      }
    }
    line.manual = true;
    line.support = 1;
    state.pointer.moved = true;
    recomputeTramlineSuggestion();
    return;
  }
  if (state.pointer.kind === "brush") {
    const sourcePoint = processedToSourceCoordinate(imagePoint);
    const points = state.pointer.stroke.points;
    const previous = points[points.length - 1];
    if (
      !previous ||
      Math.hypot(sourcePoint.x - previous.x, sourcePoint.y - previous.y) >=
        Math.max(0.7, state.pointer.stroke.radius * 0.18)
    ) {
      points.push(strokePoint(sourcePoint));
      render();
    }
    return;
  }
  if (state.pointer.kind === "sizeSample") {
    state.previewCircle.radius = Math.max(
      1,
      Math.hypot(imagePoint.x - state.pointer.startImage.x, imagePoint.y - state.pointer.startImage.y),
    );
    render();
    return;
  }
  if (state.pointer.kind === "add") {
    state.previewCircle.radius = Math.max(
      1,
      Math.hypot(imagePoint.x - state.pointer.startImage.x, imagePoint.y - state.pointer.startImage.y),
    );
    render();
    return;
  }
  const circle = state.circles.find((item) => item.id === state.pointer.circleId);
  if (!circle) return;
  if (state.pointer.kind === "move") {
    circle.x = state.pointer.startCircle.x + imagePoint.x - state.pointer.startImage.x;
    circle.y = state.pointer.startCircle.y + imagePoint.y - state.pointer.startImage.y;
  } else if (state.pointer.kind === "resize") {
    circle.radius = Math.max(0.5, Math.hypot(imagePoint.x - circle.x, imagePoint.y - circle.y));
  }
  markManual(circle);
  clearSymmetry();
  updateInspector();
  render();
});

function finishPointer(event) {
  if (!state.pointer) return;
  const pointer = state.pointer;
  state.pointer = null;
  if (pointer.kind === "tramlineAdd") {
    const length = Math.hypot(
      pointer.currentImage.x - pointer.startImage.x,
      pointer.currentImage.y - pointer.startImage.y,
    );
    if (length >= 12 / state.view.scale) {
      pushTramlineHistory();
      if (!state.tramlineSuggestion) {
        state.tramlineSuggestion = normalizeTramlineSuggestion({
          lines: [],
          requires_user_approval: true,
        });
      }
      state.tramlineCounter += 1;
      const line = {
        id: `tramline-manual-${Date.now().toString(36)}-${state.tramlineCounter}`,
        p1: { ...pointer.startImage },
        p2: { ...pointer.currentImage },
        source_p1: null,
        source_p2: null,
        support: 1,
        manual: true,
      };
      syncTramlineSourceGeometry(line);
      state.tramlineSuggestion.lines.push(line);
      state.selectedTramlineId = line.id;
      recomputeTramlineSuggestion();
      setTool("tramlineEdit");
      setStatus("Yeni tarla izi öneriye eklendi; uygulamak için ayrıca onay vermelisiniz.", "success");
    } else {
      render();
    }
  } else if (
    pointer.kind === "tramlineMove"
    || pointer.kind === "tramlineEndpoint"
  ) {
    if (pointer.moved) {
      recomputeTramlineSuggestion();
      setStatus("Tarla izi düzeltildi; onaylanmış maske henüz değiştirilmedi.", "success");
    } else {
      updateTramlineUi();
    }
  } else if (pointer.kind === "brush" && state.preprocessProfile) {
    if (pointer.tool === "samplePaint") {
      state.preprocessProfile.color_samples[pointer.sampleClass].push(pointer.stroke);
    } else {
      state.preprocessProfile.exclusion_strokes.push(pointer.stroke);
    }
    updateUi();
    setStatus("Ön işleme fırça darbesi profile eklendi.", "success");
  } else if (pointer.kind === "sizeSample" && state.previewCircle && state.preprocessProfile) {
    pushPreprocessHistory();
    const settings = state.preprocessProfile.size_classes[pointer.sizeClass];
    settings.samples.push({
      source_center: {
        x: pointer.sourceCenter.x,
        y: pointer.sourceCenter.y,
      },
      source_x: pointer.sourceCenter.x,
      source_y: pointer.sourceCenter.y,
      radius: processedRadiusToSource(
        pointer.startImage,
        Math.max(1, state.previewCircle.radius),
      ),
    });
    state.previewCircle = null;
    updateUi();
    setStatus(`${pointer.sizeClass} ortak yan-daire standardına örnek eklendi.`, "success");
  } else if (pointer.kind === "add" && state.previewCircle) {
    const preview = state.previewCircle;
    state.previewCircle = null;
    addManualCircle(preview.x, preview.y, Math.max(preview.radius, 6 / state.view.scale));
  } else if (pointer.kind === "move" || pointer.kind === "resize") {
    updateUi();
    setStatus("Manuel geometri düzeltmesi kaydedildi.", "success");
  }
  canvas.style.cursor =
    state.tool === "pan"
      ? "grab"
      : ["tramlineAdd", "perspective", "center"].includes(state.tool)
        ? "crosshair"
        : "";
  if (event && canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  render();
}

canvas.addEventListener("pointerup", finishPointer);
canvas.addEventListener("pointercancel", finishPointer);
canvas.addEventListener("contextmenu", (event) => event.preventDefault());

canvas.addEventListener("wheel", (event) => {
  if (!state.imageElement) return;
  event.preventDefault();
  const screen = pointerPosition(event);
  const anchor = screenToImage(screen);
  const factor = Math.exp(-event.deltaY * 0.0012);
  const newScale = Math.max(0.02, Math.min(24, state.view.scale * factor));
  state.view.scale = newScale;
  state.view.tx = screen.x - anchor.x * newScale;
  state.view.ty = screen.y - anchor.y * newScale;
  render();
}, { passive: false });

$("#fileInput").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const form = new FormData();
  form.append("image", file);
  setBusy(true, "Görüntü yükleniyor…");
  try {
    const record = await api("/api/upload", { method: "POST", body: form });
    await useImageRecord(record);
    await initializePreprocessState();
    setWorkspace("preprocess");
    setStatus(`${record.name} açıldı.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
    event.target.value = "";
  }
});

$("#sampleButton")?.addEventListener("click", async () => {
  setBusy(true, "Milk Hill örneği açılıyor…");
  try {
    const record = await api("/api/sample", { method: "POST" });
    await useImageRecord(record);
    await initializePreprocessState();
    setWorkspace("preprocess");
    setStatus("Milk Hill Galaxy Spiral 2001 açıldı.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

$("#preprocessSampleButton")?.addEventListener("click", async () => {
  setBusy(true, "Sentetik ön işleme örneği açılıyor…");
  try {
    const record = await api("/api/sample/preprocessing", { method: "POST" });
    await useImageRecord(record);
    await initializePreprocessState();
    setWorkspace("preprocess");
    setStatus("Eğik perspektif, değişken aydınlatma, üç boyut ve tramline içeren sentetik örnek açıldı.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

$("#perspectiveMode").addEventListener("change", (event) => {
  if (state.preprocessProfile) {
    pushPreprocessHistory();
    const perspective = operationByType("perspective");
    if (perspective) perspective.enabled = false;
  }
  state.perspectivePoints = [];
  $("#perspectiveHint").textContent = event.target.value === "field_tracks"
    ? "Paralel tarla izleri güçlü referans sağlar; iki izin uçlarını A1, A2, B1, B2 sırasıyla seçin. Tek doğrultudaki paralellik tam metrik düzeltme garantisi vermez."
    : "Görüntü köşelerini değil, zemindeki aynı düzleme ait bir dikdörtgen/paralelkenarı temsil eden dört anlamlı referans noktasını seçin.";
  invalidatePerspectiveQuality();
  updatePerspectiveUi();
  render();
});

$("#clearPerspective").addEventListener("click", () => {
  if (state.preprocessProfile) {
    pushPreprocessHistory();
    const perspective = operationByType("perspective");
    if (perspective) {
      perspective.enabled = false;
      perspective.params.points = [];
    }
  }
  state.perspectivePoints = [];
  updatePerspectiveUi();
  render();
});

$("#applyPerspective").addEventListener("click", async () => {
  if (!state.imageId || state.perspectivePoints.length !== 4 || !state.preprocessProfile) return;
  pushPreprocessHistory();
  const perspective = operationByType("perspective");
  perspective.enabled = true;
  perspective.params = {
    mode: $("#perspectiveMode").value,
    points: state.perspectivePoints.map(strokePoint),
    preserve_full_image: $("#preservePerspectiveFrame").checked,
  };
  try {
    await applyPreprocessStack();
    setStatus("Manuel perspektif işlem yığınına eklendi; kaynak görüntü korundu.", "success");
    setTool("pan");
  } catch (error) {
    setStatus(error.message, "error");
  }
});

$("#calculatePerspectiveQuality").addEventListener("click", () => {
  void calculatePerspectiveQuality();
});

$("#exportPerspectiveQuality").addEventListener("click", () => {
  void exportPerspectiveQualityReports();
});

$("#qualityViewBefore").addEventListener("click", () => {
  $("#qualityCoordinateSpace").value = "source";
  showPerspectiveQualityArtifact("source_view", "Perspektif kalitesi · source / önce");
});

$("#qualityViewAfter").addEventListener("click", () => {
  $("#qualityCoordinateSpace").value = "corrected";
  showPerspectiveQualityArtifact("corrected_view", "Perspektif kalitesi · corrected / sonra");
});

$("#qualityViewSideBySide").addEventListener("click", () => {
  showPerspectiveQualityArtifact("perspective_before_after", "Perspektif kalitesi · önce / sonra");
});

$("#qualityViewRoundness").addEventListener("click", () => {
  showPerspectiveQualityArtifact("roundness_overlay", "En büyük ana çemberler · elips uyumu");
});

$("#qualityViewTramlines").addEventListener("click", () => {
  showPerspectiveQualityArtifact("tramline_overlay_corrected", "Düzeltilmiş tramline ailesi");
});

$("#qualityViewHeatmap").addEventListener("click", () => {
  showPerspectiveQualityArtifact("arm_scale_heatmap", "Karşılıklı kol ölçek farkı ısı haritası");
});

$("#qualityCoordinateSpace").addEventListener("change", (event) => {
  const source = event.target.value === "source";
  updateInspector();
  showPerspectiveQualityArtifact(
    source ? "source_view" : "corrected_view",
    source ? "Perspektif kalitesi · source / önce" : "Perspektif kalitesi · corrected / sonra",
  );
});

$("#qualityCircleScope").addEventListener("change", renderPerspectiveQuality);

$("#showQualityEllipses").addEventListener("change", (event) => {
  state.showQualityEllipses = event.target.checked;
  render();
});

$("#showQualityBenefits").addEventListener("change", (event) => {
  state.showQualityBenefits = event.target.checked;
  render();
});

$("#selectedCircleQuality").addEventListener("click", (event) => {
  const button = event.target.closest("[data-selected-quality-view]");
  if (!button) return;
  void showSelectedCircleQualityView(button.dataset.selectedQualityView);
});

for (const id of [
  "qualityWeightTramline",
  "qualityWeightRoundness",
  "qualityWeightArm",
  "qualityWeightK5K6",
  "qualityWeightWarp",
]) {
  $(`#${id}`).addEventListener("change", invalidatePerspectiveQuality);
}

$$(".workspace-tab").forEach((button) => {
  button.addEventListener("click", () => setWorkspace(button.dataset.workspace));
});

$("#brushRadius").addEventListener("input", (event) => {
  $("#brushRadiusReadout").textContent = `${event.target.value} px`;
});

$("#maskOpacity").addEventListener("input", (event) => {
  state.maskOpacity = Number(event.target.value) / 100;
  $("#maskOpacityReadout").textContent = `${event.target.value}%`;
  render();
});

$("#showExclusionMask").addEventListener("change", (event) => {
  state.showExclusionMask = event.target.checked;
  render();
});

$("#showRoi").addEventListener("change", (event) => {
  state.showRoi = event.target.checked;
  render();
});

$("#showTramlines").addEventListener("change", render);

$("#tramlineRegularity").addEventListener("input", (event) => {
  $("#tramlineRegularityReadout").textContent = `${event.target.value}%`;
});

$("#undoTramlineEdit").addEventListener("click", () => {
  if (!state.tramlineHistory.length) return;
  const previous = state.tramlineHistory.pop();
  state.tramlineSuggestion = previous
    ? normalizeTramlineSuggestion(previous)
    : null;
  state.selectedTramlineId = null;
  updateTramlineUi();
  render();
  setStatus("Son tarla izi değişikliği geri alındı.", "success");
});

$("#parallelizeTramlines").addEventListener("click", () => {
  const lines = state.tramlineSuggestion?.lines || [];
  if (lines.length < 2) return;
  pushTramlineHistory();
  const angle = Number(state.tramlineSuggestion.dominant_angle_degrees || 0);
  for (const line of lines) {
    const midpoint = {
      x: (line.p1.x + line.p2.x) / 2,
      y: (line.p1.y + line.p2.y) / 2,
    };
    const extended = extendLineToImageBounds(midpoint, angle);
    if (!extended) continue;
    line.p1 = extended.p1;
    line.p2 = extended.p2;
    line.manual = true;
    line.support = 1;
  }
  recomputeTramlineSuggestion();
  setStatus("Tüm izler ortak yöne getirildi; onaylanmış maske henüz değiştirilmedi.", "success");
});

$("#clearRoi").addEventListener("click", () => {
  if (!state.preprocessProfile) return;
  pushPreprocessHistory();
  state.preprocessProfile.analysis_polygon = [];
  updateUi();
});

$("#clearExclusionMask").addEventListener("click", () => {
  if (!state.preprocessProfile) return;
  pushPreprocessHistory();
  state.preprocessProfile.exclusion_strokes = [];
  updateUi();
});

$("#clearColorSamples").addEventListener("click", () => {
  if (!state.preprocessProfile) return;
  pushPreprocessHistory();
  for (const sampleClass of Object.keys(SAMPLE_COLORS)) {
    state.preprocessProfile.color_samples[sampleClass] = [];
  }
  updateUi();
});

$("#clearSizeSamples").addEventListener("click", () => {
  if (!state.preprocessProfile) return;
  pushPreprocessHistory();
  for (const sizeClass of ["small", "medium", "large"]) {
    state.preprocessProfile.size_classes[sizeClass].samples = [];
  }
  updateUi();
});

$("#undoPreprocess").addEventListener("click", async () => {
  const previous = state.preprocessHistory.pop();
  if (!previous) {
    setStatus("Geri alınacak ön işleme değişikliği yok.", "error");
    return;
  }
  state.preprocessProfile = previous;
  syncControlsFromProfile();
  try {
    await applyPreprocessStack();
    setStatus("Son ön işleme değişikliği geri alındı.", "success");
  } catch (_error) {
    // applyPreprocessStack already reports the actionable message.
  }
});

$("#resetPreprocess").addEventListener("click", async () => {
  if (!state.defaultPreprocessProfile) return;
  pushPreprocessHistory();
  state.preprocessProfile = deepClone(state.defaultPreprocessProfile);
  state.perspectivePoints = [];
  state.tramlineSuggestion = null;
  state.selectedTramlineId = null;
  state.tramlineHistory = [];
  syncControlsFromProfile();
  try {
    await applyPreprocessStack();
    setStatus("Ön işleme profili varsayılan, etkisiz yığına döndürüldü.", "success");
  } catch (_error) {
    // Status is set centrally.
  }
});

$("#applyPreprocessButton").addEventListener("click", async () => {
  pushPreprocessHistory();
  try {
    await applyPreprocessStack();
  } catch (_error) {
    // Status is set centrally.
  }
});

$("#suggestTramlines").addEventListener("click", async () => {
  if (!state.imageId) {
    setStatus("Önce bir görüntü açın.", "error");
    return;
  }
  setBusy(true, "Uzun ve paralel tarla izleri aranıyor…");
  try {
    await ensurePreprocessCurrent(false);
    const suggestion = await api("/api/preprocess/tramlines/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        min_length_ratio: Number($("#tramlineMinLength").value || 0.28),
        angle_tolerance_degrees: Number($("#tramlineAngleTolerance").value || 9),
        expected_spacing_pixels: Math.max(
          0,
          Number($("#tramlineExpectedSpacing").value || 0),
        ),
        regularity: Math.max(
          0,
          Math.min(1, Number($("#tramlineRegularity").value || 72) / 100),
        ),
      }),
    });
    state.tramlineSuggestion = normalizeTramlineSuggestion(suggestion);
    state.selectedTramlineId = null;
    state.tramlineHistory = [];
    const count = suggestion.lines?.length || 0;
    updateTramlineUi();
    render();
    setStatus(
      count
        ? "Düzenli tarla izi ailesi hazır; çizgileri onaydan önce elle düzeltebilirsiniz."
        : "Güvenilir ve düzenli bir tarla izi ailesi bulunamadı; aralığı girebilir veya elle iz çizebilirsiniz.",
      count ? "success" : "warning",
    );
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

$("#approvePerspectiveSuggestion").addEventListener("click", async () => {
  const points = state.tramlineSuggestion?.perspective_points_source;
  if (!state.preprocessProfile || points?.length !== 4) return;
  pushPreprocessHistory();
  state.perspectivePoints = points.map((point) => ({ x: Number(point.x), y: Number(point.y) }));
  const perspective = operationByType("perspective");
  perspective.enabled = true;
  perspective.params = {
    mode: "field_tracks",
    points: state.perspectivePoints.map(strokePoint),
    preserve_full_image: $("#preservePerspectiveFrame").checked,
  };
  $("#perspectiveMode").value = "field_tracks";
  try {
    await applyPreprocessStack();
    setStatus("Tarla izlerinden önerilen perspektif kullanıcı onayıyla uygulandı.", "success");
  } catch (_error) {
    // Status is set centrally.
  }
});

$("#acceptTramlineMask").addEventListener("click", async () => {
  const lines = state.tramlineSuggestion?.lines;
  if (!state.preprocessProfile || !lines?.length) return;
  pushPreprocessHistory();
  state.preprocessProfile.tramline.lines = lines
    .filter((line) => line.source_p1 && line.source_p2)
    .map((line) => ({ p1: line.source_p1, p2: line.source_p2 }));
  state.preprocessProfile.tramline.enabled = true;
  $("#tramlineMaskEnabled").checked = true;
  try {
    await applyPreprocessStack();
    setStatus("Tarla izleri yalnızca dışlama maskesine eklendi; görüntü pikselleri değiştirilmedi.", "success");
  } catch (_error) {
    // Status is set centrally.
  }
});

$("#saveProfile").addEventListener("click", async () => {
  if (!state.imageId || !state.preprocessProfile) return;
  syncProfileFromControls();
  try {
    const payload = await api("/api/preprocess/profile/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.imageId, profile: state.preprocessProfile }),
    });
    $("#profileDownload").innerHTML = `<a href="${payload.url}" download="${escapeHtml(payload.name)}">JSON profili</a>`;
    $("#profileDownload").classList.remove("hidden");
    setStatus("preprocess_profile.json kaydedildi.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
});

$("#profileInput").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file || !state.imageId) return;
  const form = new FormData();
  form.append("image_id", state.imageId);
  form.append("profile", file);
  setBusy(true, "Ön işleme profili doğrulanıyor…");
  try {
    const payload = await api("/api/preprocess/profile/load", { method: "POST", body: form });
    pushPreprocessHistory();
    state.preprocessProfile = payload.profile;
    state.tramlineSuggestion = null;
    state.selectedTramlineId = null;
    state.tramlineHistory = [];
    syncControlsFromProfile();
    await ensurePreprocessCurrent(false);
    setStatus("preprocess_profile.json yüklendi ve non-destructive olarak uygulandı.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
    event.target.value = "";
  }
});

$("#exportPreprocessButton").addEventListener("click", async () => {
  if (!state.imageId) return;
  setBusy(true, "Beş ön işleme ara çıktısı hazırlanıyor…");
  try {
    await ensurePreprocessCurrent(false);
    const payload = await api("/api/preprocess/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        circles: state.circles,
        center: state.center,
        phase_degrees: state.phaseDegrees,
        ...armSettingsPayload(),
      }),
    });
    $("#preprocessDownloads").innerHTML = Object.values(payload.files).map(
      (item) => `<a href="${item.url}" download="${escapeHtml(item.name)}">${escapeHtml(item.name.replace(".png", ""))}</a>`,
    ).join("");
    $("#preprocessDownloads").classList.remove("hidden");
    setStatus("Ara çıktılar tam dosya adlarıyla outputs/ altında hazırlandı.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

$("#closeStepPreview").addEventListener("click", () => {
  closeStepPreview();
});

function automaticOverlapsManual(candidate, manualCircles) {
  return manualCircles.some((manual) => {
    const centerDistance = Math.hypot(candidate.x - manual.x, candidate.y - manual.y);
    const radiusDifference = Math.abs(candidate.radius - manual.radius) / Math.max(candidate.radius, manual.radius, 1);
    return centerDistance < Math.max(3, 0.35 * Math.min(candidate.radius, manual.radius)) && radiusDifference < 0.35;
  });
}

function manualLocksPayload() {
  return Object.fromEntries(
    state.circles
      .filter((circle) => (circle._locked_fields || []).length)
      .map((circle) => {
        circle.locked_fields = [...circle._locked_fields];
        return [circle.id, circle._locked_fields];
      }),
  );
}

function editorStatePayload() {
  return {
    deleted_automatic_ids: [...state.deletedAutoIds],
    suppressed_automatic_geometry: state.suppressedAutomaticGeometry.map(
      (geometry) => ({ ...geometry }),
    ),
  };
}

function armSettingsPayload() {
  return {
    arm_model: state.armModel,
    arm_twist_degrees_per_100px: state.armTwistDegreesPer100px,
  };
}

function projectRequestPayload() {
  return {
    image_id: state.imageId,
    circles: state.circles,
    manual_locks: manualLocksPayload(),
    editor_state: editorStatePayload(),
    edit_history: state.editHistory,
    structure_state: {
      main_chains: state.mainChains,
      main_chain_draft: state.mainChainDraft
        ? {
          arm: state.mainChainDraft.arm,
          ids: [...state.mainChainDraft.ids],
        }
        : null,
    },
    center: state.center,
    phase_degrees: state.phaseDegrees,
    ...armSettingsPayload(),
  };
}

function importedSourceRadius(rawCircle, exportedInverse) {
  const processedCenter = {
    x: Number(rawCircle.processed_x ?? rawCircle.x),
    y: Number(rawCircle.processed_y ?? rawCircle.y),
  };
  const processedEdge = {
    x: processedCenter.x + Number(rawCircle.radius),
    y: processedCenter.y,
  };
  const sourceCenter = transformCoordinate(processedCenter, exportedInverse);
  const sourceEdge = transformCoordinate(processedEdge, exportedInverse);
  return Math.max(
    0.1,
    Math.hypot(
      sourceEdge.x - sourceCenter.x,
      sourceEdge.y - sourceCenter.y,
    ),
  );
}

function restoreProjectDocument(document) {
  const exportedInverse = (
    document.coordinate_transform?.processed_to_source || IDENTITY_MATRIX
  ).map((row) => row.map(Number));
  state.circles = (document.circles || []).map((rawCircle) => {
    const sourceCenter = {
      x: Number(rawCircle.source_x ?? transformCoordinate(rawCircle, exportedInverse).x),
      y: Number(rawCircle.source_y ?? transformCoordinate(rawCircle, exportedInverse).y),
    };
    const processedCenter = sourceToProcessedCoordinate(sourceCenter);
    const sourceRadius = Number(
      rawCircle.source_radius
      ?? importedSourceRadius(rawCircle, exportedInverse),
    );
    const processedRadius = sourceRadiusToProcessed(sourceCenter, sourceRadius);
    return normalizeCircle({
      ...rawCircle,
      x: processedCenter.x,
      y: processedCenter.y,
      radius: processedRadius,
      processed_x: processedCenter.x,
      processed_y: processedCenter.y,
      source_x: sourceCenter.x,
      source_y: sourceCenter.y,
      source_radius: sourceRadius,
      corrected_x: processedCenter.x,
      corrected_y: processedCenter.y,
      corrected_radius: processedRadius,
    });
  });
  state.preserveLoadedCircleSet = true;
  const manualLocks = document.manual_locks || {};
  for (const circle of state.circles) {
    circle._locked_fields = [
      ...(circle.locked_fields || []),
      ...(Array.isArray(manualLocks[circle.id]) ? manualLocks[circle.id] : []),
    ].filter((field, index, values) => values.indexOf(field) === index);
    circle.locked_fields = [...circle._locked_fields];
  }
  const editorState = document.editor_state || {};
  state.deletedAutoIds = new Set(editorState.deleted_automatic_ids || []);
  state.suppressedAutomaticGeometry = (
    editorState.suppressed_automatic_geometry || []
  ).map((geometry) => {
    const sourceCenter = transformCoordinate(geometry, exportedInverse);
    const processedCenter = sourceToProcessedCoordinate(sourceCenter);
    const sourceRadius = importedSourceRadius(geometry, exportedInverse);
    return {
      id: geometry.id || "",
      x: processedCenter.x,
      y: processedCenter.y,
      radius: sourceRadiusToProcessed(sourceCenter, sourceRadius),
    };
  });
  state.selectedId = null;
  state.editHistory = Array.isArray(document.edit_history)
    ? document.edit_history.map((item) => ({ ...item }))
    : [];
  const rawChains = document.structure_state?.main_chains || {};
  state.mainChains = { 1: [], 2: [], 3: [], 4: [], 5: [], 6: [] };
  for (let arm = 1; arm <= 6; arm += 1) {
    const rawIds = rawChains[String(arm)] || rawChains[arm] || [];
    state.mainChains[arm] = Array.isArray(rawIds)
      ? rawIds.map(String).filter((id) => state.circles.some((circle) => circle.id === id)).slice(0, 13)
      : [];
  }
  const rawDraft = document.structure_state?.main_chain_draft;
  const draftArm = Number(rawDraft?.arm);
  const draftIds = Array.isArray(rawDraft?.ids)
    ? rawDraft.ids
      .map(String)
      .filter(
        (id, index, values) =>
          values.indexOf(id) === index &&
          state.circles.some((circle) => circle.id === id),
      )
      .slice(0, 13)
    : [];
  state.mainChainDraft =
    Number.isInteger(draftArm) &&
    draftArm >= 1 &&
    draftArm <= 6 &&
    draftIds.length
      ? { arm: draftArm, ids: draftIds }
      : null;
  state.structureUndo = [];
  state.structureReport = null;
  state.boundarySelection.clear();
  state.localSideSelection.clear();
  state.satelliteSelection.clear();
  state.center = null;
  state.centerSource = null;
  if (document.general_center) {
    const exportedCenter = {
      x: Number(document.general_center.x),
      y: Number(document.general_center.y),
    };
    state.centerSource = transformCoordinate(exportedCenter, exportedInverse);
    state.center = sourceToProcessedCoordinate(state.centerSource);
  }
  state.phaseDegrees = Number(document.arm_phase_degrees || 0);
  state.armModel = document.arm_model === "spiral" ? "spiral" : "radial";
  state.armTwistDegreesPer100px = Number(
    document.arm_twist_degrees_per_100px || 0,
  );
  $("#phaseDegrees").value = String(state.phaseDegrees);
  clearArmSuggestion();
  clearSymmetry();
  setWorkspace("measure");
  if (state.mainChainDraft) {
    $("#mainChainArm").value = String(state.mainChainDraft.arm);
    setTool("mainChain");
    setStatus(
      `K${state.mainChainDraft.arm} Ana Zincir taslağı ` +
      `${state.mainChainDraft.ids.length}/13 seçimle geri yüklendi.`,
      "success",
    );
  }
  updateUi();
}

function replaceAnalyzedCircles(rawCircles) {
  const metadata = new Map(
    state.circles.map((circle) => [
      circle.id,
      {
        locked: [...(circle._locked_fields || [])],
        origin: circle._automatic_origin ? { ...circle._automatic_origin } : null,
      },
    ]),
  );
  state.circles = rawCircles.map(normalizeCircle);
  for (const circle of state.circles) {
    const previous = metadata.get(circle.id);
    if (!previous) continue;
    circle._locked_fields = previous.locked;
    circle.locked_fields = [...previous.locked];
    if (previous.origin) circle._automatic_origin = previous.origin;
  }
}

$("#detectButton").addEventListener("click", async () => {
  if (!state.imageId) {
    setStatus("Önce bir görüntü açın.", "error");
    return;
  }
  if (state.preserveLoadedCircleSet) {
    setStatus(
      "Yüklenen çalışma kümesi korunuyor; mevcut çemberler yeniden algılanmayacak.",
      "warning",
    );
    return;
  }
  closeStepPreview();
  setBusy(true, "Ön işleme ve çok ölçekli algılama çalışıyor…");
  try {
    await ensurePreprocessCurrent(false);
    const payload = await api("/api/detect/advanced", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        circles: state.circles,
        max_candidates: Number($("#maxCandidates").value || 250),
      }),
    });
    const radiusRanges = payload.radius_ranges || {};
    const formatRange = (name) => {
      const range = radiusRanges[name];
      return range
        ? `${Number(range.min_radius).toFixed(1)}–${Number(range.max_radius).toFixed(1)}`
        : "—";
    };
    const mainRange = radiusRanges.main_variable;
    $("#detectionRangeReadout").textContent =
      `Etkin işlenmiş px · yan küçük ${formatRange("small")}, ` +
      `yan orta ${formatRange("medium")}, yan büyük ${formatRange("large")}; ` +
      `ana değişken ${mainRange?.enabled === false ? "kapalı" : formatRange("main_variable")}.`;
    const previousMetadata = new Map(
      state.circles.map((circle) => [
        circle.id,
        {
          locks: [...(circle._locked_fields || [])],
          origin: circle._automatic_origin ? { ...circle._automatic_origin } : null,
        },
      ]),
    );
    const protectedCircles = state.circles.filter(
      (circle) => circle.source !== "automatic" && !circle.locked,
    );
    const protectedIds = new Set(protectedCircles.map((circle) => circle.id));
    const returned = payload.circles.map(normalizeCircle);
    for (const circle of returned) {
      const metadata = previousMetadata.get(circle.id);
      if (!metadata) continue;
      circle._locked_fields = metadata.locks;
      circle.locked_fields = [...metadata.locks];
      if (metadata.origin) circle._automatic_origin = metadata.origin;
    }
    const lockedReturned = returned.filter((circle) => circle.locked);
    const automatic = returned
      .filter((circle) => !circle.locked)
      .filter((circle) => !state.deletedAutoIds.has(circle.id))
      .filter((circle) => !automaticOverlapsManual(circle, state.suppressedAutomaticGeometry))
      .filter((circle) => !protectedIds.has(circle.id))
      .filter((circle) => !automaticOverlapsManual(circle, protectedCircles));
    state.circles = [...protectedCircles, ...lockedReturned, ...automatic];
    if (!selectedCircle()) state.selectedId = null;
    clearSymmetry();
    updateUi();
    const mainVariableCount = Number(
      payload.detection_counts?.main_variable || 0,
    );
    const sideStandardCount = Number(
      payload.detection_counts?.side_standard || 0,
    );
    setStatus(
      `${automatic.length} yeni aday (${mainVariableCount} ana-değişken, ${sideStandardCount} yan-standardı) ` +
      `${Number(payload.elapsed_seconds || 0).toFixed(1)} saniyede birleştirildi; ` +
      `${lockedReturned.length} kilitli ve ${protectedCircles.length} manuel daire korundu.`,
      "success",
    );
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

$("#inspectWeakCircle").addEventListener("click", async () => {
  if (!state.imageId || !state.circles.length) {
    setStatus("İncelemek için daire bulunmuyor.", "error");
    return;
  }
  setBusy(true, "Dairelerin yerel görüntü desteği karşılaştırılıyor…");
  try {
    const payload = await api("/api/circles/suspects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        circles: state.circles,
        limit: 5,
      }),
    });
    const suggestion = payload.suggestions?.[0];
    const circle = state.circles.find(
      (item) => item.id === suggestion?.id,
    );
    if (!suggestion || !circle) {
      throw new Error("İncelenecek aday bulunamadı.");
    }
    focusCircle(circle);
    $("#weakCircleReadout").textContent =
      `${circle.id} · x ${circle.x.toFixed(1)}, y ${circle.y.toFixed(1)}, ` +
      `r ${circle.radius.toFixed(1)} · destek ${Number(suggestion.score).toFixed(3)} · ` +
      `kontrast ${Number(suggestion.contrast_score).toFixed(3)}`;
    setStatus(
      `${circle.id} yalnızca inceleme için seçildi ve yakınlaştırıldı; otomatik silinmedi.`,
      "success",
    );
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

$("#phaseDegrees").addEventListener("input", (event) => {
  const value = Number(event.target.value);
  state.phaseDegrees = Number.isFinite(value) ? value : 0;
  clearArmSuggestion();
  clearSymmetry();
  render();
});

$("#armModel").addEventListener("change", (event) => {
  state.armModel = event.target.value === "spiral" ? "spiral" : "radial";
  clearArmSuggestion();
  clearSymmetry();
  updateArmModelUi();
  render();
});

$("#armTwist").addEventListener("input", (event) => {
  const value = Number(event.target.value);
  state.armTwistDegreesPer100px = Number.isFinite(value)
    ? Math.max(-360, Math.min(360, value))
    : 0;
  clearArmSuggestion();
  clearSymmetry();
  render();
});

$("#suggestArmModel").addEventListener("click", async () => {
  if (!state.imageId || !state.center || state.circles.length < 6) {
    setStatus("Spiral önerisi için merkez ve en az 6 daire gerekli.", "error");
    return;
  }
  setBusy(true, "Büyük dairelerden spiral kol uyumu hesaplanıyor…");
  try {
    const suggestion = await api("/api/arms/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        circles: state.circles,
        center: state.center,
        phase_degrees: state.phaseDegrees,
      }),
    });
    state.armSuggestion = suggestion;
    const improvement = Number(suggestion.improvement_fraction || 0) * 100;
    $("#armFitReadout").textContent =
      `${Number(suggestion.sample_count)} güçlü daire · faz ${Number(suggestion.phase_degrees).toFixed(1)}° · ` +
      `dönüş ${Number(suggestion.twist_degrees_per_100px).toFixed(1)}°/100 px · ` +
      `RMS ${Number(suggestion.radial_rms_error_degrees).toFixed(1)}° → ` +
      `${Number(suggestion.weighted_rms_error_degrees).toFixed(1)}° (%${improvement.toFixed(0)} iyileşme)`;
    $("#applyArmSuggestion").classList.remove("hidden");
    setStatus("Spiral uyumu hazır; değerler henüz uygulanmadı.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

$("#applyArmSuggestion").addEventListener("click", () => {
  const suggestion = state.armSuggestion;
  if (!suggestion) return;
  pushStructureUndo("approve_spiral_model");
  const previous = {
    arm_model: state.armModel,
    phase_degrees: state.phaseDegrees,
    arm_twist_degrees_per_100px: state.armTwistDegreesPer100px,
  };
  state.armModel = "spiral";
  state.phaseDegrees = Number(suggestion.phase_degrees);
  state.armTwistDegreesPer100px = Number(
    suggestion.twist_degrees_per_100px,
  );
  $("#phaseDegrees").value = String(state.phaseDegrees);
  $("#armModel").value = "spiral";
  $("#armTwist").value = String(state.armTwistDegreesPer100px);
  state.armSuggestion = null;
  const next = {
    arm_model: state.armModel,
    phase_degrees: state.phaseDegrees,
    arm_twist_degrees_per_100px: state.armTwistDegreesPer100px,
  };
  recordStructuralEdit(
    null,
    "arm_model_settings",
    previous,
    next,
    "user",
    "approve_spiral_model",
  );
  $("#applyArmSuggestion").classList.add("hidden");
  clearSymmetry();
  updateArmModelUi();
  render();
  setStatus(
    "Spiral önerisi kontrollere uygulandı; daireleri değiştirmek için analizi çalıştırın.",
    "success",
  );
});

$("#showArms").addEventListener("change", (event) => {
  state.showArms = event.target.checked;
  updateUi();
});

$("#showLabels").addEventListener("change", (event) => {
  state.showLabels = event.target.checked;
  state.labelMode = event.target.checked ? "all" : "selected";
  $("#labelMode").value = state.labelMode;
  render();
});

$("#refreshStructureButton").addEventListener("click", () => {
  void refreshStructureAnalysis(true);
});

$("#undoStructureButton").addEventListener("click", undoLastStructureChange);
$("#clearAllCircleLocks").addEventListener("click", clearAllCircleLocks);
$("#approveCenterCandidate").addEventListener("click", approveCenterCandidate);
$("#applySelectedBoundary").addEventListener("click", applyBoundarySelection);
$("#applySelectedLocalSides").addEventListener("click", applyLocalSideSelection);
$("#startMainChain").addEventListener("click", startMainChainDraft);
$("#cancelMainChain").addEventListener("click", cancelMainChainDraft);
$("#confirmMainChain").addEventListener("click", confirmMainChainDraft);
$("#applySelectedSatellites").addEventListener("click", applySatelliteSelection);
$("#unlinkSelectedParent").addEventListener("click", unlinkSelectedParent);

$("#boundaryTopFourOnly").addEventListener("change", renderBoundaryCandidates);
$("#boundarySelectAll").addEventListener("change", (event) => {
  const rows = (state.structureReport?.boundary_candidates || []).filter(
    (item) =>
      (!$("#boundaryTopFourOnly").checked || item.most_likely_four) &&
      !item.manual_locked &&
      Number(item.difference_score) > 0,
  );
  state.boundarySelection = event.target.checked
    ? new Set(rows.map((item) => String(item.id)))
    : new Set();
  renderBoundaryCandidates();
});

$("#localSideArmFilter").addEventListener("change", renderLocalSideCandidates);
$("#localSideTypeFilter").addEventListener("change", renderLocalSideCandidates);
$("#localSideOnlyChanged").addEventListener("change", renderLocalSideCandidates);
$("#localSideOnlyLowConfidence").addEventListener("change", renderLocalSideCandidates);
$("#localSideMinConfidence").addEventListener("input", renderLocalSideCandidates);

$("#selectSafeLocalSides").addEventListener("click", () => {
  const threshold = Math.max(
    0,
    Math.min(1, Number($("#localSideMinConfidence").value || 0.6)),
  );
  for (const item of visibleLocalSideCandidates()) {
    if (
      !item.side_locked &&
      item.suggested_side !== "unknown" &&
      Number(item.confidence) >= threshold
    ) {
      state.localSideSelection.add(String(item.id));
    }
  }
  renderLocalSideCandidates();
});

$("#clearLocalSideSelection").addEventListener("click", () => {
  state.localSideSelection.clear();
  renderLocalSideCandidates();
});

$("#selectAllSatelliteSuggestions").addEventListener("click", () => {
  const arm = Number($("#mainChainArm").value || 1);
  const rows = (state.structureReport?.satellite_candidates || []).filter(
    (item) => Number(item.arm_id) === arm && !item.manual_locked,
  );
  state.satelliteSelection = new Set(rows.map((item) => String(item.id)));
  renderSatelliteCandidates();
});

$("#mainChainArm").addEventListener("change", () => {
  renderMainChainUi();
  renderSatelliteCandidates();
  if ($("#filterSelectedArm").checked) render();
});

$("#satelliteTableArm").addEventListener("change", renderSatelliteStatus);

$("#boundaryCandidatesBody").addEventListener("change", (event) => {
  const id = event.target.dataset.boundaryId;
  if (!id) return;
  if (event.target.checked) state.boundarySelection.add(id);
  else state.boundarySelection.delete(id);
  renderBoundaryCandidates();
});

$("#localSideCandidatesBody").addEventListener("change", (event) => {
  const id = event.target.dataset.localSideId;
  if (!id) return;
  if (event.target.checked) state.localSideSelection.add(id);
  else state.localSideSelection.delete(id);
  renderLocalSideCandidates();
});

$("#satelliteCandidatesBody").addEventListener("change", (event) => {
  const id = event.target.dataset.satelliteId;
  if (!id) return;
  if (event.target.checked) state.satelliteSelection.add(id);
  else state.satelliteSelection.delete(id);
  renderSatelliteCandidates();
});

$("#mainChainList").addEventListener("click", (event) => {
  const moveButton = event.target.closest("[data-chain-move]");
  if (moveButton) {
    moveMainChainItem(
      Number(moveButton.dataset.chainIndex),
      Number(moveButton.dataset.chainMove),
    );
    return;
  }
  const removeButton = event.target.closest("[data-chain-remove]");
  if (removeButton) removeMainChainDraftItem(Number(removeButton.dataset.chainRemove));
});

document.addEventListener("click", (event) => {
  const focusButton = event.target.closest("[data-focus-id]");
  if (focusButton) {
    const circle = state.circles.find((item) => item.id === focusButton.dataset.focusId);
    if (circle) focusCircle(circle);
    return;
  }
  const mergeButton = event.target.closest("[data-merge-duplicate]");
  if (mergeButton) mergeDuplicateCandidate(Number(mergeButton.dataset.mergeDuplicate));
});

for (const id of [
  "filterUnclassified",
  "filterMain",
  "filterSatellite",
  "filterCenter",
  "filterSelectedArm",
  "filterBoundary",
  "filterMissingParent",
  "filterDuplicates",
  "filterLowConfidence",
  "filterManual",
]) {
  $(`#${id}`).addEventListener("change", render);
}

$("#labelMode").addEventListener("change", (event) => {
  state.labelMode = event.target.value;
  state.showLabels = state.labelMode === "all";
  $("#showLabels").checked = state.showLabels;
  render();
});

$("#clearStructureFilters").addEventListener("click", () => {
  for (const id of [
    "filterUnclassified",
    "filterMain",
    "filterSatellite",
    "filterCenter",
    "filterSelectedArm",
    "filterBoundary",
    "filterMissingParent",
    "filterDuplicates",
    "filterLowConfidence",
    "filterManual",
  ]) {
    $(`#${id}`).checked = false;
  }
  state.labelMode = "selected";
  state.showLabels = false;
  $("#labelMode").value = "selected";
  $("#showLabels").checked = false;
  $("#localSideArmFilter").value = "all";
  $("#localSideTypeFilter").value = "all";
  $("#localSideOnlyChanged").checked = false;
  $("#localSideOnlyLowConfidence").checked = false;
  $("#localSideMinConfidence").value = "0.60";
  renderLocalSideCandidates();
  render();
});

async function runArmAnalysis({ forceReindex = false } = {}) {
  if (!state.imageId) {
    setStatus("Önce bir görüntü açın.", "error");
    return;
  }
  if (!state.center) {
    setStatus("Önce genel merkezi seçin.", "error");
    setTool("center");
    return;
  }
  const locks = manualLocksPayload();
  if (forceReindex) {
    for (const circleId of Object.keys(locks)) {
      locks[circleId] = locks[circleId].filter(
        (fieldName) => fieldName !== "sequence_index",
      );
      if (!locks[circleId].length) delete locks[circleId];
    }
  }
  setBusy(
    true,
    forceReindex
      ? "Kollar merkezden dışarı yeniden numaralanıyor…"
      : "Altılı simetri hesaplanıyor…",
  );
  const operationType = forceReindex ? "reindex_arms" : "approve_arm_analysis";
  const previousById = new Map(
    state.circles.map((circle) => [circle.id, deepClone(circle)]),
  );
  pushStructureUndo(operationType);
  try {
    const payload = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        circles: state.circles,
        manual_locks: locks,
        center: state.center,
        phase_degrees: state.phaseDegrees,
        ...armSettingsPayload(),
      }),
    });
    replaceAnalyzedCircles(payload.circles);
    const batchId = `${operationType}-${Date.now().toString(36)}`;
    let structuralChangeCount = 0;
    for (const circle of state.circles) {
      const previous = previousById.get(circle.id);
      if (!previous) continue;
      let circleChanged = false;
      for (const field of ["arm_id", "sequence_index", "side", "parent_id"]) {
        if (structuralValuesEqual(previous[field], circle[field])) continue;
        recordStructuralEdit(
          circle.id,
          field,
          previous[field] ?? null,
          circle[field] ?? null,
          "user_approved_automatic",
          operationType,
          batchId,
        );
        structuralChangeCount += 1;
        circleChanged = true;
      }
      if (circleChanged) {
        circle.classification_source = "automatic_approved";
        circle.classification_confidence = Math.max(
          Number(circle.classification_confidence || 0),
          0.5,
        );
      }
    }
    if (forceReindex) {
      for (const circle of state.circles) {
        circle._locked_fields = (circle._locked_fields || []).filter(
          (fieldName) => fieldName !== "sequence_index",
        );
        circle.locked_fields = [...circle._locked_fields];
      }
    }
    if (!structuralChangeCount) state.structureUndo.pop();
    showSymmetry(payload);
    updateUi();
    await refreshStructureAnalysis(false);
    setStatus(
      forceReindex
        ? "Manuel kol atamaları korundu; her kol merkezden dışarı 1…N yeniden numaralandı."
        : "Kutupsal koordinatlar, altı kol ve simetri hataları hesaplandı.",
      "success",
    );
  } catch (error) {
    state.structureUndo.pop();
    $("#undoStructureButton").disabled = state.structureUndo.length === 0;
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
}

$("#analyzeButton").addEventListener("click", () => {
  void runArmAnalysis();
});

$("#reindexArmsButton").addEventListener("click", () => {
  void runArmAnalysis({ forceReindex: true });
});

$("#saveProjectButton").addEventListener("click", async () => {
  if (!state.imageId) {
    setStatus("Önce bir görüntü açın.", "error");
    return;
  }
  setBusy(true, "Daire ayrıntıları çalışma JSON'una kaydediliyor…");
  try {
    const payload = await api("/api/project/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(projectRequestPayload()),
    });
    const item = payload.file;
    $("#projectDownload").innerHTML =
      `<a href="${item.url}" download="${escapeHtml(item.name)}">Çalışma JSON'u</a>`;
    $("#projectDownload").classList.remove("hidden");
    setStatus(
      `${payload.circle_count} daire ve düzenleme günlüğü çalışma JSON'una kaydedildi.`,
      "success",
    );
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

$("#projectInput").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  if (!state.imageId) {
    setStatus("Önce çalışma dosyasının kaynak görüntüsünü açın.", "error");
    event.target.value = "";
    return;
  }
  const form = new FormData();
  form.append("image_id", state.imageId);
  form.append("project", file);
  setBusy(true, "Çalışma doğrulanıyor ve daireler geri yükleniyor…");
  try {
    const payload = await api("/api/project/load", {
      method: "POST",
      body: form,
    });
    const document = payload.document;
    state.circles = [];
    state.selectedId = null;
    if (document.preprocess_profile) {
      state.preprocessProfile = deepClone(document.preprocess_profile);
      state.tramlineSuggestion = null;
      state.selectedTramlineId = null;
      state.tramlineHistory = [];
      state.preprocessApplied = false;
      state.appliedPreprocessSignature = null;
      syncControlsFromProfile();
      await applyPreprocessStack(false);
    } else if (state.preprocessApplied && state.defaultPreprocessProfile) {
      state.preprocessProfile = deepClone(state.defaultPreprocessProfile);
      state.tramlineSuggestion = null;
      state.selectedTramlineId = null;
      state.tramlineHistory = [];
      for (const operation of state.preprocessProfile.operations || []) {
        operation.enabled = false;
      }
      state.preprocessApplied = false;
      state.appliedPreprocessSignature = null;
      syncControlsFromProfile();
      await applyPreprocessStack(false);
    }
    restoreProjectDocument(document);
    await refreshStructureAnalysis(false);
    const legacyNote =
      payload.source_schema_version &&
      Number(payload.source_schema_version) < 1.4
        ? " Eski dosyada kaynak/düzeltilmiş yarıçap ve elips eksenleri bulunmayabilir."
        : "";
    const draftNote = state.mainChainDraft
      ? ` K${state.mainChainDraft.arm} Ana Zincir taslağı ` +
        `${state.mainChainDraft.ids.length}/13 seçimle geri yüklendi.`
      : "";
    setStatus(
      `${state.circles.length} daire, merkez ve kol ayarları geri yüklendi.` +
      `${payload.warning ? ` Uyarı: ${payload.warning}` : ""}${legacyNote}` +
      `${payload.backup ? ` Yedek: ${payload.backup.relative_path}.` : ""}` +
      draftNote,
      payload.warning ? "warning" : "success",
    );
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
    event.target.value = "";
  }
});

$("#exportButton").addEventListener("click", async () => {
  if (!state.imageId) {
    setStatus("Önce bir görüntü açın.", "error");
    return;
  }
  setBusy(true, "PNG, JSON ve yapısal CSV dosyaları hazırlanıyor…");
  try {
    const payload = await api("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(projectRequestPayload()),
    });
    replaceAnalyzedCircles(payload.circles);
    if (payload.summary && Object.keys(payload.summary).length) {
      // Export recomputes the same authoritative values, but the full rows stay
      // from the explicit Analyze action.
      updateUi();
    }
    const exportOrder = [
      "png",
      "json",
      "circles_classified",
      "arm_summary",
      "main_sequence",
      "satellite_relationships",
      "boundary_candidates",
      "structure_summary",
      "csv",
    ];
    $("#downloads").innerHTML = exportOrder
      .filter((kind) => payload.files[kind])
      .map((kind) => {
        const item = payload.files[kind];
        return `<a href="${item.url}" download="${escapeHtml(item.name)}">${escapeHtml(item.name)}</a>`;
      }).join("");
    $("#downloads").classList.remove("hidden");
    setStatus("Çıktılar outputs/ ve data/annotations/ klasörlerine kaydedildi.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

$$("[data-field]").forEach((input) => {
  input.addEventListener("change", (event) => {
    const circle = selectedCircle();
    if (!circle) return;
    const field = event.target.dataset.field;
    if (field === "id") return;
    let nextValue;
    if (event.target.type === "checkbox") {
      nextValue = event.target.checked;
    } else if (["x", "y", "radius", "confidence", "stroke_width", "fill_opacity"].includes(field)) {
      const value = Number(event.target.value);
      if (!Number.isFinite(value)) {
        updateInspector();
        return;
      }
      nextValue = field === "radius" ? Math.max(0.1, value) : value;
      if (field === "confidence") nextValue = Math.max(0, Math.min(1, value));
      if (field === "stroke_width") nextValue = Math.max(0.5, Math.min(20, value));
      if (field === "fill_opacity") nextValue = Math.max(0, Math.min(1, value));
    } else if (["arm_id", "sequence_index"].includes(field)) {
      nextValue = event.target.value === "" ? null : Number(event.target.value);
    } else {
      nextValue = event.target.value || null;
    }
    if (field === "parent_id" && nextValue) {
      const parent = state.circles.find((item) => item.id === nextValue);
      if (!parent || !approvedMainIds().has(parent.id)) {
        setStatus("parent_id onaylı bir Ana Zincir çemberine ait olmalı.", "error");
        updateInspector();
        return;
      }
    }
    const structuralField = [
      "circle_type",
      "arm_id",
      "sequence_index",
      "parent_id",
      "side",
    ].includes(field);
    if (structuralField) {
      if (field === "circle_type" && nextValue === "center") {
        const conflicts = ["arm_id", "sequence_index", "parent_id", "side"].filter(
          (relatedField) =>
            circleFieldLocked(circle, relatedField) &&
            !structuralValuesEqual(
              circle[relatedField],
              relatedField === "side" ? "center" : null,
            ),
        );
        if (conflicts.length) {
          setStatus(`Merkez onayı için önce ${conflicts.join(", ")} kilidini kaldırın.`, "error");
          updateInspector();
          return;
        }
      }
      pushStructureUndo(`manual_${field}`);
      const changes = {
        [field]: nextValue,
        classification_source: "user_confirmed",
        classification_confidence: 1,
      };
      if (field === "arm_id" && !circleFieldLocked(circle, "sequence_index")) {
        changes.sequence_index = null;
      }
      if (field === "circle_type" && nextValue === "center") {
        Object.assign(changes, {
          arm_id: null,
          sequence_index: null,
          parent_id: null,
          side: "center",
        });
      }
      if (field === "parent_id") {
        if (nextValue) {
          const parent = state.circles.find((item) => item.id === nextValue);
          if (
            circleFieldLocked(circle, "arm_id") &&
            !structuralValuesEqual(circle.arm_id, parent.arm_id)
          ) {
            state.structureUndo.pop();
            setStatus("Seçilen parent farklı kolda; önce arm_id kilidini kaldırın.", "error");
            updateInspector();
            return;
          }
          Object.assign(changes, relationshipMetrics(circle, parent));
          changes.arm_id = parent.arm_id;
        } else {
          changes.local_angle = null;
          changes.normalized_parent_distance = null;
        }
      }
      applyCircleChanges(
        circle,
        changes,
        {
          operationType: `manual_${field}`,
          lockFields: [field],
        },
      );
    } else {
      const previous = circle[field] ?? null;
      if (field === "locked") pushStructureUndo("manual_record_lock");
      circle[field] = nextValue;
      if (field === "locked") {
        recordStructuralEdit(
          circle.id,
          "locked",
          previous,
          nextValue,
          "user",
          "manual_record_lock",
        );
      }
      markManual(circle, field);
    }
    if (field === "locked" && circle.locked) {
      setStatus(`${circle.id} kullanıcı onayıyla kilitlendi.`, "success");
    } else if (field === "arm_id") {
      setStatus(
        `${circle.id} için eski sıra kaldırıldı; yeniden numaralama işlemini çalıştırın.`,
        "success",
      );
    }
    clearSymmetry();
    updateUi();
    if (structuralField) void refreshStructureAnalysis(false);
  });
});

$$("[data-lock-field]").forEach((input) => {
  input.addEventListener("change", (event) => {
    const circle = selectedCircle();
    if (!circle) return;
    const field = event.target.dataset.lockField;
    pushStructureUndo("change_field_lock");
    const previous = [...(circle._locked_fields || [])];
    const locks = new Set(previous);
    if (event.target.checked) locks.add(field);
    else locks.delete(field);
    circle._locked_fields = [...locks];
    circle.locked_fields = [...locks];
    recordStructuralEdit(
      circle.id,
      `locked_fields.${field}`,
      previous.includes(field),
      event.target.checked,
      "user",
      "change_field_lock",
    );
    markManual(circle);
    updateUi();
    void refreshStructureAnalysis(false);
    setStatus(
      `${circle.id} · ${field} alan kilidi ${event.target.checked ? "açıldı" : "kaldırıldı"}.`,
      "success",
    );
  });
});

$("#resetCircleAppearance").addEventListener("click", () => {
  const circle = selectedCircle();
  if (!circle) return;
  const hasCustomAppearance =
    circle.stroke_color !== null ||
    circle.stroke_width !== null ||
    circle.fill_enabled ||
    circle.fill_color !== null ||
    Math.abs(Number(circle.fill_opacity ?? 0.2) - 0.2) > 1e-9;
  if (!hasCustomAppearance) {
    setStatus(`${circle.id} zaten varsayılan kol görünümünü kullanıyor.`);
    return;
  }
  circle.stroke_color = null;
  circle.stroke_width = null;
  circle.fill_enabled = false;
  circle.fill_color = null;
  circle.fill_opacity = 0.2;
  markManual(circle);
  updateUi();
  setStatus(`${circle.id} varsayılan kol görünümüne döndürüldü.`, "success");
});

$("#deleteSelected").addEventListener("click", () => {
  if (state.selectedId) removeCircle(state.selectedId);
});

$$(".tool[data-tool]").forEach((button) => {
  button.addEventListener("click", () => setTool(button.dataset.tool));
});

$$("[data-activate-tool]").forEach((button) => {
  button.addEventListener("click", () => setTool(button.dataset.activateTool));
});

$("#fitButton").addEventListener("click", fitView);

window.addEventListener("keydown", (event) => {
  const target = event.target;
  const typing = target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement;
  if (event.code === "Space" && !typing) {
    event.preventDefault();
    state.spaceDown = true;
    canvas.style.cursor = "grab";
  }
  if (typing) return;
  const key = event.key.toLowerCase();
  if (key === "v" && state.workspace === "measure") setTool("select");
  else if (key === "h") setTool("pan");
  else if (key === "a" && state.workspace === "measure") setTool("add");
  else if (key === "d" && state.workspace === "measure") setTool("delete");
  else if (key === "c" && state.workspace === "measure") setTool("center");
  else if (key === "p" && state.workspace === "preprocess") setTool("perspective");
  else if (key === "i" && state.workspace === "preprocess") {
    setTool(event.shiftKey ? "tramlineAdd" : "tramlineEdit");
  }
  else if (key === "f") fitView();
  else if (event.key === "Delete" || event.key === "Backspace") {
    if (
      state.workspace === "preprocess"
      && state.selectedTramlineId
      && state.tramlineSuggestion
    ) {
      event.preventDefault();
      pushTramlineHistory();
      state.tramlineSuggestion.lines = state.tramlineSuggestion.lines.filter(
        (line) => line.id !== state.selectedTramlineId,
      );
      state.selectedTramlineId = null;
      recomputeTramlineSuggestion();
      setStatus("Seçili tarla izi öneriden silindi.", "success");
    } else if (state.selectedId) {
      event.preventDefault();
      removeCircle(state.selectedId);
    }
  } else if (event.key === "Escape") {
    state.previewCircle = null;
    if (state.tool === "perspective") state.perspectivePoints = [];
    state.selectedTramlineId = null;
    state.pointer = null;
    setTool(state.workspace === "measure" ? "select" : "pan");
    updateUi();
  }
});

window.addEventListener("keyup", (event) => {
  if (event.code === "Space") {
    state.spaceDown = false;
    canvas.style.cursor = state.tool === "pan" ? "grab" : "";
  }
});

window.addEventListener("blur", () => {
  state.spaceDown = false;
});

const resizeObserver = new ResizeObserver(resizeCanvas);
resizeObserver.observe(canvasWrap);
resizeCanvas();
setWorkspace("preprocess");
