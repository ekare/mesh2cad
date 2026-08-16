// The adaptive preview keeps normal browser sessions responsive. Full-resolution mode
// deliberately removes this display-only budget; conversion and fidelity always use the
// complete uploaded mesh regardless of the viewport setting.
const DEFAULT_PREVIEW_TRIANGLES = 250_000;
let fullResolutionPreview = false;

function previewTriangleBudget() {
  return fullResolutionPreview ? Number.POSITIVE_INFINITY : DEFAULT_PREVIEW_TRIANGLES;
}

const byId = (id) => document.getElementById(id);
const dropzone = byId("dropzone");
const fileInput = byId("file-input");
const convertButton = byId("convert-button");
const simplifyStlButton = byId("simplify-stl-button");
const repairButton = byId("repair-button");
const simplifyStepButton = byId("simplify-step-button");
const stlSimplifyRange = byId("stl-simplify-range");
const stlSimplifyValue = byId("stl-simplify-value");
const fullPreviewToggle = byId("full-preview-toggle");
const stepSimplifyRange = byId("step-simplify-range");
const stepSimplifyValue = byId("step-simplify-value");
const progressCard = byId("progress-card");
const resultCard = byId("result-card");
const errorCard = byId("error-card");
const previewSwitch = byId("preview-switch");
const previewButtons = [...previewSwitch.querySelectorAll("button[data-preview]")];
let selectedFile = null;
let lastConversionToken = null;

function multiply4(a, b) {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let value = 0;
      for (let index = 0; index < 4; index += 1) {
        value += a[index * 4 + row] * b[column * 4 + index];
      }
      out[column * 4 + row] = value;
    }
  }
  return out;
}

function perspective(fieldOfView, aspect, near, far) {
  const scale = 1 / Math.tan(fieldOfView / 2);
  const range = 1 / (near - far);
  return new Float32Array([
    scale / aspect, 0, 0, 0,
    0, scale, 0, 0,
    0, 0, (near + far) * range, -1,
    0, 0, 2 * near * far * range, 0,
  ]);
}

function translation(z) {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, z, 1,
  ]);
}

function rotation(yaw, pitch) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cx = Math.cos(pitch);
  const sx = Math.sin(pitch);
  const rotateY = new Float32Array([
    cy, 0, -sy, 0,
    0, 1, 0, 0,
    sy, 0, cy, 0,
    0, 0, 0, 1,
  ]);
  const rotateX = new Float32Array([
    1, 0, 0, 0,
    0, cx, sx, 0,
    0, -sx, cx, 0,
    0, 0, 0, 1,
  ]);
  return multiply4(rotateX, rotateY);
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "WebGL shader compilation failed");
  }
  return shader;
}

function createProgram(gl) {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, `
    attribute vec3 aPosition;
    attribute vec3 aNormal;
    attribute float aDeviation;
    uniform mat4 uMvp;
    uniform mat4 uModel;
    varying vec3 vNormal;
    varying vec3 vPosition;
    varying float vDeviation;
    void main() {
      vec4 world = uModel * vec4(aPosition, 1.0);
      vPosition = world.xyz;
      vNormal = normalize(mat3(uModel) * aNormal);
      vDeviation = aDeviation;
      gl_Position = uMvp * vec4(aPosition, 1.0);
    }
  `);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying vec3 vNormal;
    varying vec3 vPosition;
    varying float vDeviation;
    uniform vec3 uBaseColor;
    uniform float uUseHeatmap;
    uniform float uHeatmapScale;
    vec3 heatmap(float ratio) {
      vec3 green = vec3(0.18, 0.92, 0.43);
      vec3 yellow = vec3(1.0, 0.83, 0.16);
      vec3 red = vec3(1.0, 0.18, 0.13);
      return ratio < 0.5
        ? mix(green, yellow, ratio * 2.0)
        : mix(yellow, red, (ratio - 0.5) * 2.0);
    }
    void main() {
      vec3 normal = normalize(vNormal);
      vec3 key = normalize(vec3(-0.55, 0.82, 0.42));
      vec3 rimDirection = normalize(vec3(0.65, 0.35, -0.7));
      float diffuse = max(dot(normal, key), 0.0);
      float rim = pow(1.0 - abs(dot(normal, rimDirection)), 2.2);
      float bands = 0.96 + 0.04 * sin(vPosition.y * 7.0);
      float deviationRatio = clamp(vDeviation / max(uHeatmapScale, 1.0e-12), 0.0, 1.0);
      vec3 selectedBase = mix(uBaseColor, heatmap(deviationRatio), uUseHeatmap);
      vec3 base = selectedBase * bands;
      vec3 color = base * (0.28 + 0.82 * diffuse) + vec3(0.25, 0.9, 1.0) * rim * 0.42;
      gl_FragColor = vec4(color, 1.0);
    }
  `);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || "WebGL program link failed");
  }
  return program;
}

function faceNormal(a, b, c) {
  const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const cross = [
    ab[1] * ac[2] - ab[2] * ac[1],
    ab[2] * ac[0] - ab[0] * ac[2],
    ab[0] * ac[1] - ab[1] * ac[0],
  ];
  const length = Math.hypot(...cross) || 1;
  return cross.map((value) => value / length);
}

function appendTriangle(positions, normals, a, b, c, normal) {
  const resolved = Math.hypot(...normal) > 1e-12 ? normal : faceNormal(a, b, c);
  positions.push(...a, ...b, ...c);
  normals.push(...resolved, ...resolved, ...resolved);
}

function parseBinaryStl(buffer, triangleCount) {
  const view = new DataView(buffer);
  const stride = Math.max(1, Math.ceil(triangleCount / previewTriangleBudget()));
  const positions = [];
  const normals = [];
  for (let triangle = 0; triangle < triangleCount; triangle += stride) {
    const offset = 84 + triangle * 50;
    const normal = [0, 4, 8].map((delta) => view.getFloat32(offset + delta, true));
    const vertices = [];
    for (let vertex = 0; vertex < 3; vertex += 1) {
      const start = offset + 12 + vertex * 12;
      vertices.push([0, 4, 8].map((delta) => view.getFloat32(start + delta, true)));
    }
    appendTriangle(positions, normals, vertices[0], vertices[1], vertices[2], normal);
  }
  return { positions, normals };
}

function parseAsciiStl(buffer) {
  const lines = new TextDecoder("utf-8", { fatal: false }).decode(buffer).split(/\r?\n/);
  const triangles = [];
  let normal = [0, 0, 0];
  let vertices = [];
  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("facet normal")) {
      normal = line.split(/\s+/).slice(-3).map(Number);
    } else if (line.startsWith("vertex")) {
      vertices.push(line.split(/\s+/).slice(-3).map(Number));
      if (vertices.length === 3) {
        triangles.push({ vertices, normal });
        vertices = [];
      }
    }
  }
  if (!triangles.length) throw new Error("The STL contains no readable triangles.");
  const stride = Math.max(1, Math.ceil(triangles.length / previewTriangleBudget()));
  const positions = [];
  const normals = [];
  for (let index = 0; index < triangles.length; index += stride) {
    const triangle = triangles[index];
    appendTriangle(positions, normals, ...triangle.vertices, triangle.normal);
  }
  return { positions, normals };
}

function parseStl(buffer) {
  if (buffer.byteLength >= 84) {
    const view = new DataView(buffer);
    const triangleCount = view.getUint32(80, true);
    const expected = 84 + triangleCount * 50;
    if (triangleCount > 0 && expected <= buffer.byteLength) {
      return parseBinaryStl(buffer, triangleCount);
    }
  }
  return parseAsciiStl(buffer);
}

function normalizePositions(values, existingTransform = null) {
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  if (!existingTransform) {
    for (let index = 0; index < values.length; index += 3) {
      for (let axis = 0; axis < 3; axis += 1) {
        minimum[axis] = Math.min(minimum[axis], values[index + axis]);
        maximum[axis] = Math.max(maximum[axis], values[index + axis]);
      }
    }
  }
  const transform = existingTransform || (() => {
    const center = minimum.map((value, axis) => (value + maximum[axis]) / 2);
    const extent = Math.max(...maximum.map((value, axis) => value - minimum[axis])) || 1;
    return { center, scale: 3.3 / extent };
  })();
  return {
    positions: new Float32Array(
      values.map((value, index) => (value - transform.center[index % 3]) * transform.scale),
    ),
    transform,
  };
}

function stepPreviewGeometry(preview, transform) {
  const rawPositions = preview.positions.map(Number);
  if (!rawPositions.length || rawPositions.length % 9 !== 0) {
    throw new Error("The reconstructed STEP preview is malformed.");
  }
  const normals = [];
  const deviations = [];
  for (let index = 0; index < rawPositions.length; index += 9) {
    const a = rawPositions.slice(index, index + 3);
    const b = rawPositions.slice(index + 3, index + 6);
    const c = rawPositions.slice(index + 6, index + 9);
    const normal = faceNormal(a, b, c);
    normals.push(...normal, ...normal, ...normal);
    const deviation = Number(preview.deviations?.[index / 9] || 0);
    deviations.push(deviation, deviation, deviation);
  }
  return {
    positions: normalizePositions(rawPositions, transform).positions,
    normals: new Float32Array(normals),
    deviations: new Float32Array(deviations),
  };
}

function placeholderGeometry() {
  const positions = [];
  const normals = [];
  const majorSegments = 64;
  const minorSegments = 22;
  const point = (u, v) => {
    const major = 1.05;
    const minor = 0.34;
    return [
      (major + minor * Math.cos(v)) * Math.cos(u),
      minor * Math.sin(v),
      (major + minor * Math.cos(v)) * Math.sin(u),
    ];
  };
  for (let uIndex = 0; uIndex < majorSegments; uIndex += 1) {
    for (let vIndex = 0; vIndex < minorSegments; vIndex += 1) {
      const u0 = 2 * Math.PI * uIndex / majorSegments;
      const u1 = 2 * Math.PI * (uIndex + 1) / majorSegments;
      const v0 = 2 * Math.PI * vIndex / minorSegments;
      const v1 = 2 * Math.PI * (vIndex + 1) / minorSegments;
      const a = point(u0, v0);
      const b = point(u1, v0);
      const c = point(u1, v1);
      const d = point(u0, v1);
      appendTriangle(positions, normals, a, b, c, [0, 0, 0]);
      appendTriangle(positions, normals, a, c, d, [0, 0, 0]);
    }
  }
  return { positions: new Float32Array(positions), normals: new Float32Array(normals) };
}

class MeshViewer {
  constructor(host) {
    this.canvas = document.createElement("canvas");
    host.appendChild(this.canvas);
    this.gl = this.canvas.getContext("webgl", { antialias: true, alpha: false });
    if (!this.gl) throw new Error("WebGL is unavailable in this browser.");
    this.program = createProgram(this.gl);
    this.positionBuffer = this.gl.createBuffer();
    this.normalBuffer = this.gl.createBuffer();
    this.deviationBuffer = this.gl.createBuffer();
    this.yaw = -0.55;
    this.pitch = -0.35;
    this.distance = 5.4;
    this.dragging = false;
    this.lastPointer = [0, 0];
    this.geometries = new Map();
    this.normalization = null;
    this.setGeometry(placeholderGeometry(), [0.34, 0.46, 1.0]);
    this.bindControls();
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.draw = this.draw.bind(this);
    requestAnimationFrame(this.draw);
  }

  bindControls() {
    this.canvas.addEventListener("pointerdown", (event) => {
      this.dragging = true;
      this.lastPointer = [event.clientX, event.clientY];
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (!this.dragging) return;
      this.yaw += (event.clientX - this.lastPointer[0]) * 0.009;
      this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + (event.clientY - this.lastPointer[1]) * 0.009));
      this.lastPointer = [event.clientX, event.clientY];
    });
    this.canvas.addEventListener("pointerup", () => { this.dragging = false; });
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.distance = Math.max(2.8, Math.min(10, this.distance * Math.exp(event.deltaY * 0.001)));
    }, { passive: false });
  }

  resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }

  setGeometry({ positions, normals, deviations }, color, heatmap = false, heatmapScale = 1) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
    this.vertexCount = positions.length / 3;
    const resolvedDeviations = deviations?.length === this.vertexCount
      ? deviations
      : new Float32Array(this.vertexCount);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.deviationBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, resolvedDeviations, gl.STATIC_DRAW);
    this.baseColor = new Float32Array(color);
    this.useHeatmap = heatmap;
    this.heatmapScale = Math.max(Number(heatmapScale) || 1, 1e-12);
  }

  storeGeometry(name, geometry, color, heatmap = false, heatmapScale = 1) {
    this.geometries.set(name, { geometry, color, heatmap, heatmapScale });
  }

  showGeometry(name) {
    const stored = this.geometries.get(name);
    if (!stored) return false;
    this.setGeometry(stored.geometry, stored.color, stored.heatmap, stored.heatmapScale);
    this.activeGeometry = name;
    return true;
  }

  async showFile(file) {
    const parsed = parseStl(await file.arrayBuffer());
    const normalized = normalizePositions(parsed.positions);
    this.normalization = normalized.transform;
    this.geometries.clear();
    this.storeGeometry("original", {
      positions: normalized.positions,
      normals: new Float32Array(parsed.normals),
    }, [0.34, 0.46, 1.0]);
    this.showGeometry("original");
  }

  showReconstruction(preview) {
    const geometry = stepPreviewGeometry(preview, this.normalization);
    this.storeGeometry("step", geometry, [0.18, 0.78, 0.5]);
    this.storeGeometry(
      "deviation",
      geometry,
      [0.18, 0.78, 0.5],
      true,
      preview.deviation_scale,
    );
    this.showGeometry("step");
  }

  draw() {
    const gl = this.gl;
    this.resize();
    if (!this.dragging) this.yaw += 0.0022;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.clearColor(0.027, 0.039, 0.075, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);

    const position = gl.getAttribLocation(this.program, "aPosition");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
    const normal = gl.getAttribLocation(this.program, "aNormal");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
    gl.enableVertexAttribArray(normal);
    gl.vertexAttribPointer(normal, 3, gl.FLOAT, false, 0, 0);
    const deviation = gl.getAttribLocation(this.program, "aDeviation");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.deviationBuffer);
    gl.enableVertexAttribArray(deviation);
    gl.vertexAttribPointer(deviation, 1, gl.FLOAT, false, 0, 0);

    const model = rotation(this.yaw, this.pitch);
    const viewModel = multiply4(translation(-this.distance), model);
    const projection = perspective(Math.PI / 4.8, this.canvas.width / this.canvas.height, 0.05, 100);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.program, "uModel"), false, model);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.program, "uMvp"), false, multiply4(projection, viewModel));
    gl.uniform3fv(gl.getUniformLocation(this.program, "uBaseColor"), this.baseColor);
    gl.uniform1f(gl.getUniformLocation(this.program, "uUseHeatmap"), this.useHeatmap ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.program, "uHeatmapScale"), this.heatmapScale);
    gl.drawArrays(gl.TRIANGLES, 0, this.vertexCount);
    requestAnimationFrame(this.draw);
  }
}

let viewer;
try {
  viewer = new MeshViewer(byId("viewer"));
} catch (error) {
  byId("viewer").textContent = error.message;
}

function showError(message) {
  progressCard.hidden = true;
  resultCard.hidden = true;
  byId("heatmap-legend").hidden = true;
  errorCard.textContent = message;
  errorCard.hidden = false;
}

function setPreviewMode(mode) {
  if (!viewer?.showGeometry(mode)) return;
  for (const button of previewButtons) {
    button.classList.toggle("active", button.dataset.preview === mode);
  }
  byId("heatmap-legend").hidden = mode !== "deviation";
}

for (const button of previewButtons) {
  button.addEventListener("click", () => setPreviewMode(button.dataset.preview));
}
byId("heatmap-button").addEventListener("click", () => setPreviewMode("deviation"));

function formatLength(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (number === 0) return "0";
  return Math.abs(number) >= 0.01 ? number.toFixed(4) : number.toExponential(2);
}

function formatPercent(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(100 * number).toFixed(digits)}%` : "—";
}

async function selectFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".stl")) {
    showError("Please choose a binary or ASCII STL file.");
    return;
  }
  selectedFile = file;
  byId("drop-title").textContent = "Mesh ready";
  byId("drop-detail").textContent = `${(file.size / 1048576).toFixed(2)} MB · click to replace`;
  byId("selected-name").textContent = file.name;
  byId("selected-name").hidden = false;
  convertButton.disabled = false;
  simplifyStlButton.disabled = false;
  repairButton.disabled = false;
  resultCard.hidden = true;
  errorCard.hidden = true;
  previewSwitch.hidden = true;
  if (viewer) {
    try {
      await viewer.showFile(file);
      setPreviewMode("original");
    } catch (error) {
      showError(`Preview failed: ${error.message}. Conversion is still available.`);
    }
  }
}

function downloadPayload(payload) {
  const link = document.createElement("a");
  link.href = payload.download_url;
  link.download = payload.output_name || "mesh2cad-output";
  link.click();
}

async function runMeshTool(endpoint, fields, label) {
  if (!selectedFile) return;
  const body = new FormData();
  body.append("file", selectedFile);
  for (const [key, value] of Object.entries(fields)) body.append(key, String(value));
  const response = await fetch(endpoint, { method: "POST", body });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || `${label} failed.`);
  if (endpoint === "/api/simplify-stl" && viewer) {
    const simplifiedResponse = await fetch(payload.download_url);
    const blob = await simplifiedResponse.blob();
    selectedFile = new File([blob], payload.output_name, { type: "model/stl" });
    await viewer.showFile(selectedFile);
    byId("drop-detail").textContent = `${(selectedFile.size / 1048576).toFixed(2)} MB · simplified preview`;
  }
  downloadPayload(payload);
  return payload;
}

async function simplifyStepFromConversion() {
  if (!lastConversionToken) return;
  const body = new FormData();
  body.append("token", lastConversionToken);
  body.append("merge_faces", stepSimplifyRange.value);
  const response = await fetch("/api/simplify-step", { method: "POST", body });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "STEP simplification failed.");
  downloadPayload(payload);
}

stlSimplifyRange.addEventListener("input", () => {
  stlSimplifyValue.textContent = `${Number(stlSimplifyRange.value).toLocaleString()} triangles`;
});
fullPreviewToggle.addEventListener("change", async () => {
  fullResolutionPreview = fullPreviewToggle.checked;
  if (!selectedFile || !viewer) return;
  try {
    await viewer.showFile(selectedFile);
  } catch (error) {
    showError(`Preview failed: ${error.message}. Conversion is still available.`);
  }
});
stepSimplifyRange.addEventListener("input", () => {
  stepSimplifyValue.textContent = stepSimplifyRange.value === "1" ? "merge redundant faces" : "preserve topology";
});

simplifyStlButton.addEventListener("click", async () => {
  simplifyStlButton.disabled = true;
  try { await runMeshTool("/api/simplify-stl", { target_faces: stlSimplifyRange.value }, "STL simplification"); }
  catch (error) { showError(error.message); }
  finally { simplifyStlButton.disabled = false; }
});

simplifyStepButton.addEventListener("click", async () => {
  simplifyStepButton.disabled = true;
  try { await simplifyStepFromConversion(); }
  catch (error) { showError(error.message); }
  finally { simplifyStepButton.disabled = false; }
});

repairButton.addEventListener("click", async () => {
  repairButton.disabled = true;
  try { await runMeshTool("/api/repair", {}, "Repair"); }
  catch (error) { showError(error.message); }
  finally { repairButton.disabled = false; }
});

fileInput.addEventListener("change", () => selectFile(fileInput.files?.[0]));
for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}
dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer?.files?.[0]));

convertButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  convertButton.disabled = true;
  convertButton.textContent = "Reconstructing analytic solid…";
  resultCard.hidden = true;
  errorCard.hidden = true;
  progressCard.hidden = false;
  const body = new FormData();
  body.append("file", selectedFile);
  try {
    const response = await fetch("/api/convert", { method: "POST", body });
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : { detail: await response.text() };
    if (!response.ok) throw new Error(payload.detail || `Conversion failed (HTTP ${response.status}).`);
    lastConversionToken = payload.token;
    byId("planes").textContent = payload.surfaces.planes;
    byId("cylinders").textContent = payload.surfaces.cylinders;
    byId("fidelity-score").textContent = formatPercent(payload.metrics.geometric_fidelity);
    byId("fidelity-tolerance").textContent =
      `bidirectional coverage @ ${formatLength(payload.metrics.fidelity_tolerance)} model units`;
    byId("p95-error").textContent = formatLength(payload.metrics.bidirectional_p95_error);
    byId("max-error").textContent = formatLength(payload.metrics.bidirectional_max_error);
    byId("volume-delta").textContent = formatPercent(payload.metrics.volume_relative_error);
    byId("normal-p95").textContent =
      `${Number(payload.metrics.normal_p95_error_degrees).toFixed(2)}°`;
    byId("analytic-coverage").textContent =
      formatPercent(payload.cad_quality?.analytic_area_fraction, 1);
    byId("brep-faces").textContent =
      Number(payload.cad_quality?.brep_face_count || 0).toLocaleString();
    byId("heatmap-limit").textContent =
      `${formatLength(payload.metrics.fidelity_tolerance)} model units`;
    const faithful = payload.topology?.recognizer === "faithful_unified_mesh_brep";
    byId("result-heading").textContent = faithful
      ? "✓ faithful shape-preserving BRep ready"
      : "✓ verified analytic STEP ready";
    byId("cad-status").textContent = faithful
      ? `${payload.surfaces.engine} regions · Mesh-derived BRep · ${formatPercent(payload.cad_quality?.face_reduction_fraction, 1)} face reduction · geometry preserved, editability limited`
      : `${payload.surfaces.engine} regions · Analytic CSG · ${formatPercent(payload.cad_quality?.face_reduction_fraction, 1)} face reduction · editable primitives`;
    const meshQuality = payload.mesh_quality || {};
    const structural = formatPercent(meshQuality.structural_health, 0);
    const sampling = meshQuality.coarse_sampling_hint
      ? "coarse sampling hint"
      : "dense sampling hint";
    byId("mesh-diagnostic").textContent = meshQuality.repair_recommended
      ? `Repair recommended · structural health ${structural} · ${meshQuality.boundary_edges || 0} boundary edges · ${meshQuality.non_manifold_edges || 0} non-manifold edges`
      : `Structurally clean source · health ${structural} · ${sampling}`;
    byId("mesh-diagnostic").dataset.state = meshQuality.repair_recommended ? "warning" : "clean";
    const download = byId("download-button");
    download.href = payload.download_url;
    download.download = payload.output_name;
    byId("download-label").textContent = `Download ${payload.output_name}`;
    if (viewer && payload.preview?.positions?.length) {
      viewer.showReconstruction(payload.preview);
      previewSwitch.title = `${payload.preview.shown_triangles.toLocaleString()} preview triangles`;
      previewSwitch.hidden = false;
      setPreviewMode("step");
    } else {
      previewSwitch.hidden = true;
      setPreviewMode("original");
    }
    progressCard.hidden = true;
    resultCard.hidden = false;
  } catch (error) {
    showError(error instanceof Error ? error.message : "Conversion failed.");
  } finally {
    convertButton.disabled = false;
    convertButton.textContent = "Convert to STEP →";
  }
});
