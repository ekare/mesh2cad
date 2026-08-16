const byId = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("en-US");
const events = [];
let lastOperation = "";
let lastTrainingPhase = "";

function setText(id, value) {
  byId(id).textContent = value ?? "—";
}

function clamp(value, low = 0, high = 1) {
  return Math.min(high, Math.max(low, Number(value) || 0));
}

function percent(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function metricPercent(value) {
  return value === null || value === undefined ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
}

function duration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return "—";
  const total = Math.max(0, Math.round(Number(seconds)));
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function bytes(value) {
  if (value === null || value === undefined) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = Number(value);
  let index = 0;
  while (amount >= 1024 && index < units.length - 1) { amount /= 1024; index += 1; }
  return `${amount.toFixed(index > 2 ? 2 : 1)} ${units[index]}`;
}

function addEvent(message, timestamp = new Date()) {
  const time = timestamp instanceof Date ? timestamp : new Date(timestamp);
  events.unshift({ message, time });
  events.splice(8);
  byId("event-list").innerHTML = events.map((entry) =>
    `<li><time>${entry.time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time><span>${entry.message}</span></li>`
  ).join("");
}

function renderLossChart(points) {
  const canvas = byId("loss-chart");
  const empty = byId("chart-empty");
  if (!Array.isArray(points) || points.length < 2) {
    empty.hidden = false;
    setText("loss-range", "—");
    return;
  }
  empty.hidden = true;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);

  const values = points.map((point) => Number(point.loss)).filter(Number.isFinite);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const spread = Math.max(maximum - minimum, 1e-6);
  const inset = 8;
  const coordinates = values.map((value, index) => ({
    x: inset + index / Math.max(1, values.length - 1) * (width - inset * 2),
    y: inset + (maximum - value) / spread * (height - inset * 2),
  }));

  const fill = context.createLinearGradient(0, 0, 0, height);
  fill.addColorStop(0, "rgba(200, 255, 77, 0.30)");
  fill.addColorStop(1, "rgba(98, 233, 255, 0.01)");
  context.beginPath();
  context.moveTo(coordinates[0].x, height);
  coordinates.forEach((point) => context.lineTo(point.x, point.y));
  context.lineTo(coordinates.at(-1).x, height);
  context.closePath();
  context.fillStyle = fill;
  context.fill();

  context.beginPath();
  coordinates.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
  context.strokeStyle = "#c8ff4d";
  context.lineWidth = 2;
  context.shadowColor = "rgba(200, 255, 77, 0.45)";
  context.shadowBlur = 10;
  context.stroke();
  setText("loss-range", `${minimum.toFixed(4)} — ${maximum.toFixed(4)}`);
}

function renderGpu(gpu) {
  if (!gpu) {
    setText("gpu-live", "UNAVAILABLE");
    return;
  }
  const utilization = clamp(Number(gpu.utilization_percent) / 100);
  const memory = clamp(Number(gpu.memory_used_mib) / Math.max(1, Number(gpu.memory_total_mib)));
  setText("gpu-live", "LIVE");
  setText("gpu-name", gpu.name);
  setText("gpu-pci", `PCI ${gpu.pci_bus_id}`);
  setText("gpu-util", `${Math.round(utilization * 100)}%`);
  setText("vram-util", `${Math.round(memory * 100)}%`);
  setText("vram-detail", `${(gpu.memory_used_mib / 1024).toFixed(1)} / ${(gpu.memory_total_mib / 1024).toFixed(1)} GiB`);
  setText("gpu-temp", gpu.temperature_c === null ? "—" : `${gpu.temperature_c.toFixed(0)}°C`);
  setText("gpu-power", gpu.power_w === null ? "—" : `${gpu.power_w.toFixed(0)} / ${gpu.power_limit_w.toFixed(0)} W`);
  byId("gpu-dial").style.setProperty("--value", `${utilization * 100}%`);
  byId("vram-dial").style.setProperty("--value", `${memory * 100}%`);
}

function renderPreparation(preparation, storage, service) {
  const prep = preparation || {};
  const progress = clamp(prep.progress_fraction);
  const staleRunning = prep.status === "running" && service?.active_state !== "active";
  setText("prep-status", staleRunning ? "PAUSED" : String(prep.status || "waiting").toUpperCase());
  setText("prep-completed", prep.completed === undefined ? "—" : number.format(prep.completed));
  setText("prep-processed", prep.processed === undefined ? "—" : number.format(prep.processed));
  setText("prep-failed", prep.failed === undefined ? "—" : number.format(prep.failed));
  setText("prep-remaining", prep.remaining === undefined ? "—" : number.format(prep.remaining));
  setText("prep-rate", prep.rate_per_minute === undefined ? "—" : `${prep.rate_per_minute.toFixed(1)}/min`);
  setText("storage-free", storage ? bytes(storage.free_bytes) : "—");
  byId("prep-progress").style.width = `${progress * 100}%`;
}

function renderTraining(training) {
  const run = training || {};
  const validation = run.best_validation || run.validation || {};
  const recent = Array.isArray(run.recent_losses) ? run.recent_losses : [];
  const latestRecordedLoss = recent.length ? recent.at(-1).loss : null;
  const displayedLoss = run.current_loss ?? latestRecordedLoss;
  setText("loss-phase", String(run.phase || "no active run").toUpperCase());
  setText("current-loss", displayedLoss === null || displayedLoss === undefined ? "—" : Number(displayedLoss).toFixed(5));
  setText("mean-loss", run.mean_loss === null || run.mean_loss === undefined ? "—" : Number(run.mean_loss).toFixed(5));
  setText("epoch-value", run.epoch ? `${run.epoch} / ${run.epochs}` : "—");
  setText("batch-value", run.batch ?? "—");
  setText("global-batch", run.global_batch ?? "—");
  setText("macro-iou", metricPercent(validation.surface_macro_iou));
  setText("boundary-f1", metricPercent(validation.boundary_f1));
  setText("supported-iou", metricPercent(validation.supported_surface_macro_iou));
  setText("best-epoch", run.best_epoch || "—");
  setText("training-samples", run.training_samples === undefined ? "— training samples" : `${number.format(run.training_samples)} training samples`);
  const partition = run.partition || {};
  const rotation = run.rotation_augmentation === undefined
    ? "—"
    : (run.rotation_augmentation ? "ON" : "OFF");
  const cache = run.sample_cache || {};
  const cacheRequests = Number(cache.hits || 0) + Number(cache.misses || 0);
  const cacheHitRate = cacheRequests ? `${(Number(cache.hits || 0) / cacheRequests * 100).toFixed(0)}% hit` : "warming";
  const cacheLabel = Number(cache.max_bytes || 0) > 0
    ? `${bytes(cache.current_bytes)} cache · ${cacheHitRate}`
    : "cache OFF";
  setText(
    "training-config",
    `AMP ${run.amp ? "ON" : "OFF"} · rotation ${rotation} · select ${run.selection_metric || "legacy"} · prefetch ${partition.prefetch_batches ?? "—"} · ${cacheLabel}`,
  );
  setText("training-elapsed", `Elapsed ${duration(run.elapsed_seconds)}`);
  renderLossChart(recent);
  if (run.phase && run.phase !== lastTrainingPhase) {
    addEvent(`Training phase changed to ${run.phase}.`, run.updated_at);
    lastTrainingPhase = run.phase;
  }
}

function renderOperation(operation, preparation, training) {
  const card = byId("operation-card");
  const current = operation || { kind: "idle", label: "Ready", state: "idle" };
  card.dataset.state = current.state;
  setText("operation-state", String(current.state).toUpperCase());
  setText("operation-label", current.label);
  let progress = 0;
  let detail = "No background job is active.";
  let eta = "ETA —";
  if (current.kind === "preparation" && preparation) {
    progress = clamp(preparation.progress_fraction);
    detail = `${number.format(preparation.completed || 0)} graphs ready · ${number.format(preparation.failed || 0)} rejected`;
    eta = `ETA ${duration(preparation.eta_seconds)}`;
  } else if (current.kind === "training" && training) {
    const epoch = Number(training.epoch) || 0;
    const epochs = Math.max(1, Number(training.epochs) || 1);
    const batches = Number(training.batch) || 0;
    const batchEstimate = Number(training.batches_per_epoch_estimate) || 0;
    const completedEpochs = Math.max(0, epoch - 1);
    progress = batchEstimate > 0
      ? clamp((completedEpochs + clamp(batches / batchEstimate)) / epochs)
      : clamp(epoch / epochs);
    const evaluation = training.evaluation_progress || {};
    if (training.phase === "validation" || training.phase === "test") {
      const label = training.phase === "test" ? "Held-out test" : "Validation";
      detail = `${label} · ${number.format(evaluation.completed || 0)} / ${number.format(evaluation.total || 0)} samples`;
      if (training.phase === "test") progress = 1;
    } else {
      const longPreparation = current.label === "Preparing large mesh batch";
      detail = longPreparation
        ? `Epoch ${training.epoch || 0} · after batch ${training.batch || 0} · uncapped decode / partition`
        : `Epoch ${training.epoch || 0} of ${training.epochs || "—"} · batch ${training.batch || 0}`;
    }
    const recent = Array.isArray(training.recent_losses) ? training.recent_losses : [];
    const displayedLoss = training.current_loss ?? (recent.length ? recent.at(-1).loss : null);
    eta = `Loss ${displayedLoss === null || displayedLoss === undefined ? "—" : Number(displayedLoss).toFixed(5)}`;
  }
  setText("operation-detail", detail);
  setText("operation-percent", percent(progress, 2));
  setText("operation-eta", eta);
  byId("operation-progress").style.width = `${progress * 100}%`;
  if (current.kind !== lastOperation) {
    addEvent(`Active operation: ${current.label}.`);
    lastOperation = current.kind;
  }
}

function render(snapshot) {
  renderGpu(snapshot.gpu);
  renderPreparation(snapshot.preparation, snapshot.storage, snapshot.services?.preparation);
  renderTraining(snapshot.training);
  renderOperation(snapshot.operation, snapshot.preparation, snapshot.training);
  const updated = new Date(snapshot.server_time);
  setText("last-update", updated.toLocaleTimeString());
}

const connection = byId("connection");
const stream = new EventSource("/api/dashboard/stream");
stream.addEventListener("open", () => {
  connection.className = "connection live";
  connection.querySelector("span").textContent = "Live stream connected";
  addEvent("Live telemetry stream connected.");
});
stream.addEventListener("snapshot", (event) => {
  try { render(JSON.parse(event.data)); } catch (error) { console.error(error); }
});
stream.addEventListener("error", () => {
  connection.className = "connection failed";
  connection.querySelector("span").textContent = "Reconnecting to stream";
});

window.addEventListener("resize", () => {
  fetch("/api/dashboard").then((response) => response.json()).then(render).catch(() => {});
});
