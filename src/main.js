import { MlirEngine } from "./wasm/bridge.js";
import { CanvasRenderer } from "./render/canvas-renderer.js";
import { STATUS } from "./wasm/abi.js";
import { copySnapshot, diffSnapshots } from "./diff.js";
import { bindHighlighting } from "./mlir-highlight.js";
import {
  baselineFor,
  describeEvent,
  isPassTrace,
  parsePassTrace,
} from "./trace.js";

const SAMPLE = `module {
  func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>) -> tensor<128x64xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128x64xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
    %out = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>)
                         outs(%filled : tensor<128x64xf32>) -> tensor<128x64xf32>
    func.return %out : tensor<128x64xf32>
  }
}`;

const canvas = document.getElementById("canvas");
const input = document.getElementById("editor");
const baseline = document.getElementById("baseline");
const statusEl = document.getElementById("status");
const detailEl = document.getElementById("detail");
const diffSummary = document.getElementById("diff-summary");
const diffList = document.getElementById("diff-list");
const workspace = document.getElementById("workspace");
const docsView = document.getElementById("docs-view");
const routeLinks = document.querySelectorAll("[data-route]");
const docsSample = document.getElementById("docs-sample");
const baselineHighlight = document.getElementById("baseline-highlight");
const editorHighlight = document.getElementById("editor-highlight");
const docsSampleHighlight = document.getElementById("docs-sample-highlight");
const diagsEl = document.getElementById("diags");
const passSelect = document.getElementById("pass-select");
const fileInput = document.getElementById("file");
const baselineTitle = document.getElementById("baseline-title");
const currentTitle = document.getElementById("current-title");

let trace = null;
let traceNote = "";

const renderer = new CanvasRenderer(canvas, {
  onSelect(index, snap) {
    detailEl.textContent =
      index < 0
        ? "—"
        : `#${index} · ${snap.labelOf(index)} · parent ${snap.parentOf(index)}`;
  },
});

const engine = await MlirEngine.load(
  `${import.meta.env.BASE_URL}mlir_core.wasm`,
);

function parse(text) {
  const t0 = performance.now();
  const status = engine.parse(text);
  return {
    status,
    ms: performance.now() - t0,
    snapshot: status === STATUS.OK ? engine.snapshot() : null,
  };
}

function renderDiff(rows) {
  const counts = rows.reduce(
    (result, row) => {
      result[row.type] += 1;
      return result;
    },
    { added: 0, removed: 0, changed: 0 },
  );
  const total = counts.added + counts.removed + counts.changed;
  diffSummary.textContent = total
    ? `${total} change(s): +${counts.added}  -${counts.removed}  ~${counts.changed}`
    : "No structural changes";
  diffList.replaceChildren(
    ...rows.map((row) => {
      const item = document.createElement("li");
      item.className = row.type;
      if (row.type === "changed")
        item.textContent = `~ ${row.before.label} -> ${row.after.label}`;
      if (row.type === "added") item.textContent = `+ ${row.after.label}`;
      if (row.type === "removed") item.textContent = `- ${row.before.label}`;
      return item;
    }),
  );
}

function run() {
  if (isPassTrace(input.value)) {
    loadTrace(input.value);
    return;
  }

  let beforeCopy = null;
  if (baseline.value.trim()) {
    const before = parse(baseline.value);
    if (before.status !== STATUS.OK) {
      statusEl.textContent = `baseline error: ${engine.statusText}`;
      return;
    }
    beforeCopy = copySnapshot(before.snapshot);
  }
  const after = parse(input.value);
  if (after.status !== STATUS.OK) {
    statusEl.textContent = `current error: ${engine.statusText}`;
    renderer.setSnapshot(null);
    return;
  }

  const snap = after.snapshot;
  renderer.setSnapshot(snap);
  const diags = snap.diagnostics();
  const rows = diffSnapshots(beforeCopy, snap);
  renderDiff(rows);
  statusEl.textContent =
    traceNote +
    `${snap.nodeCount} nodes · ${snap.edgeCount} edges · ${after.ms.toFixed(2)} ms` +
    (diags.length
      ? ` · ${diags.length} warning(s): ${diags[0].message} "${diags[0].symbol}"`
      : "");
}

function loadTrace(text) {
  trace = parsePassTrace(text);
  if (!trace.events.length) {
    clearTrace();
    statusEl.textContent = "no IR dumps found in trace";
    return;
  }
  passSelect.replaceChildren(
    ...trace.events.map((event) => {
      const option = document.createElement("option");
      option.value = String(event.index);
      const flags = `${event.failed ? " ✗ failed" : ""}${event.diagnostics.length ? ` · ${event.diagnostics.length} diag` : ""}`;
      option.textContent = `${event.index + 1}. ${describeEvent(event)}${flags}`;
      return option;
    }),
  );
  passSelect.hidden = false;
  const firstFailure = trace.events.findIndex((event) => event.failed);
  selectEvent(firstFailure >= 0 ? firstFailure : 0);
}

function selectEvent(index) {
  const event = trace.events[index];
  const base = baselineFor(trace.events, index);
  passSelect.value = String(index);
  baseline.value = base?.ir ?? "";
  input.value = event.ir;
  baseline.dispatchEvent(new Event("input"));
  input.dispatchEvent(new Event("input"));
  clearTimeout(timer);

  baselineTitle.textContent = base
    ? `Baseline · ${base.reconstructed ? "rebuilt through" : "from"} #${base.event.index + 1}`
    : "Baseline · no earlier snapshot";
  currentTitle.textContent = `Current · #${index + 1} ${describeEvent(event)}`;
  traceNote = `pass ${index + 1}/${trace.events.length}${event.failed ? " · ✗ FAILED" : ""} · trace line ${event.headerLine} · `;
  renderDiagnostics(
    event.index === trace.events.length - 1
      ? [
          ...event.diagnostics,
          ...trace.diagnostics.filter((d) => d.eventIndex < 0),
        ]
      : event.diagnostics,
  );
  run();
}

function clearTrace() {
  trace = null;
  traceNote = "";
  passSelect.hidden = true;
  passSelect.replaceChildren();
  baselineTitle.textContent = "Baseline";
  currentTitle.textContent = "Current";
  renderDiagnostics([]);
}

function renderDiagnostics(diagnostics) {
  diagsEl.replaceChildren(
    ...diagnostics.map((diagnostic) => {
      const item = document.createElement("div");
      const { file, line, column } = diagnostic.location;
      item.className = diagnostic.severity;
      item.textContent =
        `${file.split("/").at(-1)}:${line}${column ? `:${column}` : ""}: ${diagnostic.severity}: ${diagnostic.message}` +
        (diagnostic.eventIndex < 0 ? "  (after last dump)" : "");
      item.title = diagnostic.detail;
      return item;
    }),
  );
}

function updateRoute() {
  const isDocs = window.location.hash === "#/docs";
  workspace.hidden = isDocs;
  docsView.hidden = !isDocs;
  routeLinks.forEach((link) => {
    link.setAttribute(
      "aria-current",
      isDocs === (link.dataset.route === "docs") ? "page" : "false",
    );
  });
}

let timer = 0;
baseline.value = SAMPLE;
input.value = SAMPLE.replace("linalg.fill", "linalg.fill_relu");
docsSample.value = SAMPLE;
bindHighlighting(baseline, baselineHighlight);
bindHighlighting(input, editorHighlight);
bindHighlighting(docsSample, docsSampleHighlight);
function scheduleRun() {
  clearTimeout(timer);
  timer = setTimeout(run, 140);
}
baseline.addEventListener("input", scheduleRun);
input.addEventListener("input", scheduleRun);
document.getElementById("reparse")?.addEventListener("click", run);

document.getElementById("fit")?.addEventListener("click", () => {
  renderer.fit();
  renderer.requestDraw();
});

passSelect.addEventListener("change", () =>
  selectEvent(Number(passSelect.value)),
);

fileInput.addEventListener("change", async () => {
  const [file] = fileInput.files;
  fileInput.value = "";
  if (!file) return;
  const text = await file.text();
  window.location.hash = "#/";
  if (isPassTrace(text)) {
    loadTrace(text);
    return;
  }
  clearTrace();
  input.value = text;
  input.dispatchEvent(new Event("input"));
  run();
});

document.getElementById("load-sample")?.addEventListener("click", () => {
  clearTrace();
  baseline.value = SAMPLE;
  input.value = SAMPLE.replace("linalg.fill", "linalg.fill_relu");
  window.location.hash = "#/";
  run();
});

window.addEventListener("hashchange", updateRoute);
updateRoute();
run();
