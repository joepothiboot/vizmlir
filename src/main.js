import { MlirEngine } from "./wasm/bridge.js";
import { CanvasRenderer } from "./render/canvas-renderer.js";
import { ABI_VERSION, STATUS } from "./wasm/abi.js";
import { copySnapshot, diffSnapshots } from "./diff.js";
import { bindHighlighting } from "./mlir-highlight.js";
import { CommandPalette } from "./palette.js";
import {
  baselineFor,
  describeEvent,
  isPassTrace,
  parsePassTrace,
} from "./trace.js";

const DIFF_GLYPH = { added: "+", removed: "−", changed: "~" };

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
const statusEl = document.getElementById("status-text");
const modeEl = document.getElementById("mode");
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
const passStrip = document.getElementById("pass-strip");
const fileInput = document.getElementById("file");
const baselineTitle = document.getElementById("baseline-title");
const currentTitle = document.getElementById("current-title");
const sourcePane = document.getElementById("source-pane");
const splitToggle = document.getElementById("split-toggle");
const sourceName = document.getElementById("source-name");
const diffTitle = document.getElementById("diff-title");
const helpDialog = document.getElementById("help");
const TABS = [
  { tab: baselineTitle, textarea: baseline },
  { tab: currentTitle, textarea: input },
];

let trace = null;
let traceNote = "";
let traceIndex = -1;
let diffRows = [];
let diffCursor = -1;
let traceDiffTitle = "";

const renderer = new CanvasRenderer(canvas, {
  onSelect(index, snap) {
    const parent = index < 0 ? -1 : snap.parentOf(index);
    detailEl.textContent =
      index < 0
        ? "—"
        : `#${index} ${snap.labelOf(index)}` +
          (parent < 0 ? "" : ` · parent ${snap.labelOf(parent)}`);
    syncDiffSelection(index);
    markSourceLine(index);
  },
});

const engine = await MlirEngine.load(
  `${import.meta.env.BASE_URL}mlir_core.wasm`,
);

function setStatus(text, { error = false } = {}) {
  statusEl.textContent = text;
  modeEl.textContent = error ? "ERROR" : trace ? "TRACE" : "IR";
  modeEl.classList.toggle("bad", error);
}

function parse(text) {
  const t0 = performance.now();
  const status = engine.parse(text);
  return {
    status,
    ms: performance.now() - t0,
    snapshot: status === STATUS.OK ? engine.snapshot() : null,
  };
}

function parentLabel(snapshot, index) {
  if (!snapshot || index < 0) return "";
  return snapshot.nodes
    ? (snapshot.nodes[index]?.label ?? "")
    : snapshot.labelOf(index);
}

function renderDiff(rows, before, after) {
  diffTitle.textContent = traceDiffTitle || "diff baseline → current";
  const counts = rows.reduce(
    (result, row) => {
      result[row.type] += 1;
      return result;
    },
    { added: 0, removed: 0, changed: 0 },
  );
  const total = counts.added + counts.removed + counts.changed;
  if (total) {
    diffSummary.replaceChildren(
      ...["added", "removed", "changed"].flatMap((type) => {
        const span = document.createElement("span");
        span.className = type;
        span.textContent = `${DIFF_GLYPH[type]}${counts[type]}`;
        return [span, " "];
      }),
    );
  } else {
    diffSummary.textContent = "No structural changes";
  }

  diffRows = rows;
  diffCursor = -1;
  diffList.replaceChildren(
    ...rows.map((row, index) => {
      const item = document.createElement("li");
      item.className = row.type;
      item.setAttribute("role", "option");
      item.setAttribute("aria-selected", "false");
      const glyph = document.createElement("span");
      glyph.className = "glyph";
      glyph.textContent = DIFF_GLYPH[row.type];
      const op = row.after ?? row.before;
      const body = document.createElement("span");
      body.className = "body";
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = op.label;
      const sub = document.createElement("span");
      sub.className = "sub";
      const parent = parentLabel(row.after ? after : before, op.parent);
      sub.textContent =
        row.type === "changed"
          ? `was ${row.before.label}`
          : parent
            ? `in ${parent}`
            : "top level";
      body.append(label, sub);
      item.append(glyph, body);
      item.addEventListener("click", () => focusChange(index));
      return item;
    }),
  );
  renderer.setMarks(
    new Map(
      rows.filter((row) => row.after).map((row) => [row.after.index, row.type]),
    ),
  );
}

function syncDiffSelection(nodeIndex) {
  diffCursor = diffRows.findIndex((row) => row.after?.index === nodeIndex);
  [...diffList.children].forEach((item, i) =>
    item.setAttribute("aria-selected", String(i === diffCursor)),
  );
  diffList.children[diffCursor]?.scrollIntoView({ block: "nearest" });
}

function focusChange(index) {
  if (!diffRows.length) return;
  diffCursor = (index + diffRows.length) % diffRows.length;
  [...diffList.children].forEach((item, i) =>
    item.setAttribute("aria-selected", String(i === diffCursor)),
  );
  diffList.children[diffCursor].scrollIntoView({ block: "nearest" });
  const row = diffRows[diffCursor];
  // Removed ops no longer exist in the current graph, so there is nothing to select.
  if (row.after) renderer.select(row.after.index, { center: true });
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
      setStatus(`baseline error: ${engine.statusText}`, { error: true });
      return;
    }
    beforeCopy = copySnapshot(before.snapshot);
  }

  const after = parse(input.value);
  if (after.status !== STATUS.OK) {
    setStatus(`current error: ${engine.statusText}`, { error: true });
    renderer.setSnapshot(null);
    return;
  }

  const snap = after.snapshot;
  renderer.setSnapshot(snap);
  const diags = snap.diagnostics();
  const rows = diffSnapshots(beforeCopy, snap);
  renderDiff(rows, beforeCopy, snap);
  setStatus(
    traceNote +
      `${snap.nodeCount} nodes · ${snap.edgeCount} edges · ${after.ms.toFixed(2)} ms` +
      (diags.length
        ? ` · ${diags.length} warning(s): ${diags[0].message} "${diags[0].symbol}"`
        : ""),
  );
}

function loadTrace(text) {
  trace = parsePassTrace(text);
  if (!trace.events.length) {
    clearTrace();
    setStatus("no IR dumps found in trace", { error: true });
    return;
  }
  const width = String(trace.events.length).length;
  passStrip.replaceChildren(
    ...trace.events.map((event) => {
      const button = document.createElement("button");
      const flags = `${event.failed ? " ✗ failed" : ""}${event.diagnostics.length ? ` · ${event.diagnostics.length} diag` : ""}`;
      const number = document.createElement("span");
      number.className = "n";
      number.textContent =
        String(event.index + 1).padStart(Math.max(2, width), "0") + flags;
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = describeEvent(event);
      button.append(number, name);
      button.title = `${event.index + 1}. ${describeEvent(event)}${flags}`;
      button.classList.toggle("failed", event.failed);
      button.addEventListener("click", () => selectEvent(event.index));
      return button;
    }),
  );
  passStrip.hidden = false;
  const firstFailure = trace.events.findIndex((event) => event.failed);
  selectEvent(firstFailure >= 0 ? firstFailure : 0);
}

function selectEvent(index) {
  const event = trace.events[index];
  const base = baselineFor(trace.events, index);
  traceIndex = index;
  [...passStrip.children].forEach((button, i) => {
    if (i === index) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  passStrip.children[index]?.scrollIntoView({
    block: "nearest",
    inline: "nearest",
  });
  baseline.value = base?.ir ?? "";
  input.value = event.ir;
  baseline.dispatchEvent(new Event("input"));
  input.dispatchEvent(new Event("input"));
  clearTimeout(timer);

  baselineTitle.textContent = base
    ? `Baseline · ${base.reconstructed ? "rebuilt through" : "from"} #${base.event.index + 1}`
    : "Baseline · no earlier snapshot";
  currentTitle.textContent = `Current · #${index + 1} ${describeEvent(event)}`;
  traceDiffTitle = `diff ${base ? `#${base.event.index + 1}` : "∅"} → #${index + 1}`;
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
  traceIndex = -1;
  traceDiffTitle = "";
  passStrip.hidden = true;
  passStrip.replaceChildren();
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
  passStrip.hidden = isDocs || !trace;
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

function fitGraph() {
  renderer.fit();
  renderer.requestDraw();
}
document.getElementById("fit")?.addEventListener("click", fitGraph);

// ---- Source tabs -----------------------------------------------------------

function showTab(textarea) {
  for (const entry of TABS) {
    const on = entry.textarea === textarea;
    entry.tab.setAttribute("aria-selected", String(on));
    entry.textarea.parentElement.hidden = !on;
  }
}

function toggleTab() {
  showTab(
    currentTitle.getAttribute("aria-selected") === "true" ? baseline : input,
  );
}

function toggleSplit() {
  const on = sourcePane.classList.toggle("split");
  splitToggle.setAttribute("aria-pressed", String(on));
}

for (const { tab, textarea } of TABS)
  tab.addEventListener("click", () => showTab(textarea));
splitToggle.addEventListener("click", toggleSplit);

// ---- Source line marker ----------------------------------------------------

const lineMark = input.parentElement.querySelector(".line-mark");
let markedLine = -1;

function positionLineMark() {
  if (markedLine < 0) {
    lineMark.hidden = true;
    return;
  }
  const style = getComputedStyle(input);
  const lineHeight = parseFloat(style.lineHeight);
  const top = parseFloat(style.paddingTop) + markedLine * lineHeight;
  lineMark.style.top = `${top - input.scrollTop}px`;
  lineMark.style.height = `${lineHeight}px`;
  lineMark.hidden = false;
}

// Nodes carry no source locations, so find the n-th line mentioning the op:
// node order follows source order, so the n-th node with this op name is
// (nearly always) on the n-th line that names it.
function markSourceLine(index) {
  markedLine = -1;
  const snap = renderer.snapshot;
  if (snap && index >= 0) {
    const op = snap.labelOf(index).split(" ")[0];
    let nth = 0;
    for (let i = 0; i < index; i++)
      if (snap.labelOf(i).split(" ")[0] === op) nth++;
    const escaped = op.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = new RegExp(`(^|[^\\w.])${escaped}([^\\w.]|$)`);
    const lines = input.value.split("\n");
    markedLine = lines.findIndex((line) => pattern.test(line) && nth-- === 0);
  }
  if (markedLine >= 0) {
    if (!sourcePane.classList.contains("split")) showTab(input);
    const lineHeight = parseFloat(getComputedStyle(input).lineHeight);
    const y = markedLine * lineHeight;
    if (y < input.scrollTop || y > input.scrollTop + input.clientHeight - 40)
      input.scrollTop = Math.max(0, y - input.clientHeight / 3);
  }
  positionLineMark();
}

// Hand edits mark the loaded source as modified, like an editor tab.
for (const textarea of [baseline, input])
  textarea.addEventListener("input", (e) => {
    if (e.isTrusted && !sourceName.textContent.endsWith(" ●"))
      sourceName.textContent += " ●";
  });

input.addEventListener("scroll", positionLineMark);
input.addEventListener("input", () => {
  markedLine = -1;
  positionLineMark();
});

// ---- Command palette -------------------------------------------------------

function goToWorkspace() {
  if (workspace.hidden) window.location.hash = "#/";
}

const palette = new CommandPalette(document.getElementById("palette"), () => {
  const items = [];
  if (trace)
    for (const event of trace.events)
      items.push({
        group: "pass",
        text: `${event.index + 1}. ${describeEvent(event)}`,
        hint: event.failed ? "✗ failed" : "",
        run: () => {
          goToWorkspace();
          selectEvent(event.index);
        },
      });
  const snap = renderer.snapshot;
  for (let i = 0; snap && i < snap.nodeCount; i++) {
    const label = snap.labelOf(i);
    const parent = snap.parentOf(i);
    items.push({
      group: label.includes("@") ? "symbol" : "op",
      text: label,
      hint: `#${i}${parent < 0 ? "" : ` · in ${snap.labelOf(parent)}`}`,
      run: () => {
        goToWorkspace();
        renderer.select(i, { center: true });
      },
    });
  }
  const actions = [
    ["Open file…", "", () => fileInput.click()],
    ["Compare", "", run],
    ["Fit graph", "f", fitGraph],
    ["Switch baseline / current", "t", toggleTab],
    ["Toggle split sources", "s", toggleSplit],
    ["Load sample", "", loadSample],
    ["Open docs", "", () => (window.location.hash = "#/docs")],
    ["Keyboard shortcuts", "?", () => helpDialog.showModal()],
  ];
  for (const [text, hint, action] of actions)
    items.push({ group: "action", text, hint, run: action });
  return items;
});

const isMac = /Mac|iPhone|iPad/.test(navigator.userAgent);
document
  .querySelectorAll(".mod-k")
  .forEach((kbd) => (kbd.textContent = isMac ? "⌘K" : "Ctrl K"));
document
  .getElementById("palette-open")
  .addEventListener("click", () => palette.open());
document
  .getElementById("help-open")
  .addEventListener("click", () => helpDialog.showModal());

// ---- Keyboard --------------------------------------------------------------

function isTyping(target) {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable ||
      ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
  );
}

const WORKSPACE_KEYS = {
  f: fitGraph,
  j: () => focusChange(diffCursor + 1),
  k: () => focusChange(diffCursor < 0 ? -1 : diffCursor - 1),
  t: toggleTab,
  s: toggleSplit,
  "[": () => trace && traceIndex > 0 && selectEvent(traceIndex - 1),
  "]": () =>
    trace &&
    traceIndex < trace.events.length - 1 &&
    selectEvent(traceIndex + 1),
};

// Keyboard-first navigation. Escape leaves an editor so the shortcuts apply.
window.addEventListener("keydown", (e) => {
  if (document.querySelector("dialog[open]")) return;
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    palette.open();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "Escape" && isTyping(e.target)) {
    e.target.blur();
    return;
  }
  if (isTyping(e.target)) return;
  if (e.key === "/") {
    e.preventDefault();
    palette.open();
  } else if (e.key === "?") {
    helpDialog.showModal();
  } else if (!workspace.hidden && WORKSPACE_KEYS[e.key]) {
    e.preventDefault();
    WORKSPACE_KEYS[e.key]();
  }
});

// ---- Loading ---------------------------------------------------------------

fileInput.addEventListener("change", async () => {
  const [file] = fileInput.files;
  fileInput.value = "";
  if (!file) return;
  const text = await file.text();
  sourceName.textContent = file.name;
  sourceName.title = file.name;
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

function loadSample() {
  clearTrace();
  sourceName.textContent = "sample.mlir";
  baseline.value = SAMPLE;
  input.value = SAMPLE.replace("linalg.fill", "linalg.fill_relu");
  baseline.dispatchEvent(new Event("input"));
  input.dispatchEvent(new Event("input"));
  window.location.hash = "#/";
  run();
}
document.getElementById("load-sample")?.addEventListener("click", loadSample);

document.getElementById("abi").textContent = `wasm abi v${ABI_VERSION}`;
showTab(input);
window.addEventListener("hashchange", updateRoute);
updateRoute();
run();
