import { MlirEngine } from "./wasm/bridge.js";
import { CanvasRenderer } from "./render/canvas-renderer.js";
import { ABI_VERSION, STATUS } from "./wasm/abi.js";
import { copySnapshot, diffSnapshots } from "./diff.js";
import { bindHighlighting } from "./mlir-highlight.js";
import { CommandPalette } from "./palette.js";
import { loadSampleState, RENAME_SAMPLE, SAMPLES } from "./samples.js";
import { bindSplitters } from "./splitters.js";
import { kv, sessionFromFile, sessions, sessionToFile } from "./storage.js";
import { canWatchFiles, FileWatcher } from "./watch.js";
import {
  diffRecords,
  diffToJSON,
  diffToMarkdown,
  diffToPatch,
  download,
  slug,
} from "./export.js";
import {
  baselineFor,
  describeEvent,
  isPassTrace,
  parsePassTrace,
} from "./trace.js";

const DIFF_GLYPH = { added: "+", removed: "−", changed: "~" };

const canvas = document.getElementById("canvas");
const input = document.getElementById("editor");
const baseline = document.getElementById("baseline");
const statusEl = document.getElementById("status-text");
const modeEl = document.getElementById("mode");
const detailEl = document.getElementById("detail");
const diffSummary = document.getElementById("diff-summary");
const diffList = document.getElementById("diff-list");
const diffCopy = document.getElementById("diff-copy");
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
const zoomLevel = document.getElementById("zoom-level");
const TABS = [
  { tab: baselineTitle, textarea: baseline },
  { tab: currentTitle, textarea: input },
];

let trace = null;
let traceText = "";
let traceNote = "";
let traceIndex = -1;
let diffRows = [];
let diffBefore = null;
let diffAfter = null;
let diffCursor = -1;
let diffPicked = new Set();
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
  onViewChange(scale) {
    zoomLevel.textContent = `${Math.round(scale * 100)}%`;
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
  diffBefore = before;
  diffAfter = after;
  diffCursor = -1;
  diffPicked = new Set();
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
      item.setAttribute("aria-checked", "false");
      item.addEventListener("click", (e) => {
        if (e.shiftKey && diffCursor >= 0) pickRange(diffCursor, index);
        else if (e.metaKey || e.ctrlKey) togglePick(index);
        focusChange(index);
      });
      return item;
    }),
  );
  syncPicks();
  renderer.setMarks(
    new Map(
      rows.filter((row) => row.after).map((row) => [row.after.index, row.type]),
    ),
  );
}

// Picked rows are what "copy" takes; with nothing picked it takes the cursor row.
function togglePick(index) {
  if (!diffPicked.delete(index)) diffPicked.add(index);
  syncPicks();
}

function pickRange(from, to) {
  for (let i = Math.min(from, to); i <= Math.max(from, to); i++)
    diffPicked.add(i);
  syncPicks();
}

function clearPicks() {
  if (!diffPicked.size) return false;
  diffPicked.clear();
  syncPicks();
  return true;
}

function syncPicks() {
  [...diffList.children].forEach((item, i) =>
    item.setAttribute("aria-checked", String(diffPicked.has(i))),
  );
  diffCopy.textContent = diffPicked.size ? `copy ${diffPicked.size}` : "copy";
  diffCopy.disabled = !diffRows.length;
}

async function copyChanges() {
  const indices = diffPicked.size
    ? [...diffPicked].sort((a, b) => a - b)
    : diffCursor >= 0
      ? [diffCursor]
      : diffRows.map((_, i) => i);
  if (!indices.length) return;
  const rows = indices.map((i) => diffRows[i]);
  const records = diffRecords(rows, (row, parent) =>
    parentLabel(row.after ? diffAfter : diffBefore, parent),
  );
  try {
    await navigator.clipboard.writeText(
      diffToPatch(diffTitle.textContent, records),
    );
    setStatus(
      `copied ${rows.length} change${rows.length === 1 ? "" : "s"} as a patch`,
    );
  } catch {
    setStatus("clipboard unavailable", { error: true });
  }
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
  scheduleAutosave();
  if (isPassTrace(input.value)) {
    // Files and watched files load traces directly; this path is a paste.
    sourceName.textContent = "pasted trace";
    sourceName.title = sourceName.textContent;
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

function loadTrace(text, { keepIndex = false } = {}) {
  const previousIndex = traceIndex;
  traceText = text;
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
  selectEvent(
    keepIndex && previousIndex >= 0
      ? Math.min(previousIndex, trace.events.length - 1)
      : firstFailure >= 0
        ? firstFailure
        : 0,
  );
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
  traceText = "";
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
docsSample.value = RENAME_SAMPLE;
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
document.getElementById("zoom-fit").addEventListener("click", fitGraph);
document
  .getElementById("zoom-in")
  .addEventListener("click", () => renderer.zoomBy(1.25));
document
  .getElementById("zoom-out")
  .addEventListener("click", () => renderer.zoomBy(0.8));

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

// ---- Theme -----------------------------------------------------------------

const THEME_KEY = "vizmlir-theme";
const themeToggle = document.getElementById("theme-toggle");
const systemLight = matchMedia("(prefers-color-scheme: light)");

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const next = theme === "dark" ? "light" : "dark";
  themeToggle.setAttribute("aria-label", `Switch to ${next} theme`);
  themeToggle.title = `Switch to ${next} theme (Shift+L)`;
  renderer.refreshTheme();
}

function toggleTheme() {
  const theme =
    document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Storage can be unavailable (private mode); the toggle still works.
  }
  applyTheme(theme);
}

// Follow the OS setting until the user picks a theme explicitly.
systemLight.addEventListener("change", (e) => {
  let saved = null;
  try {
    saved = localStorage.getItem(THEME_KEY);
  } catch {}
  if (!saved) applyTheme(e.matches ? "light" : "dark");
});
themeToggle.addEventListener("click", toggleTheme);
applyTheme(document.documentElement.dataset.theme || "dark");

// ---- Pane splitters -------------------------------------------------------

bindSplitters(document.querySelector("#workspace main"), [
  {
    el: document.getElementById("source-splitter"),
    pane: document.getElementById("source-pane"),
    side: "left",
  },
  {
    el: document.getElementById("diff-splitter"),
    pane: document.getElementById("diff-pane"),
    side: "right",
  },
]);

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
    ["Browse samples…", "", openSamples],
    ["Sessions: save, open, import", "", openSessions],
    ["Download session .json", "", downloadSession],
    ["Export graph as PNG", "", exportPNG],
    ["Export graph as SVG", "", exportSVG],
    ["Copy changes as patch", "c", copyChanges],
    ["Export diff as Markdown", "", () => exportDiff("md")],
    ["Export diff as JSON", "", () => exportDiff("json")],
    ...(canWatchFiles
      ? [
          watcher.watching
            ? ["Stop watching file", "", stopWatching]
            : ["Watch a file for changes…", "", pickWatch],
        ]
      : []),
    [
      "Show / hide menu",
      "m",
      () => topbar.togglePopover(),
    ],
    ["Switch light / dark theme", "shift L", toggleTheme],
    ["Open docs", "", () => (window.location.hash = "#/docs")],
    ["Keyboard shortcuts", "?", () => helpDialog.showModal()],
  ];
  for (const [text, hint, action] of actions)
    items.push({ group: "action", text, hint, run: action });
  for (const sample of SAMPLES)
    items.push({
      group: "sample",
      text: sample.title,
      hint: sample.trace ? "trace" : "diff",
      run: () => openSample(sample),
    });
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

// The menu is an overlay: close it once an action runs, but keep it open while
// switching themes so the change is easy to compare.
const topbar = document.getElementById("topbar");
topbar.addEventListener("click", (e) => {
  const control = e.target.closest("a, button");
  if (control && control.id !== "theme-toggle") topbar.hidePopover();
});

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
  "+": () => renderer.zoomBy(1.25),
  "=": () => renderer.zoomBy(1.25),
  "-": () => renderer.zoomBy(0.8),
  x: () => diffCursor >= 0 && togglePick(diffCursor),
  c: copyChanges,
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
  if (e.key === "Escape" && !workspace.hidden && clearPicks()) return;
  if (e.key === "/") {
    e.preventDefault();
    palette.open();
  } else if (e.key === "?") {
    helpDialog.showModal();
  } else if (e.key === "L") {
    toggleTheme();
  } else if (e.key === "m") {
    topbar.togglePopover();
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
  window.location.hash = "#/";
  const session = file.name.endsWith(".json") ? sessionFromFile(text) : null;
  if (session) {
    sessionsDialog.close();
    applyState(session.state);
    return;
  }
  sourceName.textContent = file.name;
  sourceName.title = file.name;
  if (isPassTrace(text)) {
    loadTrace(text);
    return;
  }
  clearTrace();
  input.value = text;
  input.dispatchEvent(new Event("input"));
  run();
});

// ---- Samples -----------------------------------------------------------------

const samplesDialog = document.getElementById("samples");
const sampleList = document.getElementById("sample-list");

async function fetchSampleText(path) {
  const response = await fetch(`${import.meta.env.BASE_URL}samples/${path}`);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.text();
}

async function openSample(sample) {
  samplesDialog.close();
  goToWorkspace();
  try {
    applyState(await loadSampleState(sample, fetchSampleText));
  } catch (error) {
    setStatus(`could not load sample: ${error.message}`, { error: true });
  }
}

sampleList.replaceChildren(
  ...SAMPLES.map((sample) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    const title = document.createElement("span");
    title.className = "title";
    title.textContent = sample.title;
    const kind = document.createElement("span");
    kind.className = "kind";
    kind.textContent = sample.trace ? "pass trace" : "before / after";
    const blurb = document.createElement("span");
    blurb.className = "blurb";
    blurb.textContent = sample.blurb;
    button.append(title, kind, blurb);
    button.addEventListener("click", () => openSample(sample));
    item.append(button);
    return item;
  }),
);

function openSamples() {
  samplesDialog.showModal();
}

document.getElementById("samples-open").addEventListener("click", openSamples);
document.getElementById("docs-samples").addEventListener("click", openSamples);
document.getElementById("load-sample")?.addEventListener("click", () => {
  openSample(SAMPLES[0]);
});

// ---- Workspace state (autosave, sessions) --------------------------------

function getState() {
  return {
    sourceName: sourceName.textContent,
    trace: traceText || null,
    traceIndex,
    baseline: traceText ? "" : baseline.value,
    current: traceText ? "" : input.value,
    tab:
      currentTitle.getAttribute("aria-selected") === "true"
        ? "current"
        : "baseline",
    split: sourcePane.classList.contains("split"),
  };
}

function applyState(state) {
  clearTrace();
  sourceName.textContent = state.sourceName || "untitled";
  sourceName.title = sourceName.textContent;
  sourcePane.classList.toggle("split", !!state.split);
  splitToggle.setAttribute("aria-pressed", String(!!state.split));
  if (state.trace && isPassTrace(state.trace)) {
    traceIndex = state.traceIndex ?? -1;
    loadTrace(state.trace, { keepIndex: true });
  } else {
    baseline.value = state.baseline ?? "";
    input.value = state.current ?? "";
    baseline.dispatchEvent(new Event("input"));
    input.dispatchEvent(new Event("input"));
    clearTimeout(timer);
    run();
  }
  showTab(state.tab === "baseline" ? baseline : input);
}

let autosaveTimer = 0;
let restored = false;
function scheduleAutosave() {
  if (!restored) return;
  clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(() => kv.set("autosave", getState()), 800);
}

const sessionsDialog = document.getElementById("sessions");
const sessionList = document.getElementById("session-list");
const sessionName = document.getElementById("session-name");

async function renderSessions() {
  const list = await sessions.list();
  if (!list.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No saved sessions yet.";
    sessionList.replaceChildren(empty);
    return;
  }
  sessionList.replaceChildren(
    ...list.map((session) => {
      const item = document.createElement("li");
      const meta = document.createElement("span");
      meta.className = "meta";
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = session.name;
      const when = document.createElement("span");
      when.className = "when";
      when.textContent =
        new Date(session.savedAt).toLocaleString() +
        (session.state.trace ? " · trace" : "");
      meta.append(name, when);
      const actions = [
        [
          "Open",
          () => {
            sessionsDialog.close();
            goToWorkspace();
            applyState(session.state);
          },
        ],
        [
          "Download",
          () =>
            download(
              `${slug(session.name)}.vizmlir.json`,
              sessionToFile(session.name, session.state),
              "application/json",
            ),
        ],
        [
          "Delete",
          async () => {
            await sessions.delete(session.id);
            renderSessions();
          },
        ],
      ].map(([label, action]) => {
        const button = document.createElement("button");
        button.textContent = label;
        button.setAttribute("aria-label", `${label} ${session.name}`);
        button.addEventListener("click", action);
        return button;
      });
      item.append(meta, ...actions);
      return item;
    }),
  );
}

function openSessions() {
  sessionName.value = sourceName.textContent.replace(/ ●$/, "");
  renderSessions();
  sessionsDialog.showModal();
  sessionName.select();
}

document
  .getElementById("session-save")
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = sessionName.value.trim();
    if (!name) return;
    await sessions.put({
      id: crypto.randomUUID(),
      name,
      savedAt: Date.now(),
      state: getState(),
    });
    renderSessions();
  });

function downloadSession() {
  const name = sourceName.textContent.replace(/ ●$/, "");
  download(
    `${slug(name)}.vizmlir.json`,
    sessionToFile(name, getState()),
    "application/json",
  );
}

document
  .getElementById("sessions-open")
  .addEventListener("click", openSessions);
document
  .getElementById("session-import")
  .addEventListener("click", () => fileInput.click());
document
  .getElementById("session-download")
  .addEventListener("click", downloadSession);

// ---- Export ----------------------------------------------------------------

function exportStem() {
  const name = sourceName.textContent
    .replace(/ ●$/, "")
    .replace(/\.[^.]+$/, "");
  return slug(trace ? `${name}-pass-${traceIndex + 1}` : name);
}

async function exportPNG() {
  const blob = await renderer.exportPNG();
  if (blob) download(`${exportStem()}.png`, blob);
}

function exportSVG() {
  const svg = renderer.exportSVG();
  if (svg) download(`${exportStem()}.svg`, svg, "image/svg+xml");
}

function exportDiff(format) {
  const title = `${sourceName.textContent.replace(/ ●$/, "")} · ${diffTitle.textContent}`;
  const records = diffRecords(diffRows, (row, parent) =>
    parentLabel(row.after ? diffAfter : diffBefore, parent),
  );
  if (format === "md")
    download(
      `${exportStem()}-diff.md`,
      diffToMarkdown(title, records),
      "text/markdown",
    );
  else
    download(
      `${exportStem()}-diff.json`,
      diffToJSON(title, records),
      "application/json",
    );
}

document.getElementById("export-png").addEventListener("click", exportPNG);
document.getElementById("export-svg").addEventListener("click", exportSVG);
diffCopy.addEventListener("click", copyChanges);
document
  .getElementById("export-md")
  .addEventListener("click", () => exportDiff("md"));
document
  .getElementById("export-json")
  .addEventListener("click", () => exportDiff("json"));

// ---- Watch a file ------------------------------------------------------------

const watchButton = document.getElementById("watch");
const watchLive = document.getElementById("watch-live");
const watchResume = document.getElementById("watch-resume");

const watcher = new FileWatcher((change, error) => {
  if (!change) {
    updateWatchUi();
    setStatus(`stopped watching: ${error?.message ?? "file unavailable"}`, {
      error: true,
    });
    return;
  }
  sourceName.textContent = change.name;
  sourceName.title = `${change.name} · watching`;
  goToWorkspace();
  if (isPassTrace(change.text)) {
    loadTrace(change.text, { keepIndex: !!trace });
  } else {
    clearTrace();
    input.value = change.text;
    input.dispatchEvent(new Event("input"));
    clearTimeout(timer);
    run();
  }
  statusEl.textContent = `reloaded ${new Date().toLocaleTimeString()} · ${statusEl.textContent}`;
});

function updateWatchUi() {
  watchButton.hidden = !canWatchFiles || watcher.watching;
  watchLive.hidden = !watcher.watching;
  document
    .getElementById("menu-handle")
    .classList.toggle("watching", watcher.watching);
  if (watcher.watching) {
    watchResume.hidden = true;
    watchLive.title = `Watching ${watcher.handle.name} · click to stop`;
  }
}

async function pickWatch() {
  const handle = await watcher.pick();
  if (handle) await kv.set("watch-handle", handle);
  updateWatchUi();
}

function stopWatching() {
  watcher.stop();
  kv.delete("watch-handle");
  updateWatchUi();
}

watchButton.addEventListener("click", pickWatch);
watchLive.addEventListener("click", stopWatching);

// Browsers forget file permission on reload; offer a one-click resume.
async function restoreWatch() {
  if (!canWatchFiles) return;
  const handle = await kv.get("watch-handle");
  if (!handle) return;
  if (!(await FileWatcher.needsPermission(handle))) {
    await watcher.start(handle);
    updateWatchUi();
    return;
  }
  watchResume.textContent = `Resume watching ${handle.name}`;
  watchResume.hidden = false;
  watchButton.hidden = true;
  watchResume.onclick = async () => {
    if (await FileWatcher.requestPermission(handle))
      await watcher.start(handle);
    else watchResume.hidden = true;
    updateWatchUi();
  };
}

// ---- Startup -------------------------------------------------------------------

document.getElementById("abi").textContent = `wasm abi v${ABI_VERSION}`;
showTab(input);
window.addEventListener("hashchange", updateRoute);
updateRoute();
updateWatchUi();

const saved = await kv.get("autosave");
if (saved) applyState(saved);
else applyState(await loadSampleState(SAMPLES[0]));
restored = true;
restoreWatch();
