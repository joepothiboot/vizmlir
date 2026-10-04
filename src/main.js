import { MlirEngine } from "./ir/index.js";
import { CanvasRenderer, createDescentView } from "./render/index.js";
import { ABI_VERSION, STATUS } from "./ir/index.js";
import { copySnapshot, diffSnapshots } from "./ir/index.js";
import {
  countOps,
  opCountsToCSV,
  opCountTable,
  totalOps,
} from "./trace/index.js";
import {
  analyzeBuffers,
  bufferTotals,
  buffersToJSON,
  compareBuffers,
} from "./trace/index.js";
import {
  renderFunctions,
  renderPasses,
  renderSummary,
  signedBytes,
} from "./app/index.js";
import {
  findSymbolNode,
  historyToJSON,
  symbolHistory,
  symbolTimeline,
} from "./trace/index.js";
import { diffStats, lineDiff } from "./trace/index.js";
import { analyzeGpu, memorySpace } from "./gpu/index.js";
import {
  focusGpuLine,
  renderGpuPath,
  renderGpuView,
  verdictSummary,
} from "./render/index.js";
import { warpAccess } from "./gpu/index.js";
import {
  analyzeLocalMemory,
  hasLocalMemory,
  localLineInfo,
  localTargetOf,
} from "./gpu/index.js";
import {
  focusLocalLine,
  localExplain,
  renderLocalView,
} from "./render/index.js";
import { parseMemref } from "./trace/index.js";
import { buildOpModel, nodesOnLine, opHistory, opRecords, snapshotShape } from "./trace/index.js";
import { conditionHits, mergeHits, nextHit, parseCondition } from "./trace/index.js";
import { buildLocIndex, findSource, linesOf, nodesAt, opCounts } from "./trace/index.js";
import { explainLine } from "./anatomy.js";
import {
  changedBeyond,
  compareBenchmarks,
  comparisonOrder,
  comparisonToJSON,
  formatChange,
  formatDuration,
  parseBenchmarks,
  symbolTimes,
} from "./bench.js";
import { bindHighlighting, highlightMlir } from "./app/index.js";
import { CommandPalette } from "./app/index.js";
import { createInspector } from "./app/index.js";
import { createScrubber } from "./app/index.js";
import { createSelection } from "./app/index.js";
import { createSourceView } from "./app/index.js";
import { loadSampleState, SAMPLES } from "./samples.js";
import { bindSplitters } from "./app/index.js";
import {
  kv,
  sessionFromFile,
  sessions,
  sessionToFile,
} from "./session/index.js";
import { canWatchFiles, FileWatcher } from "./session/index.js";
import {
  diffRecords,
  diffToJSON,
  diffToMarkdown,
  diffToPatch,
  download,
  slug,
} from "./session/index.js";
import {
  baselineFor,
  describeEvent,
  isPassTrace,
  moduleStateAt,
  parsePassTrace,
} from "./trace/index.js";
import {
  byteLength,
  extractReports,
  formatBytes,
  formatSeconds,
  matchTiming,
  timingToJSON,
} from "./trace/index.js";

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
const baselineHighlight = document.getElementById("baseline-highlight");
const editorHighlight = document.getElementById("editor-highlight");
const diagsEl = document.getElementById("diags");
const passScrubber = document.getElementById("pass-scrubber");
const passList = document.getElementById("pass-list");
const passListWrap = document.getElementById("pass-list-wrap");
const passListCount = document.getElementById("pass-list-count");
const fileInput = document.getElementById("file");
const baselineTitle = document.getElementById("baseline-title");
const currentTitle = document.getElementById("current-title");
const sourcePane = document.getElementById("source-pane");
const splitToggle = document.getElementById("split-toggle");
const sourceTitle = document.getElementById("source-title");
const sourcePanel = document.getElementById("source-panel");
const sourceName = document.getElementById("source-name");
const diffTitle = document.getElementById("diff-title");
const inspectorEl = document.getElementById("inspector");
const inspectorToggle = document.getElementById("inspector-toggle");
const inspectorBadge = document.getElementById("inspector-badge");
const inspectorSplitter = document.getElementById("inspector-splitter");
const gpuPath = document.getElementById("gpu-path");
const gpuAnswerSection = document.getElementById("gpu-answer-section");
const gpuAnswerHost = document.getElementById("gpu-answer-host");
const vizVerdict = document.getElementById("viz-verdict");
const helpDialog = document.getElementById("help");
const zoomLevel = document.getElementById("zoom-level");
const timingOpen = document.getElementById("timing-open");
const opCountOpen = document.getElementById("opcount-open");
const buffersOpen = document.getElementById("buffers-open");
const symbolsOpen = document.getElementById("symbols-open");
const stage = document.getElementById("stage");
const gpuViewEl = document.getElementById("gpu-view");
const viewGpu = document.getElementById("view-gpu");
const viewDescent = document.getElementById("view-descent");
const descentEl = document.getElementById("descent-view");
const localViewEl = document.getElementById("local-view");
const viewLocal = document.getElementById("view-local");
const leTitle = document.getElementById("le-title");
const leBody = document.getElementById("le-body");
const leDocs = document.getElementById("le-docs");
const TABS = [
  { tab: baselineTitle, textarea: baseline },
  { tab: currentTitle, textarea: input },
];

// Source files loaded for the Source tab, by name, and the source locations of
// the rendered IR (rebuilt on every parse, since the engine reuses its memory).
let sources = {};
let locIndex = null;

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
// Timing and peak memory from the loaded trace or pasted log; `matches` links
// trace events to timing rows.
let profile = { timing: null, memory: null, matches: [] };
// Op counts of the rendered baseline and current IR, and the text the graph
// was parsed from (so its snapshot can be restored after other parses).
let viewCounts = null;
let renderedText = "";
// Whole-module op counts for each trace event, computed when first shown.
let traceCounts = null;
// Buffer analyses of the rendered baseline and current IR, and whole-module
// buffer totals for each trace event (computed when first shown).
let viewBuffers = null;
let traceBuffers = null;
// What the canvas area shows ("graph" or "gpu"), the GPU launches and
// buffers of the rendered IR (null when it has no GPU code), and which view
// the person prefers when both apply: the GPU view unless they picked the
// graph. IR without GPU code always shows the graph.
const CANVAS_KEY = "vizmlir-canvas";
let canvasView = "graph";
let gpuModel = null;
let canvasPreference = "gpu";
try {
  if (localStorage.getItem(CANVAS_KEY) === "graph") canvasPreference = "graph";
} catch {}
// The scratchpad buffers and DMAs of the rendered IR (src/gpu/local-memory.js),
// and whether the Local memory view should open when they appear: yes until
// the person picks another view, so stepping through a trace keeps it.
let localModel = null;
let localWanted = true;
// Created / changed / lowered / removed passes for every symbol in the trace,
// computed when first shown.
let traceSymbols = null;
// How each op of a trace was made and changed, computed when first shown.
let opModel = null;
// Each pass's IR as a tree, for the Descent view (filled with opModel).
let opShapes = null;
// The history on show after jumping to the pass that removed its op, where
// nothing is selected; cleared by the next pick.
let opPinned = null;
// Imported kernel benchmark results, a baseline and a current run, each
// { name, text, mock, result } or null; `mock` marks a sample's invented data. Kept with the trace (and its live reloads)
// and saved with the session.
let benchmarks = { baseline: null, current: null };

// What is selected: the pass, the graph node, the marked source line.
const selection = createSelection();

const renderer = new CanvasRenderer(canvas, {
  onSelect(index, snap) {
    const parent = index < 0 ? -1 : snap.parentOf(index);
    detailEl.textContent =
      index < 0
        ? "—"
        : `#${index} ${snap.labelOf(index)}` +
          (parent < 0 ? "" : ` · parent ${snap.labelOf(parent)}`) +
          sourceNote(index);
    syncDiffSelection(index);
    selection.select({ node: index });
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
  // The counts also ride on the inspector's toggle, so they show while it is
  // closed.
  inspectorBadge.replaceChildren(
    ...(total
      ? [...diffSummary.childNodes].map((node) => node.cloneNode(true))
      : []),
  );

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
      setViewCounts(null);
      setViewBuffers(null);
      refreshInspector();
      return;
    }
    beforeCopy = copySnapshot(before.snapshot);
  }

  let current = input.value;
  if (!trace) {
    const reports = mayHaveReports(current)
      ? extractReports(current)
      : { text: current, timing: null, memory: null };
    current = reports.text;
    setProfile(reports.timing, reports.memory, []);
  }

  const after = parse(current);
  if (after.status !== STATUS.OK) {
    setStatus(`current error: ${engine.statusText}`, { error: true });
    renderer.setSnapshot(null);
    setViewCounts(null);
    setViewBuffers(null);
    setGpuModel(null);
    setLocalModel(null);
    refreshInspector();
    return;
  }

  const snap = after.snapshot;
  renderedText = current;
  renderer.setSnapshot(snap);
  locIndex = buildLocIndex(snap);
  refreshSources();
  setGpuModel(analyzeGpu(current));
  setLocalModel(
    analyzeLocalMemory(current, {
      target:
        trace && traceIndex >= 0
          ? localTargetOf(trace.events, traceIndex)
          : null,
    }),
  );
  setViewCounts({
    before: beforeCopy ? countOps(beforeCopy) : null,
    after: countOps(snap),
  });
  setViewBuffers({
    before: beforeCopy ? analyzeBuffers(baseline.value) : null,
    after: analyzeBuffers(current),
  });
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
  refreshInspector();
}

function loadTrace(text, { keepIndex = false } = {}) {
  const previousIndex = traceIndex;
  traceText = text;
  trace = parsePassTrace(text);
  traceCounts = null;
  traceBuffers = null;
  traceSymbols = null;
  opModel = null;
  opShapes = null;
  opPinned = null;
  bpHits = new Map();
  scrubber.setMarks([]);
  if (!trace.events.length) {
    clearTrace();
    setStatus("no IR dumps found in trace", { error: true });
    return;
  }
  setProfile(
    trace.timing,
    trace.memory,
    matchTiming(trace.events, trace.timing),
  );
  const slowest = Math.max(
    0,
    ...profile.matches.map((match) => match?.row.wall.seconds ?? 0),
  );
  passChanges = new Map();
  scrubber.setPasses(
    trace.events.map((event) => {
      const match = profile.matches[event.index];
      return {
        failed: event.failed,
        share: match && slowest > 0 ? match.row.wall.seconds / slowest : null,
      };
    }),
  );
  renderPassList(slowest);
  passScrubber.hidden = false;
  const firstFailure = trace.events.findIndex((event) => event.failed);
  selectEvent(
    keepIndex && previousIndex >= 0
      ? Math.min(previousIndex, trace.events.length - 1)
      : firstFailure >= 0
        ? firstFailure
        : 0,
  );
  updateSymbolsOpen();
  updateDescentTab();
  refreshBreakpoints({ defer: true });
}

function selectEvent(index) {
  const event = trace.events[index];
  const base = baselineFor(trace.events, index);
  traceIndex = index;
  selection.select({ pass: index });
  scrubber.setValue(index);
  [...passList.children].forEach((item, i) => {
    const button = item.firstChild;
    if (i === index) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  if (passListWrap.open)
    passList.children[index]?.scrollIntoView({ block: "nearest" });
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
  const match = profile.matches[index];
  const size = byteLength(event.ir);
  const growth = base ? size - byteLength(base.ir) : 0;
  traceNote =
    `pass ${index + 1}/${trace.events.length}${event.failed ? " · ✗ FAILED" : ""}` +
    (match ? ` · ${describeMatch(match)}` : "") +
    ` · IR ${formatBytes(size)}` +
    (base && growth
      ? ` (${growth > 0 ? "+" : "−"}${formatBytes(Math.abs(growth))})`
      : "") +
    ` · trace line ${event.headerLine} · `;
  renderDiagnostics(
    event.index === trace.events.length - 1
      ? [
          ...event.diagnostics,
          ...trace.diagnostics.filter((d) => d.eventIndex < 0),
        ]
      : event.diagnostics,
  );
  run();
  updateDebugBar();
  refreshDescent();
}

// Every pass as one compact row, in the Changes tab: number, name, wall
// time and IR size, with a bar for its time against the slowest.
function renderPassList(slowest) {
  const width = Math.max(2, String(trace.events.length).length);
  passList.replaceChildren(
    ...trace.events.map((event) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      const flags = `${event.failed ? " ✗" : ""}${event.diagnostics.length ? ` · ${event.diagnostics.length} diag` : ""}`;
      const number = document.createElement("span");
      number.className = "n";
      number.textContent = String(event.index + 1).padStart(width, "0") + flags;
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = describeEvent(event);
      const match = profile.matches[event.index];
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent =
        (match ? `${formatSeconds(match.row.wall.seconds)} · ` : "") +
        formatBytes(byteLength(event.ir));
      button.append(number, name, meta);
      if (match && slowest > 0)
        button.style.setProperty(
          "--time",
          `${(100 * match.row.wall.seconds) / slowest}%`,
        );
      button.title =
        `${event.index + 1}. ${describeEvent(event)}${event.failed ? " ✗ failed" : ""}` +
        (match ? `\n${describeMatch(match)}` : "");
      button.classList.toggle("failed", event.failed);
      button.addEventListener("click", () => selectEvent(event.index));
      item.append(button);
      return item;
    }),
  );
  passListCount.textContent = `(${trace.events.length})`;
  passListWrap.hidden = false;
}

// Ops pass `index` added, removed and changed against its baseline, worked
// out on demand for the scrubber's preview and kept until the next trace.
// The engine has one arena, so the graph's snapshot is re-parsed afterwards.
let passChanges = new Map();
function changedOpsAt(index) {
  if (index === traceIndex) {
    const counts = { added: 0, removed: 0, changed: 0 };
    for (const row of diffRows) counts[row.type] += 1;
    return counts;
  }
  if (passChanges.has(index)) return passChanges.get(index);
  const base = baselineFor(trace.events, index);
  let counts = null;
  const before = base ? parse(base.ir) : null;
  if (!before || before.status === STATUS.OK) {
    const beforeCopy = before && copySnapshot(before.snapshot);
    const after = parse(trace.events[index].ir);
    if (after.status === STATUS.OK) {
      counts = { added: 0, removed: 0, changed: 0 };
      for (const row of diffSnapshots(beforeCopy, after.snapshot))
        counts[row.type] += 1;
    }
  }
  if (renderedText) {
    const restored = parse(renderedText).snapshot;
    renderer.snapshot = restored;
    diffAfter = restored;
    renderer.requestDraw();
  }
  passChanges.set(index, counts);
  return counts;
}

let previewTimer = 0;
const scrubber = createScrubber(passScrubber, {
  onSelect: (index) => trace && index !== traceIndex && selectEvent(index),
  label: (index) => {
    const event = trace?.events[index];
    if (!event) return "";
    return (
      `Pass ${index + 1} of ${trace.events.length}: ${describeEvent(event)}` +
      (event.failed ? " · failed" : "")
    );
  },
  preview: (index) => {
    const event = trace.events[index];
    const base = baselineFor(trace.events, index);
    const size = byteLength(event.ir);
    const growth = base ? size - byteLength(base.ir) : 0;
    const match = profile.matches[index];
    const lines = [
      `IR ${formatBytes(size)}` +
        (growth
          ? ` (${growth > 0 ? "+" : "−"}${formatBytes(Math.abs(growth))})`
          : ""),
    ];
    // The op diff needs two parses; wait until the pointer settles.
    const known = index === traceIndex || passChanges.has(index);
    const counts = known ? changedOpsAt(index) : undefined;
    if (counts === undefined) {
      lines.push("changed ops: …");
      clearTimeout(previewTimer);
      previewTimer = setTimeout(() => {
        if (!trace?.events[index]) return;
        changedOpsAt(index);
        scrubber.refreshPreview(index);
      }, 120);
    } else if (counts === null) {
      lines.push("changed ops: could not parse");
    } else {
      const total = counts.added + counts.removed + counts.changed;
      lines.push(
        total
          ? `changed ops: ${total} (+${counts.added} −${counts.removed} ~${counts.changed})`
          : "changed ops: none",
      );
    }
    if (match) lines.push(describeMatch(match));
    if (event.failed) lines.push("✗ failed");
    return { title: `${index + 1}. ${describeEvent(event)}`, lines };
  },
});

function clearTrace() {
  trace = null;
  traceCounts = null;
  traceBuffers = null;
  traceSymbols = null;
  opModel = null;
  opShapes = null;
  opPinned = null;
  benchmarks = { baseline: null, current: null };
  traceText = "";
  traceNote = "";
  traceIndex = -1;
  traceDiffTitle = "";
  updateDebugBar();
  updateDescentTab();
  passScrubber.hidden = true;
  bpHits = new Map();
  scrubber.setMarks([]);
  scrubber.setPasses([]);
  passList.replaceChildren();
  passListWrap.hidden = true;
  setProfile(null, null, []);
  baselineTitle.textContent = "Baseline";
  currentTitle.textContent = "Current";
  renderDiagnostics([]);
  updateSymbolsOpen();
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


let timer = 0;
bindHighlighting(baseline, baselineHighlight);
bindHighlighting(input, editorHighlight);
function scheduleRun() {
  clearTimeout(timer);
  timer = setTimeout(run, 140);
}
baseline.addEventListener("input", scheduleRun);
input.addEventListener("input", scheduleRun);

function fitGraph() {
  renderer.fit();
  renderer.requestDraw();
}
document.getElementById("zoom-fit").addEventListener("click", fitGraph);
document
  .getElementById("zoom-in")
  .addEventListener("click", () => renderer.zoomBy(1.25));
document
  .getElementById("zoom-out")
  .addEventListener("click", () => renderer.zoomBy(0.8));

// ---- Source tabs -----------------------------------------------------------

function showTab(textarea) {
  if (sourcePane.classList.contains("split-source")) {
    if (textarea === input) return;
    setSourceSplit(false);
  }
  for (const entry of TABS) {
    const on = entry.textarea === textarea;
    entry.tab.setAttribute("aria-selected", String(on));
    entry.textarea.parentElement.hidden = !on;
  }
  sourceTitle.setAttribute("aria-selected", "false");
  sourcePanel.hidden = true;
}

function showSourceTab() {
  if (sourceTitle.hidden) return;
  if (sourcePane.classList.contains("split")) toggleSplit();
  for (const entry of TABS) {
    entry.tab.setAttribute("aria-selected", "false");
    entry.textarea.parentElement.hidden = true;
  }
  sourceTitle.setAttribute("aria-selected", "true");
  sourcePanel.hidden = false;
}

// The current IR above the source file it came from.
function setSourceSplit(on) {
  sourcePane.classList.toggle("split-source", on);
  if (on) {
    if (sourcePane.classList.contains("split")) toggleSplit();
    baselineTitle.setAttribute("aria-selected", "false");
    currentTitle.setAttribute("aria-selected", "true");
    sourceTitle.setAttribute("aria-selected", "false");
    baseline.parentElement.hidden = true;
    input.parentElement.hidden = false;
    sourcePanel.hidden = false;
  } else {
    showTab(input);
  }
}

function toggleTab() {
  showTab(
    currentTitle.getAttribute("aria-selected") === "true" ? baseline : input,
  );
}

function toggleSplit() {
  if (sourcePane.classList.contains("split-source")) setSourceSplit(false);
  const on = sourcePane.classList.toggle("split");
  splitToggle.setAttribute("aria-pressed", String(on));
}

for (const { tab, textarea } of TABS)
  tab.addEventListener("click", () => showTab(textarea));
sourceTitle.addEventListener("click", showSourceTab);
splitToggle.addEventListener("click", toggleSplit);

// ---- Debug mode ------------------------------------------------------------

// Debug mode is opt-in so the default screen stays as it was: it splits the
// sources and shows a step bar under the pass slider. Later debugger controls
// (breakpoints, op history) hang off it.
const debugToggle = document.getElementById("debug-toggle");
const debugBar = document.getElementById("debug-bar");
const debugStatus = document.getElementById("debug-status");
const debugButtons = {
  first: document.getElementById("debug-first"),
  prev: document.getElementById("debug-prev-change"),
  next: document.getElementById("debug-next-change"),
  last: document.getElementById("debug-last"),
};
let debugOn = false;
// Which split Debug turned on: "source" (IR above its source), "split", or "".
let debugSplitOwned = "";

// A pass "changed" when it added, removed or modified an op; one whose IR
// would not parse counts too, since something happened there.
function passChanged(index) {
  const counts = changedOpsAt(index);
  return !counts || counts.added + counts.removed + counts.changed > 0;
}

function changedPassFrom(index, delta) {
  for (let i = index + delta; i >= 0 && i < trace.events.length; i += delta)
    if (passChanged(i)) return i;
  return -1;
}

function updateDebugBar() {
  debugBar.hidden = !(debugOn && trace);
  if (debugBar.hidden) return;
  const last = trace.events.length - 1;
  debugButtons.first.disabled = traceIndex <= 0;
  debugButtons.last.disabled = traceIndex >= last;
  debugButtons.prev.disabled = changedPassFrom(traceIndex, -1) < 0;
  debugButtons.next.disabled = changedPassFrom(traceIndex, 1) < 0;
  const counts = changedOpsAt(traceIndex);
  debugStatus.textContent = counts
    ? `pass ${traceIndex + 1}: +${counts.added} −${counts.removed} ~${counts.changed} ops`
    : `pass ${traceIndex + 1}: IR did not parse`;
  updateOpStepButtons();
  updateBreakpointButtons();
}

function setDebug(on, { save = true } = {}) {
  debugOn = on;
  debugToggle.setAttribute("aria-pressed", String(on));
  const already =
    sourcePane.classList.contains("split") ||
    sourcePane.classList.contains("split-source");
  if (on && !already) {
    if (!sourceTitle.hidden) {
      setSourceSplit(true);
      debugSplitOwned = "source";
    } else {
      toggleSplit();
      debugSplitOwned = "split";
    }
  } else if (!on) {
    if (debugSplitOwned === "source") setSourceSplit(false);
    else if (debugSplitOwned === "split") toggleSplit();
    debugSplitOwned = "";
  }
  updateDebugBar();
  if (save)
    try {
      localStorage.setItem("vizmlir.debug", on ? "1" : "0");
    } catch {}
}

function stepToChange(delta) {
  if (!debugOn || !trace) return;
  const index = changedPassFrom(traceIndex, delta);
  if (index >= 0) selectEvent(index);
}

debugToggle.addEventListener("click", () => setDebug(!debugOn));
debugButtons.first.addEventListener("click", () => selectEvent(0));
debugButtons.last.addEventListener("click", () =>
  selectEvent(trace.events.length - 1),
);
debugButtons.prev.addEventListener("click", () => stepToChange(-1));
debugButtons.next.addEventListener("click", () => stepToChange(1));

// Steps to the previous / next pass that changed the selected op: it was made,
// renamed, inlined, fused or removed there. The op stays selected.
const opStepButtons = {
  prev: document.getElementById("debug-op-prev"),
  next: document.getElementById("debug-op-next"),
};

function updateOpStepButtons() {
  const off = !(debugOn && trace) || renderer.selected < 0;
  opStepButtons.prev.disabled = off;
  opStepButtons.next.disabled = off;
}

function stepOpChange(delta) {
  if (!debugOn || !trace) return;
  if (renderer.selected < 0) {
    setStatus("pick an op first, then step to the passes that changed it");
    return;
  }
  opModel ??= computeOpModel();
  const entries = opHistory(opModel, traceIndex, renderer.selected);
  const changes = entries.filter((entry) => entry.change !== "kept");
  const target =
    delta > 0
      ? changes.find((entry) => entry.pass > traceIndex)
      : changes.findLast((entry) => entry.pass < traceIndex);
  if (!target) {
    setStatus(`no ${delta > 0 ? "later" : "earlier"} pass changes this op`);
    return;
  }
  lastOpEntries = entries;
  goToOp(target.pass, target.node);
  debugStatus.textContent = `${OP_CHANGE[target.change]} · pass ${target.pass + 1}`;
}

opStepButtons.prev.addEventListener("click", () => stepOpChange(-1));
opStepButtons.next.addEventListener("click", () => stepOpChange(1));
selection.subscribe((_, picked) => {
  if (picked.includes("node")) updateOpStepButtons();
});

// ---- Breakpoints -----------------------------------------------------------

// Conditions that stop a run between passes (src/trace/breakpoints.js). The
// passes each one hits are worked out when a trace loads or the list changes,
// and drawn as dots on the pass slider.
const bp = {
  open: document.getElementById("bp-open"),
  pop: document.getElementById("bp-pop"),
  form: document.getElementById("bp-form"),
  input: document.getElementById("bp-input"),
  error: document.getElementById("bp-error"),
  list: document.getElementById("bp-list"),
  count: document.getElementById("bp-count"),
  back: document.getElementById("bp-back"),
  cont: document.getElementById("bp-continue"),
};
let breakpoints = []; // { text, on }
let bpHits = new Map(); // text -> passes it hits
let bpTimer = 0;

function breakpointContext() {
  return {
    events: trace.events,
    counts: () => (traceCounts ??= computeTraceCounts()),
    peaks: () => (traceBuffers ??= computeTraceBuffers()),
  };
}

const activeHits = () =>
  mergeHits(breakpoints.filter((b) => b.on).map((b) => bpHits.get(b.text) ?? []));

// Works the hits out again. After a load this waits a tick so the pass shows
// first; the whole trace is parsed to count ops.
function refreshBreakpoints({ defer = false } = {}) {
  clearTimeout(bpTimer);
  const compute = () => {
    bpHits = new Map();
    if (trace)
      for (const b of breakpoints) {
        const parsed = parseCondition(b.text);
        bpHits.set(b.text, parsed.ok ? conditionHits(parsed.cond, breakpointContext()) : []);
      }
    scrubber.setMarks(activeHits());
    renderBreakpoints();
  };
  if (defer) bpTimer = setTimeout(compute, 0);
  else compute();
}

function updateBreakpointButtons() {
  const hits = activeHits();
  bp.back.disabled = nextHit(hits, traceIndex, -1) < 0;
  bp.cont.disabled = nextHit(hits, traceIndex, 1) < 0;
}

function renderBreakpoints() {
  bp.count.textContent = breakpoints.length ? ` (${breakpoints.length})` : "";
  bp.list.replaceChildren(
    ...breakpoints.map((b, i) => {
      const item = document.createElement("li");
      const on = document.createElement("input");
      on.type = "checkbox";
      on.checked = b.on;
      on.setAttribute("aria-label", `Enable ${b.text}`);
      on.addEventListener("change", () => {
        b.on = on.checked;
        scrubber.setMarks(activeHits());
        updateBreakpointButtons();
        scheduleAutosave();
      });
      const text = document.createElement("code");
      text.textContent = b.text;
      const hits = document.createElement("span");
      hits.className = "hits";
      const passes = bpHits.get(b.text) ?? [];
      if (!trace) hits.textContent = "open a pass trace";
      else if (!passes.length) hits.textContent = "never hit";
      else {
        hits.append("hits at ");
        passes.forEach((pass, k) => {
          const go = document.createElement("button");
          go.type = "button";
          go.textContent = `#${pass + 1}`;
          go.addEventListener("click", () => selectEvent(pass));
          hits.append(go, k < passes.length - 1 ? " " : "");
        });
      }
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove";
      remove.textContent = "✕";
      remove.setAttribute("aria-label", `Remove ${b.text}`);
      remove.addEventListener("click", () => {
        breakpoints.splice(i, 1);
        refreshBreakpoints();
        scheduleAutosave();
      });
      item.append(on, text, hits, remove);
      return item;
    }),
  );
  updateBreakpointButtons();
}

// Runs to the next (or previous) pass any enabled breakpoint hits.
function runToBreakpoint(delta) {
  if (!trace) return;
  const hits = activeHits();
  const target = nextHit(hits, traceIndex, delta);
  if (target < 0) {
    setStatus(
      breakpoints.some((b) => b.on)
        ? `no breakpoint ${delta > 0 ? "after" : "before"} pass ${traceIndex + 1}`
        : "no breakpoints set: use Break when…",
    );
    return;
  }
  selectEvent(target);
  const why = breakpoints
    .filter((b) => b.on && (bpHits.get(b.text) ?? []).includes(target))
    .map((b) => b.text);
  debugStatus.textContent = `break: ${why.join(" · ")} · pass ${target + 1}`;
}

bp.open.addEventListener("click", () => {
  bp.pop.hidden = !bp.pop.hidden;
  bp.open.setAttribute("aria-expanded", String(!bp.pop.hidden));
  if (!bp.pop.hidden) bp.input.focus();
});
bp.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const parsed = parseCondition(bp.input.value);
  bp.error.hidden = parsed.ok;
  if (!parsed.ok) {
    bp.error.textContent = parsed.error;
    return;
  }
  if (!breakpoints.some((b) => b.text === parsed.cond.text))
    breakpoints.push({ text: parsed.cond.text, on: true });
  bp.input.value = "";
  refreshBreakpoints();
  scheduleAutosave();
});
bp.back.addEventListener("click", () => runToBreakpoint(-1));
bp.cont.addEventListener("click", () => runToBreakpoint(1));

// ---- Source line marker ----------------------------------------------------

const lineMark = input.parentElement.querySelector(".line-mark");

function sourceLineBox(line) {
  const style = getComputedStyle(input);
  const lineHeight = parseFloat(style.lineHeight);
  const fallback = {
    top: parseFloat(style.paddingTop) + line * lineHeight,
    height: lineHeight,
  };
  const code = input.parentElement.querySelector("code");
  const lines = input.value.split("\n");
  if (!code || line >= lines.length || !lines[line]) return fallback;
  let start = 0;
  for (let i = 0; i < line; i++) start += lines[i].length + 1;
  const end = start + lines[line].length;
  const range = document.createRange();
  const walker = document.createTreeWalker(code, NodeFilter.SHOW_TEXT);
  let seen = 0;
  let placed = false;
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const len = node.nodeValue.length;
    if (!placed && start < seen + len) {
      range.setStart(node, start - seen);
      placed = true;
    }
    if (placed && end <= seen + len) {
      range.setEnd(node, end - seen);
      const rects = range.getClientRects();
      if (!rects.length) return fallback;
      const first = rects[0];
      const last = rects[rects.length - 1];
      const origin = code.getBoundingClientRect().top;
      const pad = parseFloat(style.paddingTop);
      return {
        top: pad + first.top - origin,
        height: last.bottom - first.top,
      };
    }
    seen += len;
  }
  return fallback;
}

function positionLineMark() {
  const markedLine = selection.state.line;
  if (markedLine < 0) {
    lineMark.hidden = true;
    return;
  }
  const { top, height } = sourceLineBox(markedLine);
  lineMark.style.top = `${top - input.scrollTop}px`;
  lineMark.style.height = `${height}px`;
  lineMark.hidden = false;
}

// The parser records each node's line in the printed IR. The module node has
// none, so for it (and any engine without the table) fall back to finding the
// n-th line that mentions the op: node order follows source order, so the n-th
// node with this op name is (nearly always) on the n-th line that names it.
function markSourceLine(index) {
  let markedLine = -1;
  const snap = renderer.snapshot;
  const printed = snap && index >= 0 ? (snap.irLineOf?.(index) ?? 0) : 0;
  if (printed > 0) {
    markedLine = printed - 1;
  } else if (snap && index >= 0) {
    const op = snap.labelOf(index).split(" ")[0];
    let nth = 0;
    for (let i = 0; i < index; i++)
      if (snap.labelOf(i).split(" ")[0] === op) nth++;
    const escaped = op.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = new RegExp(`(^|[^\\w.])${escaped}([^\\w.]|$)`);
    const lines = input.value.split("\n");
    markedLine = lines.findIndex((line) => pattern.test(line) && nth-- === 0);
  }
  showSourceLine(markedLine);
}

// Marks a 0-based line of the current source. The views that follow a line
// listen to the selection (below).
function showSourceLine(line) {
  selection.select({ line });
}

selection.subscribe((state, picked) => {
  if (!picked.includes("line")) return;
  const markedLine = state.line;
  if (markedLine >= 0) {
    if (
      !sourcePane.classList.contains("split") &&
      sourceTitle.getAttribute("aria-selected") !== "true"
    )
      showTab(input);
    const y = sourceLineBox(markedLine).top;
    if (y < input.scrollTop || y > input.scrollTop + input.clientHeight - 40)
      input.scrollTop = Math.max(0, y - input.clientHeight / 3);
    renderExplain(markedLine);
  }
  focusGpu(markedLine);
  if (canvasView === "local") focusLocalLine(localViewEl, markedLine);
  positionLineMark();
});

// ---- Source tab ------------------------------------------------------------

const sourceView = createSourceView(sourcePanel, {
  onPick: pickSourceLine,
  onAdd: addSourceFiles,
});
// Which op of the ones at a line the last click on it selected.
let sourceCycle = { key: "", at: -1 };

function sourceNote(index) {
  const entry = locIndex?.byNode.get(index);
  if (!entry) return "";
  const { file, line, col } = entry.positions[0];
  return ` · ${file}:${line}:${col}`;
}

// The Source tab shows when the IR has locations; its file list, the names the
// IR refers to that are not loaded, and each line's op count follow the IR.
function refreshSources() {
  const named = locIndex?.files ?? [];
  const has = named.length > 0;
  sourceTitle.hidden = !has;
  if (!has) {
    if (sourcePane.classList.contains("split-source")) setSourceSplit(false);
    else if (sourceTitle.getAttribute("aria-selected") === "true") showTab(input);
  }
  const missing = [...new Set(named.filter((f) => !findSource(sources, f)))];
  sourceView.setSources(sources, missing);
  syncSourceView(renderer.selected);
}

// Ops per line of the file on show, and the lines of the picked op marked.
function syncSourceView(node) {
  const hit = node >= 0 && locIndex ? linesOf(locIndex, node) : null;
  const key = hit && findSource(sources, hit.file);
  if (key) sourceView.mark(key, hit.lines);
  else sourceView.mark(null, []);
  sourceView.setCounts(
    locIndex && sourceView.file ? opCounts(locIndex, sourceView.file) : new Map(),
  );
}

selection.subscribe((state, picked) => {
  if (picked.includes("node")) syncSourceView(state.node);
});

// A click on a source line selects an op that came from it; clicking again
// steps through the others (a line often makes several).
function pickSourceLine(file, line) {
  const nodes = locIndex ? nodesAt(locIndex, file, line) : [];
  if (!nodes.length) {
    sourceView.mark(file, [line]);
    setStatus(`${file}:${line} has no ops in this pass`);
    return;
  }
  const key = `${file}:${line}`;
  const at = sourceCycle.key === key ? (sourceCycle.at + 1) % nodes.length : 0;
  sourceCycle = { key, at };
  if (canvasView !== "graph") setCanvasView("graph", { remember: false });
  renderer.select(nodes[at], { center: true });
  setStatus(
    `${file}:${line} · op ${at + 1} of ${nodes.length}` +
      (nodes.length > 1 ? " (click again for the next)" : ""),
  );
}

async function addSourceFiles(files) {
  const next = { ...sources };
  const skipped = [];
  for (const file of files) {
    const text = file.size > 1024 * 1024 ? null : await file.text();
    if (text === null || text.includes("\0")) skipped.push(file.name);
    else next[file.name] = text;
  }
  sources = next;
  refreshSources();
  scheduleAutosave();
  if (skipped.length) setStatus(`not a text file under 1 MiB: ${skipped.join(", ")}`);
}

// Hand edits mark the loaded source as modified, like an editor tab.
for (const textarea of [baseline, input])
  textarea.addEventListener("input", (e) => {
    if (e.isTrusted && !sourceName.textContent.endsWith(" ●"))
      sourceName.textContent += " ●";
  });

input.addEventListener("scroll", positionLineMark);
input.addEventListener("input", () => {
  selection.clear("line");
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
    el: inspectorSplitter,
    pane: inspectorEl,
    side: "right",
  },
]);

// ---- Inspector ---------------------------------------------------------------

// Every detail view is a tab of one drawer beside the visualization, so what
// it explains stays in sight. Selecting a line, an op, or an access opens it;
// Escape closes it and returns focus to what was selected. Whether it is
// open, and on which tab, is remembered.
const INSPECTOR_KEY = "vizmlir-inspector";
let inspectorSaved = {};
try {
  inspectorSaved = JSON.parse(localStorage.getItem(INSPECTOR_KEY)) || {};
} catch {}

// Draws a tab's contents when it is shown, and again when the pass or the
// IR changes under it.
const PANELS = {
  line: renderLinePanel,
  timing: renderTiming,
  buffers: renderBuffersPanel,
  opcount: renderOpCountPanel,
  symbols: renderSymbolsPanel,
  ophistory: renderOpHistoryPanel,
};

const inspector = createInspector(inspectorEl, {
  toggle: inspectorToggle,
  closeButton: document.getElementById("inspector-close"),
  tab: inspectorSaved.tab,
  onShow: (tab) => PANELS[tab]?.(),
  onChange: ({ open, tab }) => {
    try {
      localStorage.setItem(INSPECTOR_KEY, JSON.stringify({ open, tab }));
    } catch {}
  },
});
// The column's splitter only shows beside an open inspector.
inspectorEl.addEventListener("inspector-toggle", (e) => {
  inspectorSplitter.hidden = !e.detail.open;
});

// Redraws the open tab after the IR or the pass changed.
function refreshInspector() {
  if (inspector.isOpen) PANELS[inspector.active]?.();
}

// Shows a tab from a command (palette, key, status bar), redrawing it when
// it is already on screen.
function showPanel(tab) {
  const shown = inspector.isOpen && inspector.active === tab;
  inspector.show(tab);
  if (shown) PANELS[tab]?.();
}

function toggleInspector() {
  if (inspector.isOpen) inspector.close();
  else inspector.open();
}

// j / k walk the changes, with the Changes tab in view.
function stepChange(delta) {
  focusChange(
    delta > 0 ? diffCursor + 1 : diffCursor < 0 ? -1 : diffCursor - 1,
  );
  if (diffRows.length) inspector.show("changes", { from: canvas });
}

// Opens `tab` for something the person selected (`from` gets focus back on
// close). Selecting an op keeps the Changes or Line tab already open.
function inspect(tab, from) {
  if (tab === "changes" && inspector.isOpen && inspector.active === "line") {
    inspector.open({ from });
    return;
  }
  inspector.show(tab, { from });
}

// ---- Command palette -------------------------------------------------------

const palette = new CommandPalette(document.getElementById("palette"), () => {
  const items = [];
  if (trace)
    for (const event of trace.events)
      items.push({
        group: "pass",
        text: `${event.index + 1}. ${describeEvent(event)}`,
        hint: event.failed ? "✗ failed" : "",
        run: () => {
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
        renderer.select(i, { center: true });
        inspect("changes", canvas);
      },
    });
  }
  const actions = [
    ["Open file…", "", () => fileInput.click()],
    ["Fit graph", "f", fitGraph],
    ["Switch baseline / current", "t", toggleTab],
    ["Toggle split sources", "s", toggleSplit],
    ["Toggle debug mode", "d", () => setDebug(!debugOn)],
    ["Continue to the next breakpoint", ".", () => runToBreakpoint(1)],
    ["Run back to the previous breakpoint", ",", () => runToBreakpoint(-1)],
    ["Op history: follow the selected op", "y", () => showPanel("ophistory")],
    ["Browse samples…", "", openSamples],
    ["Sessions: save, open, import", "", openSessions],
    ["Download session .json", "", downloadSession],
    ["Export graph as PNG", "", exportPNG],
    ["Export graph as SVG", "", exportSVG],
    ["Copy changes as patch", "c", copyChanges],
    ["Export diff as Markdown", "", () => exportDiff("md")],
    ["Export diff as JSON", "", () => exportDiff("json")],
    ["Inspector: changes in this pass", "", () => showPanel("changes")],
    ["Inspector: pass timing and memory", "p", openTiming],
    ["Inspector: op counts per pass", "o", openOpCounts],
    ["Inspector: buffers and peak memory", "b", openBuffers],
    [
      "Inspector: symbol history (which pass made each kernel)",
      "h",
      openSymbols,
    ],
    [
      inspector.isOpen ? "Close the inspector" : "Open the inspector",
      "\\",
      toggleInspector,
    ],
    ...(gpuModel
      ? [
          [
            canvasView === "gpu"
              ? "Show the op graph"
              : "Show the GPU view: launches and memory",
            "g",
            () => setCanvasView(canvasView === "gpu" ? "graph" : "gpu"),
          ],
        ]
      : []),
    ...(localModel
      ? [
          [
            canvasView === "local"
              ? "Show the op graph"
              : "Show the Local memory view: scratchpad buffers and DMAs",
            "",
            () => {
              localWanted = canvasView !== "local";
              setCanvasView(localWanted ? "local" : "graph");
            },
          ],
        ]
      : []),
    [
      "Inspector: what's this line? (explain the clicked line)",
      "w",
      () => showPanel("line"),
    ],
    [
      "Import current kernel benchmarks…",
      "",
      () => importBenchmarks("current"),
    ],
    [
      "Import baseline kernel benchmarks…",
      "",
      () => importBenchmarks("baseline"),
    ],
    ...(canWatchFiles
      ? [
          watcher.watching
            ? ["Stop watching file", "", stopWatching]
            : ["Watch a file for changes…", "", pickWatch],
        ]
      : []),
    ["Show / hide menu", "m", () => toggleMenu()],
    ["Switch light / dark theme", "shift L", toggleTheme],
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

// The menu is docked above the workspace; the strip under it hides and shows
// it, and the choice is remembered.
const MENU_KEY = "vizmlir-menu";
const topbar = document.getElementById("topbar");
const menuHandle = document.getElementById("menu-handle");

function setMenuHidden(hidden) {
  topbar.hidden = hidden;
  const verb = hidden ? "Show" : "Hide";
  menuHandle.setAttribute("aria-expanded", String(!hidden));
  menuHandle.setAttribute("aria-label", `${verb} menu`);
  menuHandle.title = `${verb} menu (m)`;
}

function toggleMenu() {
  setMenuHidden(!topbar.hidden);
  try {
    localStorage.setItem(MENU_KEY, topbar.hidden ? "hidden" : "shown");
  } catch {}
}

menuHandle.addEventListener("click", toggleMenu);
try {
  setMenuHidden(localStorage.getItem(MENU_KEY) === "hidden");
} catch {}

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
  j: () => stepChange(1),
  k: () => stepChange(-1),
  t: toggleTab,
  s: toggleSplit,
  d: () => setDebug(!debugOn),
  "{": () => stepToChange(-1),
  "}": () => stepToChange(1),
  ",": () => debugOn && runToBreakpoint(-1),
  ".": () => debugOn && runToBreakpoint(1),
  "<": () => stepOpChange(-1),
  ">": () => stepOpChange(1),
  p: () => inspector.toggleTab("timing"),
  o: () => inspector.toggleTab("opcount"),
  b: () => inspector.toggleTab("buffers"),
  h: () => inspector.toggleTab("symbols"),
  w: () => inspector.toggleTab("line"),
  y: () => inspector.toggleTab("ophistory"),
  "\\": toggleInspector,
  g: () => gpuModel && setCanvasView(canvasView === "gpu" ? "graph" : "gpu"),
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
  if (e.key === "Escape" && clearPicks()) return;
  if (e.key === "Escape" && inspector.handleEscape())
    return;
  if (e.key === "/") {
    e.preventDefault();
    palette.open();
  } else if (e.key === "?") {
    helpDialog.showModal();
  } else if (e.key === "L") {
    toggleTheme();
  } else if (e.key === "m") {
    toggleMenu();
  } else if (WORKSPACE_KEYS[e.key]) {
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
    benchmarks = { baseline: null, current: null };
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
  try {
    applyState(await loadSampleState(sample, fetchSampleText));
    statusEl.textContent = `loaded ${sample.title} · ${statusEl.textContent}`;
    if (sample.benchmarks) openSymbols();
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
    kind.textContent = sample.benchmarks
      ? "trace + mock benchmarks"
      : sample.trace
        ? sample.handwritten
          ? "pass trace (hand-written)"
          : "pass trace"
        : "before / after";
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
    sourceSplit: sourcePane.classList.contains("split-source"),
    sources: Object.keys(sources).length ? sources : null,
    breakpoints: breakpoints.length ? breakpoints : null,
    benchmarks: traceText
      ? Object.fromEntries(
          Object.entries(benchmarks).map(([slot, bench]) => [
            slot,
            bench && { name: bench.name, text: bench.text, mock: bench.mock },
          ]),
        )
      : null,
  };
}

function applyState(state) {
  clearTrace();
  sources = state.sources ? { ...state.sources } : {};
  breakpoints = (Array.isArray(state.breakpoints) ? state.breakpoints : [])
    .filter((b) => typeof b?.text === "string")
    .map((b) => ({ text: b.text, on: b.on !== false }));
  sourceName.textContent = state.sourceName || "untitled";
  sourceName.title = sourceName.textContent;
  sourcePane.classList.toggle("split", !!state.split);
  splitToggle.setAttribute("aria-pressed", String(!!state.split));
  if (state.trace && isPassTrace(state.trace)) {
    traceIndex = state.traceIndex ?? -1;
    loadTrace(state.trace, { keepIndex: true });
    // Sessions from before baseline benchmarks held one run, as `current`.
    const saved = state.benchmarks?.text
      ? { current: state.benchmarks }
      : state.benchmarks;
    for (const slot of ["baseline", "current"]) {
      if (!trace || !saved?.[slot]) continue;
      try {
        setBenchmarks(
          slot,
          saved[slot].name,
          saved[slot].text,
          saved[slot].mock,
        );
      } catch {}
    }
    updateSymbolsOpen();
  } else {
    baseline.value = state.baseline ?? "";
    input.value = state.current ?? "";
    baseline.dispatchEvent(new Event("input"));
    input.dispatchEvent(new Event("input"));
    clearTimeout(timer);
    run();
  }
  showTab(state.tab === "baseline" ? baseline : input);
  if (state.sourceSplit && !sourceTitle.hidden) setSourceSplit(true);
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

// ---- Timing and memory -----------------------------------------------------

const timingSummary = document.getElementById("timing-summary");
const timingTable = document.getElementById("timing-table");
const timingEmpty = document.getElementById("timing-empty");
const timingExport = document.getElementById("timing-export");

// Reports only ever appear in mlir-opt logs; skip the scan for ordinary IR.
function mayHaveReports(text) {
  return /Execution time report|resident set size|"duration":/.test(text);
}

function setProfile(timing, memory, matches) {
  profile = { timing, memory, matches };
  timingOpen.hidden = !timing && !memory;
  timingOpen.textContent = timing?.total
    ? `⏱ ${formatSeconds(timing.total)}`
    : memory
      ? `⏱ ${formatBytes(memory.peakBytes)}`
      : "⏱";
}

function describeMatch(match) {
  return (
    `${formatSeconds(match.row.wall.seconds)} wall` +
    (match.runs > 1 ? ` across ${match.runs} runs` : "")
  );
}

function openTiming() {
  showPanel("timing");
}

function renderTiming() {
  const { timing, memory, matches } = profile;
  timingEmpty.hidden = !!(timing || memory);
  const facts = [];
  if (timing?.total !== null && timing?.total !== undefined)
    facts.push(["Total", formatSeconds(timing.total)]);
  if (timing)
    facts.push([
      "Passes timed",
      String(timing.rows.filter((row) => row.kind === "pass").length),
    ]);
  if (memory)
    facts.push([
      "Peak memory",
      `${formatBytes(memory.peakBytes)} (whole process, ${memory.source})`,
    ]);
  timingSummary.replaceChildren(
    ...facts.flatMap(([term, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = value;
      return [dt, dd];
    }),
  );

  timingTable.parentElement.hidden = !timing;
  timingExport.disabled = !timing && !memory;
  if (!timing) return;
  const withUser = timing.columns.includes("user");
  const head = document.createElement("tr");
  for (const label of ["Name", ...(withUser ? ["User"] : []), "Wall", "%"]) {
    const th = document.createElement("th");
    th.textContent = label;
    head.append(th);
  }
  const firstEvent = new Map();
  matches.forEach((match, i) => {
    if (match && !firstEvent.has(match.row.index))
      firstEvent.set(match.row.index, i);
  });

  const body = timing.rows.map((row) => {
    const tr = document.createElement("tr");
    tr.className = row.kind;
    const name = document.createElement("td");
    name.style.paddingLeft = `${8 + row.depth * 14}px`;
    const eventIndex = firstEvent.get(row.index);
    if (eventIndex !== undefined) {
      const link = document.createElement("button");
      link.className = "link-btn";
      link.textContent = row.name;
      link.title = `Go to pass #${eventIndex + 1}`;
      link.addEventListener("click", () => selectEvent(eventIndex));
      name.append(link);
    } else {
      name.textContent = row.name;
    }
    const cells = [name];
    if (withUser) cells.push(timeCell(row.user?.seconds));
    cells.push(timeCell(row.wall.seconds));
    const share = document.createElement("td");
    share.className = "share";
    share.style.setProperty("--share", `${row.wall.percent}%`);
    share.textContent = `${row.wall.percent.toFixed(1)}%`;
    cells.push(share);
    tr.append(...cells);
    return tr;
  });
  const thead = document.createElement("thead");
  thead.append(head);
  const tbody = document.createElement("tbody");
  tbody.append(...body);
  timingTable.replaceChildren(thead, tbody);
}

function timeCell(seconds) {
  const td = document.createElement("td");
  td.className = "num";
  td.textContent = formatSeconds(seconds);
  return td;
}

function exportTiming() {
  const title = sourceName.textContent.replace(/ ●$/, "");
  download(
    `${slug(title)}-timing.json`,
    timingToJSON(
      title,
      profile.timing,
      profile.memory,
      trace?.events ?? [],
      profile.matches,
    ),
    "application/json",
  );
}

timingOpen.addEventListener("click", openTiming);
timingExport.addEventListener("click", exportTiming);

// ---- Op counts -------------------------------------------------------------

const opCountBody = document.getElementById("opcount-body");
const opCountNote = document.getElementById("opcount-note");
const opCountSummary = document.getElementById("opcount-summary");
const opCountTableEl = document.getElementById("opcount-table");
const opCountFilter = document.getElementById("opcount-filter");
const opCountChangedOps = document.getElementById("opcount-changed-ops");
const opCountChangedPasses = document.getElementById("opcount-changed-passes");
const opCountShown = document.getElementById("opcount-shown");
let opCountModel = null;

function signed(n) {
  return n > 0 ? `+${n}` : n < 0 ? `−${-n}` : "±0";
}

function setViewCounts(counts) {
  viewCounts = counts;
  opCountOpen.hidden = !counts;
  if (!counts) return;
  const total = totalOps(counts.after);
  const delta = counts.before ? total - totalOps(counts.before) : 0;
  opCountOpen.textContent =
    `${total} ops` + (counts.before && delta ? ` (${signed(delta)})` : "");
}

// Parses the whole-module state at every event. The engine has one arena, so
// the graph's snapshot is re-parsed afterwards to keep drawing valid data.
function computeTraceCounts() {
  const columns = trace.events.map((_, i) => {
    const result = parse(moduleStateAt(trace.events, i));
    return result.status === STATUS.OK ? countOps(result.snapshot) : null;
  });
  if (renderedText) {
    const restored = parse(renderedText).snapshot;
    renderer.snapshot = restored;
    diffAfter = restored;
    renderer.requestDraw();
  }
  return columns;
}

function buildOpCountModel() {
  if (trace) {
    traceCounts ??= computeTraceCounts();
    const failed = traceCounts.filter((column) => !column).length;
    return {
      table: opCountTable(traceCounts.map((column) => column ?? new Map())),
      headers: trace.events.map((event) => `#${event.index + 1}`),
      titles: trace.events.map(
        (event) => `${event.index + 1}. ${describeEvent(event)}`,
      ),
      events: trace.events.map((event) => event.index),
      current: traceIndex,
      note:
        "Operations in the whole module at each dump, by op name. Nested " +
        "dumps are spliced into the last module dump, as for the baseline." +
        (failed
          ? ` ${failed} dump(s) could not be parsed and count as empty.`
          : ""),
    };
  }
  const columns = viewCounts.before
    ? [viewCounts.before, viewCounts.after]
    : [viewCounts.after];
  return {
    table: opCountTable(columns),
    headers: viewCounts.before ? ["Baseline", "Current"] : ["Current"],
    titles: [],
    events: [],
    current: -1,
    note: "Operations in the baseline and current IR, by op name.",
  };
}

function openOpCounts() {
  showPanel("opcount");
}

function renderOpCountPanel() {
  const none = !viewCounts && !trace;
  opCountBody.hidden = none;
  if (none) {
    opCountModel = null;
    opCountNote.textContent = "No parsed IR to count.";
    return;
  }
  opCountModel = buildOpCountModel();
  opCountNote.textContent = opCountModel.note;
  renderOpCountSummary();
  renderOpCountTable();
}

function renderOpCountSummary() {
  const { table, headers } = opCountModel;
  const first = table.totals[0] ?? 0;
  const last = table.totals.at(-1) ?? 0;
  const facts = [
    [
      "Total ops",
      headers.length > 1
        ? `${first} → ${last} (${signed(last - first)})`
        : String(last),
    ],
    ["Distinct ops", String(table.rows.length)],
  ];
  if (trace)
    facts.push([
      "Passes that change counts",
      `${table.changedColumns.length} of ${headers.length}`,
    ]);
  opCountSummary.replaceChildren(
    ...facts.flatMap(([term, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = value;
      return [dt, dd];
    }),
  );
}

function renderOpCountTable() {
  const { table, headers, titles, events, current } = opCountModel;
  const many = headers.length > 2;
  opCountChangedOps.parentElement.hidden = headers.length < 2;
  opCountChangedPasses.parentElement.hidden = !trace;

  // Always keep the first column (the starting point) and the current pass.
  const keep = new Set([0, current, ...table.changedColumns]);
  const shownColumns = headers
    .map((_, i) => i)
    .filter((i) => !trace || !opCountChangedPasses.checked || keep.has(i));
  const needle = opCountFilter.value.trim().toLowerCase();
  const rows = table.rows.filter(
    (row) =>
      (!opCountChangedOps.checked || headers.length < 2 || row.changed) &&
      (!needle || row.op.toLowerCase().includes(needle)),
  );
  opCountShown.textContent =
    `${rows.length} of ${table.rows.length} ops` +
    (trace ? ` · ${shownColumns.length} of ${headers.length} dumps` : "");

  const head = document.createElement("tr");
  head.append(cell("th", "Op", "op"));
  for (const i of shownColumns) {
    const th = cell("th", "", i === current ? "current" : "");
    if (events.length) {
      const link = document.createElement("button");
      link.className = "link-btn";
      link.textContent = headers[i];
      link.title = `${titles[i]}\nGo to this pass`;
      link.addEventListener("click", () => selectEvent(events[i]));
      th.append(link);
    } else {
      th.textContent = headers[i];
    }
    head.append(th);
  }
  if (headers.length > 1) head.append(cell("th", "Δ", "delta"));

  const bodyRow = (label, counts, delta, className) => {
    const tr = document.createElement("tr");
    if (className) tr.className = className;
    const name = cell("td", label, "op");
    name.title = label;
    tr.append(name);
    shownColumns.forEach((i, k) => {
      // Compare with the previous dump, not the previous shown column, so a
      // cell is marked only where that pass changed the count.
      const previous = i > 0 ? counts[i - 1] : counts[i];
      const change = counts[i] - previous;
      const trend =
        k === 0
          ? ""
          : change > 0
            ? "up"
            : change < 0
              ? "down"
              : many
                ? "same"
                : "";
      const td = cell(
        "td",
        String(counts[i]),
        [trend, i === current ? "current" : ""].filter(Boolean).join(" "),
      );
      if (k > 0 && change) td.title = `${signed(change)} in this pass`;
      tr.append(td);
    });
    if (headers.length > 1)
      tr.append(
        cell(
          "td",
          signed(delta),
          `delta ${delta > 0 ? "up" : delta < 0 ? "down" : "same"}`,
        ),
      );
    return tr;
  };

  const thead = document.createElement("thead");
  thead.append(head);
  const tbody = document.createElement("tbody");
  tbody.append(
    bodyRow(
      "(all ops)",
      table.totals,
      (table.totals.at(-1) ?? 0) - (table.totals[0] ?? 0),
      "total",
    ),
    ...rows.map((row) => bodyRow(row.op, row.counts, row.delta)),
  );
  opCountTableEl.replaceChildren(thead, tbody);
}

function cell(tag, text, className) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

function exportOpCounts() {
  if (!opCountModel) return;
  download(
    `${slug(sourceName.textContent.replace(/ ●$/, ""))}-op-counts.csv`,
    opCountsToCSV(opCountModel.headers, opCountModel.table),
    "text/csv",
  );
}

opCountOpen.addEventListener("click", openOpCounts);
opCountFilter.addEventListener("input", renderOpCountTable);
opCountChangedOps.addEventListener("change", renderOpCountTable);
opCountChangedPasses.addEventListener("change", renderOpCountTable);
document
  .getElementById("opcount-export")
  .addEventListener("click", exportOpCounts);

// ---- Buffers ---------------------------------------------------------------

const buffersNone = document.getElementById("buffers-none");
const buffersBody = document.getElementById("buffers-body");
const buffersSummary = document.getElementById("buffers-summary");
const buffersEmpty = document.getElementById("buffers-empty");
const buffersFunctions = document.getElementById("buffers-functions");
const buffersPasses = document.getElementById("buffers-passes");
const buffersPassTable = document.getElementById("buffers-pass-table");
const buffersChangedPasses = document.getElementById("buffers-changed-passes");
const buffersPassShown = document.getElementById("buffers-pass-shown");
let buffersModel = null;

function setViewBuffers(analyses) {
  viewBuffers = analyses;
  const after = analyses && bufferTotals(analyses.after);
  const before = analyses?.before && bufferTotals(analyses.before);
  buffersOpen.hidden = !after || (!after.buffers && !before?.buffers);
  if (buffersOpen.hidden) return;
  const delta = before ? after.peak - before.peak : 0;
  buffersOpen.textContent =
    `▦ ${formatBytes(after.peak)}` + (delta ? ` (${signedBytes(delta)})` : "");
}

// Text only: no engine parse, so the graph's snapshot is left alone.
function computeTraceBuffers() {
  return trace.events.map((_, i) =>
    bufferTotals(analyzeBuffers(moduleStateAt(trace.events, i))),
  );
}

function openBuffers() {
  showPanel("buffers");
}

function renderBuffersPanel() {
  buffersNone.hidden = !!viewBuffers;
  buffersBody.hidden = !viewBuffers;
  if (!viewBuffers) {
    buffersModel = null;
    return;
  }
  const { before, after } = viewBuffers;
  const comparison = compareBuffers(before, after);
  if (trace) traceBuffers ??= computeTraceBuffers();
  buffersModel = { comparison, passes: trace ? traceBuffers : null };

  renderSummary(buffersSummary, {
    before: before && bufferTotals(before),
    after: bufferTotals(after),
    globals: after.globals,
  });
  const shown = renderFunctions(buffersFunctions, comparison, {
    hasBaseline: !!before,
    onLine: showSourceLine,
  });
  buffersEmpty.hidden = shown > 0;
  buffersPasses.hidden = !trace;
  if (trace) renderBufferPasses();
}

function renderBufferPasses() {
  const rows = trace.events.map((event, i) => ({
    header: `#${event.index + 1}`,
    title: describeEvent(event),
    totals: traceBuffers[i],
  }));
  const { shown, changed } = renderPasses(buffersPassTable, rows, {
    current: traceIndex,
    onlyChanged: buffersChangedPasses.checked,
    onPass: selectEvent,
  });
  buffersPassShown.textContent = `${shown} of ${rows.length} dumps · ${changed} change buffers`;
}

function exportBuffers() {
  if (!buffersModel) return;
  const passes = buffersModel.passes?.map((totals, i) => ({
    pass: describeEvent(trace.events[i]),
    ...totals,
  }));
  download(
    `${slug(sourceName.textContent.replace(/ ●$/, ""))}-buffers.json`,
    buffersToJSON(sourceName.textContent, buffersModel.comparison, passes),
    "application/json",
  );
}

buffersOpen.addEventListener("click", openBuffers);
buffersChangedPasses.addEventListener("change", renderBufferPasses);
document
  .getElementById("buffers-export")
  .addEventListener("click", exportBuffers);

// ---- Symbol history --------------------------------------------------------

const symbolsNone = document.getElementById("symbols-none");
const symbolsBody = document.getElementById("symbols-body");
const symbolsSummary = document.getElementById("symbols-summary");
const symbolsTable = document.getElementById("symbols-table");
const symbolsFilter = document.getElementById("symbols-filter");
const symbolsChanged = document.getElementById("symbols-changed");
const symbolsAtPass = document.getElementById("symbols-at-pass");
const symbolsShown = document.getElementById("symbols-shown");
const symbolsBench = document.getElementById("symbols-bench");
const symbolsUnmatched = document.getElementById("symbols-unmatched");
const symbolsBenchError = document.getElementById("symbols-bench-error");
const symbolsBenchClear = document.getElementById("symbols-bench-clear");
const symbolsBenchFile = document.getElementById("symbols-bench-file");
const symbolsMock = document.getElementById("symbols-mock");
const symbolsMinChange = document.getElementById("symbols-min-change");
const symbolsMinChangeOn = document.getElementById("symbols-min-change-on");
const BENCH_DOCS =
  "https://github.com/joepothiboot/vizmlir/blob/main/docs/benchmark-format.md";
// The slot the file picker is importing into.
let benchSlot = "current";
const STEP_TEXT = {
  created: "created",
  changed: "changed",
  lowered: "lowered",
  removed: "removed",
};

function openSymbols() {
  symbolsBenchError.hidden = true;
  showPanel("symbols");
}

function renderSymbolsPanel() {
  symbolsNone.hidden = !!trace;
  symbolsBody.hidden = !trace;
  if (!trace) return;
  renderSymbolsSummary();
  renderSymbols();
}

// Parses benchmark results into `slot` (baseline or current); throws on input
// it cannot read. `mock` marks invented sample data.
function setBenchmarks(slot, name, text, mock = false) {
  benchmarks[slot] = { name, text, mock, result: parseBenchmarks(text) };
}

// Each imported run matched to symbols, and the per-symbol comparison, or
// null when nothing is imported.
function benchmarkView() {
  if (!benchmarks.baseline && !benchmarks.current) return null;
  traceSymbols ??= symbolHistory(trace.events);
  const runs = {};
  for (const slot of ["baseline", "current"])
    runs[slot] =
      benchmarks[slot] &&
      symbolTimes(benchmarks[slot].result.entries, traceSymbols);
  return {
    runs,
    both: !!(runs.baseline && runs.current),
    comparison: compareBenchmarks(runs.baseline?.times, runs.current?.times),
  };
}

function importBenchmarks(slot) {
  if (!trace) {
    setStatus("open a pass trace before importing benchmarks", { error: true });
    return;
  }
  benchSlot = slot;
  symbolsBenchFile.click();
}

symbolsBenchFile.addEventListener("change", async () => {
  const [file] = symbolsBenchFile.files;
  symbolsBenchFile.value = "";
  if (!file || !trace) return;
  const text = await file.text();
  if (!inspector.isOpen || inspector.active !== "symbols") openSymbols();
  try {
    setBenchmarks(benchSlot, file.name, text);
    symbolsBenchError.hidden = true;
    scheduleAutosave();
  } catch (error) {
    symbolsBenchError.replaceChildren(
      `Could not read ${file.name} as ${benchSlot} benchmarks: ${error.message}. See the `,
      Object.assign(document.createElement("a"), {
        href: BENCH_DOCS,
        target: "_blank",
        rel: "noopener",
        textContent: "benchmark format",
      }),
      ".",
    );
    symbolsBenchError.hidden = false;
  }
  renderSymbolsSummary();
  renderSymbols();
});

symbolsBenchClear.addEventListener("click", () => {
  benchmarks = { baseline: null, current: null };
  scheduleAutosave();
  renderSymbolsSummary();
  renderSymbols();
});

function renderSymbolsSummary() {
  traceSymbols ??= symbolHistory(trace.events);
  const count = (predicate) => traceSymbols.filter(predicate).length;
  const facts = [
    ["Symbols", String(traceSymbols.length)],
    ["Created during the trace", String(count((r) => !r.initial))],
    ["Lowered to another op", String(count((r) => r.ops.length > 1))],
    ["Removed", String(count((r) => r.removed))],
  ];
  const view = benchmarkView();
  for (const slot of ["baseline", "current"]) {
    const bench = benchmarks[slot];
    if (!bench) continue;
    const { entries, format, timeColumn, skipped } = bench.result;
    const matched = entries.length - view.runs[slot].unmatched.length;
    facts.push([
      slot === "baseline" ? "Baseline benchmarks" : "Current benchmarks",
      `${bench.name}${bench.mock ? " (mock data)" : ""} (${format}, ${timeColumn}) · ` +
        `${matched} of ${entries.length} kernels matched` +
        (skipped ? ` · ${skipped} row(s) without a name or time skipped` : ""),
    ]);
  }
  if (view?.both) {
    const rows = [...view.comparison.values()].filter(
      (row) => row.change !== null,
    );
    const threshold = Number(symbolsMinChange.value) || 0;
    const slower = rows.filter((row) => row.change * 100 > threshold).length;
    const faster = rows.filter((row) => row.change * 100 < -threshold).length;
    facts.push([
      "Compared",
      `${rows.length} symbol(s) measured in both · ${slower} slower and ` +
        `${faster} faster by more than ${threshold}%`,
    ]);
  }
  symbolsSummary.replaceChildren(
    ...facts.flatMap(([term, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = value;
      return [dt, dd];
    }),
  );
}

function renderSymbols() {
  const view = benchmarkView();
  const comparison = view?.comparison;
  const slots = ["baseline", "current"].filter((slot) => view?.runs[slot]);
  symbolsBenchClear.hidden = !view;
  symbolsMock.hidden = !slots.some((slot) => benchmarks[slot].mock);
  symbolsMinChangeOn.parentElement.hidden = !view?.both;
  const unmatched = slots.flatMap((slot) =>
    view.runs[slot].unmatched.map((match) => ({ ...match, slot })),
  );
  symbolsBench.hidden = !unmatched.length;
  if (view) renderUnmatched(unmatched, view.both);

  const needle = symbolsFilter.value.trim().toLowerCase();
  const rows = traceSymbols.filter(
    (record) =>
      (!symbolsChanged.checked || record.changes.length) &&
      (!symbolsAtPass.checked ||
        record.changes.some((change) => change.index === traceIndex)) &&
      (!needle ||
        record.path.toLowerCase().includes(needle) ||
        record.ops.some((op) => op.toLowerCase().includes(needle))) &&
      (!view?.both ||
        !symbolsMinChangeOn.checked ||
        changedBeyond(
          comparison.get(record.path),
          Number(symbolsMinChange.value) || 0,
        )),
  );
  // Largest slowdown (or slowest) first; unmeasured symbols keep trace order.
  if (comparison)
    rows.sort((a, b) =>
      comparisonOrder(comparison.get(a.path), comparison.get(b.path)),
    );
  symbolsShown.textContent = `${rows.length} of ${traceSymbols.length} symbols`;

  const head = document.createElement("tr");
  head.append(cell("th", "Symbol"), cell("th", "Defined by"));
  // One run reads as "Time / call" (or "Baseline / call"); two as columns.
  for (const slot of slots) {
    const title =
      slot === "baseline" ? "Baseline" : view.both ? "Current" : "Time";
    head.append(cell("th", view.both ? title : `${title} / call`, "time"));
  }
  if (view?.both) head.append(cell("th", "Δ", "time"));
  head.append(cell("th", "Passes that touched it"));
  const thead = document.createElement("thead");
  thead.append(head);

  const tbody = document.createElement("tbody");
  for (const record of rows) {
    const tr = document.createElement("tr");
    if (record.removed) tr.className = "gone";
    const sym = cell("td", "", "sym");
    const show = document.createElement("button");
    show.className = "link-btn";
    show.textContent = record.path;
    show.title =
      (record.initial ? "In the first dump" : "Created during the trace") +
      "\nOpen its IR at every pass that changed it";
    show.addEventListener("click", () => openSymbolView(record));
    sym.append(show);
    const steps = document.createElement("div");
    steps.className = "steps";
    if (!record.changes.length)
      steps.append(cell("span", "unchanged through the trace", "none"));
    for (const change of record.changes) {
      const event = trace.events[change.index];
      const step = document.createElement("button");
      step.className = [
        "link-btn",
        "step",
        change.kind,
        change.index === traceIndex ? "current" : "",
      ]
        .filter(Boolean)
        .join(" ");
      step.textContent =
        `#${change.index + 1} ${event.argument || event.pass} · ` +
        STEP_TEXT[change.kind] +
        (change.kind === "lowered" ? ` to ${change.op}` : "");
      step.title = `${change.index + 1}. ${describeEvent(event)}\nGo to this pass`;
      step.addEventListener("click", () => selectEvent(change.index));
      steps.append(step);
    }
    const stepsCell = document.createElement("td");
    stepsCell.append(steps);
    tr.append(sym, cell("td", record.ops.join(" → "), "ops"));
    const row = comparison?.get(record.path);
    for (const slot of slots) {
      const time = row?.[slot];
      const td = cell("td", time ? formatDuration(time.timeNs) : "", "time");
      if (time)
        td.title =
          `${time.calls} call(s) · total ${formatDuration(time.totalNs)}\n` +
          time.kernels
            .map((kernel, i) => `${kernel} (${time.how[i]} match)`)
            .join("\n");
      tr.append(td);
    }
    if (view?.both) {
      const delta = row?.change ?? null;
      const td = cell(
        "td",
        delta === null ? "" : formatChange(delta),
        `time ${delta > 0 ? "up" : delta < 0 ? "down" : ""}`.trim(),
      );
      if (delta !== null)
        td.title =
          `${row.deltaNs > 0 ? "+" : row.deltaNs < 0 ? "−" : "±"}` +
          `${formatDuration(Math.abs(row.deltaNs))} per call`;
      else if (row)
        td.title = `Measured in the ${row.baseline ? "baseline" : "current"} run only`;
      tr.append(td);
    }
    tr.append(stepsCell);
    tbody.append(tr);
  }
  symbolsTable.replaceChildren(thead, tbody);
  updateSymbolsOpen();
}

// Goes to pass `index` (by default the last whose dump shows the symbol) and
// selects the symbol's node, which also marks its line in the source.
function showSymbol(record, index = record.lastDump) {
  if (index < 0) return;
  selectEvent(index);
  const node = findSymbolNode(renderer.snapshot, record.path);
  if (node >= 0) renderer.select(node, { center: true });
  else setStatus(`${record.path} is not drawn in this pass`, { error: true });
}

// The status-bar entry: "@ symbols", or how many kernels got slower when a
// baseline and current run are compared.
function updateSymbolsOpen() {
  symbolsOpen.hidden = !trace;
  if (!trace) return;
  const view = benchmarkView();
  if (!view?.both) {
    symbolsOpen.textContent = "@ symbols";
    return;
  }
  const threshold = Number(symbolsMinChange.value) || 0;
  const slower = [...view.comparison.values()].filter(
    (row) => row.change !== null && row.change * 100 > threshold,
  ).length;
  symbolsOpen.textContent = `@ ${slower} slower`;
}

function renderUnmatched(unmatched, labelSlots) {
  symbolsUnmatched.replaceChildren(
    ...unmatched.map(({ entry, candidates, slot }) => {
      const item = document.createElement("li");
      const name = document.createElement("code");
      name.textContent = entry.kernel;
      const reason = entry.symbol
        ? ` · symbol ${entry.symbol} is not in the trace`
        : candidates.length
          ? ` · ambiguous: ${candidates.join(", ")}; add a symbol column`
          : " · no symbol with this name";
      item.append(
        labelSlots ? `${slot}: ` : "",
        name,
        ` ${formatDuration(entry.timeNs)}${reason}`,
      );
      return item;
    }),
  );
}

function exportSymbols() {
  if (!traceSymbols) return;
  download(
    `${slug(sourceName.textContent.replace(/ ●$/, ""))}-symbols.json`,
    historyToJSON(
      sourceName.textContent,
      trace.events,
      traceSymbols,
      describeEvent,
      (() => {
        const view = benchmarkView();
        return view && comparisonToJSON(view.comparison);
      })(),
    ),
    "application/json",
  );
}

symbolsOpen.addEventListener("click", openSymbols);
symbolsFilter.addEventListener("input", renderSymbols);
symbolsChanged.addEventListener("change", renderSymbols);
symbolsAtPass.addEventListener("change", renderSymbols);
document
  .getElementById("symbols-export")
  .addEventListener("click", exportSymbols);
document
  .getElementById("symbols-bench-current")
  .addEventListener("click", () => importBenchmarks("current"));
document
  .getElementById("symbols-bench-baseline")
  .addEventListener("click", () => importBenchmarks("baseline"));
symbolsMinChange.addEventListener("input", () => {
  symbolsMinChangeOn.checked = true;
  renderSymbolsSummary();
  renderSymbols();
});
symbolsMinChangeOn.addEventListener("change", renderSymbols);

// ---- Visualization tabs and the GPU view ---------------------------------------

// Picking an access marks its line (1-based) in the current source and opens
// the inspector's Line tab, which holds its answer card and its path into the
// GPU. With a pass trace open, the answer card follows the picked access
// across passes and a click on a pass selects it.
const GPU_VIEW_OPTIONS = {
  onLine: (line) => {
    showSourceLine(line - 1);
    inspect("line", document.activeElement);
  },
  onAnswer: setGpuAnswer,
  passes: () =>
    trace?.events.length > 1
      ? {
          events: trace.events,
          current: traceIndex,
          select: selectEvent,
          describe: describeEvent,
        }
      : null,
};

// The answer card of the picked access lives in the Line tab; the bar above
// the GPU view keeps a one-line summary of it. Each launch picks an access
// when drawn; the first launch's pick stands until the person picks another.
let gpuAnswer = null;
function setGpuAnswer(card, judged, { initial }) {
  if (initial && gpuAnswer?.card.isConnected) return;
  gpuAnswer = { card, judged };
  gpuAnswerHost.replaceChildren(card);
  renderVerdictChip();
}

function renderVerdictChip() {
  const on = canvasView === "gpu" && !!gpuAnswer;
  vizVerdict.hidden = !on;
  gpuAnswerSection.hidden = !on;
  if (on) vizVerdict.replaceChildren(...verdictSummary(gpuAnswer.judged));
  placeAnswer();
}

// The answer card leads the Line tab when it is about the explained line
// (an access was picked), and follows the explanation otherwise.
function placeAnswer() {
  const first = gpuAnswer && gpuAnswer.judged.access.line === explainedLine + 1;
  leBody.parentElement.insertBefore(gpuAnswerSection, first ? leBody : gpuPath);
}

vizVerdict.addEventListener("click", () => inspect("line", vizVerdict));

// Redraws the GPU view, dropping the last answer (the cards are rebuilt).
function drawGpuView(options = GPU_VIEW_OPTIONS) {
  gpuAnswer = null;
  gpuAnswerHost.replaceChildren();
  renderGpuView(gpuViewEl, gpuModel, options);
  renderVerdictChip();
}

// The path the focused line takes into the GPU, under its answer card.
function renderPath() {
  const on = canvasView === "gpu" && !!gpuModel;
  gpuPath.hidden = !on;
  if (on) renderGpuPath(gpuPath, GPU_VIEW_OPTIONS);
}

// Traces 0-based source line `index` in the GPU view (3D IR wall, Memory
// accesses, GPU path) when that view is open.
function focusGpu(index) {
  if (canvasView !== "gpu" || !gpuModel) return;
  focusGpuLine(index >= 0 ? index + 1 : null);
  renderPath();
}

// Shows the GPU tab only when the rendered IR has GPU code, opens the
// preferred view, and redraws the GPU view when it is open.
function setGpuModel(model) {
  gpuModel = model;
  viewGpu.hidden = !model;
  if (!model) {
    if (canvasView === "gpu") setCanvasView("graph", { remember: false });
  } else if (canvasView === "gpu") {
    drawGpuView();
    focusGpu(selection.state.line);
  } else if (canvasPreference === "gpu") {
    setCanvasView("gpu", { remember: false });
  }
}

// Shows the Local memory tab when the rendered IR has local buffers or DMAs,
// and opens it unless the person picked another view.
function setLocalModel(model) {
  localModel = hasLocalMemory(model) ? model : null;
  viewLocal.hidden = !localModel;
  if (!localModel) {
    if (canvasView === "local") setCanvasView("graph", { remember: false });
  } else if (canvasView === "local") {
    drawLocalView();
  } else if (localWanted) {
    setCanvasView("local", { remember: false });
  }
}

function drawLocalView() {
  if (!localModel) return;
  const event = trace && traceIndex >= 0 ? trace.events[traceIndex] : null;
  renderLocalView(localViewEl, localModel, {
    pass: event ? `Pass ${event.index + 1} · ${describeEvent(event)}` : "",
    onPick: (line, from) => {
      showSourceLine(line);
      inspect("line", from);
    },
  });
  focusLocalLine(localViewEl, selection.state.line);
}

// The views of the visualization pane, one tab each ([data-view] in
// #viz-tabs). A new view registers here: `available()` says whether the
// current IR has anything to show in it, and `show(on)` draws or hides it.
const VIEWS = {
  graph: {
    available: () => true,
    show(on) {
      if (on) renderer.requestDraw();
    },
  },
  gpu: {
    available: () => !!gpuModel,
    show(on) {
      stage.classList.toggle("gpu-mode", on);
      gpuViewEl.hidden = !on;
      if (on) drawGpuView();
    },
  },
  descent: {
    available: () => !!trace && trace.events.length > 1,
    show(on) {
      stage.classList.toggle("descent-mode", on);
      descentEl.hidden = !on;
      if (on) drawDescent();
    },
  },
  local: {
    available: () => !!localModel,
    show(on) {
      stage.classList.toggle("local-mode", on);
      localViewEl.hidden = !on;
      if (on) drawLocalView();
    },
  },
};
const vizTabs = [...document.querySelectorAll("#viz-tabs [data-view]")];

// `remember` records the choice as the preference, for the person's own
// toggles; automatic switches pass false.
function setCanvasView(view, { remember = true } = {}) {
  if (remember && gpuModel && view !== "local" && view !== "descent") {
    canvasPreference = view === "gpu" ? "gpu" : "graph";
    try {
      localStorage.setItem(CANVAS_KEY, canvasPreference);
    } catch {}
  }
  canvasView = VIEWS[view]?.available() ? view : "graph";
  for (const tab of vizTabs) {
    const on = tab.dataset.view === canvasView;
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
  }
  for (const [name, entry] of Object.entries(VIEWS))
    entry.show(name === canvasView);
  renderVerdictChip();
  renderPath();
  if (canvasView === "gpu") focusGpu(selection.state.line);
  renderExplain(explainedLine);
}

for (const tab of vizTabs) {
  tab.addEventListener("click", () => {
    if (localModel) localWanted = tab.dataset.view === "local";
    setCanvasView(tab.dataset.view);
  });
  // Arrow keys move between the views (WAI-ARIA tabs pattern).
  tab.addEventListener("keydown", (e) => {
    const shown = vizTabs.filter((t) => !t.hidden);
    const at = shown.indexOf(tab);
    const next = { ArrowRight: at + 1, ArrowLeft: at - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    const target = shown[(next + shown.length) % shown.length];
    if (localModel) localWanted = target.dataset.view === "local";
    setCanvasView(target.dataset.view);
    target.focus();
  });
}

// An op picked in the graph opens the inspector (Changes, or Line if that is
// the tab on screen).
canvas.addEventListener("click", () => {
  if (renderer.selected >= 0) inspect("changes", canvas);
});

// ---- What's this line ------------------------------------------------------

// The inspector's Line tab explains the line of the current source that was
// clicked, reached with the arrow keys, or marked from the graph or the GPU
// view. With the GPU view open, the picked access's answer card and its path
// into the GPU follow the explanation.
let explainedLine = -1;

const VERDICT_MEANING = {
  coalesced:
    "Neighboring threads use neighboring items, so the GPU fetches them in one trip.",
  strided:
    "Neighboring threads use items far apart, so the GPU fetches much more than it uses.",
  broadcast:
    "Every thread uses the same item, so it is fetched once and shared.",
  "conflict-free":
    "Every thread gets its own shared-memory counter (bank), so nobody waits.",
};

function caretLine() {
  return input.value.slice(0, input.selectionStart).split("\n").length - 1;
}

function partElement(tag, part, text) {
  const element = document.createElement(tag);
  element.className = `k-${part.kind}`;
  element.textContent = text;
  return element;
}

// The line with each recognized part underlined in its color.
function markedLineElement(line, parts) {
  const pre = document.createElement("div");
  pre.className = "le-line";
  let cursor = 0;
  for (const part of parts) {
    const at = line.indexOf(part.text, cursor);
    if (at < 0) continue;
    pre.append(line.slice(cursor, at), partElement("mark", part, part.text));
    cursor = at + part.text.length;
  }
  pre.append(line.slice(cursor));
  return pre;
}

// What the GPU view knows about a load or store on line `index` (0-based),
// for when the graph is on screen; with the GPU view open its answer card
// says it instead.
function gpuExplain(index) {
  if (!gpuModel) return null;
  for (const launch of gpuModel.launches) {
    const kernel = gpuModel.kernels[launch.kernel];
    const access = kernel?.accesses?.find((a) => a.line === index + 1);
    if (!access) continue;
    const memref = parseMemref(access.type);
    const space = memorySpace(memref?.space ?? "");
    const result = warpAccess(access, memref, space, {
      defs: kernel.defs,
      args: kernel.args,
      block: launch.block,
      grid: launch.grid,
    });
    const box = document.createElement("div");
    box.className = "le-gpu";
    const head = document.createElement("p");
    if (!result.analyzed) {
      head.textContent = `On the GPU: not worked out, because ${result.reason}.`;
      box.append(head);
    } else {
      const strong = document.createElement("strong");
      strong.textContent = `On the GPU (${space} memory): ${result.verdict.replace("-", " ")}. `;
      head.append(
        strong,
        result.verdict === "bank-conflict"
          ? `Up to ${result.ways} threads queue at the same shared-memory counter (bank), so they are served one after another.`
          : (VERDICT_MEANING[result.verdict] ?? ""),
      );
      const example = document.createElement("p");
      const lanes = result.lanes.slice(0, 3);
      example.append(
        "For example, ",
        ...lanes.flatMap((lane, i) => {
          const code = document.createElement("code");
          code.textContent = `${access.buffer}[${lane.index.join(", ")}]`;
          return [
            i
              ? i === lanes.length - 1
                ? ", and thread "
                : ", thread "
              : "thread ",
            `(${lane.tx}, ${lane.ty}, ${lane.tz}) uses `,
            code,
          ];
        }),
        ".",
      );
      box.append(head, example);
    }
    const show = document.createElement("button");
    show.textContent = "Show in the GPU view";
    show.addEventListener("click", () => {
      setCanvasView("gpu");
      drawGpuView({ ...GPU_VIEW_OPTIONS, focusLine: index + 1 });
      focusGpu(index);
    });
    box.append(show);
    return box;
  }
  return null;
}

// Explains 0-based line `index` of the current source, or shows a hint.
function renderExplain(index) {
  explainedLine = index;
  const lines = input.value.split("\n");
  const explained = index >= 0 ? explainLine(lines, index) : null;
  leDocs.hidden = !explained?.op;
  if (!explained) {
    placeAnswer();
    leTitle.textContent = "What's this line?";
    const hint = document.createElement("p");
    hint.className = "le-hint";
    hint.textContent =
      "Click any line of code, or a box in the diagram, to see what each part of it means.";
    leBody.replaceChildren(hint);
    return;
  }
  leTitle.textContent =
    `Line ${index + 1}` + (explained.op ? ` · ${explained.op}` : "");
  if (explained.op) {
    leDocs.href = explained.docs;
    leDocs.title = `The ${explained.dialect} family in the MLIR documentation`;
  }
  const summary = document.createElement("p");
  summary.className = "le-summary";
  summary.textContent =
    explained.summary ??
    "There is no plain description of this operation yet, but its parts are labeled below.";
  const children = [summary];
  if (explained.parts.length) {
    children.push(markedLineElement(lines[index].trim(), explained.parts));
    const list = document.createElement("dl");
    list.className = "le-parts";
    for (const part of explained.parts) {
      const dt = partElement("dt", part, part.label);
      const dd = document.createElement("dd");
      // Show the part itself unless the sentence already starts with it.
      const code = document.createElement("code");
      code.textContent = part.text;
      if (part.detail.startsWith(part.text)) {
        dd.append(code, part.detail.slice(part.text.length));
      } else {
        dd.append(code, " ", part.detail);
      }
      if (part.ref !== undefined) {
        const go = document.createElement("button");
        go.className = "link-btn";
        go.textContent = `go to line ${part.ref + 1}`;
        go.addEventListener("click", () => showSourceLine(part.ref));
        dd.append(" ", go);
      }
      list.append(dt, dd);
    }
    children.push(list);
  }
  const gpu = canvasView === "gpu" ? null : gpuExplain(index);
  if (gpu) children.push(gpu);
  const local = localExplain(localLineInfo(localModel, index));
  if (local) children.push(local);
  leBody.replaceChildren(...children);
  placeAnswer();
}

// The Line tab is drawn as lines are picked; showing it only needs the path.
function renderLinePanel() {
  renderPath();
}

// A click on a line of the current source explains it in the inspector.
input.addEventListener("click", () => {
  selection.select({ line: caretLine() });
  inspect("line", input);
});
input.addEventListener("keyup", (e) => {
  if (!/^(Arrow|Page|Home|End)/.test(e.key)) return;
  selection.select({ line: caretLine() });
});
// New text (another pass, a paste) invalidates the explained line.
input.addEventListener("input", () => renderExplain(-1));

// ---- Op history ------------------------------------------------------------

const ophNone = document.getElementById("oph-none");
const ophBody = document.getElementById("oph-body");
const ophTitle = document.getElementById("oph-title");
const ophSummary = document.getElementById("oph-summary");
const ophSteps = document.getElementById("oph-steps");

// Every dump parsed once, as what history needs (the engine has one arena, so
// the graph's snapshot is re-parsed afterwards). Uses the text each pass shows,
// so node numbers agree with the graph.
function computeOpModel() {
  const shapes = [];
  const records = trace.events.map((event) => {
    const result = parse(event.ir);
    const ok = result.status === STATUS.OK;
    shapes.push(ok ? snapshotShape(result.snapshot) : null);
    return ok ? opRecords(result.snapshot) : null;
  });
  opShapes = shapes;
  if (renderedText) {
    const restored = parse(renderedText).snapshot;
    renderer.snapshot = restored;
    diffAfter = restored;
    renderer.requestDraw();
  }
  return buildOpModel(records);
}

const OP_CHANGE = {
  created: "created",
  kept: "unchanged",
  renamed: "renamed (lowered)",
  inlined: "inlined from",
  fused: "fused from",
  removed: "removed",
};

// Selects `node` of pass `pass`: the pass loads, then the op is picked in it.
function goToOp(pass, node) {
  // Going to where the op is gone keeps its history up instead of clearing it.
  opPinned = node < 0 ? lastOpEntries : null;
  if (pass !== traceIndex) selectEvent(pass);
  if (node >= 0) renderer.select(node, { center: true });
  else renderOpHistoryPanel();
}
let lastOpEntries = null;

function renderOpHistoryPanel() {
  const hint = (text) => {
    ophNone.textContent = text;
    ophNone.hidden = false;
    ophBody.hidden = true;
  };
  if (!trace || trace.events.length < 2)
    return hint("Open a pass trace to follow an op through the passes.");
  const node = renderer.selected;
  if (node >= 0) opPinned = null;
  if (node < 0 && !opPinned)
    return hint(
      "Pick an op in the graph, or a line in the Source tab, to see where it came from and what each pass did to it.",
    );
  opModel ??= computeOpModel();
  const entries = node >= 0 ? opHistory(opModel, traceIndex, node) : opPinned;
  lastOpEntries = entries;
  if (!entries.length) return hint("This pass's IR did not parse, so there is no history.");

  ophNone.hidden = true;
  ophBody.hidden = false;
  const here = entries.find((e) => e.pass === traceIndex) ?? entries[0];
  ophTitle.textContent = here.label;
  const passName = (i) => `#${i + 1} ${describeEvent(trace.events[i])}`;
  const made = entries[0];
  const gone = entries.find((e) => e.change === "removed");
  const lowered = entries.filter((e) => e.change === "renamed");
  ophSummary.textContent =
    `${made.change === "created" ? "Made" : made.change === "inlined" ? "Inlined" : "Fused"} at ${passName(made.pass)}` +
    (lowered.length ? `; lowered at #${lowered.map((e) => e.pass + 1).join(", #")}` : "") +
    (gone ? `; removed at #${gone.pass + 1}` : "; still there at the last pass") +
    (here.loc ? ` · loc ${here.loc}` : "");

  ophSteps.replaceChildren(
    ...entries.map((entry) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = `oph-step ${entry.change}`;
      if (entry.pass === traceIndex) button.setAttribute("aria-current", "step");
      const what = document.createElement("div");
      what.className = "what";
      what.textContent = `${passName(entry.pass)} · ${OP_CHANGE[entry.change]}`;
      const op = document.createElement("div");
      op.className = "op";
      op.textContent = entry.label;
      button.append(what, op);
      button.addEventListener("click", () => goToOp(entry.pass, entry.node));
      item.append(button);
      if (entry.from.length && ["inlined", "fused"].includes(entry.change)) {
        const parts = document.createElement("ul");
        parts.className = "oph-from";
        for (const from of entry.from) {
          const part = document.createElement("li");
          const go = document.createElement("button");
          go.type = "button";
          go.textContent = `${from.label} (#${from.pass + 1})`;
          go.addEventListener("click", () => goToOp(from.pass, from.node));
          part.append(go);
          parts.append(part);
        }
        item.append(parts);
      }
      return item;
    }),
  );
}

selection.subscribe((state, picked) => {
  if (picked.includes("node") && inspector.isOpen && inspector.active === "ophistory")
    renderOpHistoryPanel();
});

// ---- Descent view ------------------------------------------------------------

// The passes as layers in a 3D stack (src/render/descent.js). It reads the
// same op model as the History tab: the layers come from each pass's parse,
// the path of the selected op from its history, and the lit line from the
// selection.
const descent = createDescentView(descentEl, {
  onPickNode: goToOp,
  onPickPass: (pass) => pass !== traceIndex && selectEvent(pass),
  onStep(delta) {
    if (!trace) return;
    const last = trace.events.length - 1;
    const target = Math.max(0, Math.min(last, Number.isFinite(delta) ? traceIndex + delta : delta > 0 ? last : 0));
    if (target !== traceIndex) selectEvent(target);
  },
  onGpu: () => setCanvasView("gpu", { remember: false }),
});
let descentFor = null;

// The Descent tab is there whenever a trace has more than one pass.
function updateDescentTab() {
  const has = !!trace && trace.events.length > 1;
  viewDescent.hidden = !has;
  if (!has) {
    descent.setData(null);
    descentFor = null;
    if (canvasView === "descent") setCanvasView("graph", { remember: false });
  }
}

function drawDescent() {
  if (!trace) return;
  opModel ??= computeOpModel();
  if (descentFor !== opModel) {
    descent.setData({
      shapes: opShapes,
      titles: trace.events.map((event) => `#${event.index + 1} ${describeEvent(event)}`),
      names: trace.events.map((event) => describeEvent(event).replace(/ · .*/, "")),
      labelOf: (pass, node) => opModel.records[pass]?.[node]?.label ?? "",
    });
    descentFor = opModel;
  }
  refreshDescent();
}

function refreshDescent() {
  if (canvasView !== "descent" || !trace || !opModel || descentFor !== opModel) return;
  const node = renderer.selected;
  const line = selection.state.line;
  descent.setState({
    pass: traceIndex,
    node,
    lineage: node >= 0 ? opHistory(opModel, traceIndex, node) : (opPinned ?? []),
    lineNodes: line >= 0 ? nodesOnLine(opShapes[traceIndex], line + 1) : [],
    gpu: !!gpuModel,
  });
}

selection.subscribe((_, picked) => {
  if (picked.includes("node") || picked.includes("line")) refreshDescent();
});

// ---- Symbol view -----------------------------------------------------------

const symbolView = document.getElementById("symbol-view");
const symbolViewTitle = document.getElementById("symbol-view-title");
const symbolViewMeta = document.getElementById("symbol-view-meta");
const symbolViewSteps = document.getElementById("symbol-view-steps");
const symbolViewStep = document.getElementById("symbol-view-step");
const symbolViewCode = document.getElementById("symbol-view-code");
const symbolViewTabs = {
  diff: document.getElementById("sv-tab-diff"),
  ir: document.getElementById("sv-tab-ir"),
  asm: document.getElementById("sv-tab-asm"),
};
// { record, steps, current, mode } for the open symbol view.
let symbolViewState = null;

function openSymbolView(record) {
  if (!trace) return;
  const steps = symbolTimeline(trace.events, record.path);
  if (!steps.length) {
    setStatus(`${record.path} has no IR in this trace`, { error: true });
    return;
  }
  // Start at the step for the pass on screen, if it touched the symbol.
  const here = steps.findIndex((step) => step.index === traceIndex);
  symbolViewState = {
    record,
    steps,
    current: here >= 0 ? here : 0,
    mode: "diff",
  };
  renderSymbolViewHead();
  renderSymbolView();
  symbolView.showModal();
  symbolViewCode.focus();
}

function renderSymbolViewHead() {
  const { record, steps } = symbolViewState;
  symbolViewTitle.textContent = record.path;
  const facts = [
    record.ops.join(" → "),
    `${steps.length} step(s)`,
    record.removed ? "removed by the end" : "",
  ];
  const row = benchmarkView()?.comparison.get(record.path);
  if (row?.current || row?.baseline) {
    const time = (slot) =>
      row[slot] && `${slot} ${formatDuration(row[slot].timeNs)}`;
    facts.push(
      [time("baseline"), time("current")].filter(Boolean).join(" → ") +
        (row.change !== null ? ` (${formatChange(row.change)})` : "") +
        (benchmarks.current?.mock || benchmarks.baseline?.mock
          ? " · mock data"
          : ""),
    );
  }
  symbolViewMeta.textContent = facts.filter(Boolean).join(" · ");

  symbolViewSteps.replaceChildren(
    ...steps.map((step, i) => {
      const event = trace.events[step.index];
      const item = document.createElement("li");
      const button = document.createElement("button");
      const name = document.createElement("span");
      name.textContent = `#${step.index + 1} ${event.argument || event.pass}`;
      const kind = document.createElement("span");
      kind.className = `kind ${step.kind}`;
      kind.textContent =
        step.kind === "lowered"
          ? `lowered ${step.from} → ${step.op}`
          : step.kind === "initial"
            ? `first dump · ${step.op}`
            : `${step.kind} · ${step.op}`;
      button.append(name, kind);
      button.title = `${step.index + 1}. ${describeEvent(event)}`;
      button.addEventListener("click", () => {
        symbolViewState.current = i;
        renderSymbolView();
      });
      item.append(button);
      return item;
    }),
  );
}

// The IR of the latest step at or before `i` that has any.
function stepText(i) {
  for (let k = i; k >= 0; k--)
    if (symbolViewState.steps[k].text !== null)
      return symbolViewState.steps[k].text;
  return null;
}

function renderSymbolView() {
  const { steps, current } = symbolViewState;
  const step = steps[current];
  const event = trace.events[step.index];
  [...symbolViewSteps.children].forEach((item, i) => {
    const button = item.firstChild;
    if (i === current) {
      button.setAttribute("aria-current", "step");
      button.scrollIntoView({ block: "nearest" });
    } else button.removeAttribute("aria-current");
  });

  const assembly = step.assembly ?? [];
  const isNvvm = assembly.some((object) => object.target.startsWith("#nvvm"));
  symbolViewTabs.asm.hidden = !assembly.length;
  symbolViewTabs.asm.textContent = isNvvm ? "PTX" : "Assembly";
  // Fall back to a mode this step can show.
  let mode = symbolViewState.mode;
  if (mode === "asm" && !assembly.length) mode = "diff";
  if (mode === "diff" && step.text === null && assembly.length) mode = "asm";
  for (const [name, tab] of Object.entries(symbolViewTabs))
    tab.setAttribute("aria-selected", String(name === mode));

  const before = current > 0 ? stepText(current - 1) : null;
  const lines = [];
  const note = (text) => lines.push({ type: "note", text });
  if (mode === "asm") {
    for (const object of assembly) {
      note(`${object.target}`);
      for (const text of object.text.replace(/\n$/, "").split("\n"))
        lines.push({ type: "plain", text });
    }
  } else if (step.text === null) {
    note(
      `Removed by #${step.index + 1} ${describeEvent(event)}.` +
        (assembly.length
          ? " Its code is now embedded in the enclosing binary; see the assembly tab."
          : ""),
    );
    if (mode === "diff" && before) lines.push(...lineDiff(before, null));
  } else if (mode === "diff" && before !== null) {
    const diff = lineDiff(before, step.text);
    const { added, removed } = diffStats(diff);
    note(
      added || removed
        ? `+${added} −${removed} lines against the previous step`
        : "Same text as the previous step",
    );
    lines.push(...diff);
  } else {
    if (mode === "diff")
      note("First appearance, so there is nothing to compare.");
    for (const text of step.text.split("\n"))
      lines.push({ type: "same", text });
  }

  symbolViewStep.textContent = `#${step.index + 1} ${describeEvent(event)}`;
  symbolViewCode.replaceChildren(
    ...lines.map((line) => {
      const div = document.createElement("div");
      div.className = `line ${line.type}`;
      if (line.type === "note" || line.type === "plain") {
        // A space keeps blank lines one line tall.
        div.textContent = line.text || " ";
      } else {
        const sign =
          line.type === "add" ? "+" : line.type === "del" ? "−" : " ";
        div.innerHTML = `${sign} ${highlightMlir(line.text).slice(0, -1)}`;
      }
      return div;
    }),
  );
  symbolViewCode.scrollTop = 0;
}

function setSymbolViewMode(mode) {
  if (!symbolViewState) return;
  if (mode === "asm" && symbolViewTabs.asm.hidden) return;
  symbolViewState.mode = mode;
  renderSymbolView();
}

function stepSymbolView(delta) {
  if (!symbolViewState) return;
  const next = symbolViewState.current + delta;
  if (next < 0 || next >= symbolViewState.steps.length) return;
  symbolViewState.current = next;
  renderSymbolView();
}

for (const [mode, tab] of Object.entries(symbolViewTabs))
  tab.addEventListener("click", () => setSymbolViewMode(mode));
symbolView.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const action = {
    j: () => stepSymbolView(1),
    ArrowDown: () => stepSymbolView(1),
    k: () => stepSymbolView(-1),
    ArrowUp: () => stepSymbolView(-1),
    d: () => setSymbolViewMode("diff"),
    i: () => setSymbolViewMode("ir"),
    a: () => setSymbolViewMode("asm"),
  }[e.key];
  if (!action) return;
  e.preventDefault();
  action();
});
document.getElementById("symbol-view-go").addEventListener("click", () => {
  const { record, steps, current } = symbolViewState;
  // A removed symbol is shown in the last dump that still has it.
  const step = steps[current];
  const index =
    step.text !== null
      ? step.index
      : record.lastDump >= 0 && record.lastDump < step.index
        ? record.lastDump
        : Math.max(0, step.index - 1);
  symbolView.close();
  showSymbol(record, index);
});
document.getElementById("symbol-view-back").addEventListener("click", () => {
  symbolView.close();
  openSymbols();
});
symbolView.addEventListener("click", (e) => {
  if (e.target === symbolView) symbolView.close();
});

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
renderExplain(-1);
renderPath();
updateWatchUi();

// An autosave of an emptied workspace would open blank forever; show the
// first sample instead (a GPU trace), or the inline one if it cannot be
// fetched.
const saved = await kv.get("autosave");
const hasContent = (state) =>
  !!(state?.trace || state?.baseline?.trim() || state?.current?.trim());
if (hasContent(saved)) {
  applyState(saved);
} else {
  const [first] = SAMPLES;
  try {
    applyState(await loadSampleState(first, fetchSampleText));
  } catch {
    applyState(await loadSampleState(SAMPLES.find((sample) => sample.inline)));
  }
}
restored = true;
try {
  if (localStorage.getItem("vizmlir.debug") === "1")
    setDebug(true, { save: false });
} catch {}
if (inspectorSaved.open) inspector.open();
restoreWatch();
