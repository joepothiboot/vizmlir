import {
  renderFunctions,
  renderPasses,
  renderSummary,
  signedBytes,
} from "../app/index.js";
import { download, slug } from "../session/index.js";
import {
  analyzeBuffers,
  bufferTotals,
  buffersToJSON,
  compareBuffers,
  describeEvent,
  formatBytes,
  moduleStateAt,
} from "../trace/index.js";
import { sourceName } from "./dom.js";
import { showPanel } from "./layout.js";
import { showSourceLine } from "./source-tabs.js";
import { state } from "./state.js";
import { selectEvent } from "./trace-nav.js";

const buffersOpen = document.getElementById("buffers-open");
let viewBuffers = null;
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

export function setViewBuffers(analyses) {
  viewBuffers = analyses;

  const after = analyses && bufferTotals(analyses.after);
  const before = analyses?.before && bufferTotals(analyses.before);
  buffersOpen.hidden = !after || (!after.buffers && !before?.buffers);
  if (buffersOpen.hidden) return;

  const delta = before ? after.peak - before.peak : 0;

  buffersOpen.textContent =
    `▦ ${formatBytes(after.peak)}` + (delta ? ` (${signedBytes(delta)})` : "");
}

export function computeTraceBuffers() {
  return state.trace.events.map((_, i) =>
    bufferTotals(analyzeBuffers(moduleStateAt(state.trace.events, i))),
  );
}

export function openBuffers() {
  showPanel("buffers");
}

export function renderBuffersPanel() {
  buffersNone.hidden = !!viewBuffers;
  buffersBody.hidden = !viewBuffers;

  if (!viewBuffers) {
    buffersModel = null;

    return;
  }

  const { before, after } = viewBuffers;
  const comparison = compareBuffers(before, after);
  if (state.trace) state.traceBuffers ??= computeTraceBuffers();

  buffersModel = {
    comparison,
    passes: state.trace ? state.traceBuffers : null,
  };

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
  buffersPasses.hidden = !state.trace;
  if (state.trace) renderBufferPasses();
}

function renderBufferPasses() {
  const rows = state.trace.events.map((event, i) => ({
    header: `#${event.index + 1}`,
    title: describeEvent(event),
    totals: state.traceBuffers[i],
  }));

  const { shown, changed } = renderPasses(buffersPassTable, rows, {
    current: state.traceIndex,
    onlyChanged: buffersChangedPasses.checked,
    onPass: selectEvent,
  });

  buffersPassShown.textContent = `${shown} of ${rows.length} dumps · ${changed} change buffers`;
}

function exportBuffers() {
  if (!buffersModel) return;

  const passes = buffersModel.passes?.map((totals, i) => ({
    pass: describeEvent(state.trace.events[i]),
    ...totals,
  }));

  download(
    `${slug(sourceName.textContent.replace(/ ●$/, ""))}-buffers.json`,
    buffersToJSON(sourceName.textContent, buffersModel.comparison, passes),
    "application/json",
  );
}

export function setupBuffersPanel() {
  buffersOpen.addEventListener("click", openBuffers);

  buffersChangedPasses.addEventListener("change", renderBufferPasses);

  document
    .getElementById("buffers-export")
    .addEventListener("click", exportBuffers);
}
