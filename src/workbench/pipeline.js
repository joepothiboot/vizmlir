import { RUN_DELAY_MS } from "../constants.js";
import { bindHighlighting } from "../app/index.js";
import { analyzeGpu, analyzeLocalMemory, localTargetOf } from "../gpu/index.js";
import { STATUS, copySnapshot, diffSnapshots } from "../ir/index.js";
import {
  analyzeBuffers,
  buildLocIndex,
  countOps,
  extractReports,
  isPassTrace,
} from "../trace/index.js";
import { setViewBuffers } from "./buffers-panel.js";
import { setGpuModel, setLocalModel } from "./canvas-views.js";
import { renderDiff } from "./diff-panel.js";
import { baseline, input, setStatus, sourceName } from "./dom.js";
import { engine, parse } from "./engine.js";
import { renderer } from "./graph.js";
import { refreshInspector } from "./layout.js";
import { setViewCounts } from "./opcount-panel.js";
import { refreshSources } from "./source-files.js";
import { state } from "./state.js";
import { mayHaveReports, setProfile } from "./timing-panel.js";
import { loadTrace } from "./trace-nav.js";
import { scheduleAutosave } from "./workspace.js";

const baselineHighlight = document.getElementById("baseline-highlight");
const editorHighlight = document.getElementById("editor-highlight");

export function run() {
  scheduleAutosave();

  if (isPassTrace(input.value)) {
    sourceName.textContent = "pasted trace";
    sourceName.title = sourceName.textContent;
    state.measured = null;
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

  if (!state.trace) {
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
  state.renderedText = current;
  renderer.setSnapshot(snap);
  state.locIndex = buildLocIndex(snap);
  refreshSources();
  setGpuModel(analyzeGpu(current));

  setLocalModel(
    analyzeLocalMemory(current, {
      target:
        state.trace && state.traceIndex >= 0
          ? localTargetOf(state.trace.events, state.traceIndex)
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
    state.traceNote +
      `${snap.nodeCount} nodes · ${snap.edgeCount} edges · ${after.ms.toFixed(2)} ms` +
      (diags.length
        ? ` · ${diags.length} warning(s): ${diags[0].message} "${diags[0].symbol}"`
        : ""),
  );

  refreshInspector();
}

function scheduleRun() {
  clearTimeout(state.timer);
  state.timer = setTimeout(run, RUN_DELAY_MS);
}

export function fitGraph() {
  renderer.fit();
  renderer.requestDraw();
}

export function setupPipeline() {
  bindHighlighting(baseline, baselineHighlight);

  bindHighlighting(input, editorHighlight);

  baseline.addEventListener("input", scheduleRun);

  input.addEventListener("input", scheduleRun);

  document.getElementById("zoom-fit").addEventListener("click", fitGraph);

  document
    .getElementById("zoom-in")
    .addEventListener("click", () => renderer.zoomBy(1.25));

  document
    .getElementById("zoom-out")
    .addEventListener("click", () => renderer.zoomBy(0.8));
}
