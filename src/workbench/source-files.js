import { createSourceView } from "../app/index.js";
import { findSource, linesOf, nodesAt, opCounts } from "../trace/index.js";
import { setCanvasView } from "./canvas-views.js";
import {
  baseline,
  input,
  setStatus,
  sourceName,
  sourcePane,
  sourcePanel,
  sourceTitle,
} from "./dom.js";
import { renderer } from "./graph.js";
import { positionLineMark, setSourceSplit, showTab } from "./source-tabs.js";
import { selection, state } from "./state.js";
import { scheduleAutosave } from "./workspace.js";

const sourceView = createSourceView(sourcePanel, {
  onPick: pickSourceLine,
  onAdd: addSourceFiles,
});

let sourceCycle = { key: "", at: -1 };

export function sourceNote(index) {
  const entry = state.locIndex?.byNode.get(index);
  if (!entry) return "";

  const { file, line, col } = entry.positions[0];

  return ` · ${file}:${line}:${col}`;
}

export function refreshSources() {
  const named = state.locIndex?.files ?? [];
  const has = named.length > 0;
  sourceTitle.hidden = !has;

  if (!has) {
    if (sourcePane.classList.contains("split-source")) {
      setSourceSplit(false);
    } else if (sourceTitle.getAttribute("aria-selected") === "true") {
      showTab(input);
    }
  }

  const missing = [
    ...new Set(named.filter((f) => !findSource(state.sources, f))),
  ];

  sourceView.setSources(state.sources, missing);
  syncSourceView(renderer.selected);
}

function syncSourceView(node) {
  const hit =
    node >= 0 && state.locIndex ? linesOf(state.locIndex, node) : null;

  const key = hit && findSource(state.sources, hit.file);
  if (key) sourceView.mark(key, hit.lines);
  else sourceView.mark(null, []);

  sourceView.setCounts(
    state.locIndex && sourceView.file
      ? opCounts(state.locIndex, sourceView.file)
      : new Map(),
  );
}

function pickSourceLine(file, line) {
  const nodes = state.locIndex ? nodesAt(state.locIndex, file, line) : [];

  if (!nodes.length) {
    sourceView.mark(file, [line]);
    setStatus(`${file}:${line} has no ops in this pass`);

    return;
  }

  const key = `${file}:${line}`;
  const at = sourceCycle.key === key ? (sourceCycle.at + 1) % nodes.length : 0;
  sourceCycle = { key, at };
  if (state.canvasView !== "graph") setCanvasView("graph", { remember: false });
  renderer.select(nodes[at], { center: true });

  setStatus(
    `${file}:${line} · op ${at + 1} of ${nodes.length}` +
      (nodes.length > 1 ? " (click again for the next)" : ""),
  );
}

async function addSourceFiles(files) {
  const next = { ...state.sources };
  const skipped = [];

  for (const file of files) {
    const text = file.size > 1024 * 1024 ? null : await file.text();
    if (text === null || text.includes("\0")) skipped.push(file.name);
    else next[file.name] = text;
  }

  state.sources = next;
  refreshSources();
  scheduleAutosave();

  if (skipped.length) {
    setStatus(`not a text file under 1 MiB: ${skipped.join(", ")}`);
  }
}

export function setupSourceFiles() {
  selection.subscribe((selected, picked) => {
    if (picked.includes("node")) syncSourceView(selected.node);
  });

  for (const textarea of [baseline, input]) {
    textarea.addEventListener("input", (e) => {
      if (e.isTrusted && !sourceName.textContent.endsWith(" ●")) {
        sourceName.textContent += " ●";
      }
    });
  }

  input.addEventListener("scroll", positionLineMark);

  input.addEventListener("input", () => {
    selection.clear("line");
    positionLineMark();
  });
}
