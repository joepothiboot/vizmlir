import { PREVIEW_DELAY_MS } from "../constants.js";
import { createScrubber } from "../app/index.js";
import { STATUS, copySnapshot, diffSnapshots } from "../ir/index.js";
import {
  baselineFor,
  byteLength,
  describeEvent,
  formatBytes,
  formatSeconds,
  matchTiming,
  parsePassTrace,
} from "../trace/index.js";
import { refreshBreakpoints, updateDebugBar } from "./debugger.js";
import {
  baseline,
  baselineTitle,
  currentTitle,
  input,
  setStatus,
} from "./dom.js";
import { parse } from "./engine.js";
import { renderer } from "./graph.js";
import {
  OP_MARK_TEXT,
  refreshDescent,
  updateDescentTab,
} from "./op-history.js";
import { run } from "./pipeline.js";
import { selection, state } from "./state.js";
import { updateSymbolsOpen } from "./symbol-history.js";
import { describeMatch, setProfile } from "./timing-panel.js";

const diagsEl = document.getElementById("diags");
const passScrubber = document.getElementById("pass-scrubber");
const passList = document.getElementById("pass-list");
const passListWrap = document.getElementById("pass-list-wrap");
const passListCount = document.getElementById("pass-list-count");

export function loadTrace(text, { keepIndex = false } = {}) {
  const previousIndex = state.traceIndex;
  state.traceText = text;
  state.trace = parsePassTrace(text);
  state.traceCounts = null;
  state.traceBuffers = null;
  state.traceSymbols = null;
  state.opModel = null;
  state.opShapes = null;
  state.opPinned = null;
  state.bpHits = new Map();
  scrubber.setMarks([]);
  scrubber.setOpMarks([]);
  state.opMarkInfo = new Map();

  if (!state.trace.events.length) {
    clearTrace();
    setStatus("no IR dumps found in trace", { error: true });

    return;
  }

  setProfile(
    state.trace.timing,
    state.trace.memory,
    matchTiming(state.trace.events, state.trace.timing),
  );

  const slowest = Math.max(
    0,
    ...state.profile.matches.map((match) => match?.row.wall.seconds ?? 0),
  );

  passChanges = new Map();

  scrubber.setPasses(
    state.trace.events.map((event) => {
      const match = state.profile.matches[event.index];

      return {
        failed: event.failed,
        share: match && slowest > 0 ? match.row.wall.seconds / slowest : null,
      };
    }),
  );

  renderPassList(slowest);
  passScrubber.hidden = false;

  const firstFailure = state.trace.events.findIndex((event) => event.failed);

  let start = Math.max(0, firstFailure);

  if (keepIndex && previousIndex >= 0) {
    start = Math.min(previousIndex, state.trace.events.length - 1);
  }

  selectEvent(start);

  updateSymbolsOpen();
  updateDescentTab();
  refreshBreakpoints({ defer: true });
}

export function selectEvent(index) {
  const event = state.trace.events[index];
  const base = baselineFor(state.trace.events, index);
  state.traceIndex = index;
  selection.select({ pass: index });
  scrubber.setValue(index);

  [...passList.children].forEach((item, i) => {
    const button = item.firstChild;
    if (i === index) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });

  if (passListWrap.open) {
    passList.children[index]?.scrollIntoView({ block: "nearest" });
  }

  baseline.value = base?.ir ?? "";
  input.value = event.ir;
  baseline.dispatchEvent(new Event("input"));
  input.dispatchEvent(new Event("input"));
  clearTimeout(state.timer);

  baselineTitle.textContent = base
    ? `Baseline · ${base.reconstructed ? "rebuilt through" : "from"} #${base.event.index + 1}`
    : "Baseline · no earlier snapshot";

  currentTitle.textContent = `Current · #${index + 1} ${describeEvent(event)}`;
  state.traceDiffTitle = `diff ${base ? `#${base.event.index + 1}` : "∅"} → #${index + 1}`;

  const match = state.profile.matches[index];
  const size = byteLength(event.ir);
  const growth = base ? size - byteLength(base.ir) : 0;

  state.traceNote =
    `pass ${index + 1}/${state.trace.events.length}${event.failed ? " · ✗ FAILED" : ""}` +
    (match ? ` · ${describeMatch(match)}` : "") +
    ` · IR ${formatBytes(size)}` +
    (base && growth
      ? ` (${growth > 0 ? "+" : "−"}${formatBytes(Math.abs(growth))})`
      : "") +
    ` · trace line ${event.headerLine} · `;

  renderDiagnostics(
    event.index === state.trace.events.length - 1
      ? [
          ...event.diagnostics,
          ...state.trace.diagnostics.filter((d) => d.eventIndex < 0),
        ]
      : event.diagnostics,
  );

  run();
  updateDebugBar();
  refreshDescent();
}

function renderPassList(slowest) {
  const width = Math.max(2, String(state.trace.events.length).length);

  passList.replaceChildren(
    ...state.trace.events.map((event) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      const flags = `${event.failed ? " ✗" : ""}${event.diagnostics.length ? ` · ${event.diagnostics.length} diag` : ""}`;
      const number = document.createElement("span");
      number.className = "n";
      number.textContent = String(event.index + 1).padStart(width, "0") + flags;

      const name = document.createElement("span");
      name.className = "name";
      name.textContent = describeEvent(event);

      const match = state.profile.matches[event.index];
      const meta = document.createElement("span");
      meta.className = "meta";

      meta.textContent =
        (match ? `${formatSeconds(match.row.wall.seconds)} · ` : "") +
        formatBytes(byteLength(event.ir));

      button.append(number, name, meta);

      if (match && slowest > 0) {
        button.style.setProperty(
          "--time",
          `${(100 * match.row.wall.seconds) / slowest}%`,
        );
      }

      button.title =
        `${event.index + 1}. ${describeEvent(event)}${event.failed ? " ✗ failed" : ""}` +
        (match ? `\n${describeMatch(match)}` : "");

      button.classList.toggle("failed", event.failed);
      button.addEventListener("click", () => selectEvent(event.index));
      item.append(button);

      return item;
    }),
  );

  passListCount.textContent = `(${state.trace.events.length})`;
  passListWrap.hidden = false;
}

let passChanges = new Map();

export function changedOpsAt(index) {
  if (index === state.traceIndex) {
    const counts = { added: 0, removed: 0, changed: 0 };
    for (const row of state.diffRows) counts[row.type] += 1;

    return counts;
  }

  if (passChanges.has(index)) return passChanges.get(index);

  const base = baselineFor(state.trace.events, index);
  let counts = null;
  const before = base ? parse(base.ir) : null;

  if (!before || before.status === STATUS.OK) {
    const beforeCopy = before && copySnapshot(before.snapshot);
    const after = parse(state.trace.events[index].ir);

    if (after.status === STATUS.OK) {
      counts = { added: 0, removed: 0, changed: 0 };

      for (const row of diffSnapshots(beforeCopy, after.snapshot)) {
        counts[row.type] += 1;
      }
    }
  }

  if (state.renderedText) {
    const restored = parse(state.renderedText).snapshot;
    renderer.snapshot = restored;
    state.diffAfter = restored;
    renderer.requestDraw();
  }

  passChanges.set(index, counts);

  return counts;
}

let previewTimer = 0;

export const scrubber = createScrubber(passScrubber, {
  onSelect: (index) =>
    state.trace && index !== state.traceIndex && selectEvent(index),
  label: (index) => {
    const event = state.trace?.events[index];
    if (!event) return "";

    return (
      `Pass ${index + 1} of ${state.trace.events.length}: ${describeEvent(event)}` +
      (event.failed ? " · failed" : "")
    );
  },
  preview: (index) => {
    const event = state.trace.events[index];
    const base = baselineFor(state.trace.events, index);
    const size = byteLength(event.ir);
    const growth = base ? size - byteLength(base.ir) : 0;
    const match = state.profile.matches[index];

    const lines = [
      `IR ${formatBytes(size)}` +
        (growth
          ? ` (${growth > 0 ? "+" : "−"}${formatBytes(Math.abs(growth))})`
          : ""),
    ];

    const known = index === state.traceIndex || passChanges.has(index);
    const counts = known ? changedOpsAt(index) : undefined;

    if (counts === undefined) {
      lines.push("changed ops: …");
      clearTimeout(previewTimer);

      previewTimer = setTimeout(() => {
        if (!state.trace?.events[index]) return;
        changedOpsAt(index);
        scrubber.refreshPreview(index);
      }, PREVIEW_DELAY_MS);
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

    const life = state.opMarkInfo.get(index);
    if (life) lines.push(`selected op: ${OP_MARK_TEXT[life]}`);
    if (match) lines.push(describeMatch(match));
    if (event.failed) lines.push("✗ failed");

    return { title: `${index + 1}. ${describeEvent(event)}`, lines };
  },
});

export function clearTrace() {
  state.trace = null;
  state.traceCounts = null;
  state.traceBuffers = null;
  state.traceSymbols = null;
  state.opModel = null;
  state.opShapes = null;
  state.opPinned = null;
  state.benchmarks = { baseline: null, current: null };
  state.measured = null;
  state.traceText = "";
  state.traceNote = "";
  state.traceIndex = -1;
  state.traceDiffTitle = "";
  updateDebugBar();
  updateDescentTab();
  passScrubber.hidden = true;
  state.bpHits = new Map();
  scrubber.setMarks([]);
  scrubber.setOpMarks([]);
  state.opMarkInfo = new Map();
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
