import { STORAGE_KEYS } from "../constants.js";
import {
  conditionHits,
  mergeHits,
  nextHit,
  opHistory,
  parseCondition,
} from "../trace/index.js";
import { computeTraceBuffers } from "./buffers-panel.js";
import { setStatus, sourcePane, sourceTitle } from "./dom.js";
import { renderer } from "./graph.js";
import { OP_CHANGE, computeOpModel, goToOp } from "./op-history.js";
import { computeTraceCounts } from "./opcount-panel.js";
import { setSourceSplit, toggleSplit } from "./source-tabs.js";
import { selection, state } from "./state.js";
import { changedOpsAt, scrubber, selectEvent } from "./trace-nav.js";
import { scheduleAutosave } from "./workspace.js";

const debugToggle = document.getElementById("debug-toggle");
const debugBar = document.getElementById("debug-bar");
const debugStatus = document.getElementById("debug-status");

const debugButtons = {
  first: document.getElementById("debug-first"),
  prev: document.getElementById("debug-prev-change"),
  next: document.getElementById("debug-next-change"),
  last: document.getElementById("debug-last"),
};

let debugSplitOwned = "";

function passChanged(index) {
  const counts = changedOpsAt(index);

  return !counts || counts.added + counts.removed + counts.changed > 0;
}

function changedPassFrom(index, delta) {
  for (
    let i = index + delta;
    i >= 0 && i < state.trace.events.length;
    i += delta
  ) {
    if (passChanged(i)) return i;
  }

  return -1;
}

export function updateDebugBar() {
  debugBar.hidden = !(state.debugOn && state.trace);
  if (debugBar.hidden) return;

  const last = state.trace.events.length - 1;
  debugButtons.first.disabled = state.traceIndex <= 0;
  debugButtons.last.disabled = state.traceIndex >= last;
  debugButtons.prev.disabled = changedPassFrom(state.traceIndex, -1) < 0;
  debugButtons.next.disabled = changedPassFrom(state.traceIndex, 1) < 0;

  const counts = changedOpsAt(state.traceIndex);

  debugStatus.textContent = counts
    ? `pass ${state.traceIndex + 1}: +${counts.added} −${counts.removed} ~${counts.changed} ops`
    : `pass ${state.traceIndex + 1}: IR did not parse`;

  updateOpStepButtons();
  updateBreakpointButtons();
}

export function setDebug(on, { save = true } = {}) {
  state.debugOn = on;
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

  if (save) {
    try {
      localStorage.setItem(STORAGE_KEYS.debug, on ? "1" : "0");
    } catch {}
  }
}

export function stepToChange(delta) {
  if (!state.debugOn || !state.trace) return;

  const index = changedPassFrom(state.traceIndex, delta);
  if (index >= 0) selectEvent(index);
}

const opStepButtons = {
  prev: document.getElementById("debug-op-prev"),
  next: document.getElementById("debug-op-next"),
};

function updateOpStepButtons() {
  const off = !(state.debugOn && state.trace) || renderer.selected < 0;
  opStepButtons.prev.disabled = off;
  opStepButtons.next.disabled = off;
}

export function stepOpChange(delta) {
  if (!state.debugOn || !state.trace) return;

  if (renderer.selected < 0) {
    setStatus("pick an op first, then step to the passes that changed it");

    return;
  }

  state.opModel ??= computeOpModel();

  const entries = opHistory(state.opModel, state.traceIndex, renderer.selected);
  const changes = entries.filter((entry) => entry.change !== "kept");

  const target =
    delta > 0
      ? changes.find((entry) => entry.pass > state.traceIndex)
      : changes.findLast((entry) => entry.pass < state.traceIndex);

  if (!target) {
    setStatus(`no ${delta > 0 ? "later" : "earlier"} pass changes this op`);

    return;
  }

  state.lastOpEntries = entries;
  goToOp(target.pass, target.node);
  debugStatus.textContent = `${OP_CHANGE[target.change]} · pass ${target.pass + 1}`;
}

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

let bpTimer = 0;

function breakpointContext() {
  return {
    events: state.trace.events,
    counts: () => (state.traceCounts ??= computeTraceCounts()),
    peaks: () => (state.traceBuffers ??= computeTraceBuffers()),
  };
}

const activeHits = () =>
  mergeHits(
    state.breakpoints
      .filter((b) => b.on)
      .map((b) => state.bpHits.get(b.text) ?? []),
  );

export function refreshBreakpoints({ defer = false } = {}) {
  clearTimeout(bpTimer);

  const compute = () => {
    state.bpHits = new Map();

    if (state.trace) {
      for (const b of state.breakpoints) {
        const parsed = parseCondition(b.text);

        state.bpHits.set(
          b.text,
          parsed.ok ? conditionHits(parsed.cond, breakpointContext()) : [],
        );
      }
    }

    scrubber.setMarks(activeHits());
    renderBreakpoints();
  };

  if (defer) bpTimer = setTimeout(compute, 0);
  else compute();
}

function updateBreakpointButtons() {
  const hits = activeHits();
  bp.back.disabled = nextHit(hits, state.traceIndex, -1) < 0;
  bp.cont.disabled = nextHit(hits, state.traceIndex, 1) < 0;
}

function renderBreakpoints() {
  bp.count.textContent = state.breakpoints.length
    ? ` (${state.breakpoints.length})`
    : "";

  bp.list.replaceChildren(
    ...state.breakpoints.map((b, i) => {
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

      const passes = state.bpHits.get(b.text) ?? [];

      if (!state.trace) {
        hits.textContent = "open a pass trace";
      } else if (!passes.length) {
        hits.textContent = "never hit";
      } else {
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
        state.breakpoints.splice(i, 1);
        refreshBreakpoints();
        scheduleAutosave();
      });

      item.append(on, text, hits, remove);

      return item;
    }),
  );

  updateBreakpointButtons();
}

export function runToBreakpoint(delta) {
  if (!state.trace) return;

  const hits = activeHits();
  const target = nextHit(hits, state.traceIndex, delta);

  if (target < 0) {
    setStatus(
      state.breakpoints.some((b) => b.on)
        ? `no breakpoint ${delta > 0 ? "after" : "before"} pass ${state.traceIndex + 1}`
        : "no breakpoints set: use Break when…",
    );

    return;
  }

  selectEvent(target);

  const why = state.breakpoints
    .filter((b) => b.on && (state.bpHits.get(b.text) ?? []).includes(target))
    .map((b) => b.text);

  debugStatus.textContent = `break: ${why.join(" · ")} · pass ${target + 1}`;
}

export function setupDebugger() {
  debugToggle.addEventListener("click", () => setDebug(!state.debugOn));

  debugButtons.first.addEventListener("click", () => selectEvent(0));

  debugButtons.last.addEventListener("click", () =>
    selectEvent(state.trace.events.length - 1),
  );

  debugButtons.prev.addEventListener("click", () => stepToChange(-1));

  debugButtons.next.addEventListener("click", () => stepToChange(1));

  opStepButtons.prev.addEventListener("click", () => stepOpChange(-1));

  opStepButtons.next.addEventListener("click", () => stepOpChange(1));

  selection.subscribe((_, picked) => {
    if (picked.includes("node")) updateOpStepButtons();
  });

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

    if (!state.breakpoints.some((b) => b.text === parsed.cond.text)) {
      state.breakpoints.push({ text: parsed.cond.text, on: true });
    }

    bp.input.value = "";
    refreshBreakpoints();
    scheduleAutosave();
  });

  bp.back.addEventListener("click", () => runToBreakpoint(-1));

  bp.cont.addEventListener("click", () => runToBreakpoint(1));
}
