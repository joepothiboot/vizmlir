import { STORAGE_KEYS } from "../constants.js";
import { hasLocalMemory } from "../gpu/index.js";
import {
  focusGpuLine,
  focusLocalLine,
  renderGpuPath,
  renderGpuView,
  renderLocalView,
  verdictSummary,
} from "../render/index.js";
import { describeEvent } from "../trace/index.js";
import { canvas, descentEl, leBody, localViewEl } from "./dom.js";
import { renderer } from "./graph.js";
import { inspect } from "./layout.js";
import { renderExplain } from "./line-explain.js";
import { drawDescent } from "./op-history.js";
import { showSourceLine } from "./source-tabs.js";
import { selection, state } from "./state.js";
import { selectEvent } from "./trace-nav.js";

const gpuPath = document.getElementById("gpu-path");
const gpuAnswerSection = document.getElementById("gpu-answer-section");
const gpuAnswerHost = document.getElementById("gpu-answer-host");
const vizVerdict = document.getElementById("viz-verdict");
const stage = document.getElementById("stage");
const gpuViewEl = document.getElementById("gpu-view");
const viewGpu = document.getElementById("view-gpu");
const viewLocal = document.getElementById("view-local");
let canvasPreference = "gpu";

try {
  if (localStorage.getItem(STORAGE_KEYS.canvas) === "graph") {
    canvasPreference = "graph";
  }
} catch {}

export const GPU_VIEW_OPTIONS = {
  onLine: (line) => {
    showSourceLine(line - 1);
    inspect("line", document.activeElement);
  },
  onAnswer: setGpuAnswer,
  measured: () => state.measured,
  passes: () =>
    state.trace?.events.length > 1
      ? {
          events: state.trace.events,
          current: state.traceIndex,
          select: selectEvent,
          describe: describeEvent,
        }
      : null,
};

let gpuAnswer = null;

function setGpuAnswer(card, judged, { initial }) {
  if (initial && gpuAnswer?.card.isConnected) return;
  gpuAnswer = { card, judged };
  gpuAnswerHost.replaceChildren(card);
  renderVerdictChip();
}

function renderVerdictChip() {
  const on = state.canvasView === "gpu" && !!gpuAnswer;
  vizVerdict.hidden = !on;
  gpuAnswerSection.hidden = !on;
  if (on) vizVerdict.replaceChildren(...verdictSummary(gpuAnswer.judged));
  placeAnswer();
}

export function placeAnswer() {
  const first =
    gpuAnswer && gpuAnswer.judged.access.line === state.explainedLine + 1;

  leBody.parentElement.insertBefore(gpuAnswerSection, first ? leBody : gpuPath);
}

export function drawGpuView(options = GPU_VIEW_OPTIONS) {
  gpuAnswer = null;
  gpuAnswerHost.replaceChildren();
  renderGpuView(gpuViewEl, state.gpuModel, options);
  renderVerdictChip();
}

export function renderPath() {
  const on = state.canvasView === "gpu" && !!state.gpuModel;
  gpuPath.hidden = !on;
  if (on) renderGpuPath(gpuPath, GPU_VIEW_OPTIONS);
}

export function focusGpu(index) {
  if (state.canvasView !== "gpu" || !state.gpuModel) return;
  focusGpuLine(index >= 0 ? index + 1 : null);
  renderPath();
}

export function setGpuModel(model) {
  state.gpuModel = model;
  viewGpu.hidden = !model;

  if (!model) {
    if (state.canvasView === "gpu") setCanvasView("graph", { remember: false });
  } else if (state.canvasView === "gpu") {
    drawGpuView();
    focusGpu(selection.state.line);
  } else if (canvasPreference === "gpu") {
    setCanvasView("gpu", { remember: false });
  }
}

export function setLocalModel(model) {
  state.localModel = hasLocalMemory(model) ? model : null;
  viewLocal.hidden = !state.localModel;

  if (!state.localModel) {
    if (state.canvasView === "local") {
      setCanvasView("graph", { remember: false });
    }
  } else if (state.canvasView === "local") {
    drawLocalView();
  } else if (state.localWanted) {
    setCanvasView("local", { remember: false });
  }
}

function drawLocalView() {
  if (!state.localModel) return;

  const event =
    state.trace && state.traceIndex >= 0
      ? state.trace.events[state.traceIndex]
      : null;

  renderLocalView(localViewEl, state.localModel, {
    pass: event ? `Pass ${event.index + 1} · ${describeEvent(event)}` : "",
    onPick: (line, from) => {
      showSourceLine(line);
      inspect("line", from);
    },
  });

  focusLocalLine(localViewEl, selection.state.line);
}

export const VIEWS = {
  graph: {
    available: () => true,
    show(on) {
      if (on) renderer.requestDraw();
    },
  },
  gpu: {
    available: () => !!state.gpuModel,
    show(on) {
      stage.classList.toggle("gpu-mode", on);
      gpuViewEl.hidden = !on;
      if (on) drawGpuView();
    },
  },
  descent: {
    available: () => !!state.trace && state.trace.events.length > 1,
    show(on) {
      stage.classList.toggle("descent-mode", on);
      descentEl.hidden = !on;
      if (on) drawDescent();
    },
  },
  local: {
    available: () => !!state.localModel,
    show(on) {
      stage.classList.toggle("local-mode", on);
      localViewEl.hidden = !on;
      if (on) drawLocalView();
    },
  },
};

const vizTabs = [...document.querySelectorAll("#viz-tabs [data-view]")];

export function setCanvasView(view, { remember = true } = {}) {
  if (remember && state.gpuModel && view !== "local" && view !== "descent") {
    canvasPreference = view === "gpu" ? "gpu" : "graph";

    try {
      localStorage.setItem(STORAGE_KEYS.canvas, canvasPreference);
    } catch {}
  }

  state.canvasView = VIEWS[view]?.available() ? view : "graph";

  for (const tab of vizTabs) {
    const on = tab.dataset.view === state.canvasView;
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
  }

  for (const [name, entry] of Object.entries(VIEWS)) {
    entry.show(name === state.canvasView);
  }

  renderVerdictChip();
  renderPath();
  if (state.canvasView === "gpu") focusGpu(selection.state.line);
  renderExplain(state.explainedLine);
}

export function setupCanvasViews() {
  vizVerdict.addEventListener("click", () => inspect("line", vizVerdict));

  for (const tab of vizTabs) {
    tab.addEventListener("click", () => {
      if (state.localModel) state.localWanted = tab.dataset.view === "local";
      setCanvasView(tab.dataset.view);
    });

    tab.addEventListener("keydown", (e) => {
      const shown = vizTabs.filter((t) => !t.hidden);
      const at = shown.indexOf(tab);
      const next = { ArrowRight: at + 1, ArrowLeft: at - 1 }[e.key];
      if (next === undefined) return;
      e.preventDefault();

      const target = shown[(next + shown.length) % shown.length];
      if (state.localModel) state.localWanted = target.dataset.view === "local";
      setCanvasView(target.dataset.view);
      target.focus();
    });
  }

  canvas.addEventListener("click", () => {
    if (renderer.selected >= 0) inspect("changes", canvas);
  });
}
