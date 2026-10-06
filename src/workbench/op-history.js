import { STATUS } from "../ir/index.js";
import { createDescentView } from "../render/index.js";
import {
  buildOpModel,
  describeEvent,
  nodesOnLine,
  opHistory,
  opRecords,
  snapshotShape,
} from "../trace/index.js";
import { setCanvasView } from "./canvas-views.js";
import { descentEl } from "./dom.js";
import { parse } from "./engine.js";
import { renderer } from "./graph.js";
import { inspector } from "./layout.js";
import { selection, state } from "./state.js";
import { scrubber, selectEvent } from "./trace-nav.js";

const viewDescent = document.getElementById("view-descent");
const ophNone = document.getElementById("oph-none");
const ophBody = document.getElementById("oph-body");
const ophSliderNote = document.getElementById("oph-slider-note");
const ophTitle = document.getElementById("oph-title");
const ophSummary = document.getElementById("oph-summary");
const ophSteps = document.getElementById("oph-steps");
const OP_MARK_BUDGET = 4_000_000;

export const OP_MARK_TEXT = {
  present: "unchanged here",
  created: "made here",
  renamed: "renamed (lowered) here",
  inlined: "inlined here",
  fused: "fused here",
  removed: "removed here",
};

let opMarkTimer = 0;

function refreshOpMarks() {
  clearTimeout(opMarkTimer);
  opMarkTimer = setTimeout(applyOpMarks, 0);
}

function applyOpMarks() {
  let entries = [];
  const node = renderer.selected;

  if (
    state.trace &&
    state.trace.events.length > 1 &&
    (node >= 0 || state.opPinned)
  ) {
    const small =
      state.trace.events.reduce((n, event) => n + event.ir.length, 0) <=
      OP_MARK_BUDGET;

    if (state.opModel || small) {
      state.opModel ??= computeOpModel();

      entries =
        node >= 0
          ? opHistory(state.opModel, state.traceIndex, node)
          : state.opPinned;
    }
  }

  state.opMarkInfo = new Map(
    entries.map((e) => [e.pass, e.change === "kept" ? "present" : e.change]),
  );

  scrubber.setOpMarks(
    [...state.opMarkInfo].map(([pass, kind]) => ({
      pass,
      kind: kind === "present" ? "life" : "op",
    })),
  );
}

export function computeOpModel() {
  const shapes = [];

  const records = state.trace.events.map((event) => {
    const result = parse(event.ir);
    const ok = result.status === STATUS.OK;
    shapes.push(ok ? snapshotShape(result.snapshot) : null);

    return ok ? opRecords(result.snapshot) : null;
  });

  state.opShapes = shapes;

  if (state.renderedText) {
    const restored = parse(state.renderedText).snapshot;
    renderer.snapshot = restored;
    state.diffAfter = restored;
    renderer.requestDraw();
  }

  return buildOpModel(records);
}

export const OP_CHANGE = {
  created: "created",
  kept: "unchanged",
  renamed: "renamed (lowered)",
  inlined: "inlined from",
  fused: "fused from",
  removed: "removed",
};

export function goToOp(pass, node) {
  state.opPinned = node < 0 ? state.lastOpEntries : null;
  if (pass !== state.traceIndex) selectEvent(pass);
  if (node >= 0) renderer.select(node, { center: true });
  else renderOpHistoryPanel();
}

const MADE_TEXT = { created: "Made", inlined: "Inlined" };

export function renderOpHistoryPanel() {
  const hint = (text) => {
    ophNone.textContent = text;
    ophNone.hidden = false;
    ophBody.hidden = true;
    ophSliderNote.hidden = true;
  };

  if (!state.trace || state.trace.events.length < 2) {
    return hint("Open a pass trace to follow an op through the passes.");
  }

  const node = renderer.selected;
  if (node >= 0) state.opPinned = null;

  if (node < 0 && !state.opPinned) {
    return hint(
      "Pick an op in the graph, or a line in the Source tab, to see where it came from and what each pass did to it.",
    );
  }

  state.opModel ??= computeOpModel();

  const entries =
    node >= 0
      ? opHistory(state.opModel, state.traceIndex, node)
      : state.opPinned;

  state.lastOpEntries = entries;

  if (!entries.length) {
    return hint("This pass's IR did not parse, so there is no history.");
  }

  ophNone.hidden = true;
  ophBody.hidden = false;
  ophSliderNote.hidden = false;

  const here = entries.find((e) => e.pass === state.traceIndex) ?? entries[0];
  ophTitle.textContent = here.label;

  const passName = (i) => `#${i + 1} ${describeEvent(state.trace.events[i])}`;
  const made = entries[0];
  const gone = entries.find((e) => e.change === "removed");
  const lowered = entries.filter((e) => e.change === "renamed");

  ophSummary.textContent =
    `${MADE_TEXT[made.change] ?? "Fused"} at ${passName(made.pass)}` +
    (lowered.length
      ? `; lowered at #${lowered.map((e) => e.pass + 1).join(", #")}`
      : "") +
    (gone
      ? `; removed at #${gone.pass + 1}`
      : "; still there at the last pass") +
    (here.loc ? ` · loc ${here.loc}` : "");

  ophSteps.replaceChildren(
    ...entries.map((entry) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = `oph-step ${entry.change}`;

      if (entry.pass === state.traceIndex) {
        button.setAttribute("aria-current", "step");
      }

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

const descent = createDescentView(descentEl, {
  onPickNode: goToOp,
  onPickPass: (pass) => pass !== state.traceIndex && selectEvent(pass),
  onStep(delta) {
    if (!state.trace) return;

    const last = state.trace.events.length - 1;

    let wanted = state.traceIndex + delta;
    if (!Number.isFinite(delta)) wanted = delta > 0 ? last : 0;

    const target = Math.max(0, Math.min(last, wanted));

    if (target !== state.traceIndex) selectEvent(target);
  },
  onGpu: () => setCanvasView("gpu", { remember: false }),
});

let descentFor = null;

export function updateDescentTab() {
  const has = !!state.trace && state.trace.events.length > 1;
  viewDescent.hidden = !has;

  if (!has) {
    descent.setData(null);
    descentFor = null;

    if (state.canvasView === "descent") {
      setCanvasView("graph", { remember: false });
    }
  }
}

export function drawDescent() {
  if (!state.trace) return;
  state.opModel ??= computeOpModel();

  if (descentFor !== state.opModel) {
    descent.setData({
      shapes: state.opShapes,
      titles: state.trace.events.map(
        (event) => `#${event.index + 1} ${describeEvent(event)}`,
      ),
      names: state.trace.events.map((event) =>
        describeEvent(event).replace(/ · .*/, ""),
      ),
      labelOf: (pass, node) => state.opModel.records[pass]?.[node]?.label ?? "",
    });

    descentFor = state.opModel;
  }

  refreshDescent();
}

export function refreshDescent() {
  if (
    state.canvasView !== "descent" ||
    !state.trace ||
    !state.opModel ||
    descentFor !== state.opModel
  ) {
    return;
  }

  const node = renderer.selected;
  const line = selection.state.line;

  descent.setState({
    pass: state.traceIndex,
    node,
    lineage:
      node >= 0
        ? opHistory(state.opModel, state.traceIndex, node)
        : (state.opPinned ?? []),
    lineNodes:
      line >= 0 ? nodesOnLine(state.opShapes[state.traceIndex], line + 1) : [],
    gpu: !!state.gpuModel,
  });
}

export function setupOpHistory() {
  selection.subscribe((_, picked) => {
    if (picked.includes("node")) refreshOpMarks();
  });

  selection.subscribe((selected, picked) => {
    if (
      picked.includes("node") &&
      inspector.isOpen &&
      inspector.active === "ophistory"
    ) {
      renderOpHistoryPanel();
    }
  });

  selection.subscribe((_, picked) => {
    if (picked.includes("node") || picked.includes("line")) refreshDescent();
  });
}
