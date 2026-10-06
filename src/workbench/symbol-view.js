import { highlightMlir } from "../app/index.js";
import { formatChange, formatDuration } from "../bench.js";
import {
  describeEvent,
  diffStats,
  lineDiff,
  symbolTimeline,
} from "../trace/index.js";
import { setStatus } from "./dom.js";
import { state } from "./state.js";
import { benchmarkView, openSymbols, showSymbol } from "./symbol-history.js";

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

let symbolViewState = null;

export function openSymbolView(record) {
  if (!state.trace) return;

  const steps = symbolTimeline(state.trace.events, record.path);

  if (!steps.length) {
    setStatus(`${record.path} has no IR in this trace`, { error: true });

    return;
  }

  const here = steps.findIndex((step) => step.index === state.traceIndex);

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

function stepLabel(step) {
  if (step.kind === "lowered") return `lowered ${step.from} → ${step.op}`;
  if (step.kind === "initial") return `first dump · ${step.op}`;

  return `${step.kind} · ${step.op}`;
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
        (state.benchmarks.current?.mock || state.benchmarks.baseline?.mock
          ? " · mock data"
          : ""),
    );
  }

  symbolViewMeta.textContent = facts.filter(Boolean).join(" · ");

  symbolViewSteps.replaceChildren(
    ...steps.map((step, i) => {
      const event = state.trace.events[step.index];
      const item = document.createElement("li");
      const button = document.createElement("button");
      const name = document.createElement("span");
      name.textContent = `#${step.index + 1} ${event.argument || event.pass}`;

      const kind = document.createElement("span");
      kind.className = `kind ${step.kind}`;

      kind.textContent = stepLabel(step);

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

function stepText(i) {
  for (let k = i; k >= 0; k--) {
    if (symbolViewState.steps[k].text !== null) {
      return symbolViewState.steps[k].text;
    }
  }

  return null;
}

const LINE_SIGN = { add: "+", del: "−" };

function renderSymbolView() {
  const { steps, current } = symbolViewState;
  const step = steps[current];
  const event = state.trace.events[step.index];

  [...symbolViewSteps.children].forEach((item, i) => {
    const button = item.firstChild;

    if (i === current) {
      button.setAttribute("aria-current", "step");
      button.scrollIntoView({ block: "nearest" });
    } else {
      button.removeAttribute("aria-current");
    }
  });

  const assembly = step.assembly ?? [];
  const isNvvm = assembly.some((object) => object.target.startsWith("#nvvm"));
  symbolViewTabs.asm.hidden = !assembly.length;
  symbolViewTabs.asm.textContent = isNvvm ? "PTX" : "Assembly";

  let mode = symbolViewState.mode;
  if (mode === "asm" && !assembly.length) mode = "diff";
  if (mode === "diff" && step.text === null && assembly.length) mode = "asm";

  for (const [name, tab] of Object.entries(symbolViewTabs)) {
    tab.setAttribute("aria-selected", String(name === mode));
  }

  const before = current > 0 ? stepText(current - 1) : null;
  const lines = [];
  const note = (text) => lines.push({ type: "note", text });

  if (mode === "asm") {
    for (const object of assembly) {
      note(`${object.target}`);

      for (const text of object.text.replace(/\n$/, "").split("\n")) {
        lines.push({ type: "plain", text });
      }
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
    if (mode === "diff") {
      note("First appearance, so there is nothing to compare.");
    }

    for (const text of step.text.split("\n")) {
      lines.push({ type: "same", text });
    }
  }

  symbolViewStep.textContent = `#${step.index + 1} ${describeEvent(event)}`;

  symbolViewCode.replaceChildren(
    ...lines.map((line) => {
      const div = document.createElement("div");
      div.className = `line ${line.type}`;

      if (line.type === "note" || line.type === "plain") {
        div.textContent = line.text || " ";
      } else {
        const sign = LINE_SIGN[line.type] ?? " ";

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

function dumpIndexFor(step, record) {
  if (step.text !== null) return step.index;

  if (record.lastDump >= 0 && record.lastDump < step.index) {
    return record.lastDump;
  }

  return Math.max(0, step.index - 1);
}

export function setupSymbolView() {
  for (const [mode, tab] of Object.entries(symbolViewTabs)) {
    tab.addEventListener("click", () => setSymbolViewMode(mode));
  }

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
    const step = steps[current];

    const index = dumpIndexFor(step, record);

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
}
