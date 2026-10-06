import { direction, signOf } from "../format.js";
import {
  changedBeyond,
  compareBenchmarks,
  comparisonOrder,
  comparisonToJSON,
  formatChange,
  formatDuration,
  parseBenchmarks,
  symbolTimes,
} from "../bench.js";
import { download, slug } from "../session/index.js";
import {
  describeEvent,
  findSymbolNode,
  historyToJSON,
  symbolHistory,
} from "../trace/index.js";
import { setStatus, sourceName } from "./dom.js";
import { renderer } from "./graph.js";
import { inspector, showPanel } from "./layout.js";
import { cell } from "./opcount-panel.js";
import { state } from "./state.js";
import { openSymbolView } from "./symbol-view.js";
import { selectEvent } from "./trace-nav.js";
import { scheduleAutosave } from "./workspace.js";

const symbolsOpen = document.getElementById("symbols-open");
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

let benchSlot = "current";

const STEP_TEXT = {
  created: "created",
  changed: "changed",
  lowered: "lowered",
  removed: "removed",
};

export function openSymbols() {
  symbolsBenchError.hidden = true;
  showPanel("symbols");
}

export function renderSymbolsPanel() {
  symbolsNone.hidden = !!state.trace;
  symbolsBody.hidden = !state.trace;
  if (!state.trace) return;
  renderSymbolsSummary();
  renderSymbols();
}

export function setBenchmarks(slot, name, text, mock = false) {
  state.benchmarks[slot] = { name, text, mock, result: parseBenchmarks(text) };
}

export function benchmarkView() {
  if (!state.benchmarks.baseline && !state.benchmarks.current) return null;
  state.traceSymbols ??= symbolHistory(state.trace.events);

  const runs = {};

  for (const slot of ["baseline", "current"]) {
    runs[slot] =
      state.benchmarks[slot] &&
      symbolTimes(state.benchmarks[slot].result.entries, state.traceSymbols);
  }

  return {
    runs,
    both: !!(runs.baseline && runs.current),
    comparison: compareBenchmarks(runs.baseline?.times, runs.current?.times),
  };
}

export function importBenchmarks(slot) {
  if (!state.trace) {
    setStatus("open a pass trace before importing benchmarks", { error: true });

    return;
  }

  benchSlot = slot;
  symbolsBenchFile.click();
}

function renderSymbolsSummary() {
  state.traceSymbols ??= symbolHistory(state.trace.events);

  const count = (predicate) => state.traceSymbols.filter(predicate).length;

  const facts = [
    ["Symbols", String(state.traceSymbols.length)],
    ["Created during the trace", String(count((r) => !r.initial))],
    ["Lowered to another op", String(count((r) => r.ops.length > 1))],
    ["Removed", String(count((r) => r.removed))],
  ];

  const view = benchmarkView();

  for (const slot of ["baseline", "current"]) {
    const bench = state.benchmarks[slot];
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
  symbolsMock.hidden = !slots.some((slot) => state.benchmarks[slot].mock);
  symbolsMinChangeOn.parentElement.hidden = !view?.both;

  const unmatched = slots.flatMap((slot) =>
    view.runs[slot].unmatched.map((match) => ({ ...match, slot })),
  );

  symbolsBench.hidden = !unmatched.length;
  if (view) renderUnmatched(unmatched, view.both);

  const needle = symbolsFilter.value.trim().toLowerCase();

  const rows = state.traceSymbols.filter(
    (record) =>
      (!symbolsChanged.checked || record.changes.length) &&
      (!symbolsAtPass.checked ||
        record.changes.some((change) => change.index === state.traceIndex)) &&
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

  if (comparison) {
    rows.sort((a, b) =>
      comparisonOrder(comparison.get(a.path), comparison.get(b.path)),
    );
  }

  symbolsShown.textContent = `${rows.length} of ${state.traceSymbols.length} symbols`;

  const head = document.createElement("tr");
  head.append(cell("th", "Symbol"), cell("th", "Defined by"));

  for (const slot of slots) {
    let title = view.both ? "Current" : "Time";
    if (slot === "baseline") title = "Baseline";

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

    if (!record.changes.length) {
      steps.append(cell("span", "unchanged through the trace", "none"));
    }

    for (const change of record.changes) {
      const event = state.trace.events[change.index];
      const step = document.createElement("button");

      step.className = [
        "link-btn",
        "step",
        change.kind,
        change.index === state.traceIndex ? "current" : "",
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

      if (time) {
        td.title =
          `${time.calls} call(s) · total ${formatDuration(time.totalNs)}\n` +
          time.kernels
            .map((kernel, i) => `${kernel} (${time.how[i]} match)`)
            .join("\n");
      }

      tr.append(td);
    }

    if (view?.both) {
      const delta = row?.change ?? null;

      const td = cell(
        "td",
        delta === null ? "" : formatChange(delta),
        `time ${direction(delta, "")}`.trim(),
      );

      if (delta !== null) {
        td.title =
          `${signOf(row.deltaNs)}` +
          `${formatDuration(Math.abs(row.deltaNs))} per call`;
      } else if (row) {
        td.title = `Measured in the ${row.baseline ? "baseline" : "current"} run only`;
      }

      tr.append(td);
    }

    tr.append(stepsCell);
    tbody.append(tr);
  }

  symbolsTable.replaceChildren(thead, tbody);
  updateSymbolsOpen();
}

export function showSymbol(record, index = record.lastDump) {
  if (index < 0) return;
  selectEvent(index);

  const node = findSymbolNode(renderer.snapshot, record.path);
  if (node >= 0) renderer.select(node, { center: true });
  else setStatus(`${record.path} is not drawn in this pass`, { error: true });
}

export function updateSymbolsOpen() {
  symbolsOpen.hidden = !state.trace;
  if (!state.trace) return;

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

function unmatchedReason(entry, candidates) {
  if (entry.symbol) return ` · symbol ${entry.symbol} is not in the trace`;

  if (candidates.length) {
    return ` · ambiguous: ${candidates.join(", ")}; add a symbol column`;
  }

  return " · no symbol with this name";
}

function renderUnmatched(unmatched, labelSlots) {
  symbolsUnmatched.replaceChildren(
    ...unmatched.map(({ entry, candidates, slot }) => {
      const item = document.createElement("li");
      const name = document.createElement("code");
      name.textContent = entry.kernel;

      const reason = unmatchedReason(entry, candidates);

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
  if (!state.traceSymbols) return;

  download(
    `${slug(sourceName.textContent.replace(/ ●$/, ""))}-symbols.json`,
    historyToJSON(
      sourceName.textContent,
      state.trace.events,
      state.traceSymbols,
      describeEvent,
      (() => {
        const view = benchmarkView();

        return view && comparisonToJSON(view.comparison);
      })(),
    ),
    "application/json",
  );
}

export function setupSymbolHistory() {
  symbolsBenchFile.addEventListener("change", async () => {
    const [file] = symbolsBenchFile.files;
    symbolsBenchFile.value = "";
    if (!file || !state.trace) return;

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
    state.benchmarks = { baseline: null, current: null };
    scheduleAutosave();
    renderSymbolsSummary();
    renderSymbols();
  });

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
}
