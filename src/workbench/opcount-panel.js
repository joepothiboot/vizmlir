import { direction } from "../format.js";
import { STATUS } from "../ir/index.js";
import { download, slug } from "../session/index.js";
import {
  countOps,
  describeEvent,
  moduleStateAt,
  opCountTable,
  opCountsToCSV,
  totalOps,
} from "../trace/index.js";
import { sourceName } from "./dom.js";
import { parse } from "./engine.js";
import { renderer } from "./graph.js";
import { showPanel } from "./layout.js";
import { state } from "./state.js";
import { selectEvent } from "./trace-nav.js";

const opCountOpen = document.getElementById("opcount-open");
let viewCounts = null;
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
  if (n > 0) return `+${n}`;
  if (n < 0) return `−${-n}`;

  return "±0";
}

export function setViewCounts(counts) {
  viewCounts = counts;
  opCountOpen.hidden = !counts;
  if (!counts) return;

  const total = totalOps(counts.after);
  const delta = counts.before ? total - totalOps(counts.before) : 0;

  opCountOpen.textContent =
    `${total} ops` + (counts.before && delta ? ` (${signed(delta)})` : "");
}

export function computeTraceCounts() {
  const columns = state.trace.events.map((_, i) => {
    const result = parse(moduleStateAt(state.trace.events, i));

    return result.status === STATUS.OK ? countOps(result.snapshot) : null;
  });

  if (state.renderedText) {
    const restored = parse(state.renderedText).snapshot;
    renderer.snapshot = restored;
    state.diffAfter = restored;
    renderer.requestDraw();
  }

  return columns;
}

function buildOpCountModel() {
  if (state.trace) {
    state.traceCounts ??= computeTraceCounts();

    const failed = state.traceCounts.filter((column) => !column).length;

    return {
      table: opCountTable(
        state.traceCounts.map((column) => column ?? new Map()),
      ),
      headers: state.trace.events.map((event) => `#${event.index + 1}`),
      titles: state.trace.events.map(
        (event) => `${event.index + 1}. ${describeEvent(event)}`,
      ),
      events: state.trace.events.map((event) => event.index),
      current: state.traceIndex,
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

export function openOpCounts() {
  showPanel("opcount");
}

export function renderOpCountPanel() {
  const none = !viewCounts && !state.trace;
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

  if (state.trace) {
    facts.push([
      "Passes that change counts",
      `${table.changedColumns.length} of ${headers.length}`,
    ]);
  }

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
  opCountChangedPasses.parentElement.hidden = !state.trace;

  const keep = new Set([0, current, ...table.changedColumns]);

  const shownColumns = headers
    .map((_, i) => i)
    .filter(
      (i) => !state.trace || !opCountChangedPasses.checked || keep.has(i),
    );

  const needle = opCountFilter.value.trim().toLowerCase();

  const rows = table.rows.filter(
    (row) =>
      (!opCountChangedOps.checked || headers.length < 2 || row.changed) &&
      (!needle || row.op.toLowerCase().includes(needle)),
  );

  opCountShown.textContent =
    `${rows.length} of ${table.rows.length} ops` +
    (state.trace ? ` · ${shownColumns.length} of ${headers.length} dumps` : "");

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
      const previous = i > 0 ? counts[i - 1] : counts[i];
      const change = counts[i] - previous;

      let trend = "";
      if (k > 0 && change) trend = direction(change);
      else if (k > 0 && many) trend = "same";

      const td = cell(
        "td",
        String(counts[i]),
        [trend, i === current ? "current" : ""].filter(Boolean).join(" "),
      );

      if (k > 0 && change) td.title = `${signed(change)} in this pass`;
      tr.append(td);
    });

    if (headers.length > 1) {
      tr.append(cell("td", signed(delta), `delta ${direction(delta)}`));
    }

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

export function cell(tag, text, className) {
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

export function setupOpCountPanel() {
  opCountOpen.addEventListener("click", openOpCounts);

  opCountFilter.addEventListener("input", renderOpCountTable);

  opCountChangedOps.addEventListener("change", renderOpCountTable);

  opCountChangedPasses.addEventListener("change", renderOpCountTable);

  document
    .getElementById("opcount-export")
    .addEventListener("click", exportOpCounts);
}
