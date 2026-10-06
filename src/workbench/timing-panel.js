import { download, slug } from "../session/index.js";
import { formatBytes, formatSeconds, timingToJSON } from "../trace/index.js";
import { sourceName } from "./dom.js";
import { showPanel } from "./layout.js";
import { state } from "./state.js";
import { selectEvent } from "./trace-nav.js";

const timingOpen = document.getElementById("timing-open");
const timingSummary = document.getElementById("timing-summary");
const timingTable = document.getElementById("timing-table");
const timingEmpty = document.getElementById("timing-empty");
const timingExport = document.getElementById("timing-export");

export function mayHaveReports(text) {
  return /Execution time report|resident set size|"duration":/.test(text);
}

export function setProfile(timing, memory, matches) {
  state.profile = { timing, memory, matches };
  timingOpen.hidden = !timing && !memory;

  timingOpen.textContent = "⏱";

  if (timing?.total) {
    timingOpen.textContent = `⏱ ${formatSeconds(timing.total)}`;
  } else if (memory) {
    timingOpen.textContent = `⏱ ${formatBytes(memory.peakBytes)}`;
  }
}

export function describeMatch(match) {
  return (
    `${formatSeconds(match.row.wall.seconds)} wall` +
    (match.runs > 1 ? ` across ${match.runs} runs` : "")
  );
}

export function openTiming() {
  showPanel("timing");
}

export function renderTiming() {
  const { timing, memory, matches } = state.profile;
  timingEmpty.hidden = !!(timing || memory);

  const facts = [];

  if (timing?.total !== null && timing?.total !== undefined) {
    facts.push(["Total", formatSeconds(timing.total)]);
  }

  if (timing) {
    facts.push([
      "Passes timed",
      String(timing.rows.filter((row) => row.kind === "pass").length),
    ]);
  }

  if (memory) {
    facts.push([
      "Peak memory",
      `${formatBytes(memory.peakBytes)} (whole process, ${memory.source})`,
    ]);
  }

  timingSummary.replaceChildren(
    ...facts.flatMap(([term, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;

      const dd = document.createElement("dd");
      dd.textContent = value;

      return [dt, dd];
    }),
  );

  timingTable.parentElement.hidden = !timing;
  timingExport.disabled = !timing && !memory;
  if (!timing) return;

  const withUser = timing.columns.includes("user");
  const head = document.createElement("tr");

  for (const label of ["Name", ...(withUser ? ["User"] : []), "Wall", "%"]) {
    const th = document.createElement("th");
    th.textContent = label;
    head.append(th);
  }

  const firstEvent = new Map();

  matches.forEach((match, i) => {
    if (match && !firstEvent.has(match.row.index)) {
      firstEvent.set(match.row.index, i);
    }
  });

  const body = timing.rows.map((row) => {
    const tr = document.createElement("tr");
    tr.className = row.kind;

    const name = document.createElement("td");
    name.style.paddingLeft = `${8 + row.depth * 14}px`;

    const eventIndex = firstEvent.get(row.index);

    if (eventIndex !== undefined) {
      const link = document.createElement("button");
      link.className = "link-btn";
      link.textContent = row.name;
      link.title = `Go to pass #${eventIndex + 1}`;
      link.addEventListener("click", () => selectEvent(eventIndex));
      name.append(link);
    } else {
      name.textContent = row.name;
    }

    const cells = [name];
    if (withUser) cells.push(timeCell(row.user?.seconds));
    cells.push(timeCell(row.wall.seconds));

    const share = document.createElement("td");
    share.className = "share";
    share.style.setProperty("--share", `${row.wall.percent}%`);
    share.textContent = `${row.wall.percent.toFixed(1)}%`;
    cells.push(share);
    tr.append(...cells);

    return tr;
  });

  const thead = document.createElement("thead");
  thead.append(head);

  const tbody = document.createElement("tbody");
  tbody.append(...body);
  timingTable.replaceChildren(thead, tbody);
}

function timeCell(seconds) {
  const td = document.createElement("td");
  td.className = "num";
  td.textContent = formatSeconds(seconds);

  return td;
}

function exportTiming() {
  const title = sourceName.textContent.replace(/ ●$/, "");

  download(
    `${slug(title)}-timing.json`,
    timingToJSON(
      title,
      state.profile.timing,
      state.profile.memory,
      state.trace?.events ?? [],
      state.profile.matches,
    ),
    "application/json",
  );
}

export function setupTimingPanel() {
  timingOpen.addEventListener("click", openTiming);

  timingExport.addEventListener("click", exportTiming);
}
