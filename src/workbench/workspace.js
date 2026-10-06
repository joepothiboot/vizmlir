import { AUTOSAVE_DELAY_MS } from "../constants.js";
import { parseNcuCounters } from "../bench.js";
import {
  download,
  kv,
  sessionToFile,
  sessions,
  slug,
} from "../session/index.js";
import { isPassTrace } from "../trace/index.js";
import {
  baseline,
  currentTitle,
  fileInput,
  input,
  sourceName,
  sourcePane,
  sourceTitle,
  splitToggle,
} from "./dom.js";
import { run } from "./pipeline.js";
import { setSourceSplit, showTab } from "./source-tabs.js";
import { state } from "./state.js";
import { setBenchmarks, updateSymbolsOpen } from "./symbol-history.js";
import { clearTrace, loadTrace } from "./trace-nav.js";

function getState() {
  return {
    sourceName: sourceName.textContent,
    trace: state.traceText || null,
    traceIndex: state.traceIndex,
    baseline: state.traceText ? "" : baseline.value,
    current: state.traceText ? "" : input.value,
    tab:
      currentTitle.getAttribute("aria-selected") === "true"
        ? "current"
        : "baseline",
    split: sourcePane.classList.contains("split"),
    sourceSplit: sourcePane.classList.contains("split-source"),
    sources: Object.keys(state.sources).length ? state.sources : null,
    breakpoints: state.breakpoints.length ? state.breakpoints : null,
    benchmarks: state.traceText
      ? Object.fromEntries(
          Object.entries(state.benchmarks).map(([slot, bench]) => [
            slot,
            bench && { name: bench.name, text: bench.text, mock: bench.mock },
          ]),
        )
      : null,
    measured:
      state.traceText && state.measured
        ? { device: state.measured.device, text: state.measured.text }
        : null,
  };
}

export function applyState(workspace) {
  clearTrace();
  state.sources = workspace.sources ? { ...workspace.sources } : {};

  state.breakpoints = (
    Array.isArray(workspace.breakpoints) ? workspace.breakpoints : []
  )
    .filter((b) => typeof b?.text === "string")
    .map((b) => ({ text: b.text, on: b.on !== false }));

  sourceName.textContent = workspace.sourceName || "untitled";
  sourceName.title = sourceName.textContent;
  sourcePane.classList.toggle("split", !!workspace.split);
  splitToggle.setAttribute("aria-pressed", String(!!workspace.split));

  if (workspace.trace && isPassTrace(workspace.trace)) {
    state.traceIndex = workspace.traceIndex ?? -1;

    if (workspace.measured?.text) {
      try {
        state.measured = {
          ...workspace.measured,
          counters: parseNcuCounters(workspace.measured.text),
        };
      } catch {}
    }

    loadTrace(workspace.trace, { keepIndex: true });

    const saved = workspace.benchmarks?.text
      ? { current: workspace.benchmarks }
      : workspace.benchmarks;

    for (const slot of ["baseline", "current"]) {
      if (!state.trace || !saved?.[slot]) continue;

      try {
        setBenchmarks(
          slot,
          saved[slot].name,
          saved[slot].text,
          saved[slot].mock,
        );
      } catch {}
    }

    updateSymbolsOpen();
  } else {
    baseline.value = workspace.baseline ?? "";
    input.value = workspace.current ?? "";
    baseline.dispatchEvent(new Event("input"));
    input.dispatchEvent(new Event("input"));
    clearTimeout(state.timer);
    run();
  }

  showTab(workspace.tab === "baseline" ? baseline : input);
  if (workspace.sourceSplit && !sourceTitle.hidden) setSourceSplit(true);
}

let autosaveTimer = 0;

export function scheduleAutosave() {
  if (!state.restored) return;
  clearTimeout(autosaveTimer);

  autosaveTimer = setTimeout(
    () => kv.set("autosave", getState()),
    AUTOSAVE_DELAY_MS,
  );
}

export const sessionsDialog = document.getElementById("sessions");
const sessionList = document.getElementById("session-list");
const sessionName = document.getElementById("session-name");

async function renderSessions() {
  const list = await sessions.list();

  if (!list.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No saved sessions yet.";
    sessionList.replaceChildren(empty);

    return;
  }

  sessionList.replaceChildren(
    ...list.map((session) => {
      const item = document.createElement("li");
      const meta = document.createElement("span");
      meta.className = "meta";

      const name = document.createElement("span");
      name.className = "name";
      name.textContent = session.name;

      const when = document.createElement("span");
      when.className = "when";

      when.textContent =
        new Date(session.savedAt).toLocaleString() +
        (session.state.trace ? " · trace" : "");

      meta.append(name, when);

      const actions = [
        [
          "Open",
          () => {
            sessionsDialog.close();
            applyState(session.state);
          },
        ],
        [
          "Download",
          () =>
            download(
              `${slug(session.name)}.vizmlir.json`,
              sessionToFile(session.name, session.state),
              "application/json",
            ),
        ],
        [
          "Delete",
          async () => {
            await sessions.delete(session.id);
            renderSessions();
          },
        ],
      ].map(([label, action]) => {
        const button = document.createElement("button");
        button.textContent = label;
        button.setAttribute("aria-label", `${label} ${session.name}`);
        button.addEventListener("click", action);

        return button;
      });

      item.append(meta, ...actions);

      return item;
    }),
  );
}

export function openSessions() {
  sessionName.value = sourceName.textContent.replace(/ ●$/, "");
  renderSessions();
  sessionsDialog.showModal();
  sessionName.select();
}

export function downloadSession() {
  const name = sourceName.textContent.replace(/ ●$/, "");

  download(
    `${slug(name)}.vizmlir.json`,
    sessionToFile(name, getState()),
    "application/json",
  );
}

export function setupWorkspace() {
  document
    .getElementById("session-save")
    .addEventListener("submit", async (e) => {
      e.preventDefault();

      const name = sessionName.value.trim();
      if (!name) return;

      await sessions.put({
        id: crypto.randomUUID(),
        name,
        savedAt: Date.now(),
        state: getState(),
      });

      renderSessions();
    });

  document
    .getElementById("sessions-open")
    .addEventListener("click", openSessions);

  document
    .getElementById("session-import")
    .addEventListener("click", () => fileInput.click());

  document
    .getElementById("session-download")
    .addEventListener("click", downloadSession);
}
