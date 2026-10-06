import { STORAGE_KEYS } from "../constants.js";
import { CommandPalette } from "../app/index.js";
import { SAMPLES } from "../samples.js";
import { canWatchFiles } from "../session/index.js";
import { describeEvent } from "../trace/index.js";
import { openBuffers } from "./buffers-panel.js";
import { setCanvasView } from "./canvas-views.js";
import {
  runToBreakpoint,
  setDebug,
  stepOpChange,
  stepToChange,
} from "./debugger.js";
import { clearPicks, copyChanges, togglePick } from "./diff-panel.js";
import { canvas, fileInput } from "./dom.js";
import {
  exportDiff,
  exportPNG,
  exportSVG,
  pickWatch,
  stopWatching,
  watcher,
} from "./files.js";
import { renderer } from "./graph.js";
import {
  inspect,
  inspector,
  showPanel,
  stepChange,
  toggleInspector,
  toggleTheme,
} from "./layout.js";
import { openOpCounts } from "./opcount-panel.js";
import { fitGraph } from "./pipeline.js";
import { openSample, openSamples } from "./scenarios.js";
import { toggleSplit, toggleTab } from "./source-tabs.js";
import { state } from "./state.js";
import { importBenchmarks, openSymbols } from "./symbol-history.js";
import { openTiming } from "./timing-panel.js";
import { selectEvent } from "./trace-nav.js";
import { downloadSession, openSessions } from "./workspace.js";

const helpDialog = document.getElementById("help");

const palette = new CommandPalette(document.getElementById("palette"), () => {
  const items = [];

  if (state.trace) {
    for (const event of state.trace.events) {
      items.push({
        group: "pass",
        text: `${event.index + 1}. ${describeEvent(event)}`,
        hint: event.failed ? "✗ failed" : "",
        run: () => {
          selectEvent(event.index);
        },
      });
    }
  }

  const snap = renderer.snapshot;

  for (let i = 0; snap && i < snap.nodeCount; i++) {
    const label = snap.labelOf(i);
    const parent = snap.parentOf(i);

    items.push({
      group: label.includes("@") ? "symbol" : "op",
      text: label,
      hint: `#${i}${parent < 0 ? "" : ` · in ${snap.labelOf(parent)}`}`,
      run: () => {
        renderer.select(i, { center: true });
        inspect("changes", canvas);
      },
    });
  }

  const actions = [
    ["Open file…", "", () => fileInput.click()],
    ["Fit graph", "f", fitGraph],
    ["Switch baseline / current", "t", toggleTab],
    ["Toggle split sources", "s", toggleSplit],
    ["Toggle debug mode", "d", () => setDebug(!state.debugOn)],
    ["Continue to the next breakpoint", ".", () => runToBreakpoint(1)],
    ["Run back to the previous breakpoint", ",", () => runToBreakpoint(-1)],
    ["Op history: follow the selected op", "y", () => showPanel("ophistory")],
    ["Choose a scenario…", "", () => openSamples()],
    ["Sessions: save, open, import", "", openSessions],
    ["Download session .json", "", downloadSession],
    ["Export graph as PNG", "", exportPNG],
    ["Export graph as SVG", "", exportSVG],
    ["Copy changes as patch", "c", copyChanges],
    ["Export diff as Markdown", "", () => exportDiff("md")],
    ["Export diff as JSON", "", () => exportDiff("json")],
    ["Inspector: changes in this pass", "", () => showPanel("changes")],
    ["Inspector: pass timing and memory", "p", openTiming],
    ["Inspector: op counts per pass", "o", openOpCounts],
    ["Inspector: buffers and peak memory", "b", openBuffers],
    [
      "Inspector: symbol history (which pass made each kernel)",
      "h",
      openSymbols,
    ],
    [
      inspector.isOpen ? "Close the inspector" : "Open the inspector",
      "\\",
      toggleInspector,
    ],
    ...(state.gpuModel
      ? [
          [
            state.canvasView === "gpu"
              ? "Show the op graph"
              : "Show the GPU view: launches and memory",
            "g",
            () => setCanvasView(state.canvasView === "gpu" ? "graph" : "gpu"),
          ],
        ]
      : []),
    ...(state.localModel
      ? [
          [
            state.canvasView === "local"
              ? "Show the op graph"
              : "Show the Local memory view: scratchpad buffers and DMAs",
            "",
            () => {
              state.localWanted = state.canvasView !== "local";
              setCanvasView(state.localWanted ? "local" : "graph");
            },
          ],
        ]
      : []),
    [
      "Inspector: what's this line? (explain the clicked line)",
      "w",
      () => showPanel("line"),
    ],
    [
      "Import current kernel benchmarks…",
      "",
      () => importBenchmarks("current"),
    ],
    [
      "Import baseline kernel benchmarks…",
      "",
      () => importBenchmarks("baseline"),
    ],
    ...(canWatchFiles
      ? [
          watcher.watching
            ? ["Stop watching file", "", stopWatching]
            : ["Watch a file for changes…", "", pickWatch],
        ]
      : []),
    ["Show / hide menu", "m", () => toggleMenu()],
    ["Switch light / dark theme", "shift L", toggleTheme],
    ["Keyboard shortcuts", "?", () => helpDialog.showModal()],
  ];

  for (const [text, hint, action] of actions) {
    items.push({ group: "action", text, hint, run: action });
  }

  for (const sample of SAMPLES) {
    items.push({
      group: "sample",
      text: sample.title,
      hint: sample.trace ? "trace" : "diff",
      run: () => openSample(sample),
    });
  }

  return items;
});

const isMac = /Mac|iPhone|iPad/.test(navigator.userAgent);
const topbar = document.getElementById("topbar");
const menuHandle = document.getElementById("menu-handle");

function setMenuHidden(hidden) {
  topbar.hidden = hidden;

  const verb = hidden ? "Show" : "Hide";
  menuHandle.setAttribute("aria-expanded", String(!hidden));
  menuHandle.setAttribute("aria-label", `${verb} menu`);
  menuHandle.title = `${verb} menu (m)`;
}

function toggleMenu() {
  setMenuHidden(!topbar.hidden);

  try {
    localStorage.setItem(STORAGE_KEYS.menu, topbar.hidden ? "hidden" : "shown");
  } catch {}
}

function isTyping(target) {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable ||
      ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
  );
}

const WORKSPACE_KEYS = {
  f: fitGraph,
  "+": () => renderer.zoomBy(1.25),
  "=": () => renderer.zoomBy(1.25),
  "-": () => renderer.zoomBy(0.8),
  x: () => state.diffCursor >= 0 && togglePick(state.diffCursor),
  c: copyChanges,
  j: () => stepChange(1),
  k: () => stepChange(-1),
  t: toggleTab,
  s: toggleSplit,
  d: () => setDebug(!state.debugOn),
  "{": () => stepToChange(-1),
  "}": () => stepToChange(1),
  ",": () => state.debugOn && runToBreakpoint(-1),
  ".": () => state.debugOn && runToBreakpoint(1),
  "<": () => stepOpChange(-1),
  ">": () => stepOpChange(1),
  p: () => inspector.toggleTab("timing"),
  o: () => inspector.toggleTab("opcount"),
  b: () => inspector.toggleTab("buffers"),
  h: () => inspector.toggleTab("symbols"),
  w: () => inspector.toggleTab("line"),
  y: () => inspector.toggleTab("ophistory"),
  "\\": toggleInspector,
  g: () =>
    state.gpuModel &&
    setCanvasView(state.canvasView === "gpu" ? "graph" : "gpu"),
  "[": () =>
    state.trace && state.traceIndex > 0 && selectEvent(state.traceIndex - 1),
  "]": () =>
    state.trace &&
    state.traceIndex < state.trace.events.length - 1 &&
    selectEvent(state.traceIndex + 1),
};

export function setupCommands() {
  document
    .querySelectorAll(".mod-k")
    .forEach((kbd) => (kbd.textContent = isMac ? "⌘K" : "Ctrl K"));

  document
    .getElementById("palette-open")
    .addEventListener("click", () => palette.open());

  document
    .getElementById("help-open")
    .addEventListener("click", () => helpDialog.showModal());

  menuHandle.addEventListener("click", toggleMenu);

  try {
    setMenuHidden(localStorage.getItem(STORAGE_KEYS.menu) === "hidden");
  } catch {}

  window.addEventListener("keydown", (e) => {
    if (document.querySelector("dialog[open]")) return;

    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      palette.open();

      return;
    }

    if (e.metaKey || e.ctrlKey || e.altKey) return;

    if (e.key === "Escape" && isTyping(e.target)) {
      e.target.blur();

      return;
    }

    if (isTyping(e.target)) return;
    if (e.key === "Escape" && clearPicks()) return;
    if (e.key === "Escape" && inspector.handleEscape()) return;

    if (e.key === "/") {
      e.preventDefault();
      palette.open();
    } else if (e.key === "?") {
      helpDialog.showModal();
    } else if (e.key === "L") {
      toggleTheme();
    } else if (e.key === "m") {
      toggleMenu();
    } else if (WORKSPACE_KEYS[e.key]) {
      e.preventDefault();
      WORKSPACE_KEYS[e.key]();
    }
  });
}
