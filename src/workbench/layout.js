import { STORAGE_KEYS } from "../constants.js";
import { bindSplitters, createInspector } from "../app/index.js";
import { renderBuffersPanel } from "./buffers-panel.js";
import { focusChange } from "./diff-panel.js";
import { canvas } from "./dom.js";
import { renderer } from "./graph.js";
import { renderLinePanel } from "./line-explain.js";
import { renderOpHistoryPanel } from "./op-history.js";
import { renderOpCountPanel } from "./opcount-panel.js";
import { state } from "./state.js";
import { renderSymbolsPanel } from "./symbol-history.js";
import { renderTiming } from "./timing-panel.js";

const inspectorEl = document.getElementById("inspector");
const inspectorToggle = document.getElementById("inspector-toggle");
const inspectorSplitter = document.getElementById("inspector-splitter");
const themeToggle = document.getElementById("theme-toggle");
const systemLight = matchMedia("(prefers-color-scheme: light)");

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;

  const next = theme === "dark" ? "light" : "dark";
  themeToggle.setAttribute("aria-label", `Switch to ${next} theme`);
  themeToggle.title = `Switch to ${next} theme (Shift+L)`;
  renderer.refreshTheme();
}

export function toggleTheme() {
  const theme =
    document.documentElement.dataset.theme === "dark" ? "light" : "dark";

  try {
    localStorage.setItem(STORAGE_KEYS.theme, theme);
  } catch {}

  applyTheme(theme);
}

try {
  state.inspectorSaved =
    JSON.parse(localStorage.getItem(STORAGE_KEYS.inspector)) || {};
} catch {}

const PANELS = {
  line: renderLinePanel,
  timing: renderTiming,
  buffers: renderBuffersPanel,
  opcount: renderOpCountPanel,
  symbols: renderSymbolsPanel,
  ophistory: renderOpHistoryPanel,
};

export const inspector = createInspector(inspectorEl, {
  toggle: inspectorToggle,
  closeButton: document.getElementById("inspector-close"),
  tab: state.inspectorSaved.tab,
  onShow: (tab) => PANELS[tab]?.(),
  onChange: ({ open, tab }) => {
    try {
      localStorage.setItem(
        STORAGE_KEYS.inspector,
        JSON.stringify({ open, tab }),
      );
    } catch {}
  },
});

export function refreshInspector() {
  if (inspector.isOpen) PANELS[inspector.active]?.();
}

export function showPanel(tab) {
  const shown = inspector.isOpen && inspector.active === tab;
  inspector.show(tab);
  if (shown) PANELS[tab]?.();
}

export function toggleInspector() {
  if (inspector.isOpen) inspector.close();
  else inspector.open();
}

export function stepChange(delta) {
  let next = state.diffCursor - 1;
  if (delta > 0) next = state.diffCursor + 1;
  else if (state.diffCursor < 0) next = -1;

  focusChange(next);

  if (state.diffRows.length) inspector.show("changes", { from: canvas });
}

export function inspect(tab, from) {
  if (tab === "changes" && inspector.isOpen && inspector.active === "line") {
    inspector.open({ from });

    return;
  }

  inspector.show(tab, { from });
}

export function setupLayout() {
  systemLight.addEventListener("change", (e) => {
    let saved = null;

    try {
      saved = localStorage.getItem(STORAGE_KEYS.theme);
    } catch {}

    if (!saved) applyTheme(e.matches ? "light" : "dark");
  });

  themeToggle.addEventListener("click", toggleTheme);

  applyTheme(document.documentElement.dataset.theme || "dark");

  bindSplitters(document.querySelector("#workspace main"), [
    {
      el: document.getElementById("source-splitter"),
      pane: document.getElementById("source-pane"),
      side: "left",
    },
    {
      el: inspectorSplitter,
      pane: inspectorEl,
      side: "right",
    },
  ]);

  inspectorEl.addEventListener("inspector-toggle", (e) => {
    inspectorSplitter.hidden = !e.detail.open;
  });
}
