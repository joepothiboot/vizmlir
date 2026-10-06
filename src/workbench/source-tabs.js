import { focusLocalLine } from "../render/index.js";
import { focusGpu } from "./canvas-views.js";
import {
  baseline,
  baselineTitle,
  currentTitle,
  input,
  localViewEl,
  sourcePane,
  sourcePanel,
  sourceTitle,
  splitToggle,
} from "./dom.js";
import { renderer } from "./graph.js";
import { renderExplain } from "./line-explain.js";
import { selection, state } from "./state.js";

const TABS = [
  { tab: baselineTitle, textarea: baseline },
  { tab: currentTitle, textarea: input },
];

export function showTab(textarea) {
  if (sourcePane.classList.contains("split-source")) {
    if (textarea === input) return;
    setSourceSplit(false);
  }

  for (const entry of TABS) {
    const on = entry.textarea === textarea;
    entry.tab.setAttribute("aria-selected", String(on));
    entry.textarea.parentElement.hidden = !on;
  }

  sourceTitle.setAttribute("aria-selected", "false");
  sourcePanel.hidden = true;
}

function showSourceTab() {
  if (sourceTitle.hidden) return;
  if (sourcePane.classList.contains("split")) toggleSplit();

  for (const entry of TABS) {
    entry.tab.setAttribute("aria-selected", "false");
    entry.textarea.parentElement.hidden = true;
  }

  sourceTitle.setAttribute("aria-selected", "true");
  sourcePanel.hidden = false;
}

export function setSourceSplit(on) {
  sourcePane.classList.toggle("split-source", on);

  if (on) {
    if (sourcePane.classList.contains("split")) toggleSplit();
    baselineTitle.setAttribute("aria-selected", "false");
    currentTitle.setAttribute("aria-selected", "true");
    sourceTitle.setAttribute("aria-selected", "false");
    baseline.parentElement.hidden = true;
    input.parentElement.hidden = false;
    sourcePanel.hidden = false;
  } else {
    showTab(input);
  }
}

export function toggleTab() {
  showTab(
    currentTitle.getAttribute("aria-selected") === "true" ? baseline : input,
  );
}

export function toggleSplit() {
  if (sourcePane.classList.contains("split-source")) setSourceSplit(false);

  const on = sourcePane.classList.toggle("split");
  splitToggle.setAttribute("aria-pressed", String(on));
}

const lineMark = input.parentElement.querySelector(".line-mark");

function sourceLineBox(line) {
  const style = getComputedStyle(input);
  const lineHeight = parseFloat(style.lineHeight);

  const fallback = {
    top: parseFloat(style.paddingTop) + line * lineHeight,
    height: lineHeight,
  };

  const code = input.parentElement.querySelector("code");
  const lines = input.value.split("\n");
  if (!code || line >= lines.length || !lines[line]) return fallback;

  let start = 0;
  for (let i = 0; i < line; i++) start += lines[i].length + 1;

  const end = start + lines[line].length;
  const range = document.createRange();
  const walker = document.createTreeWalker(code, NodeFilter.SHOW_TEXT);
  let seen = 0;
  let placed = false;

  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const len = node.nodeValue.length;

    if (!placed && start < seen + len) {
      range.setStart(node, start - seen);
      placed = true;
    }

    if (placed && end <= seen + len) {
      range.setEnd(node, end - seen);

      const rects = range.getClientRects();
      if (!rects.length) return fallback;

      const first = rects[0];
      const last = rects[rects.length - 1];
      const origin = code.getBoundingClientRect().top;
      const pad = parseFloat(style.paddingTop);

      return {
        top: pad + first.top - origin,
        height: last.bottom - first.top,
      };
    }

    seen += len;
  }

  return fallback;
}

export function positionLineMark() {
  const markedLine = selection.state.line;

  if (markedLine < 0) {
    lineMark.hidden = true;

    return;
  }

  const { top, height } = sourceLineBox(markedLine);
  lineMark.style.top = `${top - input.scrollTop}px`;
  lineMark.style.height = `${height}px`;
  lineMark.hidden = false;
}

export function markSourceLine(index) {
  let markedLine = -1;
  const snap = renderer.snapshot;
  const printed = snap && index >= 0 ? (snap.irLineOf?.(index) ?? 0) : 0;

  if (printed > 0) {
    markedLine = printed - 1;
  } else if (snap && index >= 0) {
    const op = snap.labelOf(index).split(" ")[0];
    let nth = 0;

    for (let i = 0; i < index; i++) {
      if (snap.labelOf(i).split(" ")[0] === op) nth++;
    }

    const escaped = op.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = new RegExp(`(^|[^\\w.])${escaped}([^\\w.]|$)`);
    const lines = input.value.split("\n");
    markedLine = lines.findIndex((line) => pattern.test(line) && nth-- === 0);
  }

  showSourceLine(markedLine);
}

export function showSourceLine(line) {
  selection.select({ line });
}

export function setupSourceTabs() {
  for (const { tab, textarea } of TABS) {
    tab.addEventListener("click", () => showTab(textarea));
  }

  sourceTitle.addEventListener("click", showSourceTab);

  splitToggle.addEventListener("click", toggleSplit);

  selection.subscribe((selected, picked) => {
    if (!picked.includes("line")) return;

    const markedLine = selected.line;

    if (markedLine >= 0) {
      if (
        !sourcePane.classList.contains("split") &&
        sourceTitle.getAttribute("aria-selected") !== "true"
      ) {
        showTab(input);
      }

      const y = sourceLineBox(markedLine).top;

      if (
        y < input.scrollTop ||
        y > input.scrollTop + input.clientHeight - 40
      ) {
        input.scrollTop = Math.max(0, y - input.clientHeight / 3);
      }

      renderExplain(markedLine);
    }

    focusGpu(markedLine);
    if (state.canvasView === "local") focusLocalLine(localViewEl, markedLine);
    positionLineMark();
  });
}
