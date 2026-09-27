// Draggable dividers between the workspace panes. Each splitter resizes one
// side pane; the graph stage in the middle takes whatever width is left.

const STORAGE_KEY = "vizmlir-pane-widths";
const MIN_PANE = 180;
const MIN_STAGE = 240;
const KEY_STEP = 16;

function loadWidths() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch {
    return {};
  }
}

function saveWidths(widths) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(widths));
  } catch {
    // Storage can be unavailable (private mode); resizing still works.
  }
}

/**
 * @param {HTMLElement} main  flex row holding the panes and splitters
 * @param {{ el: HTMLElement, pane: HTMLElement, side: "left" | "right" }[]} splitters
 */
export function bindSplitters(main, splitters) {
  const widths = loadWidths();
  const stacked = () => getComputedStyle(main).flexDirection === "column";

  // Keep the stage at least MIN_STAGE wide, counting the other side pane.
  function clamp(pane, width) {
    const others = splitters
      .filter((s) => s.pane !== pane)
      .reduce((sum, s) => sum + s.pane.offsetWidth + s.el.offsetWidth, 0);
    const max = main.clientWidth - others - MIN_STAGE - 6;
    return Math.round(Math.max(MIN_PANE, Math.min(width, max)));
  }

  function setWidth(splitter, width, persist = true) {
    const w = clamp(splitter.pane, width);
    splitter.pane.style.width = `${w}px`;
    splitter.el.setAttribute("aria-valuenow", String(w));
    if (persist) {
      widths[splitter.pane.id] = w;
      saveWidths(widths);
    }
  }

  for (const s of splitters) {
    const { el, pane, side } = s;
    el.setAttribute("role", "separator");
    el.setAttribute("aria-orientation", "vertical");
    el.setAttribute("aria-controls", pane.id);
    el.setAttribute("aria-valuemin", String(MIN_PANE));
    el.tabIndex = 0;
    if (widths[pane.id]) setWidth(s, widths[pane.id], false);

    el.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || stacked()) return;
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      const startX = e.clientX;
      const startW = pane.offsetWidth;
      document.body.classList.add("resizing");
      el.classList.add("dragging");

      const move = (ev) => {
        const dx = ev.clientX - startX;
        setWidth(s, side === "left" ? startW + dx : startW - dx, false);
      };
      const end = () => {
        el.removeEventListener("pointermove", move);
        el.removeEventListener("pointerup", end);
        el.removeEventListener("pointercancel", end);
        document.body.classList.remove("resizing");
        el.classList.remove("dragging");
        setWidth(s, pane.offsetWidth);
      };
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", end);
      el.addEventListener("pointercancel", end);
    });

    // Double-click restores the stylesheet width.
    el.addEventListener("dblclick", () => {
      pane.style.width = "";
      delete widths[pane.id];
      saveWidths(widths);
      el.setAttribute("aria-valuenow", String(pane.offsetWidth));
    });

    el.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      e.stopPropagation();
      const grow = (e.key === "ArrowRight") === (side === "left");
      const step = e.shiftKey ? KEY_STEP * 4 : KEY_STEP;
      setWidth(s, pane.offsetWidth + (grow ? step : -step));
    });
  }

  // Re-clamp saved widths when the window shrinks so the stage never vanishes.
  window.addEventListener("resize", () => {
    if (stacked()) return;
    for (const s of splitters) {
      if (s.pane.style.width) setWidth(s, s.pane.offsetWidth, false);
    }
  });
}
