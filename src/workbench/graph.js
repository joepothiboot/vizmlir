import { CanvasRenderer } from "../render/index.js";
import { syncDiffSelection } from "./diff-panel.js";
import { canvas } from "./dom.js";
import { sourceNote } from "./source-files.js";
import { markSourceLine } from "./source-tabs.js";
import { selection } from "./state.js";

const detailEl = document.getElementById("detail");
const zoomLevel = document.getElementById("zoom-level");

export const renderer = new CanvasRenderer(canvas, {
  onSelect(index, snap) {
    const parent = index < 0 ? -1 : snap.parentOf(index);

    detailEl.textContent =
      index < 0
        ? "—"
        : `#${index} ${snap.labelOf(index)}` +
          (parent < 0 ? "" : ` · parent ${snap.labelOf(parent)}`) +
          sourceNote(index);

    syncDiffSelection(index);
    selection.select({ node: index });
    markSourceLine(index);
  },
  onViewChange(scale) {
    zoomLevel.textContent = `${Math.round(scale * 100)}%`;
  },
});
