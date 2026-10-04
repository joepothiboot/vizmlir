// Public surface of src/render/. Cross-folder imports go through here.
export { focusGpuLine, renderGpuPath, renderGpuView, verdictSummary } from "./gpu-view.js";
export { CanvasRenderer } from "./canvas-renderer.js";
export { focusLocalLine, localExplain, renderLocalView } from "./local-view.js";
export { createDescentView } from "./descent.js";
