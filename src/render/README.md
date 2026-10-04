# `src/render/` 🎨

**Everything that gets drawn: the op graph, the GPU view, the 3D launch and the Local memory view.**

These modules take the data from `gpu/`, `trace/` and `ir/` and turn it into pixels and page elements.

## 🧰 What's here

- **`canvas-renderer.js`**: the op graph on a canvas, with pan, zoom and fit.
- **`gpu-view.js`**: the GPU view. The memory accesses table, the answer card with its Plain and Compiler modes, the Across passes strip, and the "Who holds which element" grid for Triton.
- **`gpu-3d.js`**: the launch in 3D. Blocks and warps, the memory plate, and the Elements layer.
- **`descent.js`**: the Descent view: every pass as a layer in a 3D stack, with the selected op's path down through them and a camera that slides to the lit pass.
- **`local-view.js`**: the Local memory tab. The scratchpad budget bar and the DMA, wait and compute lanes.

## 🔗 How other folders use it

Import from the barrel, `index.js`. Each view exports a `render…` function that fills an element you give it.

## 🚦 Import rules

- May import `gpu/`, `trace/` and `ir/`. Nothing may import it except `main.js`.
- Analysis never flows the other way: if a view needs a new fact, compute it in `gpu/` or `trace/` and import it.

## 🚫 What doesn't belong here

- Logic that decides a verdict or reads IR text.
- The app shell (`app/`) and exporting files (`session/`).
