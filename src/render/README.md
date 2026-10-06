# `src/render/` 🎨

**Everything that gets drawn: the op graph, the GPU view, the 3D launch and the Local memory view.**

These modules take the data from `gpu/`, `trace/` and `ir/` and turn it into pixels and page elements.

## 🧰 What's here

- **`canvas-renderer.js`**: the op graph on a canvas, with pan, zoom and fit.
- **`gpu-view.js`**: the GPU view: one section per launch with its facts and the memory accesses table, plus the path a line takes into the GPU.
- **`gpu-verdict.js`**: verdict chips and the plain and compiler explanations of a verdict.
- **`gpu-answer.js`**: the answer card for the picked access, its Plain/Compiler toggle, and the measured counter beside the prediction.
- **`gpu-passes.js`**: the Across passes strip: how a verdict changes from pass to pass.
- **`gpu-layout.js`**: the "Who holds which element" grid for Triton and the lane-by-lane detail.
- **`gpu-memory.js`**: the Memory panel: buffers by memory space and the PTX note.
- **`gpu-3d.js`**: the launch in 3D: the canvas, its tools, picking and camera control.
- **`gpu-3d-model.js`**: the scene's static model (blocks, threads, IR nodes, memory sizes) built once per launch.
- **`gpu-3d-draw.js`**: one frame of the scene, drawn in steps (plates, tiles, cubes, edges, labels, titles).
- **`gpu-3d-geometry.js`**: launch layout, the camera and the hit test.
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
