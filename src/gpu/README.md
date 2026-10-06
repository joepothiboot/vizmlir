# `src/gpu/` 🖥️

**The brain of the GPU view: what a kernel launches, where its data lives, and whether each memory access is a good one.**

These modules read IR text and produce plain data: a model of each launch, a verdict for each load and store, and a proof of how far that verdict holds. Nothing here draws anything.

## 🧰 What's here

- **`model.js`**: `analyzeGpu`, which finds each `gpu.launch` (grid, block, buffers) and classifies arrays as global, shared or private memory.
- **`access-ir.js`**: reads kernel IR into definitions, loops and loads/stores, and evaluates an index for one thread.
- **`access.js`**: the verdicts. Evaluates a load or store for one warp, rules it *coalesced*, *strided*, *misaligned* or *broadcast* (or a shared-memory *bank conflict*), and proves it for every warp and loop iteration.
- **`affine.js`**: the proof engine's algebra. Turns an index into a linear expression over thread ids, block ids and loop counters.
- **`flow.js`**: the kernel's chain from ids to index math to loads and stores.
- **`triton.js`**: Triton GPU IR. Maps `#blocked` layouts to threads and gives `tt.load` and `tt.store` the same verdicts.
- **`local-memory.js`**: DSP scratchpad buffers and DMAs, with the schedule they imply.
- **`local-scan.js`**: the IR scan behind it: functions, definitions, subviews and buffer slots.
- **`local-pipeline.js`**: the DMA/compute pipeline of a scratchpad loop (single or double buffered).

## 🔗 How other folders use it

Import from the barrel, `index.js`.

## 🚦 Import rules

- Imports only `trace/` (memref parsing, symbol scans, formatters) and the root `constants.js`.
- Imported by `render/` and `main.js`.
- No DOM and no `window`, so it runs under Node and in unit tests.

## 🚫 What doesn't belong here

- Anything that creates page elements or SVG; that's `render/`.
- Pass-log parsing (`trace/`).
