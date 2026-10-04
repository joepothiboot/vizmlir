# `src/trace/` 🧭

**Everything that reads a compiler's log: the passes, what each one changed, and how long it took.**

Run `mlir-opt` with `-mlir-print-ir-after-all` and you get a long log. These modules split it into passes and answer questions about it, without touching the page.

## 🧰 What's here

- **`trace.js`**: parses a pass log into events (the pass slider, failed passes, diagnostics) and rebuilds the whole module as it stood at any pass. The format is in [docs/trace-format.md](../../docs/trace-format.md).
- **`timing.js`**: reads `-mlir-timing` reports and matches them to passes. Also home to the small `formatBytes` and `formatSeconds` helpers many views share.
- **`opcount.js`**: how many of each op the module has at every pass.
- **`buffers.js`**: memref lifetimes, the peak of live bytes, and baseline-versus-current comparison. Also parses memref types (`parseMemref`, `elementBytes`) for the GPU code.
- **`provenance.js`**: symbol history, meaning which pass created, changed, lowered or removed each function and kernel, down to the PTX.
- **`linediff.js`**: a line-by-line diff of one symbol between two passes.

## 🔗 How other folders use it

Import from the barrel, `index.js`.

## 🚦 Import rules

- Imports only `ir/` (just `opcount.js`, for node kinds).
- Imported by `gpu/`, `render/`, `app/` and `main.js`.
- No DOM, no `window`, no drawing. These run fine under Node, which is how they are unit tested.

## 🚫 What doesn't belong here

- Anything about GPU launches or memory access patterns (`gpu/`).
- Anything that builds page elements (`render/` or `app/`).
