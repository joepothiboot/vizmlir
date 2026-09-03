---
name: vizmlir-development
description: Develop and validate VizMLIR changes across its Vite frontend and Rust WASM parser.
---

# VizMLIR development

Use this workflow for changes that cross the JavaScript frontend and the Rust WASM module.

## Workflow

1. Read `CONTRIBUTING.md` and inspect the nearest implementation and call sites.
2. Make the smallest change that preserves the existing browser-only design.
3. If the exported Rust ABI changes, update `src/wasm/abi.js` in the same change.
4. Run `npm run build`.
5. For parser or rendering changes, exercise a representative valid and invalid MLIR input in the app.

## Boundaries

- `wasm/src/` owns parsing and the Rust-side ABI.
- `src/wasm/` owns JavaScript bindings and views.
- `src/render/` owns graph drawing.
- `public/mlir_core.wasm` is generated; do not edit it directly.
