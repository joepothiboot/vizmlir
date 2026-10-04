---
name: vizmlir-development
description: Develop and validate VizMLIR changes across its Vite frontend and Rust WASM parser, and find where a feature is implemented (feature-to-code map).
---

# VizMLIR development

Use this workflow for changes that cross the JavaScript frontend and the Rust WASM module.

## Workflow

1. Read `CONTRIBUTING.md`. To find where a feature lives, start from `references/feature-map.md`, then inspect the nearest implementation and call sites.
2. Make the smallest change that preserves the existing browser-only design.
3. If the exported Rust ABI changes, update `src/ir/abi.js` in the same change.
4. Run `npm run build`.
5. For parser or rendering changes, exercise a representative valid and invalid MLIR input in the app.
6. If you added, moved, or removed a feature, a module, or a `main.js` section banner, update `references/feature-map.md` in the same change.

## Bug reports

1. Ask or check whether the bug is on the live site (https://joepothiboot.github.io/vizmlir/) or a local dev server, and reproduce it there before changing code.
2. If it only fails live, check the deployment before the code: `gh api repos/joepothiboot/vizmlir/pages --jq .build_type` must be `workflow`, and the served `index.html` must load `/vizmlir/assets/index-*.js`, not `./src/main.js`. See "Deployment" in `CONTRIBUTING.md`.
3. Another session may be editing this checkout or running a dev server. Start your own server on a free port with `--strictPort`, and leave changes you did not make alone.

## Boundaries

- `wasm/src/` owns parsing and the Rust-side ABI.
- `src/ir/` owns JavaScript bindings and views.
- `src/render/` owns drawing: the graph canvas, the GPU view and the 3D scene.
- `src/gpu/` and `src/trace/` own DOM-free analysis; they must not import `render/` or `app/`.
- Cross-folder imports go through each folder's `index.js` barrel; see "Project layout" in `CONTRIBUTING.md`.
- `public/mlir_core.wasm` is generated; do not edit it directly.
