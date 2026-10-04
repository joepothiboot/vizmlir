# `src/ir/` 🔌

**The JavaScript side of the Rust parser: load the WebAssembly module, talk to it, and read what it returns.**

The MLIR text you paste is parsed by Rust code compiled to WebAssembly (`wasm/src/`). This folder is the only place the browser app touches that module, so everything else can work with plain JavaScript objects.

## 🧰 What's here

- **`abi.js`**: the shared constants (header slots, strides, status and node-kind codes, node colours). It must match `wasm/src/abi.rs`; a unit test fails if they drift.
- **`bridge.js`**: `MlirEngine`, which instantiates the module, sends it text and hands back a snapshot of the parsed graph.
- **`views.js`**: `MemoryViews`, typed windows onto the module's memory so a snapshot can be read without copying.
- **`diff.js`**: `copySnapshot` and `diffSnapshots`, which compare two parsed modules and report added, removed and changed ops.

## 🔗 How other folders use it

Import from the barrel, `index.js`. `compiler-tooling-lab` also loads `bridge.js` and `diff.js` directly to capture its demo diffs, so keep their paths stable.

## 🚦 Import rules

- Imports nothing else in `src/`.
- Imported by `trace/`, `render/` and `main.js`.

## 🚫 What doesn't belong here

- Anything that reads IR *text* (that's `trace/` or `gpu/`).
- Rust source (`wasm/src/`) or the generated `public/mlir_core.wasm`.
- Drawing code (`render/`).

If you change the Rust ABI, update `abi.js` in the same change.
