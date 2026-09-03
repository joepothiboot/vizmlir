# VizMLIR repository notes

Use [CONTRIBUTING.md](CONTRIBUTING.md) for setup, layout, and validation commands.

Important project constraints:

- The app runs entirely in the browser; do not add a backend for ordinary visualization work.
- Rust code in `wasm/src/` is compiled to `public/mlir_core.wasm` by `npm run wasm`.
- Keep the Rust exported ABI and `src/wasm/abi.js` synchronized.
- Validate changes with `npm run build`; inspect the app for parser and rendering changes.
