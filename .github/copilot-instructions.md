# VizMLIR contributor guidance

Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for repository conventions.

- Prefer small changes that fit the existing vanilla JavaScript, Rust, and Vite structure.
- Keep visualization work client-side.
- Treat `public/mlir_core.wasm`, `dist/`, and `wasm/target/` as generated output.
- Keep `wasm/src/abi.rs` and `src/wasm/abi.js` synchronized.
- Run `npm run build` after implementation changes.
