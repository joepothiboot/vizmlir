# Working on VizMLIR

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing the project. It documents the supported setup, source layout, generated files, and checks.

The shortest useful validation is:

```bash
npm run build
```

Changes to the Rust ABI require corresponding updates in `src/wasm/abi.js`. Keep generated files out of hand-edited patches.
