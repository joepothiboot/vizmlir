# `src/workbench/` 🧩

**The page: every panel, button and keyboard shortcut, wired to the analyses in `trace/`, `gpu/` and `render/`.**

Each module owns one feature and exports a `setup…()` function that attaches its event listeners. `main.js` calls them in a fixed order, so listeners and selection subscribers run in the same order on every load.

## 🧰 What's here

- **`state.js`**: the shared application state (the loaded trace, the pass on show, the diff rows, the current GPU model…) and the selection store.
- **`dom.js`**: page elements used by more than one module, and `setStatus`.
- **`engine.js`**: loads the WASM parser and wraps `parse`.
- **`graph.js`**: the op graph renderer.
- **`pipeline.js`**: parses the editors into the graph, diff and analyses.
- **`trace-nav.js`**: loading a pass trace, the pass list and the scrubber.
- **`diff-panel.js`**: the Changes tab and copying changes as a patch.
- **`debugger.js`**: Debug mode and breakpoints.
- **`source-tabs.js`**, **`source-files.js`**: the Baseline/Current/Source tabs and the source-line marker.
- **`layout.js`**: theme, pane splitters and the inspector drawer.
- **`commands.js`**: the command palette and keyboard shortcuts.
- **`scenarios.js`**: loading files and the sample dialog.
- **`workspace.js`**: autosave and named sessions.
- **`timing-panel.js`**, **`opcount-panel.js`**, **`buffers-panel.js`**, **`symbol-history.js`**, **`symbol-view.js`**: the inspector's analysis tabs.
- **`canvas-views.js`**, **`line-explain.js`**, **`op-history.js`**: the Graph/GPU/Local/Descent views, the explained line and an op's history.
- **`files.js`**: export and live file watching.

## 🔗 How other folders use it

Only `main.js` imports it, file by file.

## 🚦 Import rules

- May import every other folder through its barrel.
- State that more than one module changes lives in `state.js`; state one module owns stays in that module.
