# `src/app/` 🪟

**The shell around the visualization: the command palette, the inspector drawer, the pass slider and the other controls you work with.**

These are the pieces `main.js` mounts onto the page. They hold no compiler knowledge of their own.

## 🧰 What's here

- **`palette.js`**: the command palette (⌘K / Ctrl K) and its fuzzy search.
- **`inspector.js`**: the tabbed drawer beside the picture. Opens on a pick, closes with `esc`, and returns focus.
- **`selection.js`**: the one store for what is selected (pass, graph node, source line). Views read it and subscribe.
- **`scrubber.js`**: the pass slider, with ticks for failed and slow passes and a hover preview.
- **`splitters.js`**: resizable panes.
- **`mlir-highlight.js`**: syntax highlighting for the code editors.
- **`buffers-view.js`**: how the Buffers tab is drawn.

## 🔗 How other folders use it

Import from the barrel, `index.js`.

## 🚦 Import rules

- May import `trace/` (for example `formatBytes`).
- Imported by `main.js` only.

## 🚫 What doesn't belong here

- Compiler analysis; it goes in `trace/` or `gpu/`.
- Big views of the IR; those go in `render/`.

Most of the app's wiring (keyboard shortcuts, each inspector tab's contents) is still in `main.js`; moving it here is a known follow-up.
