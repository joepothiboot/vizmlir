# `src/session/` 💾

**Keeping your work: autosave, named sessions, share files, exports and live reload.**

Everything stays in your browser. This folder is where the app talks to browser storage, downloads and the file system.

## 🧰 What's here

- **`storage.js`**: browser-local autosave and named sessions in IndexedDB, and the `.json` share file format.
- **`export.js`**: `download`, plus the pass diff as Markdown, JSON or a patch.
- **`watch.js`**: live reload of a watched trace file through the File System Access API (Chrome and Edge).

## 🔗 How other folders use it

Import from the barrel, `index.js`. PNG and SVG export of the diagram currently lives in `main.js`.

## 🚦 Import rules

- Imports nothing else in `src/`.
- Imported by `main.js`.
- Storage and watching are guarded, so the app still runs where they are unavailable (a private window, Firefox).

## 🚫 What doesn't belong here

- Analysis or drawing.
- Anything that sends data off the machine. VizMLIR never uploads your code.
