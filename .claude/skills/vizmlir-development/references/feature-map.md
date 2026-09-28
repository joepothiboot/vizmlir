# VizMLIR feature map

Where each user-facing feature lives. User-facing descriptions are in `README.md` (Features) and `CHANGELOG.md`; this file only says where to look.

`src/main.js` wires everything. It is split by `// ---- <Section> ----` banners, cited below as "main.js › Section". Cite functions and banners rather than line numbers, which drift.

## Features

| Feature | Code | Tests / docs |
| --- | --- | --- |
| MLIR parsing (lexer, parser, diagnostics) | `wasm/src/parser/lexer.rs`, `wasm/src/parser/ast.rs` (`parse`), `wasm/src/arena.rs`, `wasm/src/intern.rs` | TEST_PLAN L6 (6.1, 6.2, 6.4) |
| Graph layout | `wasm/src/lib.rs` (`mlir_layout`) | TEST_PLAN 6.3 |
| WASM ABI and bridge | `wasm/src/abi.rs`, `wasm/src/lib.rs` (`mlir_*` exports) ↔ `src/wasm/abi.js`, `src/wasm/bridge.js` (`MlirEngine`), `src/wasm/views.js` (`MemoryViews`) | `tests/unit/abi-sync.test.js`, TEST_PLAN L5 |
| Graph canvas (draw, pan, zoom, fit, node colours) | `src/render/canvas-renderer.js` (`CanvasRenderer`); node styles in `KIND_STYLE` in `src/wasm/abi.js`; `fitGraph` in main.js | TEST_PLAN 1.3, L3 |
| Parse and render loop | main.js: `parse`, `run`, `scheduleRun` | TEST_PLAN L2 |
| Before/after diff pane | `src/diff.js` (`diffSnapshots`); main.js: `renderDiff`, `focusChange`, `syncDiffSelection` | `tests/unit/diff.test.js`, TEST_PLAN 1.4, 4.2 |
| Pick and copy changes (`x`, `c`, Esc) | main.js: `togglePick`, `pickRange`, `clearPicks`, `copyChanges` | TEST_PLAN 1.4 |
| Editors and syntax highlighting | `src/mlir-highlight.js` (`highlightMlir`, `bindHighlighting`) | `tests/unit/mlir-highlight.test.js`, TEST_PLAN 1.2, 4.1 |
| Source tabs and split (`t`, `s`) | main.js › Source tabs: `showTab`, `toggleTab`, `toggleSplit` | TEST_PLAN 1.2 |
| Change ↔ source-line marker | main.js › Source line marker: `markSourceLine`, `positionLineMark` | — |
| Pass traces (`-mlir-print-ir-*-all`) | `src/trace.js` (`isPassTrace`, `parsePassTrace`, `baselineFor`, `extractSymbolOp`); main.js: `loadTrace`, `selectEvent`, `clearTrace`, `renderDiagnostics` | `tests/unit/trace.test.js`, `tests/fixtures/traces/`, `docs/trace-format.md`, TEST_PLAN 1.6, 4.3 |
| Pass timing and peak memory (`p`) | `src/timing.js` (`extractReports`, `matchTiming`, `timingToJSON`); main.js › Timing and memory: `setProfile`, `openTiming`, `renderTiming`, `exportTiming` | `tests/unit/timing.test.js`, `tests/fixtures/traces/timing-*.txt`, `timed-run.txt` |
| Op counts per pass (`o`) | `src/opcount.js` (`countOps`, `opCountTable`, `opCountsToCSV`); `moduleStateAt` in `src/trace.js`; main.js › Op counts: `setViewCounts`, `computeTraceCounts`, `openOpCounts`, `renderOpCountTable`, `exportOpCounts` | `tests/unit/opcount.test.js` |
| Live reload of a watched trace file | `src/watch.js` (`FileWatcher`, `canWatchFiles`, Chromium only); main.js › Watch a file: `pickWatch`, `stopWatching`, `restoreWatch` | — |
| Command palette (⌘K, `/`) | `src/palette.js` (`CommandPalette`, `fuzzyScore`); items built in main.js › Command palette | `tests/unit/palette.test.js` |
| Keyboard shortcuts and `?` help | main.js › Keyboard: `WORKSPACE_KEYS` and the global `keydown` handler; help dialog markup in `index.html` | — |
| Autosave and named sessions (IndexedDB, `.json`) | `src/storage.js` (`kv`, `sessions`, `sessionToFile`, `sessionFromFile`); main.js › Workspace state: `getState`, `applyState`, `scheduleAutosave`, `renderSessions`, `downloadSession` | — |
| Export PNG / SVG / diff Markdown, JSON, patch | `src/export.js` (`download`, `diffToMarkdown`, `diffToJSON`, `diffToPatch`); main.js › Export: `exportPNG`, `exportSVG`, `exportDiff` | `tests/unit/export.test.js` |
| Built-in samples | `src/samples.js` (`SAMPLES`, `RENAME_SAMPLE`, `loadSampleState`); generated files in `public/samples/` from `samples/input/` via `samples/generate.sh`; main.js › Samples | `tests/unit/samples.test.js` |
| Open… a file (routes `.json` sessions, traces, or plain MLIR) | main.js › Loading (`fileInput` change handler) | TEST_PLAN 1.1 |
| Resizable panes | `src/splitters.js` (`bindSplitters`); main.js › Pane splitters | — |
| Theme (`L`) and menu (`m`) | main.js › Theme: `applyTheme`, `toggleTheme`; `toggleMenu`, `setMenuHidden`; CSS tokens in `index.html` | — |
| Docs route (`#/docs`) | main.js: `updateRoute`; content and styles in `index.html` (`#docs-view`) | TEST_PLAN 1.7 |
| Status bar and error states | main.js: `setStatus`; `STATUS`, `STATUS_TEXT`, `DIAG_CODE` in `src/wasm/abi.js` | TEST_PLAN 1.4, 1.5 |
| Deployment (GitHub Pages) | `.github/workflows/deploy-pages.yml`, `vite.config.js` (base path) | CONTRIBUTING.md › Deployment |
| README screenshots | `public/demo/` | — |

## Notes

- Tests exist only as Vitest unit tests (`tests/unit/`). The TEST_PLAN L1 Playwright cases and the L6 `cargo test` cases are planned, so a TEST_PLAN reference does not mean a test exists yet.
- All markup and CSS live in `index.html`; there are no component files.
