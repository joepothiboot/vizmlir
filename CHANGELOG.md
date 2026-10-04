# Changelog 📜

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- The in-app **Learn** course (`#/docs`), its `src/app/guide.js`, and the ← → chapter keys. The docs live in a separate project.

### Added

- A **Local memory** view for DSP-style scratchpads. When the IR at the current pass has buffers in nano-dsp's `#dsp.local` memory space, `memref.dma_start` / `memref.dma_wait`, or the `linalg.copy` they are lowered to, a new tab of the visualization pane draws the local budget as a bar of buffer slots (A slot 0/1, B slot 0/1) and the cache loop's first trips as DMA, wait and compute lanes, marking prologue loads, prefetches of the next tile, and waits. Each block and event picks its IR line and the inspector's Line tab explains it. The budget is read from the IR when it carries one, otherwise from nano-dsp's TargetModel and labelled as assumed, and the view states that it shows the order the IR expresses, not timing. The analysis (`src/gpu/local-memory.js`) runs per pass: nothing before `nanodsp-promote-local`, DMAs after it, copies after `nanodsp-lower-local`.
- A "DSP scratchpad: double-buffered DMA" sample, real `nanodsp-opt` output from nano-dsp-mlir 8c23df8 (`NANODSP_OPT=... samples/generate.sh`; the version is in `public/samples/NANODSP_VERSION`).
- A guide lesson, "Scratchpads and DMA (beyond GPUs)".

### Changed

- The workspace is the code, one wide picture, and an inspector drawer, instead of three columns that showed each verdict twice and dialogs that covered what you compare against. Graph and GPU are tabs of the picture, and over the GPU view one line keeps the picked access's verdict. The inspector holds every detail as a tab: Changes, Line ("What's this line?" with the GPU answer card and the GPU path), Timing, Buffers, Op count and Symbols, which redraw as you change passes. Picking a line, an op or an access opens it on the right tab; `esc` closes it and returns focus to what you picked, and the Inspector button or `\` reopens it. It takes a resizable column on wide screens, overlays the picture below 1100px, and is a bottom sheet on phones. `w`, `p`, `b`, `o` and `h` open their tabs, and the palette's actions point at them. The palette, samples, sessions, help and symbol view stay dialogs.
- The row of pass cards is a pass slider: drag it, use the arrow keys, or `[` `]`. Ticks mark failed and slow passes, and hovering or focusing it previews a pass (its name, IR size, how many ops it changed, and its time) before you jump. The full list is under All passes in the Changes tab.
- The GPU view now proves memory verdicts instead of only sampling them. When an index is a linear function of the thread ids, block ids and loop counters, VizMLIR checks the verdict for every warp of the launch and every loop iteration, and a new Checked column in the Memory accesses table says how far it reaches ("all 32,768 warps", "every iteration", or "warp 0 only"). A loop that drifts out of alignment, coalesced on its first trip but not on the rest, now shows as "varies" with how often each pattern occurs.
- Add a "Memory patterns" sample, real mlir-opt output from outlining to PTX: six small kernels, one pattern each, that look different in the GPU view: `{x, y}` pairs read one field at a time (strided, 50% useful) against separate arrays (coalesced), a read shifted by one element (misaligned), a sliding-window loop (varies: misaligned on 7 of every 8 trips), a bias every thread shares (broadcast), and 16×16 blocks (a warp covers two half rows).
- A new verdict, **misaligned**: the warp's elements are side by side but start partway into a sector, so they take one more sector than needed. It used to be called strided, which suggested the wrong fix. The card suggests a struct of arrays when neighboring threads are 2 to 8 elements apart, says under the chart when a verdict varies across warps or loop trips, and colors such passes amber in Across passes. Buffers whose rows are narrower than 32 elements are drawn as flat memory.
- Read Triton GPU IR (`tt.`/`ttg.`, or the older `triton_gpu.`). Each `tt.func` is a kernel whose program runs `num-warps` warps. `tt.load` and `tt.store` offsets are followed through `tt.addptr`, `tt.splat`, `tt.make_range`, `tt.expand_dims`, `tt.broadcast` and `ttg.convert_layout` as linear functions of the element coordinates and program ids, so each access gets the same answer card, verdict and proof as memref code, with every thread's whole vector counted. A new "Who holds which element" grid draws the `#blocked` layout of the picked tensor, colored by warp and numbered by lane. Add a Triton sample (hand-written in Triton 3.x TTGIR format, marked as such): the tritongpu-coalesce pass turns a strided 32-sector load into a coalesced one of 4-wide vectors.
- The 3D view draws the picked access's elements on the memory plate beside the opened block, with a line from each thread of warp 0 to the element it touches: straight across for a coalesced row, fanning into one column for a strided access or a bank conflict. In Compiler mode it also writes the offset's formula under the block.
- The GPU view follows kernels through the whole lowering pipeline. It reads `affine.apply` (inline maps and `#map` aliases), `affine.load`/`affine.store` index expressions and `affine.for`, and kernels after convert-gpu-to-nvvm: LLVM arithmetic, NVVM thread and block ids, loads and stores through `getelementptr` traced back to the kernel argument or workgroup global they address, memref arguments split into pointers, sizes and strides, and loops lowered to branches (a block argument that starts at a value, steps by a constant and is bounded by a compare). In the transpose and tiled matmul samples every access stays proven up to the pass that compiles the kernel to a binary, and Across passes shows it all the way.
- Each GPU launch opens with an answer card for the picked load or store: its verdict, how far it was checked, one plain sentence ("Each warp touches 32 separate 32-byte chunks, so the GPU moves 8× more data than it uses") and a bar for the share of useful memory traffic or bank passes. Its Plain | Compiler switch swaps the sentence for the index math: the element offset as a formula of the ids, the distance between neighboring threads, and what the proof covered. The choice is remembered.
- With a pass trace open, the answer card shows the picked access at every pass, colored by its verdict, with a sentence on what the passes did to it (for the transpose sample: "Strided · 32 sectors in passes 1–4: no pass changes it, so a fix belongs in the source"). A click on a pass selects it. Strided accesses and bank conflicts also get a one-line usual fix.
- The field guide explains the answer card, the proven / varies / warp 0 only chips, Compiler mode, Across passes and Elements, and the README screenshots show the new card.
- When a verdict is proven, the 3D view tints every block of the grid with it and says "Same pattern in every block: proven".
- The map of the elements a warp touches moved into the 3D view: the Elements tool (or `e`) shows the buffer from above over the scene, and stays on as you pick other accesses.
- VizMLIR is now framed as a tool for seeing how MLIR runs on the GPU. The canvas opens on the GPU view whenever the IR launches kernels, and remembers when you switch to the op graph instead (IR without GPU code always shows the graph). A first visit opens the transpose sample, the GPU samples come first in the Samples list, and the field guide, README, and page title lead with the GPU view. The op graph, pass diff, and trace tools are unchanged.
- Add "What's this line?", a panel under the code that explains the line you click (or the line picked in the graph or the GPU view) in plain words: its result, operation (with a one-sentence description for about 70 common operations and a link to its dialect's MLIR docs), inputs linked to the lines that define them, position, settings, and type (for example "a 1024 × 1024 array of 32-bit decimal numbers in memory (4.0 MB)"). For loads and stores it adds the GPU verdict, example elements for the first threads, and a button that opens that access in the GPU view. `w` hides or shows it, and the choice is remembered.
- The workspace is roomier: the source, canvas, and diff panels are separate rounded cards in a padded frame, with wider gutters (the drag handles), and more padding in the top bar, pass strip, code, diff list, and status bar.
- The GPU view now draws each launch in 3D on a canvas, replacing the flat grid and block diagrams: the grid of blocks laid out by their x, y, z ids, one block opened into its threads (each row of 32 is a warp), and floor plates for global and shared memory. Drag to turn it, shift-drag to pan, ⌘/Ctrl + scroll or pinch to zoom (or use the −, +, Top, and Reset buttons, or the arrow keys, + / −, t, and 0 when it has focus), hover a block or thread for its ids, and click a block to open it. Picking a row in Memory accesses raises warp 0 and colors each thread by the sector or bank it hits. It follows the light and dark themes. The launch's facts (grid, block, threads) sit on the canvas's top edge and the Memory section is a panel over its right side (collapsed to "Memory · N buffers" on narrow screens), so a launch reads in one place; hovering a buffer there also outlines its memory's floor plate.
- Trace how the IR reaches the GPU. The 3D view lays the kernel's IR on a board in front of the launch: the thread and block ids, the index math built from them, and each load and store, with arched edges between them and dashed drops into the block, threads, and memory they name (the IR button or `i` hides it). Click a line of kernel code, or a node, to light that line's path from the ids to memory; the load or store is picked in Memory accesses, and clicking a node marks its line in the code.
- The side panel follows the canvas: with the GPU view on it switches from Changes to a GPU path tab that walks the picked line's path as steps (where the thread is, index math, memory with its verdict), each linked to its line (a click also scrolls the GPU view to that launch's 3D view and flashes it); with nothing picked it lists the loads and stores to start from. The Changes tab is one click away.
- The buffers in the GPU view's Memory section (Global, Shared, Private) are now interactive. Hovering one highlights its loads and stores in the Memory accesses table; clicking it opens what it holds, its size (with the number of copies for shared and private buffers), where it lives, where it comes from, and each load and store with its verdict. Clicking one of those opens its lanes and marks its line in the source.
- Turn the field guide into a short course for CS students, under **Learn** in the top bar. Every chapter sits on one scrolling page inside a padded frame, grouped into Start here, Lessons, Lab, and Reference, with a sticky contents list (a scrolling row on narrow screens) that highlights the chapter you're reading; click it or press ← → to jump. Four lessons (how a GPU runs code, memory and access patterns, reading MLIR, how a compiler builds a kernel) each have learning goals, a "Try it" in the app, and "Check yourself" questions, followed by lab chapters, the model's limits, and a glossary. The URL stays `#/docs`. The README leads with the same framing.
- Keep the pass strip hidden when a trace loads while the guide is open.
- Rewrite the field guide and README for newcomers: a one-minute tour, "the GPU in four ideas" (threads, blocks and the grid, warps, memory) with everyday comparisons, what each memory verdict means and why it matters, and the `mlir-opt` flags explained in plain words.

### Added

- Read `-mlir-timing` reports (tree, list, and JSON output) from pass traces and pasted logs. Each pass in the timeline shows its wall time and IR size, and **p** opens the full report with links back to each pass.
- Show whole-process peak memory when the log includes `/usr/bin/time -l` (macOS) or `/usr/bin/time -v` (Linux) output. mlir-opt does not measure memory per pass.
- Download the timing report, peak memory, and per-pass IR sizes as JSON.
- Document the accepted trace format and the parsed structure in `docs/trace-format.md`.
- Test traces from out-of-tree `MlirOptMain` drivers with custom dialects (a `schema-opt` fixture), including namespace-qualified pass names.
- Show the buffers the IR allocates (`memref.alloc`, `memref.alloca`, `gpu.alloc`), with their static size and live range, baseline against current. **b** opens a lifetime chart per function with added, removed and changed buffers and a live-bytes curve, and for traces, buffer count, allocated bytes and peak live bytes at every pass. The status bar shows the current peak and its change. Click a buffer to show its line; download the comparison as JSON.
- Count operations by name. The status bar shows the current op count and its change from the baseline, and **o** opens a table of counts at every dump of a trace (whole module, with nested dumps spliced in). Filter by op name, hide ops or passes that change nothing, jump to a pass from its column, and download the table as CSV.
- Follow every function, kernel, GPU module and global through a trace. **h** lists, per symbol, the pass that created it (for example `gpu-kernel-outlining`), the passes that changed its body, lowered it to another op (`gpu.func` → `llvm.func`), or removed it, with links to each pass. Nested symbols are named `@module::@kernel`, as in `gpu.launch_func`. Download the history as JSON.
- Import kernel benchmark results into the symbol history: Nsight Systems kernel summaries, Nsight Compute `gpu__time_duration.sum` metrics, Google Benchmark JSON, or a generic CSV/JSON with the unit in the time header. Each kernel is matched to its symbol by name (including C++ and Itanium-mangled names) or by an explicit `symbol` column, and its time per call shows next to the passes that built it. Unmatched and ambiguous kernels are listed; the results are saved with the session and added to the JSON export. The format is documented in `docs/benchmark-format.md`.
- Compare two benchmark runs against a trace: import a baseline next to the current results to see each kernel's baseline and current time per call and its change, largest slowdown first, with a filter for kernels that changed by more than a set percentage. Sessions saved with a single run open as the current run.
- Add a "GPU kernels + benchmarks (mock)" sample: a real `mlir-opt` trace of two kernels from outlining to PTX, with invented baseline and current timings that show the benchmark comparison. The Symbol history marks mock data with a note.
- Open the Symbol history from the status bar while a trace is loaded (**@ symbols**, or **@ N slower** when two benchmark runs are compared). From the symbol view, **Show in graph** goes to that pass with the symbol's graph node selected and its line marked.
- Follow one symbol through the trace: click a symbol in the Symbol history to see its IR at each pass that created, changed, lowered, or removed it, as a line diff against the previous step or as full IR (`j`/`k` to step, `d`/`i`/`a` to switch). When a kernel is serialized by `gpu-module-to-binary{format=isa}`, the embedded PTX (or other text assembly) is decoded and shown. Benchmark times for the symbol appear in the header.
- Add a GPU view to the canvas (**Graph | GPU**, or `g`) whenever the IR has GPU code. For each `gpu.launch` / `gpu.launch_func` it shows the grid of blocks and one block opened into warps of 32 threads, with sizes read from constant launch operands, and the kernel's buffers by memory space: global, shared per block (`workgroup` attributions, address space 3), and private per thread, with sizes and load/store counts. After `gpu-module-to-binary{format=isa}` it reads PTX register and shared-memory declarations instead. Everything is read from the IR, and the view says so.
- Judge each kernel's memory accesses in the GPU view. For every `memref.load` / `memref.store`, the element each thread of warp 0 touches is worked out from its index arithmetic (thread and block ids, launch sizes, constants, loop starts, constant launch arguments), and the warp is judged: coalesced, strided (with how many 32-byte sectors it moves against the minimum), or broadcast for global memory; conflict-free or an N-way bank conflict for shared memory. Picking an access shows its 32 lanes colored by sector or bank, the elements they touch, a plain-language explanation with the usual fix, and marks its line in the source. Indices built from anything else are reported as not analyzed.
- Add a "Transpose: strided, bank conflict, fixed" sample: a real `mlir-opt` trace of a naive transpose (strided write), a 32×32 shared tile (32-way bank conflict), and the tile padded to 32×33 (conflict-free).
- Add a "Tiled matmul on the GPU" sample: a real `mlir-opt` trace of an 8×8 grid of 16×16-thread blocks that stage tiles in shared memory, opened in the GPU view.

### Fixed

- Count a `{` nested in a type or attribute (`#ttg.slice<{dim = 1, ..}>`) as well as its `}`. Triton GPU IR was reported as an unbalanced brace.
- Keep a block label (`^bb1:`) at its function's own indent inside that function when scanning symbols. Everything after the first block label of a lowered kernel was cut off, so symbol history and the GPU view missed it.
- Label ops whose operands contain `=` by their op name: `scf.for %i = %c0 to ..` was labeled `%c0`, and `linalg.generic {indexing_maps = ..}` was labeled `[`. Values bound with `%x =` inside an op (induction variables, `iter_args`) are now definitions.
- Stop treating a region op's trailing types (`} -> tensor<..>`, `} : ..`) as a separate `->` op.
- Function arguments (`func.func @f(%arg0: T)`), the ids and buffers `gpu.launch` binds in `blocks(..)`, `threads(..)`, `clusters(..)`, `workgroup(..)` and `private(..)`, and the induction variables of `scf.forall` and `scf.parallel` are now definitions. They were reported as undefined SSA values (60 warnings on the transpose sample), and their uses had no edges in the graph.
- The launch facts write the warp count with a thousands separator ("32,768 warps"), like the rest of the GPU view.

### Internal / Chore

- Group `src/` by domain instead of one flat folder: `ir/` (renamed from `wasm/`, so it no longer looks like the Rust crate in `wasm/src/`), `trace/`, `gpu/`, `render/`, `session/` and `app/`, each with an `index.js` barrel. Moves were done with `git mv` in content-free commits listed in `.git-blame-ignore-revs`; imports were then rewritten mechanically. No behavior, rendering, shortcut or export format changed. `src/main.js` is still the entry. Anything that imports `src/ir/bridge.js` or `src/ir/diff.js` directly must switch to `src/ir/bridge.js` and `src/ir/diff.js`.

## [0.3.0] - 2026-09-26

### Added

- Load `mlir-opt -mlir-print-ir-after-all` / `-mlir-print-ir-before-all` logs via **Open…** or by pasting, and step through each pass dump with its baseline IR.
- Mark failed passes and show compiler errors, warnings, notes, and remarks printed alongside the dumps.
- Rebuild module baselines from nested function dumps when `-mlir-print-ir-module-scope` is not used.

### Fixed

- `#alias = ...` definitions and trailing `} loc(...)` no longer appear as operations in the graph or diff.

## [0.2.0] - 2026-09-03

### Added

- Live baseline/current MLIR comparison with SSA-renumbering-aware structural diffs.
- Side-by-side MLIR editors with added, removed, and changed operation reporting.

### Fixed

- Restored the Rust lexer and WebAssembly parser build used by the visualizer.
- Fixed canvas and editor selectors so parsing and graph rendering initialize correctly.

## [0.1.0] - 2026-09-01

### Added

- Initial project structure and setup.
- Basic integration for MLIR code rendering.
- Interactive graph visualization interface.
- Core project documentation (`README.md`).
