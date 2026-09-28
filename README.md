# VizMLIR

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**See how your MLIR runs on the GPU.** VizMLIR follows GPU kernels from MLIR to PTX, in your browser: how each launch splits into blocks and warps, where its data lives, whether each warp's loads and stores are coalesced or bank-conflicted, and which compiler pass produced them.

**[Try VizMLIR live](https://joepothiboot.github.io/vizmlir/)**

![VizMLIR GPU view of a transpose kernel](public/demo/vizmlir-workspace.png)

![VizMLIR field guide](public/demo/vizmlir-docs.png)

## Why VizMLIR?

Much of a kernel's GPU performance is settled in the compiler pipeline, long before a profiler sees it: a lowering that leaves neighboring threads writing memory 1024 elements apart, or a shared-memory tile laid out so that every thread of a warp hits the same bank. MLIR shows you the code and a profiler shows you the cost. VizMLIR sits between them: it reads the IR at every pass and draws the GPU shape of the kernel it describes, so you can see the problem, the pass that introduced it, and whether the fix worked.

VizMLIR reads the IR; it does not run it. Launch sizes, buffers, and access patterns come from the code itself. Occupancy, caching, and real timings still need a profiler, and VizMLIR can import its results.

## Who is it for?

- **GPU compiler engineers:** check what each lowering does to a kernel's launch shape, memory placement, and access patterns, pass by pass.
- **Kernel authors on MLIR-based stacks:** see why a kernel is strided or bank-conflicted before reaching for a profiler, and line up profiler timings with the passes that built each kernel.
- **Students and newcomers to GPU compilers:** build an intuition for blocks, warps, coalescing, and shared memory on real compiler output, with the IR next to the picture.

## Features

### On the GPU

- **GPU View:** The canvas opens on the GPU view whenever the IR launches kernels (`g` switches to the op graph). Each `gpu.launch` / `gpu.launch_func` is drawn as its grid of blocks, with one block opened into warps of 32 threads, and the kernel's buffers are grouped as global, shared per block, and private per thread, with sizes and load/store counts. After `gpu-module-to-binary{format=isa}`, the PTX register and shared-memory declarations.
- **Memory Accesses:** Every `memref.load` and `memref.store` is judged for the 32 threads of the first warp: coalesced, strided (with the 32-byte sectors it moves against the minimum), or broadcast in global memory; conflict-free or an N-way bank conflict in shared memory. Pick one to see its lanes, the elements they touch, the usual fix, and its source line.
- **Kernel History and PTX:** Press `h` to see which pass created, changed, lowered (`gpu.func` → `llvm.func`), or serialized each kernel. Click one to step through just that kernel as a diff, down to the PTX it became.
- **Benchmarks:** Import kernel times from Nsight Systems, Nsight Compute, Google Benchmark, or your own CSV, and compare a baseline and a current run, so each kernel's time sits next to the passes that built it ([docs/benchmark-format.md](docs/benchmark-format.md)).

### Through the MLIR pipeline

- **Pass Traces:** Open `mlir-opt -mlir-print-ir-after-all` output (or from any out-of-tree `*-opt` driver) and step through each pass with its before/after diff, failures, and diagnostics. The accepted format is documented in [docs/trace-format.md](docs/trace-format.md).
- **Op Graph and Diff:** Each operation as a node with its SSA data flow, and a structural diff of what a pass added, removed, or changed. Walk changes with `j` `k`; each is linked to its graph node and source line.
- **Pass Timing:** Add `-mlir-timing` to see each pass's wall time and IR size in the timeline, and the full report with `p`. Run under `/usr/bin/time -l` / `-v` to add peak memory.
- **Op Counts:** Press `o` to see how many of each op every pass leaves in the module, to spot a lowering that stopped firing or an op-count blow-up. Exports as CSV.
- **Buffer Memory:** After bufferization, press `b` to see each `memref.alloc` as a live range with its size, the peak live bytes per function, and which buffers a pass added, removed, or now frees differently.

### Workflow

- **Live Reload:** Watch a trace file (Chrome/Edge) and the view refreshes each time `mlir-opt` rewrites it.
- **Save & Export:** The workspace autosaves locally (IndexedDB), named sessions can be saved and shared as `.json`, graphs export as PNG/SVG, and diffs as Markdown/JSON.
- **Keyboard-First:** Jump to any pass, op, or `@symbol` with ⌘K / Ctrl K, step passes with `[` `]`, and press `?` for the full list.
- **Browser-Native:** Runs entirely on the client; your IR never leaves the browser.

## Quick Start

To run VizMLIR locally:

```bash
git clone https://github.com/joepothiboot/vizmlir
cd vizmlir
npm install
npm run dev
```

1. Open your browser to the local dev address. The transpose sample opens in the GPU view.
2. Step through its passes with `[` `]`, and pick a memory access to see its lanes.
3. Open your own trace with **Open…**. To follow kernels from their creation, capture it with `-mlir-print-ir-before=gpu-kernel-outlining -mlir-print-ir-after-all -mlir-print-ir-module-scope`.

## Contributing

Feedback and contributions are highly appreciated. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and project conventions. If you encounter a specific MLIR construct that doesn't render as expected, please open an issue with the snippet attached so we can improve the parser.

## License

Distributed under the MIT License. See `LICENSE` for more information.
