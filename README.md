# VizMLIR

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**See how your code runs on the GPU.**

When you write a program for a GPU, a compiler reshapes it in many small steps before the GPU ever sees it. VizMLIR lets you watch those steps one at a time, and draws what the result will do on the GPU: how the work is split up, where the data is kept, and whether it's read in a way the GPU is good at. It runs in your browser, and your code never leaves it.

**[Try VizMLIR live](https://joepothiboot.github.io/vizmlir/)**: it opens on a small example with two classic GPU mistakes in it.

![VizMLIR showing a transpose kernel whose writes are strided](public/demo/vizmlir-workspace.png)

![VizMLIR field guide](public/demo/vizmlir-docs.png)

## Why VizMLIR?

A GPU is fast when thousands of its threads work side by side and read memory in tidy, neighboring chunks. Whether that happens is often decided inside the compiler, long before you run anything, and the only evidence is pages of intermediate code that are hard to read.

VizMLIR turns that code into pictures. It shows your kernel as teams of threads, sorts its data into the GPU's kinds of memory, and checks each read and write: "these 32 threads read neighbors, good" or "these 32 threads each go to a different place, and the GPU fetches eight times more than it uses." Then it shows you which compiler step made it that way.

One honest limit: VizMLIR only reads your code; it never runs it. It can spot the patterns that make a GPU slow, but it can't give you exact timings. A profiler does that, and you can bring its numbers into VizMLIR.

## Who is it for?

- **People learning how GPUs and compilers work.** See threads, warps, coalescing and shared memory on real compiler output, with the code right next to the picture. You don't need to be a compiler expert.
- **Engineers building GPU compilers on MLIR.** Check what each step does to a kernel's shape, memory, and access patterns, one step at a time.
- **Kernel authors.** Find out why a kernel is slow before reaching for a profiler, and line up profiler timings with the steps that built each kernel.

## What you can do

### See the GPU side

- **The GPU view.** Whenever your code starts work on the GPU, the middle panel shows it as a grid of blocks, with one block opened up into its warps of 32 threads, and lists every array the kernel uses as global, shared, or private memory. Press `g` to switch to a diagram of the code instead.
- **Memory checks.** Every read and write is checked for the first warp: are the threads reading neighbors (_coalesced_), far-apart places (_strided_), or the same item (_broadcast_)? In shared memory, are they queueing at the same bank (_bank conflict_)? Click one to see the 32 threads, the items they touch, and how people usually fix it.
- **A kernel's life story.** Press `h` to see which compiler step created, changed, or finished each kernel, then follow one kernel step by step, all the way to the GPU assembly (PTX) at the end.
- **Your measured timings.** Bring in results from Nsight Systems, Nsight Compute, Google Benchmark, or your own CSV, and compare two runs to see which kernels got faster or slower ([docs/benchmark-format.md](docs/benchmark-format.md)).

### Follow the compiler

- **Step through a pipeline.** Open the log `mlir-opt -mlir-print-ir-after-all` writes (or one from any `*-opt` tool built on MLIR) and move through each step with `[` `]`. Failed steps are marked, and compiler messages appear next to the code. The format is described in [docs/trace-format.md](docs/trace-format.md).
- **See what changed.** Each operation is drawn as a box with arrows for where its values go, and a side panel lists what the current step added, removed, or changed. Walk through the list with `j` `k`.
- **How long each step took**, with `-mlir-timing` (press `p`).
- **How many of each operation** every step leaves behind (press `o`), to spot a step that stopped working or one that made the code blow up.
- **Memory over time** after arrays are allocated (press `b`): when each one is alive, how big it is, and the peak.

### Work comfortably

- **Live reload.** Watch a file (Chrome/Edge), and VizMLIR refreshes every time you rerun your compiler.
- **Save and share.** Your work saves itself in the browser. Keep named sessions and share them as `.json`, save diagrams as PNG or SVG, and export changes as Markdown or JSON.
- **Keyboard friendly.** Jump to any step, operation, or function with ⌘K / Ctrl K. Press `?` for every shortcut.
- **Private by design.** Everything runs in your browser. Nothing is uploaded.

## Quick start

To run VizMLIR on your own machine:

```bash
git clone https://github.com/joepothiboot/vizmlir
cd vizmlir
npm install
npm run dev
```

1. Open the address it prints. The transpose example opens on the GPU view.
2. Scroll to **Memory accesses** and click the orange row to see what went wrong and how it's usually fixed.
3. Press `]` to step through the compiler's work and watch the picture change.
4. When you're ready, open your own `mlir-opt` log with **Open…**. The field guide (the **Docs** link in the app) explains which flags to use and what each one does.

## Contributing

Feedback and contributions are very welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up and work on the project. If some MLIR code doesn't show up the way you expect, please open an issue with the snippet, and we'll look at it.

## License

Distributed under the MIT License. See `LICENSE` for more information.
