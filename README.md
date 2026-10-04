# VizMLIR 🔍

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Learn how a compiler turns your code into GPU work, by watching it happen.**

VizMLIR is a visual companion for computer architecture, compilers, and parallel programming courses. It takes real output from MLIR, the compiler framework behind many modern GPU and machine learning compilers, and draws what it means: how the work is split among thousands of threads, where the data lives, and whether it's read in a way the GPU is good at. You can step through the compiler's work one pass at a time and watch the picture change. It runs in your browser, and your code never leaves it.

🚀 **[Try VizMLIR live](https://joepothiboot.github.io/vizmlir/)**: it opens on a small example with two classic GPU mistakes in it. See if you can spot them.

![VizMLIR showing a transpose kernel: the code, the GPU view, and the inspector, whose answer card proves the write is strided for all 32,768 warps and follows it across compiler passes, with "What's this line?" explaining the store in plain words](public/demo/vizmlir-workspace.png)

_A transpose whose write is strided. The GPU view shows the 32 threads of a warp landing 1024 items apart; beside it, the inspector's Line tab gives the verdict and takes the store apart in plain words._

## 💡 Why VizMLIR?

A GPU is fast when thousands of its threads work side by side and read memory in tidy, neighboring chunks. Whether that happens is often decided inside the compiler, long before you run anything, and the only evidence is pages of intermediate code that are hard to read.

VizMLIR turns that code into pictures. It shows your kernel as teams of threads, sorts its data into the GPU's kinds of memory, and checks every read and write: "these 32 threads read neighbors, good" or "these 32 threads each go to a different place, and the GPU fetches eight times more than it uses." Then it shows you which compiler step made it that way.

One honest limit: VizMLIR only reads your code; it never runs it. It can spot the patterns that make a GPU slow, but it can't give you exact timings. A profiler does that, and you can bring its numbers into VizMLIR.

## 🎯 Who is it for?

- 📚 **CS students and teachers.** See threads, warps, coalescing, shared memory, SSA and lowering on real compiler output, with the code right next to the picture. If you know loops and arrays, you can follow along.
- 🛠️ **Engineers building GPU compilers on MLIR.** Check what each step does to a kernel's shape, memory, and access patterns, one step at a time.
- ⚡ **Kernel authors.** Find out why a kernel is slow before reaching for a profiler, and line up profiler timings with the steps that built each kernel.

## 🧰 What you can do

### 🖥️ See the GPU side

- **The GPU view.** Whenever your code starts work on the GPU, the wide pane next to the code shows it as a grid of blocks, with one block opened up into its warps of 32 threads, and lists every array the kernel uses as global, shared, or private memory. Its **Graph** tab (or `g`) switches to a diagram of the code instead.
- **Memory checks.** Every read and write gets a verdict: are a warp's threads reading neighbors (_coalesced_), far-apart places (_strided_), or the same item (_broadcast_)? In shared memory, are they queueing at the same bank (_bank conflict_)? Click an access and the **inspector** slides in beside the picture with its answer card: what it costs ("the GPU moves 8× more data than it uses") and how people usually fix it, while a one-line verdict stays above the picture. Click another access to see its 32 threads and, with **Elements** (`e`) in the 3D view, the items they touch.
- **Proven, not guessed.** When an address is a linear function of the thread ids, block ids and loop counters, VizMLIR checks the verdict for every warp of the launch and every loop iteration, and says so ("✓ all 32,768 warps"). Switch the card to **Compiler** to see the math behind it, and follow the same access across every compiler step, down to the LLVM dialect.
- **Triton, too.** Open Triton GPU IR and see which thread holds which element of a tensor under its `#blocked` layout, with the same proven verdicts for every `tt.load` and `tt.store`. The Triton sample shows the coalesce pass turning a strided load into 16-byte vectors.
- **What's this line?** Click any line of code to see it explained in plain words: what it produces, what it does, which values it uses (each linked back to the line that made it), and what kind of data it works on. For reads and writes, it also says how the GPU handles them, with a real example like "thread (1, 0, 0) uses `%arg1[1, 0]`". It is the inspector's **Line** tab (`w`).
- **A kernel's full history.** Press `h` to see which compiler step created, changed, or finished each kernel, then follow one kernel step by step, all the way to the GPU assembly (PTX) at the end.
- **Your measured timings.** Bring in results from Nsight Systems, Nsight Compute, Google Benchmark, or your own CSV, and compare two runs to see which kernels got faster or slower ([docs/benchmark-format.md](docs/benchmark-format.md)).

### 🧱 Beyond GPUs

- **Scratchpads and DMA.** DSP and NPU accelerators often have a small local memory that the compiler fills by DMA instead of a cache. When the IR has buffers in such a memory (nano-dsp's `#dsp.local`) or `memref.dma_start` / `memref.dma_wait`, a **Local memory** tab shows the budget as a bar of buffer slots and the loop's first trips as DMA, wait and compute lanes: which tile is loaded before the loop, which is prefetched for the next trip, and where the loop waits. Click any block to see its line explained. It is the order the IR expresses, not measured timing.
- The "DSP scratchpad" sample is real output of `nanodsp-opt` from [nano-dsp-mlir](https://github.com/joepothiboot/nano-dsp-mlir): a matmul whose tiles are double-buffered in a Hexagon-like scratchpad, then lowered to plain copies.

### 🧭 Follow the compiler

- **Step through a pipeline.** Open the log `mlir-opt -mlir-print-ir-after-all` writes (or one from any `*-opt` tool built on MLIR) and move through each step with the pass slider or `[` `]`; hovering the slider previews a step (its name, size, and how many operations it changed) before you jump. Failed steps are marked, and compiler messages appear next to the code. The format is described in [docs/trace-format.md](docs/trace-format.md).
- **See what changed.** Each operation is drawn as a box with arrows for where its values go, and the inspector's **Changes** tab lists what the current step added, removed, or changed. Walk through the list with `j` `k`.
- **How long each step took**, with `-mlir-timing` (press `p` for the inspector's Timing tab).
- **How many of each operation** every step leaves behind (press `o`), to spot a step that stopped working or one that made the code blow up.
- **Memory over time** after arrays are allocated (press `b`): when each one is alive, how big it is, and the peak.

### ⚙️ Work comfortably

- **Details beside the picture.** What changed, the explained line, timing, buffers, op counts, and symbol history are tabs of one inspector next to the picture, not pop-ups, so what you compare against stays in view. `\` shows or hides it, and `esc` closes it.
- **Live reload.** Watch a file (Chrome/Edge), and VizMLIR refreshes every time you rerun your compiler.
- **Save and share.** Your work saves itself in the browser. Keep named sessions and share them as `.json`, save diagrams as PNG or SVG, and export changes as Markdown or JSON.
- **Keyboard friendly.** Jump to any step, operation, or function with ⌘K / Ctrl K. Press `?` for every shortcut.
- 🔒 **Private by design.** Everything runs in your browser. Nothing is uploaded.

## 🏁 Quick start

To run VizMLIR on your own machine:

```bash
git clone https://github.com/joepothiboot/vizmlir
cd vizmlir
npm install
npm run dev
```

1. Open the address it prints. The transpose example opens on the GPU view.
2. Scroll to **Memory accesses** and click the orange row; the inspector opens beside the picture with what went wrong and how it's usually fixed.
3. Press `]`, or drag the pass slider, to step through the compiler's work and watch the picture change.
4. When you're ready, open your own `mlir-opt` log with **Open…**. Use `-mlir-print-ir-after-all` (and `-mlir-timing` for the timing view) so the log has every pass.

## 🤝 Contributing

Feedback and contributions are very welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up and work on the project. If some MLIR code doesn't show up the way you expect, please open an issue with the snippet, and we'll look at it.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
