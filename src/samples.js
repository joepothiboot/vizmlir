// Built-in samples. The inline one needs no fetch, so it is the fallback when
// the first sample cannot be loaded; the rest are real mlir-opt output in
// public/samples (see samples/generate.sh), fetched only when picked, or the
// output of an out-of-tree driver built on MLIR (nanodsp-local-matmul, from
// nanodsp-opt; see public/samples/NANODSP_VERSION). The
// benchmark CSVs are the exception: hand-written mock timings (`*.mock.csv`,
// flagged `mock`) to show the benchmark view. So is the Triton trace
// (triton-coalesce.trace.txt, flagged `handwritten`): Triton does not run on
// every machine, so it is written by hand in the format of a Triton 3.x
// MLIR_ENABLE_DUMP=1 trace; replace it with a real one when one is at hand.

export const RENAME_SAMPLE = `module {
  func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>) -> tensor<128x64xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128x64xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
    %out = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>)
                         outs(%filled : tensor<128x64xf32>) -> tensor<128x64xf32>
    func.return %out : tensor<128x64xf32>
  }
}`;

/**
 * In the order the Samples dialog lists them; the first opens on a first
 * visit. `baseline`/`current` name a before/after pair; `trace` names a pass
 * trace, and `benchmarks` optional baseline/current kernel results for it,
 * which are invented when `mock` is set. Paths are relative to the samples
 * directory.
 */
export const SAMPLES = [
  {
    id: "gpu-transpose",
    title: "Transpose: strided, bank conflict, fixed",
    blurb:
      "The same transpose three ways: a naive strided write, a shared tile with a 32-way bank conflict, and the tile padded to 32×33.",
    trace: "gpu-transpose.trace.txt",
  },
  {
    id: "gpu-patterns",
    title: "Memory patterns: six ways a warp reads",
    blurb:
      "One small kernel per pattern: {x, y} pairs vs. separate arrays, a read shifted by one, a sliding window, a shared bias, and 16×16 blocks. Each looks different in the 3D view.",
    trace: "gpu-patterns.trace.txt",
  },
  {
    id: "gpu-tiled-matmul",
    title: "Tiled matmul on the GPU",
    blurb:
      "An 8×8 grid of 16×16-thread blocks staging tiles of A and B in shared memory, from gpu.launch to PTX.",
    trace: "gpu-tiled-matmul.trace.txt",
  },
  {
    id: "triton-coalesce",
    title: "Triton: the coalesce pass fixes a strided load",
    blurb:
      "A Triton kernel reads a 32×32 tile of a column-major matrix. The default layout puts lanes along the wrong axis; tritongpu-coalesce switches to 4-wide vectors down the columns. Hand-written in Triton 3.x TTGIR format.",
    trace: "triton-coalesce.trace.txt",
    handwritten: true,
  },
  {
    id: "gpu-kernels",
    title: "GPU kernels + benchmarks (mock)",
    blurb:
      "Two kernels outlined, lowered to NVVM and serialized to PTX, with mock baseline and current timings: saxpy_kernel looks 26% slower. The timings are invented to show the feature.",
    trace: "gpu-kernels.trace.txt",
    benchmarks: {
      baseline: "gpu-kernels.baseline.mock.csv",
      current: "gpu-kernels.current.mock.csv",
    },
    mock: true,
  },
  {
    id: "nanodsp-local-matmul",
    title: "DSP scratchpad: double-buffered DMA",
    blurb:
      "A DSP-style scratchpad: tiles copied into local memory by DMA, double-buffered. A matmul for nano-dsp-mlir's Hexagon-like target: one pass gives the A and B tiles two slots each in local memory and fetches the next tile by DMA while the current one is used; the next turns the DMAs into plain copies. Open the Local memory tab.",
    trace: "nanodsp-local-matmul.trace.txt",
  },
  {
    id: "lowering",
    title: "Lowering pipeline trace",
    blurb:
      "Seven passes, from linalg on tensors through bufferization to loops, with pass timing. Step with [ and ].",
    trace: "lowering.trace.txt",
  },
  {
    id: "failed",
    title: "Failed pass trace",
    blurb:
      "A transform script targets the wrong op; the trace stops on the failing pass.",
    trace: "failed-transform.trace.txt",
  },
  {
    id: "tiling",
    title: "Tile a matmul",
    blurb:
      "Transform dialect tiles the matmul 32×32×64 into nested scf.for loops.",
    baseline: "tiling.before.mlir",
    current: "tiling.after.mlir",
  },
  {
    id: "canonicalize",
    title: "Canonicalize + CSE",
    blurb:
      "Identity arithmetic folded, a constant computed, dead code removed.",
    baseline: "canonicalize.before.mlir",
    current: "canonicalize.after.mlir",
  },
  {
    id: "rename",
    title: "Op rename",
    blurb: "One op swapped in a matmul. The smallest possible diff.",
    inline: {
      baseline: RENAME_SAMPLE,
      current: RENAME_SAMPLE.replace("linalg.fill", "linalg.fill_relu"),
    },
  },
];

/** Every file a sample needs, relative to the samples directory. */
export function sampleFiles(sample) {
  return [
    sample.baseline,
    sample.current,
    sample.trace,
    sample.benchmarks?.baseline,
    sample.benchmarks?.current,
  ].filter(Boolean);
}

/**
 * Resolves a sample to a workspace state (the shape `applyState` takes).
 * @param {(path: string) => Promise<string>} fetchText
 */
export async function loadSampleState(sample, fetchText) {
  const sourceName = `sample: ${sample.title}`;
  if (sample.trace) {
    const state = {
      sourceName,
      trace: await fetchText(sample.trace),
      traceIndex: -1,
    };
    if (sample.benchmarks) {
      state.benchmarks = {};
      for (const [slot, path] of Object.entries(sample.benchmarks))
        state.benchmarks[slot] = {
          name: path,
          text: await fetchText(path),
          mock: !!sample.mock,
        };
    }
    return state;
  }
  const [baseline, current] = sample.inline
    ? [sample.inline.baseline, sample.inline.current]
    : await Promise.all([
        fetchText(sample.baseline),
        fetchText(sample.current),
      ]);
  return { sourceName, baseline, current, tab: "current" };
}
