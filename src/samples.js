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

const T4_COUNTERS = { file: "gpu-patterns.t4.ncu.csv", device: "Tesla T4" };

export const SAMPLES = [
  {
    id: "gpu-transpose",
    scope: "gpu",
    title: "Transpose: strided, bank conflict, fixed",
    blurb:
      "The same transpose three ways: a naive strided write, a shared tile with a 32-way bank conflict, and the tile padded to 32×33.",
    trace: "gpu-transpose.trace.txt",
    measured: T4_COUNTERS,
  },
  {
    id: "mojo-saxpy",
    scope: "debug",
    title: "Source locations: a Mojo-style saxpy through the passes",
    blurb:
      "Every op carries a loc(...) into saxpy.mojo. Watch one source line follow its ops through inlining (a call site), canonicalization (two ops fused into one fma) and lowering to NVVM. Open the Source tab and click a line, or an op. Hand-written in the format of mlir-opt -mlir-print-debuginfo output.",
    trace: "saxpy.trace.txt",
    sources: ["saxpy.mojo"],
    breakpoints: ["gone func.call", "appears math.fma"],
    handwritten: true,
  },
  {
    id: "gpu-patterns",
    scope: "gpu",
    title: "Memory patterns: six ways a warp reads",
    blurb:
      "One small kernel per pattern: {x, y} pairs vs. separate arrays, a read shifted by one, a sliding window, a shared bias, and 16×16 blocks. Each looks different in the 3D view.",
    trace: "gpu-patterns.trace.txt",
    measured: T4_COUNTERS,
  },
  {
    id: "gpu-tiled-matmul",
    scope: "gpu",
    title: "Tiled matmul on the GPU",
    blurb:
      "An 8×8 grid of 16×16-thread blocks staging tiles of A and B in shared memory, from gpu.launch to PTX.",
    trace: "gpu-tiled-matmul.trace.txt",
  },
  {
    id: "triton-coalesce",
    scope: "triton",
    title: "Triton: the coalesce pass fixes a strided load",
    blurb:
      "A Triton kernel reads a 32×32 tile of a column-major matrix. The default layout puts lanes along the wrong axis; tritongpu-coalesce switches to 4-wide vectors down the columns. Hand-written in Triton 3.x TTGIR format.",
    trace: "triton-coalesce.trace.txt",
    handwritten: true,
  },
  {
    id: "gpu-kernels",
    scope: "gpu",
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
    scope: "dsp",
    startPass: 11,
    startsAt: "nanodsp-promote-local",
    title: "DSP scratchpad: double-buffered DMA",
    blurb:
      "A DSP-style scratchpad: tiles copied into local memory by DMA, double-buffered. A matmul for nano-dsp-mlir's Hexagon-like target: one pass gives the A and B tiles two slots each in local memory and fetches the next tile by DMA while the current one is used; the next turns the DMAs into plain copies. Open the Local memory tab.",
    trace: "nanodsp-local-matmul.trace.txt",
  },
  {
    id: "lowering",
    scope: "generic",
    title: "Lowering pipeline trace",
    blurb:
      "Seven passes, from linalg on tensors through bufferization to loops, with pass timing. Step with [ and ].",
    trace: "lowering.trace.txt",
  },
  {
    id: "failed",
    scope: "generic",
    title: "Failed pass trace",
    blurb:
      "A transform script targets the wrong op; the trace stops on the failing pass.",
    trace: "failed-transform.trace.txt",
  },
  {
    id: "tiling",
    scope: "generic",
    title: "Tile a matmul",
    blurb:
      "Transform dialect tiles the matmul 32×32×64 into nested scf.for loops.",
    baseline: "tiling.before.mlir",
    current: "tiling.after.mlir",
  },
  {
    id: "canonicalize",
    scope: "generic",
    title: "Canonicalize + CSE",
    blurb:
      "Identity arithmetic folded, a constant computed, dead code removed.",
    baseline: "canonicalize.before.mlir",
    current: "canonicalize.after.mlir",
  },
  {
    id: "rename",
    scope: "generic",
    title: "Op rename",
    blurb: "One op swapped in a matmul. The smallest possible diff.",
    inline: {
      baseline: RENAME_SAMPLE,
      current: RENAME_SAMPLE.replace("linalg.fill", "linalg.fill_relu"),
    },
  },
];

export const SCOPES = [
  {
    id: "debug",
    title: "Source tracing and debugging",
    short: "Debugging",
    blurb:
      "Follow an op from its source line through the passes, and stop on a condition.",
    view: "graph",
    debug: true,
  },
  {
    id: "gpu",
    title: "GPU memory",
    short: "GPU memory",
    blurb:
      "How a kernel is launched, and whether its reads and writes suit the GPU.",
    view: "gpu",
    debug: false,
  },
  {
    id: "triton",
    title: "Triton",
    short: "Triton",
    blurb: "Tensor layouts and the coalesce pass in Triton GPU IR.",
    view: "gpu",
    debug: false,
  },
  {
    id: "dsp",
    title: "DSP and scratchpad",
    short: "DSP",
    blurb: "Local memory filled by DMA, double-buffered.",
    view: "local",
    debug: false,
  },
  {
    id: "generic",
    title: "Generic MLIR passes",
    short: "Generic",
    blurb: "Pass pipelines, diffs and failures with no GPU in them.",
    view: "graph",
    debug: false,
  },
];

export const DEFAULT_SAMPLE_ID = "mojo-saxpy";

export const scopeOf = (sample) =>
  SCOPES.find((scope) => scope.id === sample.scope) ?? SCOPES.at(-1);

export function groupSamples(samples = SAMPLES, scopes = SCOPES) {
  return scopes
    .map((scope) => ({
      scope,
      samples: samples.filter((sample) => sample.scope === scope.id),
    }))
    .filter((group) => group.samples.length);
}

export function sampleFiles(sample) {
  return [
    sample.baseline,
    sample.current,
    sample.trace,
    ...(sample.sources ?? []),
    sample.benchmarks?.baseline,
    sample.benchmarks?.current,
    sample.measured?.file,
  ].filter(Boolean);
}

export async function loadSampleState(sample, fetchText) {
  const sourceName = `sample: ${sample.title}`;

  if (sample.trace) {
    const state = {
      sourceName,
      trace: await fetchText(sample.trace),
      traceIndex: sample.startPass ?? -1,
    };

    if (sample.breakpoints) {
      state.breakpoints = sample.breakpoints.map((text) => ({
        text,
        on: true,
      }));
    }

    if (sample.sources) {
      state.sources = {};

      for (const path of sample.sources) {
        state.sources[path.slice(path.lastIndexOf("/") + 1)] =
          await fetchText(path);
      }
    }

    if (sample.measured) {
      state.measured = {
        device: sample.measured.device,
        text: await fetchText(sample.measured.file),
      };
    }

    if (sample.benchmarks) {
      state.benchmarks = {};

      for (const [slot, path] of Object.entries(sample.benchmarks)) {
        state.benchmarks[slot] = {
          name: path,
          text: await fetchText(path),
          mock: !!sample.mock,
        };
      }
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
