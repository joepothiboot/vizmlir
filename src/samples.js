// Built-in samples. The inline one loads instantly on first visit; the rest are
// real mlir-opt output in public/samples (see samples/generate.sh) and are
// fetched only when picked.

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
 * `baseline`/`current` name a before/after pair; `trace` names a pass trace.
 * Paths are relative to the samples directory.
 */
export const SAMPLES = [
  {
    id: "rename",
    title: "Op rename",
    blurb: "One op swapped in a matmul. The smallest possible diff.",
    inline: {
      baseline: RENAME_SAMPLE,
      current: RENAME_SAMPLE.replace("linalg.fill", "linalg.fill_relu"),
    },
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
    id: "tiling",
    title: "Tile a matmul",
    blurb:
      "Transform dialect tiles the matmul 32×32×64 into nested scf.for loops.",
    baseline: "tiling.before.mlir",
    current: "tiling.after.mlir",
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
];

/** Every file a sample needs, relative to the samples directory. */
export function sampleFiles(sample) {
  return [sample.baseline, sample.current, sample.trace].filter(Boolean);
}

/**
 * Resolves a sample to a workspace state (the shape `applyState` takes).
 * @param {(path: string) => Promise<string>} fetchText
 */
export async function loadSampleState(sample, fetchText) {
  const sourceName = `sample: ${sample.title}`;
  if (sample.trace) {
    return { sourceName, trace: await fetchText(sample.trace), traceIndex: -1 };
  }
  const [baseline, current] = sample.inline
    ? [sample.inline.baseline, sample.inline.current]
    : await Promise.all([
        fetchText(sample.baseline),
        fetchText(sample.current),
      ]);
  return { sourceName, baseline, current, tab: "current" };
}
