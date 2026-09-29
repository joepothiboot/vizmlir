#!/usr/bin/env bash
# Regenerates the built-in samples in public/samples from input/*.mlir.
# Usage: MLIR_OPT=/path/to/mlir-opt samples/generate.sh
set -euo pipefail
cd "$(dirname "$0")/input"
MLIR_OPT="${MLIR_OPT:-mlir-opt}"
OUT=../../public/samples
COMMON=(-mlir-disable-threading)
mkdir -p "$OUT"

opt() {
  local out="$1"; shift
  # Diagnostics go to stderr and IR dumps to stdout; the app reads both merged.
  "$MLIR_OPT" "$@" "${COMMON[@]}" > "$OUT/$out" 2>&1 || true
}

# Before/after pairs. The "before" side is round-tripped with no passes so both
# sides use the printer's SSA names.
opt canonicalize.before.mlir canonicalize.mlir
opt canonicalize.after.mlir canonicalize.mlir -canonicalize -cse
opt tiling.before.mlir matmul.mlir
opt tiling.after.mlir matmul.mlir \
  -transform-preload-library='transform-library-paths=tile-matmul.transform.mlir' \
  -transform-interpreter -canonicalize

# Pass traces. The lowering trace also carries a -mlir-timing report and, from
# /usr/bin/time (BSD -l or GNU -v), the process's peak memory.
if [[ "$(uname)" == Darwin ]]; then TIME_FLAG=-l; else TIME_FLAG=-v; fi
/usr/bin/time "$TIME_FLAG" "$MLIR_OPT" matmul.mlir "${COMMON[@]}" \
  -pass-pipeline='builtin.module(func.func(linalg-generalize-named-ops,linalg-fuse-elementwise-ops,canonicalize),one-shot-bufferize{bufferize-function-boundaries},func.func(convert-linalg-to-loops,canonicalize,cse))' \
  -mlir-print-ir-after-all -mlir-print-ir-module-scope -mlir-timing \
  > "$OUT/lowering.trace.txt" 2>&1 || true
# Two GPU kernels from outlining to PTX. The dump before outlining lets the
# symbol history credit gpu-kernel-outlining with creating each kernel. Needs
# an MLIR built with the NVPTX target. The benchmark CSVs that go with it
# (gpu-kernels.*.mock.csv) are hand-written mock data, not generated here.
opt gpu-kernels.trace.txt gpu-kernels.mlir \
  -pass-pipeline='builtin.module(gpu-kernel-outlining,canonicalize,nvvm-attach-target{chip=sm_80},gpu.module(convert-gpu-to-nvvm),gpu-module-to-binary{format=isa})' \
  -mlir-print-ir-before=gpu-kernel-outlining -mlir-print-ir-after-all -mlir-print-ir-module-scope
# A tiled matmul that stages tiles in shared (workgroup) memory, for the GPU
# view. scf.for is lowered to cf inside the gpu.module before NVVM.
opt gpu-tiled-matmul.trace.txt gpu-tiled-matmul.mlir \
  -pass-pipeline='builtin.module(gpu-kernel-outlining,canonicalize,nvvm-attach-target{chip=sm_80},gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm),gpu-module-to-binary{format=isa})' \
  -mlir-print-ir-before=gpu-kernel-outlining -mlir-print-ir-after-all -mlir-print-ir-module-scope
# A transpose three ways (naive, 32x32 shared tile, 32x33 padded tile) for the
# GPU view's coalescing and bank-conflict verdicts.
opt gpu-transpose.trace.txt gpu-transpose.mlir \
  -pass-pipeline='builtin.module(gpu-kernel-outlining,canonicalize,nvvm-attach-target{chip=sm_80},gpu.module(convert-gpu-to-nvvm),gpu-module-to-binary{format=isa})' \
  -mlir-print-ir-before=gpu-kernel-outlining -mlir-print-ir-after-all -mlir-print-ir-module-scope
# Six small kernels, one memory pattern each (stride 2, misaligned, a loop
# that drifts, broadcast, 16x16 blocks), followed down to PTX.
opt gpu-patterns.trace.txt gpu-patterns.mlir \
  -pass-pipeline='builtin.module(gpu-kernel-outlining,canonicalize,nvvm-attach-target{chip=sm_80},gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm),gpu-module-to-binary{format=isa})' \
  -mlir-print-ir-before=gpu-kernel-outlining -mlir-print-ir-after-all -mlir-print-ir-module-scope
opt failed-transform.trace.txt matmul.mlir \
  -pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=tile-wrong-op.transform.mlir},canonicalize,cse,transform-interpreter)' \
  -mlir-print-ir-after-all -mlir-print-ir-module-scope

"$MLIR_OPT" --version | grep -m1 -i "llvm version" > "$OUT/MLIR_VERSION"
