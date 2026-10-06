#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/input"
MLIR_OPT="${MLIR_OPT:-mlir-opt}"
OUT=../../public/samples
COMMON=(-mlir-disable-threading)
mkdir -p "$OUT"

opt() {
  local out="$1"; shift
  "$MLIR_OPT" "$@" "${COMMON[@]}" > "$OUT/$out" 2>&1 || true
}

opt canonicalize.before.mlir canonicalize.mlir
opt canonicalize.after.mlir canonicalize.mlir -canonicalize -cse
opt tiling.before.mlir matmul.mlir
opt tiling.after.mlir matmul.mlir \
  -transform-preload-library='transform-library-paths=tile-matmul.transform.mlir' \
  -transform-interpreter -canonicalize

if [[ "$(uname)" == Darwin ]]; then TIME_FLAG=-l; else TIME_FLAG=-v; fi
/usr/bin/time "$TIME_FLAG" "$MLIR_OPT" matmul.mlir "${COMMON[@]}" \
  -pass-pipeline='builtin.module(func.func(linalg-generalize-named-ops,linalg-fuse-elementwise-ops,canonicalize),one-shot-bufferize{bufferize-function-boundaries},func.func(convert-linalg-to-loops,canonicalize,cse))' \
  -mlir-print-ir-after-all -mlir-print-ir-module-scope -mlir-timing \
  > "$OUT/lowering.trace.txt" 2>&1 || true
opt gpu-kernels.trace.txt gpu-kernels.mlir \
  -pass-pipeline='builtin.module(gpu-kernel-outlining,canonicalize,nvvm-attach-target{chip=sm_80},gpu.module(convert-gpu-to-nvvm),gpu-module-to-binary{format=isa})' \
  -mlir-print-ir-before=gpu-kernel-outlining -mlir-print-ir-after-all -mlir-print-ir-module-scope
opt gpu-tiled-matmul.trace.txt gpu-tiled-matmul.mlir \
  -pass-pipeline='builtin.module(gpu-kernel-outlining,canonicalize,nvvm-attach-target{chip=sm_80},gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm),gpu-module-to-binary{format=isa})' \
  -mlir-print-ir-before=gpu-kernel-outlining -mlir-print-ir-after-all -mlir-print-ir-module-scope
opt gpu-transpose.trace.txt gpu-transpose.mlir \
  -pass-pipeline='builtin.module(gpu-kernel-outlining,canonicalize,nvvm-attach-target{chip=sm_80},gpu.module(convert-gpu-to-nvvm),gpu-module-to-binary{format=isa})' \
  -mlir-print-ir-before=gpu-kernel-outlining -mlir-print-ir-after-all -mlir-print-ir-module-scope
opt gpu-patterns.trace.txt gpu-patterns.mlir \
  -pass-pipeline='builtin.module(gpu-kernel-outlining,canonicalize,nvvm-attach-target{chip=sm_80},gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm),gpu-module-to-binary{format=isa})' \
  -mlir-print-ir-before=gpu-kernel-outlining -mlir-print-ir-after-all -mlir-print-ir-module-scope
opt failed-transform.trace.txt matmul.mlir \
  -pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=tile-wrong-op.transform.mlir},canonicalize,cse,transform-interpreter)' \
  -mlir-print-ir-after-all -mlir-print-ir-module-scope

"$MLIR_OPT" --version | grep -m1 -i "llvm version" > "$OUT/MLIR_VERSION"

if [[ -n "${NANODSP_OPT:-}" ]]; then
  "$NANODSP_OPT" nanodsp-local-matmul.mlir "${COMMON[@]}" \
    -convert-dsp-to-linalg -nanodsp-optimize=target=hexagon-hvx128 \
    -nanodsp-bufferize -nanodsp-promote-local=target=hexagon-hvx128 \
    -nanodsp-lower-local -mlir-print-ir-after-all \
    > "$OUT/nanodsp-local-matmul.trace.txt" 2>&1 || true
  NANODSP_COMMIT="${NANODSP_COMMIT:-$(git -C "$(dirname "$NANODSP_OPT")" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
  echo "nano-dsp-mlir $NANODSP_COMMIT (nanodsp-opt)" > "$OUT/NANODSP_VERSION"
else
  echo "NANODSP_OPT not set: skipping nanodsp-local-matmul.trace.txt" >&2
fi
