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

# Pass traces.
opt lowering.trace.txt matmul.mlir \
  -pass-pipeline='builtin.module(func.func(linalg-generalize-named-ops,linalg-fuse-elementwise-ops,canonicalize),one-shot-bufferize{bufferize-function-boundaries},func.func(convert-linalg-to-loops,canonicalize,cse))' \
  -mlir-print-ir-after-all -mlir-print-ir-module-scope
opt failed-transform.trace.txt matmul.mlir \
  -pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=tile-wrong-op.transform.mlir},canonicalize,cse,transform-interpreter)' \
  -mlir-print-ir-after-all -mlir-print-ir-module-scope

"$MLIR_OPT" --version | grep -m1 -i "llvm version" > "$OUT/MLIR_VERSION"
