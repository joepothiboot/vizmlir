#!/usr/bin/env bash
# Regenerates the pass-trace fixtures from input/*.mlir.
# Usage: MLIR_OPT=/path/to/mlir-opt tests/fixtures/traces/generate.sh
set -euo pipefail
cd "$(dirname "$0")"
MLIR_OPT="${MLIR_OPT:-mlir-opt}"
COMMON=(-mlir-disable-threading)

run() {
  local out="$1"; shift
  # Diagnostics go to stderr and IR dumps to stdout; the app reads both merged.
  "$MLIR_OPT" "$@" "${COMMON[@]}" > "$out" 2>&1 || true
}

run nested-after-all.txt input/two-funcs.mlir \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  -mlir-print-ir-after-all
run module-scope-after-all.txt input/two-funcs.mlir \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  -mlir-print-ir-after-all -mlir-print-ir-module-scope
run nested-before-all.txt input/two-funcs.mlir \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  -mlir-print-ir-before-all
run mixed-nesting.txt input/two-funcs.mlir \
  -pass-pipeline='builtin.module(symbol-dce,func.func(cse),symbol-dce)' \
  -mlir-print-ir-after-all
run failed-pass.txt input/tile-non-tileable.mlir \
  -pass-pipeline='builtin.module(cse,transform-interpreter)' \
  -mlir-print-ir-after-all -mlir-print-ir-module-scope
run failed-only.txt input/tile-non-tileable.mlir \
  -pass-pipeline='builtin.module(cse,transform-interpreter)' \
  -mlir-print-ir-after-failure -mlir-print-ir-module-scope

"$MLIR_OPT" --version | head -1 > MLIR_VERSION
