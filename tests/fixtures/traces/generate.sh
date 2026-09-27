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

# -mlir-timing prints an execution time report on stderr after the last dump.
run timing-after-all.txt input/two-funcs.mlir \
  -pass-pipeline='builtin.module(symbol-dce,func.func(cse,canonicalize,cse))' \
  -mlir-print-ir-after-all -mlir-timing
run timing-list.txt input/two-funcs.mlir \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  -mlir-timing -mlir-timing-display=list
run timing-json.txt input/two-funcs.mlir \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  -mlir-timing -mlir-output-format=json
run timing-failed.txt input/tile-non-tileable.mlir \
  -pass-pipeline='builtin.module(cse,transform-interpreter)' \
  -mlir-print-ir-after-all -mlir-print-ir-module-scope -mlir-timing
# With threading, reports add a user time column.
"$MLIR_OPT" input/two-funcs.mlir \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  -mlir-timing -mlir-timing-display=list > timing-threaded-list.txt 2>&1 || true
# Peak memory comes from the process, not mlir-opt: BSD time -l or GNU time -v.
if [[ "$(uname)" == Darwin ]]; then TIME_FLAG=-l; else TIME_FLAG=-v; fi
/usr/bin/time "$TIME_FLAG" "$MLIR_OPT" input/two-funcs.mlir "${COMMON[@]}" \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  -mlir-print-ir-after-all -mlir-timing > timed-run.txt 2>&1 || true

# Out-of-tree drivers built on MlirOptMain print the same trace format. This
# one uses schema-opt from json-schema-mlir, a custom dialect mlir-opt can't
# parse; set SCHEMA_OPT to regenerate it.
if [[ -n "${SCHEMA_OPT:-}" ]]; then
  "$SCHEMA_OPT" input/schema-person.mlir "${COMMON[@]}" \
    --schema-to-std-pipeline -mlir-print-ir-after-all \
    > schema-opt-pipeline.txt 2>&1 || true
fi

"$MLIR_OPT" --version | head -1 > MLIR_VERSION
