#!/usr/bin/env bash
set -euo pipefail
TARGET="wasm32-unknown-unknown"
CRATE="mlir_core"
OUT_DIR="../public"
RAW="target/${TARGET}/release/${CRATE}.wasm"

echo "==> building ${CRATE} (${TARGET})"
rustup target add "${TARGET}" >/dev/null 2>&1 || true
cargo build --release --target "${TARGET}"
mkdir -p "${OUT_DIR}"

if command -v wasm-opt >/dev/null 2>&1; then
  echo "==> optimising with wasm-opt"
  wasm-opt -Oz --enable-bulk-memory --strip-debug --strip-producers \
    "${RAW}" -o "${OUT_DIR}/${CRATE}.wasm"
else
  echo "!! wasm-opt not found (binaryen) — copying unoptimised binary"
  cp "${RAW}" "${OUT_DIR}/${CRATE}.wasm"
fi

echo "==> ${OUT_DIR}/${CRATE}.wasm ($(wc -c < "${OUT_DIR}/${CRATE}.wasm") bytes)"
echo "==> reminder: if abi.rs changed, sync src/wasm/abi.js"
