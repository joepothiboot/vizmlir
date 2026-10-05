#!/usr/bin/env node
// Builds the hardware-check kernels to PTX for sm_75 (a T4; newer GPUs JIT it)
// with bare-pointer kernel parameters, so a launcher passes one pointer per
// buffer. Writes samples/ptx/<kernel>.ptx. Needs an mlir-opt with the NVPTX
// target: MLIR_OPT=/path/to/mlir-opt node scripts/build-ptx.mjs
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { INPUTS } from "./predict.mjs";

const ROOT = new URL("../", import.meta.url);
const MLIR_OPT = process.env.MLIR_OPT ?? "mlir-opt";
const PIPELINE =
  "builtin.module(gpu-launch-sink-index-computations,gpu-kernel-outlining,canonicalize,nvvm-attach-target{chip=sm_75}," +
  "gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm{use-bare-ptr-memref-call-conv=1})," +
  "gpu-module-to-binary{format=isa})";

mkdirSync(new URL("samples/ptx/", ROOT), { recursive: true });
for (const input of INPUTS) {
  const ir = execFileSync(MLIR_OPT, [new URL(input, ROOT).pathname, `-pass-pipeline=${PIPELINE}`], { encoding: "utf8" });
  for (const [, name, escaped] of ir.matchAll(/gpu\.binary @(\w+)\s.*?assembly = "((?:[^"\\]|\\.)*)"/g)) {
    // MLIR escapes non-printable characters, `"` and `\` as \XX hex.
    const ptx = escaped.replace(/\\([0-9A-Fa-f]{2})/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
    writeFileSync(new URL(`samples/ptx/${name}.ptx`, ROOT), ptx);
    console.log(`samples/ptx/${name}.ptx`);
  }
}
