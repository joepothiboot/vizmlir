#!/usr/bin/env node
// Builds the hardware-check kernels to PTX for sm_75 (a T4; newer GPUs JIT it)
// with bare-pointer kernel parameters, so a launcher passes one pointer per
// buffer. Writes samples/ptx/<kernel>.ptx and samples/ptx/kernels.json (grid,
// block and buffer shapes per kernel, for scripts/run_kernels.py). Needs an
// mlir-opt with the NVPTX target:
//   MLIR_OPT=/path/to/mlir-opt node scripts/build-ptx.mjs
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { analyzeGpu } from "../src/gpu/model.js";
import { parseMemref } from "../src/trace/buffers.js";
import { INPUTS } from "./predict.mjs";

const ROOT = new URL("../", import.meta.url);
const MLIR_OPT = process.env.MLIR_OPT ?? "mlir-opt";
// Sinking constants into the launch first keeps them out of the kernel's
// parameters.
const PIPELINE =
  "builtin.module(gpu-launch-sink-index-computations,gpu-kernel-outlining,canonicalize,nvvm-attach-target{chip=sm_75}," +
  "gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm{use-bare-ptr-memref-call-conv=1})," +
  "gpu-module-to-binary{format=isa})";

mkdirSync(new URL("samples/ptx/", ROOT), { recursive: true });
const manifest = [];
for (const input of INPUTS) {
  const path = new URL(input, ROOT).pathname;
  const launches = new Map(analyzeGpu(readFileSync(path, "utf8")).launches.map((l) => [`${l.host.slice(1)}_kernel`, l]));
  const ir = execFileSync(MLIR_OPT, [path, `-pass-pipeline=${PIPELINE}`], { encoding: "utf8" });
  for (const [, name, escaped] of ir.matchAll(/gpu\.binary @(\w+)\s.*?assembly = "((?:[^"\\]|\\.)*)"/g)) {
    // MLIR escapes non-printable characters, `"` and `\` as \XX hex.
    const ptx = escaped.replace(/\\([0-9A-Fa-f]{2})/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
    writeFileSync(new URL(`samples/ptx/${name}.ptx`, ROOT), ptx);
    const args = new RegExp(`gpu\\.launch_func\\s+@${name}::@${name}\\b.*args\\((.*)\\)`).exec(ir)[1];
    const { grid, block } = launches.get(name);
    manifest.push({ name, grid, block, args: [...args.matchAll(/memref<[^>]*>/g)].map(([type]) => parseMemref(type).dims) });
  }
}
writeFileSync(new URL("samples/ptx/kernels.json", ROOT), `${JSON.stringify(manifest, null, 1)}\n`);
console.log(`samples/ptx: ${manifest.length} kernels`);
