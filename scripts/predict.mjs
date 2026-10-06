#!/usr/bin/env node
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { NCU_COUNTER_METRICS, parseNcuCounters } from "../src/bench.js";
import { expectedCost, warpAccess } from "../src/gpu/access.js";
import { analyzeGpu, memorySpace } from "../src/gpu/model.js";
import { parseMemref } from "../src/trace/buffers.js";

const ROOT = new URL("../", import.meta.url);
export const INPUTS = [
  "samples/input/gpu-patterns.mlir",
  "samples/input/gpu-transpose.mlir",
];

const OUT = "docs/predictions.json";
const TOLERANCE = 0.05;

const KIND = {
  global: { load: "globalLoad", store: "globalStore" },
  shared: { load: "sharedLoad", store: "sharedStore" },
};

export function predict(ir, file) {
  const model = analyzeGpu(ir);

  return model.launches.map((launch) => {
    const kernel = model.kernels[launch.kernel];

    const accesses = kernel.accesses.map((access) => {
      const memref = parseMemref(access.type);
      const space = memorySpace(memref.space);

      const result = warpAccess(access, memref, space, {
        defs: kernel.defs,
        args: kernel.args,
        block: launch.block,
        grid: launch.grid,
      });

      const base = {
        file,
        line: access.line,
        kind: access.kind,
        buffer: access.buffer,
        space,
        inLoop: access.inLoop,
      };

      return result.analyzed
        ? { ...base, ...expectedCost(result, space) }
        : { ...base, status: "not analyzed", reason: result.reason };
    });

    const counters = {};

    for (const [space, kinds] of Object.entries(KIND)) {
      for (const [kind, key] of Object.entries(kinds)) {
        const group = accesses.filter(
          (a) => a.space === space && a.kind === kind,
        );

        if (!group.length) continue;

        const known = group.every((a) => a.mean !== undefined);
        const mixed = new Set(group.map((a) => a.inLoop)).size > 1;

        counters[key] =
          known && !mixed
            ? group.reduce((s, a) => s + a.mean, 0) / group.length
            : null;
      }
    }

    return {
      kernel: `${launch.host.slice(1)}_kernel`,
      grid: launch.grid,
      block: launch.block,
      counters,
      accesses,
    };
  });
}

function agreementMark(agrees) {
  if (agrees === null) return "?";

  return agrees ? "✓" : "✗";
}

export function check(predictions, measured) {
  return predictions.flatMap(({ kernel, counters }) =>
    Object.entries(counters).map(([counter, predicted]) => {
      const value = measured.get(kernel)?.[counter] ?? null;

      const agrees =
        predicted !== null && value !== null
          ? Math.abs(value - predicted) <= TOLERANCE * predicted
          : null;

      return { kernel, counter, predicted, measured: value, agrees };
    }),
  );
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const read = (path) => readFileSync(new URL(path, ROOT), "utf8");
  const at = process.argv.indexOf("--check");

  if (at < 0) {
    const kernels = INPUTS.flatMap((file) => predict(read(file), file));

    const doc = {
      note: "Predicted from the IR by scripts/predict.mjs before any GPU run.",
      metrics: NCU_COUNTER_METRICS,
      tolerance: TOLERANCE,
      kernels,
    };

    writeFileSync(new URL(OUT, ROOT), `${JSON.stringify(doc, null, 2)}\n`);
    console.log(`wrote ${OUT}: ${kernels.length} kernels`);
  } else {
    const rows = check(
      JSON.parse(read(OUT)).kernels,
      parseNcuCounters(readFileSync(process.argv[at + 1], "utf8")),
    );

    const fmt = (x) => (x === null ? "—" : x.toFixed(2));

    for (const r of rows) {
      console.log(
        `${agreementMark(r.agrees)} ${r.kernel.padEnd(26)} ${r.counter.padEnd(12)} predicted ${fmt(r.predicted).padStart(6)}  measured ${fmt(r.measured).padStart(6)}`,
      );
    }

    const judged = rows.filter((r) => r.agrees !== null);

    console.log(
      `${judged.filter((r) => r.agrees).length} / ${judged.length} counters agree within ${TOLERANCE * 100}%`,
    );

    process.exitCode = judged.every((r) => r.agrees) ? 0 : 1;
  }
}
