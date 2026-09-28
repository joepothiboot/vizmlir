import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parseMemref } from "../../src/buffers.js";
import {
  buildDefs,
  evaluate,
  findAccesses,
  warpAccess,
  warpLanes,
} from "../../src/gpu-access.js";
import { analyzeGpu, memorySpace } from "../../src/gpu.js";
import { parsePassTrace } from "../../src/trace.js";

const events = (name) =>
  parsePassTrace(
    readFileSync(new URL(`../../public/samples/${name}`, import.meta.url), "utf8"),
  ).events;

// Every access of every launch in `ir`, as [kind, buffer, verdict, detail].
function verdicts(ir) {
  const model = analyzeGpu(ir);
  return model.launches.flatMap((launch) => {
    const kernel = model.kernels[launch.kernel];
    return kernel.accesses.map((access) => {
      const memref = parseMemref(access.type);
      const result = warpAccess(access, memref, memorySpace(memref.space), {
        defs: kernel.defs,
        args: kernel.args,
        block: launch.block,
        grid: launch.grid,
      });
      return [
        access.kind,
        access.buffer,
        result.analyzed ? result.verdict : "not analyzed",
        result.analyzed
          ? result.sectors !== undefined
            ? `${result.sectors}/${result.needed} sectors`
            : `${result.ways}-way`
          : result.reason,
      ];
    });
  });
}

// A function with one inline launch of a single 32-thread block around `body`.
const kernel = (body) =>
  [
    "func.func @f(%x: memref<64x64xf32>, %s: memref<64xf32, #gpu.address_space<workgroup>>, %n: index) {",
    "  %c0 = arith.constant 0 : index",
    "  %c1 = arith.constant 1 : index",
    "  %c2 = arith.constant 2 : index",
    "  %c32 = arith.constant 32 : index",
    "  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c1, %sz = %c1) {",
    ...body.map((line) => `    ${line}`),
    "    gpu.terminator",
    "  }",
    "  return",
    "}",
  ].join("\n");

describe("evaluate", () => {
  const defs = buildDefs([
    "%a = gpu.thread_id x",
    "%b = gpu.block_id x",
    "%d = gpu.block_dim x",
    "%c4 = arith.constant 4 : index",
    "%m = arith.muli %b, %d : index",
    "%i = arith.addi %m, %a : index",
    "%r = arith.remui %i, %c4 : index",
    "%v = memref.load %buf[%i] : memref<8xindex>",
    "%w = arith.addi %v, %a : index",
  ]);
  const env = { tx: 3, bx: 2, bdx: 32 };

  it("interprets thread ids, block sizes and integer arithmetic", () => {
    expect(evaluate("%i", defs, env)).toBe(67);
    expect(evaluate("%r", defs, env)).toBe(3);
  });

  it("stops at a value it cannot compute and names it", () => {
    const trace = {};
    expect(evaluate("%w", defs, env, trace)).toBe(null);
    expect(trace.stuck).toBe("%v");
  });

  it("uses kernel argument values", () => {
    const trace = {};
    expect(evaluate("%p", defs, { args: new Map([["%p", 16]]) }, trace)).toBe(16);
    expect(evaluate("%q", defs, { args: new Map([["%q", null]]) }, trace)).toBe(null);
    expect(trace.stuck).toBe("%q");
  });
});

describe("findAccesses", () => {
  it("finds loads and stores and whether they sit in a loop", () => {
    const accesses = findAccesses(
      [
        "%v = memref.load %a[%i, %j] : memref<4x4xf32>",
        "scf.for %k = %c0 to %c4 step %c1 {",
        "  memref.store %v, %b[%k] : memref<4xf32, 3>",
        "}",
      ],
      10,
    );
    expect(accesses).toEqual([
      { line: 10, kind: "load", value: null, buffer: "%a", indices: ["%i", "%j"], type: "memref<4x4xf32>", inLoop: false },
      { line: 12, kind: "store", value: "%v", buffer: "%b", indices: ["%k"], type: "memref<4xf32, 3>", inLoop: true },
    ]);
  });
});

describe("warpLanes", () => {
  it("fills x first, then y", () => {
    const lanes = warpLanes([16, 16, 1]);
    expect(lanes).toHaveLength(32);
    expect(lanes[15]).toEqual({ lane: 15, tx: 15, ty: 0, tz: 0 });
    expect(lanes[16]).toEqual({ lane: 16, tx: 0, ty: 1, tz: 0 });
  });

  it("stops at the block size", () => {
    expect(warpLanes([8, 1, 1])).toHaveLength(8);
  });
});

describe("warpAccess verdicts", () => {
  it("row by thread x is coalesced; column by thread x is strided", () => {
    const ir = kernel([
      "%v = memref.load %x[%c0, %tx] : memref<64x64xf32>",
      "memref.store %v, %x[%tx, %c0] : memref<64x64xf32>",
    ]);
    expect(verdicts(ir)).toEqual([
      ["load", "%x", "coalesced", "4/4 sectors"],
      // 32 floats need 4 sectors; a column of 64-float rows spreads them over 32.
      ["store", "%x", "strided", "32/4 sectors"],
    ]);
  });

  it("every lane on one element is a broadcast", () => {
    const ir = kernel(["%v = memref.load %x[%c0, %c0] : memref<64x64xf32>"]);
    expect(verdicts(ir)[0][2]).toBe("broadcast");
  });

  it("shared memory: consecutive words are conflict-free, stride 2 is a 2-way conflict", () => {
    const ir = kernel([
      "%v = memref.load %s[%tx] : memref<64xf32, #gpu.address_space<workgroup>>",
      "%i = arith.muli %tx, %c2 : index",
      "memref.store %v, %s[%i] : memref<64xf32, #gpu.address_space<workgroup>>",
    ]);
    expect(verdicts(ir)).toEqual([
      ["load", "%s", "conflict-free", "1-way"],
      ["store", "%s", "bank-conflict", "2-way"],
    ]);
  });

  it("says why an index cannot be analyzed", () => {
    const ir = kernel(["%v = memref.load %x[%c0, %n] : memref<64x64xf32>"]);
    expect(verdicts(ir)[0][2]).toBe("not analyzed");
    expect(verdicts(ir)[0][3]).toMatch(/%n is not computed/);
  });
});

describe("the GPU samples", () => {
  it("saxpy's loads and store are coalesced", () => {
    expect(verdicts(events("gpu-kernels.trace.txt")[1].ir)).toEqual([
      ["load", "%arg1", "coalesced", "4/4 sectors"],
      ["load", "%arg2", "coalesced", "4/4 sectors"],
      ["store", "%arg2", "coalesced", "4/4 sectors"],
      ["load", "%arg1", "coalesced", "4/4 sectors"],
      ["store", "%arg1", "coalesced", "4/4 sectors"],
    ]);
  });

  it("the transpose sample shows a strided write, a 32-way conflict, and the padded fix", () => {
    const expected = [
      // naive
      ["load", "coalesced", "4/4 sectors"],
      ["store", "strided", "32/4 sectors"],
      // 32x32 tile
      ["load", "coalesced", "4/4 sectors"],
      ["store", "conflict-free", "1-way"],
      ["load", "bank-conflict", "32-way"],
      ["store", "coalesced", "4/4 sectors"],
      // 32x33 tile
      ["load", "coalesced", "4/4 sectors"],
      ["store", "conflict-free", "1-way"],
      ["load", "conflict-free", "1-way"],
      ["store", "coalesced", "4/4 sectors"],
    ];
    const trace = events("gpu-transpose.trace.txt");
    for (const event of [trace[0], trace[1]])
      expect(verdicts(event.ir).map(([kind, , verdict, detail]) => [kind, verdict, detail])).toEqual(expected);
  });

  it("the tiled matmul is coalesced and conflict-free before and after outlining", () => {
    const expected = [
      ["load", "coalesced", "4/4 sectors"],
      ["load", "coalesced", "4/4 sectors"],
      ["store", "conflict-free", "1-way"],
      ["store", "conflict-free", "1-way"],
      ["load", "conflict-free", "1-way"],
      ["load", "conflict-free", "1-way"],
      ["store", "coalesced", "4/4 sectors"],
    ];
    const trace = events("gpu-tiled-matmul.trace.txt");
    for (const event of [trace[0], trace[1]])
      expect(verdicts(event.ir).map(([kind, , verdict, detail]) => [kind, verdict, detail])).toEqual(expected);
  });
});
