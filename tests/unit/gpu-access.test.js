import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parseMemref } from "../../src/trace/buffers.js";
import {
  buildDefs,
  evaluate,
  findAccesses,
  warpAccess,
  warpLanes,
} from "../../src/gpu/access.js";
import { analyzeGpu, loweredArgs, memorySpace } from "../../src/gpu/model.js";
import { parsePassTrace } from "../../src/trace/trace.js";

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

describe("proofs", () => {
  const proof = (body) => {
    const model = analyzeGpu(kernel(body));
    const [launch] = model.launches;
    const k = model.kernels[launch.kernel];
    return k.accesses.map((access) => {
      const memref = parseMemref(access.type);
      return warpAccess(access, memref, memorySpace(memref.space), {
        defs: k.defs,
        args: k.args,
        block: launch.block,
        grid: launch.grid,
      }).proof;
    });
  };

  it("proves a verdict for every warp of the launch", () => {
    expect(
      proof([
        "%v = memref.load %x[%c0, %tx] : memref<64x64xf32>",
        "memref.store %v, %x[%tx, %c0] : memref<64x64xf32>",
      ]),
    ).toEqual([
      {
        status: "proven",
        verdict: "coalesced",
        sectors: 4,
        warps: 1,
        iterations: false,
        formula: "tx",
        laneStride: 1,
      },
      {
        status: "proven",
        verdict: "strided",
        sectors: 32,
        warps: 1,
        iterations: false,
        formula: "64·tx",
        laneStride: 64,
      },
    ]);
  });

  it("covers every loop iteration, and finds a loop that drifts out of alignment", () => {
    const [aligned, drifting] = proof([
      "scf.for %i = %c0 to %c32 step %c32 {",
      "  %a = arith.addi %i, %tx : index",
      "  %v = memref.load %x[%c0, %a] : memref<64x64xf32>",
      "}",
      "scf.for %j = %c0 to %c32 step %c1 {",
      "  %b = arith.addi %j, %tx : index",
      "  %w = memref.load %x[%c0, %b] : memref<64x64xf32>",
      "}",
    ]);
    expect(aligned).toMatchObject({ status: "proven", verdict: "coalesced", iterations: true });
    // Warp 0 on the first iteration is aligned; 7 of every 8 iterations straddle
    // one more sector.
    expect(drifting).toMatchObject({
      status: "varies",
      outcomes: [
        { verdict: "misaligned", sectors: 5, cases: 7 },
        { verdict: "coalesced", sectors: 4, cases: 1 },
      ],
      cases: 8,
      formula: "%j + tx",
    });
  });

  it("falls back to the warp shown when an index is not affine", () => {
    const [shown] = proof([
      "%r = arith.remui %tx, %c2 : index",
      "%v = memref.load %x[%c0, %r] : memref<64x64xf32>",
    ]);
    expect(shown).toEqual({
      status: "sampled",
      reason: "%r is not affine in the thread ids, block ids and loop counters",
    });
  });
});

// Every access of the first launch in `ir`, with its warpAccess result.
function judged(ir) {
  const model = analyzeGpu(ir);
  const [launch] = model.launches;
  const k = model.kernels[launch.kernel];
  return k.accesses.map((access) => {
    const memref = parseMemref(access.type);
    const result = warpAccess(access, memref, memorySpace(memref.space), {
      defs: k.defs,
      args: k.args,
      block: launch.block,
      grid: launch.grid,
    });
    return { access, result };
  });
}

describe("affine ops", () => {
  const ir = (body) => ["#row = affine_map<(d0)[s0] -> (d0 * 64 + s0)>", kernel(body)].join("\n");

  it("reads affine.apply through a #map alias and inline", () => {
    const [aliased, inline] = judged(
      ir([
        "%i = affine.apply #row(%c0)[%tx]",
        "%v = memref.load %x[%c0, %i] : memref<64x64xf32>",
        "%j = affine.apply affine_map<(d0) -> (d0 * 64)>(%tx)",
        "memref.store %v, %x[%c0, %j] : memref<64x64xf32>",
      ]),
    );
    expect(aliased.result).toMatchObject({
      verdict: "coalesced",
      proof: { status: "proven", formula: "tx" },
    });
    expect(inline.result).toMatchObject({
      verdict: "strided",
      proof: { status: "proven", laneStride: 64 },
    });
  });

  it("reads affine.load index expressions and affine.for loops", () => {
    const [divided] = judged(
      ir([
        "affine.for %k = 0 to 64 step 8 {",
        "  %v = affine.load %x[%tx floordiv 32, %k + %tx] : memref<64x64xf32>",
        "}",
      ]),
    );
    expect(divided.access.indices).toEqual(["%tx floordiv 32", "%k + %tx"]);
    // floordiv of a thread id is not affine: warp 0 is judged, not proven.
    expect(divided.result).toMatchObject({
      analyzed: true,
      verdict: "coalesced",
      proof: { status: "sampled" },
    });
    const [aligned] = judged(
      ir([
        "affine.for %k = 0 to 64 step 8 {",
        "  %v = affine.load %x[%c1, %k + %tx] : memref<64x64xf32>",
        "}",
      ]),
    );
    expect(aligned.result.proof).toMatchObject({
      status: "proven",
      verdict: "coalesced",
      iterations: true,
    });
  });
});

describe("lowered kernels", () => {
  // A kernel after convert-gpu-to-nvvm: the memref argument is split into
  // pointers, offset, sizes and strides, and the block size is an argument.
  const struct = "!llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>";
  const lowered = [
    "module attributes {gpu.container_module} {",
    "  func.func @main(%a: memref<64x64xf32>) {",
    "    %c1 = arith.constant 1 : index",
    "    %c32 = arith.constant 32 : index",
    "    gpu.launch_func @k::@k blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%c32 : index, %a : memref<64x64xf32>)",
    "    return",
    "  }",
    "  gpu.module @k {",
    "    llvm.func @k(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) attributes {gpu.kernel, nvvm.kernel} {",
    `      %0 = llvm.mlir.poison : ${struct}`,
    `      %1 = llvm.insertvalue %arg1, %0[0] : ${struct}`,
    `      %2 = llvm.insertvalue %arg2, %1[1] : ${struct}`,
    "      %3 = nvvm.read.ptx.sreg.tid.x : i32",
    "      %4 = llvm.sext %3 : i32 to i64",
    "      %5 = llvm.mul %4, %arg6 : i64",
    `      %6 = llvm.extractvalue %2[1] : ${struct}`,
    "      %7 = llvm.getelementptr %6[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
    "      %8 = llvm.load %7 : !llvm.ptr -> f32",
    "      %9 = llvm.getelementptr %6[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
    "      llvm.store %8, %9 : f32, !llvm.ptr",
    "      llvm.return",
    "    }",
    "  }",
    "}",
  ].join("\n");

  it("expands memref launch arguments into pointers, offset, sizes and strides", () => {
    expect(loweredArgs(["%c32 : index", "%a : memref<64x64xf32>"], 8, () => 32)).toEqual([
      32,
      null,
      null,
      0,
      64,
      64,
      64,
      1,
    ]);
  });

  it("reads loads and stores through getelementptr, back to the argument", () => {
    const [load, store] = judged(lowered);
    expect(load.access).toMatchObject({
      kind: "load",
      buffer: "%arg2",
      indices: ["%4"],
      type: "memref<?xf32>",
    });
    expect(load.result).toMatchObject({
      verdict: "coalesced",
      proof: { status: "proven", formula: "tx" },
    });
    // The row stride (%arg6) comes from the expanded launch arguments.
    expect(store.result).toMatchObject({
      verdict: "strided",
      proof: { status: "proven", formula: "64·tx" },
    });
  });

  it("recognizes a loop lowered to branches as an induction variable", () => {
    const [load] = judged(
      kernel([
        "cf.br ^bb1(%c0 : index)",
        "^bb1(%i: index):",
        "  %more = arith.cmpi slt, %i, %c32 : index",
        "  cf.cond_br %more, ^bb2, ^bb3",
        "^bb2:",
        "  %a = arith.addi %i, %tx : index",
        "  %v = memref.load %x[%c0, %a] : memref<64x64xf32>",
        "  %next = arith.addi %i, %c1 : index",
        "  cf.br ^bb1(%next : index)",
        "^bb3:",
      ]),
    );
    expect(load.result.proof).toMatchObject({
      status: "varies",
      formula: "%i + tx",
      outcomes: [
        { verdict: "misaligned", sectors: 5, cases: 7 },
        { verdict: "coalesced", sectors: 4, cases: 1 },
      ],
    });
  });

  it("leaves a block argument that is not a counting loop undefined", () => {
    const [load] = judged(
      kernel([
        "cf.br ^bb1(%c1 : index)",
        "^bb1(%i: index):",
        "  %more = arith.cmpi slt, %i, %c32 : index",
        "  %v = memref.load %x[%c0, %i] : memref<64x64xf32>",
        "  %next = arith.muli %i, %c2 : index",
        "  cf.br ^bb1(%next : index)",
      ]),
    );
    expect(load.result).toMatchObject({ analyzed: false });
  });
});

describe("the GPU samples", () => {
  it("the memory patterns sample shows one pattern per kernel, from the first pass to LLVM", () => {
    const expected = [
      // aos_x: x of {x, y} pairs, every other float
      ["load", "strided", "8/4 sectors", "proven"],
      ["store", "coalesced", "4/4 sectors", "proven"],
      // soa_x
      ["load", "coalesced", "4/4 sectors", "proven"],
      ["store", "coalesced", "4/4 sectors", "proven"],
      // diff: in[i] and the shifted in[i + 1]
      ["load", "coalesced", "4/4 sectors", "proven"],
      ["load", "misaligned", "5/4 sectors", "proven"],
      ["store", "coalesced", "4/4 sectors", "proven"],
      // window_sum: the loop shifts the warp by one element per trip
      ["load", "coalesced", "4/4 sectors", "varies"],
      ["store", "coalesced", "4/4 sectors", "proven"],
      // add_bias: bias[0] for every thread
      ["load", "coalesced", "4/4 sectors", "proven"],
      ["load", "broadcast", "1/1 sectors", "proven"],
      ["store", "coalesced", "4/4 sectors", "proven"],
      // tile_copy_16x16: a warp covers two half rows
      ["load", "coalesced", "4/4 sectors", "proven"],
      ["store", "coalesced", "4/4 sectors", "proven"],
    ];
    const trace = events("gpu-patterns.trace.txt");
    // Inline launches, outlined kernels, and the kernels lowered to NVVM.
    for (const pass of [0, 1, trace.length - 2]) {
      const model = analyzeGpu(trace[pass].ir);
      const rows = model.launches.flatMap((launch) => {
        const k = model.kernels[launch.kernel];
        return k.accesses.map((access) => {
          const memref = parseMemref(access.type);
          const result = warpAccess(access, memref, memorySpace(memref.space), {
            defs: k.defs,
            args: k.args,
            block: launch.block,
            grid: launch.grid,
          });
          return [access.kind, result.verdict, `${result.sectors}/${result.needed} sectors`, result.proof.status];
        });
      });
      expect(rows).toEqual(expected);
    }
  });

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

  it("proves the transpose verdicts for all 32,768 warps", () => {
    const model = analyzeGpu(events("gpu-transpose.trace.txt")[0].ir);
    const launch = model.launches[0];
    const k = model.kernels[launch.kernel];
    const store = k.accesses.find((a) => a.kind === "store");
    const memref = parseMemref(store.type);
    expect(
      warpAccess(store, memref, memorySpace(memref.space), {
        defs: k.defs,
        args: k.args,
        block: launch.block,
        grid: launch.grid,
      }).proof,
    ).toEqual({
      status: "proven",
      verdict: "strided",
      sectors: 32,
      warps: 32768,
      iterations: false,
      formula: "32768·bx + 1024·tx + 32·by + ty",
      laneStride: 1024,
    });
  });

  it("keeps proving the transpose and tiled matmul verdicts after lowering to LLVM", () => {
    for (const [file, pass, warps, count] of [
      ["gpu-transpose.trace.txt", 6, 32768, 10],
      ["gpu-tiled-matmul.trace.txt", 5, 512, 7],
    ]) {
      const model = analyzeGpu(events(file)[pass].ir);
      let seen = 0;
      for (const launch of model.launches) {
        const k = model.kernels[launch.kernel];
        expect(k.op).toBe("llvm.func");
        for (const access of k.accesses) {
          const memref = parseMemref(access.type);
          const result = warpAccess(access, memref, memorySpace(memref.space), {
            defs: k.defs,
            args: k.args,
            block: launch.block,
            grid: launch.grid,
          });
          expect(result.proof).toMatchObject({ status: "proven", warps });
          seen++;
        }
      }
      expect(seen).toBe(count);
    }
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
