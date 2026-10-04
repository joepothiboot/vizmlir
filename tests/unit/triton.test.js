import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { buildDefs } from "../../src/gpu-access.js";
import { analyzeGpu } from "../../src/gpu.js";
import { acrossPasses } from "../../src/gpu-view.js";
import { parsePassTrace } from "../../src/trace/trace.js";
import {
  analyzeTriton,
  isTriton,
  ownedBy,
  parseLayouts,
  parseTensor,
  pointerOffset,
  tritonAccess,
} from "../../src/triton.js";

const trace = parsePassTrace(
  readFileSync(new URL("../../public/samples/triton-coalesce.trace.txt", import.meta.url), "utf8"),
).events;

describe("layouts and types", () => {
  it("reads #blocked layouts with either dialect prefix", () => {
    const layouts = parseLayouts(
      [
        "#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>",
        "#old = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>",
        "#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>",
      ].join("\n"),
    );
    expect(layouts.get("#blocked")).toEqual({
      kind: "blocked",
      sizePerThread: [1, 4],
      threadsPerWarp: [8, 4],
      warpsPerCTA: [4, 1],
      order: [1, 0],
    });
    expect(layouts.get("#old").sizePerThread).toEqual([4]);
    expect(layouts.has("#mma")).toBe(false);
  });

  it("reads tensor types, pointer tensors included", () => {
    expect(parseTensor("tensor<32x16x!tt.ptr<f32>, #blocked1>")).toEqual({
      shape: [32, 16],
      element: "!tt.ptr<f32>",
      pointee: "f32",
      encoding: "#blocked1",
    });
    expect(parseTensor("tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>")).toMatchObject({
      shape: [32],
      element: "i32",
      encoding: "#ttg.slice<{dim = 1, parent = #blocked}>",
    });
  });
});

describe("ownedBy", () => {
  // The layout from the design mockup: 4 elements per thread along the rows.
  const layout = { sizePerThread: [1, 4], threadsPerWarp: [8, 4], warpsPerCTA: [4, 1], order: [1, 0] };

  it("gives each thread a run along the fastest dimension, lanes first along it", () => {
    expect(ownedBy(layout, [32, 16], 1, 5).reps).toEqual([
      [
        [9, 4],
        [9, 5],
        [9, 6],
        [9, 7],
      ],
    ]);
  });

  it("repeats over a larger tensor and wraps over a smaller one", () => {
    expect(ownedBy(layout, [64, 16], 0, 0).reps.map((rep) => rep[0])).toEqual([
      [0, 0],
      [32, 0],
    ]);
    // 8 columns: lanes 0 and 2 of a row hold the same elements (broadcast).
    expect(ownedBy(layout, [32, 8], 0, 2).reps[0][0]).toEqual([0, 0]);
  });
});

describe("pointer offsets", () => {
  it("writes base + offsets as a linear function of coordinates and program id", () => {
    const defs = buildDefs(
      [
        "%c64 = arith.constant 64 : i32",
        "%0 = tt.get_program_id x : i32",
        "%1 = arith.muli %0, %c64 : i32",
        "%2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>",
        "%3 = tt.splat %1 : i32 -> tensor<64xi32, #blocked>",
        "%4 = arith.addi %3, %2 : tensor<64xi32, #blocked>",
        "%5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>",
        "%6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>",
      ],
      1,
    );
    const { base, offset } = pointerOffset("%6", defs);
    expect(base).toBe("%arg0");
    expect(Object.fromEntries(offset.t)).toEqual({ bx: 64, c0: 1 });
  });
});

describe("the Triton sample", () => {
  const judged = (ir) => {
    const model = analyzeGpu(ir);
    const kernel = model.kernels[0];
    return kernel.accesses.map((access) => ({ access, result: tritonAccess(access, kernel) }));
  };

  it("is read as one program of 4 warps", () => {
    expect(isTriton(trace[0].ir)).toBe(true);
    const model = analyzeTriton(trace[0].ir);
    expect(model.launches[0]).toMatchObject({ grid: [null, null, null], block: [128, 1, 1] });
    expect(model.kernels[0]).toMatchObject({ name: "transpose_tile", op: "tt.func" });
  });

  it("proves the load strided before the coalesce pass and coalesced after", () => {
    const [before, after] = [judged(trace[0].ir), judged(trace[1].ir)];
    expect(before.map(({ access, result }) => [access.kind, access.buffer, result.verdict, result.sectors])).toEqual([
      ["load", "%arg0", "strided", 32],
      ["store", "%arg1", "coalesced", 4],
    ]);
    expect(before[0].result.proof).toMatchObject({
      status: "proven",
      formula: "32768·pid_y + 1024·j + 32·pid_x + i",
      laneStride: 1024,
    });
    // #blocked1: 4 floats per thread down a column, 8 lanes per column.
    expect(after.map(({ result }) => [result.verdict, result.sectors, result.distinct])).toEqual([
      ["coalesced", 16, 128],
      ["coalesced", 16, 128],
    ]);
    expect(after[0].result.lanes[1]).toMatchObject({ index: [4, 0], vector: 4 });
    expect(after[0].result.proof).toMatchObject({ status: "proven", verdict: "coalesced" });
  });

  it("follows the load across the pass by position", () => {
    const model = analyzeGpu(trace[0].ir);
    const kernel = model.kernels[0];
    const history = acrossPasses({ events: trace }, 0, kernel, { access: kernel.accesses[0], index: 0 });
    expect(history.map((h) => h.result.verdict)).toEqual(["strided", "coalesced"]);
  });

  it("says why an offset it cannot read is not analyzed", () => {
    const ir = trace[0].ir.replace(
      "%12 = arith.muli %11, %cst_0 : tensor<1x32xi32, #blocked>",
      "%12 = arith.muli %11, %11 : tensor<1x32xi32, #blocked>",
    );
    expect(judged(ir)[0].result).toMatchObject({ analyzed: false });
    expect(judged(ir)[0].result.reason).toMatch(/%12 is not a linear function/);
  });
});
