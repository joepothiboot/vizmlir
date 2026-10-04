import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { analyzeGpu, memorySpace, ptxEntries } from "../../src/gpu.js";
import { parsePassTrace } from "../../src/trace.js";

const sample = (name) =>
  parsePassTrace(
    readFileSync(new URL(`../../public/samples/${name}`, import.meta.url), "utf8"),
  ).events;

describe("memorySpace", () => {
  it.each([
    ["", "global"],
    ["1", "global"],
    ["global", "global"],
    ["3", "shared"],
    ["workgroup", "shared"],
    ["5", "private"],
    ["private", "private"],
    ["7", "space 7"],
    ["#dsp.local", "local"],
    ["#foo.local", "space #foo.local"],
  ])("%j -> %s", (space, name) => {
    expect(memorySpace(space)).toBe(name);
  });

  it("uses the fallback for a buffer with no space", () => {
    expect(memorySpace("", "private")).toBe("private");
  });
});

describe("analyzeGpu", () => {
  it("returns null for IR without GPU code", () => {
    expect(analyzeGpu("module {\n  func.func @f() {\n    return\n  }\n}")).toBe(null);
  });

  it("reads an inline gpu.launch printed over several lines", () => {
    const ir = [
      "func.func @f(%x: memref<64xf32>, %n: index) {",
      "  %c2 = arith.constant 2 : index",
      "  %c1 = arith.constant 1 : index",
      "  %c32 = arith.constant 32 : index",
      "  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c2, %gy = %c1, %gz = %c1)",
      "             threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %n, %sz = %c1)",
      "             workgroup(%sh : memref<32xf32, #gpu.address_space<workgroup>>)",
      "             private(%p : memref<4xf32, #gpu.address_space<private>>) {",
      "    %v = memref.load %x[%tx] : memref<64xf32>",
      "    memref.store %v, %sh[%tx] : memref<32xf32, #gpu.address_space<workgroup>>",
      "    gpu.terminator",
      "  }",
      "  return",
      "}",
    ].join("\n");
    const { launches, kernels } = analyzeGpu(ir);
    expect(launches).toEqual([
      { line: 5, host: "@f", kernel: 0, grid: [2, 1, 1], block: [32, null, 1], threads: null },
    ]);
    expect(kernels[0]).toMatchObject({ inline: true, op: "gpu.launch" });
    expect(
      kernels[0].buffers.map((b) => [b.name, b.space, b.bytes, b.source, b.loads, b.stores]),
    ).toEqual([
      ["%sh", "shared", 128, "workgroup", 0, 1],
      ["%p", "private", 16, "private", 0, 0],
      ["%x", "global", 256, "captured", 1, 0],
    ]);
  });

  it("follows the saxpy sample from inline launch to PTX", () => {
    const events = sample("gpu-kernels.trace.txt");
    const before = analyzeGpu(events[0].ir);
    expect(before.launches.map((l) => [l.host, l.grid, l.block, l.threads])).toEqual([
      ["@saxpy", [4, 1, 1], [256, 1, 1], 1024],
      ["@relu", [4, 1, 1], [256, 1, 1], 1024],
    ]);

    const outlined = analyzeGpu(events[1].ir);
    const saxpy = outlined.kernels[outlined.launches[0].kernel];
    expect(saxpy).toMatchObject({ path: "@saxpy_kernel::@saxpy_kernel", op: "gpu.func" });
    expect(saxpy.buffers.map((b) => [b.name, b.space, b.source, b.loads, b.stores])).toEqual([
      ["%arg1", "global", "argument", 1, 0],
      ["%arg2", "global", "argument", 1, 1],
    ]);

    const binary = analyzeGpu(events.at(-1).ir);
    const ptx = binary.kernels[binary.launches[0].kernel];
    expect(ptx).toMatchObject({ op: "gpu.binary", lowered: true, buffers: [] });
    expect(ptx.ptx.registers).toEqual([
      { type: "b32", count: 8 },
      { type: "b64", count: 12 },
    ]);
  });

  it("finds shared memory in the tiled matmul sample", () => {
    const events = sample("gpu-tiled-matmul.trace.txt");
    const outlined = analyzeGpu(events[1].ir);
    const [launch] = outlined.launches;
    expect([launch.grid, launch.block, launch.threads]).toEqual([
      [8, 8, 1],
      [16, 16, 1],
      16384,
    ]);
    const kernel = outlined.kernels[launch.kernel];
    const shared = kernel.buffers.filter((b) => b.space === "shared");
    expect(shared.map((b) => [b.bytes, b.source, b.loads, b.stores])).toEqual([
      [1024, "workgroup", 1, 1],
      [1024, "workgroup", 1, 1],
    ]);
    expect(kernel.buffers.filter((b) => b.space === "global")).toHaveLength(3);

    const binary = analyzeGpu(events.at(-1).ir);
    expect(binary.kernels[binary.launches[0].kernel].ptx.sharedBytes).toBe(2048);
  });
});

describe("ptxEntries", () => {
  it("reads registers and shared memory per entry", () => {
    const ptx = [
      ".visible .entry a(",
      ") {",
      "\t.reg .pred \t%p<3>;",
      "\t.reg .b32 \t%r<8>;",
      "\t.shared .align 4 .b8 tile[1024];",
      "}",
      ".visible .entry b() {",
      "\t.reg .f32 \t%f<2>;",
      "}",
    ].join("\n");
    expect(ptxEntries(ptx)).toEqual([
      {
        name: "a",
        registers: [
          { type: "pred", count: 3 },
          { type: "b32", count: 8 },
        ],
        sharedBytes: 1024,
      },
      { name: "b", registers: [{ type: "f32", count: 2 }], sharedBytes: 0 },
    ]);
  });
});
