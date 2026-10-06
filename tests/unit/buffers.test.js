import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  analyzeBuffers,
  bufferTotals,
  buffersToJSON,
  compareBuffers,
  elementBytes,
  parseMemref,
} from "../../src/trace/buffers.js";
import { moduleStateAt, parsePassTrace } from "../../src/trace/trace.js";

const lowering = readFileSync(
  new URL("../../public/samples/lowering.trace.txt", import.meta.url),
  "utf8",
);

const only = (ir) => analyzeBuffers(ir).functions[0];
const byName = (fn, name) => fn.buffers.find((b) => b.name === name);

describe("elementBytes", () => {
  it.each([
    ["f32", 4],
    ["bf16", 2],
    ["f64", 8],
    ["i1", 1],
    ["i4", 1],
    ["i32", 4],
    ["ui8", 1],
    ["index", 8],
    ["f8E4M3FN", 1],
    ["complex<f32>", 8],
    ["vector<4x8xf16>", 64],
    ["!my.type", null],
  ])("%s -> %s", (type, bytes) => {
    expect(elementBytes(type)).toBe(bytes);
  });
});

describe("parseMemref", () => {
  it("computes the static size", () => {
    expect(parseMemref("memref<128x64xf32>")).toMatchObject({
      dims: [128, 64],
      element: "f32",
      bytes: 32768,
      space: "",
    });
  });

  it("treats a 0-d memref as one element", () => {
    expect(parseMemref("memref<f64>").bytes).toBe(8);
  });

  it("leaves dynamic and unranked sizes unknown", () => {
    expect(parseMemref("memref<?x64xf32>").bytes).toBeNull();
    expect(parseMemref("memref<*xf32>").bytes).toBeNull();
  });

  it("ignores layouts and reads the memory space", () => {
    const strided = parseMemref(
      "memref<128x256xf32, strided<[?, ?], offset: ?>>",
    );

    expect(strided).toMatchObject({ bytes: 131072, space: "" });
    expect(parseMemref("memref<4xf32, #map>").space).toBe("");
    expect(parseMemref("memref<4xf32, 3>").space).toBe("3");

    expect(
      parseMemref("memref<16xf16, #gpu.address_space<workgroup>>").space,
    ).toBe("workgroup");
  });
});

describe("analyzeBuffers", () => {
  it("ends a buffer at its dealloc", () => {
    const fn = only(`func.func @f() {
  %a = memref.alloc() : memref<16xf32>
  %b = memref.alloc() : memref<32xf32>
  memref.copy %a, %b : memref<16xf32> to memref<32xf32>
  memref.dealloc %a : memref<16xf32>
  "test.use"(%b) : (memref<32xf32>) -> ()
  memref.dealloc %b : memref<32xf32>
  return
}`);

    expect(fn.name).toBe("@f");

    expect(byName(fn, "%a")).toMatchObject({
      bytes: 64,
      start: 0,
      end: 3,
      freed: "dealloc",
    });

    expect(byName(fn, "%b")).toMatchObject({ start: 1, end: 5 });
    expect(fn.peak).toBe(64 + 128);
    expect(fn.peakAt).toBe(1);
    expect(fn.live).toEqual([64, 192, 192, 192, 128, 128, 0]);
  });

  it("keeps a buffer used in a loop body live until the loop ends", () => {
    const fn = only(`func.func @f(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a = memref.alloc() : memref<8xf32>
  scf.for %i = %c0 to %n step %c1 {
    %v = memref.load %a[%i] : memref<8xf32>
    "test.sink"(%v) : (f32) -> ()
  }
  "test.after"() : () -> ()
  return
}`);

    expect(byName(fn, "%a")).toMatchObject({ start: 2, end: 5 });
  });

  it("follows aliases and marks returned buffers", () => {
    const fn =
      only(`func.func @f() -> memref<4x4xf32, strided<[?, ?], offset: ?>> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>
  %cast = memref.cast %alloc : memref<4x4xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
  "test.other"() : () -> ()
  return %cast : memref<4x4xf32, strided<[?, ?], offset: ?>>
}`);

    const buffer = byName(fn, "%alloc");
    expect(buffer.aliases).toEqual(["%cast"]);
    expect(buffer.freed).toBe("returned");
    expect(buffer.end).toBe(fn.length - 1);
    expect(buffer.bytes).toBe(64);
  });

  it("reads alloca, gpu.alloc and generic-form ops", () => {
    const fn = only(`gpu.func @k() kernel {
  %s = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
  %g = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2xi64>
  %d = gpu.alloc () : memref<?xf32>
  gpu.return
}`);

    expect(
      fn.buffers.map((b) => [b.name, b.op, b.bytes, b.space, b.freed]),
    ).toEqual([
      ["%s", "memref.alloca", 64, "workgroup", "scope"],
      ["%g", "memref.alloc", 16, "", "last-use"],
      ["%d", "gpu.alloc", null, "", "last-use"],
    ]);

    expect(fn.dynamic).toBe(1);
    expect(fn.allocated).toBe(80);
  });

  it("separates functions and lists globals", () => {
    const { functions, globals } = analyzeBuffers(`module {
  memref.global "private" constant @__constant_4xf32 : memref<4xf32> = dense<1.0>
  func.func private @decl(memref<4xf32>)
  func.func @a() {
    %x = memref.alloc() : memref<4xf32>
    return
  }
  func.func @b() {
    %x = memref.alloc() : memref<8xf32>
    return
  }
}`);

    expect(functions.map((fn) => [fn.name, fn.allocated])).toEqual([
      ["@a", 16],
      ["@b", 32],
    ]);

    expect(globals).toMatchObject([
      { name: "@__constant_4xf32", bytes: 16, constant: true, line: 1 },
    ]);
  });

  it("finds nothing in tensor-level IR", () => {
    const fn = only(`func.func @f(%t: tensor<4xf32>) -> tensor<4xf32> {
  %e = tensor.empty() : tensor<4xf32>
  return %e : tensor<4xf32>
}`);

    expect(fn.buffers).toEqual([]);
    expect(fn.peak).toBe(0);
  });
});

describe("the lowering sample", () => {
  const trace = parsePassTrace(lowering);

  const states = trace.events.map((_, i) =>
    analyzeBuffers(moduleStateAt(trace.events, i)),
  );

  const totals = states.map(bufferTotals);

  it("has no buffers until bufferization, then one 32 KB result", () => {
    const first = trace.events.findIndex((e) => /Bufferize/.test(e.pass));
    expect(first).toBeGreaterThan(0);
    for (let i = 0; i < first; i++) expect(totals[i].buffers).toBe(0);

    for (let i = first; i < totals.length; i++) {
      expect(totals[i]).toMatchObject({
        buffers: 1,
        allocated: 128 * 64 * 4,
        peak: 128 * 64 * 4,
      });
    }

    const [buffer] = states.at(-1).functions[0].buffers;
    expect(buffer).toMatchObject({ name: "%alloc", freed: "returned" });
  });

  it("matches the buffer across the loop lowering", () => {
    const i = trace.events.findIndex((e) => /LinalgToLoops/.test(e.pass));
    const [dense] = compareBuffers(states[i - 1], states[i]);
    expect(dense.rows.map((row) => row.status)).toEqual(["same"]);
  });
});

describe("compareBuffers", () => {
  const before = analyzeBuffers(`func.func @f() {
  %a = memref.alloc() : memref<16xf32>
  %b = memref.alloc() : memref<16xf32>
  %c = memref.alloc() : memref<8xi8>
  "test.use"(%a, %b, %c) : (memref<16xf32>, memref<16xf32>, memref<8xi8>) -> ()
  memref.dealloc %a : memref<16xf32>
  return
}`);

  const after = analyzeBuffers(`func.func @f() {
  %alloc = memref.alloc() : memref<16xf32>
  "test.use"(%alloc) : (memref<16xf32>) -> ()
  %alloc_0 = memref.alloc() : memref<4xf64>
  "test.use"(%alloc_0) : (memref<4xf64>) -> ()
  return
}`);

  it("pairs buffers by op and type, not by SSA name", () => {
    const [f] = compareBuffers(before, after);

    expect(
      f.rows.map((r) => [
        r.status,
        r.before?.name ?? null,
        r.after?.name ?? null,
      ]),
    ).toEqual([
      ["changed", "%a", "%alloc"],
      ["added", null, "%alloc_0"],
      ["removed", "%b", null],
      ["removed", "%c", null],
    ]);
  });

  it("marks everything the same without a baseline", () => {
    const [f] = compareBuffers(null, after);
    expect(f.rows.every((r) => r.status === "same")).toBe(true);
  });

  it("lists functions from both sides", () => {
    const g = analyzeBuffers(`func.func @g() {
  return
}`);

    expect(
      compareBuffers(before, g).map((f) => [f.name, !!f.before, !!f.after]),
    ).toEqual([
      ["@f", true, false],
      ["@g", false, true],
    ]);
  });

  it("serializes with 1-based lines", () => {
    const json = JSON.parse(buffersToJSON("t", compareBuffers(before, after)));

    expect(json.functions[0].buffers[0].after).toMatchObject({
      name: "%alloc",
      line: 2,
      bytes: 64,
    });
  });
});
