import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  analyzeLocalMemory,
  hasLocalMemory,
  localLineInfo,
  localTargetOf,
} from "../../src/gpu/local-memory.js";
import { parsePassTrace } from "../../src/trace/trace.js";

const events = parsePassTrace(
  readFileSync(
    new URL(
      "../../public/samples/nanodsp-local-matmul.trace.txt",
      import.meta.url,
    ),
    "utf8",
  ),
).events;

const at = (argument) => events.findIndex((e) => e.argument === argument);

const analyze = (index) =>
  analyzeLocalMemory(events[index].ir, {
    target: localTargetOf(events, index),
  });

const PROMOTE = at("nanodsp-promote-local");
const LOWER = at("nanodsp-lower-local");

describe("analyzeLocalMemory across the nano-dsp passes", () => {
  it("finds nothing before promote-local", () => {
    expect(PROMOTE).toBeGreaterThan(0);

    for (let i = 0; i < PROMOTE; i++) {
      const analysis = analyze(i);
      expect(analysis.stage, events[i].argument).toBe("none");
      expect(hasLocalMemory(analysis)).toBe(false);
      expect(analysis.budget).toBe(null);
    }
  });

  it("finds two double-buffered tiles in #dsp.local after promote-local", () => {
    const analysis = analyze(PROMOTE);
    expect(analysis.stage).toBe("dma");

    const [fn] = analysis.functions;
    expect(fn.name).toBe("@matmul_local");

    expect(
      fn.localBuffers.map((b) => [
        b.label,
        b.name,
        b.slots,
        b.slotBytes,
        b.source,
        b.lowered,
      ]),
    ).toEqual([
      ["A", "%alloc_0", 2, 65536, "%arg0", false],
      ["B", "%alloc_2", 2, 65536, "%arg1", false],
    ]);

    expect(fn.localBuffers[0].shape).toEqual([2, 128, 128]);

    expect(fn.tags.map((t) => [t.name, t.slots])).toEqual([
      ["%alloc_1", 2],
      ["%alloc_3", 2],
    ]);

    expect(fn.peakLocalBytes).toBe(262144);

    expect(fn.budget).toMatchObject({
      bytes: 262144,
      source: "target-model",
      target: "hexagon-hvx128",
    });

    expect(fn.budget.note).toMatch(/^assumed, from nano-dsp TargetModel/);
  });

  it("tells the prologue loads, the prefetches and the waits apart", () => {
    const [fn] = analyze(PROMOTE).functions;

    const rows = fn.dmas.map((t) => [
      t.op,
      t.role,
      t.srcRoot,
      t.buffer,
      t.slot.value ?? t.slot.of,
    ]);

    expect(rows).toEqual([
      ["memref.dma_start", "prologue", "%arg0", "%alloc_0", 0],
      ["memref.dma_start", "prologue", "%arg1", "%alloc_2", 0],
      ["memref.dma_start", "prefetch", "%arg0", "%alloc_0", "i+1"],
      ["memref.dma_start", "prefetch", "%arg1", "%alloc_2", "i+1"],
      ["memref.dma_wait", "wait", "%arg0", "%alloc_0", "i"],
      ["memref.dma_wait", "wait", "%arg1", "%alloc_2", "i"],
    ]);

    expect(fn.copies).toEqual([]);

    expect(fn.dmas[0]).toMatchObject({
      elements: 16384,
      bytes: 65536,
      strided: { stride: 256, perStride: 128 },
    });

    expect(fn.dmas[1].strided).toBe(null);

    const lines = events[PROMOTE].ir.split("\n");
    for (const t of fn.dmas) expect(lines[t.line]).toContain(t.op);

    for (const b of fn.localBuffers) {
      expect(lines[b.line]).toContain(`${b.name} = memref.alloc`);
    }
  });

  it("reads the steady-state schedule of the cache loop", () => {
    const { pipeline } = analyze(PROMOTE).functions[0];

    expect(pipeline.loop).toMatchObject({
      lb: 0,
      ub: 256,
      step: 128,
      trips: 2,
      cacheLoop: true,
    });

    expect(pipeline.mode).toBe("double");
    expect(pipeline.synchronous).toBe(false);
    expect(pipeline.steady).toEqual(["prefetch", "wait", "compute"]);

    const summary = pipeline.iterations.map((it) =>
      it.events.map((e) => `${e.kind}${e.tile}${e.slot ?? ""}`),
    );

    expect(summary).toEqual([
      [
        "prologue00",
        "prologue00",
        "prefetch11",
        "prefetch11",
        "wait00",
        "wait00",
        "compute0",
      ],
      ["wait11", "wait11", "compute1"],
    ]);

    expect(
      pipeline.iterations[1].events
        .at(-1)
        .uses.map((u) => `${u.label}${u.slot}`),
    ).toEqual(["A1", "B1"]);

    expect(events[PROMOTE].ir.split("\n")[pipeline.computeLine]).toMatch(
      /scf\.for/,
    );
  });

  it("finds synchronous copies into the same slots after lower-local", () => {
    const analysis = analyze(LOWER);
    expect(analysis.stage).toBe("copies");

    const [fn] = analysis.functions;
    expect(fn.dmas).toEqual([]);
    expect(fn.tags).toEqual([]);

    expect(fn.localBuffers.map((b) => [b.label, b.slots, b.lowered])).toEqual([
      ["A", 2, true],
      ["B", 2, true],
    ]);

    expect(
      fn.copies.map((t) => [t.role, t.srcRoot, t.slot.value ?? t.slot.of]),
    ).toEqual([
      ["prologue", "%arg0", 0],
      ["prologue", "%arg1", 0],
      ["prefetch", "%arg0", "i+1"],
      ["prefetch", "%arg1", "i+1"],
    ]);

    expect(fn.peakLocalBytes).toBe(262144);
    expect(fn.pipeline.synchronous).toBe(true);
    expect(fn.pipeline.steady).toEqual(["prefetch", "compute"]);
  });

  it("looks a line up", () => {
    const analysis = analyze(PROMOTE);
    const [fn] = analysis.functions;

    expect(localLineInfo(analysis, fn.localBuffers[0].line)).toMatchObject({
      kind: "buffer",
    });

    expect(localLineInfo(analysis, fn.dmas[2].line).item.role).toBe("prefetch");
    expect(localLineInfo(analysis, fn.tags[0].line).kind).toBe("tag");
    expect(localLineInfo(analysis, 0)).toBe(null);
  });
});

describe("analyzeLocalMemory on small inputs", () => {
  it("takes the budget from the IR when it carries one", () => {
    const ir = `module attributes {dsp.local_mem_bytes = 65536} {
  func.func @f(%a: memref<64xf32>) {
    %buf = memref.alloc() : memref<64xf32, #dsp.local>
    return
  }
}`;

    const analysis = analyzeLocalMemory(ir);
    expect(analysis.budget).toMatchObject({ bytes: 65536, source: "ir" });

    expect(analysis.functions[0].localBuffers[0]).toMatchObject({
      slots: 1,
      bytes: 256,
    });

    expect(analysis.functions[0].pipeline).toBe(null);
  });

  it("ignores GPU address spaces and plain copies", () => {
    const ir = `func.func @g(%a: memref<64xf32>) {
  %s = memref.alloc() : memref<64xf32, 3>
  %t = memref.alloc() : memref<64xf32>
  memref.copy %a, %t : memref<64xf32> to memref<64xf32>
  return
}`;

    expect(analyzeLocalMemory(ir).stage).toBe("none");
  });

  it("names the target from the trace's pass options", () => {
    expect(localTargetOf(events, PROMOTE)).toBe("hexagon-hvx128");
    expect(localTargetOf([{ options: null }])).toBe(null);
  });
});
