import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { analyzeGpu } from "../../src/gpu.js";
import { flowSlice, kernelFlow } from "../../src/gpu-flow.js";
import { parsePassTrace } from "../../src/trace.js";

const transpose = () => {
  const events = parsePassTrace(
    readFileSync(new URL("../../public/samples/gpu-transpose.trace.txt", import.meta.url), "utf8"),
  ).events;
  return { ir: events[0].ir, model: analyzeGpu(events[0].ir) };
};

describe("kernelFlow", () => {
  it("traces each access back to the thread and block ids", () => {
    const { ir, model } = transpose();
    const flow = kernelFlow(model.kernels[0]);
    const kinds = new Set(flow.nodes.map((n) => n.kind));
    expect(kinds).toEqual(new Set(["block", "thread", "const", "math", "access"]));
    const load = flow.nodes.find((n) => n.kind === "access" && n.access.kind === "load");
    expect(ir.split("\n")[load.line - 1]).toContain("memref.load");
    // Accesses share the last rank; ids start at 0.
    const last = Math.max(...flow.nodes.map((n) => n.rank));
    expect(flow.nodes.filter((n) => n.kind === "access").every((n) => n.rank === last)).toBe(true);
    expect(flow.nodes.filter((n) => n.kind === "thread").every((n) => n.rank === 0)).toBe(true);
    expect(flow.nodes.find((n) => n.kind === "thread").label).toMatch(/^thread [xyz]$/);
    // Every node line points at its definition.
    for (const node of flow.nodes.filter((n) => n.kind === "math"))
      expect(ir.split("\n")[node.line - 1]).toContain(`${node.name} =`);
  });

  it("slices a clicked line to what it reads and what reads it", () => {
    const { model } = transpose();
    const flow = kernelFlow(model.kernels[0]);
    const load = flow.nodes.find((n) => n.kind === "access" && n.access.kind === "load");
    const slice = flowSlice(flow, load.line);
    expect(slice.has(load.id)).toBe(true);
    expect([...slice].some((id) => flow.nodes.find((n) => n.id === id).kind === "thread")).toBe(true);
    const store = flow.nodes.find((n) => n.kind === "access" && n.access.kind === "store");
    expect(slice.has(store.id)).toBe(false);
    expect(flowSlice(flow, 1).size).toBe(0);
  });

  it("is empty for kernels without accesses", () => {
    expect(kernelFlow({ defs: new Map(), accesses: [] })).toEqual({ nodes: [], edges: [] });
  });
});
