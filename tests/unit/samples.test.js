import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_SAMPLE_ID,
  groupSamples,
  loadSampleState,
  SAMPLES,
  SCOPES,
  sampleFiles,
  scopeOf,
} from "../../src/samples.js";
import { describeEvent, isPassTrace, parsePassTrace } from "../../src/trace/trace.js";
import { symbolHistory } from "../../src/trace/provenance.js";
import {
  compareBenchmarks,
  parseBenchmarks,
  symbolTimes,
} from "../../src/bench.js";

// Samples in public/samples are real mlir-opt output; see samples/generate.sh.
const read = async (path) =>
  readFileSync(
    new URL(`../../public/samples/${path}`, import.meta.url),
    "utf8",
  );

describe("samples", () => {
  it("each belong to a known scenario", () => {
    const known = new Set(SCOPES.map((scope) => scope.id));
    for (const sample of SAMPLES) expect(known.has(sample.scope), sample.id).toBe(true);
    expect(new Set(SCOPES.map((scope) => scope.id)).size).toBe(SCOPES.length);
    for (const scope of SCOPES) expect(scope.short.length, scope.id).toBeLessThanOrEqual(12);
  });

  it("group by scenario without losing or repeating any", () => {
    const groups = groupSamples();
    expect(groups.flatMap((g) => g.samples.map((s) => s.id)).sort()).toEqual(SAMPLES.map((s) => s.id).sort());
    for (const { scope, samples } of groups) for (const sample of samples) expect(scopeOf(sample)).toBe(scope);
  });

  it("leave out a scenario that has no samples", () => {
    const only = SAMPLES.filter((s) => s.scope === "gpu");
    expect(groupSamples(only).map((g) => g.scope.id)).toEqual(["gpu"]);
  });

  it("open a first visit on a debugging sample with Debug on", () => {
    const sample = SAMPLES.find((s) => s.id === DEFAULT_SAMPLE_ID);
    expect(sample).toBeDefined();
    expect(scopeOf(sample)).toMatchObject({ id: "debug", debug: true });
  });

  it("only turn Debug on for scenarios that have source locations", async () => {
    for (const scope of SCOPES.filter((s) => s.debug))
      for (const sample of SAMPLES.filter((s) => s.scope === scope.id))
        expect(sample.sources?.length, sample.id).toBeGreaterThan(0);
  });

  it("open on the pass they say they start at", async () => {
    for (const sample of SAMPLES.filter((s) => s.startPass !== undefined)) {
      const state = await loadSampleState(sample, read);
      const events = parsePassTrace(state.trace).events;
      expect(state.traceIndex, sample.id).toBe(sample.startPass);
      expect(describeEvent(events[sample.startPass]), sample.id).toContain(sample.startsAt);
    }
  });

  it("have unique ids", () => {
    const ids = SAMPLES.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it.each(SAMPLES.map((s) => [s.id, s]))("%s loads", async (_, sample) => {
    for (const path of sampleFiles(sample))
      expect((await read(path)).trim(), path).not.toBe("");
    const state = await loadSampleState(sample, read);
    if (sample.trace) {
      expect(isPassTrace(state.trace)).toBe(true);
      expect(parsePassTrace(state.trace).events.length).toBeGreaterThan(1);
    } else {
      expect(state.baseline).not.toBe(state.current);
      expect(state.current).not.toMatch(/error:/);
    }
  });

  it("lowering trace steps through every pass", async () => {
    const trace = parsePassTrace(await read("lowering.trace.txt"));
    expect(trace.events).toHaveLength(7);
    expect(trace.events.some((e) => e.failed)).toBe(false);
  });

  it("failed trace ends on the failing pass", async () => {
    const trace = parsePassTrace(await read("failed-transform.trace.txt"));
    expect(trace.events.at(-1).failed).toBe(true);
    expect(trace.diagnostics.length).toBeGreaterThan(0);
  });

  it("GPU sample ties mock timings to the kernels the trace built", async () => {
    const sample = SAMPLES.find((s) => s.id === "gpu-kernels");
    const state = await loadSampleState(sample, read);
    const history = symbolHistory(parsePassTrace(state.trace).events);
    const kernel = history.find(
      (r) => r.path === "@saxpy_kernel::@saxpy_kernel",
    );
    expect(kernel.ops).toEqual(["gpu.func", "llvm.func"]);
    expect(kernel.changes[0].kind).toBe("created");

    expect(state.benchmarks.current.mock).toBe(true);
    expect(state.benchmarks.current.text).toMatch(/^# MOCK DATA/);
    const runs = ["baseline", "current"].map((slot) =>
      symbolTimes(parseBenchmarks(state.benchmarks[slot].text).entries, history),
    );
    const comparison = compareBenchmarks(runs[0].times, runs[1].times);
    expect(comparison.get("@saxpy_kernel::@saxpy_kernel").change).toBeCloseTo(
      0.258,
      2,
    );
    expect(comparison.get("@relu_kernel::@relu_kernel").change).toBeLessThan(0);
    // The library kernel shows how unmatched kernels are listed.
    expect(runs[1].unmatched.map((m) => m.entry.kernel)).toEqual([
      "void cub::DeviceReduceKernel<float>(float*, int)",
    ]);
  });
});
