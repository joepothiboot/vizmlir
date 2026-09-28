import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadSampleState, SAMPLES, sampleFiles } from "../../src/samples.js";
import { isPassTrace, parsePassTrace } from "../../src/trace.js";
import { symbolHistory } from "../../src/provenance.js";
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
