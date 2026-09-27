import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadSampleState, SAMPLES, sampleFiles } from "../../src/samples.js";
import { isPassTrace, parsePassTrace } from "../../src/trace.js";

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
});
