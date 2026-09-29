import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { analyzeGpu } from "../../src/gpu.js";
import { acrossPasses, passSummary } from "../../src/gpu-view.js";
import { parsePassTrace } from "../../src/trace.js";

const events = (name) =>
  parsePassTrace(
    readFileSync(new URL(`../../public/samples/${name}`, import.meta.url), "utf8"),
  ).events;

// The picked row for access `index` of launch `launchIndex` in `ir`.
function pick(ir, launchIndex, index) {
  const model = analyzeGpu(ir);
  const kernel = model.kernels[model.launches[launchIndex].kernel];
  return { kernel, judged: { access: kernel.accesses[index], index } };
}

const verdicts = (history) =>
  history.map((h) => (h.result ? (h.result.analyzed ? h.result.verdict : "not analyzed") : null));

describe("acrossPasses", () => {
  const trace = events("gpu-transpose.trace.txt");
  const passes = { events: trace };

  it("follows an access through outlining and lowering by position, and loses it in the binary", () => {
    const { kernel, judged } = pick(trace[0].ir, 0, 1);
    expect(judged.access.kind).toBe("store");
    const history = acrossPasses(passes, 0, kernel, judged);
    expect(history).toHaveLength(trace.length);
    expect(verdicts(history)).toEqual([
      "strided",
      "strided",
      "strided",
      "strided",
      "strided",
      "strided",
      "strided",
      null,
    ]);
    // Outlining renamed the buffer, and lowering split it into pointers,
    // sizes and strides; the match is by position.
    expect(history[0].access.buffer).toBe("%arg1");
    expect(history[1].access.buffer).toBe("%arg2");
    expect(history[4].access.buffer).toBe("%arg9");
  });

  it("does not match an access of another kind at the same place", () => {
    const { kernel, judged } = pick(trace[0].ir, 0, 1);
    const load = { ...judged, access: { ...judged.access, kind: "load" } };
    expect(verdicts(acrossPasses(passes, 0, kernel, load)).every((v) => v === null)).toBe(true);
  });
});

describe("passSummary", () => {
  const at = (verdict, sectors) => ({ result: { analyzed: true, verdict, sectors } });
  const none = { result: null };

  it("says no pass changes a verdict, and where the access stops being readable", () => {
    expect(passSummary([at("strided", 32), at("strided", 32), none])).toBe(
      "Strided · 32 sectors in passes 1–2: no pass changes it, so a fix belongs in the source. From pass 3 on it can't be found: the kernel was compiled to a binary, or its loads and stores changed.",
    );
  });

  it("names the pass that changes a verdict", () => {
    expect(passSummary([at("strided", 32), at("coalesced", 4)])).toBe(
      "Pass 2 turns strided · 32 sectors into coalesced · 4 sectors.",
    );
  });

  it("says when no pass can be read", () => {
    expect(passSummary([none, none])).toBe("Not readable in any pass.");
  });
});
