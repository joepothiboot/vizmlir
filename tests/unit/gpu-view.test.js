// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { URL as FileURL } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { analyzeGpu } from "../../src/gpu/model.js";
import {
  acrossPasses,
  passSummary,
  renderGpuPath,
  renderGpuView,
  verdictSummary,
} from "../../src/gpu-view.js";
import { parsePassTrace } from "../../src/trace/trace.js";

const events = (name) =>
  parsePassTrace(
    readFileSync(new FileURL(`../../public/samples/${name}`, import.meta.url), "utf8"),
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

// The page shows the picked access's answer card in its inspector and keeps
// a one-line verdict over the GPU view, so the view itself holds the picture.
describe("GPU view in the page", () => {
  const ir = events("gpu-transpose.trace.txt")[0].ir;
  // jsdom has no 2D canvas; the 3D view draws on animation frames.
  globalThis.ResizeObserver ??= class {
    observe() {}
    disconnect() {}
  };
  window.requestAnimationFrame = () => 0;

  function draw() {
    const view = document.createElement("div");
    document.body.replaceChildren(view);
    const onAnswer = vi.fn();
    renderGpuView(view, analyzeGpu(ir), { onAnswer });
    return { view, onAnswer };
  }

  it("hands the answer card to onAnswer instead of drawing it in the view", () => {
    const { view, onAnswer } = draw();
    expect(view.querySelector(".gpu-answer")).toBeNull();
    // One initial pick per launch with loads or stores.
    expect(onAnswer).toHaveBeenCalledTimes(analyzeGpu(ir).launches.length);
    const [card, judged, { initial, launchIndex }] = onAnswer.mock.calls[0];
    expect(card.classList.contains("gpu-answer")).toBe(true);
    expect(initial).toBe(true);
    expect(launchIndex).toBe(0);
    expect(card.textContent).toContain(`Line ${judged.access.line}`);
    expect(card.textContent).toContain("Each warp touches");
  });

  it("reports a picked access as the person's pick", () => {
    const { view, onAnswer } = draw();
    const row = view.querySelector(".gpu-access-table tbody tr:first-child");
    row.click();
    const [, judged, { initial }] = onAnswer.mock.calls.at(-1);
    expect(initial).toBe(false);
    expect(row.textContent).toContain(judged.access.buffer);
  });

  it("sums up the picked access in one line", () => {
    const { onAnswer } = draw();
    const [, judged] = onAnswer.mock.calls[0];
    const line = document.createElement("span");
    line.append(...verdictSummary(judged));
    expect(line.querySelector(".where").textContent).toBe(
      `Line ${judged.access.line} · ${judged.access.kind} ${judged.access.buffer}`,
    );
    expect(line.querySelector(".gpu-chip").textContent).toMatch(/^strided · 32 sectors$/);
    expect(line.querySelector(".reach").textContent).toContain("32,768 warps");
  });

  it("does not repeat the access list in the GPU path before a line is picked", () => {
    draw();
    const path = document.createElement("div");
    renderGpuPath(path);
    expect(path.querySelector("h3").textContent).toBe("How the IR reaches the GPU");
    expect(path.querySelector(".gpu-chip")).toBeNull();
    expect(path.querySelectorAll(".gpu-step")).toHaveLength(0);
  });
});
