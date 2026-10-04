// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { URL as FileURL } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { analyzeLocalMemory, localLineInfo, localTargetOf } from "../../src/gpu/local-memory.js";
import { focusLocalLine, localExplain, renderLocalView } from "../../src/render/local-view.js";
import { parsePassTrace } from "../../src/trace/trace.js";

const events = parsePassTrace(
  readFileSync(
    new FileURL("../../public/samples/nanodsp-local-matmul.trace.txt", import.meta.url),
    "utf8",
  ),
).events;
const analyze = (argument) => {
  const index = events.findIndex((e) => e.argument === argument);
  return analyzeLocalMemory(events[index].ir, { target: localTargetOf(events, index) });
};

describe("renderLocalView", () => {
  it("draws the slots, the schedule and an honest note, all as buttons that pick lines", () => {
    const root = document.createElement("div");
    const onPick = vi.fn();
    const analysis = analyze("nanodsp-promote-local");
    renderLocalView(root, analysis, { onPick });

    const blocks = [...root.querySelectorAll(".lm-block")];
    expect(blocks.map((b) => b.textContent)).toEqual(["A slot 0", "A slot 1", "B slot 0", "B slot 1"]);
    // Four 64 KiB slots fill the 256 KiB budget.
    expect(blocks.every((b) => b.style.width === "25%")).toBe(true);
    expect(root.querySelector(".lm-free")).toBe(null);
    expect(root.textContent).toContain("assumed, from nano-dsp TargetModel");
    expect(root.textContent).toContain("not measured timing");

    const prefetch = root.querySelectorAll(".lm-prefetch");
    expect([...prefetch].map((b) => b.textContent)).toEqual([
      "A1 → slot 1 · next",
      "B1 → slot 1 · next",
    ]);
    expect(root.querySelectorAll(".lm-prologue")).toHaveLength(2);
    expect(root.querySelectorAll(".lm-wait")).toHaveLength(4);
    expect([...root.querySelectorAll(".lm-compute")].map((b) => b.textContent)).toEqual([
      "tile 0 · A0 B0",
      "tile 1 · A1 B1",
    ]);

    // Every interactive element is a labelled, focusable button.
    for (const button of root.querySelectorAll("[data-line]")) {
      expect(button.tagName).toBe("BUTTON");
      expect(button.getAttribute("aria-label")).toBeTruthy();
    }

    const fn = analysis.functions[0];
    prefetch[0].click();
    expect(onPick).toHaveBeenCalledWith(fn.dmas[2].line, prefetch[0]);
    focusLocalLine(root, fn.dmas[2].line);
    expect(prefetch[0].getAttribute("aria-pressed")).toBe("true");
    expect(prefetch[1].getAttribute("aria-pressed")).toBe("false");
  });

  it("drops the wait lane and says the copies are synchronous after lower-local", () => {
    const root = document.createElement("div");
    renderLocalView(root, analyze("nanodsp-lower-local"));
    expect(root.querySelectorAll(".lm-wait")).toHaveLength(0);
    expect(root.querySelectorAll(".lm-prefetch")).toHaveLength(2);
    expect(root.textContent).toContain("synchronous");
    expect(root.textContent).toContain("was #dsp.local");
  });
});

describe("localExplain", () => {
  it("explains a prefetch and a buffer in plain words", () => {
    const analysis = analyze("nanodsp-promote-local");
    const fn = analysis.functions[0];
    const prefetch = localExplain(localLineInfo(analysis, fn.dmas[2].line));
    expect(prefetch.textContent).toMatch(/next trip/);
    const buffer = localExplain(localLineInfo(analysis, fn.localBuffers[0].line));
    expect(buffer.textContent).toMatch(/2 slots of 64\.0 KB/);
    expect(localExplain(null)).toBe(null);
  });
});
