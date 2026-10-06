// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { URL as FileURL } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { parseNcuCounters } from "../../src/bench.js";
import { analyzeGpu } from "../../src/gpu/model.js";
import { renderGpuView } from "../../src/render/gpu-view.js";

const read = (path) =>
  readFileSync(new FileURL(`../../${path}`, import.meta.url), "utf8");

const counters = parseNcuCounters(
  read("public/samples/gpu-patterns.t4.ncu.csv"),
);

globalThis.ResizeObserver ??= class {
  observe() {}
  disconnect() {}
};

window.requestAnimationFrame = () => 0;

function measuredLines(ir, measured) {
  const view = document.createElement("div");
  document.body.replaceChildren(view);

  const onAnswer = vi.fn();
  renderGpuView(view, analyzeGpu(ir), { onAnswer, measured: () => measured });

  const out = [];

  for (const row of view.querySelectorAll(".gpu-access-table tbody tr")) {
    row.click();

    const [card, judged] = onAnswer.mock.calls.at(-1);

    out.push([
      judged.access.line,
      card.querySelector(".gpu-measured")?.textContent ?? "",
    ]);
  }

  return out;
}

describe("measured line", () => {
  const measured = { device: "Tesla T4", counters };

  it("shows the T4 counter beside every access's prediction, all agreeing", () => {
    const lines = measuredLines(
      read("samples/input/gpu-patterns.mlir"),
      measured,
    );

    const byLine = Object.fromEntries(lines);
    expect(lines.length).toBeGreaterThan(10);

    for (const [, text] of lines) {
      expect(text).toMatch(/^✓ Measured on a Tesla T4: /);
    }

    expect(byLine[20]).toBe(
      "✓ Measured on a Tesla T4: 8 chunks of 32 bytes per warp request, as predicted.",
    );

    expect(byLine[52]).toBe(
      "✓ Measured on a Tesla T4: 4.5 chunks of 32 bytes per warp request, averaged over this kernel's 2 loads; predicted 4.5.",
    );

    expect(byLine[73]).toBe(
      "✓ Measured on a Tesla T4: 4.875 chunks of 32 bytes per warp request, as predicted.",
    );
  });

  it("shows the bank conflict and its fix as measured passes", () => {
    const byLine = Object.fromEntries(
      measuredLines(read("samples/input/gpu-transpose.mlir"), measured),
    );

    expect(byLine[37]).toBe(
      "✓ Measured on a Tesla T4: 32 passes per warp, as predicted.",
    );

    expect(byLine[59]).toBe(
      "✓ Measured on a Tesla T4: 1 pass per warp, as predicted.",
    );
  });

  it("marks a counter that disagrees, and says nothing without measurements", () => {
    const wrong = new Map(counters);

    wrong.set("aos_x_kernel", {
      ...counters.get("aos_x_kernel"),
      globalLoad: 4,
    });

    const ir = read("samples/input/gpu-patterns.mlir");

    expect(
      Object.fromEntries(
        measuredLines(ir, { device: "Tesla T4", counters: wrong }),
      )[20],
    ).toBe(
      "✗ Measured on a Tesla T4: 4 chunks of 32 bytes per warp request; predicted 8.",
    );

    for (const [, text] of measuredLines(ir, null)) expect(text).toBe("");
  });
});
