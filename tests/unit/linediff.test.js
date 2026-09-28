import { describe, expect, it } from "vitest";
import { diffStats, lineDiff } from "../../src/linediff.js";

const render = (lines) =>
  lines.map(({ type, text }) => `${{ same: " ", add: "+", del: "-" }[type]}${text}`);

describe("lineDiff", () => {
  it("marks added, removed and kept lines in order", () => {
    expect(render(lineDiff("a\nb\nc\nd", "a\nc\nx\nd"))).toEqual([
      " a",
      "-b",
      " c",
      "+x",
      " d",
    ]);
  });

  it("treats null as empty", () => {
    expect(render(lineDiff(null, "a\nb"))).toEqual(["+a", "+b"]);
    expect(render(lineDiff("a", null))).toEqual(["-a"]);
    expect(lineDiff(null, null)).toEqual([]);
  });

  it("reports identical text as unchanged", () => {
    const diff = lineDiff("x\ny", "x\ny");
    expect(diffStats(diff)).toEqual({ added: 0, removed: 0 });
  });

  it("falls back to remove-then-add for very large changes", () => {
    const before = Array.from({ length: 2100 }, (_, i) => `a${i}`).join("\n");
    const after = Array.from({ length: 2100 }, (_, i) => `b${i}`).join("\n");
    const diff = lineDiff(`head\n${before}\ntail`, `head\n${after}\ntail`);
    expect(diff[0]).toEqual({ type: "same", text: "head" });
    expect(diff.at(-1)).toEqual({ type: "same", text: "tail" });
    expect(diffStats(diff)).toEqual({ added: 2100, removed: 2100 });
  });
});
