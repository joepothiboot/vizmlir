import { describe, expect, it, vi } from "vitest";
import { conditionHits, mergeHits, nextHit, parseCondition } from "../../src/trace/breakpoints.js";

const cond = (text) => {
  const parsed = parseCondition(text);
  expect(parsed.ok, text).toBe(true);
  return parsed.cond;
};

// Passes as op-name -> count maps, the way countOps gives them.
const column = (obj) => (obj === null ? null : new Map(Object.entries(obj)));
const ctxOf = ({ counts = [], peaks = [], events } = {}) => ({
  events: events ?? (counts.length ? counts : peaks).map(() => ({})),
  counts: () => counts.map(column),
  peaks: () => peaks.map((p) => (p === null ? null : { peak: p })),
});

describe("parseCondition", () => {
  it("reads op counts, totals and sizes", () => {
    expect(cond("linalg.matmul == 0")).toMatchObject({ kind: "count", op: "linalg.matmul", cmp: "==", value: 0 });
    expect(cond("arith.addf>=3")).toMatchObject({ kind: "count", op: "arith.addf", cmp: ">=", value: 3 });
    expect(cond("ops < 20")).toMatchObject({ kind: "total", cmp: "<", value: 20 });
    expect(cond("live > 1MB")).toMatchObject({ kind: "live", cmp: ">", value: 1024 ** 2 });
    expect(cond("live > 1.5 kb").value).toBe(1536);
  });

  it("accepts a single = as ==", () => {
    expect(cond("func.call = 0").cmp).toBe("==");
  });

  it("reads the word forms", () => {
    expect(cond("appears gpu.launch")).toMatchObject({ kind: "appears", op: "gpu.launch" });
    expect(cond("gone  func.call")).toMatchObject({ kind: "gone", op: "func.call", text: "gone func.call" });
    expect(cond("fails").kind).toBe("fails");
    expect(cond("diag").kind).toBe("diag");
  });

  it("explains what it could not read", () => {
    for (const bad of ["", "   ", "matmul", "appears", "live > lots", "arith.addf > many", "ops < 1.5"]) {
      const result = parseCondition(bad);
      expect(result.ok, bad).toBe(false);
      expect(result.error).toBeTypeOf("string");
    }
  });
});

describe("conditionHits", () => {
  const counts = [
    { "linalg.matmul": 1, "func.call": 1 },
    { "linalg.matmul": 1 },
    { "linalg.matmul": 0, "gpu.launch": 1 },
    { "linalg.matmul": 0, "gpu.launch": 2 },
    null,
    { "linalg.matmul": 0, "gpu.launch": 2 },
  ];

  it("hits where a count first becomes true, not at every pass it holds", () => {
    expect(conditionHits(cond("linalg.matmul == 0"), ctxOf({ counts }))).toEqual([2, 5]);
  });

  it("treats a pass that did not parse as false", () => {
    expect(conditionHits(cond("gpu.launch >= 1"), ctxOf({ counts }))).toEqual([2, 5]);
  });

  it("compares a count that is zero when the op is absent", () => {
    expect(conditionHits(cond("gpu.launch == 0"), ctxOf({ counts }))).toEqual([0]);
  });

  it("totals every op", () => {
    // 2, 1, 1, 2, (unparsed), 2 ops: more than one at the start, and again from pass 3.
    expect(conditionHits(cond("ops > 1"), ctxOf({ counts }))).toEqual([0, 3, 5]);
  });

  it("finds ops appearing and disappearing", () => {
    expect(conditionHits(cond("appears gpu.launch"), ctxOf({ counts }))).toEqual([2]);
    expect(conditionHits(cond("gone func.call"), ctxOf({ counts }))).toEqual([1]);
    expect(conditionHits(cond("gone linalg.matmul"), ctxOf({ counts }))).toEqual([2]);
  });

  it("does not call an op appearing across a pass that did not parse", () => {
    expect(conditionHits(cond("appears gpu.launch"), ctxOf({ counts: [{}, null, { "gpu.launch": 1 }] }))).toEqual([]);
  });

  it("breaks on peak live bytes", () => {
    const peaks = [0, 512, 2048, 4096, null];
    expect(conditionHits(cond("live > 1KB"), ctxOf({ peaks }))).toEqual([2]);
  });

  it("breaks on failures and diagnostics", () => {
    const events = [{}, { failed: true }, { diagnostics: [{}] }, { diagnostics: [] }];
    expect(conditionHits(cond("fails"), ctxOf({ events }))).toEqual([1]);
    expect(conditionHits(cond("diag"), ctxOf({ events }))).toEqual([2]);
  });

  it("only computes the data a condition needs", () => {
    const ctx = ctxOf({ events: [{ failed: true }] });
    ctx.counts = vi.fn();
    ctx.peaks = vi.fn();
    conditionHits(cond("fails"), ctx);
    expect(ctx.counts).not.toHaveBeenCalled();
    expect(ctx.peaks).not.toHaveBeenCalled();
  });
});

describe("running between hits", () => {
  it("finds the next hit forward and back", () => {
    const hits = [2, 5, 9];
    expect(nextHit(hits, 0, 1)).toBe(2);
    expect(nextHit(hits, 2, 1)).toBe(5);
    expect(nextHit(hits, 9, 1)).toBe(-1);
    expect(nextHit(hits, 9, -1)).toBe(5);
    expect(nextHit(hits, 2, -1)).toBe(-1);
    expect(nextHit([], 3, 1)).toBe(-1);
  });

  it("merges hit lists without repeats", () => {
    expect(mergeHits([[5, 2], [2, 9], []])).toEqual([2, 5, 9]);
  });
});
