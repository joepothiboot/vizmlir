import { describe, expect, it } from "vitest";
import {
  buildLocIndex,
  findSource,
  linesOf,
  nodesAt,
  opCounts,
  sameFile,
  sourcePositions,
} from "../../src/trace/sources.js";

function snapshotOf(nodes) {
  return {
    nodeCount: nodes.length,
    locOf: (i) => (nodes[i][1] === null ? null : { text: nodes[i][1] }),
  };
}

describe("sourcePositions", () => {
  it("reads a plain location", () => {
    expect(sourcePositions('"k.mojo":12:5')).toEqual([
      { file: "k.mojo", line: 12, col: 5 },
    ]);
  });

  it("reads every position of a call site and of a fused op, in order", () => {
    expect(
      sourcePositions('callsite("k.mojo":4:14 at "main.mojo":10:16)'),
    ).toEqual([
      { file: "k.mojo", line: 4, col: 14 },
      { file: "main.mojo", line: 10, col: 16 },
    ]);

    expect(
      sourcePositions('fused["a.mojo":1:1, "a.mojo":2:2]').map((p) => p.line),
    ).toEqual([1, 2]);
  });

  it("finds nothing in unknown or name-only locations", () => {
    expect(sourcePositions("unknown")).toEqual([]);
    expect(sourcePositions('"just a name"')).toEqual([]);
  });

  it("keeps escaped quotes inside a file name", () => {
    expect(sourcePositions('"a\\"b.mojo":3:1')[0].file).toBe('a\\"b.mojo');
  });
});

describe("matching files", () => {
  it("matches equal paths, path tails and bare file names", () => {
    expect(sameFile("k.mojo", "k.mojo")).toBe(true);
    expect(sameFile("/work/src/k.mojo", "src/k.mojo")).toBe(true);
    expect(sameFile("/work/src/k.mojo", "k.mojo")).toBe(true);
    expect(sameFile("/work/src/k.mojo", "other.mojo")).toBe(false);
  });

  it("prefers an exact name when several loaded files would match", () => {
    const sources = { "a/k.mojo": "x", "k.mojo": "y" };
    expect(findSource(sources, "k.mojo")).toBe("k.mojo");
    expect(findSource(sources, "/abs/a/k.mojo")).toBe("a/k.mojo");
    expect(findSource(sources, "nope.mojo")).toBeNull();
  });
});

describe("location index", () => {
  const snap = snapshotOf([
    ["module", null],
    ["arith.mulf", '"k.mojo":4:14'],
    ["arith.addi", '"k.mojo":8:39'],
    ["arith.muli", '"k.mojo":8:25'],
    ["math.fma", 'fused["k.mojo":4:14, "k.mojo":10:31]'],
    ["return", "unknown"],
  ]);

  const index = buildLocIndex(snap);

  it("lists the files the IR names, once each", () => {
    expect(index.files).toEqual(["k.mojo"]);
  });

  it("finds the ops at a line, including every line of a fused op", () => {
    expect(nodesAt(index, "k.mojo", 4)).toEqual([1, 4]);
    expect(nodesAt(index, "k.mojo", 8)).toEqual([2, 3]);
    expect(nodesAt(index, "k.mojo", 10)).toEqual([4]);
    expect(nodesAt(index, "k.mojo", 99)).toEqual([]);
    expect(nodesAt(index, "/work/k.mojo", 8)).toEqual([2, 3]);
  });

  it("finds the lines an op came from", () => {
    expect(linesOf(index, 4)).toEqual({ file: "k.mojo", lines: [4, 10] });
    expect(linesOf(index, 1)).toEqual({ file: "k.mojo", lines: [4] });
    expect(linesOf(index, 0)).toBeNull();
    expect(linesOf(index, 5)).toBeNull();
  });

  it("counts the ops each line produced", () => {
    expect([...opCounts(index, "k.mojo")]).toEqual([
      [4, 2],
      [8, 2],
      [10, 1],
    ]);
  });

  it("is empty for IR without locations or a locOf", () => {
    expect(buildLocIndex(snapshotOf([["module", null]])).files).toEqual([]);
    expect(buildLocIndex({ nodeCount: 2 }).byNode.size).toBe(0);
  });
});
