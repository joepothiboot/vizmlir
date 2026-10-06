import { describe, expect, it } from "vitest";
import {
  approach,
  depths,
  layoutLayer,
  layoutStack,
  lineageSegments,
  neighbours,
  nodePoint,
  nodesOnLine,
  pointAlong,
  snapshotShape,
} from "../../src/trace/descent.js";

const shape = {
  parent: [-1, 0, 1, 1],
  kind: [0, 1, 2, 2],
  line: [0, 1, 2, 3],
  edges: [
    [0, 1],
    [1, 2],
    [2, 3],
  ],
};

describe("snapshotShape", () => {
  it("copies parents, kinds, printed lines and edges", () => {
    const snapshot = {
      nodeCount: 3,
      edgeCount: 2,
      parentOf: (i) => [-1, 0, 1][i],
      kindOf: (i) => [0, 1, 2][i],
      irLineOf: (i) => [0, 1, 2][i],
      edges: new Uint32Array([0, 1, 1, 2]),
    };

    expect(snapshotShape(snapshot)).toEqual({
      parent: [-1, 0, 1],
      kind: [0, 1, 2],
      line: [0, 1, 2],
      edges: [
        [0, 1],
        [1, 2],
      ],
    });
  });

  it("copes with a snapshot that has no line table", () => {
    const shape = snapshotShape({
      nodeCount: 1,
      edgeCount: 0,
      parentOf: () => -1,
      kindOf: () => 0,
      edges: [],
    });

    expect(shape.line).toEqual([0]);
  });
});

describe("layout", () => {
  it("puts a column at each nesting depth and stacks siblings in a row", () => {
    expect(depths(shape)).toEqual([0, 1, 2, 2]);

    const layer = layoutLayer(shape, { cellX: 2, cellZ: 1 });
    const [m, f, a, b] = layer.pos;
    expect(f[0] - m[0]).toBe(2);
    expect(a[0] - f[0]).toBe(2);
    expect(a[0]).toBe(b[0]);
    expect(b[1] - a[1]).toBe(1);
  });

  it("centres the layer on the origin", () => {
    const layer = layoutLayer(shape, { cellX: 2, cellZ: 1 });
    const xs = layer.pos.map((p) => p[0]);
    const zs = layer.pos.map((p) => p[1]);
    expect(Math.min(...xs) + Math.max(...xs)).toBeCloseTo(0);
    expect(Math.min(...zs) + Math.max(...zs)).toBeCloseTo(0);
  });

  it("draws only the first maxNodes nodes and says how many it left out", () => {
    const layer = layoutLayer(shape, { maxNodes: 2 });
    expect(layer.pos.filter(Boolean)).toHaveLength(2);
    expect(layer.pos[3]).toBeNull();
    expect(layer).toMatchObject({ shown: 2, hidden: 2 });
  });

  const wide = (n) => ({
    parent: [-1, 0, ...Array(n).fill(1)],
    kind: [0, 1, ...Array(n).fill(2)],
    line: Array(n + 2).fill(0),
    edges: [],
  });

  it("wraps a tall column into side-by-side sub-columns", () => {
    const layer = layoutLayer(wide(30), {
      cellX: 3,
      cellZ: 1,
      maxRows: 10,
      subWidth: 1,
    });

    const col = layer.pos.slice(2);
    expect(new Set(col.map((p) => p[0])).size).toBe(3);
    expect(layer.depth).toBe(9);
    expect(col[10][0] - col[0][0]).toBe(1);
    expect(col[10][1]).toBe(col[0][1]);
  });

  it("keeps the next depth clear of a wrapped column", () => {
    const shape = {
      parent: [-1, 0, ...Array(30).fill(1), 2],
      kind: [],
      line: [],
      edges: [],
    };

    const layer = layoutLayer(shape, {
      cellX: 3,
      cellZ: 1,
      maxRows: 10,
      subWidth: 1,
    });

    const widest = Math.max(...layer.pos.slice(2, 32).map((p) => p[0]));
    expect(layer.pos[32][0]).toBeGreaterThan(widest);
  });

  it("never lays out a plate wider or deeper than the limit", () => {
    const stack = layoutStack([wide(400)], { limit: 20 });
    const layer = stack.layers[0];
    expect(Math.max(layer.width, layer.depth)).toBeLessThanOrEqual(20 + 1e-9);
    expect(stack.shrink).toBeLessThan(1);
    expect(stack.plate.width).toBeLessThan(40);
  });

  it("leaves a modest module at its natural size", () => {
    expect(layoutStack([shape]).shrink).toBe(1);
  });

  it("stacks layers downward, one gap apart, on plates of one size", () => {
    const small = { parent: [-1], kind: [0], line: [0], edges: [] };
    const stack = layoutStack([small, shape, null], { gap: 10 });
    expect(stack.layers[2]).toBeNull();
    expect(stack.heightOf(0)).toBe(0);
    expect(stack.heightOf(2)).toBe(-20);
    expect(stack.plate.width).toBeGreaterThan(stack.layers[1].width);
    expect(nodePoint(stack, 1, 2)[1]).toBe(-10);
    expect(nodePoint(stack, 2, 0)).toBeNull();
    expect(nodePoint(stack, 0, 5)).toBeNull();
  });
});

describe("selection helpers", () => {
  it("finds what feeds a node and what it feeds", () => {
    expect(neighbours(shape, 2)).toEqual({ inputs: [1], outputs: [3] });
    expect(neighbours(null, 2)).toEqual({ inputs: [], outputs: [] });
  });

  it("finds the nodes printed on a line", () => {
    expect(nodesOnLine(shape, 3)).toEqual([3]);
    expect(nodesOnLine(shape, 0)).toEqual([]);
    expect(nodesOnLine(null, 3)).toEqual([]);
  });
});

describe("lineageSegments", () => {
  it("joins each entry to what it came from", () => {
    const entries = [
      { pass: 0, node: 4, change: "created", from: [] },
      { pass: 1, node: 5, change: "kept", from: [{ pass: 0, node: 4 }] },
      {
        pass: 2,
        node: 7,
        change: "fused",
        from: [
          { pass: 1, node: 5 },
          { pass: 1, node: 6 },
        ],
      },
      { pass: 3, node: -1, change: "removed", from: [] },
    ];

    expect(lineageSegments(entries)).toEqual([
      { from: { pass: 0, node: 4 }, to: { pass: 1, node: 5 }, kind: "kept" },
      { from: { pass: 1, node: 5 }, to: { pass: 2, node: 7 }, kind: "fused" },
      { from: { pass: 1, node: 6 }, to: { pass: 2, node: 7 }, kind: "fused" },
    ]);
  });

  it("starts an inlined op at its callee", () => {
    const [segment] = lineageSegments([
      { pass: 1, node: 9, change: "inlined", from: [{ pass: 0, node: 2 }] },
    ]);

    expect(segment).toMatchObject({
      from: { pass: 0, node: 2 },
      kind: "inlined",
    });
  });

  it("is empty for no history", () => {
    expect(lineageSegments([])).toEqual([]);
  });
});

describe("pointAlong", () => {
  const path = [
    [0, 0, 0],
    [10, 0, 0],
    [10, 10, 0],
  ];

  it("walks the polyline by length", () => {
    expect(pointAlong(path, 0)).toEqual([0, 0, 0]);
    expect(pointAlong(path, 0.25)).toEqual([5, 0, 0]);
    expect(pointAlong(path, 0.5)).toEqual([10, 0, 0]);
    expect(pointAlong(path, 0.75)).toEqual([10, 5, 0]);
    expect(pointAlong(path, 1)).toEqual([10, 10, 0]);
  });

  it("clamps, and copes with short or flat paths", () => {
    expect(pointAlong(path, -1)).toEqual([0, 0, 0]);
    expect(pointAlong(path, 2)).toEqual([10, 10, 0]);
    expect(pointAlong([], 0.5)).toBeNull();
    expect(pointAlong([[1, 2, 3]], 0.5)).toEqual([1, 2, 3]);

    expect(
      pointAlong(
        [
          [1, 1, 1],
          [1, 1, 1],
        ],
        0.5,
      ),
    ).toEqual([1, 1, 1]);
  });
});

describe("approach", () => {
  it("covers half the distance per half-life", () => {
    expect(approach(0, 10, 140, 140)).toBeCloseTo(5);
    expect(approach(0, 10, 280, 140)).toBeCloseTo(7.5);
  });

  it("does not depend on how the time is split into frames", () => {
    let a = 0;
    for (let i = 0; i < 4; i++) a = approach(a, 10, 35, 140);
    expect(a).toBeCloseTo(approach(0, 10, 140, 140), 5);
  });

  it("snaps when close, and stays put at the target", () => {
    expect(approach(9.995, 10, 16)).toBe(10);
    expect(approach(9.5, 10, 1)).not.toBe(10);
    expect(approach(10, 10, 16)).toBe(10);
  });
});
