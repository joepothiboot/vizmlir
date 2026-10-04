import { describe, expect, it } from "vitest";
import { copySnapshot, diffSnapshots, normalizeLabel } from "../../src/ir/diff.js";
import { KIND } from "../../src/ir/abi.js";

// Builds an object shaped like MlirEngine#snapshot() from [kind, label, parent].
function liveSnapshot(nodes) {
  return {
    nodeCount: nodes.length,
    kindOf: (i) => nodes[i][0],
    labelOf: (i) => nodes[i][1],
    parentOf: (i) => nodes[i][2],
  };
}

const module = [KIND.MODULE, "module", -1];
const func = [KIND.FUNC, "func.func @f", 0];

function summary(rows) {
  return rows.map((row) => {
    if (row.type === "added") return `+ ${row.after.label}`;
    if (row.type === "removed") return `- ${row.before.label}`;
    return `~ ${row.before.label} -> ${row.after.label}`;
  });
}

describe("normalizeLabel", () => {
  it.each([
    ["%0", "%?"],
    ["%arg0", "%?"],
    ["%a.b", "%?"],
    ["%-x", "%?"],
    ["%x$1", "%?"],
    ["op %a, %b", "op %?, %?"],
    ["func.func @f", "func.func @f"],
  ])("%s -> %s", (label, expected) => {
    expect(normalizeLabel(label)).toBe(expected);
  });
});

describe("copySnapshot", () => {
  it("copies kind, parent and label of every node", () => {
    const snap = liveSnapshot([module, func, [KIND.OP, "arith.addi", 1]]);
    expect(copySnapshot(snap)).toEqual({
      nodeCount: 3,
      nodes: [
        { kind: KIND.MODULE, parent: -1, label: "module" },
        { kind: KIND.FUNC, parent: 0, label: "func.func @f" },
        { kind: KIND.OP, parent: 1, label: "arith.addi" },
      ],
    });
  });
});

describe("diffSnapshots", () => {
  const base = [
    module,
    func,
    [KIND.OP, "arith.constant", 1],
    [KIND.OP, "linalg.fill", 1],
    [KIND.TERMINATOR, "func.return", 1],
  ];

  it("reports nothing for identical snapshots", () => {
    expect(diffSnapshots(liveSnapshot(base), liveSnapshot(base))).toEqual([]);
  });

  it("treats a missing baseline as everything added", () => {
    const rows = diffSnapshots(null, liveSnapshot(base));
    expect(rows.map((row) => row.type)).toEqual(Array(5).fill("added"));
  });

  it("reports a renamed op under the same parent as changed", () => {
    const after = base.map((node) =>
      node[1] === "linalg.fill" ? [KIND.OP, "linalg.fill_relu", 1] : node,
    );
    expect(
      summary(diffSnapshots(liveSnapshot(base), liveSnapshot(after))),
    ).toEqual(["~ linalg.fill -> linalg.fill_relu"]);
  });

  it("prefers an exact match over a kind+parent match", () => {
    const before = [module, func, [KIND.OP, "a.x", 1], [KIND.OP, "a.y", 1]];
    const after = [module, func, [KIND.OP, "a.y", 1]];
    expect(
      summary(diffSnapshots(liveSnapshot(before), liveSnapshot(after))),
    ).toEqual(["- a.x"]);
  });

  it("pairs duplicate ops one-to-one", () => {
    const add = [KIND.OP, "arith.addi", 1];
    const before = [module, func, add, add];
    const after = [module, func, add, add, add];
    const rows = diffSnapshots(liveSnapshot(before), liveSnapshot(after));
    expect(summary(rows)).toEqual(["+ arith.addi"]);
    expect(rows[0].after.index).toBe(4);
  });

  it("reports an op of a new kind as added and the old one as removed", () => {
    const before = [module, func, [KIND.OP, "cf.op", 1]];
    const after = [module, func, [KIND.TERMINATOR, "cf.br", 1]];
    expect(
      summary(diffSnapshots(liveSnapshot(before), liveSnapshot(after))),
    ).toEqual(["+ cf.br", "- cf.op"]);
  });

  it("ignores SSA renumbering in labels", () => {
    const before = [module, [KIND.BLOCK, "^bb0 %0", 0]];
    const after = [module, [KIND.BLOCK, "^bb0 %arg7", 0]];
    expect(diffSnapshots(liveSnapshot(before), liveSnapshot(after))).toEqual(
      [],
    );
  });

  it("gives the same result for a copied baseline and a live one", () => {
    const after = [...base, [KIND.OP, "tensor.empty", 1]];
    const live = diffSnapshots(liveSnapshot(base), liveSnapshot(after));
    const copied = diffSnapshots(
      copySnapshot(liveSnapshot(base)),
      liveSnapshot(after),
    );
    expect(copied).toEqual(live);
  });

  // Current behaviour: "changed" pairing matches on the parent *index*, so
  // inserting a top-level op shifts later parents and a rename inside the
  // function shows up as add + remove instead of a change.
  it("pins: parent index shift turns a rename into add + remove", () => {
    const before = [module, func, [KIND.OP, "a.old", 1]];
    const after = [
      module,
      [KIND.OP, "memref.global @g", 0],
      [KIND.FUNC, "func.func @f", 0],
      [KIND.OP, "a.new", 2],
    ];
    expect(
      summary(diffSnapshots(liveSnapshot(before), liveSnapshot(after))),
    ).toEqual(["+ memref.global @g", "+ a.new", "- a.old"]);
  });

  it("stays fast on large modules", () => {
    const nodes = [module, func];
    for (let i = 0; i < 5000; i++) nodes.push([KIND.OP, `op.n${i}`, 1]);
    const reversed = [module, func, ...nodes.slice(2).reverse()];
    const t0 = performance.now();
    expect(diffSnapshots(liveSnapshot(nodes), liveSnapshot(reversed))).toEqual(
      [],
    );
    expect(performance.now() - t0).toBeLessThan(2000);
  });
});
