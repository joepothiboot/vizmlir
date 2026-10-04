import { describe, expect, it } from "vitest";
import { copySnapshot } from "../../src/ir/diff.js";
import {
  countOps,
  opCountsToCSV,
  opCountTable,
  opName,
  totalOps,
} from "../../src/trace/opcount.js";
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

const snapshot = liveSnapshot([
  [KIND.MODULE, "module", -1],
  [KIND.FUNC, "func.func @f", 0],
  [KIND.BLOCK, "^bb0", 1],
  [KIND.OP, "arith.constant", 1],
  [KIND.OP, "arith.addi", 1],
  [KIND.OP, "arith.addi", 1],
  [KIND.OP, "scf.for (2×)", 1],
  [KIND.TERMINATOR, "func.return", 1],
]);

describe("opName", () => {
  it.each([
    ["func.func @f", "func.func"],
    ["scf.for (2×)", "scf.for"],
    ["arith.addi", "arith.addi"],
  ])("%s -> %s", (label, expected) => {
    expect(opName(label)).toBe(expected);
  });
});

describe("countOps", () => {
  it("counts by op name, skipping the root and blocks", () => {
    expect(Object.fromEntries(countOps(snapshot))).toEqual({
      "func.func": 1,
      "arith.constant": 1,
      "arith.addi": 2,
      "scf.for": 1,
      "func.return": 1,
    });
  });

  it("reads copied snapshots the same way", () => {
    expect(countOps(copySnapshot(snapshot))).toEqual(countOps(snapshot));
  });

  it("returns nothing without a snapshot", () => {
    expect(totalOps(countOps(null))).toBe(0);
  });
});

describe("opCountTable", () => {
  const columns = [
    new Map([["arith.addi", 2], ["arith.constant", 1], ["func.return", 1]]),
    new Map([["arith.addi", 2], ["arith.constant", 1], ["func.return", 1]]),
    new Map([["arith.muli", 1], ["func.return", 1]]),
  ];

  it("sorts by the size of the net change and flags changed ops", () => {
    const { rows, totals, changedColumns } = opCountTable(columns);
    expect(rows.map((row) => [row.op, row.delta, row.changed])).toEqual([
      ["arith.addi", -2, true],
      ["arith.constant", -1, true],
      ["arith.muli", 1, true],
      ["func.return", 0, false],
    ]);
    expect(rows[0].counts).toEqual([2, 2, 0]);
    expect(totals).toEqual([4, 4, 2]);
    expect(changedColumns).toEqual([2]);
  });

  it("flags an op that changes and changes back", () => {
    const { rows } = opCountTable([
      new Map([["a", 1]]),
      new Map([["a", 3]]),
      new Map([["a", 1]]),
    ]);
    expect(rows[0]).toMatchObject({ delta: 0, changed: true });
  });

  it("writes CSV with a totals row", () => {
    const csv = opCountsToCSV(["#1", "#2", "#3"], opCountTable(columns));
    expect(csv.split("\n").slice(0, 3)).toEqual([
      "op,#1,#2,#3,delta",
      "(all ops),4,4,2,-2",
      "arith.addi,2,2,0,-2",
    ]);
  });
});
