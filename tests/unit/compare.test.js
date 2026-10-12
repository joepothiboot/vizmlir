import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  canonicalIr,
  compareTraces,
  rowDiff,
} from "../../src/trace/compare.js";
import { parsePassTrace } from "../../src/trace/trace.js";

const fixture = (name) =>
  parsePassTrace(readFileSync(`tests/fixtures/traces/${name}.txt`, "utf8"));

const dump = (pass, body) =>
  `// -----// IR Dump After ${pass} //----- //\nmodule {\n${body}\n}\n`;

const trace = (...dumps) => parsePassTrace(dumps.join("\n"));

describe("canonicalIr", () => {
  it("renames values by first appearance", () => {
    expect(canonicalIr("%a = f %b\n%c = g %a, %b")).toBe(
      "%v0 = f %v1\n%v2 = g %v0, %v1",
    );
  });
});

describe("compareTraces", () => {
  it("reports identical logs", () => {
    const a = fixture("order-a");
    const result = compareTraces(a, fixture("order-a"));

    expect(result.firstDivergence).toBe(-1);
    expect(result.differing).toBe(0);
    expect(result.finalSame).toBe(true);
  });

  it("finds the first pass where the IR differs", () => {
    const a = fixture("order-a");
    const b = fixture("order-b");
    const result = compareTraces(a, b);

    expect(result.firstDivergence).toBe(0);
    expect(result.rows.map((row) => row.kind)).toEqual(["ir", "ir"]);
    expect(result.finalSame).toBe(false);
    expect(rowDiff(a, b, 0).some((line) => line.type === "add")).toBe(true);
  });

  it("ignores value numbering", () => {
    const a = trace(dump("CSEPass: cse", "  %0 = f %arg0"));
    const b = trace(dump("CSEPass: cse", "  %7 = f %arg0"));

    expect(compareTraces(a, b).firstDivergence).toBe(-1);
  });

  it("flags different pass sequences", () => {
    const a = trace(dump("CSEPass: cse", "  x"));
    const b = trace(dump("CanonicalizerPass: canonicalize", "  x"));

    expect(compareTraces(a, b).rows[0].kind).toBe("pass");
  });

  it("flags a log that is shorter", () => {
    const one = dump("CSEPass: cse", "  x");
    const result = compareTraces(trace(one), trace(one, one));

    expect(result.rows.map((row) => row.kind)).toEqual(["same", "length"]);
    expect(result.firstDivergence).toBe(1);
  });
});
