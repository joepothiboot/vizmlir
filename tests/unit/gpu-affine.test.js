import { describe, expect, it } from "vitest";
import { buildDefs } from "../../src/gpu-access.js";
import { formatAffine, linearize } from "../../src/gpu-affine.js";

const env = { bdx: 32, bdy: 32, bdz: 1, gdx: 32, gdy: null, gdz: 1, args: new Map() };
const terms = (e) => Object.fromEntries(e.t);

describe("linearize", () => {
  const defs = buildDefs([
    "%c0 = arith.constant 0 : index",
    "%c4 = arith.constant 4 : index",
    "%c32 = arith.constant 32 : index",
    "%c128 = arith.constant 128 : index",
    "%tx = gpu.thread_id x",
    "%bx = gpu.block_id x",
    "%bd = gpu.block_dim x",
    "%gd = gpu.grid_dim y",
    "%0 = arith.muli %bx, %bd : index",
    "%1 = arith.addi %0, %tx : index",
    "%2 = arith.shli %tx, %c4 : index",
    "%3 = arith.subi %1, %tx : index",
    "scf.for %i = %c0 to %c128 step %c32 {",
    "  %4 = arith.addi %i, %tx : index",
    "}",
    "%sq = arith.muli %tx, %tx : index",
    "%r = arith.remui %tx, %c4 : index",
    "%f = arith.remui %c128, %c32 : index",
    "%g = arith.muli %gd, %tx : index",
  ]);

  it("folds sizes and constants into coefficients of the ids", () => {
    expect(linearize("%1", defs, env)).toEqual({ c: 0, t: new Map([["bx", 32], ["tx", 1]]) });
    expect(terms(linearize("%2", defs, env))).toEqual({ tx: 16 });
    // Terms that cancel disappear.
    expect(terms(linearize("%3", defs, env))).toEqual({ bx: 32 });
    expect(linearize("%f", defs, env).c).toBe(0);
  });

  it("writes a loop variable as lower bound + step · iteration, and records the trip count", () => {
    const ctx = {};
    expect(terms(linearize("%4", defs, env, ctx))).toEqual({ "iter:%i": 32, tx: 1 });
    expect(ctx.loops).toEqual(new Map([["iter:%i", 4]]));
  });

  it("stops at what is not affine and names it", () => {
    for (const [name, stuck] of [
      ["%sq", "%sq"],
      ["%r", "%r"],
      ["%g", "%gd"],
    ]) {
      const ctx = {};
      expect(linearize(name, defs, env, ctx)).toBeNull();
      expect(ctx.stuck).toBe(stuck);
    }
  });

  it("keeps an unknown kernel argument as a variable", () => {
    const args = new Map([["%n", null], ["%k", 3]]);
    expect(terms(linearize("%n", defs, { ...env, args }))).toEqual({ "arg:%n": 1 });
    expect(linearize("%k", defs, { ...env, args }).c).toBe(3);
  });
});

describe("formatAffine", () => {
  const defs = buildDefs([
    "%c1 = arith.constant 1 : index",
    "%c2 = arith.constant 2 : index",
    "%c64 = arith.constant 64 : index",
    "%tx = gpu.thread_id x",
    "%by = gpu.block_id y",
    "scf.for %k = %c1 to %c64 step %c2 {",
    "  %0 = arith.muli %by, %c64 : index",
    "  %1 = arith.addi %0, %k : index",
    "  %2 = arith.subi %1, %tx : index",
    "}",
  ]);

  it("prints largest coefficients first and loop variables by name", () => {
    const ctx = {};
    const e = linearize("%2", defs, env, ctx);
    // %k = 1 + 2·iter, written back as %k.
    expect(formatAffine(e, ctx.ivs)).toBe("64·by + %k - tx");
  });

  it("prints a constant alone", () => {
    expect(formatAffine({ c: 0, t: new Map() })).toBe("0");
  });
});
