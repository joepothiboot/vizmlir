import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { MlirEngine } from "../../src/ir/bridge.js";
import { copySnapshot } from "../../src/ir/diff.js";

// The wasm is built by `npm run wasm` and is not committed.
const wasmPath = new URL("../../public/mlir_core.wasm", import.meta.url);
const built = existsSync(wasmPath);

async function engine() {
  const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
  return new MlirEngine(instance);
}

const SOURCE = [
  "func.func @f(%a: index) {",
  '  %b = arith.addi %a, %a : index loc(#loc2)',
  "  scf.for %i = %a to %a step %a {",
  "    scf.yield",
  '  } loc("k.mojo":5:3)',
  "  return",
  "}",
  '#loc1 = loc("k.mojo":7:3)',
  '#loc2 = loc(callsite(#loc1 at "main.mojo":1:1))',
].join("\n");

describe.skipIf(!built)("source locations through the wasm bridge", () => {
  it("reads each op's location, resolving aliases defined after use", async () => {
    const mlir = await engine();
    mlir.parse(SOURCE);
    const snap = mlir.snapshot();
    const byLabel = (prefix) =>
      [...Array(snap.nodeCount).keys()].find((i) => snap.labelOf(i).startsWith(prefix));

    const add = snap.locOf(byLabel("arith.addi"));
    expect(add).toMatchObject({ file: "k.mojo", line: 7, col: 3, callsite: true });
    expect(add.text).toBe('callsite("k.mojo":7:3 at "main.mojo":1:1)');

    expect(snap.locOf(byLabel("scf.for"))).toMatchObject({ file: "k.mojo", line: 5, col: 3 });
    expect(snap.locOf(byLabel("scf.yield"))).toBeNull();
    expect(snap.locOf(0)).toBeNull();
  });

  it("gives each op its line in the printed IR", async () => {
    const mlir = await engine();
    mlir.parse(SOURCE);
    const snap = mlir.snapshot();
    const lines = [...Array(snap.nodeCount).keys()].map((i) => [snap.labelOf(i), snap.irLineOf(i)]);
    expect(lines).toContainEqual(["arith.addi", 2]);
    expect(lines).toContainEqual(["scf.for", 3]);
    expect(lines).toContainEqual(["scf.yield", 4]);
  });

  it("reports no locations for IR printed without debug info", async () => {
    const mlir = await engine();
    mlir.parse("func.func @f() {\n  return\n}\n");
    const snap = mlir.snapshot();
    for (let i = 0; i < snap.nodeCount; i++) expect(snap.locOf(i)).toBeNull();
  });

  it("keeps locations in a copied snapshot", async () => {
    const mlir = await engine();
    mlir.parse(SOURCE);
    const copy = copySnapshot(mlir.snapshot());
    mlir.parse("func.func @g() {\n  return\n}\n");
    expect(copy.nodes.some((n) => n.loc?.line === 7 && n.line === 2)).toBe(true);
  });
});
