import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { MlirEngine } from "../../src/ir/bridge.js";
import { buildOpModel, matchPasses, opHistory, opRecords } from "../../src/trace/ophistory.js";
import { parsePassTrace } from "../../src/trace/trace.js";

// [label, loc text] -> a record, the way opRecords would copy it.
const rec = (label, loc = "") => ({
  label,
  op: label.split(" ")[0],
  loc,
  keys: [...loc.matchAll(/"([^"]*)":(\d+):(\d+)/g)].map((m) => `${m[1]}:${m[2]}:${m[3]}`),
});
const L = (line, col = 1) => `"k.mojo":${line}:${col}`;

describe("matchPasses", () => {
  it("keeps ops with the same location and label", () => {
    const links = matchPasses([rec("a.op", L(1)), rec("b.op", L(2))], [rec("b.op", L(2)), rec("a.op", L(1))]);
    expect(links.get(0)).toMatchObject({ from: [1], kind: "kept" });
    expect(links.get(1)).toMatchObject({ from: [0], kind: "kept" });
  });

  it("calls a different op at the same location a lowering", () => {
    const links = matchPasses([rec("gpu.thread_id x", L(8, 41))], [rec("nvvm.tid", L(8, 41))]);
    expect(links.get(0)).toMatchObject({ from: [0], kind: "renamed" });
  });

  it("links an inlined op to its callee, which stays", () => {
    const before = [rec("arith.mulf", L(4, 14)), rec("func.call @s", L(10, 16))];
    const after = [rec("arith.mulf", L(4, 14)), rec("arith.mulf", `callsite(${L(4, 14)} at ${L(10, 16)})`)];
    const links = matchPasses(before, after);
    expect(links.get(0)).toMatchObject({ from: [0], kind: "kept" });
    expect(links.get(1)).toMatchObject({ from: [0], kind: "inlined", clone: true });
    expect(links.has(2)).toBe(false);
  });

  it("links a fused op to every op it merged", () => {
    const before = [rec("arith.mulf", L(4)), rec("arith.addf", L(10)), rec("memref.load", L(11))];
    const after = [rec("math.fma", `fused[${L(4)}, ${L(10)}]`), rec("memref.load", L(11))];
    const links = matchPasses(before, after);
    expect(links.get(0)).toMatchObject({ from: [0, 1], kind: "fused" });
    expect(links.get(1)).toMatchObject({ from: [2], kind: "kept" });
  });

  it("does not take a loaded op for a fused part twice", () => {
    const before = [rec("a", L(4)), rec("a", L(4))];
    const after = [rec("a", L(4)), rec("f", `fused[${L(4)}, ${L(5)}]`)];
    const links = matchPasses(before, after);
    expect(links.get(0).kind).toBe("kept");
    expect(links.get(1)).toMatchObject({ kind: "fused", from: [1] });
  });

  it("falls back to name and order without locations", () => {
    const before = [rec("arith.constant"), rec("arith.addi"), rec("arith.constant")];
    const after = [rec("arith.constant"), rec("arith.constant")];
    const links = matchPasses(before, after);
    expect(links.get(0)).toMatchObject({ from: [0], kind: "kept" });
    expect(links.get(1)).toMatchObject({ from: [2], kind: "kept" });
  });
});

describe("opHistory", () => {
  const passes = [
    [rec("module"), rec("a.op", L(1)), rec("b.op", L(2))],
    [rec("module"), rec("a.op", L(1)), rec("b.op", L(2)), rec("c.op", L(3))],
    [rec("module"), rec("lowered.a", L(1)), rec("c.op", L(3))],
  ];
  const model = buildOpModel(passes);

  it("follows an op forward through a rename", () => {
    const history = opHistory(model, 0, 1);
    expect(history.map((e) => [e.pass, e.label, e.change])).toEqual([
      [0, "a.op", "created"],
      [1, "a.op", "kept"],
      [2, "lowered.a", "renamed"],
    ]);
  });

  it("is the same from any pass of the lifetime", () => {
    expect(opHistory(model, 2, 1)).toEqual(opHistory(model, 0, 1));
  });

  it("ends with the pass that removed the op", () => {
    const history = opHistory(model, 1, 2);
    expect(history.map((e) => [e.pass, e.change])).toEqual([
      [0, "created"],
      [1, "kept"],
      [2, "removed"],
    ]);
    expect(history.at(-1).node).toBe(-1);
  });

  it("starts at the pass that made an op", () => {
    expect(opHistory(model, 1, 3).map((e) => [e.pass, e.change])).toEqual([
      [1, "created"],
      [2, "kept"],
    ]);
  });

  it("returns nothing for a pass that did not parse or a missing node", () => {
    const broken = buildOpModel([passes[0], null, passes[2]]);
    expect(opHistory(broken, 1, 0)).toEqual([]);
    expect(opHistory(model, 0, 99)).toEqual([]);
  });

  it("does not guess across a pass that did not parse", () => {
    const broken = buildOpModel([passes[0], null, passes[2]]);
    expect(opHistory(broken, 0, 1).map((e) => e.change)).toEqual(["created"]);
  });
});

const wasmPath = new URL("../../public/mlir_core.wasm", import.meta.url);
describe.skipIf(!existsSync(wasmPath))("the saxpy sample through the real parser", () => {
  async function model() {
    const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
    const mlir = new MlirEngine(instance);
    const trace = parsePassTrace(
      readFileSync(new URL("../../public/samples/saxpy.trace.txt", import.meta.url), "utf8"),
    );
    const records = trace.events.map((event) => {
      mlir.parse(event.ir);
      return opRecords(mlir.snapshot());
    });
    return { records, model: buildOpModel(records) };
  }
  const find = (records, pass, op) => records[pass].findIndex((r) => r.op === op);

  it("follows a thread id through to its NVVM lowering", async () => {
    const { records, model: m } = await model();
    const history = opHistory(m, 0, find(records, 0, "gpu.thread_id"));
    expect(history.map((e) => e.change)).toEqual(["created", "kept", "kept", "kept", "renamed"]);
    expect(history.at(-1).label).toBe("nvvm.read.ptx.sreg.tid.x");
  });

  it("shows the call being replaced by the inlined body", async () => {
    const { records, model: m } = await model();
    const call = opHistory(m, 0, find(records, 0, "func.call"));
    expect(call.map((e) => e.change)).toEqual(["created", "removed"]);

    const inlined = records[1].findIndex((r) => r.op === "arith.mulf" && r.loc.startsWith("callsite"));
    const history = opHistory(m, 1, inlined);
    expect(history[0]).toMatchObject({ pass: 1, change: "inlined" });
    expect(history[0].from[0]).toMatchObject({ pass: 0, label: "arith.mulf" });
  });

  it("shows the inlined mulf and the addf fusing into one fma", async () => {
    const { records, model: m } = await model();
    const inlined = records[1].findIndex((r) => r.op === "arith.mulf" && r.loc.startsWith("callsite"));
    const addf = find(records, 1, "arith.addf");
    const viaMul = opHistory(m, 1, inlined);
    const viaAdd = opHistory(m, 1, addf);
    expect(viaMul.map((e) => [e.pass, e.change])).toEqual([[1, "inlined"], [2, "fused"], [3, "kept"], [4, "kept"]]);
    const fused = viaAdd.find((e) => e.pass === 2);
    expect(fused).toMatchObject({ label: "math.fma", change: "fused" });
    expect(fused.from.map((f) => f.label).sort()).toEqual(["arith.addf", "arith.mulf"]);
    // The addf has been there since the first dump.
    expect(viaAdd[0]).toMatchObject({ pass: 0, change: "created" });
  });

  it("removes @scale when symbol-dce runs", async () => {
    const { records, model: m } = await model();
    const scale = records[0].findIndex((r) => r.label === "func.func @scale");
    const history = opHistory(m, 0, scale);
    expect(history.at(-1)).toMatchObject({ pass: 3, change: "removed" });
  });
});
