import { describe, expect, it } from "vitest";
import { describeType, explainLine, findDefinition } from "../../src/anatomy.js";

const IR = [
  "func.func @f(%a: memref<64x64xf32>, %n: index) {", // 0
  "  %c0 = arith.constant 0 : index", // 1
  "  %c32 = arith.constant 32 : index", // 2
  "  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c32, %gy = %c32, %gz = %c32) threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c32, %sz = %c32) {", // 3
  "    %i = arith.muli %bx, %c32 : index", // 4
  "    %v = memref.load %a[%i, %tx] : memref<64x64xf32>", // 5
  "    memref.store %v, %a[%tx, %i] : memref<64x64xf32, #gpu.address_space<workgroup>>", // 6
  "    %s = scf.for %k = %c0 to %n step %c32 iter_args(%acc = %v) -> (f32) {", // 7
  "      scf.yield %acc : f32", // 8
  "    }", // 9
  "    gpu.terminator", // 10
  "  }", // 11
  "  return", // 12
  "}", // 13
];

const parts = (at) =>
  explainLine(IR, at).parts.map((p) => [p.kind, p.text, p.ref ?? null]);

describe("explainLine", () => {
  it("names each part of a load and links inputs to their definitions", () => {
    const explained = explainLine(IR, 5);
    expect(explained.op).toBe("memref.load");
    expect(explained.summary).toBe("Reads one item from an array in memory.");
    expect(explained.docs).toBe("https://mlir.llvm.org/docs/Dialects/MemRef/");
    expect(parts(5)).toEqual([
      ["result", "%v", null],
      ["op", "memref.load", null],
      ["input", "%a", 0],
      ["position", "[%i, %tx]", null],
      ["input", "%i", 4],
      ["input", "%tx", 3],
      ["type", ": memref<64x64xf32>", null],
    ]);
    const details = Object.fromEntries(explained.parts.map((p) => [p.text, p.detail]));
    expect(details["%a"]).toBe("%a is an input to the function on line 1.");
    expect(details["%tx"]).toBe("%tx is a thread or block number from the launch on line 4.");
    expect(details["[%i, %tx]"]).toBe("Which item: row %i, column %tx.");
    expect(details[": memref<64x64xf32>"]).toBe(
      "A 64 × 64 array of 32-bit decimal numbers in memory (16.0 KB).",
    );
  });

  it("says a constant's value and explains its type", () => {
    const explained = explainLine(IR, 2);
    expect(explained.summary).toBe("Makes the fixed value 32.");
    expect(explained.parts.at(-1).detail).toBe(
      "An index: a whole number used for positions and sizes.",
    );
  });

  it("explains what a gpu.launch binds, axis by axis", () => {
    const details = explainLine(IR, 3)
      .parts.filter((p) => p.kind === "binding")
      .map((p) => [p.text, p.detail]);
    expect(details).toContainEqual(["%by", "Inside the launch, each block's own y number."]);
    expect(details).toContainEqual(["%tx", "Inside the launch, each thread's own x number."]);
    expect(details).toContainEqual(["%gz", "Inside the launch, how many blocks there are along z."]);
    expect(explainLine(IR, 3).parts.at(-1)).toMatchObject({ kind: "region", text: "{" });
  });

  it("explains a loop's counter and carried value", () => {
    const bindings = Object.fromEntries(
      explainLine(IR, 7).parts.filter((p) => p.kind === "binding").map((p) => [p.text, p.detail]),
    );
    expect(bindings["%k"]).toMatch(/loop counter/);
    expect(bindings["%acc"]).toMatch(/carried from one trip around the loop/);
    const inputs = explainLine(IR, 7).parts.filter((p) => p.kind === "input").map((p) => p.text);
    expect(inputs).toEqual(["%c0", "%n", "%c32", "%v"]);
  });

  it("names a function and its inputs", () => {
    expect(parts(0)).toEqual([
      ["op", "func.func", null],
      ["symbol", "@f", null],
      ["binding", "%a", null],
      ["binding", "%n", null],
      ["region", "{", null],
    ]);
  });

  it("marks shared memory in a type", () => {
    expect(explainLine(IR, 6).parts.at(-1).detail).toBe(
      "A 64 × 64 array of 32-bit decimal numbers in shared memory (16.0 KB).",
    );
  });

  it("explains closing braces, blank lines and unknown ops", () => {
    expect(explainLine(IR, 9).summary).toMatch(/^Closes the block/);
    expect(explainLine(["", "x"], 0)).toBe(null);
    const unknown = explainLine(["%r = mydialect.frob %x : i32"], 0);
    expect(unknown.summary).toBe(null);
    expect(unknown.docs).toBe("https://mlir.llvm.org/docs/Dialects/");
    expect(unknown.parts.map((p) => p.kind)).toEqual(["result", "op", "input", "type"]);
  });
});

describe("describeType", () => {
  it.each([
    ["i32", "a 32-bit whole number"],
    ["f16", "a 16-bit decimal number"],
    ["i1", "true or false"],
    ["tensor<128x64xf32>", "a 128 × 64 tensor (an array treated as a value) of 32-bit decimal numbers"],
    ["vector<4xf32>", "a vector of 4 32-bit decimal numbers, processed together"],
    ["memref<?x8xindex>", "a ? × 8 array of indices in memory"],
    ["!llvm.ptr", "a pointer: the address of something in memory"],
    ["(i32, f32) -> f32", "a function type: takes 2 inputs and gives 1 result"],
    ["!mystery.type", null],
  ])("%s", (type, text) => {
    expect(describeType(type)).toBe(text);
  });
});

describe("findDefinition", () => {
  it("prefers the nearest definition above", () => {
    const lines = ["%x = a.b", "func.func @g() {", "  %x = c.d", "  e.f %x"];
    expect(findDefinition(lines, "%x", 3)).toEqual({ line: 2, how: "result" });
  });

  it("finds loop counters and launch ids", () => {
    expect(findDefinition(IR, "%k", 8)).toEqual({ line: 7, how: "loop" });
    expect(findDefinition(IR, "%acc", 8)).toEqual({ line: 7, how: "loop-carried" });
    expect(findDefinition(IR, "%bx", 4)).toEqual({ line: 3, how: "launch" });
  });
});
