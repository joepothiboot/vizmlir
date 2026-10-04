import { describe, expect, it } from "vitest";
import {
  changesAt,
  embeddedAssembly,
  findSymbolNode,
  historyToJSON,
  scanSymbols,
  symbolHistory,
  symbolTimeline,
  unescapeMlirString,
} from "../../src/trace/provenance.js";

describe("findSymbolNode", () => {
  // [label, parent] shaped like MlirEngine#snapshot().
  const snapshot = (nodes) => ({
    nodeCount: nodes.length,
    labelOf: (i) => nodes[i][0],
    parentOf: (i) => nodes[i][1],
  });
  const module = snapshot([
    ["module", -1],
    ["func.func @main", 0],
    ["gpu.launch_func @k::@k", 1],
    ["gpu.module @k", 0],
    ["gpu.func @k", 3],
    ["gpu.return", 4],
  ]);

  it("finds a nested kernel rather than its module", () => {
    expect(findSymbolNode(module, "@k::@k")).toBe(4);
    expect(findSymbolNode(module, "@k")).toBe(3);
    expect(findSymbolNode(module, "@main")).toBe(1);
  });

  it("finds a kernel in a dump of its module alone", () => {
    const nested = snapshot([
      ["gpu.module @k", -1],
      ["llvm.func @k", 0],
    ]);
    expect(findSymbolNode(nested, "@k::@k")).toBe(1);
  });

  it("returns -1 when the symbol is not drawn", () => {
    expect(findSymbolNode(module, "@other")).toBe(-1);
    expect(findSymbolNode(null, "@k")).toBe(-1);
  });
});
import { describeEvent, parsePassTrace } from "../../src/trace/trace.js";

const header = (pass, arg, anchor) =>
  `// -----// IR Dump After ${pass}: ${arg} (${anchor}) //----- //`;

// Outline a GPU kernel, lower it to NVVM, then serialize the module.
const TRACE = [
  header("CanonicalizerPass", "canonicalize", "'builtin.module' operation"),
  "module {",
  "  func.func @main(%arg0: memref<8xf32>) {",
  "    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c8, %sy = %c1, %sz = %c1) {",
  "      gpu.terminator",
  "    }",
  "    return",
  "  }",
  "  func.func private @helper()",
  "}",
  "",
  header("GpuKernelOutliningPass", "gpu-kernel-outlining", "'builtin.module' operation"),
  "module attributes {gpu.container_module} {",
  "  func.func @main(%arg0: memref<8xf32>) {",
  "    gpu.launch_func @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c8, %c1, %c1)",
  "    return",
  "  }",
  "  func.func private @helper()",
  "  gpu.module @main_kernel {",
  "    gpu.func @main_kernel() kernel {",
  "      gpu.return",
  "    }",
  "  }",
  "}",
  "",
  header("ConvertGpuOpsToNVVMOps", "convert-gpu-to-nvvm", "'gpu.module' operation: @main_kernel"),
  "gpu.module @main_kernel {",
  "  llvm.func @main_kernel() attributes {gpu.kernel, nvvm.kernel} {",
  "    llvm.return",
  "  }",
  "}",
  "",
  header("CSEPass", "cse", "'builtin.module' operation"),
  "module attributes {gpu.container_module} {",
  "  func.func @main(%arg0: memref<8xf32>) {",
  "    gpu.launch_func @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c8, %c1, %c1)",
  "    return",
  "  }",
  "  func.func private @helper()",
  "  gpu.module @main_kernel {",
  "    llvm.func @main_kernel() attributes {gpu.kernel, nvvm.kernel} {",
  "      llvm.return",
  "    }",
  "  }",
  "}",
  "",
  header("GpuModuleToBinaryPass", "gpu-module-to-binary", "'builtin.module' operation"),
  "module attributes {gpu.container_module} {",
  "  func.func @main(%arg0: memref<8xf32>) {",
  "    gpu.launch_func @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c8, %c1, %c1)",
  "    return",
  "  }",
  "  gpu.binary @main_kernel  [#gpu.object<#nvvm.target, \"BLOB\">]",
  "}",
  "",
].join("\n");

describe("scanSymbols", () => {
  it("finds defining ops with nested paths and skips uses", () => {
    const { events } = parsePassTrace(TRACE);
    const symbols = scanSymbols(events[1].ir);
    expect([...symbols.keys()]).toEqual([
      "@main",
      "@helper",
      "@main_kernel",
      "@main_kernel::@main_kernel",
    ]);
    expect(symbols.get("@main_kernel").op).toBe("gpu.module");
    expect(symbols.get("@main_kernel::@main_kernel").op).toBe("gpu.func");
  });

  it("keeps nested symbols out of their parent's text", () => {
    const { events } = parsePassTrace(TRACE);
    const symbols = scanSymbols(events[1].ir);
    expect(symbols.get("@main_kernel").text).toBe(
      "gpu.module @main_kernel {\n}",
    );
    expect(symbols.get("@helper").text).toBe("func.func private @helper()");
  });

  it("keeps block labels at the function's own indent inside it", () => {
    const ir = [
      "module {",
      "  func.func @f(%n: index) {",
      "    cf.br ^bb1",
      "  ^bb1:",
      "    return",
      "  }",
      "  func.func @g() {",
      "    return",
      "  }",
      "}",
    ].join("\n");
    expect(scanSymbols(ir).get("@f").full.split("\n")).toEqual([
      "func.func @f(%n: index) {",
      "  cf.br ^bb1",
      "^bb1:",
      "  return",
      "}",
    ]);
  });

  it("reads quoted symbols, visibility keywords and globals", () => {
    const symbols = scanSymbols(
      [
        "module {",
        '  memref.global "private" constant @cst : memref<2xf32> = dense<0.0>',
        "  llvm.mlir.global internal constant @str(\"hi\") : !llvm.array<2 x i8>",
        '  func.func @"odd name"() {',
        "    func.call @cst() : () -> ()",
        "    return",
        "  }",
        "}",
      ].join("\n"),
    );
    expect([...symbols.values()].map((s) => [s.path, s.op])).toEqual([
      ["@cst", "memref.global"],
      ["@str", "llvm.mlir.global"],
      ["@odd name", "func.func"],
    ]);
  });
});

describe("symbolHistory", () => {
  const { events } = parsePassTrace(TRACE);
  const history = symbolHistory(events);
  const byPath = Object.fromEntries(history.map((r) => [r.path, r]));
  const kinds = (path) =>
    byPath[path].changes.map((c) => [c.index, c.kind, c.from ?? c.op]);

  it("marks symbols present in the first dump as initial", () => {
    expect(byPath["@main"].initial).toBe(true);
    expect(byPath["@main_kernel"].initial).toBe(false);
  });

  it("attributes creation to the pass that outlined the kernel", () => {
    expect(kinds("@main_kernel")[0]).toEqual([1, "created", "gpu.module"]);
    expect(kinds("@main_kernel::@main_kernel")[0]).toEqual([
      1,
      "created",
      "gpu.func",
    ]);
  });

  it("follows a kernel through lowering and serialization", () => {
    expect(kinds("@main_kernel::@main_kernel")).toEqual([
      [1, "created", "gpu.func"],
      [2, "lowered", "gpu.func"],
      [4, "removed", "llvm.func"],
    ]);
    expect(byPath["@main_kernel::@main_kernel"].ops).toEqual([
      "gpu.func",
      "llvm.func",
    ]);
    expect(kinds("@main_kernel")).toEqual([
      [1, "created", "gpu.module"],
      [4, "lowered", "gpu.module"],
    ]);
    expect(byPath["@main_kernel"].op).toBe("gpu.binary");
  });

  it("does not report a pass that left a symbol alone", () => {
    expect(changesAt(history, 3)).toEqual([]);
    expect(kinds("@main")).toEqual([[1, "changed", "func.func"]]);
  });

  it("reports removed symbols", () => {
    expect(kinds("@helper")).toEqual([[4, "removed", "func.func"]]);
    expect(byPath["@helper"].removed).toBe(true);
  });

  it("records the last dump that shows each symbol", () => {
    // #3 dumps only the gpu.module; #5 serializes the kernel away.
    expect(byPath["@main_kernel::@main_kernel"].lastDump).toBe(3);
    expect(byPath["@main_kernel"].lastDump).toBe(4);
    expect(byPath["@helper"].lastDump).toBe(3);
  });

  it("exports changes with pass names", () => {
    const json = JSON.parse(
      historyToJSON("t.txt", events, history, describeEvent),
    );
    const kernel = json.symbols.find(
      (s) => s.symbol === "@main_kernel::@main_kernel",
    );
    expect(kernel.changes[1]).toEqual({
      pass: describeEvent(events[2]),
      dump: 3,
      kind: "lowered",
      op: "llvm.func",
      from: "gpu.func",
    });
  });
});

describe("scanSymbols full text", () => {
  it("keeps nested symbols in `full` but not in `text`", () => {
    const { events } = parsePassTrace(TRACE);
    const module = scanSymbols(events[1].ir).get("@main_kernel");
    expect(module.text).toBe("gpu.module @main_kernel {\n}");
    expect(module.full).toBe(
      [
        "gpu.module @main_kernel {",
        "  gpu.func @main_kernel() kernel {",
        "    gpu.return",
        "  }",
        "}",
      ].join("\n"),
    );
  });
});

describe("symbolTimeline", () => {
  const { events } = parsePassTrace(TRACE);

  it("lists the kernel's IR at each pass that touched it", () => {
    const steps = symbolTimeline(events, "@main_kernel::@main_kernel");
    expect(steps.map((s) => [s.index, s.kind, s.op])).toEqual([
      [1, "created", "gpu.func"],
      [2, "lowered", "llvm.func"],
      [4, "removed", "llvm.func"],
    ]);
    expect(steps[0].text).toMatch(/^gpu\.func @main_kernel\(\) kernel \{/);
    expect(steps[1].text).toMatch(/^llvm\.func @main_kernel\(\)/);
    expect(steps[2].text).toBe(null);
  });

  it("starts with the first dump for symbols that were already there", () => {
    const steps = symbolTimeline(events, "@main");
    expect(steps.map((s) => [s.index, s.kind])).toEqual([
      [0, "initial"],
      [1, "changed"],
    ]);
  });

  it("counts a nested change against the module's full text", () => {
    const steps = symbolTimeline(events, "@main_kernel");
    expect(steps.map((s) => [s.index, s.kind])).toEqual([
      [1, "created"],
      [2, "changed"],
      [4, "lowered"],
    ]);
  });

  it("returns nothing for an unknown symbol", () => {
    expect(symbolTimeline(events, "@nope")).toEqual([]);
  });

  it("attaches the enclosing binary's assembly when a kernel is serialized", () => {
    const withPtx = parsePassTrace(
      TRACE.replace(
        '[#gpu.object<#nvvm.target, "BLOB">]',
        '[#gpu.object<#nvvm.target<chip = "sm_80">, properties = {O = 2 : i32}, assembly = ".version 7.0\\0A.entry main_kernel()\\0A">]',
      ),
    ).events;
    const removed = symbolTimeline(withPtx, "@main_kernel::@main_kernel").at(-1);
    expect(removed.kind).toBe("removed");
    expect(removed.assembly).toEqual([
      {
        target: '#nvvm.target<chip = "sm_80">',
        text: ".version 7.0\n.entry main_kernel()\n",
      },
    ]);
    const binary = symbolTimeline(withPtx, "@main_kernel").at(-1);
    expect(binary.kind).toBe("lowered");
    expect(binary.assembly).toHaveLength(1);
  });
});

describe("unescapeMlirString and embeddedAssembly", () => {
  it("decodes hex, quote and backslash escapes, and UTF-8", () => {
    expect(unescapeMlirString('a\\0Ab\\09c\\"d\\\\e\\C3\\A9')).toBe(
      'a\nb\tc"d\\eé',
    );
  });

  it("skips binary objects", () => {
    expect(
      embeddedAssembly(
        'gpu.binary @k [#gpu.object<#nvvm.target, bin = "\\7FELF">]',
      ),
    ).toEqual([]);
  });

  it("reads the real GPU sample's PTX", async () => {
    const { readFileSync } = await import("node:fs");
    const trace = readFileSync(
      new URL("../../public/samples/gpu-kernels.trace.txt", import.meta.url),
      "utf8",
    );
    const { events } = parsePassTrace(trace);
    const [ptx] = symbolTimeline(events, "@saxpy_kernel::@saxpy_kernel").at(-1)
      .assembly;
    expect(ptx.target).toBe('#nvvm.target<chip = "sm_80">');
    expect(ptx.text).toContain(".visible .entry saxpy_kernel(");
    expect(ptx.text).not.toContain("\\0A");
  });
});
