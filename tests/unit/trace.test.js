import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  baselineFor,
  describeEvent,
  extractSymbolOp,
  isPassTrace,
  moduleStateAt,
  parseHeader,
  parsePassTrace,
} from "../../src/trace.js";

// Traces in tests/fixtures/traces are real mlir-opt output; see generate.sh.
function fixture(name) {
  return readFileSync(
    new URL(`../fixtures/traces/${name}.txt`, import.meta.url),
    "utf8",
  );
}

const header = (body) => `// -----// IR Dump ${body} //----- //`;
const legacyHeader = (body) => `// *** IR Dump ${body} ***`;

const CANON_OPTS =
  "cse-between-iterations=false max-iterations=10 max-num-rewrites=-1 " +
  "region-simplify=normal test-convergence=false top-down=true";

describe("parseHeader", () => {
  it.each([
    [
      "After CSEPass: cse ('builtin.module' operation)",
      {
        phase: "after",
        pass: "CSEPass",
        argument: "cse",
        options: null,
        failed: false,
        anchor: { op: "builtin.module", symbol: null },
      },
    ],
    [
      "After CanonicalizerPass: canonicalize{cse-between-iterations=false    max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true} ('func.func' operation: @f)",
      {
        phase: "after",
        pass: "CanonicalizerPass",
        argument: "canonicalize",
        options: CANON_OPTS,
        failed: false,
        anchor: { op: "func.func", symbol: "f" },
      },
    ],
    [
      "After InterpreterPass Failed: transform-interpreter{ debug-payload-root-tag= entry-point=__transform_main}",
      {
        phase: "after",
        pass: "InterpreterPass",
        argument: "transform-interpreter",
        options: "debug-payload-root-tag= entry-point=__transform_main",
        failed: true,
        anchor: null,
      },
    ],
    [
      "After CSE (cse) ('func.func' operation: @f)",
      {
        phase: "after",
        pass: "CSE",
        argument: "cse",
        options: null,
        failed: false,
        anchor: { op: "func.func", symbol: "f" },
      },
    ],
    [
      "After CSE Failed (cse)",
      {
        phase: "after",
        pass: "CSE",
        argument: "cse",
        options: null,
        failed: true,
        anchor: null,
      },
    ],
    [
      "Before Canonicalizer",
      {
        phase: "before",
        pass: "Canonicalizer",
        argument: null,
        options: null,
        failed: false,
        anchor: null,
      },
    ],
    [
      "After Foo Failed",
      {
        phase: "after",
        pass: "Foo",
        argument: null,
        options: null,
        failed: true,
        anchor: null,
      },
    ],
    [
      `After CSEPass: cse ('func.func' operation: @"my fn")`,
      {
        phase: "after",
        pass: "CSEPass",
        argument: "cse",
        options: null,
        failed: false,
        anchor: { op: "func.func", symbol: "my fn" },
      },
    ],
  ])("%s", (body, expected) => {
    expect(parseHeader(header(body))).toEqual(expected);
  });

  it("accepts the legacy *** header", () => {
    expect(parseHeader(legacyHeader("After CSE (cse)"))).toMatchObject({
      phase: "after",
      pass: "CSE",
      argument: "cse",
    });
  });

  it("tolerates trailing whitespace and \\r", () => {
    expect(parseHeader(`${header("After CSEPass: cse")} \r`)).toMatchObject({
      argument: "cse",
    });
  });

  it.each([
    "func.func @f() {",
    "// a regular comment",
    header("Middle CSEPass: cse"),
    "// -----// IR Dump After CSEPass: cse",
  ])("rejects %s", (line) => {
    expect(parseHeader(line)).toBeNull();
  });
});

describe("isPassTrace", () => {
  it("detects every fixture", () => {
    for (const name of [
      "nested-after-all",
      "module-scope-after-all",
      "nested-before-all",
      "mixed-nesting",
      "failed-pass",
      "failed-only",
    ]) {
      expect(isPassTrace(fixture(name)), name).toBe(true);
    }
  });

  it("finds a header after diagnostics and CRLF line endings", () => {
    const text = [
      "in.mlir:1:1: warning: hi",
      "",
      header("After CSEPass: cse"),
      "module {",
      "}",
    ].join("\r\n");
    expect(isPassTrace(text)).toBe(true);
  });

  it("is false for plain MLIR", () => {
    expect(isPassTrace("module {\n  // IR Dump\n}\n")).toBe(false);
    expect(isPassTrace("")).toBe(false);
  });

  it("only probes the first 2000 lines", () => {
    const filler = "// filler\n".repeat(2000);
    expect(isPassTrace(`${filler}${header("After CSEPass: cse")}\n`)).toBe(
      false,
    );
  });
});

describe("parsePassTrace on real traces", () => {
  it("nested after-all: one event per function dump, final module as output", () => {
    const trace = parsePassTrace(fixture("nested-after-all"));
    expect(trace.events.map(describeEvent)).toEqual([
      "After cse · func.func @f",
      "After canonicalize · func.func @f",
      "After cse · func.func @g",
      "After canonicalize · func.func @g",
    ]);
    expect(trace.events.map((event) => event.headerLine)).toEqual([
      1, 9, 15, 22,
    ]);
    expect(trace.events[0].irLine).toBe(2);
    expect(trace.events[0].ir.split("\n")[0]).toBe(
      "func.func @f(%arg0: i32) -> i32 {",
    );
    expect(trace.events[0].ir.split("\n").at(-1)).toBe("}");
    expect(trace.events[1].options).toBe(CANON_OPTS);
    expect(trace.events.every((event) => event.trailing === "")).toBe(true);
    expect(trace.output.startsWith("module {")).toBe(true);
    expect(trace.preamble).toBe("");
    expect(trace.diagnostics).toEqual([]);
  });

  it("module scope: roots are modules while anchors name the function", () => {
    const trace = parsePassTrace(fixture("module-scope-after-all"));
    expect(trace.events).toHaveLength(4);
    for (const event of trace.events) {
      expect(event.root).toEqual({ op: "builtin.module", symbol: null });
      expect(event.anchor.op).toBe("func.func");
    }
    expect(trace.events.map((event) => event.anchor.symbol)).toEqual([
      "f",
      "f",
      "g",
      "g",
    ]);
  });

  it("before-all dumps have phase before", () => {
    const trace = parsePassTrace(fixture("nested-before-all"));
    expect(trace.events.map((event) => event.phase)).toEqual(
      Array(4).fill("before"),
    );
  });

  it("failed pass: diagnostics attach to the failed dump with their snippets", () => {
    const trace = parsePassTrace(fixture("failed-pass"));
    expect(trace.events.map((event) => event.failed)).toEqual([false, true]);
    const [first, failed] = trace.events;
    expect(first.diagnostics).toEqual([]);
    expect(first.trailing).toBe("");
    expect(failed.diagnostics.map((d) => [d.severity, d.location])).toEqual([
      ["error", { file: "input/tile-non-tileable.mlir", line: 10, column: 14 }],
      ["note", { file: "input/tile-non-tileable.mlir", line: 4, column: 10 }],
    ]);
    expect(failed.diagnostics[0].message).toBe(
      "only ops implementing TilingInterface are supported",
    );
    expect(failed.diagnostics[0].detail.split("\n")).toHaveLength(2);
    expect(failed.diagnostics[0].detail.split("\n")[1].trim()).toBe("^");
    expect(trace.diagnostics.every((d) => d.eventIndex === 1)).toBe(true);
  });

  it("after-failure only: diagnostics before the first header still attach", () => {
    const trace = parsePassTrace(fixture("failed-only"));
    expect(trace.events).toHaveLength(1);
    expect(trace.events[0].failed).toBe(true);
    expect(trace.events[0].diagnostics).toHaveLength(2);
    expect(trace.preamble).toBe("");
  });
});

describe("parsePassTrace edge cases", () => {
  it("keeps aliases with the dump and preamble text before the first header", () => {
    const trace = parsePassTrace(
      [
        "starting pipeline",
        header("After CSEPass: cse"),
        "#map = affine_map<(d0) -> (d0)>",
        "module {",
        "}",
        "",
        "#loc = loc(unknown)",
        header("After CSEPass: cse"),
        "module {",
        "}",
      ].join("\n"),
    );
    expect(trace.preamble).toBe("starting pipeline");
    expect(trace.events[0].ir).toBe(
      "#map = affine_map<(d0) -> (d0)>\nmodule {\n}\n\n#loc = loc(unknown)",
    );
    expect(trace.events[0].root).toEqual({
      op: "builtin.module",
      symbol: null,
    });
  });

  it("ignores braces in strings and comments when finding the end of a dump", () => {
    const ir = [
      "module {",
      '  "test.op"() {s = "}}"} : () -> ()',
      '  "test.op"() {s = "\\"}"} : () -> () // }',
      "}",
    ];
    const trace = parsePassTrace(
      [header("After CSEPass: cse"), ...ir, "after the dump"].join("\n"),
    );
    expect(trace.events[0].ir).toBe(ir.join("\n"));
    expect(trace.output).toBe("after the dump");
  });

  it("parses CRLF traces", () => {
    const trace = parsePassTrace(
      [header("After CSEPass: cse"), "module {", "}", ""].join("\r\n"),
    );
    expect(trace.events[0].ir).toBe("module {\n}");
  });

  it("holds diagnostics printed before a Before dump until the next After dump", () => {
    const trace = parsePassTrace(
      [
        header("Before CSEPass: cse"),
        "module {",
        "}",
        "a.mlir:3: warning: no column",
        header("Before CanonicalizerPass: canonicalize"),
        "module {",
        "}",
        header("After CanonicalizerPass: canonicalize"),
        "module {",
        "}",
      ].join("\n"),
    );
    expect(trace.events.map((event) => event.diagnostics.length)).toEqual([
      0, 0, 1,
    ]);
    expect(trace.events[2].diagnostics[0]).toMatchObject({
      severity: "warning",
      message: "no column",
      location: { file: "a.mlir", line: 3, column: null },
      traceLine: 4,
      eventIndex: 2,
    });
  });

  it("marks diagnostics after the last dump with eventIndex -1", () => {
    const trace = parsePassTrace(
      [
        header("After CSEPass: cse"),
        "module {",
        "}",
        "a.mlir:1:2: error: pass without a dump failed",
        "  snippet",
        "",
        "a.mlir:1:2: remark: another",
      ].join("\n"),
    );
    expect(trace.events[0].diagnostics).toEqual([]);
    expect(
      trace.diagnostics.map((d) => [d.severity, d.eventIndex, d.detail]),
    ).toEqual([
      ["error", -1, "  snippet"],
      ["remark", -1, ""],
    ]);
    expect(trace.output).toBe("");
  });

  it("does not throw on empty or truncated input", () => {
    expect(parsePassTrace("")).toMatchObject({
      events: [],
      output: "",
      preamble: "",
    });
    const trace = parsePassTrace(
      [
        header("After CSEPass: cse"),
        header("After CSEPass: cse"),
        "module {",
        "  func.func @f() {",
      ].join("\n"),
    );
    expect(trace.events).toHaveLength(2);
    expect(trace.events[0].ir).toBe("");
    expect(trace.events[0].root).toBeNull();
    expect(trace.events[1].ir).toBe("module {\n  func.func @f() {");
  });
});

describe("baselineFor", () => {
  it("nested dumps compare against the previous dump of the same function", () => {
    const { events } = parsePassTrace(fixture("nested-after-all"));
    expect(baselineFor(events, 0)).toBeNull();
    expect(baselineFor(events, 1)).toMatchObject({
      event: events[0],
      ir: events[0].ir,
      reconstructed: false,
    });
    expect(baselineFor(events, 2)).toBeNull();
    expect(baselineFor(events, 3).event).toBe(events[2]);
  });

  it("module-scope dumps compare against the previous module dump", () => {
    const { events } = parsePassTrace(fixture("module-scope-after-all"));
    expect(baselineFor(events, 0)).toBeNull();
    for (const index of [1, 2, 3]) {
      expect(baselineFor(events, index)).toMatchObject({
        event: events[index - 1],
        reconstructed: false,
      });
    }
  });

  it("a function dump pulls its baseline out of an earlier module dump", () => {
    const { events } = parsePassTrace(fixture("mixed-nesting"));
    const base = baselineFor(events, 1);
    expect(base.event).toBe(events[0]);
    expect(base.reconstructed).toBe(false);
    expect(base.ir).toBe(extractSymbolOp(events[0].ir, "func.func", "f"));
    expect(base.ir.split("\n")[0]).toBe("func.func @f(%arg0: i32) -> i32 {");
    expect(base.ir.split("\n").at(-1)).toBe("}");
    expect(base.ir.split("\n")[1]).toBe("  %c0_i32 = arith.constant 0 : i32");
  });

  it("rebuilds a module baseline by splicing newer function dumps into the last module", () => {
    const { events } = parsePassTrace(fixture("mixed-nesting"));
    const base = baselineFor(events, 3);
    expect(base.reconstructed).toBe(true);
    expect(base.event).toBe(events[2]);
    // symbol-dce changed nothing, so the rebuilt input equals its output.
    expect(base.ir).toBe(events[3].ir);
  });

  it("builds a module from scratch when only function dumps came before", () => {
    const events = parsePassTrace(
      [
        header("After CSEPass: cse"),
        "func.func @f() {",
        "  return",
        "}",
        header("After SymbolDCEPass: symbol-dce"),
        "module {",
        "}",
      ].join("\n"),
    ).events;
    expect(baselineFor(events, 1)).toEqual({
      event: events[0],
      ir: "module {\n  func.func @f() {\n    return\n  }\n}",
      reconstructed: true,
    });
  });

  it("appends a function that the base module does not contain yet", () => {
    const events = parsePassTrace(
      [
        header("After A: a"),
        "module {",
        "  func.func @old() {",
        "  }",
        "}",
        header("After B: b"),
        "func.func @new() {",
        "}",
        header("After C: c"),
        "module {",
        "}",
      ].join("\n"),
    ).events;
    expect(baselineFor(events, 2).ir).toBe(
      "module {\n  func.func @old() {\n  }\n  func.func @new() {\n  }\n}",
    );
  });

  it("returns null for events without a root", () => {
    const events = parsePassTrace(
      [header("After CSEPass: cse"), ""].join("\n"),
    ).events;
    expect(baselineFor(events, 0)).toBeNull();
    expect(baselineFor(events, 5)).toBeNull();
  });
});

describe("extractSymbolOp", () => {
  const ir = [
    "module {",
    "  func.func private @a.b$c(%x: i32) {",
    "    return",
    "  }",
    '  llvm.func @"quoted name"() {',
    "  }",
    "  %g = test.global @sym : i32",
    "}",
  ].join("\n");

  it.each([
    ["func.func", "a.b$c", "func.func private @a.b$c(%x: i32) {\n  return\n}"],
    ["llvm.func", "quoted name", 'llvm.func @"quoted name"() {\n}'],
    ["test.global", "sym", "%g = test.global @sym : i32"],
  ])("%s @%s", (op, symbol, expected) => {
    expect(extractSymbolOp(ir, op, symbol)).toBe(expected);
  });

  it("does not match a symbol that is only a prefix", () => {
    expect(extractSymbolOp(ir, "func.func", "a")).toBeNull();
    expect(extractSymbolOp(ir, "func.func", "missing")).toBeNull();
  });
});

describe("describeEvent", () => {
  it("prefers argument over pass name and anchor over root", () => {
    expect(
      describeEvent({
        phase: "after",
        pass: "CSEPass",
        argument: "cse",
        anchor: { op: "func.func", symbol: "f" },
        root: { op: "builtin.module", symbol: null },
      }),
    ).toBe("After cse · func.func @f");
    expect(
      describeEvent({
        phase: "before",
        pass: "Canonicalizer",
        argument: null,
        anchor: null,
        root: null,
      }),
    ).toBe("Before Canonicalizer");
  });
});

// schema-opt (json-schema-mlir) is an out-of-tree MlirOptMain driver with its
// own dialect; see generate.sh.
describe("out-of-tree driver trace", () => {
  const trace = parsePassTrace(fixture("schema-opt-pipeline"));

  it("splits every pass of the pipeline", () => {
    expect(trace.events.map((event) => event.argument)).toEqual([
      "canonicalize",
      "schema-canonicalize",
      "lower-schema-to-std",
      "reconcile-unrealized-casts",
      "canonicalize",
      "cse",
      "symbol-dce",
    ]);
    expect(trace.diagnostics).toEqual([]);
  });

  it("keeps namespace-qualified pass names whole", () => {
    expect(trace.events[1].pass).toBe(
      "(anonymous namespace)::SchemaCanonicalizerPass",
    );
    expect(trace.events[1].argument).toBe("schema-canonicalize");
  });

  it("finds the root op of nested dumps of custom-dialect IR", () => {
    expect(trace.events[1].root).toEqual({
      op: "func.func",
      symbol: "validate_person",
    });
    expect(trace.events[1].ir).toContain("schema.validate_number");
    // The baseline is the function cut out of the module dump before it.
    const baseline = baselineFor(trace.events, 1);
    expect(baseline.event.index).toBe(0);
    expect(baseline.ir).toMatch(/^func\.func @validate_person/);
    expect(baseline.ir).toContain("arith.andi");
    expect(trace.events[1].ir).not.toContain("arith.andi");
  });
});

describe("moduleStateAt", () => {
  it("returns a module dump as is", () => {
    const { events } = parsePassTrace(fixture("mixed-nesting"));
    expect(moduleStateAt(events, 0)).toBe(events[0].ir);
    expect(moduleStateAt(events, 3)).toBe(events[3].ir);
  });

  it("splices a function dump into the last module dump", () => {
    const { events } = parsePassTrace(fixture("mixed-nesting"));
    const state = moduleStateAt(events, 1);
    expect(state.split("\n")[0]).toBe("module {");
    expect(extractSymbolOp(state, "func.func", "f")).toBe(
      extractSymbolOp(events[1].ir, "func.func", "f"),
    );
    // @g has not been rewritten yet, so it still matches the module dump.
    expect(extractSymbolOp(state, "func.func", "g")).toBe(
      extractSymbolOp(events[0].ir, "func.func", "g"),
    );
    // After both function dumps the module matches the next module dump.
    expect(moduleStateAt(events, 2)).toBe(events[3].ir);
  });

  it("builds a module when no module dump came before", () => {
    const { events } = parsePassTrace(fixture("nested-after-all"));
    const state = moduleStateAt(events, 2);
    expect(state).toContain("func.func @f(");
    expect(state).toContain("func.func @g(");
  });
});
