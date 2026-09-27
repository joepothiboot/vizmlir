import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  byteLength,
  extractReports,
  formatBytes,
  formatSeconds,
  hasTimingReport,
  matchTiming,
  timingToJSON,
} from "../../src/timing.js";
import { parsePassTrace } from "../../src/trace.js";

// Traces in tests/fixtures/traces are real mlir-opt output; see generate.sh.
function fixture(name) {
  return readFileSync(
    new URL(`../fixtures/traces/${name}.txt`, import.meta.url),
    "utf8",
  );
}

const names = (timing) =>
  timing.rows.map((row) => `${"  ".repeat(row.depth)}${row.name}`);

describe("extractReports", () => {
  it("reads the tree display and keeps the line count", () => {
    const text = fixture("timing-after-all");
    const { text: rest, timing, memory } = extractReports(text);
    expect(memory).toBeNull();
    expect(rest.split("\n")).toHaveLength(text.split(/\r?\n/).length);
    expect(rest).not.toMatch(/Execution time report|Wall Time/);
    expect(timing.display).toBe("tree");
    expect(timing.columns).toEqual(["wall"]);
    expect(timing.total).toBeGreaterThan(0);
    expect(names(timing)).toEqual([
      "Parser",
      "SymbolDCEPass",
      "'func.func' Pipeline",
      "  CSEPass",
      "    (A) DominanceInfo",
      "  CanonicalizerPass",
      "  CSEPass",
      "    (A) DominanceInfo",
      "Output",
      "Rest",
    ]);
    const cse = timing.rows[3];
    expect(cse).toMatchObject({ kind: "pass", anchor: "func.func", parent: 2 });
    expect(timing.rows[1]).toMatchObject({ kind: "pass", anchor: null });
    expect(timing.rows[4].kind).toBe("analysis");
    expect(timing.rows[0].kind).toBe("other");
  });

  it("reads the list display", () => {
    const { timing } = extractReports(fixture("timing-list"));
    expect(timing.display).toBe("list");
    expect(timing.columns).toEqual(["wall"]);
    expect(timing.rows.map((row) => row.name)).not.toContain("root");
  });

  it("reads user and wall columns from a threaded run", () => {
    const { timing } = extractReports(fixture("timing-threaded-list"));
    expect(timing.display).toBe("list");
    expect(timing.columns).toEqual(["user", "wall"]);
    expect(timing.rows.map((row) => row.name)).not.toContain("root");
    const cse = timing.rows.find((row) => row.name === "CSEPass");
    expect(cse.user.seconds).toBeGreaterThanOrEqual(0);
    expect(cse.wall.percent).toBeGreaterThanOrEqual(0);
  });

  it("reads -mlir-output-format=json", () => {
    const { text, timing } = extractReports(fixture("timing-json"));
    expect(timing.display).toBe("tree");
    expect(names(timing)).toEqual([
      "Parser",
      "'func.func' Pipeline",
      "  CSEPass",
      "    (A) DominanceInfo",
      "  CanonicalizerPass",
      "Output",
      "Rest",
    ]);
    expect(timing.total).toBeGreaterThan(0);
    expect(text).not.toMatch(/"duration"/);
    expect(text).toMatch(/func\.func @f/);
  });

  it("reads peak memory from BSD time -l output", () => {
    const { text, memory, timing } = extractReports(fixture("timed-run"));
    expect(timing).not.toBeNull();
    expect(memory.peakBytes).toBeGreaterThan(1024 * 1024);
    expect(text).not.toMatch(/resident set size|Maximum resident/);
  });

  it("reads peak memory from GNU time -v output", () => {
    const log = [
      "module {",
      "}",
      '\tCommand being timed: "mlir-opt in.mlir"',
      "\tUser time (seconds): 0.01",
      "\tMaximum resident set size (kbytes): 20480",
      "\tPage size (bytes): 4096",
      "\tExit status: 0",
    ].join("\n");
    const { text, memory } = extractReports(log);
    expect(memory).toEqual({ peakBytes: 20480 * 1024, source: "time -v" });
    expect(text).toBe("module {\n}\n\n\n\n\n");
  });

  it("returns the input untouched when there is no report", () => {
    const text = "module {\n}\n";
    expect(extractReports(text)).toEqual({ text, timing: null, memory: null });
    expect(hasTimingReport(text)).toBe(false);
    expect(hasTimingReport(fixture("timing-list"))).toBe(true);
  });
});

describe("parsePassTrace with timing", () => {
  it("keeps the report out of the dumps and the final output", () => {
    const trace = parsePassTrace(fixture("timing-after-all"));
    expect(trace.events).toHaveLength(7);
    expect(trace.timing.rows.length).toBeGreaterThan(0);
    for (const event of trace.events)
      expect(event.ir).not.toMatch(/Execution time|Wall Time/);
    expect(trace.output).toMatch(/^module \{/);
    expect(trace.output).not.toMatch(/Wall Time/);
  });

  it("still reports a failed pass", () => {
    const trace = parsePassTrace(fixture("timing-failed"));
    expect(trace.events.some((event) => event.failed)).toBe(true);
    expect(trace.timing.rows.map((row) => row.name)).toContain(
      "InterpreterPass",
    );
  });
});

describe("matchTiming", () => {
  it("maps repeated passes in a nested pipeline by position", () => {
    const trace = parsePassTrace(fixture("timing-after-all"));
    const matches = matchTiming(trace.events, trace.timing);
    const rowNames = matches.map(
      (match) => match && `${match.row.index}:${match.row.name}`,
    );
    // symbol-dce on the module, then cse, canonicalize, cse on @f and on @g.
    expect(rowNames).toEqual([
      "1:SymbolDCEPass",
      "3:CSEPass",
      "5:CanonicalizerPass",
      "6:CSEPass",
      "3:CSEPass",
      "5:CanonicalizerPass",
      "6:CSEPass",
    ]);
    expect(matches[0].runs).toBe(1);
    expect(matches[1].runs).toBe(2);
  });

  it("maps top-level passes in a module-scope trace", () => {
    const trace = parsePassTrace(fixture("timing-failed"));
    const matches = matchTiming(trace.events, trace.timing);
    expect(matches.map((match) => match?.row.name)).toEqual(
      trace.events.map((event) => event.pass),
    );
  });

  it("maps every run to the merged row in the list display", () => {
    const events = [
      { phase: "after", pass: "CSEPass", root: { op: "func.func" } },
      { phase: "after", pass: "CSEPass", root: { op: "func.func" } },
      { phase: "after", pass: "Unknown", root: { op: "func.func" } },
    ];
    const { timing } = extractReports(fixture("timing-list"));
    const matches = matchTiming(events, timing);
    expect(matches[0].row).toBe(matches[1].row);
    expect(matches[0].runs).toBe(2);
    expect(matches[2]).toBeNull();
  });

  it("leaves extra runs unmatched instead of guessing", () => {
    const trace = parsePassTrace(fixture("timing-after-all"));
    const events = [...trace.events, trace.events[1], trace.events[3]];
    const matches = matchTiming(events, trace.timing);
    expect(matches.slice(-2)).toEqual([null, null]);
  });

  it("returns nulls without a report", () => {
    expect(matchTiming([{ pass: "CSEPass" }], null)).toEqual([null]);
  });
});

describe("formatting", () => {
  it("formats seconds from a 4-decimal report", () => {
    expect(formatSeconds(0)).toBe("<0.1 ms");
    expect(formatSeconds(0.0004)).toBe("0.4 ms");
    expect(formatSeconds(0.0126)).toBe("12.6 ms");
    expect(formatSeconds(1.2345)).toBe("1.23 s");
    expect(formatSeconds(null)).toBe("—");
  });

  it("formats bytes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(34652160)).toBe("33.0 MB");
  });

  it("counts UTF-8 bytes", () => {
    expect(byteLength("abc")).toBe(3);
    expect(byteLength("é→😀")).toBe(2 + 3 + 4);
  });
});

describe("timingToJSON", () => {
  it("serialises rows, memory, and event links", () => {
    const trace = parsePassTrace(fixture("timed-run"));
    const matches = matchTiming(trace.events, trace.timing);
    const json = JSON.parse(
      timingToJSON("t", trace.timing, trace.memory, trace.events, matches),
    );
    expect(json.peakMemoryBytes).toBe(trace.memory.peakBytes);
    expect(json.rows[0]).toHaveProperty("wallSeconds");
    expect(json.events[0].timingRow).toBe(matches[0].row.index);
    expect(json.events[0].irBytes).toBeGreaterThan(0);
  });
});

describe("lowering sample", () => {
  const text = readFileSync(
    new URL("../../public/samples/lowering.trace.txt", import.meta.url),
    "utf8",
  );

  it("times every pass and tells repeated passes apart", () => {
    const trace = parsePassTrace(text);
    const matches = matchTiming(trace.events, trace.timing);
    expect(trace.memory.peakBytes).toBeGreaterThan(0);
    expect(matches.every(Boolean)).toBe(true);
    const canonicalize = trace.events
      .map((event, i) => event.pass === "CanonicalizerPass" && matches[i])
      .filter(Boolean);
    expect(canonicalize).toHaveLength(2);
    expect(canonicalize[0].row).not.toBe(canonicalize[1].row);
    expect(trace.output).toMatch(/^module/);
    expect(trace.output).not.toMatch(/real|Wall Time/);
  });
});
