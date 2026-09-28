import { describe, expect, it } from "vitest";
import {
  changedBeyond,
  cleanKernelName,
  compareBenchmarks,
  comparisonOrder,
  comparisonToJSON,
  formatChange,
  formatDuration,
  matchBenchmarks,
  parseBenchmarks,
  parseCSV,
  symbolTimes,
} from "../../src/bench.js";

const summarize = (result) =>
  result.entries.map((e) => [e.kernel, e.calls, Math.round(e.timeNs)]);

describe("parseCSV", () => {
  it("handles quotes, escaped quotes, commas and CRLF", () => {
    expect(parseCSV('a,"b, c","say ""hi"""\r\n1,2,3\n')).toEqual([
      ["a", "b, c", 'say "hi"'],
      ["1", "2", "3"],
    ]);
  });
});

describe("parseBenchmarks", () => {
  it("reads a generic CSV with the unit in the header", () => {
    const result = parseBenchmarks("kernel,time_us\nmain_kernel,12.5\nother,3\n");
    expect(result.format).toBe("csv");
    expect(result.timeColumn).toBe("time_us");
    expect(summarize(result)).toEqual([
      ["main_kernel", 1, 12500],
      ["other", 1, 3000],
    ]);
  });

  it("averages repeated rows for the same kernel", () => {
    const result = parseBenchmarks("name,duration [ms]\nk,1\nk,3\n");
    expect(summarize(result)).toEqual([["k", 2, 2e6]]);
  });

  it("reads an nsys cuda_gpu_kern_sum CSV after progress lines", () => {
    const text = [
      "Generating SQLite file report.sqlite from report.nsys-rep",
      "Processing [report.sqlite] with [cuda_gpu_kern_sum.py]... ",
      "",
      '"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"',
      '"91.2","182400","4","45600.0","45500.0","45000","46400","600.0","main_kernel"',
      '"8.8","17600","4","4400.0","4400.0","4300","4500","80.0","void reduce<float>(float*, int)"',
    ].join("\n");
    const result = parseBenchmarks(text);
    expect(result.timeColumn).toBe("Avg (ns)");
    expect(summarize(result)).toEqual([
      ["main_kernel", 4, 45600],
      ["void reduce<float>(float*, int)", 4, 4400],
    ]);
  });

  it("reads ncu --csv metrics, one row per launch", () => {
    const text = [
      "==PROF== Connected to process 1234",
      '"ID","Process ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"',
      '"0","1234","main_kernel","GPU Speed Of Light Throughput","gpu__time_duration.sum","usecond","10.00"',
      '"0","1234","main_kernel","GPU Speed Of Light Throughput","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","42"',
      '"1","1234","main_kernel","GPU Speed Of Light Throughput","gpu__time_duration.sum","usecond","14.00"',
      '"2","1234","big","GPU Speed Of Light Throughput","gpu__time_duration.sum","msecond","1,200"',
    ].join("\n");
    const result = parseBenchmarks(text);
    expect(result.format).toBe("ncu");
    expect(summarize(result)).toEqual([
      ["main_kernel", 2, 12000],
      ["big", 1, 1.2e9],
    ]);
  });

  it("reads JSON results with an explicit symbol", () => {
    const result = parseBenchmarks(
      JSON.stringify({
        kernels: [{ kernel: "k", symbol: "@m::@k", time_ms: 0.5, calls: 10 }],
      }),
    );
    expect(result.format).toBe("json");
    expect(result.entries[0]).toMatchObject({
      kernel: "k",
      symbol: "@m::@k",
      calls: 10,
      timeNs: 5e5,
    });
  });

  it("reads Google Benchmark JSON and skips its aggregate rows", () => {
    const result = parseBenchmarks(
      JSON.stringify({
        context: {},
        benchmarks: [
          { name: "BM_matmul", run_type: "iteration", iterations: 100, real_time: 20, cpu_time: 19, time_unit: "us" },
          { name: "BM_matmul", run_type: "iteration", iterations: 300, real_time: 40, cpu_time: 39, time_unit: "us" },
          { name: "BM_matmul_stddev", run_type: "aggregate", aggregate_name: "stddev", iterations: 2, real_time: 1, cpu_time: 1, time_unit: "us" },
        ],
      }),
    );
    expect(result.format).toBe("google-benchmark");
    expect(summarize(result)).toEqual([["BM_matmul", 400, 35000]]);
  });

  it.each([
    ["no name column", "foo,time_us\na,1\n", /kernel name column/],
    ["no unit", "kernel,time\na,1\n", /with a unit/],
    ["bad JSON", "{nope", /not valid JSON/],
    ["JSON without results", '{"a":1}', /no array/],
  ])("explains what is missing: %s", (_, text, message) => {
    expect(() => parseBenchmarks(text)).toThrow(message);
  });
});

describe("cleanKernelName", () => {
  it.each([
    ["main_kernel", "main_kernel"],
    ["void reduce<float>(float*, int)", "reduce"],
    ["ns::inner::k(int)", "k"],
    ["void k<cute::tuple<int, int>>(Params)", "k"],
  ])("%s -> %s", (name, expected) => {
    expect(cleanKernelName(name)).toBe(expected);
  });
});

describe("matchBenchmarks", () => {
  const record = (path, ops) => ({
    path,
    symbol: path.split("::").at(-1).slice(1),
    ops,
  });
  const history = [
    record("@main", ["func.func"]),
    record("@main_kernel", ["gpu.module", "gpu.binary"]),
    record("@main_kernel::@main_kernel", ["gpu.func", "llvm.func"]),
    record("@a::@dup", ["gpu.func"]),
    record("@b::@dup", ["gpu.func"]),
    record("@reduce", ["func.func"]),
  ];
  const entry = (kernel, symbol = null) => ({ kernel, symbol, calls: 1, timeNs: 1 });
  const match = (e) => {
    const [m] = matchBenchmarks([e], history);
    return [m.path, m.how, m.candidates];
  };

  it("prefers the kernel function over the module that holds it", () => {
    expect(match(entry("main_kernel"))).toEqual([
      "@main_kernel::@main_kernel",
      "exact",
      ["@main_kernel::@main_kernel"],
    ]);
  });

  it("matches C++ names and Itanium-mangled names", () => {
    expect(match(entry("void reduce<float>(float*, int)"))[0]).toBe("@reduce");
    expect(match(entry("_Z6reducePfi"))).toEqual(["@reduce", "mangled", ["@reduce"]]);
  });

  it("reports ambiguous names with their candidates", () => {
    expect(match(entry("dup"))).toEqual([null, "exact", ["@a::@dup", "@b::@dup"]]);
  });

  it("uses an explicit symbol and accepts it without @", () => {
    expect(match(entry("whatever", "b::dup"))).toEqual([
      "@b::@dup",
      "symbol",
      ["@b::@dup"],
    ]);
    expect(match(entry("whatever", "@nope"))[0]).toBe(null);
  });

  it("leaves unknown kernels unmatched", () => {
    expect(match(entry("cublas_gemm"))).toEqual([null, null, []]);
  });
});

describe("formatDuration", () => {
  it.each([
    [512, "512 ns"],
    [4400, "4.40 µs"],
    [45600, "45.6 µs"],
    [1.2e6, "1.20 ms"],
    [2.5e9, "2.50 s"],
  ])("%d ns -> %s", (ns, text) => {
    expect(formatDuration(ns)).toBe(text);
  });
});

describe("symbolTimes", () => {
  const history = [
    { path: "@k", symbol: "k", ops: ["func.func"] },
    { path: "@m::@g", symbol: "g", ops: ["gpu.func"] },
  ];

  it("combines kernels that match one symbol, weighted by calls", () => {
    const { times, unmatched } = symbolTimes(
      [
        { kernel: "k", symbol: null, calls: 1, totalNs: 10, timeNs: 10 },
        { kernel: "_Z1kv", symbol: null, calls: 3, totalNs: 90, timeNs: 30 },
        { kernel: "cublas", symbol: null, calls: 1, totalNs: 5, timeNs: 5 },
      ],
      history,
    );
    expect(times.get("@k")).toMatchObject({
      calls: 4,
      totalNs: 100,
      timeNs: 25,
      kernels: ["k", "_Z1kv"],
      how: ["exact", "mangled"],
    });
    expect(unmatched.map((m) => m.entry.kernel)).toEqual(["cublas"]);
  });
});

describe("compareBenchmarks", () => {
  const time = (timeNs) => ({ timeNs, totalNs: timeNs, calls: 1, kernels: ["x"] });
  const baseline = new Map([
    ["@a", time(100)],
    ["@b", time(200)],
    ["@gone", time(50)],
  ]);
  const current = new Map([
    ["@a", time(150)],
    ["@b", time(190)],
    ["@new", time(70)],
  ]);
  const comparison = compareBenchmarks(baseline, current);

  it("joins both runs by symbol with the relative change", () => {
    expect(comparison.get("@a")).toMatchObject({ deltaNs: 50, change: 0.5 });
    expect(comparison.get("@b").change).toBeCloseTo(-0.05);
    expect(comparison.get("@gone")).toMatchObject({ current: null, change: null });
    expect(comparison.get("@new")).toMatchObject({ baseline: null, change: null });
  });

  it("works with one run", () => {
    expect(compareBenchmarks(null, current).get("@a")).toMatchObject({
      baseline: null,
      change: null,
    });
  });

  it("filters by change, keeping one-sided symbols", () => {
    const kept = [...comparison]
      .filter(([, row]) => changedBeyond(row, 5))
      .map(([path]) => path);
    expect(kept).toEqual(["@a", "@gone", "@new"]);
    expect(changedBeyond(comparison.get("@b"), 4)).toBe(true);
    expect(changedBeyond(undefined, 0)).toBe(false);
  });

  it("orders by largest slowdown, then one-sided by time, then unmeasured", () => {
    const paths = ["@b", undefined, "@new", "@a", "@gone"];
    const sorted = paths.sort((x, y) =>
      comparisonOrder(comparison.get(x), comparison.get(y)),
    );
    expect(sorted).toEqual(["@a", "@b", "@new", "@gone", undefined]);
  });

  it("exports per-side times and the change", () => {
    const json = comparisonToJSON(comparison);
    expect(json.get("@a")).toEqual({
      baseline: { time_ns: 100, calls: 1, kernels: ["x"] },
      current: { time_ns: 150, calls: 1, kernels: ["x"] },
      delta_ns: 50,
      change: 0.5,
    });
    expect(json.get("@new")).toEqual({
      current: { time_ns: 70, calls: 1, kernels: ["x"] },
    });
  });
});

describe("formatChange", () => {
  it.each([
    [0.5, "+50%"],
    [-0.053, "−5.3%"],
    [0, "±0.0%"],
    [null, ""],
  ])("%s -> %s", (change, text) => {
    expect(formatChange(change)).toBe(text);
  });
});
