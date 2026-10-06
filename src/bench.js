import { signOf } from "./format.js";

const UNIT_NS = {
  ns: 1,
  nsecond: 1,
  us: 1e3,
  µs: 1e3,
  usecond: 1e3,
  ms: 1e6,
  msecond: 1e6,
  s: 1e9,
  second: 1e9,
};

const UNIT = /\s*(?:\(([^)]+)\)|\[([^\]]+)\]|[_ ]([a-zµ]+))$/;

const NAME =
  /^(?:kernel[ _]?name|kernel|name|function|func|demangled[ _]name)$/;

const SYMBOL = /^symbol$/;

const CALLS =
  /^(?:instances|calls|count|launches|invocations|num[ _]?calls|iterations)$/;

const TIME_PREFERENCE = [
  /^(?:avg|average|mean)(?:[ _]?time)?$/,
  /^(?:med|median)(?:[ _]?time)?$/,
  /^(?:time|duration|latency|runtime|elapsed|real[ _]time|gpu[ _]time)$/,
  /^total(?:[ _]?time)?$/,
];

export function unitScale(unit) {
  return UNIT_NS[String(unit).trim().toLowerCase()] ?? null;
}

function splitHeader(header) {
  const lower = header.trim().toLowerCase();
  const match = UNIT.exec(lower);

  if (match) {
    const scale = unitScale(match[1] ?? match[2] ?? match[3]);
    if (scale) return { base: lower.slice(0, match.index).trim(), scale };
  }

  return { base: lower, scale: null };
}

export function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];

    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (c === '"') {
        quoted = false;
      } else {
        field += c;
      }
    } else if (c === '"') {
      quoted = true;
    } else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i += 1;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += c;
    }
  }

  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}

function toNumber(value) {
  if (typeof value === "number") return value;

  const n = Number(
    String(value ?? "")
      .replaceAll(",", "")
      .trim(),
  );

  return Number.isFinite(n) && String(value).trim() !== "" ? n : null;
}

function pickColumns(headers) {
  const parts = headers.map(splitHeader);
  const find = (pattern) => parts.findIndex((part) => pattern.test(part.base));
  const name = find(NAME);
  if (name < 0) return null;

  const unitColumn = find(/^(?:time[ _])?unit$/);
  let time = -1;
  let total = false;

  for (const [rank, pattern] of TIME_PREFERENCE.entries()) {
    time = parts.findIndex(
      (part, i) =>
        pattern.test(part.base) &&
        (part.scale || (unitColumn >= 0 && i !== unitColumn)),
    );

    if (time >= 0) {
      total = rank === TIME_PREFERENCE.length - 1;
      break;
    }
  }

  return {
    name,
    symbol: find(SYMBOL),
    calls: find(CALLS),
    time,
    total,
    scale: time >= 0 ? parts[time].scale : null,
    unitColumn,
  };
}

function fromTable(headers, records, format) {
  const columns = pickColumns(headers);

  if (!columns) {
    throw new Error("no kernel name column (kernel, name, or Kernel Name)");
  }

  if (columns.time < 0) {
    throw new Error(
      "no time column with a unit, such as time_us, Avg (ns) or duration [ms]",
    );
  }

  const samples = [];
  let skipped = 0;

  for (const record of records) {
    const kernel = String(record[columns.name] ?? "").trim();
    const value = toNumber(record[columns.time]);

    const scale =
      columns.scale ??
      (columns.unitColumn >= 0 ? unitScale(record[columns.unitColumn]) : null);

    if (!kernel || value === null || !scale) {
      skipped += 1;
      continue;
    }

    const calls =
      columns.calls >= 0 ? (toNumber(record[columns.calls]) ?? 1) : 1;

    const ns = value * scale;

    samples.push({
      kernel,
      symbol:
        columns.symbol >= 0
          ? String(record[columns.symbol] ?? "").trim() || null
          : null,
      calls,
      totalNs: columns.total ? ns : ns * calls,
    });
  }

  return {
    format,
    timeColumn: headers[columns.time],
    ...aggregate(samples),
    skipped,
  };
}

function fromNcu(headers, records) {
  const col = (name) =>
    headers.findIndex((header) => header.trim().toLowerCase() === name);

  const [name, metric, unit, value] = [
    col("kernel name"),
    col("metric name"),
    col("metric unit"),
    col("metric value"),
  ];

  const samples = [];

  for (const record of records) {
    if (
      !/^(?:gpu__time_duration\.sum|duration)$/.test(record[metric]?.trim())
    ) {
      continue;
    }

    const scale = unitScale(record[unit]);
    const ns = toNumber(record[value]);
    if (!scale || ns === null) continue;

    samples.push({
      kernel: record[name].trim(),
      symbol: null,
      calls: 1,
      totalNs: ns * scale,
    });
  }

  if (!samples.length) {
    throw new Error("ncu CSV has no gpu__time_duration.sum metric rows");
  }

  return {
    format: "ncu",
    timeColumn: "gpu__time_duration.sum",
    ...aggregate(samples),
    skipped: 0,
  };
}

export const NCU_COUNTERS = {
  globalLoad: [
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
  ],
  globalStore: [
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
  ],
  sharedLoad: [
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "smsp__inst_executed_op_shared_ld.sum",
  ],
  sharedStore: [
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "smsp__inst_executed_op_shared_st.sum",
  ],
};
export const NCU_COUNTER_METRICS = Object.values(NCU_COUNTERS).flat();

export function parseNcuCounters(text) {
  const rows = parseCSV(text).filter((row) => row.some((cell) => cell.trim()));

  const start = rows.findIndex((row) => {
    const lower = row.map((cell) => cell.trim().toLowerCase());

    return lower.includes("kernel name") && lower.includes("metric name");
  });

  if (start < 0) {
    throw new Error(
      "not ncu --csv output: no Kernel Name / Metric Name header",
    );
  }

  const col = (name) =>
    rows[start].findIndex((header) => header.trim().toLowerCase() === name);

  const [name, metric, value] = [
    col("kernel name"),
    col("metric name"),
    col("metric value"),
  ];

  const sums = new Map();

  for (const record of rows.slice(start + 1)) {
    const n = toNumber(record[value]);
    const key = record[metric]?.trim();
    if (n === null || !NCU_COUNTER_METRICS.includes(key)) continue;

    const kernel = record[name].trim();
    if (!sums.has(kernel)) sums.set(kernel, new Map());
    sums.get(kernel).set(key, (sums.get(kernel).get(key) ?? 0) + n);
  }

  if (!sums.size) {
    throw new Error(
      `ncu CSV has none of the counter metrics: ${NCU_COUNTER_METRICS.join(", ")}`,
    );
  }

  return new Map(
    [...sums].map(([kernel, metrics]) => [
      kernel,
      Object.fromEntries(
        Object.entries(NCU_COUNTERS).map(([kind, [num, den]]) => [
          kind,
          metrics.get(den) ? metrics.get(num) / metrics.get(den) : null,
        ]),
      ),
    ]),
  );
}

function aggregate(samples) {
  const byKey = new Map();

  for (const sample of samples) {
    const key = `${sample.symbol ?? ""}\u0000${sample.kernel}`;

    const entry = byKey.get(key) ?? {
      kernel: sample.kernel,
      symbol: sample.symbol,
      calls: 0,
      totalNs: 0,
    };

    entry.calls += sample.calls;
    entry.totalNs += sample.totalNs;
    byKey.set(key, entry);
  }

  const entries = [...byKey.values()].map((entry) => ({
    ...entry,
    timeNs: entry.calls ? entry.totalNs / entry.calls : entry.totalNs,
  }));

  return { entries };
}

function jsonRecords(data) {
  if (Array.isArray(data)) return data;

  for (const key of ["kernels", "results", "benchmarks"]) {
    if (Array.isArray(data?.[key])) return data[key];
  }

  return null;
}

export function parseBenchmarks(text) {
  const trimmed = text.trim();

  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    let data;

    try {
      data = JSON.parse(trimmed);
    } catch (error) {
      throw new Error(`not valid JSON: ${error.message}`);
    }

    const records = jsonRecords(data)?.filter(
      (record) => record && typeof record === "object",
    );

    if (!records?.length) {
      throw new Error("JSON has no array of results (or kernels, benchmarks)");
    }

    const runs = records.filter(
      (record) => !record.aggregate_name || record.aggregate_name === "mean",
    );

    const headers = [...new Set(runs.flatMap(Object.keys))];
    const table = runs.map((record) => headers.map((header) => record[header]));

    const format = records.some((record) => "time_unit" in record)
      ? "google-benchmark"
      : "json";

    return fromTable(headers, table, format);
  }

  const rows = parseCSV(text).filter((row) => row.some((cell) => cell.trim()));

  for (let i = 0; i < rows.length; i++) {
    const lower = rows[i].map((cell) => cell.trim().toLowerCase());

    if (lower.includes("kernel name") && lower.includes("metric name")) {
      return fromNcu(rows[i], rows.slice(i + 1));
    }

    if (rows[i].length > 1 && pickColumns(rows[i])) {
      return fromTable(rows[i], rows.slice(i + 1), "csv");
    }
  }

  throw new Error("no header row with a kernel name column");
}

export function cleanKernelName(name) {
  let clean = name.trim();
  const paren = clean.indexOf("(");
  if (paren > 0) clean = clean.slice(0, paren);

  let depth = 0;
  let out = "";

  for (const c of clean) {
    if (c === "<") depth += 1;
    else if (c === ">") depth = Math.max(0, depth - 1);
    else if (!depth) out += c;
  }

  clean = out.trim().split(/\s+/).at(-1) ?? "";

  return clean.slice(clean.lastIndexOf("::") + 1).replace(/^:/, "");
}

function normalizePath(symbol) {
  return symbol
    .split("::")
    .map(
      (part) =>
        `@${part
          .trim()
          .replace(/^@/, "")
          .replace(/^"(.*)"$/, "$1")}`,
    )
    .join("::");
}

const isFunction = (record) => record.ops.some((op) => /\.func$/.test(op));

export function matchBenchmarks(entries, history) {
  const byLeaf = new Map();

  for (const record of history) {
    const list = byLeaf.get(record.symbol) ?? [];
    list.push(record);
    byLeaf.set(record.symbol, list);
  }

  const byPath = new Map(history.map((record) => [record.path, record]));

  const pick = (records) => {
    const functions = records.filter(isFunction);

    return functions.length ? functions : records;
  };

  return entries.map((entry) => {
    if (entry.symbol) {
      const path = normalizePath(entry.symbol);

      return byPath.has(path)
        ? { entry, path, how: "symbol", candidates: [path] }
        : { entry, path: null, how: "symbol", candidates: [] };
    }

    const tiers = [
      ["exact", byLeaf.get(entry.kernel) ?? []],
      ["cleaned", byLeaf.get(cleanKernelName(entry.kernel)) ?? []],
      [
        "mangled",
        history.filter((record) =>
          entry.kernel.includes(`${record.symbol.length}${record.symbol}`),
        ),
      ],
    ];

    for (const [how, found] of tiers) {
      if (!found.length) continue;

      const candidates = pick(found).map((record) => record.path);

      return {
        entry,
        path: candidates.length === 1 ? candidates[0] : null,
        how,
        candidates,
      };
    }

    return { entry, path: null, how: null, candidates: [] };
  });
}

export function symbolTimes(entries, history) {
  const times = new Map();
  const unmatched = [];

  for (const match of matchBenchmarks(entries, history)) {
    if (!match.path) {
      unmatched.push(match);
      continue;
    }

    const time = times.get(match.path) ?? {
      calls: 0,
      totalNs: 0,
      kernels: [],
      how: [],
    };

    time.calls += match.entry.calls;
    time.totalNs += match.entry.totalNs;
    time.timeNs = time.calls ? time.totalNs / time.calls : time.totalNs;
    time.kernels.push(match.entry.kernel);
    time.how.push(match.how);
    times.set(match.path, time);
  }

  return { times, unmatched };
}

export function compareBenchmarks(baseline, current) {
  const paths = new Set([
    ...(baseline?.keys() ?? []),
    ...(current?.keys() ?? []),
  ]);

  const comparison = new Map();

  for (const path of paths) {
    const before = baseline?.get(path) ?? null;
    const after = current?.get(path) ?? null;
    const both = before && after;

    comparison.set(path, {
      baseline: before,
      current: after,
      deltaNs: both ? after.timeNs - before.timeNs : null,
      change:
        both && before.timeNs
          ? (after.timeNs - before.timeNs) / before.timeNs
          : null,
    });
  }

  return comparison;
}

export function changedBeyond(row, percent) {
  if (!row) return false;
  if (!row.baseline || !row.current) return true;

  return row.change !== null && Math.abs(row.change) * 100 > percent;
}

export function comparisonOrder(a, b) {
  if (!a || !b) return (b ? 1 : 0) - (a ? 1 : 0);

  if (a.change !== null || b.change !== null) {
    return (b.change ?? -Infinity) - (a.change ?? -Infinity);
  }

  const time = (row) => (row.current ?? row.baseline).timeNs;

  return time(b) - time(a);
}

export function comparisonToJSON(comparison) {
  const side = (time) =>
    time && { time_ns: time.timeNs, calls: time.calls, kernels: time.kernels };

  return new Map(
    [...comparison].map(([path, row]) => [
      path,
      {
        ...(row.baseline ? { baseline: side(row.baseline) } : {}),
        ...(row.current ? { current: side(row.current) } : {}),
        ...(row.change !== null
          ? { delta_ns: row.deltaNs, change: row.change }
          : {}),
      },
    ]),
  );
}

export function formatChange(change) {
  if (change === null || change === undefined) return "";

  const percent = change * 100;

  const text =
    Math.abs(percent) < 10
      ? Math.abs(percent).toFixed(1)
      : Math.abs(percent).toFixed(0);

  return `${signOf(percent)}${text}%`;
}

export function formatDuration(ns) {
  if (ns === null || ns === undefined) return "—";
  if (ns < 1e3) return `${ns.toFixed(0)} ns`;
  if (ns < 1e6) return `${(ns / 1e3).toFixed(ns < 1e4 ? 2 : 1)} µs`;
  if (ns < 1e9) return `${(ns / 1e6).toFixed(ns < 1e7 ? 2 : 1)} ms`;

  return `${(ns / 1e9).toFixed(2)} s`;
}
