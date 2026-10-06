const RULE = /^===-+===\s*$/;
const TITLE = /^\s*\.\.\. Execution time report \.\.\.\s*$/;
const TOTAL_TIME = /^\s*Total Execution Time: ([\d.]+) seconds\s*$/;
const COLUMNS = /----(User|Wall) Time----/g;
const TIME = String.raw`([\d.]+) \(\s*([\d.]+)%\)`;
const ROW = new RegExp(String.raw`^\s*${TIME}(?:\s+${TIME})?  ( *)(\S.*?)\s*$`);
const JSON_ROW = /^\s*\{"(?:user|wall)": \{"duration"/;

const BSD_TIMES = /^\s*[\d.]+ real\s+[\d.]+ user\s+[\d.]+ sys\s*$/;
const BSD_STAT = /^\s*(\d+)\s+([a-z][a-z ]*[a-z])\s*$/;
const GNU_START = /^\s*Command being timed: /;
const GNU_RSS = /^\s*Maximum resident set size \(kbytes\): (\d+)\s*$/;
const GNU_END = /^\s*Exit status: -?\d+\s*$/;

export function extractReports(text) {
  const lines = text.split(/\r?\n/);
  let timing = null;
  let memory = null;

  for (let i = 0; i < lines.length; i++) {
    const found =
      !timing && (readTextReport(lines, i) ?? readJsonReport(lines, i));

    const measured = !found && !memory && readTimeOutput(lines, i);
    const block = found || measured;
    if (!block) continue;
    if (found) timing = found.report;
    else memory = measured.memory;
    for (let j = i; j < block.end; j++) lines[j] = "";
    i = block.end - 1;
  }

  if (!timing && !memory) return { text, timing, memory };

  return { text: lines.join("\n"), timing, memory };
}

export function hasTimingReport(text) {
  const probe = text.length > 262144 ? text.slice(-262144) : text;

  return /\.\.\. Execution time report \.\.\./.test(probe);
}

function readTextReport(lines, i) {
  if (!RULE.test(lines[i]) || !TITLE.test(lines[i + 1] ?? "")) return null;
  if (!RULE.test(lines[i + 2] ?? "")) return null;

  let total = null;
  let columns = ["wall"];
  const raw = [];
  let j = i + 3;

  for (; j < lines.length; j++) {
    const line = lines[j];
    const totalMatch = TOTAL_TIME.exec(line);

    if (totalMatch) {
      total = Number(totalMatch[1]);
      continue;
    }

    if (line.includes("----Name----")) {
      columns = [...line.matchAll(COLUMNS)].map((m) => m[1].toLowerCase());
      continue;
    }

    if (line.trim() === "") {
      if (raw.length) break;
      continue;
    }

    const row = ROW.exec(line);
    if (!row) break;

    const [, a, aPct, b, bPct, indent, name] = row;
    const times = [time(a, aPct), b === undefined ? null : time(b, bPct)];

    raw.push({
      name,
      depth: indent.length / 2,
      ...(columns.length === 2
        ? { user: times[0], wall: times[1] }
        : { wall: times[0] }),
    });

    if (name === "Total" && indent === "") {
      j += 1;
      break;
    }
  }

  if (!raw.length) return null;

  return { end: j, report: buildReport(raw, total, columns) };
}

function readJsonReport(lines, i) {
  if (lines[i].trim() !== "[" || !JSON_ROW.test(lines[i + 1] ?? "")) {
    return null;
  }

  let j = i + 1;
  while (j < lines.length && lines[j].trim() !== "]") j += 1;
  if (j >= lines.length) return null;

  let entries;

  try {
    entries = JSON.parse(lines.slice(i, j + 1).join("\n"));
  } catch {
    return null;
  }

  const raw = [];

  const walk = (list, depth) => {
    for (const entry of list) {
      if (!entry?.name) continue;

      raw.push({
        name: entry.name,
        depth,
        wall: jsonTime(entry.wall),
        ...(entry.user ? { user: jsonTime(entry.user) } : {}),
      });

      if (Array.isArray(entry.passes)) walk(entry.passes, depth + 1);
    }
  };

  walk(entries, 0);
  if (!raw.length) return null;

  const columns = raw.some((row) => row.user) ? ["user", "wall"] : ["wall"];
  const totalRow = raw.find((row) => row.name === "Total" && row.depth === 0);

  return {
    end: j + 1,
    report: buildReport(raw, totalRow?.wall.seconds ?? null, columns),
  };
}

function buildReport(raw, total, columns) {
  const display = displayOf(raw);
  const rows = [];
  const stack = [];
  let totalRow = null;

  for (const entry of raw) {
    if (entry.depth === 0 && entry.name === "Total") {
      totalRow = entry;
      continue;
    }

    if (entry.depth === 0 && entry.name === "root" && display === "list") {
      continue;
    }

    stack.length = entry.depth;

    const parent = stack.at(-1) ?? null;

    const row = {
      index: rows.length,
      ...entry,
      kind: kindOf(entry.name),
      parent: parent ? parent.index : -1,
      anchor: null,
    };

    const pipeline = /^'([^']+)' Pipeline$/.exec(entry.name);
    if (pipeline) row.anchor = pipeline[1];
    else if (parent) row.anchor = parent.anchor;
    rows.push(row);
    stack.push(row);
  }

  return {
    display,
    columns,
    total: total ?? totalRow?.wall.seconds ?? null,
    rows,
  };
}

function displayOf(raw) {
  if (raw.some((row) => row.depth > 0)) return "tree";

  const names = raw.map((row) => row.name);

  return new Set(names).size === names.length ? "list" : "tree";
}

function kindOf(name) {
  if (/^'[^']+' Pipeline$/.test(name)) return "pipeline";
  if (name.startsWith("Pipeline Collection")) return "pipeline";
  if (name.startsWith("(A) ")) return "analysis";
  if (["Parser", "Output", "Rest"].includes(name)) return "other";

  return "pass";
}

function time(seconds, percent) {
  return { seconds: Number(seconds), percent: Number(percent) };
}

function jsonTime(value) {
  return {
    seconds: Number(value?.duration ?? 0),
    percent: Number(value?.percentage ?? 0),
  };
}

function readTimeOutput(lines, i) {
  if (BSD_TIMES.test(lines[i])) {
    let bytes = null;
    let j = i + 1;

    for (; j < lines.length; j++) {
      const stat = BSD_STAT.exec(lines[j]);
      if (!stat) break;
      if (stat[2] === "maximum resident set size") bytes = Number(stat[1]);
    }

    return bytes === null
      ? null
      : { end: j, memory: { peakBytes: bytes, source: "time -l" } };
  }

  if (GNU_START.test(lines[i])) {
    let bytes = null;
    let j = i + 1;

    for (; j < lines.length; j++) {
      const rss = GNU_RSS.exec(lines[j]);
      if (rss) bytes = Number(rss[1]) * 1024;

      if (GNU_END.test(lines[j])) {
        j += 1;
        break;
      }
    }

    return bytes === null
      ? null
      : { end: j, memory: { peakBytes: bytes, source: "time -v" } };
  }

  return null;
}

export function matchTiming(events, timing) {
  const matches = events.map(() => null);
  if (!timing) return matches;

  const passes = timing.rows.filter((row) => row.kind === "pass");
  const seen = new Map();

  events.forEach((event, i) => {
    const on = event.anchor ?? event.root;
    const named = passes.filter((row) => row.name === event.pass);
    if (!named.length) return;

    if (timing.display === "list") {
      matches[i] = { row: named[0] };

      return;
    }

    const exact = named.filter((row) => row.anchor === on?.op);

    const candidates = exact.length
      ? exact
      : named.filter((row) => row.anchor === null);

    const key = `${event.phase}|${on?.op}@${on?.symbol ?? ""}|${event.pass}|${exact.length > 0}`;
    const nth = seen.get(key) ?? 0;
    seen.set(key, nth + 1);
    if (nth < candidates.length) matches[i] = { row: candidates[nth] };
  });

  const runs = new Map();

  events.forEach((event, i) => {
    if (!matches[i]) return;

    const key = `${event.phase}|${matches[i].row.index}`;
    runs.set(key, (runs.get(key) ?? 0) + 1);
  });

  events.forEach((event, i) => {
    if (matches[i]) {
      matches[i].runs = runs.get(`${event.phase}|${matches[i].row.index}`);
    }
  });

  return matches;
}

export function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 0.0001) return "<0.1 ms";
  if (seconds < 1) return `${(seconds * 1000).toFixed(1)} ms`;

  return `${seconds.toFixed(2)} s`;
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;

  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

export function byteLength(text) {
  let bytes = 0;

  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);

    if (code < 0x80) {
      bytes += 1;
    } else if (code < 0x800) {
      bytes += 2;
    } else if (code >= 0xd800 && code < 0xdc00) {
      bytes += 4;
      i += 1;
    } else {
      bytes += 3;
    }
  }

  return bytes;
}

export function timingToJSON(title, timing, memory, events, matches) {
  return JSON.stringify(
    {
      title,
      totalSeconds: timing?.total ?? null,
      peakMemoryBytes: memory?.peakBytes ?? null,
      peakMemorySource: memory?.source ?? null,
      columns: timing?.columns ?? [],
      rows: (timing?.rows ?? []).map((row) => ({
        name: row.name,
        kind: row.kind,
        depth: row.depth,
        parent: row.parent,
        anchor: row.anchor,
        wallSeconds: row.wall.seconds,
        wallPercent: row.wall.percent,
        ...(row.user
          ? { userSeconds: row.user.seconds, userPercent: row.user.percent }
          : {}),
      })),
      events: events.map((event, i) => ({
        index: event.index + 1,
        phase: event.phase,
        pass: event.pass,
        failed: event.failed,
        irBytes: byteLength(event.ir),
        timingRow: matches[i]?.row.index ?? null,
      })),
    },
    null,
    2,
  );
}
