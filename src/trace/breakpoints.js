// Conditional breakpoints over the passes of a trace. A condition is a line of
// text; evaluating it over the trace gives the passes where it first becomes
// true, which is where a run forward or back stops.
//
//   linalg.matmul == 0     how many of an op the module has (== != < <= > >=)
//   ops < 20               how many ops in all
//   live > 1MB             peak live buffer bytes (B, KB, MB, GB; 1024-based)
//   appears gpu.launch     the op is in this pass and was not in the last
//   gone func.call         the op was in the last pass and is not in this one
//   fails                  the pass failed
//   diag                   the pass printed a diagnostic
//
// A comparison "hits" at a pass where it is true and was not in the pass
// before, so a count that stays at zero for six passes stops once.

const COMPARE = {
  "==": (a, b) => a === b,
  "!=": (a, b) => a !== b,
  "<": (a, b) => a < b,
  "<=": (a, b) => a <= b,
  ">": (a, b) => a > b,
  ">=": (a, b) => a >= b,
};

const UNITS = { "": 1, b: 1, k: 1024, kb: 1024, m: 1024 ** 2, mb: 1024 ** 2, g: 1024 ** 3, gb: 1024 ** 3 };

const SIZE = /^(\d+(?:\.\d+)?)\s*([a-z]*)$/i;

function parseSize(text) {
  const m = SIZE.exec(text.trim());
  if (!m) return null;
  const unit = UNITS[m[2].toLowerCase()];
  return unit === undefined ? null : Math.round(Number(m[1]) * unit);
}

/** @returns {{ ok: true, cond: object } | { ok: false, error: string }} */
export function parseCondition(text) {
  const source = text.trim().replace(/\s+/g, " ");
  const fail = (error) => ({ ok: false, error });
  if (!source) return fail("Type a condition, such as linalg.matmul == 0.");

  if (source === "fails") return { ok: true, cond: { kind: "fails", text: source } };
  if (source === "diag") return { ok: true, cond: { kind: "diag", text: source } };

  const word = /^(appears|gone) (\S+)$/.exec(source);
  if (word) return { ok: true, cond: { kind: word[1], op: word[2], text: source } };
  if (/^(appears|gone)\b/.test(source)) return fail(`"${source.split(" ")[0]}" needs an op name, as in appears gpu.launch.`);

  const compare = /^(\S+?) ?(==|!=|<=|>=|<|>|=) ?(.+)$/.exec(source);
  if (!compare) return fail(`Cannot read "${source}". Try linalg.matmul == 0, ops < 20, live > 1MB, appears gpu.launch, gone func.call, fails or diag.`);
  const [, subject, rawCmp, rawValue] = compare;
  const cmp = rawCmp === "=" ? "==" : rawCmp;

  if (subject === "live") {
    const value = parseSize(rawValue);
    if (value === null) return fail(`"${rawValue}" is not a size. Use a number with B, KB, MB or GB.`);
    return { ok: true, cond: { kind: "live", cmp, value, text: `live ${cmp} ${rawValue.trim()}` } };
  }
  if (!/^\d+$/.test(rawValue.trim())) return fail(`"${rawValue}" is not a whole number.`);
  const value = Number(rawValue);
  if (subject === "ops") return { ok: true, cond: { kind: "total", cmp, value, text: `ops ${cmp} ${value}` } };
  return { ok: true, cond: { kind: "count", op: subject, cmp, value, text: `${subject} ${cmp} ${value}` } };
}

/**
 * The passes where `cond` hits, in order.
 * @param {object} ctx
 *   events:  the trace's events (`failed`, `diagnostics`)
 *   counts:  () => per pass, op name -> count, or null where the IR did not parse
 *   peaks:   () => per pass, { peak } in bytes, or null
 * `counts` and `peaks` are only called when a condition needs them.
 */
export function conditionHits(cond, ctx) {
  const n = ctx.events.length;
  const hits = [];
  const hit = (i, now, before) => {
    if (now && !before) hits.push(i);
  };

  if (cond.kind === "fails" || cond.kind === "diag") {
    for (let i = 0; i < n; i++) {
      const e = ctx.events[i];
      if (cond.kind === "fails" ? e.failed : e.diagnostics?.length > 0) hits.push(i);
    }
    return hits;
  }

  if (cond.kind === "live") {
    const peaks = ctx.peaks();
    const holds = (i) => peaks[i] != null && COMPARE[cond.cmp](peaks[i].peak, cond.value);
    for (let i = 0; i < n; i++) hit(i, holds(i), i > 0 && holds(i - 1));
    return hits;
  }

  const counts = ctx.counts();
  const count = (i, op) => {
    const column = counts[i];
    if (!column) return null;
    if (cond.kind === "total" || op === undefined) {
      let total = 0;
      for (const c of column.values()) total += c;
      return total;
    }
    return column.get(op) ?? 0;
  };

  if (cond.kind === "appears" || cond.kind === "gone") {
    for (let i = 1; i < n; i++) {
      const before = count(i - 1, cond.op);
      const now = count(i, cond.op);
      if (before === null || now === null) continue;
      if (cond.kind === "appears" ? before === 0 && now > 0 : before > 0 && now === 0) hits.push(i);
    }
    return hits;
  }

  const holds = (i) => {
    const value = count(i, cond.kind === "total" ? undefined : cond.op);
    return value !== null && COMPARE[cond.cmp](value, cond.value);
  };
  for (let i = 0; i < n; i++) hit(i, holds(i), i > 0 && holds(i - 1));
  return hits;
}

/** The first hit after `from` (delta 1) or before it (delta -1), or -1. */
export function nextHit(hits, from, delta) {
  return delta > 0
    ? (hits.find((h) => h > from) ?? -1)
    : (hits.findLast((h) => h < from) ?? -1);
}

/** The sorted union of several hit lists. */
export function mergeHits(lists) {
  return [...new Set(lists.flat())].sort((a, b) => a - b);
}
