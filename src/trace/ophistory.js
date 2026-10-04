// Op history: follow one op through the passes of a trace. Ops in neighbouring
// passes are matched by their source location and name, so a lowered op, an
// inlined body and a fused op stay connected to where they came from. IR
// without locations falls back to matching by name and order.
//
// The model is built from per-pass "records" copied out of the parser (the
// engine reuses its memory, so a snapshot cannot be kept):
//   records[pass] = [{ label, op, loc, keys }] indexed by node, or null when
//   that pass's IR did not parse.

import { sourcePositions } from "./sources.js";

const posKey = (p) => `${p.file}:${p.line}:${p.col}`;

/** Copies what history needs out of a parser snapshot. */
export function opRecords(snapshot) {
  const out = [];
  for (let i = 0; i < snapshot.nodeCount; i++) {
    const label = snapshot.labelOf(i);
    const loc = snapshot.locOf?.(i)?.text ?? "";
    out.push({
      label,
      op: label.split(" ")[0],
      loc,
      keys: sourcePositions(loc).map(posKey),
    });
  }
  return out;
}

const sameSet = (a, b) => a.length === b.length && a.every((k) => b.includes(k));
const within = (small, big) => small.length > 0 && small.every((k) => big.includes(k));

/**
 * Matches the ops of one pass to the next. Returns links keyed by node of
 * `after`: { from: node indexes in `before`, kind, clone }.
 *  - kept:    same location and label
 *  - renamed: same location, different op (a lowering)
 *  - inlined: a call-site location whose callee is an op of `before`; the
 *             callee stays where it was (`clone`)
 *  - fused:   a fused location merging ops of `before`
 * Anything left over in `after` is new, and in `before` is gone.
 */
export function matchPasses(before, after) {
  const links = new Map();
  const taken = new Set();

  const take = (b, a, kind, extra = {}) => {
    links.set(a, { from: [b], kind, ...extra });
    taken.add(b);
  };
  const unmatchedAfter = () => after.map((_, i) => i).filter((i) => !links.has(i));
  const free = (i) => !taken.has(i);

  // Same location and label, in order.
  const buckets = new Map();
  before.forEach((r, i) => {
    const key = `${r.loc}\u0000${r.label}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(i);
  });
  after.forEach((r, a) => {
    const queue = buckets.get(`${r.loc}\u0000${r.label}`);
    if (queue?.length) take(queue.shift(), a, "kept");
  });

  // Same location, another op: lowered or renamed.
  const byLoc = new Map();
  before.forEach((r, i) => {
    if (!r.loc || !free(i)) return;
    if (!byLoc.has(r.loc)) byLoc.set(r.loc, []);
    byLoc.get(r.loc).push(i);
  });
  for (const a of unmatchedAfter()) {
    const queue = byLoc.get(after[a].loc);
    if (after[a].loc && queue?.length) take(queue.shift(), a, "renamed");
  }

  // Inlined: the callee is the first position of a call-site location.
  for (const a of unmatchedAfter()) {
    const r = after[a];
    if (!r.loc.startsWith("callsite") || !r.keys.length) continue;
    const callee = before.findIndex((o) => o.keys.length && sameSet(o.keys, [r.keys[0]]));
    if (callee >= 0) links.set(a, { from: [callee], kind: "inlined", clone: true });
  }

  // Fused: a fused location naming the locations of ops that are gone.
  for (const a of unmatchedAfter()) {
    const r = after[a];
    if (!r.loc.startsWith("fused") || !r.keys.length) continue;
    const parts = before.map((_, i) => i).filter((i) => free(i) && within(before[i].keys, r.keys));
    if (!parts.length) continue;
    links.set(a, { from: parts, kind: "fused" });
    for (const p of parts) taken.add(p);
  }
  return links;
}

/**
 * @param {Array<Array<object> | null>} records per pass; null for a pass that failed to parse
 */
export function buildOpModel(records) {
  const links = [null]; // links[p]: how pass p's ops came from pass p-1's
  const forward = [null]; // forward[p]: node of pass p -> node of pass p+1
  for (let p = 1; p < records.length; p++) {
    const before = records[p - 1];
    const after = records[p];
    links.push(before && after ? matchPasses(before, after) : null);
  }
  for (let p = 0; p < records.length - 1; p++) forward[p] = new Map();
  for (let p = 1; p < records.length; p++)
    for (const [node, link] of links[p] ?? [])
      if (!link.clone) for (const from of link.from) forward[p - 1].set(from, node);
  return { records, links, forward };
}

/**
 * The life of one op: an entry per pass it exists in, from where it first
 * appears, then a final entry for the pass that removed it (if any).
 * Entry: { pass, node, label, loc, change, from? } where `change` is how it
 * got here: created, inlined, fused, renamed or kept, and `from` the ops it
 * came from (fused or inlined) as { pass, node, label }.
 * Returns [] for a pass that failed to parse or a node out of range.
 */
export function opHistory(model, pass, node) {
  const { records, links, forward } = model;
  if (!records[pass]?.[node]) return [];

  // Back to where the op begins. A fused op continues through its first part.
  let p = pass;
  let n = node;
  while (p > 0) {
    const link = links[p]?.get(n);
    if (!link || link.kind === "inlined" || link.kind === "fused") break;
    n = link.from[0];
    p -= 1;
  }
  const start = { pass: p, node: n };

  const entries = [];
  let at = start;
  while (at) {
    const record = records[at.pass][at.node];
    const link = at.pass > 0 ? links[at.pass]?.get(at.node) : null;
    entries.push({
      pass: at.pass,
      node: at.node,
      label: record.label,
      loc: record.loc,
      change: link ? link.kind : "created",
      from: link
        ? link.from.map((i) => ({
            pass: at.pass - 1,
            node: i,
            label: records[at.pass - 1][i].label,
          }))
        : [],
    });
    const next = forward[at.pass]?.get(at.node);
    if (next === undefined) {
      if (at.pass + 1 < records.length && records[at.pass + 1])
        entries.push({
          pass: at.pass + 1,
          node: -1,
          label: record.label,
          loc: record.loc,
          change: "removed",
          from: [],
        });
      break;
    }
    at = { pass: at.pass + 1, node: next };
  }
  return entries;
}
