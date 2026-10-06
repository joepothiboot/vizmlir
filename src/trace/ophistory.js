import { sourcePositions } from "./sources.js";

const posKey = (p) => `${p.file}:${p.line}:${p.col}`;

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

const sameSet = (a, b) =>
  a.length === b.length && a.every((k) => b.includes(k));

const within = (small, big) =>
  small.length > 0 && small.every((k) => big.includes(k));

export function matchPasses(before, after) {
  const links = new Map();
  const taken = new Set();

  const take = (b, a, kind, extra = {}) => {
    links.set(a, { from: [b], kind, ...extra });
    taken.add(b);
  };

  const unmatchedAfter = () =>
    after.map((_, i) => i).filter((i) => !links.has(i));

  const free = (i) => !taken.has(i);

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

  for (const a of unmatchedAfter()) {
    const r = after[a];
    if (!r.loc.startsWith("callsite") || !r.keys.length) continue;

    const callee = before.findIndex(
      (o) => o.keys.length && sameSet(o.keys, [r.keys[0]]),
    );

    if (callee >= 0) {
      links.set(a, { from: [callee], kind: "inlined", clone: true });
    }
  }

  for (const a of unmatchedAfter()) {
    const r = after[a];
    if (!r.loc.startsWith("fused") || !r.keys.length) continue;

    const parts = before
      .map((_, i) => i)
      .filter((i) => free(i) && within(before[i].keys, r.keys));

    if (!parts.length) continue;
    links.set(a, { from: parts, kind: "fused" });
    for (const p of parts) taken.add(p);
  }

  return links;
}

export function buildOpModel(records) {
  const links = [null];
  const forward = [null];

  for (let p = 1; p < records.length; p++) {
    const before = records[p - 1];
    const after = records[p];
    links.push(before && after ? matchPasses(before, after) : null);
  }

  for (let p = 0; p < records.length - 1; p++) forward[p] = new Map();

  for (let p = 1; p < records.length; p++) {
    for (const [node, link] of links[p] ?? []) {
      if (!link.clone) {
        for (const from of link.from) forward[p - 1].set(from, node);
      }
    }
  }

  return { records, links, forward };
}

export function opHistory(model, pass, node) {
  const { records, links, forward } = model;
  if (!records[pass]?.[node]) return [];

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
      if (at.pass + 1 < records.length && records[at.pass + 1]) {
        entries.push({
          pass: at.pass + 1,
          node: -1,
          label: record.label,
          loc: record.loc,
          change: "removed",
          from: [],
        });
      }

      break;
    }

    at = { pass: at.pass + 1, node: next };
  }

  return entries;
}
