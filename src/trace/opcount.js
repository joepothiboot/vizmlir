import { KIND } from "../ir/index.js";

export function opName(label) {
  const space = label.indexOf(" ");

  return space < 0 ? label : label.slice(0, space);
}

export function countOps(snapshot) {
  const counts = new Map();
  if (!snapshot) return counts;

  for (let i = 0; i < snapshot.nodeCount; i++) {
    const node = snapshot.nodes?.[i];
    const kind = node ? node.kind : snapshot.kindOf(i);
    const parent = node ? node.parent : snapshot.parentOf(i);
    if (parent < 0 || kind === KIND.BLOCK) continue;

    const name = opName(node ? node.label : snapshot.labelOf(i));
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }

  return counts;
}

export function totalOps(counts) {
  let total = 0;
  for (const count of counts.values()) total += count;

  return total;
}

export function opCountTable(columns) {
  const names = new Set();

  for (const counts of columns) {
    for (const name of counts.keys()) names.add(name);
  }

  const rows = [...names].map((op) => {
    const counts = columns.map((column) => column.get(op) ?? 0);
    const delta = counts.length ? counts.at(-1) - counts[0] : 0;
    const changed = counts.some((count) => count !== counts[0]);

    return { op, counts, delta, changed };
  });

  rows.sort(
    (a, b) =>
      Math.abs(b.delta) - Math.abs(a.delta) ||
      Number(b.changed) - Number(a.changed) ||
      a.op.localeCompare(b.op),
  );

  const totals = columns.map(totalOps);

  const changedColumns = columns
    .map((_, i) => i)
    .filter(
      (i) => i > 0 && rows.some((row) => row.counts[i] !== row.counts[i - 1]),
    );

  return { rows, totals, changedColumns };
}

export function opCountsToCSV(headers, table) {
  const quote = (value) =>
    /[",\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;

  const lines = [
    ["op", ...headers, "delta"].map(quote).join(","),
    ["(all ops)", ...table.totals, table.totals.at(-1) - table.totals[0]].join(
      ",",
    ),
    ...table.rows.map((row) =>
      [quote(row.op), ...row.counts, row.delta].join(","),
    ),
  ];

  return `${lines.join("\n")}\n`;
}
