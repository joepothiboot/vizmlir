// Line-level diff between two texts, for showing how one symbol's IR changed
// between two passes. The structural diff (diff.js) works on parsed ops; this
// one keeps the printed text, including attributes and types.

// Above this many cells the LCS table is skipped and the changed middle is
// shown as removed then added, which is still correct, only less precise.
const MAX_CELLS = 4_000_000;

// Returns [{ type: "same" | "add" | "del", text }] turning `before` into
// `after`. Either may be null or empty.
export function lineDiff(before, after) {
  const a = before ? before.split("\n") : [];
  const b = after ? after.split("\n") : [];

  let start = 0;
  while (start < a.length && start < b.length && a[start] === b[start]) start++;
  let endA = a.length;
  let endB = b.length;
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) {
    endA--;
    endB--;
  }

  const head = a.slice(0, start).map((text) => ({ type: "same", text }));
  const tail = a.slice(endA).map((text) => ({ type: "same", text }));
  const midA = a.slice(start, endA);
  const midB = b.slice(start, endB);

  if (midA.length * midB.length > MAX_CELLS)
    return [
      ...head,
      ...midA.map((text) => ({ type: "del", text })),
      ...midB.map((text) => ({ type: "add", text })),
      ...tail,
    ];

  // lcs[i][j] is the LCS length of midA[i..] and midB[j..].
  const n = midA.length;
  const m = midB.length;
  const lcs = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      lcs[i][j] =
        midA[i] === midB[j]
          ? lcs[i + 1][j + 1] + 1
          : Math.max(lcs[i + 1][j], lcs[i][j + 1]);

  const middle = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (midA[i] === midB[j]) {
      middle.push({ type: "same", text: midA[i] });
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      middle.push({ type: "del", text: midA[i++] });
    } else {
      middle.push({ type: "add", text: midB[j++] });
    }
  }
  while (i < n) middle.push({ type: "del", text: midA[i++] });
  while (j < m) middle.push({ type: "add", text: midB[j++] });

  return [...head, ...middle, ...tail];
}

export function diffStats(lines) {
  let added = 0;
  let removed = 0;
  for (const line of lines) {
    if (line.type === "add") added++;
    else if (line.type === "del") removed++;
  }
  return { added, removed };
}
