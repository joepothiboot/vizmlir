// Source locations, read from the `loc(...)` of each op: which lines of which
// source file an op came from, and the other way round. Source text is
// matched to the file names in the IR; nothing here touches the page.

// Every `"file":line:col` in a resolved location, in order. The first is where
// the op was written; a call site adds the caller after it, and a fused op one
// per op it merged.
export function sourcePositions(locText) {
  const out = [];
  for (const m of locText.matchAll(/"((?:[^"\\]|\\.)*)":(\d+):(\d+)/g))
    out.push({ file: m[1], line: Number(m[2]), col: Number(m[3]) });
  return out;
}

const baseName = (path) => path.slice(path.lastIndexOf("/") + 1);

// Whether a file named in the IR is the same file as one that was loaded:
// the paths agree, or one is the tail of the other (`/work/a/kernel.mojo` and
// `a/kernel.mojo`), or only the file names agree.
export function sameFile(a, b) {
  if (a === b) return true;
  if (a.endsWith(`/${b}`) || b.endsWith(`/${a}`)) return true;
  return baseName(a) === baseName(b);
}

// The key of the loaded source for `file`, preferring an exact match.
export function findSource(sources, file) {
  const names = Object.keys(sources);
  return (
    names.find((name) => name === file) ??
    names.find((name) => sameFile(name, file)) ??
    null
  );
}

// The location index of one snapshot, copied out of the engine's memory so it
// survives the next parse: `files` the distinct names in the IR, `byNode` each
// node's positions, `byFile` the nodes at each line of each file.
export function buildLocIndex(snapshot) {
  const files = [];
  const byNode = new Map();
  const byFile = new Map();
  for (let node = 0; node < snapshot.nodeCount; node++) {
    const loc = snapshot.locOf?.(node);
    if (!loc) continue;
    const positions = sourcePositions(loc.text);
    if (!positions.length) continue;
    byNode.set(node, { positions, text: loc.text });
    for (const { file, line } of positions) {
      if (!byFile.has(file)) {
        byFile.set(file, new Map());
        files.push(file);
      }
      const lines = byFile.get(file);
      if (!lines.has(line)) lines.set(line, []);
      const nodes = lines.get(line);
      if (nodes[nodes.length - 1] !== node) nodes.push(node);
    }
  }
  return { files, byNode, byFile };
}

// Ops at `line` of `file`, matching file names the way `sameFile` does.
export function nodesAt(index, file, line) {
  const out = [];
  for (const [name, lines] of index.byFile)
    if (sameFile(name, file)) for (const n of lines.get(line) ?? []) out.push(n);
  return out.sort((a, b) => a - b);
}

// The lines of `file` that node `node` came from, as { file, lines } for the
// first file it names, or null when it has no location.
export function linesOf(index, node) {
  const entry = index.byNode.get(node);
  if (!entry) return null;
  const file = entry.positions[0].file;
  return {
    file,
    lines: entry.positions.filter((p) => sameFile(p.file, file)).map((p) => p.line),
  };
}

// How many ops each line of `file` produced.
export function opCounts(index, file) {
  const counts = new Map();
  for (const [name, lines] of index.byFile)
    if (sameFile(name, file))
      for (const [line, nodes] of lines)
        counts.set(line, (counts.get(line) ?? 0) + nodes.length);
  return counts;
}
