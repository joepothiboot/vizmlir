// The geometry behind the Descent view: the IR of every pass as a layer in a
// stack, an op's life drawn as a path down through the layers. This file is
// plain data and arithmetic (no page, no canvas) so it can be tested.
//
// A pass's IR is a "shape": { parent, kind, line, edges }, indexed by node,
// copied out of the parser (see `snapshotShape`). A layer is that shape laid
// out as a tree by nesting depth, like the 2D graph: a column per depth, ops
// of one depth stacked in a row.

/** Copies what the layout needs out of a parser snapshot. */
export function snapshotShape(snapshot) {
  const n = snapshot.nodeCount;
  const parent = new Array(n);
  const kind = new Array(n);
  const line = new Array(n);
  for (let i = 0; i < n; i++) {
    parent[i] = snapshot.parentOf(i);
    kind[i] = snapshot.kindOf(i);
    line[i] = snapshot.irLineOf?.(i) ?? 0;
  }
  const edges = [];
  for (let e = 0; e < snapshot.edgeCount; e++)
    edges.push([snapshot.edges[e * 2], snapshot.edges[e * 2 + 1]]);
  return { parent, kind, line, edges };
}

/** Nesting depth of each node (the module is 0). Parents come before children. */
export function depths(shape) {
  const out = new Array(shape.parent.length).fill(0);
  for (let i = 0; i < out.length; i++) {
    const p = shape.parent[i];
    out[i] = p >= 0 && p < i ? out[p] + 1 : 0;
  }
  return out;
}

export const LAYOUT = Object.freeze({
  cellX: 3,
  cellZ: 1.5,
  gap: 6,
  maxNodes: 400,
  // A column taller than this wraps into side-by-side sub-columns, so a
  // layer with hundreds of ops at one depth stays a compact plate instead of
  // a strip far longer than the camera's perspective can show.
  maxRows: 12,
  subWidth: 1.4,
  // No plate is laid out wider or deeper than this, in world units.
  limit: 60,
});

/**
 * One layer's nodes on its plate, centred on (0, 0): `pos[i] = [x, z]`, or
 * null for a node past `maxNodes`, which is not drawn. `width` and `depth`
 * are the extent of what is drawn.
 */
export function layoutLayer(
  shape,
  {
    cellX = LAYOUT.cellX,
    cellZ = LAYOUT.cellZ,
    maxNodes = LAYOUT.maxNodes,
    maxRows = LAYOUT.maxRows,
    subWidth = LAYOUT.subWidth,
  } = {},
) {
  const d = depths(shape);
  const shown = Math.min(maxNodes, d.length);
  const counts = [];
  for (let i = 0; i < shown; i++) counts[d[i]] = (counts[d[i]] ?? 0) + 1;
  // Where each depth's columns start: one cell apart, plus a sub-column's
  // width for each time the column wraps.
  const start = [0];
  for (let k = 0; k < counts.length - 1; k++)
    start.push(start[k] + cellX + (Math.ceil(counts[k] / maxRows) - 1) * subWidth);
  const seen = [];
  const pos = new Array(d.length).fill(null);
  let right = 0;
  let deepest = 0;
  for (let i = 0; i < shown; i++) {
    const row = seen[d[i]] ?? 0;
    seen[d[i]] = row + 1;
    const x = start[d[i]] + Math.floor(row / maxRows) * subWidth;
    const z = (row % maxRows) * cellZ;
    pos[i] = [x, z];
    right = Math.max(right, x);
    deepest = Math.max(deepest, z);
  }
  for (const p of pos) if (p) (p[0] -= right / 2), (p[1] -= deepest / 2);
  return { pos, shown, hidden: d.length - shown, width: right, depth: deepest };
}

/**
 * Every pass as a layer. Passes whose IR did not parse (null shapes) get an
 * empty layer so the stack keeps its count. All plates are the size of the
 * largest layer, and layer `p` lies at height `-p * gap`.
 */
export function layoutStack(shapes, options = {}) {
  const layers = shapes.map((shape) => (shape ? layoutLayer(shape, options) : null));
  let width = Math.max(0, ...layers.map((l) => l?.width ?? 0));
  let depth = Math.max(0, ...layers.map((l) => l?.depth ?? 0));
  // However large the module, keep the stack inside the camera's range: shrink
  // every layer together, so layers still line up.
  const limit = options.limit ?? LAYOUT.limit;
  const k = Math.min(1, limit / Math.max(width, depth, 1e-9));
  if (k < 1) {
    for (const layer of layers)
      if (layer) {
        for (const p of layer.pos) if (p) (p[0] *= k), (p[1] *= k);
        layer.width *= k;
        layer.depth *= k;
      }
    width *= k;
    depth *= k;
  }
  const gap = options.gap ?? LAYOUT.gap;
  return {
    layers,
    plate: { width: width + LAYOUT.cellX * 2, depth: depth + LAYOUT.cellZ * 4 },
    gap,
    // How much the layout was shrunk (1 for none); nodes are drawn at this
    // fraction of their usual size.
    shrink: k,
    heightOf: (pass) => (pass === 0 ? 0 : -pass * gap),
  };
}

/** World position [x, y, z] of node `node` in layer `pass`, or null. */
export function nodePoint(stack, pass, node) {
  const at = stack.layers[pass]?.pos[node];
  return at ? [at[0], stack.heightOf(pass), at[1]] : null;
}

/** The nodes that feed `node` and the ones it feeds, within one layer. */
export function neighbours(shape, node) {
  const inputs = [];
  const outputs = [];
  for (const [from, to] of shape?.edges ?? []) {
    if (to === node) inputs.push(from);
    if (from === node) outputs.push(to);
  }
  return { inputs, outputs };
}

/** The nodes printed on `line` (1-based) of a pass's IR. */
export function nodesOnLine(shape, line) {
  const out = [];
  if (!shape || line < 1) return out;
  shape.line.forEach((l, i) => {
    if (l === line) out.push(i);
  });
  return out;
}

/**
 * The segments of an op's life, from `opHistory` entries. Every entry lists
 * the ops it came from (its earlier self when it was kept or renamed, the
 * callee when inlined, each merged op when fused), so a segment runs from each
 * of those to the entry. An entry for a removed op has no node and ends the
 * path. Each segment: { from: {pass, node}, to: {pass, node}, kind }.
 */
export function lineageSegments(entries) {
  const out = [];
  for (const entry of entries)
    if (entry.node >= 0)
      for (const from of entry.from ?? [])
        out.push({
          from: { pass: from.pass, node: from.node },
          to: { pass: entry.pass, node: entry.node },
          kind: entry.change,
        });
  return out;
}

/** A point `t` (0 … 1) of the way along a polyline of [x, y, z] points. */
export function pointAlong(points, t) {
  if (!points.length) return null;
  if (points.length === 1 || t <= 0) return points[0];
  if (t >= 1) return points.at(-1);
  const lengths = [];
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const d = Math.hypot(...points[i].map((v, k) => v - points[i - 1][k]));
    lengths.push(d);
    total += d;
  }
  if (total === 0) return points[0];
  let left = t * total;
  for (let i = 0; i < lengths.length; i++) {
    if (left <= lengths[i] || i === lengths.length - 1) {
      const f = lengths[i] === 0 ? 0 : left / lengths[i];
      return points[i].map((v, k) => v + (points[i + 1][k] - v) * f);
    }
    left -= lengths[i];
  }
  return points.at(-1);
}

/**
 * Moves `current` toward `target` by exponential smoothing: it covers half
 * the remaining distance every `halfLife` ms, whatever the frame rate. Snaps
 * within 0.01 (a fraction of a pixel at any zoom the view allows), so an
 * animation ends instead of creeping.
 */
export function approach(current, target, dtMs, halfLife = 110) {
  const next = target + (current - target) * 0.5 ** (dtMs / halfLife);
  return Math.abs(next - target) < 0.01 ? target : next;
}
