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

  for (let e = 0; e < snapshot.edgeCount; e++) {
    edges.push([snapshot.edges[e * 2], snapshot.edges[e * 2 + 1]]);
  }

  return { parent, kind, line, edges };
}

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
  maxRows: 12,
  subWidth: 1.4,
  limit: 60,
});

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

  const start = [0];

  for (let k = 0; k < counts.length - 1; k++) {
    start.push(
      start[k] + cellX + (Math.ceil(counts[k] / maxRows) - 1) * subWidth,
    );
  }

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

  for (const p of pos) if (p) ((p[0] -= right / 2), (p[1] -= deepest / 2));

  return { pos, shown, hidden: d.length - shown, width: right, depth: deepest };
}

export function layoutStack(shapes, options = {}) {
  const layers = shapes.map((shape) =>
    shape ? layoutLayer(shape, options) : null,
  );

  let width = Math.max(0, ...layers.map((l) => l?.width ?? 0));
  let depth = Math.max(0, ...layers.map((l) => l?.depth ?? 0));
  const limit = options.limit ?? LAYOUT.limit;
  const k = Math.min(1, limit / Math.max(width, depth, 1e-9));

  if (k < 1) {
    for (const layer of layers) {
      if (layer) {
        for (const p of layer.pos) if (p) ((p[0] *= k), (p[1] *= k));
        layer.width *= k;
        layer.depth *= k;
      }
    }

    width *= k;
    depth *= k;
  }

  const gap = options.gap ?? LAYOUT.gap;

  return {
    layers,
    plate: { width: width + LAYOUT.cellX * 2, depth: depth + LAYOUT.cellZ * 4 },
    gap,
    shrink: k,
    heightOf: (pass) => (pass === 0 ? 0 : -pass * gap),
  };
}

export function nodePoint(stack, pass, node) {
  const at = stack.layers[pass]?.pos[node];

  return at ? [at[0], stack.heightOf(pass), at[1]] : null;
}

export function neighbours(shape, node) {
  const inputs = [];
  const outputs = [];

  for (const [from, to] of shape?.edges ?? []) {
    if (to === node) inputs.push(from);
    if (from === node) outputs.push(to);
  }

  return { inputs, outputs };
}

export function nodesOnLine(shape, line) {
  const out = [];
  if (!shape || line < 1) return out;

  shape.line.forEach((l, i) => {
    if (l === line) out.push(i);
  });

  return out;
}

export function lineageSegments(entries) {
  const out = [];

  for (const entry of entries) {
    if (entry.node >= 0) {
      for (const from of entry.from ?? []) {
        out.push({
          from: { pass: from.pass, node: from.node },
          to: { pass: entry.pass, node: entry.node },
          kind: entry.change,
        });
      }
    }
  }

  return out;
}

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

export function approach(current, target, dtMs, halfLife = 110) {
  const next = target + (current - target) * 0.5 ** (dtMs / halfLife);

  return Math.abs(next - target) < 0.01 ? target : next;
}
