import { THREAD_MAX, WARP } from "../constants.js";

const GRID_MAX = [16, 16, 4];
export const BLOCK = { size: 1, pitch: 1.35 };
const THREAD = { size: 0.52, pitch: 0.68 };
export const TILE = { size: 0.4, pitch: 0.48, rows: 32, cols: 32 };
const GAP = 6;
export const DISTANCE = 90;
export const HOME = { yaw: -0.32, pitch: 0.82 };
export const NODE = { size: 1.2, step: 2.4, gap: 4.6 };

export const CORNERS = [
  [0, 0, 0],
  [1, 0, 0],
  [1, 1, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [1, 1, 1],
  [0, 1, 1],
];

export const FACES = [
  { v: [3, 2, 6, 7], n: [0, 1, 0], light: 1 },
  { v: [0, 1, 5, 4], n: [0, -1, 0], light: 0.5 },
  { v: [0, 1, 2, 3], n: [0, 0, -1], light: 0.8 },
  { v: [4, 5, 6, 7], n: [0, 0, 1], light: 0.8 },
  { v: [0, 3, 7, 4], n: [-1, 0, 0], light: 0.64 },
  { v: [1, 2, 6, 5], n: [1, 0, 0], light: 0.64 },
];

const known = (dims) => [0, 1, 2].map((i) => Math.max(1, dims?.[i] ?? 1));

function coords(index, [x, y]) {
  return [index % x, Math.floor(index / x) % y, Math.floor(index / (x * y))];
}

export function layoutLaunch(grid, block) {
  const g = known(grid);
  const shown = g.map((d, i) => Math.min(d, GRID_MAX[i]));
  const blocks = [];

  for (let bz = 0; bz < shown[2]; bz++) {
    for (let by = 0; by < shown[1]; by++) {
      for (let bx = 0; bx < shown[0]; bx++) {
        blocks.push({
          kind: "block",
          id: [bx, by, bz],
          at: [bx * BLOCK.pitch, bz * BLOCK.pitch, -by * BLOCK.pitch],
          size: BLOCK.size,
        });
      }
    }
  }

  const gridWidth = (shown[0] - 1) * BLOCK.pitch + BLOCK.size;

  const b = known(block);
  const total = b[0] * b[1] * b[2];
  const wrap = b[0] > 64;
  const threads = [];
  const left = gridWidth + GAP;

  for (let t = 0; t < Math.min(total, THREAD_MAX); t++) {
    const id = coords(t, b);
    const [x, y, z] = wrap ? [t % WARP, Math.floor(t / WARP), 0] : id;

    threads.push({
      kind: "thread",
      id,
      warp: Math.floor(t / WARP),
      lane: t % WARP,
      at: [left + x * THREAD.pitch, z * THREAD.pitch, -y * THREAD.pitch],
      size: THREAD.size,
    });
  }

  return {
    blocks,
    threads,
    grid: g,
    block: b,
    clipped: {
      grid: shown.some((d, i) => d < g[i]),
      threads: total > THREAD_MAX,
    },
    shownGrid: shown,
  };
}

export const matrixWidth = (dims) =>
  dims.length >= 2 && dims.at(-1) >= 32 ? dims.at(-1) : null;

export function elementTiles({ access, memref, space }, result, order) {
  const dims = memref?.dims ?? [];
  const width = matrixWidth(dims) ?? 32;
  if (!width) return null;

  const at = (offset) => [Math.floor(offset / width), offset % width];
  const points = result.lanes.map((l) => at(l.offset));
  const rowStart = Math.min(...points.map((p) => p[0]));
  const rowEnd = Math.max(...points.map((p) => p[0]));
  const colMin = Math.min(...points.map((p) => p[1]));
  const colMax = Math.max(...points.map((p) => p[1]));
  const rows = Math.min(rowEnd - rowStart + 1, TILE.rows);

  const colStart = Math.max(
    0,
    Math.min(colMin, colMax - TILE.cols + 1, width - TILE.cols),
  );

  const cols = Math.min(width - colStart, TILE.cols);
  const cells = new Map();
  const of = new Map();

  result.lanes.forEach((lane, k) => {
    const [r, c] = [points[k][0] - rowStart, points[k][1] - colStart];
    if (r < 0 || r >= rows || c < 0 || c >= cols) return;
    cells.set(r * cols + c, { ...lane, order: order.get(lane.group) });
    of.set(lane.lane, [r, c]);
  });

  const unit = space === "shared" ? 128 : 32;

  const shade = (r, c) =>
    Math.floor(
      (((rowStart + r) * width + colStart + c) * result.elementBytes) / unit,
    ) %
      2 ===
    1;

  const label = matrixWidth(dims)
    ? `${access.buffer} · rows ${rowStart}–${rowStart + rows - 1} × columns ${colStart}–${colStart + cols - 1}`
    : `${access.buffer} · elements ${rowStart * width}–${(rowStart + rows) * width - 1}`;

  return { rows, cols, cells, of, shade, label };
}

export function bounds(cubes) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];

  for (const cube of cubes) {
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], cube.at[i]);
      max[i] = Math.max(max[i], cube.at[i] + cube.size);
    }
  }

  return { min, max };
}

export function camera({ yaw, pitch, center, scale, cx, cy }) {
  const [sy, cyaw] = [Math.sin(yaw), Math.cos(yaw)];
  const [sp, cp] = [Math.sin(pitch), Math.cos(pitch)];

  const turn = ([x, y, z]) => {
    const x1 = x * cyaw - z * sy;
    const z1 = x * sy + z * cyaw;

    return [x1, y * cp + z1 * sp, -y * sp + z1 * cp];
  };

  return {
    turn,
    view: (p) => turn([p[0] - center[0], p[1] - center[1], p[2] - center[2]]),
    screen: ([x, y, z]) => {
      const f = DISTANCE / (DISTANCE + z);

      return [cx + x * scale * f, cy - y * scale * f];
    },
  };
}

export function inside([px, py], poly) {
  let hit = false;

  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];

    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      hit = !hit;
    }
  }

  return hit;
}

export const idText = (id) => `(${id.join(", ")})`;
export const dimText = (dims) => dims.join(" × ");
