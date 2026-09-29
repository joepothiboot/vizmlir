// The 3D picture of one kernel launch, drawn on a <canvas>: the grid of
// blocks laid out by their (x, y, z) ids, one block opened into its threads
// (rows of 32 are warps), and floor plates for the memory each level reaches.
// Drag to turn it, shift-drag to pan, ⌘/Ctrl + scroll (or pinch) to zoom,
// hover for ids, click a block to open it. Plain canvas 2D with a small
// perspective projection: no WebGL and no library.

import { flowSlice, kernelFlow } from "./gpu-flow.js";
import { formatBytes } from "./timing.js";

const WARP = 32;
// Drawing limits; larger launches are drawn in part and the hint says so.
const GRID_MAX = [16, 16, 4];
const THREAD_MAX = 1024;
const BLOCK = { size: 1, pitch: 1.35 };
const THREAD = { size: 0.52, pitch: 0.68 };
// Elements of the picked access on the memory plate.
const TILE = { size: 0.4, pitch: 0.48, rows: 32, cols: 32 };
// Gap between the grid and the opened block, in world units.
const GAP = 6;
// Distance from the camera to the scene's center: smaller is more perspective.
const DISTANCE = 90;
const HOME = { yaw: -0.32, pitch: 0.82 };
// The kernel's IR lies on a board in front of the launch: ranks left to
// right, a rank's nodes in a column running toward the viewer.
const NODE = { size: 1.2, step: 2.4, gap: 4.6 };

// A unit cube: corners, and faces as corner loops with outward normals and
// how much light they catch.
const CORNERS = [
  [0, 0, 0],
  [1, 0, 0],
  [1, 1, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [1, 1, 1],
  [0, 1, 1],
];
const FACES = [
  { v: [3, 2, 6, 7], n: [0, 1, 0], light: 1 },
  { v: [0, 1, 5, 4], n: [0, -1, 0], light: 0.5 },
  { v: [0, 1, 2, 3], n: [0, 0, -1], light: 0.8 },
  { v: [4, 5, 6, 7], n: [0, 0, 1], light: 0.8 },
  { v: [0, 3, 7, 4], n: [-1, 0, 0], light: 0.64 },
  { v: [1, 2, 6, 5], n: [1, 0, 0], light: 0.64 },
];

const known = (dims) => [0, 1, 2].map((i) => Math.max(1, dims?.[i] ?? 1));
// Linear index → (x, y, z) for `dims`.
function coords(index, [x, y]) {
  return [index % x, Math.floor(index / x) % y, Math.floor(index / (x * y))];
}

// World layout, in units where a block is 1 wide. Block ids map x → x,
// y → -z (row 0 at the back, like the top row of a page), z → up. Threads
// use their own ids the same way, except a wide 1-D block, which wraps into
// rows of 32 so each row is a warp.
export function layoutLaunch(grid, block) {
  const g = known(grid);
  const shown = g.map((d, i) => Math.min(d, GRID_MAX[i]));
  const blocks = [];
  for (let bz = 0; bz < shown[2]; bz++)
    for (let by = 0; by < shown[1]; by++)
      for (let bx = 0; bx < shown[0]; bx++)
        blocks.push({
          kind: "block",
          id: [bx, by, bz],
          at: [bx * BLOCK.pitch, bz * BLOCK.pitch, -by * BLOCK.pitch],
          size: BLOCK.size,
        });
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

// The window of the picked access's buffer drawn on the memory plate: rows by
// the first index, or memory as rows of 32 elements for a 1-D buffer and one
// whose rows are narrower than 32 (such as {x, y} pairs), at most TILE.rows ×
// TILE.cols around the elements warp 0 touches. `cells` maps a tile to the
// lane that touches it, `of` a lane to its tile, and `shade(r, c)` alternates
// by 32-byte sector (global) or by 128-byte row of the 32 banks (shared).
export const matrixWidth = (dims) => (dims.length >= 2 && dims.at(-1) >= 32 ? dims.at(-1) : null);
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
  const colStart = Math.max(0, Math.min(colMin, colMax - TILE.cols + 1, width - TILE.cols));
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
    Math.floor((((rowStart + r) * width + colStart + c) * result.elementBytes) / unit) % 2 === 1;
  const label =
    matrixWidth(dims)
      ? `${access.buffer} · rows ${rowStart}–${rowStart + rows - 1} × columns ${colStart}–${colStart + cols - 1}`
      : `${access.buffer} · elements ${rowStart * width}–${(rowStart + rows) * width - 1}`;
  return { rows, cols, cells, of, shade, label };
}

// The box [min, max] around a list of cubes.
function bounds(cubes) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const cube of cubes)
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], cube.at[i]);
      max[i] = Math.max(max[i], cube.at[i] + cube.size);
    }
  return { min, max };
}

// A camera turned by `yaw` about the vertical axis and tilted by `pitch`,
// looking at `center`. `view` gives camera space (z grows away from the
// viewer); `screen` projects it with perspective, `scale` pixels per unit.
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

function inside([px, py], poly) {
  let hit = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi)
      hit = !hit;
  }
  return hit;
}

// Colors come from the page's theme tokens, read at draw time so a theme
// switch just redraws.
function rgb(text) {
  const hex = /^#([0-9a-f]{6})$/i.exec(text.trim());
  if (hex) return [0, 2, 4].map((i) => parseInt(hex[1].slice(i, i + 2), 16));
  const parts = /rgba?\(([^)]+)\)/.exec(text)?.[1].split(",").map(Number);
  return parts ? parts.slice(0, 3) : [128, 128, 128];
}
const mix = (a, b, t) => a.map((v, i) => v * t + b[i] * (1 - t));
const css = (c, alpha = 1) =>
  `rgba(${c.map((v) => Math.round(v)).join(", ")}, ${alpha})`;

function palette() {
  const style = getComputedStyle(document.documentElement);
  const token = (name) => rgb(style.getPropertyValue(name) || "#808080");
  const t = {
    bg: token("--bg"),
    raised: token("--raised"),
    fg: token("--fg"),
    dim: token("--dim"),
    good: token("--good"),
    warn: token("--warn"),
    accent: token("--accent"),
  };
  return {
    ...t,
    font: style.getPropertyValue("--ui") || "sans-serif",
    block: mix(t.dim, t.bg, 0.4),
    open: t.accent,
    hover: t.warn,
    warpA: mix(t.good, t.raised, 0.5),
    warpB: mix(t.good, t.raised, 0.32),
    faded: mix(t.good, t.raised, 0.14),
    provenGood: mix(t.good, t.bg, 0.35),
    provenBad: mix(t.warn, t.bg, 0.35),
    // Same order as the lane strip in Memory accesses (.g0 … .g3).
    groups: [
      mix(t.good, t.raised, 0.8),
      mix(t.warn, t.raised, 0.8),
      mix(t.good, t.raised, 0.4),
      mix(t.warn, t.raised, 0.4),
    ],
    global: mix(t.dim, t.bg, 0.16),
    shared: mix(t.warn, t.bg, 0.2),
  };
}

// Redraw every live scene when the theme changes.
const scenes = new Set();
if (typeof MutationObserver !== "undefined" && typeof document !== "undefined")
  new MutationObserver(() => {
    for (const scene of scenes)
      if (scene.canvas.isConnected) scene.schedule();
      else if (scene.shown) scenes.delete(scene);
  }).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const idText = (id) => `(${id.join(", ")})`;
const dimText = (dims) => dims.join(" × ");

// Builds the figure for `launch` (grid/block dims) and `kernel` (for its
// buffers and IR). Returns { figure, showAccess, showSpace, showLine,
// overlay }. showLine(line) lights the kernel IR nodes on that source line
// and everything they read or feed, and returns how many it lit.
// showAccess({ access, result }) colors warp 0's threads by the sector or
// bank each one hits, showSpace(space) outlines that memory's floor plate,
// overlay(facts, panel) lays the launch facts and a Memory panel over the
// canvas, so the whole launch reads in one place, and showElements(node)
// sets the picked access's elements seen from above: a layer over the canvas
// that the Elements tool (e) turns on and off.
export function gpuScene(launch, kernel, { onLine } = {}) {
  const layout = layoutLaunch(launch.grid, launch.block);
  const floor = [...layout.blocks, ...layout.threads];
  // The floor: blocks and threads, which the memory plates sit under.
  const { min, max } = bounds(floor);
  const threadBox = layout.threads.length ? bounds(layout.threads) : null;
  // The picked access's elements lie on the memory plate right of the opened
  // block, row 0 level with the block's far row (warp 0), in a window of up
  // to 32 × 32 that is reserved whenever the kernel has loads or stores, so
  // picking another one does not refit the view.
  const tileArea =
    threadBox && kernel?.accesses?.length
      ? { x: threadBox.max[0] + 2, far: threadBox.max[2] }
      : null;
  const tileBounds = tileArea
    ? [
        { at: [tileArea.x, 0, tileArea.far - TILE.rows * TILE.pitch], size: 0 },
        { at: [tileArea.x + TILE.cols * TILE.pitch, 0, tileArea.far], size: 0 },
      ]
    : [];
  const plateBox = bounds([...floor, ...tileBounds]);
  const threadCube = new Map(layout.threads.map((cube) => [cube.id.join(","), cube]));

  // The kernel IR as cubes on the board, one per flow node.
  const flow = kernelFlow(kernel);
  const ranks = Math.max(1, ...flow.nodes.map((n) => n.rank + 1));
  const stacked = new Map();
  const boardFar = min[2] - NODE.gap;
  const nodes = [...flow.nodes]
    .sort((a, b) => a.rank - b.rank || a.line - b.line)
    .map((node) => {
      const level = stacked.get(node.rank) ?? 0;
      stacked.set(node.rank, level + 1);
      const x =
        min[0] +
        (node.rank / Math.max(1, ranks - 1)) * (max[0] - min[0] - NODE.size);
      return {
        kind: "node",
        node,
        at: [x, 0, boardFar - NODE.size - level * NODE.step],
        size: NODE.size,
      };
    });
  const byId = new Map(nodes.map((cube) => [cube.node.id, cube]));
  const boardNear = Math.min(boardFar, ...nodes.map((c) => c.at[2]));
  const spaceOf = new Map(
    (kernel?.buffers ?? []).map((b) => [b.name, b.space]),
  );
  let showIR = nodes.length > 0;
  let slice = null;
  let focusLine = null;
  // What is drawn and fitted: the floor, plus the IR wall when it is shown.
  let all;
  let center;
  let box;
  const rebuild = () => {
    all = showIR ? [...floor, ...nodes] : floor;
    box = bounds([...all, ...tileBounds]);
    center = box.min.map((v, i) => (v + box.max[i]) / 2);
  };
  rebuild();
  const gridBox = bounds(layout.blocks);
  const buffers = kernel?.buffers ?? [];
  const bytesIn = (space) =>
    buffers
      .filter((b) => b.space === space)
      .reduce((sum, b) => sum + (b.bytes ?? 0), 0);
  const memory = {
    global: bytesIn("global"),
    shared: bytesIn("shared"),
    private: bytesIn("private"),
  };

  const view = { ...HOME, zoom: 1, pan: [0, 0] };
  let fit = 1;
  let selected = layout.blocks[0];
  let hovered = null;
  let access = null;
  let size = [0, 0];
  let focusSpace = null;
  // Width the Memory panel covers on the right; the scene centers left of it.
  let inset = 0;
  // Height the facts overlay covers at the top.
  let topSpace = 0;

  const figure = el("figure", "gpu-3d");
  const stage = el("div", "gpu-3d-stage");
  const canvas = el("canvas");
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "img");
  const tip = el("div", "gpu-3d-tip");
  tip.hidden = true;
  const tools = el("div", "gpu-3d-tools");
  const tool = (text, title, action) => {
    const button = el("button", "", text);
    button.type = "button";
    button.title = title;
    button.addEventListener("click", () => {
      action();
      schedule();
    });
    tools.append(button);
  };
  tool("−", "Zoom out (−)", () => (view.zoom /= 1.25));
  tool("+", "Zoom in (+)", () => (view.zoom *= 1.25));
  tool("Top", "Look straight down (t)", () =>
    Object.assign(view, { yaw: 0, pitch: Math.PI / 2 - 0.01 }),
  );
  tool("Reset", "Reset the view (0)", () =>
    Object.assign(view, { ...HOME, zoom: 1, pan: [0, 0] }),
  );
  // The picked access's elements from above, over the canvas; off until the
  // Elements tool turns it on, and kept on or off as other accesses are picked.
  const elements = el("div", "gpu-3d-elements");
  elements.hidden = true;
  let showTop = false;
  tool("Elements", "Show the elements the picked warp touches, seen from above (e)", () =>
    toggleElements(),
  );
  const elementsButton = tools.lastChild;
  elementsButton.setAttribute("aria-pressed", "false");
  elementsButton.disabled = true;
  function toggleElements(on = !showTop) {
    showTop = on && !elementsButton.disabled;
    elementsButton.setAttribute("aria-pressed", String(showTop));
    placeElements();
  }
  // Below the facts, left of the Memory panel, above the tools.
  function placeElements() {
    elements.hidden = !showTop;
    elements.style.top = `${topSpace + 4}px`;
    elements.style.right = `${inset ? inset : 8}px`;
  }
  let irButton = null;
  if (nodes.length) {
    tool("IR", "Show or hide the kernel IR wall (i)", () => toggleIR());
    irButton = tools.lastChild;
    irButton.setAttribute("aria-pressed", "true");
  }
  function toggleIR() {
    showIR = !showIR;
    irButton.setAttribute("aria-pressed", String(showIR));
    rebuild();
    resize();
  }
  const hint = el("div", "gpu-3d-hint");
  stage.append(canvas, elements, tip, tools, hint);
  figure.append(stage);

  function describe() {
    const perBlock = layout.block[0] * layout.block[1] * layout.block[2];
    const warps = Math.ceil(perBlock / WARP);
    const notes = [];
    if (layout.clipped.grid)
      notes.push(
        `showing ${dimText(layout.shownGrid)} of the ${dimText(layout.grid)} blocks`,
      );
    if (layout.clipped.threads)
      notes.push(`showing the first ${THREAD_MAX} of ${perBlock} threads`);
    if ([launch.grid, launch.block].some((d) => !d || d.includes(null)))
      notes.push("sizes known only at runtime are drawn as 1");
    hint.textContent =
      (notes.length ? `${notes.join(" · ")}. ` : "") +
      "Drag to turn · shift-drag to pan · ⌘/Ctrl + scroll to zoom · click a block to open it" +
      (nodes.length ? " · click an IR node or a line of code to trace it" : "");
    canvas.setAttribute(
      "aria-label",
      `3D view. Left: the grid of ${dimText(layout.grid)} blocks, which run independently. ` +
        `Right: block ${idText(selected.id)} opened into ${perBlock} threads in ${warps} warp${warps > 1 ? "s" : ""} of ${WARP}, ` +
        "each row of 32 running in lockstep. Floor plates: global memory under everything, shared memory under the block.",
    );
  }

  // Fits the home view in the canvas; zoom and pan are kept relative to it.
  function refit() {
    const cam = camera({ ...HOME, center, scale: 1, cx: 0, cy: 0 });
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const cube of [
      { at: box.min, size: 0 },
      { at: box.max, size: 0 },
      ...all.filter((_, i) => i % 7 === 0),
    ])
      for (const corner of CORNERS) {
        const [x, y] = cam.screen(
          cam.view(cube.at.map((v, i) => v + corner[i] * cube.size)),
        );
        [x0, y0, x1, y1] = [
          Math.min(x0, x),
          Math.min(y0, y),
          Math.max(x1, x),
          Math.max(y1, y),
        ];
      }
    fit = Math.min(
      (size[0] - inset - 40) / (x1 - x0),
      (size[1] - topSpace - 110) / (y1 - y0),
    );
  }

  let faces = [];
  let frame = 0;
  function schedule() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(draw);
  }

  function colorOf(cube, colors) {
    if (cube === hovered) return colors.hover;
    if (cube.kind === "node") {
      const node = cube.node;
      if (node.line === focusLine) return colors.accent;
      const base =
        node.kind === "thread" || node.kind === "block"
          ? colors.warpA
          : node.kind === "access"
            ? spaceOf.get(node.name) === "shared"
              ? colors.groups[1]
              : colors.block
            : node.kind === "math"
              ? mix(colors.fg, colors.bg, 0.35)
              : mix(colors.dim, colors.bg, 0.3);
      return slice && !slice.has(node.id) ? mix(base, colors.bg, 0.25) : base;
    }
    if (cube.kind === "block") {
      if (cube === selected) return colors.open;
      // A proven verdict holds in every block: tint them all with it.
      if (access?.proven) return access.good ? colors.provenGood : colors.provenBad;
      return colors.block;
    }
    const lane = access?.lanes.get(cube.id.join(","));
    if (lane) return colors.groups[lane.order % 4];
    if (access) return colors.faded;
    return cube.warp % 2 ? colors.warpB : colors.warpA;
  }

  function draw() {
    const ctx = canvas.getContext("2d");
    if (!ctx || !size[0]) return;
    const colors = palette();
    const ratio = window.devicePixelRatio || 1;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size[0], size[1]);
    const cam = camera({
      yaw: view.yaw,
      pitch: view.pitch,
      center,
      scale: fit * view.zoom,
      cx: (size[0] - inset) / 2 + view.pan[0],
      cy: (size[1] + topSpace) / 2 + view.pan[1],
    });
    const eye = [0, 0, -DISTANCE];
    const point = (p) => cam.screen(cam.view(p));

    // Floor plates: global memory under the whole scene, shared memory
    // under the opened block.
    const plateLabels = [];
    const plate = (space, lo, hi, y, color, label) => {
      const poly = [
        [lo[0], y, lo[2]],
        [hi[0], y, lo[2]],
        [hi[0], y, hi[2]],
        [lo[0], y, hi[2]],
      ].map(point);
      ctx.beginPath();
      poly.forEach(([x, y2], i) => (i ? ctx.lineTo(x, y2) : ctx.moveTo(x, y2)));
      ctx.closePath();
      ctx.fillStyle = css(color, 0.9);
      ctx.fill();
      const focused = space === focusSpace;
      ctx.strokeStyle = focused ? css(colors.accent) : css(colors.dim, 0.35);
      ctx.lineWidth = focused ? 2 : 1;
      ctx.stroke();
      plateLabels.push([label, point([lo[0], y, lo[2]])]);
    };
    const pad = 1.2;
    plate(
      "global",
      [plateBox.min[0] - pad, 0, plateBox.min[2] - pad],
      [plateBox.max[0] + pad, 0, plateBox.max[2] + pad],
      -0.35,
      colors.global,
      `Global memory${memory.global ? ` · ${formatBytes(memory.global)}` : ""} · every thread`,
    );
    if (threadBox && memory.shared)
      plate(
        "shared",
        [threadBox.min[0] - 0.5, 0, threadBox.min[2] - 0.5],
        [threadBox.max[0] + 0.5, 0, threadBox.max[2] + 0.5],
        -0.15,
        colors.shared,
        `Shared memory · ${formatBytes(memory.shared)} · this block's threads`,
      );

    // The picked access's buffer around the elements warp 0 touches, as tiles
    // on the plate (row 0 farthest, level with warp 0), touched ones colored
    // like their lanes and the rest shaded by 32-byte sector or by bank.
    const tileAt = (r, c) => [
      tileArea.x + c * TILE.pitch,
      -0.3,
      tileArea.far - (r + 1) * TILE.pitch,
    ];
    if (access?.tiles && tileArea) {
      const { rows, cols, cells, shade } = access.tiles;
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          const [x, y, z] = tileAt(r, c);
          const poly = [
            [x, y, z],
            [x + TILE.size, y, z],
            [x + TILE.size, y, z + TILE.size],
            [x, y, z + TILE.size],
          ].map(point);
          const lane = cells.get(r * cols + c);
          ctx.beginPath();
          poly.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
          ctx.closePath();
          ctx.fillStyle = lane
            ? css(colors.groups[lane.order % 4])
            : css(mix(colors.dim, colors.bg, shade(r, c) ? 0.3 : 0.2));
          ctx.fill();
        }
      plateLabels.push([access.tiles.label, point(tileAt(-1, 0))]);
    }

    // The IR board under its nodes.
    if (showIR) {
      const y = -0.25;
      const [near, far] = [boardNear - 0.9, boardFar + 0.9];
      const poly = [
        [min[0] - 0.8, y, near],
        [max[0] + 0.8, y, near],
        [max[0] + 0.8, y, far],
        [min[0] - 0.8, y, far],
      ].map(point);
      ctx.beginPath();
      poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fillStyle = css(colors.global, 0.55);
      ctx.fill();
      ctx.strokeStyle = css(colors.dim, 0.25);
      ctx.lineWidth = 1;
      ctx.stroke();
      plateLabels.push([
        "Kernel IR · ids → index math → memory",
        point([min[0] - 0.8, y, near]),
      ]);
    }

    // Cubes, far to near, with only the faces that point at the camera.
    faces = [];
    const lift = (cube) =>
      access && cube.kind === "thread" && access.lanes.has(cube.id.join(","))
        ? 0.3
        : 0;
    const order = all
      .map((cube) => {
        const c = cube.at.map((v) => v + cube.size / 2);
        return { cube, depth: cam.view(c)[2] };
      })
      .sort((a, b) => b.depth - a.depth);
    ctx.lineJoin = "round";
    for (const { cube } of order) {
      const up = lift(cube);
      const corners = CORNERS.map((k) =>
        cam.view([
          cube.at[0] + k[0] * cube.size,
          cube.at[1] + up + k[1] * cube.size,
          cube.at[2] + k[2] * cube.size,
        ]),
      );
      const base = colorOf(cube, colors);
      for (const face of FACES) {
        const n = cam.turn(face.n);
        const c = corners[face.v[0]];
        const toEye = [eye[0] - c[0], eye[1] - c[1], eye[2] - c[2]];
        if (n[0] * toEye[0] + n[1] * toEye[1] + n[2] * toEye[2] <= 0) continue;
        const poly = face.v.map((i) => cam.screen(corners[i]));
        ctx.beginPath();
        poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.closePath();
        ctx.fillStyle = css(mix(base, [0, 0, 0], face.light * 0.35 + 0.65));
        ctx.fill();
        ctx.strokeStyle = css(colors.bg, 0.35);
        ctx.lineWidth = 0.6;
        ctx.stroke();
        faces.push({ poly, cube });
      }
    }

    // A callout from the picked block to the block drawn opened.
    if (threadBox && selected) {
      const s = selected.at;
      const top = s[1] + BLOCK.size;
      const from = [
        [s[0], top, s[2]],
        [s[0] + 1, top, s[2]],
        [s[0] + 1, top, s[2] + 1],
        [s[0], top, s[2] + 1],
      ];
      const y = threadBox.max[1];
      const to = [
        [threadBox.min[0], y, threadBox.min[2]],
        [threadBox.max[0], y, threadBox.min[2]],
        [threadBox.max[0], y, threadBox.max[2]],
        [threadBox.min[0], y, threadBox.max[2]],
      ];
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = css(colors.accent, 0.35);
      ctx.lineWidth = 1;
      for (let i = 0; i < 4; i++) {
        const [a, b] = [point(from[i]), point(to[i])];
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // From each thread of warp 0 down to the element it touches: a row of
    // threads fanning out to a column of elements is a strided access.
    if (access?.tiles && tileArea) {
      ctx.lineWidth = 1;
      for (const [key, lane] of access.lanes) {
        const cube = threadCube.get(key);
        const tile = access.tiles.of.get(lane.lane);
        if (!cube || !tile) continue;
        const from = point([
          cube.at[0] + cube.size / 2,
          cube.at[1] + 0.3 + cube.size,
          cube.at[2] + cube.size / 2,
        ]);
        const [x, y, z] = tileAt(tile[0], tile[1]);
        const to = point([x + TILE.size / 2, y, z + TILE.size / 2]);
        ctx.strokeStyle = css(colors.groups[lane.order % 4], 0.55);
        ctx.beginPath();
        ctx.moveTo(from[0], from[1]);
        ctx.lineTo(to[0], to[1]);
        ctx.stroke();
      }
    }

    // The IR's edges: value → user along the wall, and dashed drops from the
    // ids to the block and threads they name and from each access to its
    // memory. With a line picked, its slice is drawn strong and the rest faint.
    if (showIR) {
      const mid = (cube) => cube.at.map((v) => v + cube.size / 2);
      const line = (a, b, strong, dashed) => {
        const [p, q] = [point(a), point(b)];
        ctx.setLineDash(dashed ? [3, 4] : []);
        ctx.strokeStyle = strong
          ? css(colors.accent, 0.9)
          : css(colors.dim, slice ? 0.12 : 0.45);
        ctx.lineWidth = strong ? 1.8 : 1;
        // Arched: through a point lifted above the middle, so edges rise off
        // the board instead of running flat across it.
        const rise = Math.min(
          4,
          0.8 + Math.hypot(b[0] - a[0], b[2] - a[2]) * 0.18,
        );
        const c = point([
          (a[0] + b[0]) / 2,
          Math.max(a[1], b[1]) + rise,
          (a[2] + b[2]) / 2,
        ]);
        ctx.beginPath();
        ctx.moveTo(p[0], p[1]);
        ctx.quadraticCurveTo(
          2 * c[0] - (p[0] + q[0]) / 2,
          2 * c[1] - (p[1] + q[1]) / 2,
          q[0],
          q[1],
        );
        ctx.stroke();
      };
      const lit = (id) => !!slice?.has(id);
      for (const edge of flow.edges) {
        const [a, b] = [byId.get(edge.from), byId.get(edge.to)];
        if (a && b) line(mid(a), mid(b), lit(edge.from) && lit(edge.to), false);
      }
      for (const cube of nodes) {
        const { node } = cube;
        let target = null;
        if (node.kind === "thread" && threadBox)
          target = [
            (threadBox.min[0] + threadBox.max[0]) / 2,
            threadBox.max[1],
            threadBox.min[2],
          ];
        else if (node.kind === "block")
          target = selected.at.map((v, i) => v + (i === 1 ? BLOCK.size : 0.5));
        else if (node.kind === "access")
          target =
            spaceOf.get(node.name) === "shared" && threadBox
              ? [
                  Math.min(
                    Math.max(mid(cube)[0], threadBox.min[0]),
                    threadBox.max[0],
                  ),
                  -0.15,
                  threadBox.min[2] - 0.4,
                ]
              : [mid(cube)[0], -0.35, min[2] - 1.2];
        if (target && (!slice || lit(node.id)))
          line(mid(cube), target, lit(node.id), true);
      }
      ctx.setLineDash([]);
      ctx.textAlign = "left";
      for (const cube of nodes) {
        const { node } = cube;
        const [x, y] = point([
          cube.at[0] + cube.size,
          cube.at[1] + cube.size / 2,
          cube.at[2],
        ]);
        const faded = slice && !slice.has(node.id);
        ctx.font = `${node.kind === "access" || node.line === focusLine ? 600 : 400} 10px ${colors.font}`;
        ctx.fillStyle = css(faded ? colors.dim : colors.fg, faded ? 0.4 : 1);
        ctx.fillText(node.label, x + 4, y + 3);
      }
    }

    // Axis arrows along the grid's back and left edges, from block (0, 0, 0).
    const arrow = (from, to, text) => {
      const [a, b] = [point(from), point(to)];
      const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
      ctx.strokeStyle = ctx.fillStyle = css(colors.dim);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      for (const side of [-1, 1])
        (ctx.lineTo(
          ...[
            b[0] - 7 * Math.cos(angle + side * 0.45),
            b[1] - 7 * Math.sin(angle + side * 0.45),
          ],
        ),
          ctx.moveTo(b[0], b[1]));
      ctx.stroke();
      ctx.font = `11px ${colors.font}`;
      ctx.textAlign = "center";
      ctx.fillText(
        text,
        b[0] + 12 * Math.cos(angle),
        b[1] + 12 * Math.sin(angle) + 4,
      );
    };
    const back = gridBox.max[2] + 0.5;
    const side = gridBox.min[0] - 0.5;
    arrow([side, 0, back], [Math.min(gridBox.max[0], side + 6), 0, back], "x");
    arrow([side, 0, back], [side, 0, Math.max(gridBox.min[2], back - 6)], "y");

    ctx.font = `11px ${colors.font}`;
    ctx.textAlign = "left";
    ctx.fillStyle = css(colors.dim);
    // Kept inside the canvas, like the titles.
    for (const [text, [x, y]] of plateLabels) {
      const right = size[0] - ctx.measureText(text).width - 8;
      ctx.fillText(text, Math.max(8, Math.min(x + 4, right)), y + 14);
    }

    // Titles over the grid and the opened block; a second line says what
    // warp 0's colors mean.
    const title = (lines, p) => {
      const [x, y] = point(p);
      ctx.textAlign = "center";
      lines.forEach((text, i) => {
        ctx.font = `${i ? 400 : 600} ${i ? 11 : 12}px ${colors.font}`;
        ctx.fillStyle = css(i ? colors.dim : colors.fg);
        // Kept inside the canvas however the scene is turned.
        const half = ctx.measureText(text).width / 2;
        const left = Math.min(Math.max(x, half + 8), size[0] - half - 8);
        ctx.fillText(text, left, y - 14 + i * 15);
      });
    };
    const top = Math.max(gridBox.max[1], threadBox?.max[1] ?? 0) + 1.2;
    title(
      access?.proven
        ? [`Grid · ${dimText(layout.grid)} blocks`, "Same pattern in every block: proven"]
        : [`Grid · ${dimText(layout.grid)} blocks`],
      [(gridBox.min[0] + gridBox.max[0]) / 2, top, gridBox.max[2] + 1.6],
    );
    if (threadBox) {
      const lines = [
        `Block ${idText(selected.id)} · ${dimText(layout.block)} threads`,
      ];
      if (access)
        lines.push(`Warp 0 raised, colored by ${access.unit}: ${access.text}`);
      if (access?.formula) lines.push(`offset = ${access.formula}`);
      if (memory.private)
        lines.push(`Private: ${formatBytes(memory.private)} per thread`);
      title(lines, [
        (threadBox.min[0] + threadBox.max[0]) / 2,
        top,
        threadBox.max[2] + 1.6,
      ]);
    }
  }

  function hit(x, y) {
    for (let i = faces.length - 1; i >= 0; i--)
      if (inside([x, y], faces[i].poly)) return faces[i].cube;
    return null;
  }

  function tipText(cube) {
    if (cube.kind === "node")
      return `line ${cube.node.line} · ${cube.node.text}\nclick to find it in the code`;
    if (cube.kind === "block")
      return `block ${idText(cube.id)}${cube === selected ? " · opened" : " · click to open"}`;
    let text = `thread ${idText(cube.id)} · warp ${cube.warp}, lane ${cube.lane}`;
    const lane = access?.lanes.get(cube.id.join(","));
    if (lane)
      text += `\n${access.buffer}[${lane.index.join(", ")}] · ${access.unit} ${lane.group}`;
    return text;
  }

  // Pointer: drag turns (shift-drag pans), a click without a drag picks.
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, moved: false, pan: e.shiftKey };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    const box = canvas.getBoundingClientRect();
    if (drag) {
      const [dx, dy] = [e.clientX - drag.x, e.clientY - drag.y];
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (drag.moved) {
        if (drag.pan) {
          view.pan[0] += dx;
          view.pan[1] += dy;
        } else {
          view.yaw -= dx * 0.008;
          view.pitch = Math.min(
            Math.PI / 2 - 0.01,
            Math.max(0.08, view.pitch + dy * 0.006),
          );
        }
        drag.x = e.clientX;
        drag.y = e.clientY;
        tip.hidden = true;
        schedule();
        return;
      }
    }
    const cube = hit(e.clientX - box.left, e.clientY - box.top);
    if (cube !== hovered) {
      hovered = cube;
      canvas.style.cursor =
        cube?.kind === "block" || cube?.kind === "node" ? "pointer" : "grab";
      schedule();
    }
    tip.hidden = !cube;
    if (cube) {
      tip.textContent = tipText(cube);
      tip.style.left = `${e.clientX - box.left + 12}px`;
      tip.style.top = `${e.clientY - box.top + 12}px`;
    }
  });
  canvas.addEventListener("pointerup", (e) => {
    if (drag && !drag.moved) {
      const box = canvas.getBoundingClientRect();
      const cube = hit(e.clientX - box.left, e.clientY - box.top);
      if (cube?.kind === "node") {
        showLine(cube.node.line);
        onLine?.(cube.node.line);
      } else if (cube?.kind === "block" && cube !== selected) {
        selected = cube;
        describe();
        tip.textContent = tipText(cube);
        schedule();
      }
    }
    drag = null;
  });
  canvas.addEventListener("pointerleave", () => {
    tip.hidden = true;
    if (hovered) {
      hovered = null;
      schedule();
    }
  });
  // Plain scrolling keeps scrolling the panel; ⌘/Ctrl + scroll and trackpad
  // pinches (which arrive with ctrlKey) zoom.
  canvas.addEventListener(
    "wheel",
    (e) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      view.zoom = Math.min(
        8,
        Math.max(0.3, view.zoom * Math.exp(-e.deltaY * 0.004)),
      );
      schedule();
    },
    { passive: false },
  );
  canvas.addEventListener("dblclick", () => {
    Object.assign(view, { ...HOME, zoom: 1, pan: [0, 0] });
    schedule();
  });
  // Keys while the canvas has focus stay here, away from the app shortcuts.
  const KEYS = {
    ArrowLeft: () => (view.yaw += 0.12),
    ArrowRight: () => (view.yaw -= 0.12),
    ArrowUp: () =>
      (view.pitch = Math.min(Math.PI / 2 - 0.01, view.pitch + 0.1)),
    ArrowDown: () => (view.pitch = Math.max(0.08, view.pitch - 0.1)),
    "+": () => (view.zoom *= 1.25),
    "=": () => (view.zoom *= 1.25),
    "-": () => (view.zoom /= 1.25),
    t: () => Object.assign(view, { yaw: 0, pitch: Math.PI / 2 - 0.01 }),
    i: () => nodes.length && toggleIR(),
    e: () => toggleElements(),
    0: () => Object.assign(view, { ...HOME, zoom: 1, pan: [0, 0] }),
  };
  canvas.addEventListener("keydown", (e) => {
    const action = KEYS[e.key];
    if (!action || e.metaKey || e.ctrlKey) return;
    e.preventDefault();
    e.stopPropagation();
    action();
    schedule();
  });

  const resize = () => {
    const width = stage.clientWidth;
    if (!width) return;
    const height = Math.round(
      Math.min(640, Math.max(380, width * (showIR ? 0.7 : 0.58))),
    );
    // The Memory panel sits over the scene's right side on wide stages; on
    // narrow ones it starts closed and simply covers the scene when opened.
    const panel = stage.querySelector(".gpu-3d-panel");
    inset = panel?.open && width >= 640 ? panel.offsetWidth + 16 : 0;
    topSpace = (stage.querySelector(".gpu-3d-facts")?.offsetHeight ?? 0) + 8;
    placeElements();
    const ratio = window.devicePixelRatio || 1;
    size = [width, height];
    scene.shown = true;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    refit();
    draw();
  };
  if (typeof ResizeObserver !== "undefined")
    new ResizeObserver(resize).observe(stage);

  // A scene is dropped once it has been on the page and left it (the GPU
  // view builds every launch before inserting them, so not before).
  const scene = { canvas, schedule, shown: false };
  for (const old of scenes)
    if (old.shown && !old.canvas.isConnected) scenes.delete(old);
  scenes.add(scene);
  describe();

  function showLine(line) {
    const lit = line ? flowSlice(flow, line) : new Set();
    slice = lit.size ? lit : null;
    focusLine = slice ? line : null;
    schedule();
    return lit.size;
  }

  return {
    figure,
    showLine,
    flow,
    lit: (id) => !!slice?.has(id),
    showSpace(space) {
      focusSpace = space;
      schedule();
    },
    overlay(facts, panelBody, summary) {
      facts.classList.add("gpu-3d-facts");
      stage.append(facts);
      if (!panelBody) return;
      const panel = el("details", "gpu-3d-panel");
      panel.open = true;
      const head = el("summary", "", summary);
      panel.append(head, panelBody);
      // Starts closed where it would cover most of the scene.
      requestAnimationFrame(() => {
        if (stage.clientWidth && stage.clientWidth < 760) panel.open = false;
      });
      panel.addEventListener("toggle", resize);
      stage.append(panel);
    },
    // `node` is the element map of the picked access, or null when it has none.
    showElements(node) {
      elements.replaceChildren(...(node ? [node] : []));
      elementsButton.disabled = !node;
      elementsButton.title = node
        ? "Show the elements the picked warp touches, seen from above (e)"
        : "No elements to show: pick an access that could be analyzed";
      toggleElements(showTop);
    },
    // `judged` is a Memory accesses row: { access, result, space }.
    // `formula` also writes the offset's formula under the opened block
    // (Compiler mode).
    showAccess(judged, { formula = false } = {}) {
      const result = judged?.result;
      if (!result?.analyzed) {
        access = null;
      } else {
        const groups = [...new Set(result.lanes.map((l) => l.group))];
        const order = new Map(groups.map((g, i) => [g, i]));
        const unit = judged.space === "shared" ? "bank" : "sector";
        access = {
          buffer: judged.access.buffer,
          unit,
          proven: result.proof?.status === "proven",
          good: ["coalesced", "broadcast", "conflict-free"].includes(result.verdict),
          formula: formula ? (result.proof?.formula ?? null) : null,
          text: `${judged.access.kind === "load" ? "load" : "store"} ${judged.access.buffer} · ${groups.length} ${unit}${groups.length > 1 ? "s" : ""}`,
          lanes: new Map(
            result.lanes.map((l) => [
              [l.tx, l.ty, l.tz].join(","),
              { ...l, order: order.get(l.group) },
            ]),
          ),
          tiles: elementTiles(judged, result, order),
        };
      }
      schedule();
    },
  };
}
