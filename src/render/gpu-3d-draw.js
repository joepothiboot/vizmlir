import { formatBytes } from "../trace/index.js";
import {
  BLOCK,
  CORNERS,
  DISTANCE,
  FACES,
  TILE,
  camera,
  dimText,
  idText,
} from "./gpu-3d-geometry.js";

function rgb(text) {
  const hex = /^#([0-9a-f]{6})$/i.exec(text.trim());
  if (hex) return [0, 2, 4].map((i) => parseInt(hex[1].slice(i, i + 2), 16));

  const parts = /rgba?\(([^)]+)\)/.exec(text)?.[1].split(",").map(Number);

  return parts ? parts.slice(0, 3) : [128, 128, 128];
}

export const mix = (a, b, t) => a.map((v, i) => v * t + b[i] * (1 - t));

export const css = (c, alpha = 1) =>
  `rgba(${c.map((v) => Math.round(v)).join(", ")}, ${alpha})`;

export function palette() {
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

const tileAt = (tileArea, r, c) => [
  tileArea.x + c * TILE.pitch,
  -0.3,
  tileArea.far - (r + 1) * TILE.pitch,
];

function nodeColor(node, colors, spaceOf) {
  if (node.kind === "thread" || node.kind === "block") return colors.warpA;

  if (node.kind === "access") {
    return spaceOf.get(node.name) === "shared"
      ? colors.groups[1]
      : colors.block;
  }

  if (node.kind === "math") return mix(colors.fg, colors.bg, 0.35);

  return mix(colors.dim, colors.bg, 0.3);
}

function colorOf(cube, colors, model, ui) {
  const { spaceOf } = model;
  const { access, focusLine, hovered, selected, slice } = ui;

  if (cube === hovered) return colors.hover;

  if (cube.kind === "node") {
    const node = cube.node;
    if (node.line === focusLine) return colors.accent;

    const base = nodeColor(node, colors, spaceOf);

    return slice && !slice.has(node.id) ? mix(base, colors.bg, 0.25) : base;
  }

  if (cube.kind === "block") {
    if (cube === selected) return colors.open;

    if (access?.proven) {
      return access.good ? colors.provenGood : colors.provenBad;
    }

    return colors.block;
  }

  const lane = access?.lanes.get(cube.id.join(","));
  if (lane) return colors.groups[lane.order % 4];
  if (access) return colors.faded;

  return cube.warp % 2 ? colors.warpB : colors.warpA;
}

export function drawScene(canvas, model, ui) {
  const { center, fit, inset, size, topSpace, view } = ui;

  const ctx = canvas.getContext("2d");
  if (!ctx || !size[0]) return null;

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

  const frame = { ctx, colors, cam, point, eye, labels: [], model, ui };

  drawPlates(frame);
  drawTiles(frame);
  drawBoard(frame);

  const faces = drawCubes(frame);

  drawCallout(frame);
  drawLaneLinks(frame);
  drawFlow(frame);
  drawAxes(frame);
  drawLabels(frame);
  drawTitles(frame);

  return faces;
}

function drawPlates({ colors, ctx, labels, point, model, ui }) {
  const { memory, plateBox, threadBox } = model;
  const { focusSpace } = ui;

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
    labels.push([label, point([lo[0], y, lo[2]])]);
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

  if (threadBox && memory.shared) {
    plate(
      "shared",
      [threadBox.min[0] - 0.5, 0, threadBox.min[2] - 0.5],
      [threadBox.max[0] + 0.5, 0, threadBox.max[2] + 0.5],
      -0.15,
      colors.shared,
      `Shared memory · ${formatBytes(memory.shared)} · this block's threads`,
    );
  }
}

function drawTiles({ colors, ctx, labels, point, model, ui }) {
  const { tileArea } = model;
  const { access } = ui;

  if (access?.tiles && tileArea) {
    const { rows, cols, cells, shade } = access.tiles;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [x, y, z] = tileAt(tileArea, r, c);

        const poly = [
          [x, y, z],
          [x + TILE.size, y, z],
          [x + TILE.size, y, z + TILE.size],
          [x, y, z + TILE.size],
        ].map(point);

        const lane = cells.get(r * cols + c);
        ctx.beginPath();

        poly.forEach(([px, py], i) =>
          i ? ctx.lineTo(px, py) : ctx.moveTo(px, py),
        );

        ctx.closePath();

        ctx.fillStyle = lane
          ? css(colors.groups[lane.order % 4])
          : css(mix(colors.dim, colors.bg, shade(r, c) ? 0.3 : 0.2));

        ctx.fill();
      }
    }

    labels.push([access.tiles.label, point(tileAt(tileArea, -1, 0))]);
  }
}

function drawBoard({ colors, ctx, labels, point, model, ui }) {
  const { boardFar, boardNear, max, min } = model;
  const { showIR } = ui;

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

    labels.push([
      "Kernel IR · ids → index math → memory",
      point([min[0] - 0.8, y, near]),
    ]);
  }
}

function drawCubes({ cam, colors, ctx, eye, model, ui }) {
  const { access, all } = ui;

  const faces = [];

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

    const base = colorOf(cube, colors, model, ui);

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

  return faces;
}

function drawCallout({ colors, ctx, point, model, ui }) {
  const { threadBox } = model;
  const { selected } = ui;

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
}

function drawLaneLinks({ colors, ctx, point, model, ui }) {
  const { threadCube, tileArea } = model;
  const { access } = ui;

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

      const [x, y, z] = tileAt(tileArea, tile[0], tile[1]);
      const to = point([x + TILE.size / 2, y, z + TILE.size / 2]);
      ctx.strokeStyle = css(colors.groups[lane.order % 4], 0.55);
      ctx.beginPath();
      ctx.moveTo(from[0], from[1]);
      ctx.lineTo(to[0], to[1]);
      ctx.stroke();
    }
  }
}

function drawFlow({ colors, ctx, point, model, ui }) {
  const { byId, flow, min, nodes, spaceOf, threadBox } = model;
  const { focusLine, selected, showIR, slice } = ui;

  if (showIR) {
    const mid = (cube) => cube.at.map((v) => v + cube.size / 2);

    const line = (a, b, strong, dashed) => {
      const [p, q] = [point(a), point(b)];
      ctx.setLineDash(dashed ? [3, 4] : []);

      ctx.strokeStyle = strong
        ? css(colors.accent, 0.9)
        : css(colors.dim, slice ? 0.12 : 0.45);

      ctx.lineWidth = strong ? 1.8 : 1;

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

      if (node.kind === "thread" && threadBox) {
        target = [
          (threadBox.min[0] + threadBox.max[0]) / 2,
          threadBox.max[1],
          threadBox.min[2],
        ];
      } else if (node.kind === "block") {
        target = selected.at.map((v, i) => v + (i === 1 ? BLOCK.size : 0.5));
      } else if (node.kind === "access") {
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
      }

      if (target && (!slice || lit(node.id))) {
        line(mid(cube), target, lit(node.id), true);
      }
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
}

function drawAxes({ colors, ctx, point, model }) {
  const { gridBox } = model;

  const arrow = (from, to, text) => {
    const [a, b] = [point(from), point(to)];
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    ctx.strokeStyle = ctx.fillStyle = css(colors.dim);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);

    for (const side of [-1, 1]) {
      (ctx.lineTo(
        ...[
          b[0] - 7 * Math.cos(angle + side * 0.45),
          b[1] - 7 * Math.sin(angle + side * 0.45),
        ],
      ),
        ctx.moveTo(b[0], b[1]));
    }

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
}

function drawLabels({ colors, ctx, labels, ui }) {
  const { size } = ui;

  ctx.font = `11px ${colors.font}`;
  ctx.textAlign = "left";
  ctx.fillStyle = css(colors.dim);

  for (const [text, [x, y]] of labels) {
    const right = size[0] - ctx.measureText(text).width - 8;
    ctx.fillText(text, Math.max(8, Math.min(x + 4, right)), y + 14);
  }
}

function drawTitles({ colors, ctx, point, model, ui }) {
  const { gridBox, layout, memory, threadBox } = model;
  const { access, selected, size } = ui;

  const title = (lines, p) => {
    const [x, y] = point(p);
    ctx.textAlign = "center";

    lines.forEach((text, i) => {
      ctx.font = `${i ? 400 : 600} ${i ? 11 : 12}px ${colors.font}`;
      ctx.fillStyle = css(i ? colors.dim : colors.fg);

      const half = ctx.measureText(text).width / 2;
      const left = Math.min(Math.max(x, half + 8), size[0] - half - 8);
      ctx.fillText(text, left, y - 14 + i * 15);
    });
  };

  const top = Math.max(gridBox.max[1], threadBox?.max[1] ?? 0) + 1.2;

  title(
    access?.proven
      ? [
          `Grid · ${dimText(layout.grid)} blocks`,
          "Same pattern in every block: proven",
        ]
      : [`Grid · ${dimText(layout.grid)} blocks`],
    [(gridBox.min[0] + gridBox.max[0]) / 2, top, gridBox.max[2] + 1.6],
  );

  if (threadBox) {
    const lines = [
      `Block ${idText(selected.id)} · ${dimText(layout.block)} threads`,
    ];

    if (access) {
      lines.push(`Warp 0 raised, colored by ${access.unit}: ${access.text}`);
    }

    if (access?.formula) lines.push(`offset = ${access.formula}`);

    if (memory.private) {
      lines.push(`Private: ${formatBytes(memory.private)} per thread`);
    }

    title(lines, [
      (threadBox.min[0] + threadBox.max[0]) / 2,
      top,
      threadBox.max[2] + 1.6,
    ]);
  }
}
