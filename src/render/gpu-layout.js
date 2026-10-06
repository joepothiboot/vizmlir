import { el, svg } from "../dom.js";
import { ownedBy } from "../gpu/index.js";
import { matrixWidth } from "./gpu-3d.js";
import { explain, explainProof, groupOrder } from "./gpu-verdict.js";

const LAYOUT_MAX = 64;

export function renderLayout(figure, { access, result }, kernel) {
  const encoding = access.tensor?.encoding;
  const layout = kernel.triton.layouts.get(encoding);
  const shape = access.tensor?.shape ?? [];

  if (!layout || !shape.length || shape.length > 2) {
    figure.replaceChildren(
      el(
        "p",
        "gpu-note",
        `The layout ${encoding ?? "?"} is not a #blocked layout VizMLIR draws yet.`,
      ),
    );

    return;
  }

  const [rows, cols] = shape.length === 2 ? shape : [1, shape[0]];
  const shownRows = Math.min(rows, LAYOUT_MAX);
  const shownCols = Math.min(cols, LAYOUT_MAX);
  const at = (coord) => (shape.length === 2 ? coord : [0, coord[0]]);
  const owner = new Map();
  const { numWarps, warpSize } = kernel.triton;

  for (let warp = 0; warp < numWarps; warp++) {
    for (let lane = 0; lane < warpSize; lane++) {
      ownedBy(layout, shape, warp, lane).reps.forEach((coords, rep) =>
        coords.forEach((coord, k) => {
          const [r, c] = at(coord);
          const key = r * cols + c;

          if (!owner.has(key)) {
            owner.set(key, { warp, lane, rep, first: k === 0 });
          }
        }),
      );
    }
  }

  const cell = shownCols > 32 ? 8 : 14;

  const grid = svg("svg", {
    class: "gpu-layout-grid",
    viewBox: `0 0 ${shownCols * cell} ${shownRows * cell}`,
    width: shownCols * cell,
    height: shownRows * cell,
    role: "img",
    "aria-label": `Which warp and lane hold each element of ${access.buffer}'s tensor`,
  });

  for (let r = 0; r < shownRows; r++) {
    for (let c = 0; c < shownCols; c++) {
      const o = owner.get(r * cols + c);
      if (!o) continue;

      const picked = o.warp === 0 && o.rep === 0;

      grid.append(
        svg(
          "rect",
          {
            x: c * cell,
            y: r * cell,
            width: cell,
            height: cell,
            class: `w${o.warp % 4}${o.rep ? " later" : ""}${picked ? " picked" : ""}`,
          },
          `element [${shape.length === 2 ? `${r}, ${c}` : c}] · warp ${o.warp}, lane ${o.lane}${o.rep ? `, repetition ${o.rep + 1}` : ""}`,
        ),
      );

      if (o.first && cell >= 14) {
        const label = svg("text", { x: c * cell + 2, y: r * cell + cell - 4 });
        label.textContent = String(o.lane);
        grid.append(label);
      }
    }
  }

  const legend = el("div", "gpu-layout-legend");

  for (let warp = 0; warp < Math.min(numWarps, 4); warp++) {
    const item = el(
      "span",
      "",
      `warp ${warp}${numWarps > 4 && warp === 3 ? ` (and every 4th after)` : ""}`,
    );

    item.style.setProperty("--swatch", `var(--warp-${warp})`);
    legend.append(item);
  }

  const clipped =
    rows > shownRows || cols > shownCols
      ? ` Showing the first ${shownRows} × ${shownCols}.`
      : "";

  figure.replaceChildren(
    el("span", "gpu-kicker", `Who holds which element · ${encoding}`),
    el(
      "p",
      "",
      describeLayout(layout, shape, result.elementBytes, numWarps) + clipped,
    ),
    grid,
    legend,
    el(
      "p",
      "gpu-note",
      `sizePerThread [${layout.sizePerThread.join(", ")}] · threadsPerWarp [${layout.threadsPerWarp.join(", ")}] · warpsPerCTA [${layout.warpsPerCTA.join(", ")}] · order [${layout.order.join(", ")}]. Outlined: what warp 0 ${access.kind === "load" ? "loads" : "stores"} in one go, judged above. Numbers are lane ids.`,
    ),
  );
}

function describeLayout(layout, shape, elementBytes, numWarps) {
  const names = shape.length === 2 ? ["i", "j"] : ["i"];
  const fast = layout.order[0];
  const run = layout.sizePerThread[fast];
  const each = layout.sizePerThread.reduce((a, b) => a * b, 1);

  const warp = shape.map(
    (_, d) => layout.sizePerThread[d] * layout.threadsPerWarp[d],
  );

  const program = shape.map((_, d) => warp[d] * layout.warpsPerCTA[d]);

  const reps = shape.reduce(
    (n, size, d) => n * Math.max(1, Math.ceil(size / program[d])),
    1,
  );

  const bytes = run * (elementBytes ?? 0);

  const first =
    run > 1
      ? `Each thread holds ${each} elements, ${run} side by side along ${names[fast]}: one ${bytes}-byte access.`
      : `Each thread holds ${each === 1 ? "one element" : `${each} elements`} at a time, so neighboring lanes decide what a warp touches.`;

  const cover = (dims) => dims.join(" × ");

  return (
    `${first} Lanes are laid out along ${names[fast]} first. A warp covers ${cover(warp)} elements, and the program's ${numWarps} warps ${cover(program)}` +
    (reps > 1 ? `, repeated ${reps} times to fill ${cover(shape)}.` : ".")
  );
}

const groupClass = (order) => `g${order % 4}`;

export function renderLaneDetail(container, { access, result, space }) {
  const children = [el("p", "gpu-explain", explain(result, space))];

  if (result.analyzed && result.proof?.status !== "proven") {
    children.push(el("p", "gpu-note", explainProof(result.proof)));
  }

  if (!result.analyzed) {
    container.replaceChildren(...children);

    return;
  }

  const order = groupOrder(result);
  const groups = [...order.keys()];
  const unit = space === "shared" ? "bank" : "sector";

  const cell = 16;

  const strip = svg("svg", {
    class: "gpu-lanes",
    width: result.lanes.length * (cell + 2),
    height: cell + 14,
    role: "img",
    "aria-label": `Lanes by ${unit}`,
  });

  for (const lane of result.lanes) {
    strip.append(
      svg(
        "rect",
        {
          x: lane.lane * (cell + 2),
          y: 0,
          width: cell,
          height: cell,
          rx: 2,
          class: `lane ${groupClass(order.get(lane.group))}`,
        },
        `lane ${lane.lane} · thread (${lane.tx}, ${lane.ty}, ${lane.tz})\n` +
          `${access.buffer}[${lane.index.join(", ")}] · byte ${lane.byte} · ${unit} ${lane.group}`,
      ),
    );

    if (lane.lane % 8 === 0) {
      const label = svg("text", {
        x: lane.lane * (cell + 2),
        y: cell + 11,
        class: "lane-label",
      });

      label.textContent = String(lane.lane);
      strip.append(label);
    }
  }

  const lanesFigure = el("figure");

  lanesFigure.append(
    strip,
    el(
      "figcaption",
      "",
      `Lanes 0–${result.lanes.length - 1}, colored by the ${unit} they hit: ${groups.length} ${unit}${groups.length > 1 ? "s" : ""}. Hover a lane for its thread and element; Elements in the 3D view shows them in the buffer from above.`,
    ),
  );

  children.push(lanesFigure);
  container.replaceChildren(...children);
}

function elementTitle(buffer, lane, index) {
  if (lane) return `${buffer}[${lane.index.join(", ")}] · lane ${lane.lane}`;

  return `${buffer}[${index.join(", ")}]`;
}

export function elementMap(access, result, memref, order) {
  const twoD = matrixWidth(memref.dims) !== null;
  const width = matrixWidth(memref.dims) ?? 32;

  const at = (offset) => ({
    row: Math.floor(offset / width),
    col: offset % width,
  });

  const touched = new Map();
  for (const lane of result.lanes) touched.set(lane.offset, lane);

  const points = [...touched.keys()].map(at);
  const minRow = Math.min(...points.map((p) => p.row));
  const maxRow = Math.max(...points.map((p) => p.row));
  const minCol = Math.min(...points.map((p) => p.col));
  const maxCol = Math.max(...points.map((p) => p.col));
  const MAX_ROWS = 34;
  const MAX_COLS = 48;
  const rows = Math.min(maxRow - minRow + 1, MAX_ROWS);

  const colStart = Math.max(
    0,
    Math.min(minCol, maxCol - MAX_COLS + 1, width - MAX_COLS),
  );

  const cols = Math.min(width - colStart, MAX_COLS);
  const cell = 9;
  const pad = 30;
  const mapWidth = pad + cols * (cell + 1);
  const mapHeight = 14 + rows * (cell + 1);

  const map = svg("svg", {
    class: "gpu-elements",
    viewBox: `0 0 ${mapWidth} ${mapHeight}`,
    width: mapWidth,
    height: mapHeight,
    role: "img",
    "aria-label": `Elements of ${access.buffer} touched by warp 0`,
  });

  for (let r = 0; r < rows; r++) {
    const rowLabel = svg("text", {
      x: 0,
      y: 14 + r * (cell + 1) + cell - 1,
      class: "lane-label",
    });

    rowLabel.textContent = twoD
      ? String(minRow + r)
      : String((minRow + r) * width);

    map.append(rowLabel);

    for (let c = 0; c < cols; c++) {
      const offset = (minRow + r) * width + colStart + c;
      const lane = touched.get(offset);

      map.append(
        svg(
          "rect",
          {
            x: pad + c * (cell + 1),
            y: 14 + r * (cell + 1),
            width: cell,
            height: cell,
            class: lane
              ? `element touched ${groupClass(order.get(lane.group))}`
              : "element",
          },
          elementTitle(
            access.buffer,
            lane,
            twoD ? [minRow + r, colStart + c] : [offset],
          ),
        ),
      );
    }
  }

  const colLabel = svg("text", { x: pad, y: 9, class: "lane-label" });

  colLabel.textContent = twoD
    ? `columns ${colStart}–${colStart + cols - 1}`
    : "element offset";

  map.append(colLabel);

  const figure = el("figure");

  const clipped =
    maxRow - minRow + 1 > MAX_ROWS || maxCol - minCol + 1 > MAX_COLS;

  figure.append(
    map,
    el(
      "figcaption",
      "",
      `${access.buffer} (${memref.type}) around the elements warp 0 touches` +
        (clipped ? "; only part of the range fits here." : "."),
    ),
  );

  return figure;
}
