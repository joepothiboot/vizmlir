// Draws the GPU view: for each kernel launch, the grid of blocks and one block
// opened into its warps and threads, and the kernel's buffers by memory
// space. Built from analyzeGpu() (src/gpu.js); plain DOM and SVG, no canvas.

import { parseMemref } from "./buffers.js";
import { warpAccess } from "./gpu-access.js";
import { memorySpace } from "./gpu.js";
import { formatBytes } from "./timing.js";

const SVG = "http://www.w3.org/2000/svg";
const WARP = 32;
// Drawing limits; larger launches are drawn in part and say so.
const MAX_BLOCKS = 64;
const MAX_WARPS = 32;

const SPACE_TEXT = {
  global: ["Global", "device memory, visible to every thread in the launch"],
  shared: ["Shared", "one copy per block, visible to that block's threads"],
  private: ["Private", "one copy per thread (registers or local memory)"],
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function svg(tag, attrs, title) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (title) {
    const t = document.createElementNS(SVG, "title");
    t.textContent = title;
    node.append(t);
  }
  return node;
}

const dim = (dims) => (dims ? dims.map((d) => d ?? "?").join(" × ") : "?");
const count = (dims) =>
  dims && dims.every((d) => d !== null) ? dims.reduce((a, b) => a * b, 1) : null;
// Linear index → (x, y, z) for `dims`.
function coords(index, dims) {
  const [x, y] = dims.map((d) => d ?? 1);
  return [index % x, Math.floor(index / x) % y, Math.floor(index / (x * y))];
}

// The grid of blocks, block (0,0,0) highlighted as the one opened below.
function gridDiagram(grid) {
  const total = count(grid);
  const x = Math.max(1, grid?.[0] ?? 1);
  const shown = Math.min(total ?? 1, MAX_BLOCKS);
  const columns = Math.min(x, 16);
  const rows = Math.ceil(shown / columns);
  const size = 18;
  const gap = 3;
  const root = svg("svg", {
    class: "gpu-grid",
    width: columns * (size + gap),
    height: rows * (size + gap),
    role: "img",
    "aria-label": `Grid of ${total ?? "?"} blocks`,
  });
  for (let i = 0; i < shown; i++) {
    const [bx, by, bz] = coords(i, grid ?? [1, 1, 1]);
    root.append(
      svg(
        "rect",
        {
          x: (i % columns) * (size + gap),
          y: Math.floor(i / columns) * (size + gap),
          width: size,
          height: size,
          rx: 2,
          class: i === 0 ? "block open" : "block",
        },
        `block (${bx}, ${by}, ${bz})`,
      ),
    );
  }
  return root;
}

// One block: its threads in rows of 32 (warps), which run in lockstep.
function blockDiagram(block) {
  const total = count(block);
  const warps = total === null ? 0 : Math.ceil(total / WARP);
  const shownWarps = Math.min(warps, MAX_WARPS);
  const cell = 9;
  const gap = 1;
  const label = 54;
  const root = svg("svg", {
    class: "gpu-block",
    width: label + WARP * (cell + gap),
    height: shownWarps * (cell + gap + 2),
    role: "img",
    "aria-label": `Block of ${total ?? "?"} threads in ${warps} warps`,
  });
  for (let w = 0; w < shownWarps; w++) {
    const y = w * (cell + gap + 2);
    const name = svg("text", { x: 0, y: y + cell - 1, class: "warp-label" });
    name.textContent = `warp ${w}`;
    root.append(name);
    for (let lane = 0; lane < WARP; lane++) {
      const t = w * WARP + lane;
      if (t >= total) break;
      const [tx, ty, tz] = coords(t, block);
      root.append(
        svg(
          "rect",
          {
            x: label + lane * (cell + gap),
            y,
            width: cell,
            height: cell,
            rx: 1,
            class: "thread",
          },
          `thread (${tx}, ${ty}, ${tz}) · warp ${w}, lane ${lane}`,
        ),
      );
    }
  }
  return root;
}

function launchSection(launch, kernel, options) {
  const section = el("section", "gpu-launch");
  const head = el("h3");
  head.append(
    el("code", "", kernel?.path ?? kernel?.name ?? "kernel"),
    el(
      "span",
      "gpu-dim",
      kernel?.inline
        ? " · written inline, not outlined into a kernel yet"
        : launch.host
          ? ` launched from ${launch.host}`
          : "",
    ),
  );
  section.append(head);

  const blocks = count(launch.grid);
  const perBlock = count(launch.block);
  const facts = el("dl", "gpu-facts");
  for (const [term, value] of [
    ["Grid", `${dim(launch.grid)} = ${blocks ?? "?"} blocks`],
    ["Block", `${dim(launch.block)} = ${perBlock ?? "?"} threads`],
    [
      "Threads",
      launch.threads === null
        ? "some sizes are only known at runtime"
        : `${launch.threads.toLocaleString()} in ${Math.ceil((perBlock ?? 0) / WARP) * (blocks ?? 0)} warps of ${WARP}`,
    ],
  ]) {
    facts.append(el("dt", "", term), el("dd", "", value));
  }
  section.append(facts);

  const figures = el("div", "gpu-figures");
  const gridFigure = el("figure");
  gridFigure.append(
    gridDiagram(launch.grid),
    el(
      "figcaption",
      "",
      (blocks ?? 0) > MAX_BLOCKS
        ? `First ${MAX_BLOCKS} of ${blocks} blocks. Blocks run independently, spread over the GPU's multiprocessors.`
        : "Blocks run independently, spread over the GPU's multiprocessors.",
    ),
  );
  figures.append(gridFigure);
  if (perBlock !== null) {
    const blockFigure = el("figure");
    const warps = Math.ceil(perBlock / WARP);
    blockFigure.append(
      blockDiagram(launch.block),
      el(
        "figcaption",
        "",
        `Block (0, 0, 0) opened: ${perBlock} threads as ${warps} warp(s) of ${WARP} that run in lockstep` +
          (warps > MAX_WARPS ? `; first ${MAX_WARPS} warps shown.` : ".") +
          " Hover a thread for its ids.",
      ),
    );
    figures.append(blockFigure);
  }
  section.append(figures);
  if (kernel) section.append(memorySection(kernel, launch));
  if (kernel?.accesses?.length) section.append(accessSection(kernel, launch, options));
  return section;
}

const VERDICT_TEXT = {
  coalesced: "coalesced",
  strided: "strided",
  broadcast: "broadcast",
  "conflict-free": "no bank conflicts",
  "bank-conflict": "bank conflict",
};
const GOOD = new Set(["coalesced", "broadcast", "conflict-free"]);

function judge(access, kernel, launch) {
  const memref = parseMemref(access.type);
  const space = memorySpace(memref?.space ?? "");
  return {
    space,
    memref,
    result: warpAccess(access, memref, space, {
      defs: kernel.defs,
      args: kernel.args,
      block: launch.block,
      grid: launch.grid,
    }),
  };
}

function verdictChip(result) {
  if (!result.analyzed) return el("span", "gpu-chip muted", "not analyzed");
  const detail =
    result.sectors !== undefined
      ? ` · ${result.sectors} sector${result.sectors > 1 ? "s" : ""}`
      : result.ways > 1
        ? ` · ${result.ways}-way`
        : "";
  return el(
    "span",
    `gpu-chip ${GOOD.has(result.verdict) ? "good" : "bad"}`,
    VERDICT_TEXT[result.verdict] + detail,
  );
}

// One sentence on what the verdict means for this warp.
function explain(result, space) {
  if (!result.analyzed) return `Not analyzed: ${result.reason}.`;
  const bytes = result.distinct * result.elementBytes;
  switch (result.verdict) {
    case "broadcast":
      return "Every thread of the warp uses the same element, so it is fetched once and shared.";
    case "coalesced":
      return `The warp's 32 threads use ${result.distinct} element(s), ${bytes} B, in ${result.sectors} sector(s) of 32 B: the fewest possible, so every byte moved is used.`;
    case "strided": {
      const stride = result.lanes.length > 1 ? result.lanes[1].offset - result.lanes[0].offset : 0;
      return (
        `Neighboring threads are ${stride} elements apart, so the warp moves ${result.sectors} sectors of 32 B ` +
        `(${result.sectors * 32} B) to use ${bytes} B: ${Math.round(result.efficiency * 100)}% of the traffic is useful. ` +
        "Making thread x walk the last (contiguous) index, or staging the data through shared memory, usually fixes this."
      );
    }
    case "conflict-free":
      return `Each thread uses its own bank of ${space} memory (or shares a word with another thread), so the warp is served in one pass.`;
    case "bank-conflict":
      return `Up to ${result.ways} threads need different words in the same bank, so the warp is served in ${result.ways} passes instead of one. Padding the inner dimension by one element is the usual fix.`;
    default:
      return "";
  }
}

// Loads and stores of the kernel, judged for warp 0 of block (0, 0, 0), with
// a lane map of the one picked. `options.onLine(line)` marks a source line.
function accessSection(kernel, launch, options) {
  const section = el("div", "gpu-accesses");
  section.append(
    el("h4", "", "Memory accesses"),
    el(
      "p",
      "gpu-note",
      "For warp 0 of block (0, 0, 0): which element each of its 32 threads touches. Global memory is judged by 32-byte sectors, shared memory by its 32 banks. Accesses inside a loop are shown for the first iteration.",
    ),
  );
  const judged = kernel.accesses.map((access) => ({ access, ...judge(access, kernel, launch) }));
  // Start on the first access worth a look.
  let picked = judged.findIndex(
    (j) => j.result.analyzed && !GOOD.has(j.result.verdict),
  );
  if (picked < 0) picked = 0;

  const table = el("table", "gpu-access-table");
  const head = el("tr");
  for (const title of ["Line", "Access", "Space", "Warp 0"]) head.append(el("th", "", title));
  const thead = el("thead");
  thead.append(head);
  const tbody = el("tbody");
  const detail = el("div", "gpu-lane-detail");

  const pick = (i, { mark = false } = {}) => {
    picked = i;
    [...tbody.children].forEach((row, k) => row.setAttribute("aria-selected", String(k === i)));
    renderLaneDetail(detail, judged[i]);
    if (mark) options.onLine?.(judged[i].access.line);
  };
  judged.forEach((j, i) => {
    const row = el("tr");
    row.tabIndex = 0;
    const text = `${j.access.kind === "load" ? "load" : "store"} ${j.access.buffer}[${j.access.indices.join(", ")}]`;
    row.append(
      el("td", "gpu-dim", String(j.access.line)),
      el("td", "gpu-code", text + (j.access.inLoop ? "  ↻" : "")),
      el("td", "", j.space),
    );
    const verdict = el("td");
    verdict.append(verdictChip(j.result));
    row.append(verdict);
    row.title =
      (j.access.inLoop ? "Inside a loop: first iteration shown\n" : "") +
      "Show its lanes and mark its line in the source";
    row.addEventListener("click", () => pick(i, { mark: true }));
    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pick(i, { mark: true });
      }
    });
    tbody.append(row);
  });
  table.append(thead, tbody);
  section.append(table, detail);
  pick(picked);
  return section;
}

// Group colors cycle so neighboring sectors or banks are told apart.
const groupClass = (order) => `g${order % 4}`;

function renderLaneDetail(container, { access, result, space, memref }) {
  const children = [el("p", "gpu-explain", explain(result, space))];
  if (!result.analyzed) {
    container.replaceChildren(...children);
    return;
  }
  const groups = [...new Set(result.lanes.map((l) => l.group))];
  const order = new Map(groups.map((g, i) => [g, i]));
  const unit = space === "shared" ? "bank" : "sector";

  // The 32 lanes, colored by the sector or bank they hit.
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
      const label = svg("text", { x: lane.lane * (cell + 2), y: cell + 11, class: "lane-label" });
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
      `Lanes 0–${result.lanes.length - 1}, colored by the ${unit} they hit: ${groups.length} ${unit}${groups.length > 1 ? "s" : ""}. Hover a lane for its thread and element.`,
    ),
  );
  children.push(lanesFigure);
  children.push(elementMap(access, result, memref, order));
  container.replaceChildren(...children);
}

// The buffer around the touched elements, rows by the first index (or rows
// of 32 for a 1-D buffer), touched elements colored like their lanes.
function elementMap(access, result, memref, order) {
  const twoD = memref.dims.length >= 2;
  const width = twoD ? memref.dims.at(-1) : 32;
  const at = (offset) => ({ row: Math.floor(offset / width), col: offset % width });
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
  const colStart = Math.max(0, Math.min(minCol, maxCol - MAX_COLS + 1, width - MAX_COLS));
  const cols = Math.min(width - colStart, MAX_COLS);
  const cell = 9;
  const pad = 30;
  const map = svg("svg", {
    class: "gpu-elements",
    width: pad + cols * (cell + 1),
    height: 14 + rows * (cell + 1),
    role: "img",
    "aria-label": `Elements of ${access.buffer} touched by warp 0`,
  });
  for (let r = 0; r < rows; r++) {
    const rowLabel = svg("text", { x: 0, y: 14 + r * (cell + 1) + cell - 1, class: "lane-label" });
    rowLabel.textContent = twoD ? String(minRow + r) : String((minRow + r) * width);
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
            class: lane ? `element touched ${groupClass(order.get(lane.group))}` : "element",
          },
          lane
            ? `${access.buffer}[${lane.index.join(", ")}] · lane ${lane.lane}`
            : twoD
              ? `${access.buffer}[${minRow + r}, ${colStart + c}]`
              : `${access.buffer}[${offset}]`,
        ),
      );
    }
  }
  const colLabel = svg("text", { x: pad, y: 9, class: "lane-label" });
  colLabel.textContent = twoD ? `columns ${colStart}–${colStart + cols - 1}` : "element offset";
  map.append(colLabel);
  const figure = el("figure");
  const clipped = maxRow - minRow + 1 > MAX_ROWS || maxCol - minCol + 1 > MAX_COLS;
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

function memorySection(kernel, launch) {
  const section = el("div", "gpu-memory");
  section.append(el("h4", "", "Memory"));
  if (kernel.lowered && !kernel.buffers.length && !kernel.ptx) {
    section.append(
      el(
        "p",
        "gpu-note",
        `This kernel is ${kernel.op} here, so its buffers are no longer memrefs. Step back to a pass where it is a gpu.func to see them.`,
      ),
    );
    return section;
  }
  if (!kernel.op && !kernel.buffers.length) {
    section.append(el("p", "gpu-note", "The kernel's body is not in this IR."));
    return section;
  }
  if (!kernel.buffers.length && kernel.ptx) {
    // Only the PTX is left: its buffers are plain pointers by now.
    section.append(
      el(
        "p",
        "gpu-note",
        "Only the generated PTX is left here, where buffers are plain pointers. Step back to a pass where the kernel is a gpu.func to see them by memory space.",
      ),
    );
    if (kernel.ptx.sharedBytes)
      section.append(
        el("p", "gpu-total", `PTX declares ${formatBytes(kernel.ptx.sharedBytes)} of shared memory per block`),
      );
    if (kernel.ptx.registers.length) section.append(ptxNote(kernel.ptx));
    return section;
  }

  const columns = el("div", "gpu-spaces");
  const spaces = new Map([["global", []], ["shared", []], ["private", []]]);
  for (const buffer of kernel.buffers) {
    if (!spaces.has(buffer.space)) spaces.set(buffer.space, []);
    spaces.get(buffer.space).push(buffer);
  }
  const blocks = count(launch?.grid);
  const threads = launch?.threads ?? null;
  for (const [space, buffers] of spaces) {
    if (!buffers.length && space === "private" && !kernel.ptx) continue;
    const column = el("div", `gpu-space ${space.replace(/\s+/g, "-")}`);
    const [title, blurb] = SPACE_TEXT[space] ?? [space, ""];
    const known = buffers.every((b) => b.bytes !== null);
    const bytes = buffers.reduce((sum, b) => sum + (b.bytes ?? 0), 0);
    column.append(el("h5", "", title), el("p", "gpu-note", blurb));
    const list = el("ul");
    for (const buffer of buffers) {
      const item = el("li");
      const access = [
        buffer.loads ? `${buffer.loads} load${buffer.loads > 1 ? "s" : ""}` : "",
        buffer.stores ? `${buffer.stores} store${buffer.stores > 1 ? "s" : ""}` : "",
      ].filter(Boolean);
      item.append(
        el("code", "", buffer.name),
        el("span", "gpu-type", ` ${buffer.type}`),
        el(
          "span",
          "gpu-dim",
          ` · ${buffer.bytes === null ? "dynamic size" : formatBytes(buffer.bytes)}` +
            ` · ${buffer.source}` +
            (access.length ? ` · ${access.join(", ")}` : " · not accessed"),
        ),
      );
      list.append(item);
    }
    if (!buffers.length) list.append(el("li", "gpu-dim", "none"));
    column.append(list);
    if (buffers.length && space === "shared")
      column.append(
        el(
          "p",
          "gpu-total",
          `${known ? formatBytes(bytes) : "≥ " + formatBytes(bytes)} per block` +
            (blocks ? ` · ${blocks} copies across the grid` : ""),
        ),
      );
    else if (buffers.length && space === "private")
      column.append(
        el(
          "p",
          "gpu-total",
          `${formatBytes(bytes)} per thread` + (threads ? ` · ${threads.toLocaleString()} copies` : ""),
        ),
      );
    else if (buffers.length)
      column.append(el("p", "gpu-total", `${known ? "" : "≥ "}${formatBytes(bytes)} in total`));
    if (space === "private" && kernel.ptx?.registers.length)
      column.append(ptxNote(kernel.ptx));
    columns.append(column);
  }
  section.append(columns);
  return section;
}

function ptxNote(ptx) {
  const text = ptx.registers
    .map((reg) => `${reg.count} × .${reg.type}`)
    .join(", ");
  return el(
    "p",
    "gpu-note",
    `PTX declares ${text} virtual registers per thread. ptxas maps them to physical registers later, so the real count can differ.`,
  );
}

// Fills `container` for `model` (from analyzeGpu), or explains why it is
// empty. `options.onLine(line)` is called with a 1-based source line when an
// access is picked.
export function renderGpuView(container, model, options = {}) {
  const children = [
    el(
      "p",
      "gpu-note gpu-intro",
      "Read from the IR, not measured: launch sizes come from constants and buffers from their memref types. Occupancy, caching and timing need a profiler.",
    ),
  ];
  if (!model) {
    children.push(el("p", "gpu-note", "No GPU kernels or launches in this IR."));
    container.replaceChildren(...children);
    return;
  }
  const launched = new Set();
  for (const launch of model.launches) {
    launched.add(launch.kernel);
    children.push(launchSection(launch, model.kernels[launch.kernel], options));
  }
  // Kernels with no launch in this IR (a gpu.module dumped on its own).
  model.kernels.forEach((kernel, i) => {
    if (launched.has(i)) return;
    const section = el("section", "gpu-launch");
    const head = el("h3");
    head.append(
      el("code", "", kernel.path ?? kernel.name),
      el("span", "gpu-dim", " · not launched in this IR, so launch sizes are unknown"),
    );
    section.append(head, memorySection(kernel, null));
    children.push(section);
  });
  container.replaceChildren(...children);
}
