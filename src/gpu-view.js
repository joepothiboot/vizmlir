// Draws the GPU view: for each kernel launch, the grid of blocks and one block
// opened into its warps and threads, and the kernel's buffers by memory
// space. Built from analyzeGpu() (src/gpu.js); plain DOM and SVG, no canvas.

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

function launchSection(launch, kernel) {
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
  return section;
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

// Fills `container` for `model` (from analyzeGpu), or explains why it is empty.
export function renderGpuView(container, model) {
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
    children.push(launchSection(launch, model.kernels[launch.kernel]));
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
