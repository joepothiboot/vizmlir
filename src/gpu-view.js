// Draws the GPU view: for each kernel launch, the grid of blocks and one block
// opened into its warps and threads, and the kernel's buffers by memory
// space. Built from analyzeGpu() (src/gpu.js); plain DOM and SVG, no canvas.

import { parseMemref } from "./buffers.js";
import { warpAccess } from "./gpu-access.js";
import { memorySpace } from "./gpu.js";
import { formatBytes } from "./timing.js";
import { gpuScene } from "./gpu-3d.js";

const SVG = "http://www.w3.org/2000/svg";
const WARP = 32;

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
    // Each term and value wrap together in the overlay's single line.
    const pair = el("div");
    pair.append(el("dt", "", term), el("dd", "", value));
    facts.append(pair);
  }
  // The launch in 3D, with the facts and the Memory panel laid over it.
  // Picking a row in Memory accesses colors warp 0 in it; hovering a buffer
  // outlines its memory's floor plate.
  const scene = gpuScene(launch, kernel, { onLine: options.onLine });
  section.append(scene.figure);
  const accesses = kernel?.accesses?.length
    ? accessSection(kernel, launch, { ...options, onPick: scene.showAccess })
    : null;
  const memory = kernel
    ? memorySection(kernel, launch, {
        accesses,
        onLine: options.onLine,
        onSpace: scene.showSpace,
        heading: false,
      })
    : null;
  scene.overlay(facts, memory, memorySummary(kernel));
  if (accesses) section.append(accesses);
  shown.push({ section, scene, accesses, kernel, name: kernel?.path ?? kernel?.name ?? "kernel" });
  return section;
}

// The launches on screen, for focusGpuLine and renderGpuPath.
let shown = [];
let focused = null;

// Lights source line `line` (1-based) in whichever launch's kernel IR has it,
// picks it in Memory accesses when it is a load or store, and remembers it
// for renderGpuPath. Returns true when some launch has the line.
export function focusGpuLine(line) {
  shown = shown.filter((entry) => entry.section.isConnected);
  focused = null;
  for (const entry of shown) {
    const lit = entry.scene.showLine(focused ? null : line);
    if (!lit || focused) continue;
    focused = { entry, line };
    entry.accesses?.pickLine(line);
  }
  return !!focused;
}

// The path the focused line takes into the GPU, as steps a person can click
// through (`onLine(line)` marks a 1-based line): where the thread is, the
// index math, and the memory it reaches with the verdict. With nothing
// focused, lists the loads and stores to start from.
export function renderGpuPath(container, { onLine } = {}) {
  shown = shown.filter((entry) => entry.section.isConnected);
  const step = (node, entry) => {
    const button = el("button", "gpu-step");
    button.type = "button";
    button.append(el("span", "gpu-dim", `line ${node.line}`), el("code", "", node.text));
    if (node.kind === "access") {
      const judged = entry.accesses?.judged.find((j) => j.access.line === node.line);
      if (judged) button.append(el("span", "gpu-dim", ` ${judged.space} memory `), verdictChip(judged.result));
    } else if (node.kind === "thread" || node.kind === "block" || node.kind === "size") {
      button.append(el("span", "gpu-dim", ` ${node.label}`));
    }
    if (focused && node.line === focused.line) button.classList.add("current");
    button.addEventListener("click", () => onLine?.(node.line));
    return button;
  };
  const children = [];
  if (!shown.length) {
    children.push(el("p", "gpu-note", "No kernels with loads or stores to trace here."));
  } else if (!focused) {
    children.push(
      el("h3", "", "How the IR reaches the GPU"),
      el(
        "p",
        "gpu-note",
        "Click a line of kernel code (or a node on the IR wall in the 3D view) to trace it from the thread and block ids, through the index math, to the memory it touches. Or start from a load or store:",
      ),
    );
    for (const entry of shown) {
      const starts = entry.scene.flow.nodes.filter((n) => n.kind === "access");
      if (!starts.length) continue;
      children.push(el("h4", "", entry.name));
      const list = el("ul", "gpu-steps");
      for (const node of starts) {
        const item = el("li");
        item.append(step(node, entry));
        list.append(item);
      }
      children.push(list);
    }
  } else {
    const { entry, line } = focused;
    const { flow } = entry.scene;
    const lit = new Set(flow.nodes.filter((n) => entry.scene.lit?.(n.id)).map((n) => n.id));
    const nodes = flow.nodes
      .filter((n) => lit.has(n.id))
      .sort((a, b) => a.rank - b.rank || a.line - b.line);
    children.push(
      el("h3", "", `Line ${line} on its way to the GPU`),
      el("p", "gpu-note", entry.name),
    );
    for (const [title, kinds, note] of [
      ["Where the thread is", ["thread", "block", "size"], "The ids the launch gives each thread and block."],
      ["Index math", ["math", "const", "loop", "arg"], "Ops that turn those ids into an element index."],
      ["Memory", ["access"], "The element each thread reads or writes, and how the warp's accesses land."],
    ]) {
      const group = nodes.filter((n) => kinds.includes(n.kind));
      if (!group.length) continue;
      children.push(el("h4", "", title), el("p", "gpu-note", note));
      const list = el("ol", "gpu-steps");
      for (const node of group) {
        const item = el("li");
        item.append(step(node, entry));
        list.append(item);
      }
      children.push(list);
    }
    const clear = el("button", "link-btn", "show all loads and stores");
    clear.addEventListener("click", () => {
      focusGpuLine(null);
      renderGpuPath(container, { onLine });
    });
    children.push(clear);
  }
  container.replaceChildren(...children);
}

// "Memory · 3 buffers", the Memory panel's heading (the plates in the scene
// already show the sizes).
function memorySummary(kernel) {
  const n = kernel?.buffers?.length ?? 0;
  return n ? `Memory · ${n} buffer${n > 1 ? "s" : ""}` : "Memory";
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
  // Start on the access asked for, else the first one worth a look.
  const focused = judged.findIndex((j) => j.access.line === options.focusLine);
  let picked =
    focused >= 0
      ? focused
      : judged.findIndex((j) => j.result.analyzed && !GOOD.has(j.result.verdict));
  if (picked < 0) picked = 0;
  if (focused >= 0) section.dataset.focus = "true";

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
    options.onPick?.(judged[i]);
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
  // Used by the Memory section: its buffers light up and open their rows.
  section.judged = judged;
  section.pick = (i) => {
    pick(i, { mark: true });
    tbody.children[i].scrollIntoView({ block: "nearest" });
  };
  section.pickLine = (line) => {
    const i = judged.findIndex((j) => j.access.line === line);
    if (i >= 0) pick(i);
  };
  section.highlight = (name) =>
    judged.forEach((j, i) =>
      tbody.children[i].classList.toggle("gpu-linked", j.access.buffer === name),
    );
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

// Where a buffer comes from, in plain words, by its `source`.
const SOURCE_TEXT = {
  argument: "Passed in by the host when the kernel launches.",
  workgroup: "Declared with workgroup(...) on the kernel: the GPU sets aside one copy per block.",
  private: "Declared with private(...) on the kernel: one copy per thread.",
  alloc: "Allocated inside the kernel.",
  alloca: "Allocated on each thread's stack inside the kernel.",
  captured: "A host value that the launch body uses directly.",
};

// What opens under a buffer in the Memory section: its size, where it comes
// from, and its loads and stores, each linked to the Memory accesses row (or
// straight to the source line when there is no table).
function bufferDetail(buffer, { blocks, threads, accesses, onLine, kernel }) {
  const detail = el("div", "gpu-buffer-detail");
  const memref = parseMemref(buffer.type);
  const facts = el("dl", "gpu-facts");
  const shape = memref?.dims?.length
    ? `${memref.dims.map((d) => d ?? "?").join(" × ")} of ${memref.element}`
    : (memref?.element ?? buffer.type);
  const size = buffer.bytes === null ? "only known at runtime" : formatBytes(buffer.bytes);
  const copies =
    buffer.space === "shared" && blocks
      ? ` per block, ${blocks.toLocaleString()} copies across the grid`
      : buffer.space === "private" && threads
        ? ` per thread, ${threads.toLocaleString()} copies`
        : "";
  const [spaceName, spaceBlurb] = SPACE_TEXT[buffer.space] ?? [buffer.space, "a custom memory space"];
  for (const [term, value] of [
    ["Holds", shape],
    ["Size", size + copies],
    ["Lives in", `${spaceName} memory: ${spaceBlurb}`],
    ["From", SOURCE_TEXT[buffer.source] ?? buffer.source],
  ])
    facts.append(el("dt", "", term), el("dd", "", value));
  detail.append(facts);

  const uses = (accesses?.judged ?? (kernel.accesses ?? []).map((access) => ({ access })))
    .map((j, i) => ({ ...j, i }))
    .filter((j) => j.access.buffer === buffer.name);
  if (!uses.length) {
    detail.append(
      el(
        "p",
        "gpu-note",
        buffer.loads || buffer.stores
          ? "Its loads and stores could not be placed on a line here."
          : "Never read or written in this kernel.",
      ),
    );
    return detail;
  }
  const list = el("ul", "gpu-buffer-uses");
  for (const j of uses) {
    const button = el("button");
    button.type = "button";
    button.append(
      el("span", "gpu-dim", `line ${j.access.line}`),
      el(
        "span",
        "gpu-code",
        ` ${j.access.kind === "load" ? "load" : "store"} ${j.access.buffer}[${j.access.indices.join(", ")}]${j.access.inLoop ? "  ↻" : ""} `,
      ),
    );
    if (j.result) button.append(verdictChip(j.result));
    button.title = accesses ? "Show its lanes below and mark its line" : "Mark its line in the source";
    button.addEventListener("click", () =>
      accesses ? accesses.pick(j.i) : onLine?.(j.access.line),
    );
    const item = el("li");
    item.append(button);
    list.append(item);
  }
  detail.append(list);
  return detail;
}

function memorySection(kernel, launch, { accesses = null, onLine, onSpace, heading = true } = {}) {
  const section = el("div", "gpu-memory");
  if (heading) section.append(el("h4", "", "Memory"));
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
      const toggle = el("button", "gpu-buffer");
      toggle.type = "button";
      toggle.setAttribute("aria-expanded", "false");
      toggle.title = "Show its size, origin, and every load and store";
      const access = [
        buffer.loads ? `${buffer.loads} load${buffer.loads > 1 ? "s" : ""}` : "",
        buffer.stores ? `${buffer.stores} store${buffer.stores > 1 ? "s" : ""}` : "",
      ].filter(Boolean);
      toggle.append(
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
      let detail = null;
      toggle.addEventListener("click", () => {
        detail ??= bufferDetail(buffer, { blocks, threads, accesses, onLine, kernel });
        const open = toggle.getAttribute("aria-expanded") !== "true";
        toggle.setAttribute("aria-expanded", String(open));
        if (open) item.append(detail);
        else detail.remove();
      });
      const light = (on) => {
        accesses?.highlight(on ? buffer.name : null);
        onSpace?.(on ? buffer.space : null);
      };
      toggle.addEventListener("pointerenter", () => light(true));
      toggle.addEventListener("pointerleave", () => light(false));
      toggle.addEventListener("focus", () => light(true));
      toggle.addEventListener("blur", () => light(false));
      item.append(toggle);
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
// access is picked; `options.focusLine` opens the access on that line and
// scrolls to it.
export function renderGpuView(container, model, options = {}) {
  shown = [];
  focused = null;
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
    section.append(head, memorySection(kernel, null, { onLine: options.onLine }));
    children.push(section);
  });
  container.replaceChildren(...children);
  container.querySelector('[data-focus="true"]')?.scrollIntoView({ block: "start" });
}
