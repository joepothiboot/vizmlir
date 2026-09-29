// Draws the GPU view: for each kernel launch, the grid of blocks and one block
// opened into its warps and threads, and the kernel's buffers by memory
// space. Built from analyzeGpu() (src/gpu.js); plain DOM and SVG, no canvas.

import { parseMemref } from "./buffers.js";
import { warpAccess } from "./gpu-access.js";
import { analyzeGpu, memorySpace } from "./gpu.js";
import { formatBytes } from "./timing.js";
import { gpuScene, matrixWidth } from "./gpu-3d.js";
import { ownedBy, tritonAccess } from "./triton.js";

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
      kernel?.triton
        ? " · Triton GPU IR"
        : kernel?.inline
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
  // A Triton program is a block of num-warps warps; the grid is set by the
  // host at launch, which is not part of the IR.
  const rows = kernel?.triton
    ? [
        ["Programs", "set at launch (not in this IR)"],
        ["Program", `${kernel.triton.numWarps} warps = ${perBlock} threads`],
      ]
    : [
        ["Grid", `${dim(launch.grid)} = ${blocks ?? "?"} blocks`],
        ["Block", `${dim(launch.block)} = ${perBlock ?? "?"} threads`],
        [
          "Threads",
          launch.threads === null
            ? "some sizes are only known at runtime"
            : `${launch.threads.toLocaleString("en-US")} in ${(Math.ceil((perBlock ?? 0) / WARP) * (blocks ?? 0)).toLocaleString("en-US")} warps of ${WARP}`,
        ],
      ];
  for (const [term, value] of rows) {
    // Each term and value wrap together in the overlay's single line.
    const pair = el("div");
    pair.append(el("dt", "", term), el("dd", "", value));
    facts.append(pair);
  }
  // The launch in 3D, with the facts and the Memory panel laid over it.
  // Picking a row in Memory accesses colors warp 0 in it; hovering a buffer
  // outlines its memory's floor plate.
  const scene = gpuScene(launch, kernel, { onLine: options.onLine });
  // The answer for the picked access sits above the scene; picking also
  // colors warp 0 in the scene and sets its elements-from-above layer.
  const answer = kernel?.accesses?.length ? answerCard(kernel, options) : null;
  if (answer) section.append(answer);
  // Triton: which warp and lane hold each element of the picked tensor.
  const layout = kernel?.triton && answer ? el("figure", "gpu-layout") : null;
  if (layout) section.append(layout);
  section.append(scene.figure);
  const onPick = (judged) => {
    answer.show(judged);
    if (layout) renderLayout(layout, judged, kernel);
    scene.showAccess(judged, { formula: depth === "compiler" });
    // Compiler mode also writes the offset's formula in the 3D view.
    answer.onDepth = () => scene.showAccess(judged, { formula: depth === "compiler" });
    scene.showElements(
      judged.result.analyzed
        ? elementMap(judged.access, judged.result, judged.memref, groupOrder(judged.result))
        : null,
    );
  };
  const accesses = kernel?.accesses?.length
    ? accessSection(kernel, launch, { ...options, onPick })
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
// picks it in Memory accesses when it is a load or store, scrolls that
// launch's 3D view into sight (unless it already is) with a short flash,
// and remembers the line for renderGpuPath. Returns true when some launch
// has the line.
export function focusGpuLine(line) {
  shown = shown.filter((entry) => entry.section.isConnected);
  focused = null;
  for (const entry of shown) {
    const lit = entry.scene.showLine(focused ? null : line);
    if (!lit || focused) continue;
    focused = { entry, line };
    entry.accesses?.pickLine(line);
  }
  if (focused) reveal(focused.entry.scene.figure);
  return !!focused;
}

// Scrolls `figure` into its scrolling GPU view when any of it is hidden, and
// flashes its outline so the eye finds it.
function reveal(figure) {
  const view = figure.closest("#gpu-view");
  if (!view || view.hidden) return;
  const outer = view.getBoundingClientRect();
  const inner = figure.getBoundingClientRect();
  if (inner.top < outer.top || inner.bottom > outer.bottom)
    view.scrollTo({
      top: view.scrollTop + inner.top - outer.top - 12,
      behavior: "smooth",
    });
  figure.classList.remove("gpu-flash");
  void figure.offsetWidth;
  figure.classList.add("gpu-flash");
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
  misaligned: "misaligned",
  broadcast: "broadcast",
  "conflict-free": "no bank conflicts",
  "bank-conflict": "bank conflict",
};
const GOOD = new Set(["coalesced", "broadcast", "conflict-free"]);

function judge(access, kernel, launch) {
  const memref = parseMemref(access.type);
  const space = memorySpace(memref?.space ?? "");
  if (kernel.triton) return { space, memref, result: tritonAccess(access, kernel) };
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

// Plain words or the index math in the answer card; remembered per browser.
let depth = "plain";
try {
  if (localStorage.getItem("vizmlir-depth") === "compiler") depth = "compiler";
} catch {
  // Storage can be blocked; plain words it is.
}
const cards = new Set();

// The answer for the picked access: its verdict, how far that verdict was
// checked, and either one plain sentence with the share of useful traffic or
// the index math behind it. `card.show(judged)` fills it for a Memory
// accesses row. `options.passes()` (a pass trace) adds the verdict at every
// pass.
function answerCard(kernel, options) {
  const card = el("div", "gpu-answer");
  let judged = null;
  const render = () => {
    if (!judged) return;
    const { access, result, space } = judged;
    const head = el("div", "gpu-answer-top");
    head.append(
      el("span", "gpu-kicker", `Line ${access.line} · ${access.kind} ${access.buffer}`),
      verdictChip(result),
    );
    if (result.analyzed) head.append(reachChip(result.proof));
    const toggle = el("div", "gpu-depth");
    toggle.setAttribute("role", "group");
    toggle.setAttribute("aria-label", "Explain in");
    for (const [value, label] of [
      ["plain", "Plain"],
      ["compiler", "Compiler"],
    ]) {
      const button = el("button", "", label);
      button.type = "button";
      button.setAttribute("aria-pressed", String(depth === value));
      button.addEventListener("click", () => {
        depth = value;
        try {
          localStorage.setItem("vizmlir-depth", value);
        } catch {
          // Remembering is a nicety.
        }
        for (const other of cards) {
          if (!other.isConnected) {
            cards.delete(other);
            continue;
          }
          other.render();
          other.onDepth?.();
        }
      });
      toggle.append(button);
    }
    head.append(toggle);
    const body =
      depth === "compiler" && result.analyzed ? compilerBody(judged) : plainBody(result, space);
    const passes = options.passes?.();
    const history = passes
      ? passStrip(passes, acrossPasses(passes, options.launchIndex, kernel, judged))
      : [];
    card.replaceChildren(head, ...body, ...history);
  };
  card.render = render;
  card.show = (next) => {
    judged = next;
    render();
  };
  cards.add(card);
  return card;
}

function reachChip(proof) {
  if (proof?.status === "proven") return el("span", "gpu-chip proven", `✓ ${reach(proof)}`);
  if (proof?.status === "varies") return el("span", "gpu-chip bad", "varies across warps");
  return el("span", "gpu-chip muted", "warp 0 only");
}

// One sentence a beginner can act on, and a bar for how much of the memory
// traffic (global) or how many bank passes (shared) are useful.
function plainBody(result, space) {
  if (!result.analyzed) return [el("p", "gpu-answer-line", `Not analyzed: ${result.reason}.`)];
  const used = result.distinct * result.elementBytes;
  let line;
  let meter = null;
  switch (result.verdict) {
    case "broadcast":
      line = "Every thread of the warp uses the same element, so it is fetched once and shared.";
      break;
    case "coalesced":
      line = `The warp's ${result.distinct} elements sit together in ${result.sectors} chunk${result.sectors > 1 ? "s" : ""} of 32 bytes, so every byte moved is used.`;
      break;
    case "strided": {
      const times = Math.round((result.sectors * 32) / used);
      line = `Each warp touches ${result.sectors} separate 32-byte chunks, so the GPU moves ${times > 1 ? `${times}× ` : ""}more data than it uses.`;
      break;
    }
    case "misaligned":
      line = `Neighbors read neighbors, but the run starts partway into a 32-byte chunk, so the warp needs ${result.sectors} chunks instead of ${result.needed}.`;
      break;
    case "conflict-free":
      line = `Each thread uses its own bank of ${space} memory, so the warp is served in one pass.`;
      break;
    case "bank-conflict":
      line = `Up to ${result.ways} threads need the same bank, so the warp is served in ${result.ways} passes instead of one.`;
      break;
  }
  if (result.sectors !== undefined && result.verdict !== "broadcast") {
    const moved = result.sectors * 32;
    meter = trafficMeter(
      "Useful traffic",
      `${formatBytes(used)} used of ${formatBytes(moved)} moved · ${Math.round((used / moved) * 100)}%`,
      used / moved,
      result.verdict === "coalesced",
    );
  } else if (result.ways !== undefined && result.verdict !== "broadcast") {
    meter = trafficMeter(
      "Bank passes",
      `1 needed, ${result.ways} taken`,
      1 / result.ways,
      result.ways === 1,
    );
  }
  // A small stride is usually fields of a struct read one at a time.
  const stride = result.lanes.length > 1 ? Math.abs(result.lanes[1].offset - result.lanes[0].offset) : 0;
  const fix =
    !result.layout && result.verdict === "strided" && stride >= 2 && stride <= 8
      ? `neighboring threads are ${stride} elements apart, as when reading one field of an array of structs. Keep each field in its own array (a struct of arrays) so neighbors read neighbors.`
      : (result.layout ? TRITON_FIX_TEXT : FIX_TEXT)[result.verdict];
  return [
    el("p", "gpu-answer-line", line),
    ...(meter ? [meter] : []),
    // What the sentence and bar show is warp 0 on the first loop trip.
    ...(result.proof?.status === "varies" ? [el("p", "gpu-answer-fix", explainProof(result.proof))] : []),
    ...(fix ? [el("p", "gpu-answer-fix", `Usual fix: ${fix}`)] : []),
  ];
}

const FIX_TEXT = {
  strided:
    "make thread x walk the last (contiguous) index, or read a tile into shared memory and write it out row by row.",
  misaligned: "start each warp's run on a multiple of 32 bytes: shift the index, or pad the array.",
  "bank-conflict": "pad the inner dimension of the shared buffer by one element.",
};
// In Triton the layout decides which elements a warp touches together.
const TRITON_FIX_TEXT = {
  strided:
    "a layout whose order starts with the contiguous dimension, with a few elements per thread along it. The tritongpu-coalesce pass picks one when the compiler can see that dimension is contiguous.",
  misaligned: "start the tile on a multiple of 32 bytes, or tell Triton the base is aligned (tt.divisibility).",
};

function trafficMeter(label, value, share, good) {
  const meter = el("div", "gpu-meter");
  const labels = el("div", "gpu-meter-labels");
  labels.append(el("span", "", label), el("span", "gpu-dim", value));
  const bar = el("div", `gpu-meter-bar ${good ? "good" : "bad"}`);
  const fill = el("div");
  fill.style.width = `${Math.max(2, Math.min(100, share * 100))}%`;
  bar.append(fill);
  meter.append(labels, bar);
  return meter;
}

// The index math: the element offset as a function of the ids, how far apart
// neighboring threads land, and how far the verdict was checked.
function compilerBody({ access, result }) {
  const proof = result.proof;
  const rows = [];
  if (proof?.formula) {
    rows.push(["offset", `${proof.formula}   (elements of ${access.buffer})`]);
    const bytes = proof.laneStride * result.elementBytes;
    rows.push([
      proof.laneLabel ?? "∂/∂tx",
      `${proof.laneStride} element${Math.abs(proof.laneStride) === 1 ? "" : "s"} = ${bytes} B between neighboring threads`,
    ]);
  }
  rows.push([
    "warp 0",
    result.sectors !== undefined
      ? `${result.distinct} elements in ${result.sectors} sector${result.sectors > 1 ? "s" : ""} of 32 B (${result.needed} needed)`
      : `${result.distinct} words, most crowded bank ${result.ways}-way`,
  ]);
  if (proof?.status === "proven")
    rows.push([
      "holds for",
      `${proof.perProgram ? "every warp of every program" : proof.warps === null ? "every warp" : `${proof.warps.toLocaleString("en-US")} warps`}${proof.iterations ? " × every loop iteration" : ""}: affine index, every warp shape and alignment checked`,
    ]);
  else if (proof?.status === "varies")
    rows.push([
      "varies",
      proof.outcomes
        .map((o) => `${VERDICT_TEXT[o.verdict]} ${o.sectors ?? `${o.ways}-way`} in ${o.cases}/${proof.cases}`)
        .join(" · "),
    ]);
  else if (proof) rows.push(["warp 0 only", proof.reason]);
  const math = el("dl", "gpu-math");
  for (const [term, value] of rows) {
    const pair = el("div");
    pair.append(el("dt", "", term), el("dd", "", value));
    math.append(pair);
  }
  return [math];
}

// analyzeGpu for each pass's IR, worked out once per text.
const passModels = new Map();
function passModel(ir) {
  if (!passModels.has(ir)) {
    if (passModels.size > 200) passModels.clear();
    passModels.set(ir, analyzeGpu(ir));
  }
  return passModels.get(ir);
}

// The picked access in every pass of the trace. Passes rename values and
// outline kernels, so it is matched by position: the same launch, the same
// number of loads and stores in its kernel, and a load or store of the same
// kind at the same place (LLVM loads and stores count, so the match survives
// lowering). Passes where nothing matches, as once the kernel is compiled to a
// binary, get a null result.
export function acrossPasses(passes, launchIndex, kernel, judged) {
  return passes.events.map((event) => {
    const model = event.ir ? passModel(event.ir) : null;
    const launch = model?.launches[launchIndex];
    const other = launch ? model.kernels[launch.kernel] : null;
    const access = other?.accesses?.[judged.index];
    if (!access || other.accesses.length !== kernel.accesses.length || access.kind !== judged.access.kind)
      return { event, result: null };
    return { event, access, ...judge(access, other, launch) };
  });
}

// One small cell per pass, colored by its verdict, with a sentence on what
// the passes did to it. A click selects that pass.
function passStrip(passes, history) {
  const wrap = el("div", "gpu-passes");
  const cells = el("div", "gpu-pass-cells");
  const width = Math.max(2, String(history.length).length);
  history.forEach(({ event, result }, i) => {
    // A verdict that varies across warps or loop trips is not all good.
    const state = !result?.analyzed
      ? "none"
      : GOOD.has(result.verdict) && result.proof?.status !== "varies"
        ? "good"
        : "bad";
    const cell = el("button", `gpu-pass ${state}`, String(i + 1).padStart(width, "0"));
    cell.type = "button";
    if (i === passes.current) cell.setAttribute("aria-current", "step");
    cell.title =
      `${i + 1}. ${passes.describe(event)}\n` +
      (result?.analyzed
        ? verdictText(result)
        : result
          ? `not analyzed: ${result.reason}`
          : "not found: the kernel was compiled to a binary, or its loads and stores changed");
    cell.addEventListener("click", () => passes.select(i));
    cells.append(cell);
  });
  wrap.append(el("span", "gpu-kicker", "Across passes"), cells);
  wrap.append(el("p", "gpu-note", passSummary(history)));
  return [wrap];
}

const verdictText = (result) =>
  VERDICT_TEXT[result.verdict] +
  (result.sectors !== undefined
    ? ` · ${result.sectors} sector${result.sectors === 1 ? "" : "s"}`
    : result.ways > 1
      ? ` · ${result.ways}-way`
      : "") +
  (result.proof?.status === "varies" ? " for warp 0, varying across warps" : "");

// "1–3" for a run of pass numbers (0-based in), "5" for one.
const span = (first, last) => (first === last ? `${first + 1}` : `${first + 1}–${last + 1}`);

export function passSummary(history) {
  const seen = history
    .map((h, i) => ({ i, text: h.result?.analyzed ? verdictText(h.result) : null }))
    .filter((h) => h.text);
  if (!seen.length) return "Not readable in any pass.";
  const first = seen[0].i;
  const last = seen.at(-1).i;
  const lowered =
    last < history.length - 1
      ? ` From pass ${last + 2} on it can't be found: the kernel was compiled to a binary, or its loads and stores changed.`
      : "";
  const changes = seen.filter((h, k) => k && h.text !== seen[k - 1].text);
  if (!changes.length)
    return `${seen[0].text[0].toUpperCase()}${seen[0].text.slice(1)} in passes ${span(first, last)}: no pass changes it, so a fix belongs in the source.${lowered}`;
  const steps = changes.map((h) => {
    const before = seen[seen.indexOf(h) - 1];
    return `pass ${h.i + 1} turns ${before.text} into ${h.text}`;
  });
  return `${steps.join("; ")}.${lowered}`.replace(/^./, (c) => c.toUpperCase());
}

// Triton's #blocked layout for the picked tensor as a grid, one cell per
// element (up to 64 × 64), colored by the warp that holds it and labeled with
// the lane at the start of each thread's piece. Warp 0's elements on the
// layout's first repetition, the ones judged above, are outlined; later
// repetitions are paler.
const LAYOUT_MAX = 64;
function renderLayout(figure, { access, result }, kernel) {
  const encoding = access.tensor?.encoding;
  const layout = kernel.triton.layouts.get(encoding);
  const shape = access.tensor?.shape ?? [];
  if (!layout || !shape.length || shape.length > 2) {
    figure.replaceChildren(
      el("p", "gpu-note", `The layout ${encoding ?? "?"} is not a #blocked layout VizMLIR draws yet.`),
    );
    return;
  }
  const [rows, cols] = shape.length === 2 ? shape : [1, shape[0]];
  const shownRows = Math.min(rows, LAYOUT_MAX);
  const shownCols = Math.min(cols, LAYOUT_MAX);
  const at = (coord) => (shape.length === 2 ? coord : [0, coord[0]]);
  const owner = new Map();
  const { numWarps, warpSize } = kernel.triton;
  for (let warp = 0; warp < numWarps; warp++)
    for (let lane = 0; lane < warpSize; lane++)
      ownedBy(layout, shape, warp, lane).reps.forEach((coords, rep) =>
        coords.forEach((coord, k) => {
          const [r, c] = at(coord);
          const key = r * cols + c;
          if (!owner.has(key)) owner.set(key, { warp, lane, rep, first: k === 0 });
        }),
      );
  const cell = shownCols > 32 ? 8 : 14;
  const grid = svg("svg", {
    class: "gpu-layout-grid",
    viewBox: `0 0 ${shownCols * cell} ${shownRows * cell}`,
    width: shownCols * cell,
    height: shownRows * cell,
    role: "img",
    "aria-label": `Which warp and lane hold each element of ${access.buffer}'s tensor`,
  });
  for (let r = 0; r < shownRows; r++)
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
  const legend = el("div", "gpu-layout-legend");
  for (let warp = 0; warp < Math.min(numWarps, 4); warp++) {
    const item = el("span", "", `warp ${warp}${numWarps > 4 && warp === 3 ? ` (and every 4th after)` : ""}`);
    item.style.setProperty("--swatch", `var(--warp-${warp})`);
    legend.append(item);
  }
  const clipped = rows > shownRows || cols > shownCols ? ` Showing the first ${shownRows} × ${shownCols}.` : "";
  figure.replaceChildren(
    el("span", "gpu-kicker", `Who holds which element · ${encoding}`),
    el("p", "", describeLayout(layout, shape, result.elementBytes, numWarps) + clipped),
    grid,
    legend,
    el(
      "p",
      "gpu-note",
      `sizePerThread [${layout.sizePerThread.join(", ")}] · threadsPerWarp [${layout.threadsPerWarp.join(", ")}] · warpsPerCTA [${layout.warpsPerCTA.join(", ")}] · order [${layout.order.join(", ")}]. Outlined: what warp 0 ${access.kind === "load" ? "loads" : "stores"} in one go, judged above. Numbers are lane ids.`,
    ),
  );
}

// "Each thread holds 4 elements side by side along i, one 16-byte access. A
// warp covers 32 × 4 elements, and 4 warps 32 × 16, repeated 2 times."
function describeLayout(layout, shape, elementBytes, numWarps) {
  const names = shape.length === 2 ? ["i", "j"] : ["i"];
  const fast = layout.order[0];
  const run = layout.sizePerThread[fast];
  const each = layout.sizePerThread.reduce((a, b) => a * b, 1);
  const warp = shape.map((_, d) => layout.sizePerThread[d] * layout.threadsPerWarp[d]);
  const program = shape.map((_, d) => warp[d] * layout.warpsPerCTA[d]);
  const reps = shape.reduce((n, size, d) => n * Math.max(1, Math.ceil(size / program[d])), 1);
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

// Sector or bank → its color slot, in the order lanes first meet them.
function groupOrder(result) {
  const groups = [...new Set(result.lanes.map((l) => l.group))];
  return new Map(groups.map((g, i) => [g, i]));
}

// How far the verdict reaches: "all 32,768 warps", "varies" or "warp 0 only".
function reach(proof) {
  if (proof?.status === "varies") return "varies";
  if (proof?.status !== "proven") return "warp 0 only";
  const warps = proof.perProgram
    ? "every warp, every program"
    : proof.warps === null
      ? "every warp"
      : `all ${proof.warps.toLocaleString("en-US")} warps`;
  return proof.iterations ? `${warps}, every iteration` : warps;
}

// One sentence on why the warp shown may not speak for the whole launch (a
// proven verdict says so on the answer card instead).
function explainProof(proof) {
  if (proof?.status === "varies") {
    const parts = proof.outcomes.map(
      (o) =>
        `${VERDICT_TEXT[o.verdict]}${o.sectors !== undefined ? ` (${o.sectors} sectors)` : o.ways > 1 ? ` (${o.ways}-way)` : ""} in ${o.cases} of ${proof.cases}`,
    );
    return `This warp is not the whole story. Across every warp and loop iteration the address lands on different alignments: ${parts.join(", ")}.`;
  }
  return proof ? `Only this warp was checked: ${proof.reason}.` : "";
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
    case "misaligned": {
      const offBy = (result.lanes[0].byte % 32 + 32) % 32;
      return (
        `The warp's ${result.distinct} elements (${bytes} B) are side by side, but the first starts ${offBy} B into a 32-byte sector, ` +
        `so they straddle ${result.sectors} sectors instead of ${result.needed}: ${Math.round(result.efficiency * 100)}% of the traffic is useful. ` +
        "Starting each warp's run on a multiple of 32 bytes (shift the index, or pad the array) fixes this; the cost is one extra sector, not a stride."
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
      "The lanes shown are warp 0 of block (0, 0, 0) on the first loop iteration: which element each of its 32 threads touches. Global memory is judged by 32-byte sectors, shared memory by its 32 banks. When an index is a linear function of the thread ids, block ids and loop counters, the verdict is also checked for every warp and iteration.",
    ),
  );
  const judged = kernel.accesses.map((access, index) => ({
    access,
    index,
    ...judge(access, kernel, launch),
  }));
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
  for (const title of ["Line", "Access", "Space", "Verdict", "Checked"]) head.append(el("th", "", title));
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
    const checked = j.result.analyzed
      ? j.result.proof?.status === "varies"
        ? el("span", "gpu-chip bad", "varies")
        : el("span", "gpu-dim", reach(j.result.proof))
      : el("span", "gpu-dim", "");
    const reachCell = el("td");
    reachCell.append(checked);
    row.append(verdict, reachCell);
    row.title =
      (j.access.inLoop && j.result.proof?.status !== "proven" ? "Inside a loop: first iteration shown\n" : "") +
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

function renderLaneDetail(container, { access, result, space }) {
  const children = [el("p", "gpu-explain", explain(result, space))];
  // A proven verdict is already on the answer card; the rest need saying.
  if (result.analyzed && result.proof?.status !== "proven")
    children.push(el("p", "gpu-note", explainProof(result.proof)));
  if (!result.analyzed) {
    container.replaceChildren(...children);
    return;
  }
  const order = groupOrder(result);
  const groups = [...order.keys()];
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
      `Lanes 0–${result.lanes.length - 1}, colored by the ${unit} they hit: ${groups.length} ${unit}${groups.length > 1 ? "s" : ""}. Hover a lane for its thread and element; Elements in the 3D view shows them in the buffer from above.`,
    ),
  );
  children.push(lanesFigure);
  container.replaceChildren(...children);
}

// The buffer around the touched elements, rows by the first index (or rows
// of 32 for a 1-D buffer, or one with rows narrower than 32), touched
// elements colored like their lanes.
function elementMap(access, result, memref, order) {
  const twoD = matrixWidth(memref.dims) !== null;
  const width = matrixWidth(memref.dims) ?? 32;
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
  const mapWidth = pad + cols * (cell + 1);
  const mapHeight = 14 + rows * (cell + 1);
  // Sized by its container (the 3D view's Elements layer); the viewBox keeps
  // the cells square.
  const map = svg("svg", {
    class: "gpu-elements",
    viewBox: `0 0 ${mapWidth} ${mapHeight}`,
    width: mapWidth,
    height: mapHeight,
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
  model.launches.forEach((launch, launchIndex) => {
    launched.add(launch.kernel);
    children.push(launchSection(launch, model.kernels[launch.kernel], { ...options, launchIndex }));
  });
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
