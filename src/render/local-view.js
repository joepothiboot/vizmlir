// The "Local memory" view: for IR that stages tiles in a software-managed
// local memory (see src/gpu/local-memory.js), a memory map of the budget with
// each buffer's slots as blocks, and a timeline of the first loop trips with
// a DMA lane, the wait points and a compute lane. Every block and event is a
// button that picks its IR line.
//
// The timeline is the order the IR issues things in, not measured time: the
// boxes have equal widths because the IR says nothing about durations.

import { formatBytes } from "../trace/index.js";

const ROLE_TEXT = {
  prologue: "loaded before the loop",
  prefetch: "prefetch of the next tile",
  load: "loaded in the loop (single-buffered)",
  wait: "wait for the tile",
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// A button that picks `line` (0-based) of the IR.
function pickButton(className, line, label, onPick) {
  const button = el("button", className);
  button.type = "button";
  button.dataset.line = String(line);
  button.setAttribute("aria-label", label);
  button.title = `${label} · line ${line + 1}`;
  button.addEventListener("click", () => onPick?.(line, button));
  return button;
}

// Buffers A, B, C, D take the theme's warp hues, the two most distinct first.
const HUES = [0, 2, 1, 3];
const colorOf = (label) => `var(--warp-${HUES[(label.charCodeAt(0) - 65) % 4]})`;
const percent = (part, whole) => `${Math.max(0, Math.min(100, (part / whole) * 100))}%`;

function stageNote(analysis) {
  if (analysis.stage === "copies")
    return "After -nanodsp-lower-local: each DMA is now a synchronous linalg.copy into the same slot, and the buffers are back in the default memory space. The copy of the next tile runs before the compute, not beside it.";
  return "Tiles are copied into local memory by DMA. A dma_start only starts a transfer; the matching dma_wait blocks until it is done, so the work between the two is what the transfer can hide behind.";
}

// ---- Memory map ------------------------------------------------------------

function memoryMap(fn, onPick) {
  const section = el("section", "lm-section");
  section.append(el("h4", "", "Memory map"));
  const budget = fn.budget?.bytes ?? null;
  const total = budget && budget >= fn.peakLocalBytes ? budget : Math.max(fn.peakLocalBytes, 1);
  const facts = el("p", "lm-facts");
  facts.append(
    `Peak ${formatBytes(fn.peakLocalBytes)}`,
    budget !== null
      ? ` of ${formatBytes(budget)} (${Math.round((fn.peakLocalBytes / budget) * 100)}%)`
      : "",
  );
  section.append(facts);
  if (fn.budget) section.append(el("p", "lm-note", `Budget: ${fn.budget.note}.`));
  if (budget !== null && fn.peakLocalBytes > budget)
    section.append(el("p", "lm-warn", "The buffers need more than the budget."));

  const bar = el("div", "lm-bar");
  bar.setAttribute("role", "list");
  bar.setAttribute("aria-label", "Local memory, buffer slots in allocation order");
  for (const buffer of fn.localBuffers) {
    const slots = buffer.slots;
    for (let slot = 0; slot < slots; slot++) {
      const bytes = buffer.slotBytes ?? 0;
      const name = slots > 1 ? `${buffer.label} slot ${slot}` : buffer.label;
      const block = pickButton(
        "lm-block",
        buffer.line,
        `${name}: ${formatBytes(bytes)} from ${buffer.source ?? "?"} (${buffer.name})`,
        onPick,
      );
      block.setAttribute("role", "listitem");
      block.style.width = percent(bytes, total);
      block.style.setProperty("--lm-color", colorOf(buffer.label));
      block.append(el("span", "lm-block-name", name));
      bar.append(block);
    }
  }
  if (budget !== null && budget > fn.peakLocalBytes) {
    const free = el("div", "lm-free");
    free.setAttribute("role", "listitem");
    free.style.width = percent(budget - fn.peakLocalBytes, total);
    free.textContent = `free ${formatBytes(budget - fn.peakLocalBytes)}`;
    bar.append(free);
  }
  section.append(bar);

  const list = el("ul", "lm-legend");
  for (const buffer of fn.localBuffers) {
    const item = el("li");
    const swatch = el("span", "lm-swatch");
    swatch.style.background = colorOf(buffer.label);
    swatch.setAttribute("aria-hidden", "true");
    const button = pickButton(
      "link-btn",
      buffer.line,
      `${buffer.label}: ${buffer.name}, line ${buffer.line + 1}`,
      onPick,
    );
    button.textContent = `${buffer.label} = ${buffer.name}`;
    item.append(
      swatch,
      button,
      ` · tile of ${buffer.source ?? "?"} · ${buffer.slots > 1 ? `${buffer.slots} slots of ${formatBytes(buffer.slotBytes)}` : formatBytes(buffer.bytes)}`,
      buffer.lowered ? " · was #dsp.local" : "",
    );
    list.append(item);
  }
  for (const tag of fn.tags) {
    const item = el("li", "lm-dim");
    const button = pickButton("link-btn", tag.line ?? 0, `DMA tag ${tag.name}`, onPick);
    button.textContent = tag.name;
    item.append("DMA tags ", button, ` · ${tag.slots} slot${tag.slots === 1 ? "" : "s"}, in main memory`);
    list.append(item);
  }
  section.append(list);
  return section;
}

// ---- Timeline --------------------------------------------------------------

function eventLabel(event, fn) {
  const buffer = fn.localBuffers.find((b) => b.name === event.transfer?.buffer);
  const tile = buffer?.label ?? "?";
  if (event.kind === "wait") return `wait ${tile}${event.tile} (slot ${event.slot})`;
  return `${tile}${event.tile} → slot ${event.slot}`;
}

function timeline(fn, onPick) {
  const { pipeline } = fn;
  const section = el("section", "lm-section");
  section.append(el("h4", "", "Schedule"));
  const loop = pipeline.loop;
  const head = el("p", "lm-facts");
  head.append(
    `Loop at line ${loop.line + 1}`,
    loop.trips !== null ? ` · ${loop.trips} trip${loop.trips === 1 ? "" : "s"}` : " · trip count not known",
    loop.cacheLoop ? " · marked nanodsp.cache_loop" : "",
    ` · ${{ double: "double-buffered", single: "single-buffered", resident: "loaded once", none: "no transfers" }[pipeline.mode]}`,
  );
  section.append(head);
  section.append(
    el(
      "p",
      "lm-note",
      "The order the IR expresses, not measured timing: every box has the same width because the IR says nothing about how long a transfer or a compute takes." +
        (pipeline.synchronous ? " Copies are synchronous, so nothing overlaps here." : ""),
    ),
  );

  // Scrolls sideways on its own on narrow screens; focusable so the arrow
  // keys can scroll it too.
  const scroller = el("div", "lm-scroll");
  scroller.tabIndex = 0;
  scroller.setAttribute("role", "region");
  scroller.setAttribute("aria-label", "Schedule, scrolls sideways");
  const grid = el("div", "lm-grid");
  grid.setAttribute("role", "table");
  grid.setAttribute("aria-label", "Schedule of the first loop trips");
  const columns = pipeline.iterations.length + 1;
  grid.style.setProperty("--lm-cols", String(columns));

  const row = (cells) => {
    const r = el("div", "lm-row");
    r.setAttribute("role", "row");
    r.append(...cells);
    grid.append(r);
  };
  const cell = (className, header) => {
    const c = el("div", `lm-cell ${className}`);
    c.setAttribute("role", header ? "columnheader" : "cell");
    return c;
  };
  row([
    cell("lm-lane", true),
    ...[
      "before the loop",
      ...pipeline.iterations.map((it) => `i = ${it.k}${it.value !== null ? ` (${loop.iv ?? "iv"} = ${it.value})` : ""}`),
    ].map((text) => {
      const c = cell("lm-col", true);
      c.textContent = text;
      return c;
    }),
  ]);

  const lanes = [
    ["dma", pipeline.synchronous ? "Copy" : "DMA"],
    ["wait", "Wait"],
    ["compute", "Compute"],
  ].filter(([lane]) => lane !== "wait" || pipeline.steady.includes("wait"));
  for (const [lane, name] of lanes) {
    const label = cell("lm-lane");
    label.setAttribute("role", "rowheader");
    label.textContent = name;
    const cells = [cell("")];
    for (const it of pipeline.iterations) cells.push(cell(""));
    for (const it of pipeline.iterations)
      for (const event of it.events) {
        if (event.lane !== lane) continue;
        const column = event.kind === "prologue" ? 0 : it.k + 1;
        let button;
        if (lane === "compute") {
          const uses = event.uses.map((u) => `${u.label} slot ${u.slot}`).join(", ");
          button = pickButton("lm-event lm-compute", event.line, `compute tile ${event.tile}, reads ${uses}`, onPick);
          button.textContent = `tile ${event.tile} · ${event.uses.map((u) => `${u.label}${u.slot}`).join(" ")}`;
        } else {
          const t = event.transfer;
          const text = eventLabel(event, fn);
          button = pickButton(
            `lm-event lm-${event.kind}`,
            t.line,
            `${text}: ${ROLE_TEXT[event.kind]} (${t.op})`,
            onPick,
          );
          const buffer = fn.localBuffers.find((b) => b.name === t.buffer);
          if (buffer) button.style.setProperty("--lm-color", colorOf(buffer.label));
          button.textContent = event.kind === "prefetch" ? `${text} · next` : text;
        }
        cells[column].append(button);
      }
    row([label, ...cells]);
  }
  scroller.append(grid);
  section.append(scroller);

  const steady = el("p", "lm-note");
  steady.textContent = `Each trip: ${pipeline.steady
    .map((step) => ({ prefetch: "start the next tile", load: "load this tile", wait: "wait for this tile", compute: "compute on it" })[step])
    .join(", then ")}.${pipeline.truncated ? ` Only the first ${pipeline.iterations.length} trips are drawn.` : ""}`;
  section.append(steady);
  return section;
}

function transferTable(fn, onPick) {
  const section = el("section", "lm-section");
  section.append(el("h4", "", pipelineTitle(fn)));
  const list = el("ul", "lm-transfers");
  for (const t of fn.transfers) {
    const item = el("li");
    const buffer = fn.localBuffers.find((b) => b.name === t.buffer);
    const button = pickButton("link-btn", t.line, `${t.op} on line ${t.line + 1}`, onPick);
    button.textContent = `line ${t.line + 1}`;
    const slot = t.slot
      ? t.slot.value !== null
        ? `slot ${t.slot.value}`
        : t.slot.of
          ? `slot of ${t.slot.of === "i+1" ? "the next trip" : "this trip"}`
          : "slot ?"
      : "";
    item.append(
      button,
      ` ${t.op.replace(/^memref\./, "")} · ${ROLE_TEXT[t.role]} · ${buffer?.label ?? "?"} ${slot}`,
      t.srcRoot && t.role !== "wait" ? ` · from ${t.srcRoot}` : "",
      t.bytes !== null && t.role !== "wait" ? ` · ${formatBytes(t.bytes)}` : "",
      t.strided ? ` · ${t.strided.perStride} elements every ${t.strided.stride}` : "",
    );
    list.append(item);
  }
  section.append(list);
  return section;
}

const pipelineTitle = (fn) => (fn.dmas.length ? "DMA operations" : "Copies");

/**
 * Draws the view for `analysis` (from analyzeLocalMemory) into `root`.
 * `onPick(line, element)` is called with a 0-based IR line.
 */
export function renderLocalView(root, analysis, { onPick, pass } = {}) {
  root.replaceChildren();
  const intro = el("div", "lm-intro");
  intro.append(el("p", "lm-note", stageNote(analysis)));
  if (pass) intro.prepend(el("p", "lm-pass", pass));
  root.append(intro);
  for (const fn of analysis.functions) {
    const card = el("article", "lm-fn");
    const title = el("h3");
    const code = el("code", "", fn.name);
    title.append("Local memory in ", code);
    card.append(title, memoryMap(fn, onPick));
    if (fn.pipeline) card.append(timeline(fn, onPick));
    if (fn.transfers.length) card.append(transferTable(fn, onPick));
    root.append(card);
  }
}

/** Marks the blocks and events of 0-based line `line` as picked. */
export function focusLocalLine(root, line) {
  for (const button of root.querySelectorAll("[data-line]")) {
    const on = Number(button.dataset.line) === line;
    button.classList.toggle("picked", on);
    if (button.classList.contains("lm-block") || button.classList.contains("lm-event"))
      button.setAttribute("aria-pressed", String(on));
  }
}

/**
 * Plain-words details of a line for the inspector's Line tab, from
 * localLineInfo(), or null.
 */
export function localExplain(info) {
  if (!info) return null;
  const box = el("div", "le-local");
  const head = el("p");
  const strong = el("strong", "", "Local memory: ");
  head.append(strong);
  const { fn, item } = info;
  if (info.kind === "buffer") {
    head.append(
      `buffer ${item.label} holds tiles of ${item.source ?? "an input"} in local memory` +
        (item.slots > 1
          ? `, in ${item.slots} slots of ${formatBytes(item.slotBytes)} (the loop fills one slot while it computes on the other).`
          : `, ${formatBytes(item.bytes ?? 0)}.`) +
        (item.lowered ? " The pass that lowered the DMAs dropped its #dsp.local memory space." : ""),
    );
  } else if (info.kind === "tag") {
    head.append(`${item.name} holds the DMA tags: each dma_start marks one, and the dma_wait on the same tag blocks until that transfer is done.`);
  } else if (info.kind === "transfer") {
    const buffer = fn.localBuffers.find((b) => b.name === item.buffer);
    const tile = buffer ? `the ${buffer.label} tile` : "a tile";
    const sentence = {
      prologue: `starts the copy of ${tile} for the first trip into slot ${item.slot?.value ?? 0}, before the loop.`,
      prefetch: `starts the copy of ${tile} for the next trip into the slot the previous trip used, while this trip computes on the other slot. The guard skips it on the last trip.`,
      load: `copies ${tile} for this trip; with one slot, the compute has to wait for it.`,
      wait: `blocks until ${tile} for this trip has arrived; the compute after it may then read the slot.`,
    }[item.role];
    head.append(
      (item.op === "linalg.copy" || item.op === "memref.copy"
        ? `this synchronous copy (lowered from a DMA) `
        : "this DMA ") + sentence,
    );
    if (item.bytes !== null && item.role !== "wait")
      box.append(el("p", "lm-dim", `${formatBytes(item.bytes)} from ${item.srcRoot}${item.strided ? `, ${item.strided.perStride} elements every ${item.strided.stride}` : ""}.`));
  } else if (info.kind === "compute") {
    head.append("the compute of one trip starts here, reading the slots the waits before it made ready.");
  }
  box.prepend(head);
  return box;
}
