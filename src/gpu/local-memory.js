// Reads a software-managed local memory (a DSP scratchpad, such as nano-dsp's
// `#dsp.local`, VTCM on Hexagon) out of IR text: which buffers live there and
// how many slots each has (memref::multiBuffer gives a double-buffered tile a
// leading dimension of 2), the DMAs that fill them (`memref.dma_start` /
// `memref.dma_wait` with their tag buffers), and the order the loop issues
// them in. After the DMAs are lowered to `linalg.copy` and the memory space is
// dropped, the same buffers are found as copy destinations used in a loop.
//
// It reads the schedule the IR expresses, nothing more: which tile is fetched
// before which compute, and into which slot. It does not know how long a DMA
// or a compute takes, so it never claims that they overlap in time.
//
// Lines are 0-based line numbers in `ir`, as in src/trace/buffers.js.

import { analyzeBuffers, elementBytes, parseMemref } from "../trace/index.js";
import { memorySpace } from "./model.js";

/**
 * Local memory per target, from nano-dsp-mlir's TargetModel
 * (lib/Schedule/TargetModel.cpp at 8c23df8). The IR does not carry it, so a
 * budget taken from here is labelled as assumed.
 */
export const TARGET_LOCAL_BYTES = {
  "hexagon-hvx128": 256 * 1024,
  "host-neon": 0,
  "x86-avx2": 0,
};
// The one nano-dsp target with local memory, assumed when the IR uses
// #dsp.local but nothing names a target.
const DEFAULT_TARGET = "hexagon-hvx128";

const FUNC = /^\s*"?func\.func"?\s+(?:(?:private|public|nested)\s+)?@([\w$.-]+)/;
const DEF = /^\s*(%[\w$.-]+)(?::\d+)?\s*=\s*"?([\w$.-]+)"?/;
const OP = /^\s*"?([A-Za-z_][\w$.-]*)"?/;
const SSA_NAME = /^%[\w$.-]+/;
const ALIAS_OPS = new Set([
  "memref.subview",
  "memref.cast",
  "memref.view",
  "memref.reinterpret_cast",
  "memref.collapse_shape",
  "memref.expand_shape",
  "memref.memory_space_cast",
  "memref.assume_alignment",
]);
const COPY_OPS = new Set(["linalg.copy", "memref.copy"]);
const MAP_ALIAS = /^\s*(#[\w$.-]+)\s*=\s*affine_map<(.*)>\s*$/;
const IR_BUDGET = /\b(?:dsp\.)?(?:local_mem_bytes|localMemBytes)\s*=\s*(\d+)/;

// Splits on commas that are not nested inside <>, (), [] or {}.
function splitTop(s) {
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if ("<([{".includes(c)) depth++;
    else if (">)]}".includes(c) && s[i - 1] !== "-") depth--;
    else if (c === "," && depth === 0) {
      parts.push(s.slice(start, i).trim());
      start = i + 1;
    }
  }
  if (s.slice(start).trim()) parts.push(s.slice(start).trim());
  return parts;
}

// The index of the ` : ` that starts an op's trailing types, outside brackets.
function typeColon(s) {
  let depth = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if ("<([{".includes(c)) depth++;
    else if (">)]}".includes(c) && s[i - 1] !== "-") depth--;
    else if (c === ":" && depth === 0 && s[i - 1] === " ") return i;
  }
  return -1;
}

const stripStrings = (line) => line.replace(/"(?:[^"\\]|\\.)*"/g, '""');

// `%x[%a, %b]` → { name: "%x", indices: ["%a", "%b"] }.
function indexed(part) {
  const match = /^(%[\w$.-]+)(?:\[(.*)\])?$/.exec(part.trim());
  return match
    ? { name: match[1], indices: match[2] ? splitTop(match[2]) : [] }
    : null;
}

// ---- Scan ------------------------------------------------------------------

// One pass over the text: functions, with every op's line, text, results and
// the regions (scf.for, scf.if, ...) it sits in.
function scan(ir) {
  const lines = ir.split("\n");
  const maps = new Map();
  const functions = [];
  let fn = null;
  let stack = [];

  for (let lineNo = 0; lineNo < lines.length; lineNo++) {
    const text = stripStrings(lines[lineNo]);
    const trimmed = text.trim();
    if (!trimmed || trimmed.startsWith("//")) continue;
    if (!fn) {
      const map = MAP_ALIAS.exec(text);
      if (map) maps.set(map[1], map[2]);
      const head = FUNC.exec(text);
      if (head && trimmed.endsWith("{")) {
        fn = { name: `@${head[1]}`, line: lineNo, ops: [], defs: new Map(), regions: [] };
        stack = [{ op: "func.func", line: lineNo }];
      }
      continue;
    }
    let rest = trimmed;
    let popped = null;
    while (rest.startsWith("}") && fn) {
      popped = stack.pop();
      popped.end = lineNo;
      rest = rest.slice(1).trimStart();
      if (stack.length === 0) {
        functions.push(fn);
        fn = null;
      }
    }
    if (!fn) continue;
    // `} {nanodsp.cache_loop}`: the attributes of the region op just closed.
    if (popped && rest.startsWith("{")) {
      popped.attrs = rest;
      continue;
    }
    if (!rest || /^[)\]:]/.test(rest) || rest.startsWith("->")) continue;

    const def = DEF.exec(rest);
    const op = def ? def[2] : OP.exec(rest)?.[1] ?? "";
    const body = def ? rest.slice(def[0].length) : rest.slice(rest.indexOf(op) + op.length);
    const entry = {
      line: lineNo,
      op,
      result: def?.[1] ?? null,
      body: body.trim(),
      regions: stack.slice(1),
    };
    fn.ops.push(entry);
    if (entry.result) {
      if (!fn.defs.has(entry.result)) fn.defs.set(entry.result, []);
      fn.defs.get(entry.result).push(entry);
    }

    let net = 0;
    for (const c of rest) net += c === "{" ? 1 : c === "}" ? -1 : 0;
    for (let i = 0; i < net; i++) {
      const region = { op, line: lineNo, end: null, attrs: "" };
      if (op === "scf.for") {
        const loop = /^(%[\w$.-]+)\s*=\s*(\S+)\s+to\s+(\S+)\s+step\s+(\S+)/.exec(entry.body);
        if (loop) [region.iv, region.lb, region.ub, region.step] = loop.slice(1);
      }
      if (op === "scf.if") region.cond = /^(%[\w$.-]+)/.exec(entry.body)?.[1] ?? null;
      fn.regions.push(region);
      stack.push(region);
    }
  }
  if (fn) functions.push(fn);
  return { functions, maps };
}

// ---- Values ----------------------------------------------------------------

// The definition of `name` visible at op `at`. The printer reuses names in
// sibling regions (a `%subview_11` in an scf.if and another in the loop
// after it), so this is the last definition before `at` in a region that
// encloses it.
function defAt(fn, name, at) {
  const defs = fn.defs.get(name);
  if (!defs) return null;
  if (!at) return defs[0];
  for (let i = defs.length - 1; i >= 0; i--) {
    const def = defs[i];
    if (def.line <= at.line && def.regions.every((r) => at.regions.includes(r)))
      return def;
  }
  return null;
}

function constantOf(fn, value, at) {
  if (/^-?\d+$/.test(value)) return Number(value);
  const def = defAt(fn, value, at);
  if (def?.op !== "arith.constant") return null;
  const match = /^(-?\d+)(?:\s*:|$)/.exec(def.body);
  return match ? Number(match[1]) : null;
}

// Follows aliases (subviews, casts) back to the value they view. `chain`
// lists the alias ops from the root outwards.
function rootOf(fn, name, at) {
  const chain = [];
  let current = name;
  for (let guard = 0; guard < 64; guard++) {
    const def = defAt(fn, current, at);
    if (!def || !ALIAS_OPS.has(def.op)) break;
    chain.unshift(def);
    const source = SSA_NAME.exec(def.body)?.[0];
    if (!source) break;
    current = source;
    at = def;
  }
  return { name: current, chain };
}

// `memref.subview %a[%o0, 0] [1, 128] [1, 1]` → offsets and sizes.
function subviewParts(def) {
  const match = /^(%[\w$.-]+)\[([^\]]*)\]\s*\[([^\]]*)\]/.exec(def.body);
  return match
    ? { source: match[1], offsets: splitTop(match[2]), sizes: splitTop(match[3]) }
    : null;
}

// The innermost loop of a list of regions, or null.
const loopOf = (regions) => [...regions].reverse().find((r) => r.op === "scf.for") ?? null;

/**
 * Which slot an index names: a constant ({ value }), or a function of the
 * enclosing loop's counter ({ of: "i" } or { of: "i+1" } for the next
 * iteration, with the map that turns it into a slot number).
 */
function slotOf(fn, maps, value, at) {
  const regions = at.regions;
  const constant = constantOf(fn, value, at);
  if (constant !== null) return { value: constant, of: null, expr: String(constant) };
  const def = defAt(fn, value, at);
  let base = value;
  let expr = value;
  if (def?.op === "affine.apply") {
    const match = /^(#[\w$.-]+|affine_map<.*?>)\s*\((%[\w$.-]+)\)/.exec(def.body);
    if (match) {
      base = match[2];
      const map = match[1].startsWith("#") ? maps.get(match[1]) : match[1].slice(11, -1);
      expr = map ? map.replace(/^.*->\s*\((.*)\)\s*$/, "$1") : match[1];
    }
  }
  const loop = loopOf(regions);
  let of = null;
  if (loop && base === loop.iv) of = "i";
  else if (loop) {
    const add = defAt(fn, base, at);
    if (add?.op === "arith.addi") {
      const [a, b] = splitTop(add.body.split(" : ")[0]);
      const other = a === loop.iv ? b : b === loop.iv ? a : null;
      if (other && constantOf(fn, other, at) === constantOf(fn, loop.step, at)) of = "i+1";
    }
  }
  return { value: null, of, expr };
}

// ---- Analysis --------------------------------------------------------------

/**
 * The target the trace's passes name (`target=` / `local-target=`), looking
 * back from event `index`, or null.
 */
export function localTargetOf(events, index = events.length - 1) {
  for (let i = Math.min(index, events.length - 1); i >= 0; i--) {
    const match = /(?:^|[\s{])(?:local-)?target=([\w.-]+)/.exec(events[i].options ?? "");
    if (match) return match[1];
  }
  return null;
}

function budgetFor(ir, target) {
  const carried = IR_BUDGET.exec(ir);
  if (carried)
    return { bytes: Number(carried[1]), source: "ir", target, note: "from the IR" };
  const name = target && target in TARGET_LOCAL_BYTES ? target : DEFAULT_TARGET;
  return {
    bytes: TARGET_LOCAL_BYTES[name],
    source: "target-model",
    target: name,
    note:
      name === target
        ? `assumed, from nano-dsp TargetModel (${name})`
        : `assumed, from nano-dsp TargetModel (${name}, its only target with local memory)`,
  };
}

/**
 * Analyzes one IR text. Returns `{ stage, target, budget, functions }`, where
 * stage is "none" (no local memory), "dma" (buffers in #dsp.local filled by
 * DMA) or "copies" (DMAs lowered to synchronous copies), and each function
 * that uses local memory has:
 *
 * - `localBuffers`: `{ name, label, line, type, shape, element, bytes,
 *   slots, slotBytes, source, lowered }`, label "A", "B", ... in source order;
 * - `tags`: the DMA tag buffers `{ name, line, slots }`;
 * - `peakLocalBytes`: the most local-buffer bytes live at once (from the
 *   lifetimes of src/trace/buffers.js);
 * - `transfers`: `{ line, op, role, src, dst, buffer, slot, tag, elements,
 *   bytes, strided }`, role "prologue" (before the loop), "prefetch" (the
 *   next iteration's tile, under a guard), "load" (this iteration's tile,
 *   single-buffered) or "wait"; `dmas` and `copies` split them by op;
 * - `pipeline`: the loop and the schedule it expresses, or null.
 *
 * @param {string} ir
 * @param {{ target?: string | null }} [options] the target named by the
 *   trace's pass options (see localTargetOf), for the budget.
 */
export function analyzeLocalMemory(ir, { target = null } = {}) {
  const { functions: scanned, maps } = scan(ir);
  const buffers = analyzeBuffers(ir);
  const functions = [];
  for (const fn of scanned) {
    const result = analyzeFunction(fn, maps, buffers.functions.find((f) => f.name === fn.name));
    if (result) functions.push(result);
  }
  const stage = !functions.length
    ? "none"
    : functions.some((f) => f.stage === "dma")
      ? "dma"
      : "copies";
  const budget = stage === "none" ? null : budgetFor(ir, target);
  for (const f of functions) f.budget = budget;
  return { stage, target: budget?.target ?? target, budget, functions };
}

/** True when the analysis has local buffers or transfers to show. */
export const hasLocalMemory = (analysis) => !!analysis && analysis.stage !== "none";

function analyzeFunction(fn, maps, bufferFn) {
  const allocs = new Map((bufferFn?.buffers ?? []).map((b) => [b.name, b]));
  const local = new Map();
  const tags = new Map();
  const transfers = [];

  const addLocal = (name, lowered) => {
    if (local.has(name) || !allocs.has(name)) return local.get(name) ?? null;
    const buffer = allocs.get(name);
    const memref = parseMemref(buffer.type);
    const entry = {
      name,
      label: "",
      line: buffer.line,
      type: buffer.type,
      shape: memref?.dims ?? null,
      element: memref?.element ?? null,
      bytes: buffer.bytes,
      slots: 1,
      slotBytes: buffer.bytes,
      source: null,
      lowered,
      start: buffer.start,
      end: buffer.end,
    };
    local.set(name, entry);
    return entry;
  };

  for (const buffer of allocs.values())
    if (memorySpace(buffer.space) === "local") addLocal(buffer.name, false);

  // Where a transfer's destination lands: the buffer and its slot.
  const target = (name, at) => {
    const { name: root, chain } = rootOf(fn, name, at);
    const first = chain[0] && subviewParts(chain[0]);
    let slot = null;
    if (first && first.source === root && first.sizes[0] === "1" && first.offsets.length > 1)
      slot = slotOf(fn, maps, first.offsets[0], chain[0]);
    return { root, slot };
  };

  const roleOf = (op, slot) => {
    const loop = loopOf(op.regions);
    if (!loop) return "prologue";
    const guarded = op.regions.slice(op.regions.indexOf(loop) + 1).some((r) => r.op === "scf.if");
    return guarded || slot?.of === "i+1" ? "prefetch" : "load";
  };

  for (const op of fn.ops) {
    if (op.op === "memref.dma_start") {
      const colon = typeColon(op.body);
      const parts = splitTop(colon < 0 ? op.body : op.body.slice(0, colon));
      const types = colon < 0 ? [] : splitTop(op.body.slice(colon + 1));
      const src = indexed(parts[0] ?? "");
      const dst = indexed(parts[1] ?? "");
      const tag = indexed(parts[3] ?? "");
      if (!src || !dst) continue;
      const into = target(dst.name, op);
      const buffer = local.get(into.root) ?? addLocal(into.root, false);
      const tagInto = tag ? target(tag.name, op) : null;
      if (tagInto) tags.set(tagInto.root, tagInto.root);
      const elements = constantOf(fn, parts[2] ?? "", op);
      const element = parseMemref(types[1] ?? types[0] ?? "")?.element;
      const elemBytes = element ? elementBytes(element) : null;
      const slot = into.slot ?? tagInto?.slot ?? null;
      if (buffer && !buffer.source) buffer.source = rootOf(fn, src.name, op).name;
      transfers.push({
        line: op.line,
        op: op.op,
        role: roleOf(op, slot),
        src: src.name,
        srcRoot: rootOf(fn, src.name, op).name,
        dst: dst.name,
        buffer: into.root,
        slot,
        tag: tagInto?.root ?? null,
        elements,
        bytes: elements !== null && elemBytes !== null ? elements * elemBytes : null,
        strided:
          parts.length > 5
            ? { stride: constantOf(fn, parts[4], op), perStride: constantOf(fn, parts[5], op) }
            : null,
        loopLine: loopOf(op.regions)?.line ?? null,
      });
    } else if (op.op === "memref.dma_wait") {
      const parts = splitTop(op.body.slice(0, Math.max(0, typeColon(op.body))) || op.body);
      const tag = indexed(parts[0] ?? "");
      if (!tag) continue;
      const into = target(tag.name, op);
      tags.set(into.root, into.root);
      transfers.push({
        line: op.line,
        op: op.op,
        role: "wait",
        src: null,
        srcRoot: null,
        dst: null,
        buffer: null, // filled from the matching dma_start below
        slot: into.slot,
        tag: into.root,
        elements: constantOf(fn, parts[1] ?? "", op),
        bytes: null,
        strided: null,
        loopLine: loopOf(op.regions)?.line ?? null,
      });
    }
  }

  // Copies: into a local buffer, or (after the memory space is dropped) into
  // a buffer of this function that a loop then reads, from a value it does
  // not own.
  const readInLoop = (root) =>
    fn.ops.some(
      (op) =>
        loopOf(op.regions) &&
        !COPY_OPS.has(op.op) &&
        (op.body.match(/%[\w$.-]+/g) ?? []).some((n) => rootOf(fn, n, op).name === root),
    );
  for (const op of fn.ops) {
    if (!COPY_OPS.has(op.op)) continue;
    let srcName;
    let dstName;
    if (op.op === "linalg.copy") {
      srcName = /ins\(\s*(%[\w$.-]+)/.exec(op.body)?.[1];
      dstName = /outs\(\s*(%[\w$.-]+)/.exec(op.body)?.[1];
    } else {
      [srcName, dstName] = (op.body.match(/%[\w$.-]+/g) ?? []).slice(0, 2);
    }
    if (!srcName || !dstName) continue;
    const into = target(dstName, op);
    const from = rootOf(fn, srcName, op).name;
    if (from === into.root || !allocs.has(into.root)) continue;
    const isLocal = local.has(into.root);
    if (!isLocal && (allocs.has(from) || !readInLoop(into.root))) continue;
    const buffer = local.get(into.root) ?? addLocal(into.root, true);
    if (!buffer.source) buffer.source = from;
    const srcType = parseMemref(/ins\([^:]*:\s*(memref<.*?>)\)/.exec(op.body)?.[1] ?? "");
    const elements = srcType?.dims?.every((d) => d !== null)
      ? srcType.dims.reduce((a, b) => a * b, 1)
      : null;
    transfers.push({
      line: op.line,
      op: op.op,
      role: roleOf(op, into.slot),
      src: srcName,
      srcRoot: from,
      dst: dstName,
      buffer: into.root,
      slot: into.slot,
      tag: null,
      elements,
      bytes: elements !== null && srcType?.bytes !== null ? srcType.bytes : null,
      strided: null,
      loopLine: loopOf(op.regions)?.line ?? null,
    });
  }

  if (!local.size && !transfers.length) return null;

  // Slots: a buffer every subview of which picks one entry of the leading
  // dimension is split into that many slots (memref::multiBuffer).
  for (const buffer of local.values()) {
    const views = fn.ops.filter(
      (op) => op.op === "memref.subview" && SSA_NAME.exec(op.body)?.[0] === buffer.name,
    );
    if (
      views.length &&
      buffer.shape?.length > 1 &&
      buffer.shape[0] > 1 &&
      views.every((v) => subviewParts(v)?.sizes[0] === "1")
    ) {
      buffer.slots = buffer.shape[0];
      buffer.slotBytes = buffer.bytes === null ? null : buffer.bytes / buffer.slots;
    }
  }
  const ordered = [...local.values()].sort((a, b) => a.line - b.line);
  ordered.forEach((b, i) => {
    b.label = String.fromCharCode(65 + (i % 26));
  });

  // A wait finishes the start that used its tag; for a slotted tag, the one
  // that will have filled the same slot.
  for (const wait of transfers.filter((t) => t.role === "wait")) {
    const start = transfers.find((t) => t.op === "memref.dma_start" && t.tag === wait.tag);
    if (start) {
      wait.buffer = start.buffer;
      wait.src = start.src;
      wait.srcRoot = start.srcRoot;
      wait.bytes = start.bytes;
    }
  }
  transfers.sort((a, b) => a.line - b.line);

  const tagList = [...tags.keys()].map((name) => {
    const buffer = allocs.get(name);
    const memref = buffer ? parseMemref(buffer.type) : null;
    return {
      name,
      line: buffer?.line ?? null,
      slots: memref?.dims?.length > 1 && memref.dims[0] > 1 ? memref.dims[0] : 1,
    };
  });

  // Peak: the most local bytes live at one position.
  let peakLocalBytes = 0;
  if (bufferFn) {
    for (let p = 0; p < bufferFn.length; p++) {
      let live = 0;
      for (const b of local.values())
        if (b.bytes !== null && b.start <= p && p <= b.end) live += b.bytes;
      peakLocalBytes = Math.max(peakLocalBytes, live);
    }
  }
  for (const b of local.values()) {
    delete b.start;
    delete b.end;
  }

  return {
    name: fn.name,
    line: fn.line,
    stage: transfers.some((t) => t.op.startsWith("memref.dma")) || ordered.some((b) => !b.lowered)
      ? "dma"
      : "copies",
    localBuffers: ordered,
    tags: tagList,
    peakLocalBytes,
    budget: null,
    transfers,
    dmas: transfers.filter((t) => t.op.startsWith("memref.dma")),
    copies: transfers.filter((t) => COPY_OPS.has(t.op)),
    pipeline: pipelineOf(fn, ordered, transfers),
  };
}

// ---- Schedule --------------------------------------------------------------

const SHOWN_ITERATIONS = 4;

/**
 * The loop the transfers serve and the order of events it expresses:
 * `{ loop, slots, mode, synchronous, computeLine, steady, iterations }`.
 * `iterations` unrolls the first few trips into lanes: `dma` (prologue
 * loads, prefetches, single-buffered loads), `wait` points, and `compute`,
 * each with the tile (iteration) and slot it touches.
 */
function pipelineOf(fn, buffers, transfers) {
  const inLoop = transfers.filter((t) => t.loopLine !== null);
  const loopLine = inLoop[0]?.loopLine ?? null;
  let loop = loopLine === null ? null : fn.regions.find((r) => r.op === "scf.for" && r.line === loopLine);
  if (!loop) {
    // No transfer in a loop: the loop that reads the buffers.
    const names = new Set(buffers.map((b) => b.name));
    const reader = fn.ops.find(
      (op) =>
        loopOf(op.regions) &&
        (op.body.match(/%[\w$.-]+/g) ?? []).some((n) => names.has(rootOf(fn, n, op).name)),
    );
    loop = reader ? loopOf(reader.regions) : null;
  }
  if (!loop) return null;

  const loopOp = fn.ops.find((op) => op.line === loop.line);
  const lb = constantOf(fn, loop.lb ?? "", loopOp);
  const ub = constantOf(fn, loop.ub ?? "", loopOp);
  const step = constantOf(fn, loop.step ?? "", loopOp);
  const trips =
    lb !== null && ub !== null && step ? Math.max(0, Math.ceil((ub - lb) / step)) : null;
  const slots = Math.max(1, ...buffers.map((b) => b.slots));
  const ofLoop = (t) => t.loopLine === loop.line;
  const prologue = transfers.filter((t) => t.role === "prologue" && t.line < loop.line);
  const prefetch = transfers.filter((t) => t.role === "prefetch" && ofLoop(t));
  const loads = transfers.filter((t) => t.role === "load" && ofLoop(t));
  const waits = transfers.filter((t) => t.role === "wait" && ofLoop(t));
  const lastTransfer = Math.max(loop.line, ...transfers.filter(ofLoop).map((t) => t.line));
  const guards = new Set(
    fn.regions.filter((r) => r.op === "scf.if" && r.line > loop.line && r.end <= loop.end).map((r) => r.line),
  );
  const compute =
    fn.ops.find(
      (op) =>
        op.line > lastTransfer &&
        op.line < (loop.end ?? Infinity) &&
        !op.regions.some((r) => guards.has(r.line)) &&
        !/^(memref\.subview|affine\.apply|arith\.|memref\.dma_)/.test(op.op),
    ) ?? null;

  const slotAt = (slot, k) => {
    if (slot?.value !== null && slot?.value !== undefined) return slot.value;
    if (slot?.of === "i") return k % slots;
    if (slot?.of === "i+1") return (k + 1) % slots;
    return 0;
  };
  // A tile that no transfer in the loop refills stays put: loaded once.
  const refilled = new Set([...prefetch, ...loads].map((t) => t.buffer));
  const shown = trips === null ? SHOWN_ITERATIONS : Math.min(trips, SHOWN_ITERATIONS);
  const iterations = [];
  for (let k = 0; k < shown; k++) {
    const events = [];
    if (k === 0)
      for (const t of prologue)
        events.push({ lane: "dma", kind: "prologue", tile: 0, slot: slotAt(t.slot, 0), transfer: t });
    for (const t of loads)
      events.push({ lane: "dma", kind: "load", tile: k, slot: slotAt(t.slot, k), transfer: t });
    const last = trips !== null && k + 1 >= trips;
    for (const t of prefetch)
      if (!last)
        events.push({ lane: "dma", kind: "prefetch", tile: k + 1, slot: slotAt(t.slot, k), transfer: t });
    for (const t of waits)
      events.push({ lane: "wait", kind: "wait", tile: k, slot: slotAt(t.slot, k), transfer: t });
    events.push({
      lane: "compute",
      kind: "compute",
      tile: k,
      uses: buffers.map((b) => ({
        buffer: b.name,
        label: b.label,
        slot: refilled.has(b.name) && b.slots > 1 ? k % b.slots : 0,
      })),
      line: compute?.line ?? loop.line,
    });
    iterations.push({ k, value: lb !== null && step ? lb + k * step : null, events });
  }

  const synchronous = !transfers.some((t) => t.op === "memref.dma_start");
  const steady = [];
  if (prefetch.length) steady.push("prefetch");
  if (loads.length) steady.push("load");
  if (waits.length) steady.push("wait");
  steady.push("compute");
  return {
    loop: {
      line: loop.line,
      end: loop.end,
      iv: loop.iv ?? null,
      lb,
      ub,
      step,
      trips,
      cacheLoop: /nanodsp\.cache_loop/.test(loop.attrs ?? ""),
    },
    slots,
    mode: prefetch.length && slots > 1 ? "double" : loads.length ? "single" : prologue.length ? "resident" : "none",
    synchronous,
    computeLine: compute?.line ?? null,
    steady,
    iterations,
    truncated: trips === null || trips > shown,
  };
}

// ---- Lookup ----------------------------------------------------------------

/**
 * What the analysis knows about 0-based line `line`: `{ kind, fn, item }`
 * with kind "buffer", "tag", "transfer" or "compute", or null.
 */
export function localLineInfo(analysis, line) {
  for (const fn of analysis?.functions ?? []) {
    const buffer = fn.localBuffers.find((b) => b.line === line);
    if (buffer) return { kind: "buffer", fn, item: buffer };
    const tag = fn.tags.find((t) => t.line === line);
    if (tag) return { kind: "tag", fn, item: tag };
    const transfer = fn.transfers.find((t) => t.line === line);
    if (transfer) return { kind: "transfer", fn, item: transfer };
    if (fn.pipeline && fn.pipeline.computeLine === line)
      return { kind: "compute", fn, item: fn.pipeline };
  }
  return null;
}
