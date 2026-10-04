// Reads Triton GPU IR (TTGIR, the `tt.` and `ttg.` dialects) for the GPU view:
// each tt.func kernel, its `#blocked` layouts, and its tt.load / tt.store.
//
// A Triton kernel works on whole tensors. A distributed layout decides which
// thread holds which element: for `#blocked`, each thread holds a small tile
// (sizePerThread), lanes of a warp and warps of the program are laid out over
// the tensor in `order` (fastest dimension first), and the pattern repeats to
// cover a larger tensor or is shared (broadcast) over a smaller one.
//
// A load or store takes a tensor of pointers, built as base + offsets with
// tt.splat / tt.addptr from index tensors (tt.make_range, tt.expand_dims,
// tt.broadcast, elementwise arith). Offsets are written as linear expressions
// over the element coordinates (`i`, `j`, ...) and program ids, so the bytes
// each lane touches are exact and the verdict can be proven for every warp,
// every repetition of the layout and every program, like src/gpu-access.js
// does for memref code.

import { add, constant, formatAffine, isConstant, scale } from "./gpu-affine.js";
import { buildDefs, judgeWarp, WARP } from "./gpu-access.js";
import { elementBytes as elementSize } from "./trace/index.js";

const SECTOR = 32;
const LAYOUT = /^\s*(#[\w$.-]+)\s*=\s*#(?:ttg|triton_gpu)\.blocked<\{(.*)\}>\s*$/;
const FUNC = /^(\s*)tt\.func\b(?:\s+(?:public|private|nested))*\s+@([\w$.-]+)\s*\(/;
const LOAD = /^\s*(%[\w$.-]+)\s*=\s*tt\.load\s+(%[\w$.-]+)/;
const STORE = /^\s*tt\.store\s+(%[\w$.-]+)\s*,\s*(%[\w$.-]+)/;

export const isTriton = (ir) => /\btt\.func\b/.test(ir);

// `#blocked = #ttg.blocked<{sizePerThread = [1, 4], ...}>` aliases, by name.
export function parseLayouts(ir) {
  const layouts = new Map();
  for (const line of ir.split("\n")) {
    const match = LAYOUT.exec(line);
    if (!match) continue;
    const field = (name) =>
      /\[([\d,\s]*)\]/.exec(new RegExp(`\\b${name}\\s*=\\s*(\\[[\\d,\\s]*\\])`).exec(match[2])?.[1] ?? "")?.[1]
        .split(",")
        .map((s) => Number(s.trim()))
        .filter((n) => !Number.isNaN(n));
    const layout = {
      kind: "blocked",
      sizePerThread: field("sizePerThread"),
      threadsPerWarp: field("threadsPerWarp"),
      warpsPerCTA: field("warpsPerCTA"),
      order: field("order"),
    };
    if (Object.values(layout).every((v) => v !== undefined)) layouts.set(match[1], layout);
  }
  return layouts;
}

// `tensor<32x32x!tt.ptr<f32>, #blocked>` → { shape, element, pointee, encoding }.
export function parseTensor(type) {
  const open = type.indexOf("tensor<");
  if (open < 0) return null;
  let depth = 0;
  let end = -1;
  for (let i = open + 6; i < type.length; i++) {
    if (type[i] === "<") depth++;
    else if (type[i] === ">" && --depth === 0) {
      end = i;
      break;
    }
  }
  if (end < 0) return null;
  const body = type.slice(open + 7, end);
  const comma = body.search(/,\s*#/);
  const [shapeText, encoding] = comma < 0 ? [body, null] : [body.slice(0, comma), body.slice(comma + 1).trim()];
  const shape = [];
  let rest = shapeText;
  for (let m = /^(\d+)x/.exec(rest); m; m = /^(\d+)x/.exec(rest)) {
    shape.push(Number(m[1]));
    rest = rest.slice(m[0].length);
  }
  return { shape, element: rest, pointee: /^!tt\.ptr<(\w+)/.exec(rest)?.[1] ?? null, encoding };
}

// The coordinates thread (warp, lane) holds in a tensor of `shape` with a
// blocked `layout`, as { reps: [[coord, ...] per repetition] }. Coordinates
// wrap around a dimension smaller than the layout's tile (broadcast).
export function ownedBy(layout, shape, warp, lane) {
  const rank = shape.length;
  const split = (id, sizes) => {
    const parts = new Array(rank).fill(0);
    let rest = id;
    for (const d of layout.order) {
      parts[d] = rest % sizes[d];
      rest = Math.floor(rest / sizes[d]);
    }
    return parts;
  };
  const l = split(lane, layout.threadsPerWarp);
  const w = split(warp, layout.warpsPerCTA);
  const tile = shape.map((_, d) => layout.sizePerThread[d] * layout.threadsPerWarp[d] * layout.warpsPerCTA[d]);
  const count = shape.map((n, d) => Math.max(1, Math.ceil(n / tile[d])));
  const reps = [];
  const each = (dims, visit, prefix = []) =>
    prefix.length === dims.length
      ? visit(prefix)
      : Array.from({ length: dims[prefix.length] }, (_, k) => each(dims, visit, [...prefix, k]));
  each(count, (rep) => {
    const coords = [];
    each(layout.sizePerThread, (s) =>
      coords.push(
        shape.map(
          (n, d) =>
            (rep[d] * tile[d] + (w[d] * layout.threadsPerWarp[d] + l[d]) * layout.sizePerThread[d] + s[d]) % n,
        ),
      ),
    );
    reps.push(coords);
  });
  return { reps };
}

// Renames coordinate variables c<k> by `map(k)` (for tt.expand_dims).
function remap(e, map) {
  const t = new Map();
  for (const [v, k] of e.t) {
    const c = /^c(\d+)$/.exec(v);
    t.set(c ? `c${map(Number(c[1]))}` : v, k);
  }
  return { c: e.c, t };
}

// The offset (in elements of the pointee) of pointer tensor `name` from its
// base pointer, as { base, offset } with `offset` linear in the coordinates
// c0, c1, ... of that tensor and the program ids bx, by, bz; null (with
// ctx.stuck) when it is not. `ctx.firstIteration` is set when a loop-carried
// pointer is read at its first iteration.
export function pointerOffset(name, defs, ctx = {}, depth = 0) {
  const def = defs.get(name);
  if (depth > 200) return null;
  if (!def) return { base: name, offset: constant(0) };
  if (def.op === "tt.addptr") {
    const ptr = pointerOffset(def.operands[0], defs, ctx, depth + 1);
    const off = ptr && tensorLinear(def.operands[1], defs, ctx, depth + 1);
    return ptr && off ? { base: ptr.base, offset: add(ptr.offset, off) } : null;
  }
  if (def.op === "tt.splat" || def.op === "tt.broadcast" || /convert_layout$/.test(def.op))
    return pointerOffset(def.operands[0], defs, ctx, depth + 1);
  if (def.op === "tt.expand_dims") {
    const inner = pointerOffset(def.operands[0], defs, ctx, depth + 1);
    const axis = Number(/axis\s*=\s*(\d+)/.exec(def.rest)?.[1] ?? 0);
    return inner && { base: inner.base, offset: remap(inner.offset, (k) => (k >= axis ? k + 1 : k)) };
  }
  if (def.op === "iter") {
    ctx.firstIteration = true;
    return pointerOffset(def.operands[0], defs, ctx, depth + 1);
  }
  ctx.stuck ??= name;
  return null;
}

// An integer tensor (or scalar) as a linear expression over its coordinates
// c0, c1, ..., the program ids and constants; null (with ctx.stuck) when not.
export function tensorLinear(name, defs, ctx = {}, depth = 0) {
  const stuck = () => {
    ctx.stuck ??= name;
    return null;
  };
  if (/^-?\d+$/.test(name)) return constant(Number(name));
  const def = defs.get(name);
  if (!def || depth > 200) return stuck();
  const operand = (i) => tensorLinear(def.operands[i], defs, ctx, depth + 1);
  switch (def.op) {
    case "arith.constant": {
      const value = /^\s*(?:dense<)?\s*(-?\d+)\b(?!\.)/.exec(def.rest);
      return value ? constant(Number(value[1])) : stuck();
    }
    case "gpu.block_id": {
      const dim = /^\s*([xyz])\b/.exec(def.rest)?.[1];
      return dim ? { c: 0, t: new Map([[`b${dim}`, 1]]) } : stuck();
    }
    case "tt.make_range": {
      const start = Number(/start\s*=\s*(-?\d+)/.exec(def.rest)?.[1] ?? 0);
      return { c: start, t: new Map([["c0", 1]]) };
    }
    case "tt.splat":
    case "tt.broadcast":
    case "arith.extsi":
    case "arith.extui":
    case "arith.trunci":
    case "arith.index_cast":
      return operand(0);
    case "tt.expand_dims": {
      const inner = operand(0);
      const axis = Number(/axis\s*=\s*(\d+)/.exec(def.rest)?.[1] ?? 0);
      return inner && remap(inner, (k) => (k >= axis ? k + 1 : k));
    }
    case "iter":
      ctx.firstIteration = true;
      return operand(0);
  }
  if (/convert_layout$/.test(def.op)) return operand(0);
  if (def.operands.length !== 2) return stuck();
  const a = operand(0);
  const b = a && operand(1);
  if (!a || !b) return null;
  if (def.op === "arith.addi") return add(a, b);
  if (def.op === "arith.subi") return add(a, b, -1);
  if (def.op === "arith.muli") return isConstant(a) ? scale(b, a.c) : isConstant(b) ? scale(a, b.c) : stuck();
  if (def.op === "arith.shli" && isConstant(b)) return scale(a, 2 ** b.c);
  return stuck();
}

// The value of `e` at coordinates `coords` with program ids `pid`.
const valueAt = (e, coords, pid = {}) => {
  let v = e.c;
  for (const [name, k] of e.t) {
    const c = /^c(\d+)$/.exec(name);
    v += k * (c ? coords[Number(c[1])] : (pid[name] ?? 0));
  }
  return v;
};

// Loads and stores of a kernel's lines, as memref-style accesses the GPU view
// already shows: one buffer (the base pointer argument), `indices` [the
// pointer tensor], a 1-D `memref<?xT>` type, and `tensor` { shape, encoding }.
export function tritonAccesses(lines, firstLine, defs) {
  const accesses = [];
  let loops = 0;
  lines.forEach((line, i) => {
    if (/\bscf\.for\b/.test(line)) loops++;
    const load = LOAD.exec(line);
    const store = load ? null : STORE.exec(line);
    if (!load && !store) return;
    const pointer = load ? load[2] : store[1];
    const colon = line.lastIndexOf(" : ");
    const tensor = parseTensor(colon < 0 ? "" : line.slice(colon + 3));
    if (!tensor?.pointee) return;
    const origin = pointerOffset(pointer, defs);
    accesses.push({
      line: firstLine + i,
      kind: load ? "load" : "store",
      value: store ? store[2] : null,
      buffer: origin?.base ?? pointer,
      indices: [pointer],
      type: `memref<?x${tensor.pointee}>`,
      tensor: { shape: tensor.shape, encoding: tensor.encoding },
      inLoop: loops > 0,
    });
  });
  return accesses;
}

// Judges `access` like warpAccess does for memref code, for warp 0 of
// program 0 on the layout's first repetition: each lane touches all the
// elements it holds there (a vector of sizePerThread). Its `proof` checks
// every warp of the program, every repetition and every alignment the
// program ids can give.
export function tritonAccess(access, kernel) {
  const layout = kernel.triton.layouts.get(access.tensor?.encoding);
  if (!layout)
    return { analyzed: false, reason: `the layout ${access.tensor?.encoding ?? "?"} is not a #blocked layout VizMLIR reads yet` };
  if (layout.order.length !== access.tensor.shape.length)
    return { analyzed: false, reason: "the layout and the tensor have different ranks" };
  const elementBytes = elementSize(/memref<\?x(\w+)>/.exec(access.type)?.[1] ?? "");
  if (!elementBytes) return { analyzed: false, reason: "unknown element size" };
  const ctx = {};
  const origin = pointerOffset(access.indices[0], kernel.defs, ctx);
  if (!origin)
    return {
      analyzed: false,
      reason: `${ctx.stuck ?? "the offset"} is not a linear function of the element coordinates and program ids`,
    };
  const { shape } = access.tensor;
  const lanes = [];
  const bytes = [];
  const owner = [];
  for (let lane = 0; lane < kernel.triton.warpSize; lane++) {
    const [coords] = ownedBy(layout, shape, 0, lane).reps;
    const offsets = coords.map((c) => valueAt(origin.offset, c));
    lanes.push({
      lane,
      tx: lane,
      ty: 0,
      tz: 0,
      index: coords[0],
      offset: offsets[0],
      byte: offsets[0] * elementBytes,
      vector: offsets.length,
    });
    for (const o of offsets) {
      bytes.push(o * elementBytes);
      owner.push(lane);
    }
  }
  const judged = judgeWarp(bytes, elementBytes, "global");
  lanes.forEach((lane) => (lane.group = judged.groups[owner.indexOf(lane.lane)]));
  delete judged.groups;
  const result = { analyzed: true, lanes, elementBytes, ...judged, layout };
  result.proof = tritonProof(origin.offset, layout, shape, elementBytes, kernel.triton, ctx, result);
  return result;
}

const mod = (a, n) => ((a % n) + n) % n;

// Every warp of the program, every repetition of the layout, and every
// 32-byte alignment the program ids (unbounded: the grid is set at launch)
// can shift the tensor by.
function tritonProof(offset, layout, shape, elementBytes, { numWarps, warpSize }, ctx, sample) {
  let residues = new Set([mod(offset.c * elementBytes, SECTOR)]);
  for (const [v, k] of offset.t) {
    if (/^c\d+$/.test(v)) continue;
    const next = new Set();
    for (const r of residues) for (let i = 0; i < SECTOR; i++) next.add(mod(r + k * elementBytes * i, SECTOR));
    residues = next;
  }
  const coordOnly = { c: 0, t: new Map([...offset.t].filter(([v]) => /^c\d+$/.test(v))) };
  const outcomes = new Map();
  let cases = 0;
  for (let warp = 0; warp < numWarps; warp++) {
    const perLane = Array.from({ length: warpSize }, (_, lane) => ownedBy(layout, shape, warp, lane).reps);
    for (let rep = 0; rep < perLane[0].length; rep++) {
      const shift = perLane.flatMap((reps) => reps[rep].map((c) => valueAt(coordOnly, c) * elementBytes));
      for (const r of residues) {
        const judged = judgeWarp(
          shift.map((x) => r + x),
          elementBytes,
          "global",
        );
        const key = `${judged.verdict}:${judged.sectors}`;
        if (!outcomes.has(key)) outcomes.set(key, { verdict: judged.verdict, sectors: judged.sectors, cases: 0 });
        outcomes.get(key).cases++;
        cases++;
      }
    }
  }
  const names = new Map([["bx", "pid_x"], ["by", "pid_y"], ["bz", "pid_z"], ...shape.map((_, d) => [`c${d}`, "ijk"[d] ?? `c${d}`])]);
  const named = { c: offset.c, t: new Map([...offset.t].map(([v, k]) => [names.get(v) ?? v, k])) };
  const neighbor = sample.lanes.length > 1 ? sample.lanes[1].offset - sample.lanes[0].offset : 0;
  const math = { formula: formatAffine(named), laneStride: neighbor, laneLabel: "lane → lane + 1" };
  if (ctx.firstIteration)
    return { status: "sampled", reason: "a pointer carried around a loop is read at its first iteration only", ...math };
  if (outcomes.size > 1)
    return { status: "varies", outcomes: [...outcomes.values()].sort((a, b) => b.cases - a.cases), cases, ...math };
  const [{ verdict, sectors }] = outcomes.values();
  return { status: "proven", verdict, sectors, warps: null, iterations: false, perProgram: numWarps, ...math };
}

// The GPU model of a Triton module: one kernel per tt.func and a launch for
// each whose grid is set by the host at runtime (not in this IR) and whose
// program runs num-warps warps.
export function analyzeTriton(ir) {
  const lines = ir.split("\n");
  const layouts = parseLayouts(ir);
  const numWarps = Number(/"(?:ttg|triton_gpu)\.num-warps"\s*=\s*(\d+)/.exec(ir)?.[1] ?? 4);
  const warpSize = Number(/"(?:ttg|triton_gpu)\.threads-per-warp"\s*=\s*(\d+)/.exec(ir)?.[1] ?? WARP);
  const kernels = [];
  const launches = [];
  lines.forEach((line, i) => {
    const func = FUNC.exec(line);
    if (!func) return;
    let end = i + 1;
    while (end < lines.length && !new RegExp(`^${func[1]}}`).test(lines[end])) end++;
    const body = lines.slice(i, end + 1);
    const defs = buildDefs(body, i + 1);
    kernels.push({
      path: `@${func[2]}`,
      name: func[2],
      op: "tt.func",
      line: i + 1,
      inline: false,
      lowered: false,
      buffers: [],
      ptx: null,
      params: [],
      args: new Map(),
      defs,
      accesses: tritonAccesses(body, i + 1, defs),
      triton: { layouts, numWarps, warpSize },
    });
    launches.push({
      line: i + 1,
      host: null,
      kernel: kernels.length - 1,
      grid: [null, null, null],
      block: [numWarps * warpSize, 1, 1],
      threads: null,
    });
  });
  return kernels.length ? { launches, kernels, triton: true } : null;
}
