// Works out which element each thread of a warp touches for every memref.load
// and memref.store in a kernel, and whether the warp's accesses are coalesced
// (global memory) or conflict-free (shared memory).
//
// Indices are evaluated, not solved: for one thread at a time, the SSA values
// an index is computed from are interpreted with that thread's ids, the launch
// sizes, and constants. Loop induction variables take their first value, so
// an access inside a loop is judged on its first iteration. An index built
// from anything else (a loaded value, a block argument after lowering to cf,
// affine.apply) is reported as not analyzed rather than guessed.

import { elementBytes as elementSize } from "./buffers.js";

export const WARP = 32;
// Global memory moves 32-byte sectors; shared memory has 32 banks of 4 bytes.
const SECTOR = 32;
const BANKS = 32;
const BANK_BYTES = 4;

const DEF = /^\s*(%[\w$.-]+)(?::\d+)?\s*=\s*"?([\w.]+)"?\s*(.*)$/;
const FOR = /\bscf\.for\s+(%[\w$.-]+)\s*=\s*(%[\w$.-]+|-?\d+)\s+to\b/;
const ACCESS =
  /\b(memref\.load|memref\.store)\s+(?:(%[\w$.-]+)\s*,\s*)?(%[\w$.-]+)\[([^\]]*)\]\s*:\s*(memref<.*>)\s*$/;
const LAUNCH_IDS = {
  blocks: ["bx", "by", "bz"],
  threads: ["tx", "ty", "tz"],
};
const LAUNCH_SIZES = {
  blocks: ["gdx", "gdy", "gdz"],
  threads: ["bdx", "bdy", "bdz"],
};
const ID_OPS = {
  "gpu.thread_id": "t",
  "gpu.block_id": "b",
  "gpu.block_dim": "bd",
  "gpu.grid_dim": "gd",
};
const BINARY = {
  "arith.addi": (a, b) => a + b,
  "arith.subi": (a, b) => a - b,
  "arith.muli": (a, b) => a * b,
  "arith.divsi": (a, b) => (b ? Math.trunc(a / b) : null),
  "arith.divui": (a, b) => (b ? Math.floor(a / b) : null),
  "arith.floordivsi": (a, b) => (b ? Math.floor(a / b) : null),
  "arith.ceildivsi": (a, b) => (b ? Math.ceil(a / b) : null),
  "arith.remsi": (a, b) => (b ? a % b : null),
  "arith.remui": (a, b) => (b ? a % b : null),
  "arith.minsi": Math.min,
  "arith.minui": Math.min,
  "arith.maxsi": Math.max,
  "arith.maxui": Math.max,
  "arith.andi": (a, b) => a & b,
  "arith.ori": (a, b) => a | b,
  "arith.xori": (a, b) => a ^ b,
  "arith.shli": (a, b) => a << b,
  "arith.shrui": (a, b) => a >>> b,
  "arith.shrsi": (a, b) => a >> b,
};
const CASTS = new Set([
  "arith.index_cast",
  "arith.index_castui",
  "arith.extsi",
  "arith.extui",
  "arith.trunci",
]);

// Definitions of SSA values in `lines`: Map(name → { op, operands, rest }).
// scf.for induction variables are "loop" (their lower bound), and the ids
// and sizes a gpu.launch binds are "env" values.
export function buildDefs(lines) {
  const defs = new Map();
  for (const line of lines) {
    const loop = FOR.exec(line);
    if (loop) defs.set(loop[1], { op: "loop", operands: [loop[2]], rest: "" });
    if (/\bgpu\.launch\b(?!_)/.test(line) || /^\s*threads\(/.test(line)) {
      for (const [keyword, ids] of Object.entries(LAUNCH_IDS)) {
        const match = new RegExp(`\\b${keyword}\\(([^)]*)\\)\\s+in\\s+\\(([^)]*)\\)`).exec(line);
        if (!match) continue;
        match[1].split(",").forEach((name, i) =>
          defs.set(name.trim(), { op: "env", operands: [], rest: ids[i] }),
        );
        match[2].split(",").forEach((part, i) => {
          const name = part.split("=")[0].trim();
          defs.set(name, { op: "env", operands: [], rest: LAUNCH_SIZES[keyword][i] });
        });
      }
    }
    const def = DEF.exec(line);
    if (!def || defs.has(def[1])) continue;
    defs.set(def[1], {
      op: def[2],
      operands: def[3].match(/%[\w$.-]+/g) ?? [],
      rest: def[3],
    });
  }
  return defs;
}

// Evaluates SSA value `name` for one thread. `env` holds tx..tz, bx..bz,
// bdx..bdz, gdx..gdz and kernel argument values by name. Returns a number, or
// null with the value that stopped it in `trace.stuck`.
export function evaluate(name, defs, env, trace = {}, depth = 0) {
  if (/^-?\d+$/.test(name)) return Number(name);
  if (env.args?.has(name)) {
    const value = env.args.get(name);
    if (value === null) trace.stuck ??= name;
    return value;
  }
  const def = defs.get(name);
  if (!def || depth > 200) {
    trace.stuck ??= name;
    return null;
  }
  const operand = (i) => evaluate(def.operands[i], defs, env, trace, depth + 1);
  if (def.op === "env") return env[def.rest] ?? null;
  if (def.op === "loop") return operand(0);
  if (def.op === "arith.constant") {
    const match = /^(-?\d+)\b(?!\.)/.exec(def.rest.trim());
    if (match) return Number(match[1]);
  } else if (ID_OPS[def.op]) {
    const dim = /^\s*([xyz])\b/.exec(def.rest)?.[1];
    if (dim) return env[`${ID_OPS[def.op]}${dim}`] ?? null;
  } else if (BINARY[def.op] && def.operands.length === 2) {
    const a = operand(0);
    const b = a === null ? null : operand(1);
    if (a !== null && b !== null) return BINARY[def.op](a, b);
    return null;
  } else if (CASTS.has(def.op) && def.operands.length === 1) {
    return operand(0);
  }
  trace.stuck ??= name;
  return null;
}

// The loads and stores in `lines`, as [{ line, kind, value, buffer, indices,
// type, inLoop }] where `line` is 1-based from `firstLine`.
export function findAccesses(lines, firstLine = 1) {
  const accesses = [];
  const loops = [];
  lines.forEach((line, i) => {
    if (line.trim() === "") return;
    const indent = /^\s*/.exec(line)[0].length;
    while (loops.length && indent <= loops.at(-1)) loops.pop();
    if (FOR.test(line)) loops.push(indent);
    const match = ACCESS.exec(line);
    if (!match) return;
    accesses.push({
      line: firstLine + i,
      kind: match[1] === "memref.load" ? "load" : "store",
      value: match[2] ?? null,
      buffer: match[3],
      indices: match[4].split(",").map((s) => s.trim()).filter(Boolean),
      type: match[5],
      inLoop: loops.length > (FOR.test(line) ? 1 : 0),
    });
  });
  return accesses;
}

// Thread ids of each lane of warp `warp` in a block of size `block`.
export function warpLanes(block, warp = 0) {
  const [x, y, z] = block.map((d) => d ?? 1);
  const lanes = [];
  for (let lane = 0; lane < WARP; lane++) {
    const t = warp * WARP + lane;
    if (t >= x * y * z) break;
    lanes.push({ lane, tx: t % x, ty: Math.floor(t / x) % y, tz: Math.floor(t / (x * y)) });
  }
  return lanes;
}

// Evaluates `access` for every lane of the first warp of block (0, 0, 0) and
// judges it. `memref` is parseMemref(access.type) and `space` its memory
// space name. Returns { analyzed, reason, lanes, verdict, ... }.
export function warpAccess(access, memref, space, { defs, args, block, grid }) {
  if (!block || block.some((d) => d === null))
    return { analyzed: false, reason: "the block size is only known at runtime" };
  // Row-major offsets need every dimension but the first.
  if (!memref?.dims || memref.dims.slice(1).some((d) => d === null))
    return { analyzed: false, reason: "the buffer has a dynamic shape" };
  const elementBytes = elementSize(memref.element);
  if (!elementBytes) return { analyzed: false, reason: `unknown element size of ${memref.element}` };
  const [bdx, bdy, bdz] = block;
  const [gdx, gdy, gdz] = (grid ?? [null, null, null]).map((d) => d ?? 1);

  const lanes = [];
  for (const ids of warpLanes(block)) {
    const env = { ...ids, bx: 0, by: 0, bz: 0, bdx, bdy, bdz, gdx, gdy, gdz, args };
    const trace = {};
    const index = access.indices.map((name) => evaluate(name, defs, env, trace));
    if (index.some((v) => v === null))
      return {
        analyzed: false,
        reason: `${trace.stuck ?? "an index"} is not computed from thread ids, loop starts and constants`,
      };
    // Row-major offset in elements.
    let offset = 0;
    for (let k = 0; k < index.length; k++) offset = offset * (memref.dims[k] ?? 1) + index[k];
    lanes.push({ ...ids, index, offset, byte: offset * elementBytes });
  }
  if (!lanes.length) return { analyzed: false, reason: "the block has no threads" };

  const distinct = new Set(lanes.map((l) => l.offset));
  const result = { analyzed: true, lanes, elementBytes, distinct: distinct.size };
  if (space === "shared") {
    // Lanes that read the same word share it (broadcast); different words in
    // one bank are served one after another.
    const banks = new Map();
    for (const lane of lanes) {
      const word = Math.floor(lane.byte / BANK_BYTES);
      lane.group = word % BANKS;
      if (!banks.has(lane.group)) banks.set(lane.group, new Set());
      banks.get(lane.group).add(word);
    }
    const ways = Math.max(...[...banks.values()].map((words) => words.size));
    return {
      ...result,
      ways,
      verdict:
        distinct.size === 1 ? "broadcast" : ways === 1 ? "conflict-free" : "bank-conflict",
    };
  }
  for (const lane of lanes) lane.group = Math.floor(lane.byte / SECTOR);
  const sectors = new Set(lanes.map((l) => l.group)).size;
  const needed = Math.ceil((distinct.size * elementBytes) / SECTOR);
  const efficiency = (distinct.size * elementBytes) / (sectors * SECTOR);
  return {
    ...result,
    sectors,
    needed,
    efficiency,
    verdict:
      distinct.size === 1
        ? "broadcast"
        : sectors <= needed
          ? "coalesced"
          : "strided",
  };
}
