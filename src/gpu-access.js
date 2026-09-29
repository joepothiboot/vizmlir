// Works out which element each thread of a warp touches for every memref.load
// and memref.store in a kernel, and whether the warp's accesses are coalesced
// (global memory) or conflict-free (shared memory).
//
// The lanes shown are evaluated, not solved: for one thread at a time, the SSA
// values an index is computed from are interpreted with that thread's ids, the
// launch sizes, and constants. Loop induction variables take their first
// value, so the lanes are those of warp 0 of block (0, 0, 0) on the first
// iteration. An index built from anything else (a loaded value, a block
// argument after lowering to cf, affine.apply) is reported as not analyzed
// rather than guessed.
//
// When the index is affine in the thread ids, block ids and loop counters
// (src/gpu-affine.js), proveAccess() checks the verdict for every warp of the
// launch and every iteration, not just the one shown.

import { elementBytes as elementSize } from "./buffers.js";
import { add, constant, formatAffine, linearize, scale } from "./gpu-affine.js";

export const WARP = 32;
// Global memory moves 32-byte sectors; shared memory has 32 banks of 4 bytes.
const SECTOR = 32;
const BANKS = 32;
const BANK_BYTES = 4;

const DEF = /^\s*(%[\w$.-]+)(?::\d+)?\s*=\s*"?([\w.]+)"?\s*(.*)$/;
const FOR = /\bscf\.for\s+(%[\w$.-]+)\s*=\s*(%[\w$.-]+|-?\d+)\s+to\b/;
const FOR_BOUNDS = /\bto\s+(%[\w$.-]+|-?\d+)\s+step\s+(%[\w$.-]+|-?\d+)/;
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
// scf.for induction variables are "loop" with operands [lower, upper, step],
// and the ids and sizes a gpu.launch binds are "env" values.
export function buildDefs(lines, firstLine = 1) {
  const defs = new Map();
  for (const [i, line] of lines.entries()) {
    const at = firstLine + i;
    const loop = FOR.exec(line);
    if (loop) {
      const bounds = FOR_BOUNDS.exec(line.slice(loop.index));
      const operands = bounds ? [loop[2], bounds[1], bounds[2]] : [loop[2]];
      defs.set(loop[1], { op: "loop", operands, rest: "", line: at });
    }
    if (/\bgpu\.launch\b(?!_)/.test(line) || /^\s*threads\(/.test(line)) {
      for (const [keyword, ids] of Object.entries(LAUNCH_IDS)) {
        const match = new RegExp(`\\b${keyword}\\(([^)]*)\\)\\s+in\\s+\\(([^)]*)\\)`).exec(line);
        if (!match) continue;
        match[1].split(",").forEach((name, i) =>
          defs.set(name.trim(), { op: "env", operands: [], rest: ids[i], line: at }),
        );
        match[2].split(",").forEach((part, i) => {
          const name = part.split("=")[0].trim();
          defs.set(name, { op: "env", operands: [], rest: LAUNCH_SIZES[keyword][i], line: at });
        });
      }
    }
    const def = DEF.exec(line);
    if (!def || defs.has(def[1])) continue;
    defs.set(def[1], {
      op: def[2],
      operands: def[3].match(/%[\w$.-]+/g) ?? [],
      rest: def[3],
      line: at,
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

// Judges one warp from the byte address each lane touches: global memory by
// the 32-byte sectors it moves, shared memory by its most crowded bank.
// `groups` is each lane's sector or bank, for coloring.
function judgeWarp(bytes, elementBytes, space) {
  const distinct = new Set(bytes).size;
  if (space === "shared") {
    // Lanes that read the same word share it (broadcast); different words in
    // one bank are served one after another.
    const banks = new Map();
    const groups = bytes.map((byte) => {
      const word = Math.floor(byte / BANK_BYTES);
      const bank = ((word % BANKS) + BANKS) % BANKS;
      if (!banks.has(bank)) banks.set(bank, new Set());
      banks.get(bank).add(word);
      return bank;
    });
    const ways = Math.max(...[...banks.values()].map((words) => words.size));
    return {
      distinct,
      groups,
      ways,
      verdict: distinct === 1 ? "broadcast" : ways === 1 ? "conflict-free" : "bank-conflict",
    };
  }
  const groups = bytes.map((byte) => Math.floor(byte / SECTOR));
  const sectors = new Set(groups).size;
  const needed = Math.ceil((distinct * elementBytes) / SECTOR);
  return {
    distinct,
    groups,
    sectors,
    needed,
    efficiency: (distinct * elementBytes) / (sectors * SECTOR),
    verdict: distinct === 1 ? "broadcast" : sectors <= needed ? "coalesced" : "strided",
  };
}

// Evaluates `access` for every lane of the first warp of block (0, 0, 0) and
// judges it. `memref` is parseMemref(access.type) and `space` its memory
// space name. Returns { analyzed, reason, lanes, verdict, proof, ... }; see
// proveAccess() for `proof`.
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

  const judged = judgeWarp(
    lanes.map((l) => l.byte),
    elementBytes,
    space,
  );
  lanes.forEach((lane, i) => (lane.group = judged.groups[i]));
  delete judged.groups;
  const result = { analyzed: true, lanes, elementBytes, ...judged };
  result.proof = proveAccess(access, memref, space, result, { defs, args, block, grid });
  return result;
}

const mod = (a, n) => ((a % n) + n) % n;

// Whether the verdict of the warp shown holds for every warp of the launch and
// every loop iteration. `sample` is warpAccess()'s result for that warp.
//
// The byte address is c + Σ k·v over thread ids, block ids, loop counters and
// unknown arguments. Thread ids differ within a warp, so each warp shape of
// the block (at most 32) is checked lane by lane. Everything else moves the
// whole warp by the same amount, and a verdict only depends on that amount
// modulo a period: 32 bytes for global sectors, 128 bytes (32 banks of 4
// bytes) for shared memory. So it is enough to check each residue those
// values can reach, found by stepping each one over its range (a block id
// over its grid size, a loop counter over its trip count; every residue when
// the range is unknown). Unknown ranges may add residues that never occur, so
// "varies" can be pessimistic; "proven" is always exact.
//
// Returns one of
//   { status: "proven", verdict, sectors | ways, warps, iterations, formula, laneStride }
//   { status: "varies", outcomes: [{ verdict, sectors | ways, cases }], cases, formula, laneStride }
//   { status: "sampled", reason }
// where `warps` counts the warps of the launch (null when the grid is only
// known at runtime), `iterations` says whether loop iterations were covered,
// `cases` counts the (warp shape, alignment) pairs checked, `formula` is the
// element offset as text, and `laneStride` is how many elements apart
// neighboring threads in x land.
export function proveAccess(access, memref, space, sample, { defs, args, block, grid }) {
  const [bdx, bdy, bdz] = block;
  const [gdx, gdy, gdz] = grid ?? [null, null, null];
  const ctx = {};
  let offset = constant(0);
  for (const [k, name] of access.indices.entries()) {
    const index = linearize(name, defs, { bdx, bdy, bdz, gdx, gdy, gdz, args }, ctx);
    if (!index)
      return {
        status: "sampled",
        reason: `${ctx.stuck ?? name} is not affine in the thread ids, block ids and loop counters`,
      };
    offset = add(scale(offset, memref.dims[k] ?? 1), index);
  }
  const bytes = scale(offset, sample.elementBytes);
  const period = space === "shared" ? BANKS * BANK_BYTES : SECTOR;

  const range = { bx: gdx, by: gdy, bz: gdz, ...Object.fromEntries(ctx.loops) };
  let residues = new Set([mod(bytes.c, period)]);
  for (const [v, k] of bytes.t) {
    if (v === "tx" || v === "ty" || v === "tz") continue;
    const steps = Math.max(1, Math.min(range[v] ?? period, period));
    const next = new Set();
    for (const r of residues) for (let i = 0; i < steps; i++) next.add(mod(r + k * i, period));
    residues = next;
  }

  const lane = ["tx", "ty", "tz"].map((v) => bytes.t.get(v) ?? 0);
  const perBlock = Math.ceil((bdx * bdy * bdz) / WARP);
  const count = (judged) => judged.sectors ?? judged.ways;
  const outcomes = new Map();
  let cases = 0;
  for (let w = 0; w < perBlock; w++) {
    const shift = warpLanes(block, w).map((l) => lane[0] * l.tx + lane[1] * l.ty + lane[2] * l.tz);
    for (const r of residues) {
      const judged = judgeWarp(
        shift.map((x) => r + x),
        sample.elementBytes,
        space,
      );
      const key = `${judged.verdict}:${count(judged)}`;
      if (!outcomes.has(key))
        outcomes.set(key, {
          verdict: judged.verdict,
          [space === "shared" ? "ways" : "sectors"]: count(judged),
          cases: 0,
        });
      outcomes.get(key).cases++;
      cases++;
    }
  }
  const math = { formula: formatAffine(offset, ctx.ivs), laneStride: offset.t.get("tx") ?? 0 };
  if (outcomes.size > 1)
    return {
      status: "varies",
      outcomes: [...outcomes.values()].sort((a, b) => b.cases - a.cases),
      cases,
      ...math,
    };
  const { verdict, sectors, ways } = [...outcomes.values()][0];
  const blocks = gdx !== null && gdy !== null && gdz !== null ? gdx * gdy * gdz : null;
  return {
    status: "proven",
    verdict,
    ...(sectors !== undefined ? { sectors } : { ways }),
    warps: blocks === null ? null : blocks * perBlock,
    iterations: ctx.loops.size > 0,
    ...math,
  };
}
