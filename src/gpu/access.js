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

import { elementBytes as elementSize } from "../trace/index.js";
import {
  add,
  constant,
  evaluateAffine,
  formatAffine,
  linearize,
  parseAffine,
  parseAffineMap,
  scale,
} from "./affine.js";

export const WARP = 32;
// Global memory moves 32-byte sectors; shared memory has 32 banks of 4 bytes.
const SECTOR = 32;
const BANKS = 32;
const BANK_BYTES = 4;

const DEF = /^\s*(%[\w$.-]+)(?::\d+)?\s*=\s*"?([\w.]+)"?\s*(.*)$/;
const FOR = /\b(?:scf|affine)\.for\s+(%[\w$.-]+)\s*=\s*(%[\w$.-]+|-?\d+)\s+to\b/;
const FOR_BOUNDS = /\bto\s+(%[\w$.-]+|-?\d+)(?:\s+step\s+(%[\w$.-]+|-?\d+))?/;
const ACCESS =
  /\b((?:memref|affine)\.(?:load|store))\s+(?:(%[\w$.-]+)\s*,\s*)?(%[\w$.-]+)\[([^\]]*)\]\s*:\s*(memref<.*>)\s*$/;
// After lowering to the LLVM dialect: an address from one getelementptr, then
// a load or store through it.
const GEP =
  /^\s*(%[\w$.-]+)\s*=\s*llvm\.getelementptr\b[^%]*(%[\w$.-]+)\[(%[\w$.-]+|-?\d+)\]\s*:\s*\(!llvm\.ptr(?:<(\d+)>)?,[^)]*\)\s*->\s*!llvm\.ptr(?:<\d+>)?,\s*(\w+)\s*$/;
const LLVM_LOAD = /^\s*(%[\w$.-]+)\s*=\s*llvm\.load\b[^%]*(%[\w$.-]+)\s*:/;
const LLVM_STORE = /^\s*llvm\.store\b[^%]*(%[\w$.-]+)\s*,\s*(%[\w$.-]+)\s*:/;
const EXTRACT = /^\s*(%[\w$.-]+)\s*=\s*llvm\.extractvalue\s+(%[\w$.-]+)\[([\d,\s]+)\]/;
const INSERT = /^\s*(%[\w$.-]+)\s*=\s*llvm\.insertvalue\s+(%[\w$.-]+)\s*,\s*(%[\w$.-]+)\[([\d,\s]+)\]/;
// Loops lowered to branches: block arguments, the branches that pass them,
// and the comparison that ends the loop.
const BLOCK_HEAD = /^\s*\^([\w$.-]+)\(([^)]*)\)\s*:/;
const BRANCH = /\b(?:cf|llvm)\.(?:br|cond_br)\b/;
const TARGET = /\^([\w$.-]+)(?:\(([^)]*)\))?/g;
const COMPARE = /\b(?:arith\.cmpi\s+(?:slt|ult)\s*,|llvm\.icmp\s+"(?:slt|ult)")\s*(%[\w$.-]+)\s*,\s*(%[\w$.-]+|-?\d+)/;
const ADDRESS_OF = /^\s*(%[\w$.-]+)\s*=\s*llvm\.mlir\.addressof\s+(@[\w$.-]+)/;
const ZERO_GEP = /^\s*(%[\w$.-]+)\s*=\s*llvm\.getelementptr\b[^%]*(%[\w$.-]+)\[0(?:\s*,\s*0)*\]/;
const MAP_ALIAS = /^\s*(#[\w$.-]+)\s*=\s*(affine_map<.*>)\s*$/;

// LLVM and NVVM (or ROCDL) ops written as their arith and gpu equivalents, so
// lowered kernels are read like unlowered ones.
const LLVM_OPS = {
  "llvm.add": "arith.addi",
  "llvm.sub": "arith.subi",
  "llvm.mul": "arith.muli",
  "llvm.sdiv": "arith.divsi",
  "llvm.udiv": "arith.divui",
  "llvm.srem": "arith.remsi",
  "llvm.urem": "arith.remui",
  "llvm.shl": "arith.shli",
  "llvm.lshr": "arith.shrui",
  "llvm.ashr": "arith.shrsi",
  "llvm.and": "arith.andi",
  "llvm.or": "arith.ori",
  "llvm.xor": "arith.xori",
  "llvm.sext": "arith.extsi",
  "llvm.zext": "arith.extui",
  "llvm.trunc": "arith.trunci",
};
const SREG = {
  tid: "gpu.thread_id",
  ctaid: "gpu.block_id",
  ntid: "gpu.block_dim",
  nctaid: "gpu.grid_dim",
  "workitem.id": "gpu.thread_id",
  "workgroup.id": "gpu.block_id",
};
function normalize(op, rest) {
  if (LLVM_OPS[op]) return { op: LLVM_OPS[op], rest };
  if (op === "llvm.mlir.constant") {
    const value = /^\s*\(\s*(-?\d+)\b(?!\.)/.exec(rest);
    if (value) return { op: "arith.constant", rest: `${value[1]} : index` };
  }
  const sreg = /^(?:nvvm\.read\.ptx\.sreg|rocdl)\.(tid|ctaid|ntid|nctaid|workitem\.id|workgroup\.id)\.([xyz])$/.exec(op);
  if (sreg) return { op: SREG[sreg[1]], rest: sreg[2] };
  // A Triton program is a block: its id and count are block id and grid size.
  if (op === "tt.get_program_id") return { op: "gpu.block_id", rest };
  if (op === "tt.get_num_programs") return { op: "gpu.grid_dim", rest };
  return { op, rest };
}

// The `#map = affine_map<...>` aliases of a module, by name.
export function affineAliases(ir) {
  const maps = new Map();
  for (const line of ir.split("\n")) {
    const alias = MAP_ALIAS.exec(line);
    if (alias) maps.set(alias[1], alias[2]);
  }
  return maps;
}
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
// scf.for and affine.for induction variables are "loop" with operands
// [lower, upper, step], the ids and sizes a gpu.launch binds are "env" values,
// and affine.apply carries its parsed `map` (inline, or from `maps`, the
// module's aliases). LLVM and NVVM ops are stored as their arith and gpu
// equivalents (see normalize).
export function buildDefs(lines, firstLine = 1, maps = new Map()) {
  const defs = new Map();
  for (const [i, line] of lines.entries()) {
    const at = firstLine + i;
    const loop = FOR.exec(line);
    if (loop) {
      const bounds = FOR_BOUNDS.exec(line.slice(loop.index));
      const operands = bounds ? [loop[2], bounds[1], bounds[2] ?? "1"] : [loop[2]];
      defs.set(loop[1], { op: "loop", operands, rest: "", line: at });
      // Values carried around the loop: "iter", holding their first value.
      const carried = /\biter_args\s*\(([^)]*)\)/.exec(line)?.[1] ?? "";
      for (const pair of carried.matchAll(/(%[\w$.-]+)\s*=\s*(%[\w$.-]+)/g))
        defs.set(pair[1], { op: "iter", operands: [pair[2]], rest: "", line: at });
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
    const { op, rest } = normalize(def[2], def[3]);
    const entry = { op, operands: rest.match(/%[\w$.-]+/g) ?? [], rest, line: at };
    if (op === "affine.apply") {
      const alias = /^\s*(#[\w$.-]+)/.exec(rest)?.[1];
      entry.map = parseAffineMap(alias ? (maps.get(alias) ?? "") : (/affine_map<.*>(?=\s*\()/.exec(rest)?.[0] ?? ""));
    }
    defs.set(def[1], entry);
  }
  branchLoops(lines, firstLine, defs);
  return defs;
}

// After scf-to-cf, `scf.for %i = %lb to %ub step %s` is a block argument that
// one branch starts at %lb, the loop's back edge passes as `%i + %s`, and a
// `cmpi slt %i, %ub` ends. Such block arguments become "loop" defs, like
// scf.for induction variables; other block arguments stay undefined.
function branchLoops(lines, firstLine, defs) {
  const params = new Map(); // block → its argument names
  const incoming = new Map(); // block argument → the values branches pass it
  const heads = [];
  lines.forEach((line, i) => {
    const head = BLOCK_HEAD.exec(line);
    if (head) {
      const names = head[2].split(",").map((part) => part.split(":")[0].trim());
      params.set(head[1], names);
      heads.push({ names, line: firstLine + i });
    }
  });
  for (const line of lines) {
    if (!BRANCH.test(line)) continue;
    for (const target of line.matchAll(TARGET)) {
      const names = params.get(target[1]);
      if (!names || !target[2]) continue;
      target[2]
        .split(":")[0]
        .split(",")
        .map((v) => v.trim())
        .forEach((value, k) => {
          if (!names[k]) return;
          if (!incoming.has(names[k])) incoming.set(names[k], []);
          incoming.get(names[k]).push(value);
        });
    }
  }
  const bounds = new Map();
  for (const line of lines) {
    const compare = COMPARE.exec(line);
    if (compare) bounds.set(compare[1], compare[2]);
  }
  for (const { names, line } of heads)
    for (const name of names) {
      const values = incoming.get(name) ?? [];
      if (defs.has(name) || values.length !== 2 || !bounds.has(name)) continue;
      // The back edge: `%next = arith.addi %name, %step` (either order).
      const back = values.findIndex((v) => {
        const d = defs.get(v);
        return d?.op === "arith.addi" && d.operands.length === 2 && d.operands.includes(name);
      });
      if (back < 0) continue;
      const next = defs.get(values[back]);
      const step = next.operands[0] === name ? next.operands[1] : next.operands[0];
      defs.set(name, { op: "loop", operands: [values[1 - back], bounds.get(name), step], rest: "", line });
    }
}

// Evaluates SSA value `name` for one thread. `env` holds tx..tz, bx..bz,
// bdx..bdz, gdx..gdz and kernel argument values by name. Returns a number, or
// null with the value that stopped it in `trace.stuck`.
export function evaluate(name, defs, env, trace = {}, depth = 0) {
  if (/^-?\d+$/.test(name)) return Number(name);
  // An affine.load index such as `%i * 2 + %j`.
  if (!/^%[\w$.-]+$/.test(name)) {
    const tree = parseAffine(name);
    if (!tree) {
      trace.stuck ??= name;
      return null;
    }
    return evaluateAffine(tree, (leaf) => evaluate(leaf, defs, env, trace, depth + 1));
  }
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
  } else if (def.op === "affine.apply" && def.map) {
    const names = [...def.map.dims, ...def.map.syms];
    const value = evaluateAffine(def.map.expr, (leaf) => {
      const k = names.indexOf(leaf);
      return k < 0 ? null : operand(k);
    });
    if (value !== null) return value;
  }
  trace.stuck ??= name;
  return null;
}

// The loads and stores in `lines`, as [{ line, kind, value, buffer, indices,
// type, inLoop }] where `line` is 1-based from `firstLine`. affine.load and
// affine.store indices are affine expressions (`%i * 2 + %j`). An LLVM load or
// store through a getelementptr becomes a 1-D access whose buffer is the
// kernel argument the pointer came from and whose type is `memref<?xT>` (with
// the pointer's address space).
export function findAccesses(lines, firstLine = 1) {
  const accesses = [];
  const loops = [];
  const geps = new Map();
  const extracts = new Map();
  const inserts = new Map();
  // Pointers that are another one unchanged: a global's address, or a
  // zero-offset getelementptr of one (how workgroup buffers are lowered).
  const aliases = new Map();
  for (const line of lines) {
    const address = ADDRESS_OF.exec(line);
    if (address) aliases.set(address[1], address[2]);
    const zero = ZERO_GEP.exec(line);
    if (zero) aliases.set(zero[1], zero[2]);
    const gep = GEP.exec(line);
    if (gep) geps.set(gep[1], { base: gep[2], index: gep[3], space: gep[4] ?? "", element: gep[5] });
    const extract = EXTRACT.exec(line);
    if (extract) extracts.set(extract[1], { from: extract[2], at: extract[3].replace(/\s+/g, "") });
    const insert = INSERT.exec(line);
    if (insert) inserts.set(insert[1], { value: insert[2], into: insert[3], at: insert[4].replace(/\s+/g, "") });
  }
  // `%p = llvm.extractvalue %desc[1]` back through the insertvalue chain that
  // built %desc, to the argument (or global) stored at [1].
  const root = (name, depth = 0) => {
    if (depth > 64) return name;
    if (aliases.has(name)) return root(aliases.get(name), depth + 1);
    const extract = extracts.get(name);
    if (!extract) return name;
    for (let s = extract.from, n = 0; inserts.has(s) && n < 64; n++) {
      const insert = inserts.get(s);
      if (insert.at === extract.at) return root(insert.value, depth + 1);
      s = insert.into;
    }
    return name;
  };
  const llvmAccess = (line, kind, value, pointer) => {
    const gep = geps.get(pointer);
    if (!gep) return;
    accesses.push({
      line,
      kind,
      value,
      buffer: root(gep.base),
      indices: [gep.index],
      type: `memref<?x${gep.element}${gep.space ? `, ${gep.space}` : ""}>`,
      inLoop: false,
    });
  };
  lines.forEach((line, i) => {
    if (line.trim() === "") return;
    const indent = /^\s*/.exec(line)[0].length;
    while (loops.length && indent <= loops.at(-1)) loops.pop();
    if (FOR.test(line)) loops.push(indent);
    const load = LLVM_LOAD.exec(line);
    if (load) return llvmAccess(firstLine + i, "load", null, load[2]);
    const store = LLVM_STORE.exec(line);
    if (store) return llvmAccess(firstLine + i, "store", store[1], store[2]);
    const match = ACCESS.exec(line);
    if (!match) return;
    accesses.push({
      line: firstLine + i,
      kind: match[1].endsWith("load") ? "load" : "store",
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
export function judgeWarp(bytes, elementBytes, space) {
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
  // Elements side by side (no gaps) that still take an extra sector start
  // partway into one: misaligned, not strided.
  const span = Math.max(...bytes) - Math.min(...bytes) + elementBytes;
  const dense = span === distinct * elementBytes;
  return {
    distinct,
    groups,
    sectors,
    needed,
    efficiency: (distinct * elementBytes) / (sectors * SECTOR),
    verdict:
      distinct === 1
        ? "broadcast"
        : sectors <= needed
          ? "coalesced"
          : dense
            ? "misaligned"
            : "strided",
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
