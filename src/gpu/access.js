import { BANKS, BANK_BYTES, SECTOR, WARP } from "../constants.js";
import { elementBytes as elementSize } from "../trace/index.js";
import { evaluate } from "./access-ir.js";
import { add, constant, formatAffine, linearize, scale } from "./affine.js";

export function warpLanes(block, warp = 0) {
  const [x, y, z] = block.map((d) => d ?? 1);
  const lanes = [];

  for (let lane = 0; lane < WARP; lane++) {
    const t = warp * WARP + lane;
    if (t >= x * y * z) break;

    lanes.push({
      lane,
      tx: t % x,
      ty: Math.floor(t / x) % y,
      tz: Math.floor(t / (x * y)),
    });
  }

  return lanes;
}

function bankVerdict(distinct, ways) {
  if (distinct === 1) return "broadcast";
  if (ways === 1) return "conflict-free";

  return "bank-conflict";
}

function globalVerdict(distinct, sectors, needed, dense) {
  if (distinct === 1) return "broadcast";
  if (sectors <= needed) return "coalesced";
  if (dense) return "misaligned";

  return "strided";
}

export function judgeWarp(bytes, elementBytes, space) {
  const distinct = new Set(bytes).size;

  if (space === "shared") {
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
      verdict: bankVerdict(distinct, ways),
    };
  }

  const groups = bytes.map((byte) => Math.floor(byte / SECTOR));
  const sectors = new Set(groups).size;
  const needed = Math.ceil((distinct * elementBytes) / SECTOR);
  const span = Math.max(...bytes) - Math.min(...bytes) + elementBytes;
  const dense = span === distinct * elementBytes;

  return {
    distinct,
    groups,
    sectors,
    needed,
    efficiency: (distinct * elementBytes) / (sectors * SECTOR),
    verdict: globalVerdict(distinct, sectors, needed, dense),
  };
}

export function warpAccess(access, memref, space, { defs, args, block, grid }) {
  if (!block || block.some((d) => d === null)) {
    return {
      analyzed: false,
      reason: "the block size is only known at runtime",
    };
  }

  if (!memref?.dims || memref.dims.slice(1).some((d) => d === null)) {
    return { analyzed: false, reason: "the buffer has a dynamic shape" };
  }

  const elementBytes = elementSize(memref.element);

  if (!elementBytes) {
    return {
      analyzed: false,
      reason: `unknown element size of ${memref.element}`,
    };
  }

  const [bdx, bdy, bdz] = block;
  const [gdx, gdy, gdz] = (grid ?? [null, null, null]).map((d) => d ?? 1);

  const lanes = [];

  for (const ids of warpLanes(block)) {
    const env = {
      ...ids,
      bx: 0,
      by: 0,
      bz: 0,
      bdx,
      bdy,
      bdz,
      gdx,
      gdy,
      gdz,
      args,
    };

    const trace = {};

    const index = access.indices.map((name) =>
      evaluate(name, defs, env, trace),
    );

    if (index.some((v) => v === null)) {
      return {
        analyzed: false,
        reason: `${trace.stuck ?? "an index"} is not computed from thread ids, loop starts and constants`,
      };
    }

    let offset = 0;

    for (let k = 0; k < index.length; k++) {
      offset = offset * (memref.dims[k] ?? 1) + index[k];
    }

    lanes.push({ ...ids, index, offset, byte: offset * elementBytes });
  }

  if (!lanes.length) {
    return { analyzed: false, reason: "the block has no threads" };
  }

  const judged = judgeWarp(
    lanes.map((l) => l.byte),
    elementBytes,
    space,
  );

  lanes.forEach((lane, i) => (lane.group = judged.groups[i]));
  delete judged.groups;

  const result = { analyzed: true, lanes, elementBytes, ...judged };

  result.proof = proveAccess(access, memref, space, result, {
    defs,
    args,
    block,
    grid,
  });

  return result;
}

const mod = (a, n) => ((a % n) + n) % n;

export function proveAccess(
  access,
  memref,
  space,
  sample,
  { defs, args, block, grid },
) {
  const [bdx, bdy, bdz] = block;
  const [gdx, gdy, gdz] = grid ?? [null, null, null];
  const ctx = {};
  let offset = constant(0);

  for (const [k, name] of access.indices.entries()) {
    const index = linearize(
      name,
      defs,
      { bdx, bdy, bdz, gdx, gdy, gdz, args },
      ctx,
    );

    if (!index) {
      return {
        status: "sampled",
        reason: `${ctx.stuck ?? name} is not affine in the thread ids, block ids and loop counters`,
      };
    }

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

    for (const r of residues) {
      for (let i = 0; i < steps; i++) next.add(mod(r + k * i, period));
    }

    residues = next;
  }

  const lane = ["tx", "ty", "tz"].map((v) => bytes.t.get(v) ?? 0);
  const perBlock = Math.ceil((bdx * bdy * bdz) / WARP);
  const count = (judged) => judged.sectors ?? judged.ways;
  const outcomes = new Map();
  let cases = 0;

  for (let w = 0; w < perBlock; w++) {
    const shift = warpLanes(block, w).map(
      (l) => lane[0] * l.tx + lane[1] * l.ty + lane[2] * l.tz,
    );

    for (const r of residues) {
      const judged = judgeWarp(
        shift.map((x) => r + x),
        sample.elementBytes,
        space,
      );

      const key = `${judged.verdict}:${count(judged)}`;

      if (!outcomes.has(key)) {
        outcomes.set(key, {
          verdict: judged.verdict,
          [space === "shared" ? "ways" : "sectors"]: count(judged),
          cases: 0,
        });
      }

      outcomes.get(key).cases++;
      cases++;
    }
  }

  const math = {
    formula: formatAffine(offset, ctx.ivs),
    laneStride: offset.t.get("tx") ?? 0,
  };

  if (outcomes.size > 1) {
    return {
      status: "varies",
      outcomes: [...outcomes.values()].sort((a, b) => b.cases - a.cases),
      cases,
      ...math,
    };
  }

  const { verdict, sectors, ways } = [...outcomes.values()][0];

  const blocks =
    gdx !== null && gdy !== null && gdz !== null ? gdx * gdy * gdz : null;

  return {
    status: "proven",
    verdict,
    ...(sectors !== undefined ? { sectors } : { ways }),
    warps: blocks === null ? null : blocks * perBlock,
    iterations: ctx.loops.size > 0,
    ...math,
  };
}

export function expectedCost(result, space) {
  const unit = space === "shared" ? "ways" : "sectors";
  const proof = result.proof;

  if (proof?.status === "proven") {
    const n = proof[unit];

    return {
      status: "proven",
      verdict: proof.verdict,
      min: n,
      max: n,
      mean: n,
    };
  }

  if (proof?.status === "varies") {
    const values = proof.outcomes.map((o) => o[unit]);

    const mean =
      proof.outcomes.reduce((s, o) => s + o[unit] * o.cases, 0) / proof.cases;

    return {
      status: "varies",
      verdict: proof.outcomes.map((o) => o.verdict).join("/"),
      min: Math.min(...values),
      max: Math.max(...values),
      mean,
    };
  }

  return {
    status: "sampled",
    verdict: result.verdict,
    min: result[unit],
    max: result[unit],
    mean: result[unit],
  };
}
