import { el } from "../dom.js";
import { memorySpace, tritonAccess, warpAccess } from "../gpu/index.js";
import { formatBytes, parseMemref } from "../trace/index.js";

export const VERDICT_TEXT = {
  coalesced: "coalesced",
  strided: "strided",
  misaligned: "misaligned",
  broadcast: "broadcast",
  "conflict-free": "no bank conflicts",
  "bank-conflict": "bank conflict",
};

export const GOOD = new Set(["coalesced", "broadcast", "conflict-free"]);

export function judge(access, kernel, launch) {
  const memref = parseMemref(access.type);
  const space = memorySpace(memref?.space ?? "");

  if (kernel.triton) {
    return { space, memref, result: tritonAccess(access, kernel) };
  }

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

function costDetail({ sectors, ways }) {
  if (sectors !== undefined) {
    return ` · ${sectors} sector${sectors > 1 ? "s" : ""}`;
  }

  if (ways > 1) return ` · ${ways}-way`;

  return "";
}

export function verdictChip(result) {
  if (!result.analyzed) return el("span", "gpu-chip muted", "not analyzed");

  const detail = costDetail(result);

  return el(
    "span",
    `gpu-chip ${GOOD.has(result.verdict) ? "good" : "bad"}`,
    VERDICT_TEXT[result.verdict] + detail,
  );
}

export function verdictSummary({ access, result }) {
  const where = el(
    "span",
    "where",
    `Line ${access.line} · ${access.kind} ${access.buffer}`,
  );

  const parts = [where, verdictChip(result)];

  if (result.analyzed) {
    const reached = reachChip(result.proof);
    reached.classList.add("reach");
    parts.push(reached);
  }

  return parts;
}

export function reachChip(proof) {
  if (proof?.status === "proven") {
    return el("span", "gpu-chip proven", `✓ ${reach(proof)}`);
  }

  if (proof?.status === "varies") {
    return el("span", "gpu-chip bad", "varies across warps");
  }

  return el("span", "gpu-chip muted", "warp 0 only");
}

export function plainBody(result, space) {
  if (!result.analyzed) {
    return [el("p", "gpu-answer-line", `Not analyzed: ${result.reason}.`)];
  }

  const used = result.distinct * result.elementBytes;
  let line;
  let meter = null;

  switch (result.verdict) {
    case "broadcast":
      line =
        "Every thread of the warp uses the same element, so it is fetched once and shared.";

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

  const stride =
    result.lanes.length > 1
      ? Math.abs(result.lanes[1].offset - result.lanes[0].offset)
      : 0;

  const fix =
    !result.layout && result.verdict === "strided" && stride >= 2 && stride <= 8
      ? `neighboring threads are ${stride} elements apart, as when reading one field of an array of structs. Keep each field in its own array (a struct of arrays) so neighbors read neighbors.`
      : (result.layout ? TRITON_FIX_TEXT : FIX_TEXT)[result.verdict];

  return [
    el("p", "gpu-answer-line", line),
    ...(meter ? [meter] : []),
    ...(result.proof?.status === "varies"
      ? [el("p", "gpu-answer-fix", explainProof(result.proof))]
      : []),
    ...(fix ? [el("p", "gpu-answer-fix", `Usual fix: ${fix}`)] : []),
  ];
}

const FIX_TEXT = {
  strided:
    "make thread x walk the last (contiguous) index, or read a tile into shared memory and write it out row by row.",
  misaligned:
    "start each warp's run on a multiple of 32 bytes: shift the index, or pad the array.",
  "bank-conflict":
    "pad the inner dimension of the shared buffer by one element.",
};

const TRITON_FIX_TEXT = {
  strided:
    "a layout whose order starts with the contiguous dimension, with a few elements per thread along it. The tritongpu-coalesce pass picks one when the compiler can see that dimension is contiguous.",
  misaligned:
    "start the tile on a multiple of 32 bytes, or tell Triton the base is aligned (tt.divisibility).",
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

function warpScope(proof, perProgram, countPrefix) {
  if (proof.perProgram) return perProgram;
  if (proof.warps === null) return "every warp";

  return `${countPrefix}${proof.warps.toLocaleString("en-US")} warps`;
}

export function compilerBody({ access, result }) {
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

  if (proof?.status === "proven") {
    rows.push([
      "holds for",
      `${warpScope(proof, "every warp of every program", "")}${proof.iterations ? " × every loop iteration" : ""}: affine index, every warp shape and alignment checked`,
    ]);
  } else if (proof?.status === "varies") {
    rows.push([
      "varies",
      proof.outcomes
        .map(
          (o) =>
            `${VERDICT_TEXT[o.verdict]} ${o.sectors ?? `${o.ways}-way`} in ${o.cases}/${proof.cases}`,
        )
        .join(" · "),
    ]);
  } else if (proof) {
    rows.push(["warp 0 only", proof.reason]);
  }

  const math = el("dl", "gpu-math");

  for (const [term, value] of rows) {
    const pair = el("div");
    pair.append(el("dt", "", term), el("dd", "", value));
    math.append(pair);
  }

  return [math];
}

export function groupOrder(result) {
  const groups = [...new Set(result.lanes.map((l) => l.group))];

  return new Map(groups.map((g, i) => [g, i]));
}

export function reach(proof) {
  if (proof?.status === "varies") return "varies";
  if (proof?.status !== "proven") return "warp 0 only";

  const warps = warpScope(proof, "every warp, every program", "all ");

  return proof.iterations ? `${warps}, every iteration` : warps;
}

function outcomeCost({ sectors, ways }) {
  if (sectors !== undefined) return ` (${sectors} sectors)`;
  if (ways > 1) return ` (${ways}-way)`;

  return "";
}

export function explainProof(proof) {
  if (proof?.status === "varies") {
    const parts = proof.outcomes.map(
      (o) =>
        `${VERDICT_TEXT[o.verdict]}${outcomeCost(o)} in ${o.cases} of ${proof.cases}`,
    );

    return `This warp is not the whole story. Across every warp and loop iteration the address lands on different alignments: ${parts.join(", ")}.`;
  }

  return proof ? `Only this warp was checked: ${proof.reason}.` : "";
}

export function explain(result, space) {
  if (!result.analyzed) return `Not analyzed: ${result.reason}.`;

  const bytes = result.distinct * result.elementBytes;

  switch (result.verdict) {
    case "broadcast":
      return "Every thread of the warp uses the same element, so it is fetched once and shared.";
    case "coalesced":
      return `The warp's 32 threads use ${result.distinct} element(s), ${bytes} B, in ${result.sectors} sector(s) of 32 B: the fewest possible, so every byte moved is used.`;

    case "strided": {
      const stride =
        result.lanes.length > 1
          ? result.lanes[1].offset - result.lanes[0].offset
          : 0;

      return (
        `Neighboring threads are ${stride} elements apart, so the warp moves ${result.sectors} sectors of 32 B ` +
        `(${result.sectors * 32} B) to use ${bytes} B: ${Math.round(result.efficiency * 100)}% of the traffic is useful. ` +
        "Making thread x walk the last (contiguous) index, or staging the data through shared memory, usually fixes this."
      );
    }

    case "misaligned": {
      const offBy = ((result.lanes[0].byte % 32) + 32) % 32;

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
