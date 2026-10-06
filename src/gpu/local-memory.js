import { analyzeBuffers, elementBytes, parseMemref } from "../trace/index.js";
import { pipelineOf } from "./local-pipeline.js";
import {
  COPY_OPS,
  SSA_NAME,
  constantOf,
  indexed,
  loopOf,
  rootOf,
  scan,
  slotOf,
  splitTop,
  subviewParts,
  typeColon,
} from "./local-scan.js";
import { memorySpace } from "./model.js";

export const TARGET_LOCAL_BYTES = {
  "hexagon-hvx128": 256 * 1024,
  "host-neon": 0,
  "x86-avx2": 0,
};

const DEFAULT_TARGET = "hexagon-hvx128";
const IR_BUDGET = /\b(?:dsp\.)?(?:local_mem_bytes|localMemBytes)\s*=\s*(\d+)/;

export function localTargetOf(events, index = events.length - 1) {
  for (let i = Math.min(index, events.length - 1); i >= 0; i--) {
    const match = /(?:^|[\s{])(?:local-)?target=([\w.-]+)/.exec(
      events[i].options ?? "",
    );

    if (match) return match[1];
  }

  return null;
}

function budgetFor(ir, target) {
  const carried = IR_BUDGET.exec(ir);

  if (carried) {
    return {
      bytes: Number(carried[1]),
      source: "ir",
      target,
      note: "from the IR",
    };
  }

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

export function analyzeLocalMemory(ir, { target = null } = {}) {
  const { functions: scanned, maps } = scan(ir);
  const buffers = analyzeBuffers(ir);
  const functions = [];

  for (const fn of scanned) {
    const result = analyzeFunction(
      fn,
      maps,
      buffers.functions.find((f) => f.name === fn.name),
    );

    if (result) functions.push(result);
  }

  let stage = "none";

  if (functions.length) {
    stage = functions.some((f) => f.stage === "dma") ? "dma" : "copies";
  }

  const budget = stage === "none" ? null : budgetFor(ir, target);
  for (const f of functions) f.budget = budget;

  return { stage, target: budget?.target ?? target, budget, functions };
}

export const hasLocalMemory = (analysis) =>
  !!analysis && analysis.stage !== "none";

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

  for (const buffer of allocs.values()) {
    if (memorySpace(buffer.space) === "local") addLocal(buffer.name, false);
  }

  const target = (name, at) => {
    const { name: root, chain } = rootOf(fn, name, at);
    const first = chain[0] && subviewParts(chain[0]);
    let slot = null;

    if (
      first &&
      first.source === root &&
      first.sizes[0] === "1" &&
      first.offsets.length > 1
    ) {
      slot = slotOf(fn, maps, first.offsets[0], chain[0]);
    }

    return { root, slot };
  };

  const roleOf = (op, slot) => {
    const loop = loopOf(op.regions);
    if (!loop) return "prologue";

    const guarded = op.regions
      .slice(op.regions.indexOf(loop) + 1)
      .some((r) => r.op === "scf.if");

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

      if (buffer && !buffer.source) {
        buffer.source = rootOf(fn, src.name, op).name;
      }

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
        bytes:
          elements !== null && elemBytes !== null ? elements * elemBytes : null,
        strided:
          parts.length > 5
            ? {
                stride: constantOf(fn, parts[4], op),
                perStride: constantOf(fn, parts[5], op),
              }
            : null,
        loopLine: loopOf(op.regions)?.line ?? null,
      });
    } else if (op.op === "memref.dma_wait") {
      const parts = splitTop(
        op.body.slice(0, Math.max(0, typeColon(op.body))) || op.body,
      );

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
        buffer: null,
        slot: into.slot,
        tag: into.root,
        elements: constantOf(fn, parts[1] ?? "", op),
        bytes: null,
        strided: null,
        loopLine: loopOf(op.regions)?.line ?? null,
      });
    }
  }

  const readInLoop = (root) =>
    fn.ops.some(
      (op) =>
        loopOf(op.regions) &&
        !COPY_OPS.has(op.op) &&
        (op.body.match(/%[\w$.-]+/g) ?? []).some(
          (n) => rootOf(fn, n, op).name === root,
        ),
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

    const srcType = parseMemref(
      /ins\([^:]*:\s*(memref<.*?>)\)/.exec(op.body)?.[1] ?? "",
    );

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
      bytes:
        elements !== null && srcType?.bytes !== null ? srcType.bytes : null,
      strided: null,
      loopLine: loopOf(op.regions)?.line ?? null,
    });
  }

  if (!local.size && !transfers.length) return null;

  for (const buffer of local.values()) {
    const views = fn.ops.filter(
      (op) =>
        op.op === "memref.subview" &&
        SSA_NAME.exec(op.body)?.[0] === buffer.name,
    );

    if (
      views.length &&
      buffer.shape?.length > 1 &&
      buffer.shape[0] > 1 &&
      views.every((v) => subviewParts(v)?.sizes[0] === "1")
    ) {
      buffer.slots = buffer.shape[0];

      buffer.slotBytes =
        buffer.bytes === null ? null : buffer.bytes / buffer.slots;
    }
  }

  const ordered = [...local.values()].sort((a, b) => a.line - b.line);

  ordered.forEach((b, i) => {
    b.label = String.fromCharCode(65 + (i % 26));
  });

  for (const wait of transfers.filter((t) => t.role === "wait")) {
    const start = transfers.find(
      (t) => t.op === "memref.dma_start" && t.tag === wait.tag,
    );

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
      slots:
        memref?.dims?.length > 1 && memref.dims[0] > 1 ? memref.dims[0] : 1,
    };
  });

  let peakLocalBytes = 0;

  if (bufferFn) {
    for (let p = 0; p < bufferFn.length; p++) {
      let live = 0;

      for (const b of local.values()) {
        if (b.bytes !== null && b.start <= p && p <= b.end) live += b.bytes;
      }

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
    stage:
      transfers.some((t) => t.op.startsWith("memref.dma")) ||
      ordered.some((b) => !b.lowered)
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

export function localLineInfo(analysis, line) {
  for (const fn of analysis?.functions ?? []) {
    const buffer = fn.localBuffers.find((b) => b.line === line);
    if (buffer) return { kind: "buffer", fn, item: buffer };

    const tag = fn.tags.find((t) => t.line === line);
    if (tag) return { kind: "tag", fn, item: tag };

    const transfer = fn.transfers.find((t) => t.line === line);
    if (transfer) return { kind: "transfer", fn, item: transfer };

    if (fn.pipeline && fn.pipeline.computeLine === line) {
      return { kind: "compute", fn, item: fn.pipeline };
    }
  }

  return null;
}
