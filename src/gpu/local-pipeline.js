import { constantOf, loopOf, rootOf } from "./local-scan.js";

const SHOWN_ITERATIONS = 4;

function bufferingMode({ prefetch, slots, loads, prologue }) {
  if (prefetch.length && slots > 1) return "double";
  if (loads.length) return "single";
  if (prologue.length) return "resident";

  return "none";
}

export function pipelineOf(fn, buffers, transfers) {
  const inLoop = transfers.filter((t) => t.loopLine !== null);
  const loopLine = inLoop[0]?.loopLine ?? null;

  let loop =
    loopLine === null
      ? null
      : fn.regions.find((r) => r.op === "scf.for" && r.line === loopLine);

  if (!loop) {
    const names = new Set(buffers.map((b) => b.name));

    const reader = fn.ops.find(
      (op) =>
        loopOf(op.regions) &&
        (op.body.match(/%[\w$.-]+/g) ?? []).some((n) =>
          names.has(rootOf(fn, n, op).name),
        ),
    );

    loop = reader ? loopOf(reader.regions) : null;
  }

  if (!loop) return null;

  const loopOp = fn.ops.find((op) => op.line === loop.line);
  const lb = constantOf(fn, loop.lb ?? "", loopOp);
  const ub = constantOf(fn, loop.ub ?? "", loopOp);
  const step = constantOf(fn, loop.step ?? "", loopOp);

  const trips =
    lb !== null && ub !== null && step
      ? Math.max(0, Math.ceil((ub - lb) / step))
      : null;

  const slots = Math.max(1, ...buffers.map((b) => b.slots));
  const ofLoop = (t) => t.loopLine === loop.line;

  const prologue = transfers.filter(
    (t) => t.role === "prologue" && t.line < loop.line,
  );

  const prefetch = transfers.filter((t) => t.role === "prefetch" && ofLoop(t));
  const loads = transfers.filter((t) => t.role === "load" && ofLoop(t));
  const waits = transfers.filter((t) => t.role === "wait" && ofLoop(t));

  const lastTransfer = Math.max(
    loop.line,
    ...transfers.filter(ofLoop).map((t) => t.line),
  );

  const guards = new Set(
    fn.regions
      .filter(
        (r) => r.op === "scf.if" && r.line > loop.line && r.end <= loop.end,
      )
      .map((r) => r.line),
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

  const refilled = new Set([...prefetch, ...loads].map((t) => t.buffer));

  const shown =
    trips === null ? SHOWN_ITERATIONS : Math.min(trips, SHOWN_ITERATIONS);

  const iterations = [];

  for (let k = 0; k < shown; k++) {
    const events = [];

    if (k === 0) {
      for (const t of prologue) {
        events.push({
          lane: "dma",
          kind: "prologue",
          tile: 0,
          slot: slotAt(t.slot, 0),
          transfer: t,
        });
      }
    }

    for (const t of loads) {
      events.push({
        lane: "dma",
        kind: "load",
        tile: k,
        slot: slotAt(t.slot, k),
        transfer: t,
      });
    }

    const last = trips !== null && k + 1 >= trips;

    for (const t of prefetch) {
      if (!last) {
        events.push({
          lane: "dma",
          kind: "prefetch",
          tile: k + 1,
          slot: slotAt(t.slot, k),
          transfer: t,
        });
      }
    }

    for (const t of waits) {
      events.push({
        lane: "wait",
        kind: "wait",
        tile: k,
        slot: slotAt(t.slot, k),
        transfer: t,
      });
    }

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

    iterations.push({
      k,
      value: lb !== null && step ? lb + k * step : null,
      events,
    });
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
    mode: bufferingMode({ prefetch, slots, loads, prologue }),
    synchronous,
    computeLine: compute?.line ?? null,
    steady,
    iterations,
    truncated: trips === null || trips > shown,
  };
}
