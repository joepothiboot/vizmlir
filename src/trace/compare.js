import { lineDiff } from "./linediff.js";
import { describeEvent } from "./trace.js";

const SSA_NAME = /%[-A-Za-z0-9_.$]+/g;

export function canonicalIr(text) {
  const names = new Map();

  return text.replace(SSA_NAME, (name) => {
    if (!names.has(name)) names.set(name, `%v${names.size}`);

    return names.get(name);
  });
}

function passKey(event) {
  return `${event.pass}|${event.argument ?? ""}`;
}

function kindOf(eventA, eventB) {
  if (!eventA || !eventB) return "length";
  if (passKey(eventA) !== passKey(eventB)) return "pass";
  if (canonicalIr(eventA.ir) !== canonicalIr(eventB.ir)) return "ir";

  return "same";
}

function finalIr(trace) {
  return trace.output || trace.events.at(-1)?.ir || "";
}

export function compareTraces(a, b) {
  const count = Math.max(a.events.length, b.events.length);

  const rows = Array.from({ length: count }, (_, index) => {
    const eventA = a.events[index];
    const eventB = b.events[index];

    return {
      index,
      kind: kindOf(eventA, eventB),
      labelA: eventA ? describeEvent(eventA) : "",
      labelB: eventB ? describeEvent(eventB) : "",
    };
  });

  return {
    rows,
    firstDivergence: rows.findIndex((row) => row.kind !== "same"),
    differing: rows.filter((row) => row.kind !== "same").length,
    finalSame: canonicalIr(finalIr(a)) === canonicalIr(finalIr(b)),
  };
}

export function rowDiff(a, b, index) {
  return lineDiff(
    canonicalIr(a.events[index]?.ir ?? ""),
    canonicalIr(b.events[index]?.ir ?? ""),
  );
}
