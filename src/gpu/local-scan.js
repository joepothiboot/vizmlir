const FUNC =
  /^\s*"?func\.func"?\s+(?:(?:private|public|nested)\s+)?@([\w$.-]+)/;

const DEF = /^\s*(%[\w$.-]+)(?::\d+)?\s*=\s*"?([\w$.-]+)"?/;
const OP = /^\s*"?([A-Za-z_][\w$.-]*)"?/;
export const SSA_NAME = /^%[\w$.-]+/;

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

export const COPY_OPS = new Set(["linalg.copy", "memref.copy"]);
const MAP_ALIAS = /^\s*(#[\w$.-]+)\s*=\s*affine_map<(.*)>\s*$/;

export function splitTop(s) {
  const parts = [];
  let depth = 0;
  let start = 0;

  for (let i = 0; i < s.length; i++) {
    const c = s[i];

    if ("<([{".includes(c)) {
      depth++;
    } else if (">)]}".includes(c) && s[i - 1] !== "-") {
      depth--;
    } else if (c === "," && depth === 0) {
      parts.push(s.slice(start, i).trim());
      start = i + 1;
    }
  }

  if (s.slice(start).trim()) parts.push(s.slice(start).trim());

  return parts;
}

export function typeColon(s) {
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

export function indexed(part) {
  const match = /^(%[\w$.-]+)(?:\[(.*)\])?$/.exec(part.trim());

  return match
    ? { name: match[1], indices: match[2] ? splitTop(match[2]) : [] }
    : null;
}

const BRACE_DEPTH = { "{": 1, "}": -1 };

export function scan(ir) {
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
        fn = {
          name: `@${head[1]}`,
          line: lineNo,
          ops: [],
          defs: new Map(),
          regions: [],
        };

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

    if (popped && rest.startsWith("{")) {
      popped.attrs = rest;
      continue;
    }

    if (!rest || /^[)\]:]/.test(rest) || rest.startsWith("->")) continue;

    const def = DEF.exec(rest);
    const op = def ? def[2] : (OP.exec(rest)?.[1] ?? "");

    const body = def
      ? rest.slice(def[0].length)
      : rest.slice(rest.indexOf(op) + op.length);

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
    for (const c of rest) net += BRACE_DEPTH[c] ?? 0;

    for (let i = 0; i < net; i++) {
      const region = { op, line: lineNo, end: null, attrs: "" };

      if (op === "scf.for") {
        const loop =
          /^(%[\w$.-]+)\s*=\s*(\S+)\s+to\s+(\S+)\s+step\s+(\S+)/.exec(
            entry.body,
          );

        if (loop) {
          [region.iv, region.lb, region.ub, region.step] = loop.slice(1);
        }
      }

      if (op === "scf.if") {
        region.cond = /^(%[\w$.-]+)/.exec(entry.body)?.[1] ?? null;
      }

      fn.regions.push(region);
      stack.push(region);
    }
  }

  if (fn) functions.push(fn);

  return { functions, maps };
}

function defAt(fn, name, at) {
  const defs = fn.defs.get(name);
  if (!defs) return null;
  if (!at) return defs[0];

  for (let i = defs.length - 1; i >= 0; i--) {
    const def = defs[i];

    if (
      def.line <= at.line &&
      def.regions.every((r) => at.regions.includes(r))
    ) {
      return def;
    }
  }

  return null;
}

export function constantOf(fn, value, at) {
  if (/^-?\d+$/.test(value)) return Number(value);

  const def = defAt(fn, value, at);
  if (def?.op !== "arith.constant") return null;

  const match = /^(-?\d+)(?:\s*:|$)/.exec(def.body);

  return match ? Number(match[1]) : null;
}

export function rootOf(fn, name, at) {
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

export function subviewParts(def) {
  const match = /^(%[\w$.-]+)\[([^\]]*)\]\s*\[([^\]]*)\]/.exec(def.body);

  return match
    ? {
        source: match[1],
        offsets: splitTop(match[2]),
        sizes: splitTop(match[3]),
      }
    : null;
}

export const loopOf = (regions) =>
  [...regions].reverse().find((r) => r.op === "scf.for") ?? null;

export function slotOf(fn, maps, value, at) {
  const regions = at.regions;
  const constant = constantOf(fn, value, at);

  if (constant !== null) {
    return { value: constant, of: null, expr: String(constant) };
  }

  const def = defAt(fn, value, at);
  let base = value;
  let expr = value;

  if (def?.op === "affine.apply") {
    const match = /^(#[\w$.-]+|affine_map<.*?>)\s*\((%[\w$.-]+)\)/.exec(
      def.body,
    );

    if (match) {
      base = match[2];

      const map = match[1].startsWith("#")
        ? maps.get(match[1])
        : match[1].slice(11, -1);

      expr = map ? map.replace(/^.*->\s*\((.*)\)\s*$/, "$1") : match[1];
    }
  }

  const loop = loopOf(regions);
  let of = null;

  if (loop && base === loop.iv) {
    of = "i";
  } else if (loop) {
    const add = defAt(fn, base, at);

    if (add?.op === "arith.addi") {
      const [a, b] = splitTop(add.body.split(" : ")[0]);
      let other = null;
      if (a === loop.iv) other = b;
      else if (b === loop.iv) other = a;

      if (
        other &&
        constantOf(fn, other, at) === constantOf(fn, loop.step, at)
      ) {
        of = "i+1";
      }
    }
  }

  return { value: null, of, expr };
}
