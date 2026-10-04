// Finds the buffers a function allocates (memref.alloc, memref.alloca,
// gpu.alloc), their static size, and how long each one stays live, so two
// versions of the IR can be compared by memory rather than by text.
//
// Positions count the non-blank lines of a function body (closing-brace lines
// excluded), so they follow source order. A buffer is live from its allocation
// to the first of:
//
// - its `memref.dealloc` / `bufferization.dealloc` (freed "dealloc"),
// - the end of the function, when it or an alias is returned ("returned"),
// - its last use, otherwise ("last-use", or "scope" for memref.alloca).
//
// A use inside a region nested below the allocation (a loop body) keeps the
// buffer live until that region's op ends, since the body may run again.
// Aliases (casts, subviews, reshapes) extend the lifetime of what they view.

const ALLOC_OPS = new Set(["memref.alloc", "memref.alloca", "gpu.alloc"]);
const DEALLOC_OPS = new Set([
  "memref.dealloc",
  "gpu.dealloc",
  "bufferization.dealloc",
]);
const ALIAS_OPS = new Set([
  "memref.cast",
  "memref.subview",
  "memref.view",
  "memref.reinterpret_cast",
  "memref.collapse_shape",
  "memref.expand_shape",
  "memref.transpose",
  "memref.memory_space_cast",
  "memref.assume_alignment",
  "bufferization.to_tensor",
  "bufferization.to_buffer",
  "bufferization.to_memref",
]);
const RETURN_OPS = /^(?:[\w]+\.)?return$/;
const FUNC = /^\s*"?[\w.]*func"?\s+(?:(?:private|public|nested)\s+)?@([\w$.-]+|"[^"]*")/;
const DEF = /^\s*((?:%[\w$.-]+(?::\d+)?\s*,\s*)*%[\w$.-]+(?::\d+)?)\s*=\s*"?([\w$.-]+)"?/;
const OP = /^\s*"?([A-Za-z_][\w$.-]*)"?/;
const SSA = /%[\w$.-]+/g;
const GLOBAL = /^\s*memref\.global\b.*?@([\w$.-]+)\s*:\s*(memref<.*)$/;

// ---- Types -----------------------------------------------------------------

// Splits `s` on commas that are not nested inside <>, (), [] or {}.
function splitTop(s) {
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if ("<([{".includes(c)) depth++;
    else if (">)]}".includes(c)) depth--;
    else if (c === "," && depth === 0) {
      parts.push(s.slice(start, i).trim());
      start = i + 1;
    }
  }
  parts.push(s.slice(start).trim());
  return parts;
}

// Returns the text of the balanced `<...>` starting at `open`, or null.
function angleBody(s, open) {
  let depth = 0;
  for (let i = open; i < s.length; i++) {
    if (s[i] === "<") depth++;
    else if (s[i] === ">" && s[i - 1] !== "-" && --depth === 0)
      return s.slice(open + 1, i);
  }
  return null;
}

/** Bytes of one element of `type`, or null if unknown. */
export function elementBytes(type) {
  const t = type.trim();
  if (t === "index") return 8;
  if (/^f8/.test(t)) return 1;
  if (t === "f16" || t === "bf16") return 2;
  if (t === "f32" || t === "tf32") return 4;
  if (t === "f64") return 8;
  if (t === "f80" || t === "f128") return 16;
  const int = /^[su]?i(\d+)$/.exec(t);
  if (int) return Math.ceil(Number(int[1]) / 8);
  if (t.startsWith("complex<")) {
    const inner = elementBytes(t.slice(8, -1));
    return inner === null ? null : inner * 2;
  }
  if (t.startsWith("vector<")) {
    const shape = parseShape(t.slice(7, -1));
    return shape.elements !== null && shape.bytes !== null
      ? shape.elements * shape.bytes
      : null;
  }
  return null;
}

// `4x?x8xf32` → dims [4, null, 8], element `f32`.
function parseShape(body) {
  if (body.startsWith("*")) return { dims: null, element: body.slice(2), elements: null, bytes: elementBytes(body.slice(2)) };
  const dims = [];
  let rest = body;
  for (;;) {
    const m = /^(\d+|\?|\[\d+\])x/.exec(rest);
    if (!m) break;
    dims.push(m[1] === "?" ? null : Number(m[1].replace(/[[\]]/g, "")));
    rest = rest.slice(m[0].length);
  }
  const elements = dims.every((d) => d !== null)
    ? dims.reduce((a, b) => a * b, 1)
    : null;
  return { dims, element: rest, elements, bytes: elementBytes(rest) };
}

/**
 * Parses `memref<128x?xf32, strided<..>, 3>` into its shape, element type,
 * static size in bytes (null when a dimension is dynamic or the element type
 * is unknown) and memory space ("" for the default space).
 */
export function parseMemref(type) {
  const open = type.indexOf("<");
  if (!type.startsWith("memref") || open < 0) return null;
  const body = angleBody(type, open);
  if (body === null) return null;
  const [shapeText, ...rest] = splitTop(body);
  const shape = parseShape(shapeText);
  let space = "";
  for (const part of rest) {
    if (/^\d+(\s*:\s*i\d+)?$/.test(part)) space = part.split(":")[0].trim();
    else if (/^#gpu\.address_space<(\w+)>$/.test(part))
      space = /<(\w+)>/.exec(part)[1];
    else if (part.startsWith("#") && !/^#map\d*$/.test(part)) space = part;
  }
  const bytes =
    shape.elements !== null && shape.bytes !== null
      ? shape.elements * shape.bytes
      : null;
  return {
    type: `memref<${body}>`,
    dims: shape.dims,
    element: shape.element,
    bytes,
    space,
  };
}

// The last `memref<...>` in a line, which for an allocation is its result.
function lastMemref(line) {
  const at = line.lastIndexOf("memref<");
  if (at < 0) return null;
  const body = angleBody(line, at + 6);
  return body === null ? null : `memref<${body}>`;
}

// ---- Analysis --------------------------------------------------------------

// Blanks string literals that could hide braces or `%` from the scan, keeping
// plain names such as a generic-form op (`"memref.alloc"() ..`).
function stripStrings(line) {
  return line.replace(/"(?:[^"\\]|\\.)*"/g, (s) =>
    /^"[\w$.-]*"$/.test(s) ? s : '""',
  );
}

/**
 * Analyzes one IR text. Returns `{ functions, globals }`: each function has
 * `{ name, line, length, buffers, peak, peakAt, allocated, dynamic, live }`
 * where `live[p]` is the static bytes live at position p, and each buffer has
 * `{ name, op, type, bytes, space, line, start, end, freed, aliases }`.
 * Lines are 0-based line numbers in `ir`.
 */
export function analyzeBuffers(ir) {
  const functions = [];
  const globals = [];
  const lines = ir.split("\n");

  let fn = null;
  // Open regions inside the current function; each records the buffers used
  // in it so their lifetime can be stretched to its closing brace.
  let stack = [];
  // SSA name → buffer, for buffers and their aliases in the current function.
  let owner = new Map();

  const close = (pos) => {
    // `pos` is the next op's position; the region ends with the op before it.
    const region = stack.pop();
    for (const buffer of region.used)
      buffer.end = Math.max(buffer.end, pos - 1);
    if (stack.length === 0) {
      finish(fn);
      functions.push(fn);
      fn = null;
      owner = new Map();
    }
  };

  const use = (buffer, pos) => {
    buffer.end = Math.max(buffer.end, pos);
    buffer.uses++;
    // The outermost region opened after the allocation.
    const region = stack[buffer.depth];
    if (region) region.used.add(buffer);
  };

  for (let lineNo = 0; lineNo < lines.length; lineNo++) {
    const raw = lines[lineNo];
    const text = stripStrings(raw);
    const trimmed = text.trim();
    if (!trimmed || trimmed.startsWith("//")) continue;

    if (!fn) {
      const global = GLOBAL.exec(raw);
      if (global) {
        const type = parseMemref(global[2]);
        if (type)
          globals.push({
            name: `@${global[1]}`,
            type: type.type,
            bytes: type.bytes,
            space: type.space,
            constant: /\bconstant\b/.test(raw),
            line: lineNo,
          });
        continue;
      }
      const head = FUNC.exec(text);
      if (!head || !trimmed.endsWith("{")) continue;
      fn = {
        name: `@${head[1].replace(/"/g, "")}`,
        line: lineNo,
        length: 0,
        buffers: [],
      };
      stack = [{ used: new Set() }];
      continue;
    }

    // Leading `}` close regions before this line's own op.
    let rest = trimmed;
    while (rest.startsWith("}") && fn) {
      close(fn.length);
      rest = rest.slice(1).trimStart();
    }
    if (!fn) continue;
    if (!rest || /^[})\]]/.test(rest) || rest.startsWith("->") || rest.startsWith(":")) {
      // Trailing types of a region op (`} -> ..`, `}) : ..`) or `)`: no op.
      for (const c of rest) if (c === "{") stack.push({ used: new Set() });
      continue;
    }

    const pos = fn.length++;
    const def = DEF.exec(rest);
    const op = def ? def[2] : OP.exec(rest)?.[1] ?? "";
    const results = def ? def[1].match(SSA) : [];
    const operandText = def ? rest.slice(def[0].length) : rest.slice(op.length);
    const operands = (operandText.match(SSA) ?? []).map((name) =>
      name.replace(/#\d+$/, ""),
    );

    if (ALLOC_OPS.has(op) && results.length) {
      const type = parseMemref(lastMemref(rest) ?? "");
      const buffer = {
        name: results[0],
        op,
        type: type?.type ?? "memref<?>",
        bytes: type?.bytes ?? null,
        space: type?.space ?? "",
        line: lineNo,
        start: pos,
        end: pos,
        depth: stack.length,
        freed: op === "memref.alloca" ? "scope" : "last-use",
        aliases: [],
        uses: 0,
      };
      fn.buffers.push(buffer);
      owner.set(buffer.name, buffer);
    } else {
      const touched = new Set();
      for (const name of operands) {
        const buffer = owner.get(name);
        if (buffer && !touched.has(buffer)) {
          touched.add(buffer);
          use(buffer, pos);
        }
      }
      if (ALIAS_OPS.has(op) && results.length) {
        const source = owner.get(operands[0]);
        if (source) {
          owner.set(results[0], source);
          source.aliases.push(results[0]);
        }
      }
      if (DEALLOC_OPS.has(op))
        for (const buffer of touched) buffer.freed = "dealloc";
      if (RETURN_OPS.test(op))
        for (const buffer of touched) buffer.freed = "returned";
    }

    for (const c of rest) {
      if (c === "{") stack.push({ used: new Set() });
      else if (c === "}" && stack.length > 1) stack.pop();
    }
  }
  if (fn) {
    finish(fn);
    functions.push(fn);
  }
  return { functions, globals };
}

function finish(fn) {
  fn.length = Math.max(fn.length, 1);
  for (const buffer of fn.buffers) {
    if (buffer.freed === "returned") buffer.end = fn.length - 1;
    delete buffer.depth;
  }
  fn.live = new Array(fn.length).fill(0);
  fn.allocated = 0;
  fn.dynamic = 0;
  for (const buffer of fn.buffers) {
    if (buffer.bytes === null) {
      fn.dynamic++;
      continue;
    }
    fn.allocated += buffer.bytes;
    const last = Math.min(buffer.end, fn.length - 1);
    for (let p = buffer.start; p <= last; p++) fn.live[p] += buffer.bytes;
  }
  fn.peak = 0;
  fn.peakAt = 0;
  fn.live.forEach((bytes, p) => {
    if (bytes > fn.peak) {
      fn.peak = bytes;
      fn.peakAt = p;
    }
  });
}

/** Totals across functions. Peak is the largest single function's peak. */
export function bufferTotals(analysis) {
  let buffers = 0;
  let allocated = 0;
  let dynamic = 0;
  let peak = 0;
  for (const fn of analysis.functions) {
    buffers += fn.buffers.length;
    allocated += fn.allocated;
    dynamic += fn.dynamic;
    peak = Math.max(peak, fn.peak);
  }
  return { buffers, allocated, dynamic, peak };
}

// ---- Comparison ------------------------------------------------------------

const bufferKey = (buffer) => `${buffer.op} ${buffer.type}`;

/**
 * Pairs the buffers of two analyses. Functions match by name; buffers within
 * a function match by allocating op and type, in source order, since SSA
 * names are renumbered by most passes. Each row is
 * `{ status, before, after }` with status "added", "removed", "changed"
 * (same buffer, released differently or in another memory space) or "same".
 */
export function compareBuffers(before, after) {
  const names = [];
  for (const fn of [...(before?.functions ?? []), ...after.functions])
    if (!names.includes(fn.name)) names.push(fn.name);

  return names.map((name) => {
    const a = before?.functions.find((fn) => fn.name === name) ?? null;
    const b = after.functions.find((fn) => fn.name === name) ?? null;
    const pending = new Map();
    for (const buffer of a?.buffers ?? []) {
      const key = bufferKey(buffer);
      if (!pending.has(key)) pending.set(key, []);
      pending.get(key).push(buffer);
    }
    const rows = [];
    for (const buffer of b?.buffers ?? []) {
      const match = pending.get(bufferKey(buffer))?.shift() ?? null;
      if (!before) rows.push({ status: "same", before: null, after: buffer });
      else if (!match) rows.push({ status: "added", before: null, after: buffer });
      else
        rows.push({
          status:
            match.freed !== buffer.freed || match.space !== buffer.space
              ? "changed"
              : "same",
          before: match,
          after: buffer,
        });
    }
    for (const left of pending.values())
      for (const buffer of left)
        rows.push({ status: "removed", before: buffer, after: null });
    return { name, before: a, after: b, rows };
  });
}

export function buffersToJSON(title, comparison, passes) {
  const buffer = (b) =>
    b && {
      name: b.name,
      op: b.op,
      type: b.type,
      bytes: b.bytes,
      space: b.space,
      line: b.line + 1,
      start: b.start,
      end: b.end,
      freed: b.freed,
      aliases: b.aliases,
    };
  const fn = (f) =>
    f && {
      length: f.length,
      allocated: f.allocated,
      peak: f.peak,
      peakAt: f.peakAt,
      dynamic: f.dynamic,
    };
  return JSON.stringify(
    {
      title,
      functions: comparison.map((entry) => ({
        name: entry.name,
        before: fn(entry.before),
        after: fn(entry.after),
        buffers: entry.rows.map((row) => ({
          status: row.status,
          before: buffer(row.before),
          after: buffer(row.after),
        })),
      })),
      ...(passes ? { passes } : {}),
    },
    null,
    2,
  );
}
