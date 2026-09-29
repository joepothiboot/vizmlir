// Reads how GPU kernels are launched and where their data lives, from IR text
// only: the grid and block sizes of each gpu.launch / gpu.launch_func, and each
// kernel's buffers by memory space (global, shared per block, private per
// thread). After gpu-module-to-binary{format=isa}, the PTX register and shared
// memory declarations are read too. Nothing here runs the kernel, so launch
// sizes that are only known at runtime stay null. Triton GPU IR is read by
// src/triton.js into the same model.

import { parseMemref } from "./buffers.js";
import { affineAliases, buildDefs, findAccesses } from "./gpu-access.js";
import { embeddedAssembly, scanSymbols } from "./provenance.js";
import { analyzeTriton, isTriton } from "./triton.js";

const LAUNCH_FUNC =
  /\bgpu\.launch_func\b.*?@([\w$.-]+|"[^"]*")::@([\w$.-]+|"[^"]*")/;
const LAUNCH = /\bgpu\.launch\b(?!_)/;
const HOST_FUNC = /^(\s*)(?:"?[\w.]+\.)?func"?\s+(?:\w+\s+)*@([\w$.-]+|"[^"]*")/;
const ACCESS =
  /\b(memref\.load|memref\.store|vector\.load|vector\.store|vector\.transfer_read|vector\.transfer_write|memref\.atomic_rmw)\b.*?(%[\w$.-]+)\[/;

const unquote = (name) => name.replace(/^"(.*)"$/, "$1");
const indentOf = (line) => /^\s*/.exec(line)[0].length;

// "" (no space on a kernel argument), 1 and `global` are global memory; 3 and
// `workgroup` are shared by a block; 5 and `private` belong to one thread.
export function memorySpace(space, fallback = "global") {
  if (space === "" || space === undefined) return fallback;
  if (space === "1" || space === "global") return "global";
  if (space === "3" || space === "workgroup") return "shared";
  if (space === "5" || space === "private") return "private";
  return `space ${space}`;
}

// Splits on commas that are not nested inside <>, (), [] or {}.
function splitTop(s) {
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if ("<([{".includes(c)) depth++;
    else if (">)]}".includes(c) && s[i - 1] !== "-") depth--;
    else if (c === "," && depth === 0) {
      parts.push(s.slice(start, i).trim());
      start = i + 1;
    }
  }
  if (s.slice(start).trim()) parts.push(s.slice(start).trim());
  return parts;
}

// The text inside the parentheses that open at `open`, or null.
function parenBody(s, open) {
  let depth = 0;
  for (let i = open; i < s.length; i++) {
    if (s[i] === "(") depth++;
    else if (s[i] === ")" && --depth === 0) return s.slice(open + 1, i);
  }
  return null;
}

// `%a: memref<4xf32>` / `%a : memref<4xf32>` → { name, type }.
function binding(part) {
  const match = /^(%[\w$.-]+)\s*:\s*(.+)$/.exec(part);
  return match ? { name: match[1], type: match[2].trim() } : null;
}

// Joins a launch op printed over several lines, up to its `{`.
function joinedHeader(lines, from) {
  let text = lines[from];
  for (let i = from + 1; i < lines.length && !/\{\s*$/.test(text); i++)
    text += ` ${lines[i].trim()}`;
  return text;
}

// The region of the op that starts at `from`: the lines after its header
// (which may span lines, up to the `{`) that are indented deeper than the op,
// blank lines included so positions stay aligned. `start` is the index of the
// first one.
function regionLines(lines, from) {
  const indent = indentOf(lines[from]);
  let i = from;
  while (i < lines.length && !/\{\s*$/.test(lines[i])) i++;
  const start = i + 1;
  let end = start;
  for (let k = start; k < lines.length; k++) {
    if (lines[k].trim() === "") continue;
    if (indentOf(lines[k]) <= indent) break;
    end = k + 1;
  }
  return { start, lines: lines.slice(start, end) };
}

// Resolves `%c256` to 256 from the nearest `%c256 = arith.constant 256`
// above line `at`, or null (a runtime value).
function constantBefore(lines, at, value) {
  const name = value.trim();
  if (/^-?\d+$/.test(name)) return Number(name);
  const escaped = name.replace(/[$.]/g, "\\$&");
  const def = new RegExp(`^\\s*${escaped}\\s*=\\s*(?:arith|llvm\\.mlir)\\.constant\\s*\\(?\\s*(-?\\d+)\\b`);
  for (let i = at; i >= 0; i--) {
    const match = def.exec(lines[i]);
    if (match) return Number(match[1]);
  }
  return null;
}

// `in (%c4, %c1, %c1)` or `in (%gx = %c4, ...)` → [4, 1, 1].
function launchDims(lines, at, header, keyword) {
  const match = new RegExp(`\\b${keyword}\\b(?:\\([^)]*\\))?\\s+in\\s+\\(`).exec(header);
  if (!match) return null;
  const body = parenBody(header, match.index + match[0].length - 1);
  if (body === null) return null;
  const dims = splitTop(body).map((part) =>
    constantBefore(lines, at, part.includes("=") ? part.split("=").at(-1) : part),
  );
  return dims.length === 3 ? dims : null;
}

// The function around line `at`: { name, index } of its header, or null.
function hostOf(lines, at) {
  const indent = indentOf(lines[at]);
  for (let i = at - 1; i >= 0; i--) {
    const match = HOST_FUNC.exec(lines[i]);
    if (match && match[1].length < indent)
      return { name: `@${unquote(match[2])}`, index: i };
  }
  return null;
}

// Buffers named in `workgroup(...)` / `private(...)` attributions.
function attributions(header) {
  const buffers = [];
  for (const match of header.matchAll(/\b(workgroup|private)\s*\(/g)) {
    const body = parenBody(header, match.index + match[0].length - 1);
    for (const part of splitTop(body ?? "")) {
      const bound = binding(part);
      if (!bound) continue;
      buffers.push({
        ...bound,
        source: match[1],
        fallback: match[1] === "workgroup" ? "shared" : "private",
      });
    }
  }
  return buffers;
}

// Buffers a kernel body allocates, and the memrefs its loads and stores use,
// with the type printed after the access.
function bodyBuffers(body) {
  const allocs = [];
  const used = new Map();
  for (const line of body) {
    const alloc = /^\s*(%[\w$.-]+)\s*=\s*(memref\.alloca?|gpu\.alloc)\b.*:\s*(memref<.*>)\s*$/.exec(line);
    if (alloc)
      allocs.push({
        name: alloc[1],
        type: alloc[3],
        source: alloc[2].replace(/^\w+\./, ""),
        fallback: alloc[2] === "memref.alloca" ? "private" : "global",
      });
    const access = ACCESS.exec(line);
    if (!access) continue;
    const type = /:\s*(memref<.*?>)(?:\s*,|\s*$)/.exec(line)?.[1] ?? null;
    const entry = used.get(access[2]) ?? { name: access[2], type, loads: 0, stores: 0 };
    entry.type ??= type;
    if (/store|write/.test(access[1])) entry.stores += 1;
    else if (/rmw/.test(access[1])) {
      entry.loads += 1;
      entry.stores += 1;
    } else entry.loads += 1;
    used.set(access[2], entry);
  }
  return { allocs, used };
}

// Merges declared buffers (arguments, attributions, allocations) with the
// accesses in the body into [{ name, type, space, bytes, source, loads,
// stores }]. Memrefs that are only accessed (a gpu.launch region using a
// value from the host) come in as source "captured".
function kernelBuffers(declared, body) {
  const { allocs, used } = bodyBuffers(body);
  const buffers = [];
  const seen = new Set();
  for (const buffer of [...declared, ...allocs]) {
    const memref = parseMemref(buffer.type);
    if (!memref) continue;
    seen.add(buffer.name);
    const access = used.get(buffer.name);
    buffers.push({
      name: buffer.name,
      type: memref.type,
      space: memorySpace(memref.space, buffer.fallback),
      bytes: memref.bytes,
      source: buffer.source,
      loads: access?.loads ?? 0,
      stores: access?.stores ?? 0,
    });
  }
  for (const access of used.values()) {
    if (seen.has(access.name) || !access.type) continue;
    const memref = parseMemref(access.type);
    if (!memref) continue;
    buffers.push({
      name: access.name,
      type: memref.type,
      space: memorySpace(memref.space),
      bytes: memref.bytes,
      source: "captured",
      loads: access.loads,
      stores: access.stores,
    });
  }
  return buffers;
}

// Register and shared-memory declarations of one `.entry` in PTX text.
export function ptxEntries(ptx) {
  const entries = [];
  const starts = [...ptx.matchAll(/\.entry\s+([\w$]+)\s*\(/g)];
  starts.forEach((start, i) => {
    const text = ptx.slice(start.index, starts[i + 1]?.index ?? ptx.length);
    const registers = [];
    for (const reg of text.matchAll(/\.reg\s+\.(\w+)\s+%\w+<(\d+)>/g))
      registers.push({ type: reg[1], count: Number(reg[2]) });
    let sharedBytes = 0;
    for (const shared of text.matchAll(/\.shared\s+(?:\.align\s+\d+\s+)?\.(\w+)\s+[\w$]+\[(\d+)\]/g)) {
      const bits = Number(/\d+/.exec(shared[1])?.[0] ?? 8);
      sharedBytes += (bits / 8) * Number(shared[2]);
    }
    entries.push({ name: start[1], registers, sharedBytes });
  });
  return entries;
}

// Returns { launches, kernels } for `ir`, or null when it has no GPU code.
//   launches: [{ line, host, kernel, grid, block, threads }] where `kernel`
//     indexes `kernels`, grid/block are [x, y, z] with null for runtime
//     values, and `threads` is their product when all are known.
//   kernels: [{ path, name, op, line, inline, lowered, buffers, ptx }] where
//     `ptx` is { registers, sharedBytes } from an embedded PTX entry.
// The values a launch passes, one per kernel parameter. When the kernel was
// lowered to LLVM and takes more parameters than the launch has operands, each
// memref operand is expanded the way MLIR lowers it: allocated and aligned
// pointers (unknown), then the offset, sizes and strides, known for a static
// row-major memref. `valueOf(part)` gives any other operand's constant value.
export function loweredArgs(operands, params, valueOf) {
  if (operands.length === params) return operands.map(valueOf);
  const values = [];
  for (const part of operands) {
    const type = part.slice(part.indexOf(":") + 1).trim();
    const memref = type.startsWith("memref") ? parseMemref(type) : null;
    if (!memref) {
      values.push(valueOf(part));
      continue;
    }
    const dims = memref.dims;
    const strides = dims.map((_, k) =>
      dims.slice(k + 1).reduce((a, d) => (a === null || d === null ? null : a * d), 1),
    );
    values.push(null, null, 0, ...dims, ...strides);
  }
  return values;
}

export function analyzeGpu(ir) {
  if (isTriton(ir)) return analyzeTriton(ir);
  if (!/\bgpu\./.test(ir)) return null;
  const lines = ir.split("\n");
  const maps = affineAliases(ir);
  const kernels = [];
  const launches = [];
  const byPath = new Map();

  // Outlined kernels: gpu.func ... kernel, or an llvm.func marked as one.
  for (const symbol of scanSymbols(ir).values()) {
    if (symbol.op !== "gpu.func" && symbol.op !== "llvm.func") continue;
    const header = joinedHeader(lines, symbol.line - 1);
    const open = header.indexOf("(");
    const args = open < 0 ? null : parenBody(header, open);
    // What follows the argument list: attributions, `kernel`, attributes.
    const rest = args === null ? "" : header.slice(open + args.length + 2);
    const isGpuKernel = symbol.op === "gpu.func" && /\bkernel\b/.test(rest);
    const isLlvmKernel =
      symbol.op === "llvm.func" && /\b(?:gpu|nvvm|rocdl)\.kernel\b/.test(rest);
    if (!isGpuKernel && !isLlvmKernel) continue;
    const declared = [];
    for (const part of splitTop(args)) {
      const bound = binding(part);
      if (bound) declared.push({ ...bound, source: "argument", fallback: "global" });
    }
    const params = declared.map((d) => d.name);
    declared.push(...attributions(rest));
    const full = symbol.full.split("\n");
    const body = full.slice(1);
    // Lowered (llvm.func) kernels are read too: their LLVM and NVVM ops are
    // normalized by buildDefs, and their loads and stores go through
    // getelementptr (findAccesses).
    const kernel = {
      path: symbol.path,
      name: symbol.symbol,
      op: symbol.op,
      line: symbol.line,
      inline: false,
      lowered: symbol.op !== "gpu.func",
      buffers: isGpuKernel ? kernelBuffers(declared, body) : [],
      ptx: null,
      params,
      args: new Map(),
      defs: buildDefs(full, symbol.line, maps),
      accesses: findAccesses(full, symbol.line),
    };
    byPath.set(symbol.path, kernels.length);
    kernels.push(kernel);
  }

  // PTX embedded by gpu-module-to-binary{format=isa}.
  for (const symbol of scanSymbols(ir).values()) {
    if (symbol.op !== "gpu.binary") continue;
    for (const object of embeddedAssembly(symbol.full))
      for (const entry of ptxEntries(object.text)) {
        const path = `${symbol.path}::@${entry.name}`;
        const ptx = { registers: entry.registers, sharedBytes: entry.sharedBytes };
        if (byPath.has(path)) {
          kernels[byPath.get(path)].ptx = ptx;
          continue;
        }
        byPath.set(path, kernels.length);
        kernels.push({
          path,
          name: entry.name,
          op: "gpu.binary",
          line: symbol.line,
          inline: false,
          lowered: true,
          buffers: [],
          ptx,
        });
      }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const outlined = LAUNCH_FUNC.exec(line);
    if (!outlined && !LAUNCH.test(line)) continue;
    const header = joinedHeader(lines, i);
    const grid = launchDims(lines, i, header, "blocks");
    const block = launchDims(lines, i, header, "threads");
    const host = hostOf(lines, i);
    let kernel = -1;
    if (outlined) {
      const path = `@${unquote(outlined[1])}::@${unquote(outlined[2])}`;
      kernel = byPath.get(path) ?? -1;
      // Constant launch arguments resolve the kernel's parameters, as the
      // `%c16` that outlining passes in for a block size.
      const entry = kernels[kernel];
      if (entry?.params && !entry.args.size) {
        const match = /\bargs\s*\(/.exec(header);
        const operands = match
          ? splitTop(parenBody(header, match.index + match[0].length - 1) ?? "")
          : [];
        loweredArgs(operands, entry.params.length, (part) =>
          constantBefore(lines, i, part.split(":")[0].trim()),
        ).forEach((value, k) => {
          if (entry.params[k]) entry.args.set(entry.params[k], value);
        });
      }
      if (kernel < 0) {
        kernel = kernels.length;
        byPath.set(path, kernel);
        kernels.push({
          path,
          name: unquote(outlined[2]),
          op: null,
          line: null,
          inline: false,
          lowered: false,
          buffers: [],
          ptx: null,
        });
      }
    } else {
      const region = regionLines(lines, i);
      // Values the region uses may be defined earlier in the host function.
      const scope = lines.slice(host?.index ?? i, region.start + region.lines.length);
      kernel = kernels.length;
      kernels.push({
        path: null,
        name: `gpu.launch in ${host?.name ?? "?"}`,
        op: "gpu.launch",
        line: i + 1,
        inline: true,
        lowered: false,
        buffers: kernelBuffers(attributions(header), region.lines),
        ptx: null,
        params: [],
        args: new Map(),
        defs: buildDefs(scope, (host?.index ?? i) + 1, maps),
        accesses: findAccesses(region.lines, region.start + 1),
      });
    }
    const dims = [...(grid ?? []), ...(block ?? [])];
    launches.push({
      line: i + 1,
      host: host?.name ?? null,
      kernel,
      grid,
      block,
      threads:
        grid && block && dims.every((d) => d !== null)
          ? dims.reduce((a, b) => a * b, 1)
          : null,
    });
  }

  return kernels.length || launches.length ? { launches, kernels } : null;
}
