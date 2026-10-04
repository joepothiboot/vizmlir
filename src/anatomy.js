// "What's this line?": splits one line of printed MLIR into the parts a
// newcomer needs named (result, operation, inputs, position, settings, type)
// and explains each in plain words. It reads the printed form one line at a
// time, so it labels what it recognizes and leaves the rest undescribed.

import { parseMemref } from "./trace/index.js";
import { formatBytes } from "./trace/index.js";

const DOCS = "https://mlir.llvm.org/docs/Dialects/";
const DIALECT_DOCS = {
  arith: "ArithOps/",
  memref: "MemRef/",
  scf: "SCFDialect/",
  gpu: "GPU/",
  func: "Func/",
  linalg: "Linalg/",
  tensor: "TensorOps/",
  vector: "Vector/",
  llvm: "LLVM/",
  nvvm: "NVVMDialect/",
  cf: "ControlFlowDialect/",
  affine: "Affine/",
  math: "MathOps/",
  bufferization: "BufferizationOps/",
  builtin: "Builtin/",
  transform: "Transform/",
};

const DIALECT_TEXT = {
  arith: "basic arithmetic on numbers",
  memref: "arrays that live in memory",
  scf: "loops and if/else",
  gpu: "running code on the GPU",
  func: "functions",
  linalg: "whole-array math like matrix multiply",
  tensor: "arrays treated as values",
  vector: "several numbers processed at once",
  llvm: "LLVM, the low-level form just before machine code",
  nvvm: "NVIDIA GPU instructions",
  cf: "jumps between blocks of code",
  affine: "loops and indices with simple arithmetic",
  math: "math functions like exp and sqrt",
  bufferization: "turning tensor values into memory",
  builtin: "the program's outer structure",
  transform: "scripts that tell the compiler how to transform code",
};

// One plain sentence per common operation. Anything else still gets its
// parts labeled, without a description.
const OPS = {
  "arith.constant": "Makes a fixed value, like the number 32.",
  "arith.addi": "Adds two whole numbers.",
  "arith.subi": "Subtracts one whole number from another.",
  "arith.muli": "Multiplies two whole numbers.",
  "arith.divsi": "Divides two whole numbers, rounding toward zero.",
  "arith.divui": "Divides two non-negative whole numbers.",
  "arith.remsi": "Gives the remainder of a whole-number division.",
  "arith.remui": "Gives the remainder of a whole-number division.",
  "arith.addf": "Adds two decimal (floating-point) numbers.",
  "arith.subf": "Subtracts one decimal number from another.",
  "arith.mulf": "Multiplies two decimal numbers.",
  "arith.divf": "Divides two decimal numbers.",
  "arith.maximumf": "Picks the larger of two decimal numbers.",
  "arith.minimumf": "Picks the smaller of two decimal numbers.",
  "arith.cmpi": "Compares two whole numbers and gives true or false.",
  "arith.cmpf": "Compares two decimal numbers and gives true or false.",
  "arith.select": "Picks one of two values depending on a true/false condition.",
  "arith.index_cast": "Converts a number between the index type and a fixed-size integer.",
  "arith.sitofp": "Turns a whole number into a decimal number.",
  "arith.fptosi": "Turns a decimal number into a whole number.",
  "memref.load": "Reads one item from an array in memory.",
  "memref.store": "Writes one item into an array in memory.",
  "memref.alloc": "Sets aside memory for a new array.",
  "memref.alloca": "Sets aside short-lived memory for a new array, freed when the function ends.",
  "memref.dealloc": "Gives an array's memory back.",
  "memref.copy": "Copies one array into another.",
  "memref.subview": "Looks at part of an array without copying it.",
  "memref.cast": "Views an array with a slightly different type, without copying it.",
  "memref.global": "Declares an array that lives for the whole program.",
  "memref.get_global": "Gets hold of an array declared with memref.global.",
  "scf.for": "Repeats the code inside it, counting from a start to an end in steps.",
  "scf.parallel": "Runs the code inside it for every index, in any order.",
  "scf.if": "Runs one block of code or another depending on a condition.",
  "scf.yield": "Hands values back to the loop or if that contains it.",
  "scf.while": "Repeats the code inside it while a condition holds.",
  "func.func": "Defines a function: a named piece of code with inputs and outputs.",
  "func.return": "Ends the function and hands back its results.",
  return: "Ends the function and hands back its results.",
  "func.call": "Calls another function.",
  "gpu.launch": "Starts work on the GPU: the code inside runs once per thread, in a grid of blocks.",
  "gpu.launch_func": "Starts a GPU kernel defined elsewhere, as a grid of blocks of threads.",
  "gpu.module": "Holds the code that will be compiled for the GPU.",
  "gpu.func": "Defines a function that runs on the GPU. Marked kernel, it is where a launch starts.",
  "gpu.thread_id": "Gives this thread's position inside its block (along x, y or z).",
  "gpu.block_id": "Gives this block's position inside the grid (along x, y or z).",
  "gpu.block_dim": "Gives how many threads a block has along x, y or z.",
  "gpu.grid_dim": "Gives how many blocks the grid has along x, y or z.",
  "gpu.barrier": "Makes every thread in the block wait here until all of them arrive.",
  "gpu.terminator": "Marks the end of the code inside a gpu.launch.",
  "gpu.return": "Ends a GPU function.",
  "gpu.binary": "Holds the finished GPU program, compiled from a gpu.module.",
  "gpu.alloc": "Sets aside memory on the GPU for a new array.",
  "gpu.memcpy": "Copies an array between the CPU and the GPU.",
  "linalg.matmul": "Multiplies two matrices.",
  "linalg.fill": "Fills an array with one value.",
  "linalg.generic": "Applies the same calculation to every element, as its body describes.",
  "tensor.empty": "Makes a new tensor whose contents don't matter yet.",
  "cf.br": "Jumps to another block of code.",
  "cf.cond_br": "Jumps to one of two blocks depending on a condition.",
  "llvm.func": "Defines a function in LLVM's low-level form.",
  "llvm.return": "Ends an LLVM function.",
  "llvm.load": "Reads a value from memory through a pointer.",
  "llvm.store": "Writes a value to memory through a pointer.",
  "llvm.getelementptr": "Works out the address of an item inside an array.",
  "llvm.insertvalue": "Puts a value into one field of a struct.",
  "llvm.extractvalue": "Takes a value out of one field of a struct.",
  "llvm.mlir.constant": "Makes a fixed value in LLVM's form.",
  "llvm.mlir.poison": "Makes a placeholder value, to be filled in field by field.",
  "nvvm.read.ptx.sreg.tid.x": "Reads this thread's x position inside its block (threadIdx.x in CUDA).",
  "nvvm.read.ptx.sreg.ctaid.x": "Reads this block's x position inside the grid (blockIdx.x in CUDA).",
  "nvvm.barrier0": "Makes every thread in the block wait here (__syncthreads in CUDA).",
  module: "The outermost container that holds the whole program.",
};

// Element types as a plural noun phrase, for "an array of …".
function elementName(type) {
  const t = type.trim();
  if (t === "index") return "indices";
  if (t === "i1") return "true/false values";
  const int = /^[su]?i(\d+)$/.exec(t);
  if (int) return `${int[1]}-bit whole numbers`;
  const float = /^(?:f|bf)(\d+)/.exec(t);
  if (float) return `${float[1]}-bit decimal numbers`;
  return t;
}

// Splits `s` on `sep` where it is not nested in <>, (), [] or {}.
function splitTop(s, sep = ",") {
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if ("<([{".includes(c)) depth++;
    else if (">)]}".includes(c) && s[i - 1] !== "-") depth--;
    else if (depth === 0 && s.startsWith(sep, i)) {
      parts.push(s.slice(start, i));
      start = i + sep.length;
    }
  }
  parts.push(s.slice(start));
  return parts;
}

// Index of the bracket that closes the one at `open`, or -1.
function closing(s, open) {
  const want = { "(": ")", "[": "]", "{": "}", "<": ">" }[s[open]];
  let depth = 0;
  for (let i = open; i < s.length; i++) {
    if (s[i] === s[open]) depth++;
    else if (s[i] === want && !(want === ">" && s[i - 1] === "-") && --depth === 0)
      return i;
  }
  return -1;
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** Describes a type in plain words, or null when it is not recognized. */
export function describeType(type) {
  const t = type.trim();
  if (t === "index") return "an index: a whole number used for positions and sizes";
  if (t === "i1") return "true or false";
  const name = elementName(t);
  if (name !== t && name !== "indices") return `a ${name.replace(/s$/, "")}`;
  if (t.startsWith("memref<")) {
    const memref = parseMemref(t);
    if (!memref) return null;
    const dims = memref.dims.map((d) => d ?? "?");
    const shape = dims.length ? `a ${dims.join(" × ")} array` : "a one-item array";
    const size = memref.bytes !== null ? ` (${formatBytes(memref.bytes)})` : "";
    const space = { 3: "shared", workgroup: "shared", 5: "private", private: "private", 1: "global", global: "global" }[memref.space];
    return `${shape} of ${elementName(memref.element)} in ${space ? `${space} ` : ""}memory${size}`;
  }
  const shaped = /^(tensor|vector)<(.*)>$/.exec(t);
  if (shaped) {
    const dims = shaped[2].split("x");
    const element = elementName(dims.pop());
    if (shaped[1] === "vector")
      return `a vector of ${dims.join(" × ")} ${element}, processed together`;
    return `a ${dims.length ? `${dims.join(" × ")} ` : ""}tensor (an array treated as a value) of ${element}`;
  }
  if (t === "!llvm.ptr" || t.startsWith("!llvm.ptr<")) return "a pointer: the address of something in memory";
  if (t.startsWith("!llvm.struct<")) return "a struct: several values bundled together";
  if (t.startsWith("(") && t.includes("->")) {
    const count = (s) => {
      const inner = s.trim().replace(/^\((.*)\)$/, "$1").trim();
      return inner ? splitTop(inner).length : 0;
    };
    const [inputs, outputs] = splitTop(t, "->");
    return `a function type: takes ${plural(count(inputs), "input")} and gives ${plural(count(outputs), "result")}`;
  }
  return null;
}

// Where `name` gets its value, searching up from line `at` (0-based), then
// down: { line, how } or null.
export function findDefinition(lines, name, at) {
  const escaped = name.replace(/[$.]/g, "\\$&");
  const tests = [
    ["result", new RegExp(`^\\s*(?:%[\\w$.-]+(?::\\d+)?\\s*,\\s*)*${escaped}(?::\\d+)?\\s*(?:,[^=]*)?=`)],
    ["loop", new RegExp(`\\bfor\\s+${escaped}\\s*=`)],
    ["loop-carried", new RegExp(`iter_args\\s*\\([^)]*${escaped}\\s*=`)],
    ["launch", new RegExp(`\\b(?:blocks|threads)\\s*\\([^)]*${escaped}\\b`)],
    ["argument", new RegExp(`[(,]\\s*${escaped}\\s*:`)],
  ];
  const check = (i) => {
    if (!lines[i]?.includes(name)) return null;
    const hit = tests.find(([, test]) => test.test(lines[i]));
    return hit ? { line: i, how: hit[0] } : null;
  };
  // Nearest above first, since SSA names are reused across functions.
  for (let i = Math.min(at, lines.length - 1); i >= 0; i--) {
    const found = check(i);
    if (found) return found;
  }
  for (let i = at + 1; i < lines.length; i++) {
    const found = check(i);
    if (found) return found;
  }
  return null;
}

const HOW = {
  result: "made on line",
  loop: "the loop counter from line",
  "loop-carried": "a value carried around the loop, from line",
  launch: "a thread or block number from the launch on line",
  argument: "an input to the function on line",
};

/**
 * Explains line `at` (0-based) of `lines`. Returns null when there is
 * nothing to explain, else { op, dialect, summary, dialectText, docs, parts }
 * where each part is { kind, text, label, detail, ref? } and `ref` is the
 * 0-based line an input is defined on.
 */
export function explainLine(lines, at) {
  const line = (lines[at] ?? "").trim();
  if (!line || line.startsWith("//")) return null;
  const note = (summary) => ({ op: null, summary, parts: [] });
  if (line.startsWith("}"))
    return note("Closes the block of code opened above it, at the line with the matching {.");
  if (/^[#!][\w.$-]+\s*=/.test(line))
    return note("Gives a short name to a long setting or type, so other lines can refer to it.");
  if (/^\^bb\d+/.test(line))
    return note("Starts a block: a stretch of code that other lines can jump to. The names in brackets are the values it receives.");

  const parts = [];
  let rest = line;
  const results = /^((?:%[\w$.-]+(?::\d+)?\s*,\s*)*%[\w$.-]+(?::\d+)?)\s*=\s*/.exec(rest);
  if (results) {
    const names = results[1].split(",").map((s) => s.trim());
    parts.push({
      kind: "result",
      text: results[1],
      label: names.length > 1 ? "Results" : "Result",
      detail:
        names.length > 1
          ? `Names the ${names.length} values this line produces, so later lines can use them.`
          : `Names the value this line produces. Later lines use it as ${names[0]}.`,
    });
    rest = rest.slice(results[0].length);
  }

  const opMatch = /^"?([A-Za-z_][\w$-]*(?:\.[\w$-]+)*)"?/.exec(rest);
  if (!opMatch) return null;
  const op = opMatch[1];
  const dialect = op.includes(".") ? op.slice(0, op.indexOf(".")) : "builtin";
  parts.push({
    kind: "op",
    text: opMatch[0],
    label: "Operation",
    detail: op.includes(".")
      ? `"${op.slice(op.indexOf(".") + 1)}" from the ${dialect} family: ${DIALECT_TEXT[dialect] ?? "a group of related operations"}.`
      : "A built-in operation.",
  });
  rest = rest.slice(opMatch[0].length);

  // The trailing `: type`, outside any brackets.
  let type = null;
  const typed = splitTop(rest, " : ");
  if (typed.length > 1 && typed.at(-1).trim()) {
    type = typed.pop().trim().replace(/\s*\{$/, "").trim();
    rest = typed.join(" : ");
  }

  const seen = new Set();
  for (let i = 0; i < rest.length; i++) {
    const c = rest[i];
    const before = rest.slice(0, i);
    const after = rest.slice(i + 1);
    if (c === "%") {
      const name = /^%[\w$.-]+/.exec(rest.slice(i))[0];
      i += name.length - 1;
      if (seen.has(name)) continue;
      seen.add(name);
      const next = rest.slice(i + 1);
      // A gpu.launch binds ids in `blocks(%bx, %by, %bz)` and sizes in
      // `in (%gx = %c8, ...)`; the position in the list is the axis.
      const launchIds = /\b(blocks|threads|clusters)\s*\(([^)]*)$/.exec(before);
      const launchSizes = /\b(blocks|threads|clusters)\s*\([^)]*\)\s*in\s*\(([^)]*)$/.exec(before);
      if (launchIds || (launchSizes && /^\s*=/.test(next))) {
        const [, what, list] = launchIds ?? launchSizes;
        const axis = "xyz"[list.split(",").length - 1] ?? "?";
        const unit = what === "threads" ? "thread" : what === "blocks" ? "block" : "cluster";
        parts.push({
          kind: "binding",
          text: name,
          label: "New name",
          detail: launchIds
            ? `Inside the launch, each ${unit}'s own ${axis} number.`
            : `Inside the launch, how many ${unit}s there are along ${axis}.`,
        });
        continue;
      }
      // Other names this op binds for its body: `for %i =`,
      // `iter_args(%acc =`, a function's `(%arg0: type`.
      if (/^\s*=/.test(next) || (/^\s*:/.test(next) && /\(/.test(before))) {
        parts.push({
          kind: "binding",
          text: name,
          label: "New name",
          detail: /iter_args\s*\([^)]*$/.test(before)
              ? `A value carried from one trip around the loop to the next, called ${name} inside it.`
              : /\.for$/.test(op) && !before.trim()
                ? `The loop counter: ${name} takes each value in turn inside the loop.`
                : /^\s*:/.test(next)
                ? `An input of this function, called ${name} inside it.`
                : `Introduces ${name} for the code inside this operation.`,
        });
        continue;
      }
      const def = findDefinition(lines, name, at - 1);
      parts.push({
        kind: "input",
        text: name,
        label: "Input",
        detail: def ? `${name} is ${HOW[def.how]} ${def.line + 1}.` : `${name} comes from elsewhere in the program.`,
        ...(def ? { ref: def.line } : {}),
      });
    } else if (c === "[" && parts.at(-1)?.kind === "input") {
      const end = closing(rest, i);
      if (end < 0) continue;
      const indices = splitTop(rest.slice(i + 1, end)).map((s) => s.trim()).filter(Boolean);
      parts.push({
        kind: "position",
        text: `[${indices.join(", ")}]`,
        label: "Position",
        detail:
          indices.length === 2
            ? `Which item: row ${indices[0]}, column ${indices[1]}.`
            : indices.length === 1
              ? `Which item: number ${indices[0]}.`
              : `Which item, one number per dimension (${indices.length}).`,
      });
      // The loop goes on to visit the names inside the brackets.
    } else if (c === "@") {
      const symbol = /^@(?:"(?:\\.|[^"\\])*"|[\w$.-]+)(?:::@(?:"(?:\\.|[^"\\])*"|[\w$.-]+))*/.exec(rest.slice(i))?.[0];
      if (!symbol) continue;
      i += symbol.length - 1;
      parts.push({
        kind: "symbol",
        text: symbol,
        label: "Name",
        detail: symbol.includes("::")
          ? `Refers to ${symbol.split("::").at(-1)} inside ${symbol.split("::")[0]}.`
          : /(?:func|module|global|binary)$/.test(op)
            ? `The name other code uses to call or find this ${op.split(".").at(-1)}.`
            : `Refers to ${symbol} by name.`,
      });
    } else if (c === "{" && closing(rest, i) > 0) {
      // A `{` that closes on this line holds settings; one that stays open
      // starts the op's body.
      const end = closing(rest, i);
      parts.push({
        kind: "attributes",
        text: rest.slice(i, end + 1),
        label: "Settings",
        detail: "Extra fixed information attached to the operation (its attributes).",
      });
      i = end;
    } else if (/^\s*$/.test(after) && c === "{") {
      break;
    }
  }

  if (type) {
    const described = splitTop(type.replace(/^\(([^()]*)\)$/, "$1"))
      .map((t) => t.trim())
      .filter(Boolean)
      .map((t) => describeType(t));
    parts.push({
      kind: "type",
      text: `: ${type}`,
      label: "Type",
      detail: described.length && described.every(Boolean)
        ? `${described.join("; ")}.`.replace(/^./, (c) => c.toUpperCase())
        : "What kind of data the values are.",
    });
  }
  if (/\{$/.test(line))
    parts.push({
      kind: "region",
      text: "{",
      label: "Body",
      detail: "Opens the code this operation contains. It ends at the matching }.",
    });

  const constant = op.endsWith(".constant") && /^\s*(-?[\d.e+-]+|true|false)\b/.exec(rest);
  return {
    op,
    dialect,
    summary: constant ? `Makes the fixed value ${constant[1]}.` : (OPS[op] ?? null),
    dialectText: DIALECT_TEXT[dialect] ?? null,
    docs: DOCS + (DIALECT_DOCS[dialect] ?? ""),
    parts,
  };
}
