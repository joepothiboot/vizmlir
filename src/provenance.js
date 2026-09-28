// Follows each symbol (functions, kernels, GPU modules, globals) through a pass
// trace: which pass created it, which passes changed its body, which lowered it
// to another op (func.func → llvm.func), and which removed it. Reads IR text
// only, so it needs no engine parse.

import { moduleStateAt } from "./trace.js";

// Ops that define a symbol, by the last segment of their name: func.func,
// gpu.func, llvm.func, tt.func, gpu.module, spirv.module, gpu.binary,
// memref.global, llvm.mlir.global, transform.named_sequence. Uses such as
// func.call @f or gpu.launch_func @k::@f do not match.
const DEFINING = /^(?:func|module|binary|global|named_sequence)$/;
const HEADER =
  /^(\s*)(?:%[^=]+=\s*)?"?([A-Za-z_][\w$-]*(?:\.[\w$-]+)+)"?\s+(?:(?:\w+|"\w+")\s+)*@("(?:\\.|[^"\\])*"|[\w.$-]+)/;

function unquote(symbol) {
  return symbol.startsWith('"')
    ? symbol.slice(1, -1).replace(/\\(.)/g, "$1")
    : symbol;
}

function indentOf(line) {
  return /^\s*/.exec(line)[0].length;
}

// Returns a Map from symbol path (`@outer::@inner`, as in gpu.launch_func) to
// { path, symbol, op, line, text }. `text` is the op's own lines without the
// nested symbol ops, so a change inside a gpu.func is not also reported as a
// change to its gpu.module. Nesting is read from indentation, which is how
// MLIR prints; `line` is 1-based.
export function scanSymbols(ir) {
  const lines = ir.split("\n");
  const symbols = new Map();
  const stack = [];
  let owner = null;
  const own = new Map();

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim() === "") continue;
    const indent = indentOf(line);

    // Leave every op this line is not inside. A closing brace at the op's own
    // indent still belongs to it.
    while (
      stack.length &&
      (indent < stack.at(-1).indent ||
        (indent === stack.at(-1).indent && !/^\s*}/.test(line)))
    )
      stack.pop();
    owner = stack.at(-1) ?? null;

    const match = HEADER.exec(line);
    const op = match?.[2];
    if (op && DEFINING.test(op.slice(op.lastIndexOf(".") + 1)) && op !== "builtin.module") {
      const symbol = unquote(match[3]);
      const path = [...stack.map((entry) => entry.symbol), symbol]
        .map((name) => `@${name}`)
        .join("::");
      const entry = { path, symbol, op, line: i + 1, indent };
      if (!symbols.has(path)) {
        symbols.set(path, entry);
        own.set(path, []);
      }
      stack.push(entry);
      owner = entry;
    }
    if (owner) own.get(owner.path).push(line.slice(Math.min(owner.indent, indent)));
  }

  for (const [path, entry] of symbols)
    symbols.set(path, {
      path,
      symbol: entry.symbol,
      op: entry.op,
      line: entry.line,
      text: own.get(path).join("\n"),
    });
  return symbols;
}

// Compares the whole-module state at every event with the event before it.
// Returns one record per symbol ever seen, in order of first appearance:
//   { path, symbol, op, ops, initial, removed, changes }
// `initial` is true when the symbol is already in the first dump. `changes`
// lists { index, kind, op, from } where kind is created, changed, lowered
// (the defining op changed name; `from` is the old one) or removed. `op` is
// the latest defining op; `ops` every op it was defined by, in order.
export function symbolHistory(events) {
  const history = new Map();
  let previous = new Map();

  events.forEach((event, index) => {
    const current = scanSymbols(moduleStateAt(events, index));
    for (const [path, now] of current) {
      const before = previous.get(path);
      let record = history.get(path);
      if (!record) {
        record = {
          path,
          symbol: now.symbol,
          op: now.op,
          ops: [now.op],
          initial: index === 0,
          removed: false,
          changes: [],
        };
        history.set(path, record);
      }
      if (!before) {
        if (index > 0) record.changes.push({ index, kind: "created", op: now.op });
      } else if (before.op !== now.op) {
        record.changes.push({ index, kind: "lowered", op: now.op, from: before.op });
      } else if (before.text !== now.text) {
        record.changes.push({ index, kind: "changed", op: now.op });
      }
      if (record.op !== now.op) record.ops.push(now.op);
      record.op = now.op;
      record.removed = false;
    }
    for (const [path, then] of previous) {
      if (current.has(path)) continue;
      const record = history.get(path);
      record.changes.push({ index, kind: "removed", op: then.op });
      record.removed = true;
    }
    previous = current;
  });

  return [...history.values()];
}

// The symbols that `events[index]` created, changed, lowered or removed.
export function changesAt(history, index) {
  return history.flatMap((record) =>
    record.changes
      .filter((change) => change.index === index)
      .map((change) => ({ ...change, path: record.path })),
  );
}

// `benchmarks`, when given, maps a symbol path to the JSON object to export
// as that symbol's `benchmark` (see comparisonToJSON in bench.js).
export function historyToJSON(
  source,
  events,
  history,
  describe,
  benchmarks = null,
) {
  return `${JSON.stringify(
    {
      source,
      symbols: history.map((record) => ({
        symbol: record.path,
        op: record.op,
        ops: record.ops,
        initial: record.initial,
        removed: record.removed,
        changes: record.changes.map((change) => ({
          pass: describe(events[change.index]),
          dump: change.index + 1,
          kind: change.kind,
          op: change.op,
          ...(change.from ? { from: change.from } : {}),
        })),
        ...(benchmarks?.has(record.path)
          ? { benchmark: benchmarks.get(record.path) }
          : {}),
      })),
    },
    null,
    2,
  )}\n`;
}
