import { moduleStateAt } from "./trace.js";

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

export function scanSymbols(ir) {
  const lines = ir.split("\n");
  const symbols = new Map();
  const stack = [];
  let owner = null;
  const own = new Map();
  let last = -1;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim() === "") continue;

    const indent = indentOf(line);

    while (
      stack.length &&
      (indent < stack.at(-1).indent ||
        (indent === stack.at(-1).indent && !/^\s*[}^]/.test(line)))
    ) {
      stack.pop().end = last;
    }

    owner = stack.at(-1) ?? null;
    last = i;

    const match = HEADER.exec(line);
    const op = match?.[2];

    if (
      op &&
      DEFINING.test(op.slice(op.lastIndexOf(".") + 1)) &&
      op !== "builtin.module"
    ) {
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

    if (owner) {
      own.get(owner.path).push(line.slice(Math.min(owner.indent, indent)));
    }
  }

  for (const entry of stack) entry.end = last;

  for (const [path, entry] of symbols) {
    symbols.set(path, {
      path,
      symbol: entry.symbol,
      op: entry.op,
      line: entry.line,
      text: own.get(path).join("\n"),
      full: lines
        .slice(entry.line - 1, entry.end + 1)
        .map((line) => line.slice(Math.min(entry.indent, indentOf(line))))
        .join("\n"),
    });
  }

  return symbols;
}

function stepKind(previous, now, seenBefore) {
  if (!previous) return seenBefore ? "created" : "initial";
  if (previous.op !== now.op) return "lowered";
  if (previous.full !== now.full) return "changed";

  return null;
}

export function symbolTimeline(events, path) {
  const steps = [];
  let previous = null;

  events.forEach((event, index) => {
    const symbols = scanSymbols(moduleStateAt(events, index));
    const now = symbols.get(path);

    if (!now) {
      if (previous) {
        const outer = [...symbols.values()]
          .filter((symbol) => path.startsWith(`${symbol.path}::`))
          .flatMap((symbol) => embeddedAssembly(symbol.full));

        steps.push({
          index,
          kind: "removed",
          op: previous.op,
          text: null,
          ...(outer.length ? { assembly: outer } : {}),
        });
      }

      previous = null;

      return;
    }

    const kind = stepKind(previous, now, steps.length || index > 0);

    if (kind) {
      const assembly = embeddedAssembly(now.full);

      steps.push({
        index,
        kind,
        op: now.op,
        text: now.full,
        ...(kind === "lowered" ? { from: previous.op } : {}),
        ...(assembly.length ? { assembly } : {}),
      });
    }

    previous = now;
  });

  return steps;
}

export function unescapeMlirString(text) {
  const bytes = text.replace(/\\(?:([0-9A-Fa-f]{2})|(.))/g, (_, hex, c) =>
    String.fromCharCode(
      hex ? parseInt(hex, 16) : ({ n: 10, t: 9 }[c] ?? c.charCodeAt(0)),
    ),
  );

  return new TextDecoder().decode(
    Uint8Array.from(bytes, (c) => c.charCodeAt(0) & 0xff),
  );
}

export function embeddedAssembly(ir) {
  const objects = [];

  const pattern =
    /#gpu\.object<(#[\w.]+(?:<[^>]*>)?)[^"]*?\bassembly = "((?:\\.|[^"\\])*)"/g;

  for (const match of ir.matchAll(pattern)) {
    objects.push({ target: match[1], text: unescapeMlirString(match[2]) });
  }

  return objects;
}

export function symbolHistory(events) {
  const history = new Map();
  let previous = new Map();

  events.forEach((event, index) => {
    const current = scanSymbols(moduleStateAt(events, index));
    const dumped = [...scanSymbols(event.ir).keys()];

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
          lastDump: -1,
        };

        history.set(path, record);
      }

      if (!before) {
        if (index > 0) {
          record.changes.push({ index, kind: "created", op: now.op });
        }
      } else if (before.op !== now.op) {
        record.changes.push({
          index,
          kind: "lowered",
          op: now.op,
          from: before.op,
        });
      } else if (before.text !== now.text) {
        record.changes.push({ index, kind: "changed", op: now.op });
      }

      if (record.op !== now.op) record.ops.push(now.op);
      record.op = now.op;
      record.removed = false;

      if (dumped.some((local) => pathEndsWith(path, local))) {
        record.lastDump = index;
      }
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

function pathEndsWith(path, local) {
  return path === local || path.endsWith(`::${local}`);
}

export function findSymbolNode(snapshot, path) {
  if (!snapshot) return -1;

  const label = (i) => snapshot.nodes?.[i]?.label ?? snapshot.labelOf(i);
  const parent = (i) => snapshot.nodes?.[i]?.parent ?? snapshot.parentOf(i);

  const symbolOf = (i) => {
    const match = HEADER.exec(label(i));
    const op = match?.[2];

    return op &&
      op !== "builtin.module" &&
      DEFINING.test(op.slice(op.lastIndexOf(".") + 1))
      ? unquote(match[3])
      : null;
  };

  let best = -1;
  let bestDepth = 0;

  for (let i = 0; i < snapshot.nodeCount; i++) {
    const symbol = symbolOf(i);
    if (!symbol) continue;

    const names = [symbol];

    for (let p = parent(i); p >= 0; p = parent(p)) {
      const outer = symbolOf(p);
      if (outer) names.unshift(outer);
    }

    const local = names.map((name) => `@${name}`).join("::");

    if (pathEndsWith(path, local) && names.length > bestDepth) {
      best = i;
      bestDepth = names.length;
    }
  }

  return best;
}

export function changesAt(history, index) {
  return history.flatMap((record) =>
    record.changes
      .filter((change) => change.index === index)
      .map((change) => ({ ...change, path: record.path })),
  );
}

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
