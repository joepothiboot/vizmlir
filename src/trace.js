// Splits `mlir-opt -mlir-print-ir-{before,after}-all` logs into pass events.
//
// Each event keeps the IR printed for one pass boundary. Compiler diagnostics
// interleaved on stderr are kept as structured records instead of being mixed
// into the IR, and anything printed after the last dump (usually the final
// module from stdout) is kept as `output`.

const HEADER_NEW = /^\/\/ -----\/\/ IR Dump (.*) \/\/----- \/\/\s*$/;
const HEADER_OLD = /^\/\/ \*\*\* IR Dump (.*) \*\*\*\s*$/;
const ANCHOR = / \('([^']+)' operation(?:: @("(?:\\.|[^"\\])*"|[^\s)]+))?\)$/;
const DIAGNOSTIC = /^(.+?):(\d+)(?::(\d+))?: (error|warning|note|remark): (.*)$/;
const ALIAS = /^[#!][\w.$-]+ = /;
const ROOT_OP = /^(?:%[^=]+=\s*)?"?([A-Za-z_][\w.$-]*)"?(?:\s+(@(?:"(?:\\.|[^"\\])*"|[\w.$-]+)))?/;

export function isPassTrace(text) {
  const probe = text.length > 65536 ? text.slice(0, 65536) : text;
  return probe.split(/\r?\n/, 2000).some((line) => HEADER_NEW.test(line) || HEADER_OLD.test(line));
}

export function parseHeader(line) {
  const match = HEADER_NEW.exec(line) ?? HEADER_OLD.exec(line);
  if (!match) return null;
  let body = match[1];

  let anchor = null;
  const anchorMatch = ANCHOR.exec(body);
  if (anchorMatch) {
    anchor = { op: anchorMatch[1], symbol: anchorMatch[2] ? unquoteSymbol(anchorMatch[2]) : null };
    body = body.slice(0, anchorMatch.index);
  }

  const phase = /^(Before|After) /.exec(body);
  if (!phase) return null;
  body = body.slice(phase[0].length);

  let pass = body;
  let argument = null;
  let failed = false;
  // LLVM 20+: `CSEPass: cse` / `InterpreterPass Failed: transform-interpreter{...}`
  // Older:    `CSE (cse)`    / `CSE Failed (cse)`
  const modern = /^(.+?)( Failed)?: (\S.*)$/.exec(body);
  const legacy = /^(.+?)( Failed)? \(([^()]*)\)$/.exec(body);
  if (modern) {
    [, pass, , argument] = modern;
    failed = Boolean(modern[2]);
  } else if (legacy) {
    [, pass, , argument] = legacy;
    failed = Boolean(legacy[2]);
  } else if (body.endsWith(' Failed')) {
    pass = body.slice(0, -' Failed'.length);
    failed = true;
  }

  let options = null;
  if (argument) {
    const brace = argument.indexOf('{');
    if (brace >= 0) {
      options = argument.slice(brace + 1, argument.lastIndexOf('}')).trim().replace(/\s+/g, ' ');
      argument = argument.slice(0, brace);
    }
  }

  return { phase: phase[1].toLowerCase(), pass, argument, options, failed, anchor };
}

export function parsePassTrace(text) {
  const lines = text.split(/\r?\n/);
  const events = [];
  const diagnostics = [];
  const preamble = [];
  let pending = [];

  let i = 0;
  while (i < lines.length && !parseHeader(lines[i])) {
    i = collectLoose(lines, i, preamble, pending);
  }

  while (i < lines.length) {
    const header = parseHeader(lines[i]);
    const event = {
      index: events.length,
      ...header,
      headerLine: i + 1,
      irLine: i + 2,
      ir: '',
      root: null,
      diagnostics: [],
      trailing: '',
    };
    if (event.phase === 'after') pending = attach(pending, event, diagnostics);
    events.push(event);

    const irEnd = findIrEnd(lines, i + 1);
    event.ir = trimBlankLines(lines.slice(i + 1, irEnd)).join('\n');
    event.root = rootOf(event.ir);
    event.scope = scopeKey(event.root);

    const loose = [];
    i = irEnd;
    while (i < lines.length && !parseHeader(lines[i])) {
      i = collectLoose(lines, i, loose, pending);
    }
    event.trailing = trimBlankLines(loose).join('\n');
  }

  // Diagnostics after the final dump come from a pass that printed no dump,
  // usually one that failed without -mlir-print-ir-after-failure.
  for (const diagnostic of pending) diagnostics.push({ ...diagnostic, eventIndex: -1 });

  const last = events.at(-1);
  const output = last?.trailing ?? '';
  if (last) last.trailing = '';

  return { events, diagnostics, preamble: trimBlankLines(preamble).join('\n'), output };
}

// Returns the IR that `events[index]` should be compared against, or null
// when the trace has no earlier snapshot of the same operation.
//
// Without -mlir-print-ir-module-scope, nested passes dump only the function
// they ran on, so the latest state of an operation may live in a nested dump
// or inside an enclosing module dump. Module baselines are rebuilt by splicing
// newer nested dumps into the last module dump (or into an empty module).
export function baselineFor(events, index) {
  const event = events[index];
  if (!event?.root) return null;

  if (event.root.op === 'builtin.module') return moduleBaseline(events, index);

  for (let j = index - 1; j >= 0; j--) {
    if (events[j].scope === event.scope) return { event: events[j], ir: events[j].ir, reconstructed: false };
    if (!event.root.symbol) continue;
    const range = findSymbolOp(events[j].ir, event.root.op, event.root.symbol);
    if (range) return { event: events[j], ir: range.text, reconstructed: false };
  }
  return null;
}

function moduleBaseline(events, index) {
  let base = -1;
  for (let j = index - 1; j >= 0; j--) {
    if (events[j].scope === events[index].scope) {
      base = j;
      break;
    }
  }

  const latest = new Map();
  for (let j = base + 1; j < index; j++) {
    if (events[j].root?.symbol) latest.set(events[j].scope, events[j]);
  }
  if (base >= 0 && latest.size === 0) return { event: events[base], ir: events[base].ir, reconstructed: false };
  if (latest.size === 0) return null;

  let ir = base >= 0 ? events[base].ir : 'module {\n}';
  for (const nested of latest.values()) ir = spliceSymbolOp(ir, nested);
  return { event: [...latest.values()].at(-1), ir, reconstructed: true };
}

// Replaces the op named by `nested.root` inside `ir`, or appends it before the
// closing brace of the enclosing op when it is not there yet.
function spliceSymbolOp(ir, nested) {
  const lines = ir.split('\n');
  const range = findSymbolOp(ir, nested.root.op, nested.root.symbol);
  const body = nested.ir.split('\n').filter((line) => !ALIAS.test(line));
  if (range) {
    const indent = ' '.repeat(range.indent);
    lines.splice(range.from, range.to - range.from, ...body.map((line) => (line ? indent + line : line)));
    return lines.join('\n');
  }
  const close = lines.findLastIndex((line) => /^}/.test(line));
  if (close < 0) return ir;
  lines.splice(close, 0, ...body.map((line) => (line ? `  ${line}` : line)));
  return lines.join('\n');
}

export function describeEvent(event) {
  const name = event.argument || event.pass;
  const on = event.anchor ?? event.root;
  const target = on ? ` · ${on.op}${on.symbol ? ` @${on.symbol}` : ''}` : '';
  return `${event.phase === 'before' ? 'Before' : 'After'} ${name}${target}`;
}

export function extractSymbolOp(ir, op, symbol) {
  return findSymbolOp(ir, op, symbol)?.text ?? null;
}

function findSymbolOp(ir, op, symbol) {
  const lines = ir.split('\n');
  const escaped = symbol.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const start = new RegExp(`^\\s*(?:%[^=]+=\\s*)?${op.replace(/\./g, '\\.')}\\s+(?:\\w+\\s+)*@"?${escaped}"?[\\s(<{:]`);
  const from = lines.findIndex((line) => start.test(line));
  if (from < 0) return null;
  let to = findIrEnd(lines, from);
  while (to > from + 1 && (lines[to - 1].trim() === '' || ALIAS.test(lines[to - 1]))) to -= 1;
  const indent = /^\s*/.exec(lines[from])[0].length;
  const text = lines.slice(from, to).map((line) => line.slice(Math.min(indent, /^\s*/.exec(line)[0].length))).join('\n');
  return { from, to, indent, text };
}

function attach(pending, event, diagnostics) {
  for (const diagnostic of pending) {
    const record = { ...diagnostic, eventIndex: event.index };
    diagnostics.push(record);
    event.diagnostics.push(record);
  }
  return [];
}

// Consumes one loose line (or a whole diagnostic block) outside of IR.
function collectLoose(lines, i, sink, pending) {
  const match = DIAGNOSTIC.exec(lines[i]);
  if (!match) {
    sink.push(lines[i]);
    return i + 1;
  }
  const [, file, line, column, severity, message] = match;
  const detail = [];
  let j = i + 1;
  // Continuation lines hold the source snippet and caret, or the op printed by
  // "see current operation". They end at a blank line or the next record.
  while (j < lines.length && lines[j].trim() !== '' && !DIAGNOSTIC.test(lines[j]) && !parseHeader(lines[j])) {
    detail.push(lines[j]);
    j += 1;
  }
  pending.push({
    severity,
    message,
    location: { file, line: Number(line), column: column ? Number(column) : null },
    detail: detail.join('\n'),
    traceLine: i + 1,
  });
  return j;
}

// A dump prints one top-level operation, surrounded by attribute and type
// aliases. Returns the first line index after it.
function findIrEnd(lines, from) {
  let i = from;
  let depth = 0;
  let started = false;
  for (; i < lines.length; i++) {
    const line = lines[i];
    if (parseHeader(line)) return i;
    if (started && depth === 0) break;
    if (line.trim() === '' || ALIAS.test(line)) continue;
    if (DIAGNOSTIC.test(line)) return i;
    started = true;
    depth += braceDelta(line);
  }
  while (i < lines.length && (lines[i].trim() === '' || ALIAS.test(lines[i]))) i += 1;
  return i;
}

function braceDelta(line) {
  let delta = 0;
  let inString = false;
  for (let k = 0; k < line.length; k++) {
    const c = line[k];
    if (inString) {
      if (c === '\\') k += 1;
      else if (c === '"') inString = false;
    } else if (c === '"') inString = true;
    else if (c === '{') delta += 1;
    else if (c === '}') delta -= 1;
    else if (c === '/' && line[k + 1] === '/') break;
  }
  return delta;
}

function rootOf(ir) {
  const line = ir.split('\n').find((candidate) => candidate.trim() && !ALIAS.test(candidate));
  const match = line && ROOT_OP.exec(line.trim());
  if (!match) return null;
  const op = match[1] === 'module' ? 'builtin.module' : match[1];
  return { op, symbol: match[2] ? unquoteSymbol(match[2].slice(1)) : null };
}

function scopeKey(root) {
  return root ? `${root.op}@${root.symbol ?? ''}` : '';
}

function unquoteSymbol(symbol) {
  return symbol.startsWith('"') ? symbol.slice(1, -1) : symbol;
}

function trimBlankLines(lines) {
  let start = 0;
  let end = lines.length;
  while (start < end && lines[start].trim() === '') start += 1;
  while (end > start && lines[end - 1].trim() === '') end -= 1;
  return lines.slice(start, end);
}
