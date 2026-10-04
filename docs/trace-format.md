# Trace format

VizMLIR reads the plain-text log that an MLIR driver writes when it is asked
to print IR between passes. It has no format of its own; this document
describes what the parser (`src/trace/trace.js`, `src/trace/timing.js`) accepts and the
structure it builds, so other tools can produce or consume the same thing.

## Producing a trace

Any driver built on `MlirOptMain` works: `mlir-opt` itself, or an
out-of-tree driver such as `nanodsp-opt` or `schema-opt`. Merge stdout and
stderr, because IR dumps go to one and diagnostics to the other:

```bash
mlir-opt input.mlir -pass-pipeline='builtin.module(...)' \
  -mlir-print-ir-after-all -mlir-disable-threading > trace.txt 2>&1
```

| Flag                                    | Effect in VizMLIR                                                           |
| --------------------------------------- | --------------------------------------------------------------------------- |
| `-mlir-print-ir-after-all`              | one event per pass run (the usual choice)                                   |
| `-mlir-print-ir-before-all`             | "before" events; can be combined with the above                             |
| `-mlir-print-ir-after-failure`          | only the IR at the failing pass                                             |
| `-mlir-print-ir-module-scope`           | every dump is the whole module (otherwise nested passes dump only their op) |
| `-mlir-timing`                          | per-pass wall time in the timeline (tree, `list` or JSON output)            |
| `-mlir-disable-threading`               | keeps dumps from interleaving; recommended                                  |
| `/usr/bin/time -l` (macOS) / `-v` (GNU) | peak memory for the whole process                                           |

`tests/fixtures/traces/generate.sh` produces every supported variant,
including `schema-opt-pipeline.txt` from an out-of-tree driver with its own
dialect.

## Input grammar

A trace is a sequence of lines. The parser recognizes four kinds of block and
treats everything else as loose text.

### Dump header

Two header styles are accepted:

```
// -----// IR Dump After CSEPass: cse ('func.func' operation: @f) //----- //     LLVM 17+
// *** IR Dump After CSE (cse) ***                                              older
```

The header body is read as:

```
(Before|After) <pass>[ Failed][: <argument>[{<options>}]] [('<op>' operation[: @<symbol>])]
```

- `<pass>` is the C++ pass name and may contain namespaces, for example
  `(anonymous namespace)::SchemaCanonicalizerPass`.
- `<argument>` is the command-line name (`cse`, `schema-canonicalize`), and
  it's what the UI shows. The legacy form puts it in parentheses instead.
- `<options>` are the pass options, with whitespace collapsed.
- `Failed` marks the pass that failed (`-mlir-print-ir-after-failure`).
- The trailing anchor names the op the pass ran on, when the driver prints it.

### IR body

After a header comes exactly one top-level operation (a `module`, or with
nested passes a `func.func` or other op), optionally with attribute and type
alias lines (`#map = ...`, `!t = ...`) before or after it. The body ends at
the line where braces balance again, ignoring braces in strings and `//`
comments. The first non-alias line gives the event's **root** op and symbol.

### Diagnostics

```
<file>:<line>[:<col>]: (error|warning|note|remark): <message>
<continuation lines until a blank line, the next diagnostic, or a header>
```

Diagnostics printed before an "After" dump are attached to that event, since
the pass that emitted them is the one whose dump follows. Diagnostics after
the last dump belong to a pass that printed nothing (usually a failure
without `-mlir-print-ir-after-failure`) and get `eventIndex: -1`.

### Reports

- **Timing:** the `-mlir-timing` block, from the `===---===` rule and
  `... Execution time report ...` title to the first blank line after the
  rows, or a JSON array whose rows start with `{"user": {"duration"` /
  `{"wall": {"duration"`.
- **Memory:** a BSD `time -l` block (`real … user … sys` followed by
  `<number> <stat name>` lines) or a GNU `time -v` block (`Command being
timed:` through `Exit status:`).

Report lines are blanked, not removed, so line numbers in the rest of the log
stay valid.

## Parsed structure

`parsePassTrace(text)` returns:

```js
{
  events: [{
    index,        // position in the trace
    phase,        // "before" | "after"
    pass,         // C++ pass name
    argument,     // CLI name, or null
    options,      // "k=v k=v", or null
    failed,       // boolean
    anchor,       // { op, symbol } from the header, or null
    root,         // { op, symbol } from the IR itself, or null
    scope,        // "<op>@<symbol>", the key used to find a baseline
    headerLine,   // 1-based line numbers in the original log
    irLine,
    ir,           // the dump, blank lines trimmed
    diagnostics,  // diagnostics attached to this event
    trailing,     // loose text after the dump
  }],
  diagnostics,    // every diagnostic, each with eventIndex
  preamble,       // loose text before the first dump
  output,         // loose text after the last dump (usually the final module)
  timing,         // parsed -mlir-timing report, or null
  memory,         // { peakBytes, source }, or null
}
```

`baselineFor(events, i)` returns what event `i` is diffed against: the latest
earlier dump with the same scope, or the op cut out of an enclosing module
dump. When nested dumps are newer than the last module dump, the module
baseline is rebuilt by splicing them in (`reconstructed: true`).

## Compatibility

The format belongs to MLIR, not VizMLIR, and it changes between LLVM
releases. The fixtures record the version they were generated with in
`tests/fixtures/traces/MLIR_VERSION`. When a new release changes the header
format, add a case to the `parseHeader` table in `tests/unit/trace.test.js`
rather than replacing the old one, so both keep parsing.
