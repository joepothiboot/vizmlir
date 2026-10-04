# VizMLIR test plan

This plan covers VizMLIR 0.3.0. It starts at what a user sees (the view) and
works down through the app controller, the renderer, the pure JavaScript
modules, and the JS↔WASM bridge, ending at the Rust lexer and parser.

The project has no automated tests today. The only gate is `npm run build` in
the GitHub Pages workflow, so each layer below lists the tool to add, the
behaviours to cover, and the cases that matter most.

## Layers at a glance

| #   | Layer             | Code                                                                | Tool                 | Runs in                                             |
| --- | ----------------- | ------------------------------------------------------------------- | -------------------- | --------------------------------------------------- |
| L1  | View / end-to-end | `index.html`, whole app                                             | Playwright           | Chromium + Firefox + WebKit, against `vite preview` |
| L2  | App controller    | `src/main.js`                                                       | Playwright (via L1)  | browser                                             |
| L3  | Canvas renderer   | `src/render/canvas-renderer.js`                                     | Vitest + stub canvas | Node (jsdom)                                        |
| L4  | Pure JS modules   | `src/ir/diff.js`, `src/trace/trace.js`, `src/app/mlir-highlight.js`              | Vitest               | Node                                                |
| L5  | Bridge and ABI    | `src/ir/bridge.js`, `views.js`, `abi.js` + built `mlir_core.wasm` | Vitest               | Node (real WASM)                                    |
| L6  | Rust engine       | `wasm/src/*.rs`                                                     | `cargo test`         | native host                                         |

Priorities: **P0** blocks a release, **P1** should be covered, **P2** nice to have.

Items marked **⚠ suspected bug** come from reading the code. Each needs a
failing test written first to confirm it.

---

## L1 — View / end-to-end (Playwright)

Run against `npm run build && npm run preview`, so the `/vizmlir/` base path and
the WASM fetch behave as they do on GitHub Pages.

### 1.1 Boot and routing

| ID   | Case                                 | Expect                                                                                                                                   | P   |
| ---- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | --- |
| V-01 | Open `/vizmlir/`                     | Workspace visible, docs hidden, both editors pre-filled with the sample (Current has `linalg.fill_relu`), graph drawn, no console errors | P0  |
| V-02 | WASM loads                           | `mlir_core.wasm` request returns 200 with `application/wasm`; there is no ABI-mismatch or bad-magic error                                | P0  |
| V-03 | WASM served with the wrong MIME type | `instantiateStreaming` fallback still loads the engine                                                                                   | P1  |
| V-04 | Click **Docs**, then **Workspace**   | Views swap; `aria-current="page"` moves to the active link                                                                               | P0  |
| V-06 | Browser back/forward between routes  | View follows the hash                                                                                                                    | P2  |

### 1.2 Editors and syntax highlighting

| ID   | Case                                                 | Expect                                                                                      | P   |
| ---- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------- | --- |
| V-10 | Type in Current                                      | Highlight layer updates on every keystroke; caret stays aligned with the highlighted text   | P0  |
| V-11 | Scroll a long editor horizontally and vertically     | Highlight layer scrolls in sync with the textarea                                           | P1  |
| V-12 | Paste `<script>alert(1)</script>` or `<img onerror>` | Rendered as text; no script runs (highlight writes through `innerHTML`)                     | P0  |
| V-13 | Token colours                                        | Comment, string, `%ssa`, `@sym`, keyword, type, number, and dialect op each get their class | P2  |
| V-14 | Docs sample editor                                   | Read-only and highlighted                                                                   | P2  |

### 1.3 Graph canvas

| ID   | Case                                                                      | Expect                                                                                                                                                                          | P   |
| ---- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| V-20 | Initial render                                                            | Screenshot of the sample graph matches the golden image (per browser, with tolerance)                                                                                           | P1  |
| V-21 | Drag                                                                      | Graph pans; releasing after a drag does **not** change the selection                                                                                                            | P0  |
| V-22 | Wheel                                                                     | Zoom is centred on the cursor and clamped to [0.05, 4]                                                                                                                          | P1  |
| V-23 | Click a node                                                              | Status `#detail` shows `#i · label · parent p`; node outline turns white                                                                                                        | P0  |
| V-24 | Click empty space                                                         | Selection clears; detail shows `—`                                                                                                                                              | P1  |
| V-25 | **Fit** after pan/zoom                                                    | Whole graph back in view                                                                                                                                                        | P1  |
| V-26 | Zoom out below 0.42                                                       | Labels hidden; boxes still drawn                                                                                                                                                | P2  |
| V-27 | Empty or failed Current                                                   | Canvas shows "No module loaded"                                                                                                                                                 | P1  |
| V-28 | HiDPI (`deviceScaleFactor: 2`)                                            | Crisp rendering; hit-testing still lands on the clicked node                                                                                                                    | P1  |
| V-29 | Resize the window                                                         | Canvas backing store is resized; no stretching                                                                                                                                  | P1  |
| V-30 | **⚠ suspected bug:** drag the resizable Baseline/Current pane wider       | The canvas only resizes on `window.resize`, not when the pane changes size, so the graph stretches or blurs until the window is resized. Fix idea: `ResizeObserver` on `#stage` | P1  |
| V-31 | **⚠ suspected bug:** resize the window while on Docs, then open Workspace | Canvas was measured while hidden (1×1) and is never re-measured, so the graph is blank or blurry                                                                                | P1  |

### 1.4 Diff pane and status bar

| ID   | Case                                 | Expect                                                                                                  | P   |
| ---- | ------------------------------------ | ------------------------------------------------------------------------------------------------------- | --- |
| V-40 | Default sample                       | Summary `1 change(s): +0  -0  ~1`; row `~ linalg.fill -> linalg.fill_relu` in the "changed" colour      | P0  |
| V-41 | Baseline equals Current              | "No structural changes"; list empty                                                                     | P0  |
| V-42 | Baseline empty                       | Every op is listed as added                                                                             | P1  |
| V-43 | Add, remove, and rename ops          | `+` / `-` / `~` rows with the right counts                                                              | P0  |
| V-44 | Rename SSA values only (`%0` → `%x`) | No changes reported                                                                                     | P0  |
| V-45 | Status line                          | `N nodes · M edges · t ms`; a use of an undefined value adds `· 1 warning(s): undefined SSA value "%x"` | P1  |
| V-46 | Typing debounce                      | Parsing runs about 140 ms after the last keystroke, not on every key                                    | P2  |

### 1.5 Error states

| ID   | Case                                                        | Expect                                                                                                                                                                                                               | P   |
| ---- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| V-50 | Clear Current                                               | `current error: empty input`; canvas shows the placeholder                                                                                                                                                           | P0  |
| V-51 | Current over 4 MiB                                          | Error message names the size limit. **⚠ suspected bug:** `bridge.parse` returns `TOO_LARGE` without writing the header, so `engine.statusText` shows the _previous_ status and the message reads `current error: ok` | P0  |
| V-52 | Baseline over 4 MiB                                         | `baseline error: …` with the correct text (same bug as V-51)                                                                                                                                                         | P1  |
| V-53 | Baseline fails but Current is fine                          | Status shows the baseline error; the graph does not show corrupt data from the half-finished baseline parse                                                                                                          | P1  |
| V-54 | Malformed MLIR (unbalanced `}`, garbage text)               | No crash or console error; something reasonable renders                                                                                                                                                              | P0  |
| V-55 | Non-ASCII labels and symbols (`@"héllo"`, emoji in strings) | Parses; labels decode correctly                                                                                                                                                                                      | P1  |

### 1.6 Pass-trace workflow

Fixtures come from real `mlir-opt` runs (see [Fixtures](#fixtures)).

| ID   | Case                                                                      | Expect                                                                                                                                                     | P   |
| ---- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| V-60 | **Open…** an after-all trace with module scope                            | Pass `<select>` appears; one option per dump, labelled `n. After <arg> · op @sym`                                                                          | P0  |
| V-61 | Paste the same trace into Current                                         | Same result as V-60                                                                                                                                        | P0  |
| V-62 | Pick a pass                                                               | Baseline gets the previous snapshot; titles read `Baseline · from #k` and `Current · #n …`; status begins `pass n/N · trace line L`                        | P0  |
| V-63 | Trace with a failed pass                                                  | That pass is auto-selected; option shows `✗ failed`; status shows `✗ FAILED`                                                                               | P0  |
| V-64 | Diagnostics in the trace                                                  | Overlay lists `file:line:col: severity: msg` coloured by severity; hovering shows the detail snippet                                                       | P1  |
| V-65 | Diagnostics after the last dump                                           | Shown on the last pass with `(after last dump)`                                                                                                            | P1  |
| V-66 | Trace without module scope (nested `func.func` dumps)                     | Module baseline title reads `rebuilt through #k`, and the rebuilt IR parses                                                                                | P0  |
| V-67 | Before-all trace; legacy `// *** IR Dump … ***` header                    | Both load                                                                                                                                                  | P1  |
| V-68 | Log with no dumps opened via **Open…**                                    | Treated as plain MLIR                                                                                                                                      | P1  |
| V-69 | After a trace, **Load in workspace** from Docs or open a plain `.mlir`    | Pass select hidden; titles, trace note, and diagnostics cleared                                                                                            | P0  |
| V-70 | **⚠ suspected bug:** after a trace, hand-edit Current into non-trace MLIR | `trace`/`traceNote` are never cleared, so the status keeps the stale `pass n/N` prefix and the select stays visible. Decide what should happen and test it | P1  |
| V-71 | Opening the same file twice                                               | Second open still fires (input value is reset)                                                                                                             | P2  |

### 1.7 Docs view

| ID   | Case                  | Expect                                                                                                                                         | P   |
| ---- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| V-80 | **Load in workspace** | Switches to Workspace with the sample loaded and compared                                                                                      | P0  |
| V-81 | External links        | `target=_blank rel=noreferrer`                                                                                                                 | P2  |
| V-82 | Legend colours        | Match `KIND_STYLE` strokes in `abi.js`                                                                                                         | P2  |
| V-84 | Contents follows      | Scrolling highlights the chapter in view in the side list (the last one at the bottom); on phones the chip row stays pinned and scrolls to it  | P2  |

### 1.8 Responsive and accessibility

| ID   | Case                         | Expect                                                                         | P   |
| ---- | ---------------------------- | ------------------------------------------------------------------------------ | --- |
| V-90 | Width 680 px and 900 px      | Stacked or narrow layouts; no horizontal page scroll; canvas has non-zero size | P1  |
| V-91 | Keyboard only                | Every header control is reachable; the pass select works from the keyboard     | P1  |
| V-92 | axe-core scan of both routes | No serious violations (e.g. the canvas has no text alternative)                | P2  |

---

## L2 — App controller (`src/main.js`)

`main.js` runs top-level `await` and touches the DOM at import time, so it is
covered through L1 rather than unit tests. The L1 cases above should also check
these controller rules:

- `run()` sends trace-shaped Current text to `loadTrace` and never parses it as MLIR.
- The baseline is copied (`copySnapshot`) **before** Current is parsed, because
  both parses share WASM memory. A regression here shows up as "no changes"
  every time. Add a test with different Baseline and Current that asserts a
  non-empty diff.
- `selectEvent` cancels the pending debounce so the synthetic `input` events do
  not cause a second parse.
- `clearTrace` resets every piece of trace UI state (V-69).

**Optional refactor:** move `run`, `loadTrace`, and `selectEvent` into a module
that receives the engine, renderer, and elements as arguments. They could then
be unit-tested in jsdom with a fake engine.

---

## L3 — Canvas renderer (Vitest + jsdom)

Build it with a stub canvas (`getBoundingClientRect` returns a fixed rect;
`getContext` returns a recording mock). Feed it hand-built snapshot objects
(`xywh`, `edges`, `bounds`, `kindOf`, `labelOf`).

| ID   | Case                                                                                                                                    | P   |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------- | --- |
| R-01 | `fit()` centres the bounds, caps scale at 2, floors it at 0.05, and ignores empty snapshots                                             | P1  |
| R-02 | `hitTest` inverts the camera transform; overlapping nodes return the topmost (highest index); a miss returns -1; no snapshot returns -1 | P0  |
| R-03 | Wheel zoom keeps the world point under the cursor fixed                                                                                 | P1  |
| R-04 | Pointer down, then up with less than 4 px of movement selects and fires `onSelect`; 4 px or more pans without selecting                 | P0  |
| R-05 | `requestDraw` batches several calls into one rAF                                                                                        | P2  |
| R-06 | Culling: offscreen nodes and edges issue no draw calls; nodes that cross the edge of the view are still drawn                           | P1  |
| R-07 | Unknown `kind` falls back to the OP style                                                                                               | P2  |
| R-08 | `resize()` sets the backing store to rect × DPR with a minimum of 1 px                                                                  | P1  |

---

## L4 — Pure JS modules (Vitest)

These modules have no DOM or WASM dependency and give the most coverage for the
least effort. **Write these first.**

### 4.1 `mlir-highlight.js`

- Escapes all of `& < > " '`, both inside and outside tokens (the XSS guard for V-12).
- Classifies each token: comment, string (including escaped quotes), SSA,
  symbol, keyword, type (`i1`, `i64`, `f16`, `bf16`, `tensor`), number
  (`1.5e-3`), dialect op (`arith.addi`), plain identifier.
- Round trip: removing the tags and unescaping gives back the input, plus the trailing `\n`.
- `bindHighlighting` syncs on `input` and `scroll` and toggles `focused`.

### 4.2 `diff.js`

| ID   | Case                                                                                                                                                                                                                                 | P   |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --- |
| D-01 | `normalizeLabel` replaces every `%name` (including `%arg0`, `%0#1`, `%a.b`, `%-x`) with `%?`                                                                                                                                         | P0  |
| D-02 | Identical snapshots → `[]`; `before = null` → all added                                                                                                                                                                              | P0  |
| D-03 | Exact signature match wins over a kind+parent "changed" match                                                                                                                                                                        | P0  |
| D-04 | Duplicate ops (two `arith.addi`) pair up one-to-one; one extra shows as added                                                                                                                                                        | P0  |
| D-05 | Changed pairing needs the same kind **and** the same parent index. Pin the current behaviour when inserting a top-level op shifts later parent indices (a known weakness of index-based parents)                                     | P1  |
| D-06 | `copySnapshot` output diffs the same as a live snapshot                                                                                                                                                                              | P0  |
| D-07 | Changes to operands, types, or attributes only are **not** detected, because labels hold only the op name and symbol. Write this as a test that documents current behaviour, so a future improvement shows up as a deliberate change | P1  |
| D-08 | Performance: 5k ops against 5k ops finish in under 200 ms (the `findIndex` loop is O(n²))                                                                                                                                            | P2  |

### 4.3 `trace.js`

Parsing pass headers is where the tool is most likely to break as LLVM versions
change. Cover it with table-driven tests.

**`parseHeader`**

| Header body                                                                     | phase    | pass            | argument              | options                    | failed | anchor          |
| ------------------------------------------------------------------------------- | -------- | --------------- | --------------------- | -------------------------- | ------ | --------------- |
| `After CSEPass: cse ('builtin.module' operation)`                               | after    | CSEPass         | cse                   | –                          | no     | builtin.module  |
| `After CSE (cse) ('func.func' operation: @f)`                                   | after    | CSE             | cse                   | –                          | no     | func.func @f    |
| `After InterpreterPass Failed: transform-interpreter{debug-payload-root-tag=x}` | after    | InterpreterPass | transform-interpreter | `debug-payload-root-tag=x` | yes    | –               |
| `After CSE Failed (cse)`                                                        | after    | CSE             | cse                   | –                          | yes    | –               |
| `Before Canonicalizer` (no argument)                                            | before   | Canonicalizer   | –                     | –                          | no     | –               |
| `After Foo Failed`                                                              | after    | Foo             | –                     | –                          | yes    | –               |
| Quoted symbol `@"my fn"` in the anchor                                          |          |                 |                       |                            |        | symbol unquoted |
| Options with nested whitespace and newlines                                     |          |                 |                       | spaces collapsed           |        |                 |
| Not a header, or `Middle Foo`                                                   | → `null` |                 |                       |                            |        |                 |

**`isPassTrace`**: finds a header past line 1; ignores headers beyond 64 KiB or
2000 lines (confirm this is intended); handles CRLF.

**`parsePassTrace`**

- Splits IR per event; `headerLine` and `irLine` are 1-based and correct.
- Keeps `#map = …` / `!t = …` aliases around a dump with the dump, not as loose lines.
- `findIrEnd`/`braceDelta`: braces inside strings and after `//` comments are
  ignored; escaped quotes inside strings are handled.
- Diagnostics: parses the location with and without a column; continuation
  lines end at a blank line, the next diagnostic, or a header. A diagnostic
  _before_ an `After` dump attaches to that event; one _before_ a `Before` dump
  stays pending until the next `After`; any left at the end get
  `eventIndex: -1`.
- Preamble text before the first header goes to `preamble`; text after the
  last dump goes to `output` and is removed from the last event's `trailing`.
- Empty input, headers only, and a dump truncated mid-region all return without throwing.

**`baselineFor` / module rebuild**

- Same-scope earlier dump → `reconstructed: false`.
- A nested func dump with no earlier func dump pulls the function out of an
  earlier module dump through `findSymbolOp`.
- A module dump after several nested dumps splices the latest version of each
  function into the last module dump, keeping indentation and dropping aliases
  → `reconstructed: true`.
- No module dump at all → built from `module {\n}`.
- A function that is new (not in the base module) is appended before the closing `}`.
- Symbols with regex metacharacters (`@a.b$c`) and quoted symbols.
- `func.func private @f`, `%x = op @sym`, `llvm.func @f(` — the `findSymbolOp` start-pattern variants.
- **Property check:** every rebuilt baseline from the fixture traces parses with
  WASM status OK (links L4 and L5).

**`describeEvent`**: prefers the argument over the pass name, and the anchor over the root.

---

## L5 — Bridge and ABI (Vitest, real WASM in Node)

Load `public/mlir_core.wasm` with `WebAssembly.instantiate(readFileSync(...))`
and wrap it in `new MlirEngine(instance)`. These tests need `npm run wasm` to
have run first.

### 5.1 ABI sync (P0, cheap, catches the most common contributor mistake)

Read `wasm/src/abi.rs` as text, pull out every `pub const`, and compare with
`src/ir/abi.js`:

- `ABI_MAGIC`, `ABI_VERSION`, all `HDR_*`, `STRIDE_*`, `STATUS_*`, `KIND_*`,
  `DIAG_*`, `NONE`.
- The `STATUS_TEXT[4]` wording matches `INPUT_CAP` (4 MiB).
- Every `KIND_*` has an entry in `KIND_STYLE`, and every `DIAG_*` has one in `DIAG_CODE`.

### 5.2 Engine contract

| ID   | Case                                                                                                                                                                                                           | P   |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| B-01 | The constructor throws a clear error on an ABI version mismatch and on bad magic (use a fake `instance.exports`)                                                                                               | P0  |
| B-02 | `parse('')` → EMPTY; the header counts are zeroed                                                                                                                                                              | P0  |
| B-03 | `parse(sample)` → OK; `snapshot()` gives node and edge counts, kinds, labels, parents that match a golden JSON                                                                                                 | P0  |
| B-04 | Input larger than `inputCapacity` → `TOO_LARGE`. The size is counted in **UTF-8 bytes**, not characters (test with multi-byte text just under and just over the limit). Also assert `statusText` agrees (V-51) | P0  |
| B-05 | Reusing the engine: parse A, parse B, parse A again gives a snapshot identical to the first A (no leftover state from the interner or AST)                                                                     | P0  |
| B-06 | Snapshot views go stale after the next `parse`. A `copySnapshot` taken earlier is unaffected (guards the L2 ordering rule)                                                                                     | P0  |
| B-07 | Memory growth: parse a large module that forces `memory.grow`; `MemoryViews.refresh` picks up the new buffer and no detached-buffer errors occur                                                               | P1  |
| B-08 | `layout({...})` with custom sizes rewrites `xywh` and `bounds`                                                                                                                                                 | P2  |
| B-09 | `MemoryViews`: misaligned pointers and out-of-range reads throw; zero-length reads return the shared empty arrays                                                                                              | P1  |
| B-10 | `diagnostics()` decodes code, line, symbol, and message; unknown codes → "unknown diagnostic"                                                                                                                  | P1  |
| B-11 | Every reported pointer is 4-byte aligned; every `labelOf` span lies inside `STRINGS_LEN`                                                                                                                       | P0  |

---

## L6 — Rust engine (`cargo test`)

Put unit tests in `#[cfg(test)] mod tests` inside each file. `cargo test`
builds the library's test harness for a `cdylib` crate. Add `"rlib"` to
`crate-type` only if integration tests under `wasm/tests/` are wanted.

**Native caveat:** `mlir_header_ptr`, `mlir_input_ptr`, and `Arena::addr` cast
pointers to `u32`, which truncates on 64-bit hosts. Test the exported
`extern "C"` functions and pointer layout **in WASM (L5)**. Natively, test
`Lexer`, `ast::parse`, `Interner`, `Arena`, and the layout maths. The global
`static mut ENGINE` also means native tests must never run exported functions
in parallel.

### 6.1 Lexer

- Each token kind on a minimal input; `->` becomes `Arrow`, but a lone `-` and
  `-1` lex as `Ident`.
- Line counting across `\n`, `\r\n`, and newlines inside strings.
- Strings with escapes, an unterminated string at EOF, and a trailing `\` at
  EOF: no panic, `end <= len`.
- Non-ASCII bytes lex as `Other` one byte at a time. Check that token slices
  never cut through a UTF-8 character, since `ast.rs` indexes `&src[start..end]`
  and a mid-character slice **panics** (and `panic = "abort"` in WASM). Inputs:
  `%é`, `@日本`, `^bb€`.
- Numbers `0x1F`, `1.0e-5`, `.5` — pin the current splitting.
- The final token is always `Eof`.

### 6.2 Parser (`ast::parse`)

Assert on a compact dump such as `kind label parent depth` per node, plus the
edge list, compared against golden text files (`insta` snapshots work well).

| ID   | Input                                                                                                            | Check                                                                                                                                                                           | P   |
| ---- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| A-01 | The README / `SAMPLE` matmul                                                                                     | Exact node, edge, and parent structure                                                                                                                                          | P0  |
| A-02 | `classify`: `func.func`, `llvm.func`, `memref.global`, `*.return`, `cf.br`, `cf.cond_br`, `scf.yield`, other ops | Right `KIND_*`                                                                                                                                                                  | P0  |
| A-03 | Blocks `^bb0(%a: i32):`                                                                                          | BLOCK node; block arguments count as definitions; a use-before-def inside a block is not reported                                                                               | P0  |
| A-04 | Multi-result `%a:2 = …`, `%a, %b = …`                                                                            | `(2×)` label suffix, both values defined                                                                                                                                        | P1  |
| A-05 | Use of an undefined `%x`                                                                                         | One `DIAG_UNDEF_SSA` with the right line and symbol                                                                                                                             | P0  |
| A-06 | An extra `}`                                                                                                     | `DIAG_UNBALANCED`, no panic; a missing `}` gives no panic                                                                                                                       | P0  |
| A-07 | `} loc(#loc3)` and `#loc = loc(...)`, `#map = affine_map<…>`, `!t = …`                                           | No nodes created (regression test for the 0.3.0 fix)                                                                                                                            | P0  |
| A-08 | `scf.if … { } else { }`                                                                                          | Pin what `else` becomes (currently an OP node called `else` that opens a region)                                                                                                | P1  |
| A-09 | Generic form `"scf.for"(%a) ({ ^bb0: … }) : …`                                                                   | Pin current behaviour: a region inside parentheses does not open a region and is collapsed into one statement                                                                   | P1  |
| A-10 | `affine.if #set` / integer sets using `>=`, and `arith.cmpi sge`                                                 | `Lt`/`Gt` depth tracking is not thrown off by `>=` or `->`                                                                                                                      | P1  |
| A-11 | Braces inside attribute dicts `{foo = 1}` on the same line                                                       | Does not open a region                                                                                                                                                          | P0  |
| A-12 | Parent → child fallback edge                                                                                     | Added only when the node has no data predecessor. Note the check only looks at the last 8 edges; build a case where an op with more than 8 operands gets a spurious parent edge | P2  |
| A-13 | Deep nesting (depth 100)                                                                                         | No overflow; layout `rows` grows past 64 columns                                                                                                                                | P1  |
| A-14 | `parse` called twice on one `Ast`/`Interner`                                                                     | Second result is independent of the first                                                                                                                                       | P0  |

### 6.3 Layout (`mlir_layout` maths)

Consider pulling this into a pure function `layout(&Ast, &Interner, params) -> (boxes, bounds)` so it can be tested natively.

- Column = depth; row = running count per column; node width is at least `node_w` and grows with label length.
- Bounds enclose every box.
- **Overlap check:** a label longer than about 21 characters makes the node
  wider than `node_w + col_gap`, so it runs into the next column. Pin this or
  fix it (e.g. per-column max width).

### 6.4 Arena and Interner

- `alloc` rounds up to the requested alignment, sets `oom` when the arena
  overflows, and `reset` clears it; `checked_add` overflow → `None`.
- `intern` deduplicates; spans point into `pool`; `clear` resets IDs to 0; an
  unknown ID's `span` → `(0,0)`.

### 6.5 Fuzzing (P2)

Use `cargo fuzz` on `ast::parse(&str)`: it must not panic on any valid UTF-8
input. Seed it with the fixture corpus. Because a panic aborts the WASM module
and the page goes dead until reload, fuzzing gives a lot of value here.

---

## Cross-cutting

### Fixtures

Add `tests/fixtures/` and share it across layers:

- `mlir/` — valid: sample matmul, `scf.for`/`scf.if`, `cf` blocks,
  multi-result ops, generic form, `llvm` dialect, aliases plus `loc`, unicode
  symbols. Malformed: unbalanced braces, truncated file, binary garbage.
- `traces/` — generated with `mlir-opt` at a pinned LLVM version:
  after-all with module scope, after-all without module scope (nested funcs),
  before-all, a failing pass with diagnostics, a diagnostic after the last
  dump, a legacy `*** IR Dump ***` header, CRLF line endings.
- Record the LLVM version and command line for each trace in `traces/README.md`
  so the fixtures can be regenerated when the header format changes.

### Performance budgets (P2)

- Parse plus layout of a 1 MiB module: under 50 ms in Node WASM.
- Diff of 5k ops: under 200 ms.
- Pan/zoom a 10k-node graph: frame time under 16 ms (Playwright trace).

### CI

Add a `test` job to `.github/workflows/deploy-pages.yml` and make `deploy`
depend on it:

1. `cargo fmt --check`, `cargo clippy`, `cargo test` (L6)
2. `npm run wasm`, then `vitest run` (L3–L5)
3. `vite build && vite preview`, then `playwright test` (L1–L2)

Suggested `package.json` scripts: `test:rust`, `test:unit`, `test:e2e`, and
`test` to run all three.

---

## Rollout order

Build from the cheapest checks to the most expensive, so each step protects the next:

1. **L5.1 ABI sync** and **L4 pure-module tests** — no browser needed, fast,
   and they cover the trace and diff logic that changes most often.
2. **L6 parser golden tests and the UTF-8 panic check.**
3. **L5 engine contract** against the real WASM, including the `TOO_LARGE`
   status bug.
4. **L1 P0 smoke tests** (boot, edit, diff, trace load, error states).
5. L3 renderer, the remaining L1 P1 cases, visual goldens, then fuzzing and
   performance budgets.

## Suspected bugs to confirm first

These come from reading the code. Write a failing test for each before fixing it.

| Ref         | Where                  | Issue                                                                                                                                                                                                                                                                                                      |
| ----------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V-51 / B-04 | `bridge.js` `parse`    | `TOO_LARGE` is returned without updating the header, so `statusText` is stale ("current error: ok")                                                                                                                                                                                                        |
| 6.1         | `lexer.rs` + `ast.rs`  | **Reproduced.** The lexer emits one `Other` token per byte of a non-ASCII character. When such a token starts a statement, `text()` at `ast.rs:65` slices through the character and panics (`byte index 6 is not a char boundary`). Inputs: `%x = é` or `module {\n  é\n}`. In WASM this aborts the module |
| V-30, V-31  | `canvas-renderer.js`   | Canvas does not resize when the pane is resized or when it was measured while hidden                                                                                                                                                                                                                       |
| V-70        | `main.js`              | Trace state is not cleared when Current is edited into plain MLIR                                                                                                                                                                                                                                          |
| 6.3         | `lib.rs` `mlir_layout` | Long labels overlap the next column                                                                                                                                                                                                                                                                        |
| A-12        | `ast.rs`               | The parent-edge check only looks at the last 8 edges                                                                                                                                                                                                                                                       |
| 4.1         | `mlir-highlight.js`    | **Reproduced** (`it.fails` in `tests/unit/mlir-highlight.test.js`). `func` comes before `func\.func` in `tokenPattern`, so `func.func` is highlighted as `func` `.` `func`                                                                                                                                 |

Known bugs are kept as `it.fails` tests so the suite stays green. When a fix
lands, the test starts passing, Vitest reports it as a failure, and it should
then become a normal `it`.

## Progress

- [x] **Step 1:** Vitest set up (`npm test`); tests under `tests/unit/` for
      the ABI sync check (L5.1), `diff.js`, `trace.js`, and `mlir-highlight.js`
      (L4). Real `mlir-opt` trace fixtures are in `tests/fixtures/traces/`;
      regenerate them with `generate.sh`.
- [ ] Step 2: Rust parser golden tests and UTF-8 panic test (L6)
- [ ] Step 3: Engine contract against the real WASM (L5)
- [ ] Step 4: Playwright P0 smoke tests (L1)
- [ ] Step 5: Renderer, remaining L1, visual goldens, fuzzing, performance
