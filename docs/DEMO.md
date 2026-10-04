# A three-minute tour

Start the app (`npm run dev`). A dialog offers a scenario to start from; close
it, or pick **Source tracing and debugging**. (**Scenario ▾** in the top bar
reopens it.) The sample is a small hand-written trace in the
format of `mlir-opt -mlir-print-ir-after-all -mlir-print-debuginfo`, with a
real `saxpy.mojo` beside it: a kernel that calls `scale(x[i], a)` and adds
`y[i]`. Five passes: inlining, canonicalization, dead-symbol removal, and
lowering to NVVM.

## 1. Source to ops (30 s)

1. Debug mode is already on in this scenario (`d` toggles it). The current IR
   sits above `saxpy.mojo`.
2. Click line 10 (`y[i] = scale(x[i], a) + y[i]`) in the **Source** tab. The
   graph selects the first op from it, and the status line says "op 1 of 5".
   Click again to step through the rest.
3. Each line carries a count of the ops it made.

## 2. One op's life (60 s)

1. Press `.` (continue). The sample comes with two breakpoints: the run stops
   at pass 2, where `func.call @scale` is gone, because the inliner replaced it.
2. Click line 4 until the inlined `arith.mulf` is selected. It marks lines 4
   *and* 10: its location is a call site, `callsite(line 4 at line 10)`.
3. Look at the pass slider: passes 2 and 3 now have lit ticks (inlined, then
   fused) and passes 4 and 5 faint ones (still there, unchanged). Hover the
   slider for the details, or open **History** (`y`) for the full list. It was
   inlined at pass 2 from the callee's `mulf`, then fused into `math.fma` at
   pass 3.
4. Press `op ▶` (or `>`). The run goes to pass 3. The `fma` marks lines 4 and
   10, and the History tab lists the two ops it merged, each one clickable.

## 3. Breakpoints (30 s)

1. Click **Break when…** and add `ops < 17`. A third dot appears on the pass
   slider, at the pass where the module drops below 17 ops.
2. `.` and `,` run forward and back between the dots.
3. Try `live > 1MB`, `appears gpu.launch`, `fails` and `diag` on the other
   samples.

## 4. Descent (60 s)

1. Open the **Descent** tab. Every pass is a layer: first at the top, last at
   the bottom. The lit one is the pass you are on.
2. With the `fma` selected, its path runs up through the layers, dashed where
   ops merged. Press `]` and the camera slides down a layer. Select an op and
   press `op ▶` (or click a step in History) and a pulse follows it down its
   path.
3. Put the caret on a line of the current IR (in the editor) and use the arrow
   keys: that line's ops light up, with the edges into and out of them.
4. Drag to turn, scroll to zoom, click a layer to go to that pass, and
   **Open launch view** to reach the 3D GPU view.

## Using your own IR

Run your pipeline with `-mlir-print-ir-after-all -mlir-print-debuginfo
-mlir-disable-threading`, merge stderr into the log, and use **Open…**. Add
the source files with **Add file…** or by dropping them on the Source tab.
See [trace-format.md](trace-format.md).
