# Benchmark format

VizMLIR can attach measured kernel times to the symbols in a pass trace, so
each kernel's time appears next to the passes that created, changed, or
lowered it. It does not run anything: you import results from a profiler or
your own harness. This document describes what the importer (`src/bench.js`)
accepts and how it matches a kernel to a symbol.

## Try it with mock data

**Samples → GPU kernels + benchmarks (mock)** opens a real `mlir-opt` trace
of two kernels (outlined, lowered to NVVM, serialized to PTX) with a baseline
and a current benchmark file, and opens the Symbol history. The timings in
`public/samples/gpu-kernels.*.mock.csv` are invented to show the feature and
were not measured on any GPU; the dialog marks them as mock data. They follow
the Nsight Systems CSV shape below, and their `# MOCK DATA` comment lines are
skipped like any text before the header.

## Importing

Open a pass trace, press **h** (Symbol history), and choose **Import
current…**, or run "Import current kernel benchmarks…" from ⌘K. The results
are kept while the trace is open (including live reloads of a watched file)
and are saved with the session. Opening a different trace file clears them.

## Comparing two runs

Import a second file with **Import baseline…** to compare two runs of the same
pipeline, for example before and after a driver, GPU, or launch-parameter
change. The two files can come from different tools. The table then shows the
baseline and current time per call and the change for every symbol measured in
either run, largest slowdown first; symbols measured in one run only follow.
**Only kernels that changed by more than N%** (5% by default) hides the rest;
a symbol measured in one run only always counts as changed.

Both runs are matched against the one trace that is open. To compare two
different pipelines (with and without a pass), open each trace in turn; a
side-by-side view of two traces is not supported yet.

## Accepted input

### Nsight Systems

The kernel summary report, as CSV:

```bash
nsys profile -o report ./app
nsys stats --report cuda_gpu_kern_sum --format csv report.nsys-rep > kernels.csv
```

The progress lines before the header are skipped. The time per call is the
`Avg (ns)` column and the call count is `Instances`.

### Nsight Compute

Raw metrics as CSV, one row per launch and metric:

```bash
ncu --csv --metrics gpu__time_duration.sum ./app > kernels.csv
```

Rows whose `Metric Name` is `gpu__time_duration.sum` are kept, with the unit
from `Metric Unit` (`nsecond`, `usecond`, `msecond`, `second`). Launches of the
same kernel are averaged.

### Google Benchmark

`--benchmark_format=json` (or `--benchmark_out=results.json`). Each run's
`real_time` is read in its `time_unit`, weighted by `iterations`. The
`stddev`, `median` and `cv` aggregate rows are ignored. The benchmark `name`
has to match a symbol, so name benchmarks after the kernel they time or add a
`symbol` field.

### Generic CSV or JSON

For your own harness, write one row per kernel (or per run: rows with the same
name are averaged):

```csv
kernel,symbol,time_us,calls
main_kernel,@main_kernel::@main_kernel,45.6,4
matmul_kernel,,120.3,1
```

```json
{
  "kernels": [{ "kernel": "main_kernel", "time_us": 45.6, "calls": 4 }]
}
```

JSON may be an array of these objects or an object holding one under
`kernels`, `results` or `benchmarks`. Its fields are read the same way as CSV
columns:

| Column      | Accepted headers (case-insensitive)                                                                                                             | Required |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| Kernel name | `kernel`, `name`, `kernel name`, `function`, `func`, `demangled name`                                                                           | yes      |
| Time        | `avg`, `mean`, `median`, `time`, `duration`, `latency`, `runtime`, `elapsed`, `real_time`; or `total` / `total time`, which is divided by calls | yes      |
| Unit        | in the time header: `time_us`, `Avg (ns)`, `duration [ms]`; or a `unit` / `time_unit` column                                                    | yes      |
| Calls       | `calls`, `instances`, `count`, `launches`, `invocations`, `iterations`                                                                          | no       |
| Symbol      | `symbol`, as `@module::@kernel` (the `@` is optional)                                                                                           | no       |

Units: `ns`, `us`, `µs`, `ms`, `s`, and ncu's `nsecond`, `usecond`,
`msecond`, `second`. A time column without a unit is rejected rather than
guessed. When a file has several time columns, a per-call time (avg, mean)
wins over a median, then a plain time, then a total.

## Matching kernels to symbols

Symbols are named as in `gpu.launch_func`: `@main_kernel::@main_kernel` is
the `gpu.func` (later `llvm.func`) `@main_kernel` inside `gpu.module
@main_kernel`. For each kernel the importer tries, in order:

1. the `symbol` column, when given, as an exact symbol path;
2. the kernel name as given, against each symbol's own name (the last part of
   its path);
3. the name without its return type, template arguments, argument list and
   namespaces: `void ns::reduce<float>(float*, int)` → `reduce`;
4. an Itanium-mangled `<length><name>`: `_Z6reducePfi` contains `6reduce`.

When a name matches a function and the module that holds it, the function
wins. When it still matches more than one symbol (the same kernel name in two
GPU modules), the kernel is listed as ambiguous with its candidates; add a
`symbol` column to pick one. Kernels that match nothing, such as library
kernels from cuBLAS or CUB, are listed under "Benchmark kernels not matched to
a symbol".

Several kernels that match the same symbol are combined into one time per
call, weighted by calls.

## Export

**Download .json** in the Symbol history dialog adds a `benchmark` object to
every measured symbol, with a `baseline` and `current` entry for each run that
measured it (`time_ns` per call, `calls`, and the matched `kernels`), and, when
both did, `delta_ns` and `change` (the relative change, `0.14` for 14%
slower).
