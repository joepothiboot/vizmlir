# Hardware check

VizMLIR's memory verdicts are proven over the IR: for each load and store, how
many 32-byte sectors a warp's request moves, or how many ways a shared-memory
read conflicts. Real hardware runs the PTX that `ptxas` makes from that IR, so
this page checks the predictions against Nsight Compute counters measured on
an actual GPU. VizMLIR itself still never runs code; the measurement is taken
elsewhere and compared here.

**Status: predictions committed, GPU run pending.** No column below is
measured yet.

## What is predicted

`node scripts/predict.mjs` runs the same analysis as the GPU view over
`samples/input/gpu-patterns.mlir` and `samples/input/gpu-transpose.mlir` and
writes [`predictions.json`](predictions.json). It was committed before any GPU
run, so the predictions can't be fitted to the measurement.

Each kernel's prediction is the counter ratio ncu reports: the mean cost of
that kind of access in the kernel.

| Kernel                   | Counter             | Predicted | Why                                          | Measured (T4) |
| ------------------------ | ------------------- | --------: | -------------------------------------------- | ------------- |
| `soa_x_kernel`           | global ld sectors/req |       4 | 32 neighboring floats = 128 B                | pending       |
| `aos_x_kernel`           | global ld sectors/req |       8 | every other float: twice the sectors         | pending       |
| `diff_kernel`            | global ld sectors/req |     4.5 | `in[i]` takes 4, `in[i+1]` straddles into 5  | pending       |
| `window_sum_kernel`      | global ld sectors/req |   4.875 | the warp drifts by 4 B a trip: 1 trip in 8 aligned | pending |
| `add_bias_kernel`        | global ld sectors/req |     2.5 | 4 for `in[i]`, 1 for the broadcast `bias[0]` | pending       |
| `tile_copy_16x16_kernel` | global ld sectors/req |       4 | two half rows of 64 B                        | pending       |
| `transpose_naive_kernel` | global st sectors/req |      32 | each lane writes a different row             | pending       |
| `transpose_tiled_kernel` | shared ld wavefronts/inst | 32  | reading a 32×32 tile down a column           | pending       |
| `transpose_padded_kernel`| shared ld wavefronts/inst |  1  | the 33rd column shifts each row by one bank  | pending       |

Every store and the remaining loads are predicted too (4 sectors, or 1
wavefront); see `predictions.json`.

## Measuring

The counters, as ratios:

| Ratio                     | Numerator / denominator                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| global ld sectors/req     | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`     |
| global st sectors/req     | same with `_st`                                                                                           |
| shared ld wavefronts/inst | `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` / `smsp__inst_executed_op_shared_ld.sum`          |
| shared st wavefronts/inst | same with `_st`                                                                                           |

Wavefronts are used rather than `l1tex__data_bank_conflicts_*`, which counts
more than bank conflicts on some architectures.

## Running it on Colab

The kernels are already built: `samples/ptx/` holds their PTX for `sm_75` and
`kernels.json` their launch sizes, made by
`MLIR_OPT=… node scripts/build-ptx.mjs` from the same IR the predictions read.
The GPU side needs only Python, CuPy and `ncu`; no LLVM. About 10 minutes on a
free T4.

1. In Colab, **Runtime → Change runtime type → T4 GPU**, then run in a cell:

   ```
   !nvidia-smi --query-gpu=name,driver_version --format=csv
   !git clone --depth 1 https://github.com/joepothiboot/vizmlir
   %cd vizmlir
   !python scripts/run_kernels.py
   ```

   `run_kernels.py` launches each kernel once and checks its output against
   NumPy. It must print `9 / 9 kernels correct` before the counters mean
   anything.

2. Find `ncu` with `!which ncu || ls /usr/local/cuda/bin/ncu`. If neither
   exists, `!apt-cache search nsight-compute` and `apt-get install` the
   package it lists. Then measure:

   ```
   !ncu --csv --log-file t4.ncu.csv -k regex:'_kernel$' --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,smsp__inst_executed_op_shared_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,smsp__inst_executed_op_shared_st.sum python scripts/run_kernels.py
   !head -5 t4.ncu.csv
   from google.colab import files; files.download("t4.ncu.csv")
   ```

   If the log says `ERR_NVGPUCTRPERM`, this host doesn't allow counter
   access. Try a Kaggle notebook (T4) with the same commands.

3. Back on the Mac:

   ```bash
   cp ~/Downloads/t4.ncu.csv public/samples/gpu-patterns.t4.ncu.csv
   node scripts/predict.mjs --check public/samples/gpu-patterns.t4.ncu.csv
   ```

   `--check` prints each counter with ✓ or ✗ and exits non-zero unless all
   of them agree within 5%. Copy the measured values into the table above,
   with the GPU and driver from step 1, and list every disagreement here with
   its explanation rather than dropping it.

## Known gaps before the run

- `ptxas` may still rewrite accesses (merge, vectorize, or hoist loads) in
  ways the IR doesn't show. Any such case would show up as a disagreement here.
- Some hosted GPUs refuse counter access (`ERR_NVGPUCTRPERM`). Without
  counters there is no check, only timings.
- The PTX is built with bare-pointer parameters and constants sunk into the
  kernel, unlike the `sm_80` sample traces. The memory accesses are the same
  IR; only how arguments are passed differs.
