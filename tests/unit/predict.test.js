import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parseNcuCounters } from "../../src/bench.js";
import { check, INPUTS, predict } from "../../scripts/predict.mjs";

const kernels = INPUTS.flatMap((file) =>
  predict(readFileSync(new URL(`../../${file}`, import.meta.url), "utf8"), file),
);
const counters = Object.fromEntries(kernels.map((k) => [k.kernel, k.counters]));

// Shaped like `ncu --csv --metrics ...` (columns trimmed), values invented.
const NCU = `==PROF== Connected to process 1234
"ID","Kernel Name","Block Size","Grid Size","Metric Name","Metric Unit","Metric Value"
"0","aos_x_kernel","(256, 1, 1)","(16, 1, 1)","l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum","sector","1,024"
"0","aos_x_kernel","(256, 1, 1)","(16, 1, 1)","l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum","request","128"
"0","aos_x_kernel","(256, 1, 1)","(16, 1, 1)","gpu__time_duration.sum","usecond","2.1"
"1","transpose_tiled_kernel","(32, 32, 1)","(32, 32, 1)","l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum","","1,048,576"
"1","transpose_tiled_kernel","(32, 32, 1)","(32, 32, 1)","smsp__inst_executed_op_shared_ld.sum","inst","32,768"
"2","transpose_tiled_kernel","(32, 32, 1)","(32, 32, 1)","l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum","","1,048,576"
"2","transpose_tiled_kernel","(32, 32, 1)","(32, 32, 1)","smsp__inst_executed_op_shared_ld.sum","inst","32,768"
`;

describe("hardware-check predictions", () => {
  it("predicts textbook counters from the IR", () => {
    expect(counters.soa_x_kernel).toEqual({ globalLoad: 4, globalStore: 4 });
    expect(counters.aos_x_kernel.globalLoad).toBe(8);
    expect(counters.diff_kernel.globalLoad).toBe(4.5); // 4 aligned + 5 shifted
    expect(counters.add_bias_kernel.globalLoad).toBe(2.5); // 4 + broadcast 1
    expect(counters.window_sum_kernel.globalLoad).toBe(4.875); // 1 aligned trip in 8
    expect(counters.transpose_naive_kernel.globalStore).toBe(32);
    expect(counters.transpose_tiled_kernel.sharedLoad).toBe(32);
    expect(counters.transpose_padded_kernel.sharedLoad).toBe(1);
  });

  it("reads ncu counters, sums launches, and checks them", () => {
    const measured = parseNcuCounters(NCU);
    expect(measured.get("aos_x_kernel")).toMatchObject({ globalLoad: 8, globalStore: null });
    expect(measured.get("transpose_tiled_kernel").sharedLoad).toBe(32);
    const rows = check(kernels, measured);
    expect(rows.find((r) => r.kernel === "aos_x_kernel" && r.counter === "globalLoad").agrees).toBe(true);
    expect(rows.find((r) => r.kernel === "soa_x_kernel").agrees).toBe(null); // not measured
  });

  it("rejects CSV without counter metrics", () => {
    expect(() => parseNcuCounters('"Kernel Name","Metric Name","Metric Value"\n"k","gpu__time_duration.sum","1"\n')).toThrow(/none of the counter metrics/);
  });
});
