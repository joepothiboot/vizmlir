#!/usr/bin/env python3
import json
import pathlib
import sys

import cupy as cp
import numpy as np

PTX = pathlib.Path(__file__).resolve().parent.parent / "samples" / "ptx"

EXPECTED = {
    "aos_x_kernel": lambda points: points[:, 0],
    "soa_x_kernel": lambda xs: xs,
    "diff_kernel": lambda x: x[1:] - x[:-1],
    "window_sum_kernel": lambda x: np.array([x[i : i + 32].sum() for i in range(4096)], np.float32),
    "add_bias_kernel": lambda x, bias: x + bias[0],
    "tile_copy_16x16_kernel": lambda x: x,
    "transpose_naive_kernel": lambda x: x.T,
    "transpose_tiled_kernel": lambda x: x.T,
    "transpose_padded_kernel": lambda x: x.T,
}

kernels = json.loads((PTX / "kernels.json").read_text())
failed = []

for k in kernels:
    name = k["name"]
    buffers = [cp.arange(np.prod(shape), dtype=cp.float32).reshape(shape) for shape in k["args"]]
    buffers[-1].fill(-1)
    kernel = cp.RawModule(path=str(PTX / f"{name}.ptx")).get_function(name)
    kernel(tuple(k["grid"]), tuple(k["block"]), tuple(buffers))
    cp.cuda.Device().synchronize()
    *inputs, out = (b.get() for b in buffers)
    ok = np.array_equal(out, EXPECTED[name](*inputs))
    print(f"{'ok  ' if ok else 'FAIL'} {name}")

    if not ok:
        failed.append(name)

print(f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}: {len(kernels) - len(failed)} / {len(kernels)} kernels correct")
sys.exit(1 if failed else 0)
