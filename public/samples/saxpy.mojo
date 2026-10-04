from gpu import block_dim, block_idx, thread_idx

fn scale(x: Float32, k: Float32) -> Float32:
    return x * k

fn saxpy(a: Float32, x: UnsafePointer[Float32],
         y: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        y[i] = scale(x[i], a) + y[i]
