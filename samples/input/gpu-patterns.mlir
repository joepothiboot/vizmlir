// Six small kernels, one memory pattern each, for the GPU view: how a warp's
// 32 threads spread over memory decides how much of what the GPU fetches is
// used. 1-D kernels run 16 blocks of 256 threads over 4096 elements.
//   aos_x            x of {x, y} pairs: every other float (stride 2)
//   soa_x            x kept in its own array: neighbors read neighbors
//   diff             in[i + 1] - in[i]: the shifted read straddles a sector
//   window_sum       a sliding window: each loop trip shifts the warp by one
//   add_bias         every thread reads the same bias[0]
//   tile_copy_16x16  16x16 blocks: a warp covers two half rows
module attributes {gpu.container_module} {
  func.func @aos_x(%points: memref<4096x2xf32>, %out: memref<4096xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c16, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c256, %sy = %c1, %sz = %c1) {
      %base = arith.muli %bx, %c256 : index
      %i = arith.addi %base, %tx : index
      %x = memref.load %points[%i, %c0] : memref<4096x2xf32>
      memref.store %x, %out[%i] : memref<4096xf32>
      gpu.terminator
    }
    return
  }

  func.func @soa_x(%xs: memref<4096xf32>, %out: memref<4096xf32>) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c16, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c256, %sy = %c1, %sz = %c1) {
      %base = arith.muli %bx, %c256 : index
      %i = arith.addi %base, %tx : index
      %x = memref.load %xs[%i] : memref<4096xf32>
      memref.store %x, %out[%i] : memref<4096xf32>
      gpu.terminator
    }
    return
  }

  func.func @diff(%in: memref<4097xf32>, %out: memref<4096xf32>) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c16, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c256, %sy = %c1, %sz = %c1) {
      %base = arith.muli %bx, %c256 : index
      %i = arith.addi %base, %tx : index
      %next = arith.addi %i, %c1 : index
      %a = memref.load %in[%i] : memref<4097xf32>
      %b = memref.load %in[%next] : memref<4097xf32>
      %d = arith.subf %b, %a : f32
      memref.store %d, %out[%i] : memref<4096xf32>
      gpu.terminator
    }
    return
  }

  func.func @window_sum(%in: memref<4128xf32>, %out: memref<4096xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant 0.0 : f32
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c16, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c256, %sy = %c1, %sz = %c1) {
      %base = arith.muli %bx, %c256 : index
      %i = arith.addi %base, %tx : index
      %sum = scf.for %k = %c0 to %c32 step %c1 iter_args(%acc = %zero) -> (f32) {
        %j = arith.addi %i, %k : index
        %v = memref.load %in[%j] : memref<4128xf32>
        %s = arith.addf %acc, %v : f32
        scf.yield %s : f32
      }
      memref.store %sum, %out[%i] : memref<4096xf32>
      gpu.terminator
    }
    return
  }

  func.func @add_bias(%in: memref<4096xf32>, %bias: memref<1xf32>, %out: memref<4096xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c16, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c256, %sy = %c1, %sz = %c1) {
      %base = arith.muli %bx, %c256 : index
      %i = arith.addi %base, %tx : index
      %v = memref.load %in[%i] : memref<4096xf32>
      %b = memref.load %bias[%c0] : memref<1xf32>
      %s = arith.addf %v, %b : f32
      memref.store %s, %out[%i] : memref<4096xf32>
      gpu.terminator
    }
    return
  }

  func.func @tile_copy_16x16(%in: memref<1024x1024xf32>, %out: memref<1024x1024xf32>) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c64, %gy = %c64, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c16, %sy = %c16, %sz = %c1) {
      %x0 = arith.muli %bx, %c16 : index
      %x = arith.addi %x0, %tx : index
      %y0 = arith.muli %by, %c16 : index
      %y = arith.addi %y0, %ty : index
      %v = memref.load %in[%y, %x] : memref<1024x1024xf32>
      memref.store %v, %out[%y, %x] : memref<1024x1024xf32>
      gpu.terminator
    }
    return
  }
}
