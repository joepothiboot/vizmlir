// Transposing a 1024x1024 matrix three ways, for the GPU view's memory
// accesses: naive (the write strides down a column), through a 32x32 shared
// tile (coalesced writes, but reading the tile down a column is a 32-way bank
// conflict), and through a tile padded to 32x33 (conflict-free).
module attributes {gpu.container_module} {
  func.func @transpose_naive(%in: memref<1024x1024xf32>, %out: memref<1024x1024xf32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c32, %gy = %c32, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c32, %sz = %c1) {
      %x0 = arith.muli %bx, %c32 : index
      %x = arith.addi %x0, %tx : index
      %y0 = arith.muli %by, %c32 : index
      %y = arith.addi %y0, %ty : index
      %v = memref.load %in[%y, %x] : memref<1024x1024xf32>
      memref.store %v, %out[%x, %y] : memref<1024x1024xf32>
      gpu.terminator
    }
    return
  }

  func.func @transpose_tiled(%in: memref<1024x1024xf32>, %out: memref<1024x1024xf32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c32, %gy = %c32, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c32, %sz = %c1)
               workgroup(%tile : memref<32x32xf32, #gpu.address_space<workgroup>>) {
      %x0 = arith.muli %bx, %c32 : index
      %x = arith.addi %x0, %tx : index
      %y0 = arith.muli %by, %c32 : index
      %y = arith.addi %y0, %ty : index
      %v = memref.load %in[%y, %x] : memref<1024x1024xf32>
      memref.store %v, %tile[%ty, %tx] : memref<32x32xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      %ox = arith.addi %y0, %tx : index
      %oy = arith.addi %x0, %ty : index
      %t = memref.load %tile[%tx, %ty] : memref<32x32xf32, #gpu.address_space<workgroup>>
      memref.store %t, %out[%oy, %ox] : memref<1024x1024xf32>
      gpu.terminator
    }
    return
  }

  func.func @transpose_padded(%in: memref<1024x1024xf32>, %out: memref<1024x1024xf32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c32, %gy = %c32, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c32, %sz = %c1)
               workgroup(%tile : memref<32x33xf32, #gpu.address_space<workgroup>>) {
      %x0 = arith.muli %bx, %c32 : index
      %x = arith.addi %x0, %tx : index
      %y0 = arith.muli %by, %c32 : index
      %y = arith.addi %y0, %ty : index
      %v = memref.load %in[%y, %x] : memref<1024x1024xf32>
      memref.store %v, %tile[%ty, %tx] : memref<32x33xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      %ox = arith.addi %y0, %tx : index
      %oy = arith.addi %x0, %ty : index
      %t = memref.load %tile[%tx, %ty] : memref<32x33xf32, #gpu.address_space<workgroup>>
      memref.store %t, %out[%oy, %ox] : memref<1024x1024xf32>
      gpu.terminator
    }
    return
  }
}
