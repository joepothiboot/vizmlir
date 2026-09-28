// A 128x128 matmul tiled for the GPU, for the tiled matmul sample: an 8x8
// grid of 16x16-thread blocks, each staging 16x16 tiles of A and B in shared
// (workgroup) memory between barriers. Outlined, lowered to NVVM, and
// serialized to PTX.
module attributes {gpu.container_module} {
  func.func @matmul(%A: memref<128x128xf32>, %B: memref<128x128xf32>, %C: memref<128x128xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %zero = arith.constant 0.0 : f32
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c8, %gy = %c8, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c16, %sy = %c16, %sz = %c1)
               workgroup(%tileA : memref<16x16xf32, #gpu.address_space<workgroup>>,
                         %tileB : memref<16x16xf32, #gpu.address_space<workgroup>>) {
      %rowBase = arith.muli %by, %c16 : index
      %row = arith.addi %rowBase, %ty : index
      %colBase = arith.muli %bx, %c16 : index
      %col = arith.addi %colBase, %tx : index
      %sum = scf.for %t = %c0 to %c128 step %c16 iter_args(%acc = %zero) -> (f32) {
        %aCol = arith.addi %t, %tx : index
        %bRow = arith.addi %t, %ty : index
        %a = memref.load %A[%row, %aCol] : memref<128x128xf32>
        %b = memref.load %B[%bRow, %col] : memref<128x128xf32>
        memref.store %a, %tileA[%ty, %tx] : memref<16x16xf32, #gpu.address_space<workgroup>>
        memref.store %b, %tileB[%ty, %tx] : memref<16x16xf32, #gpu.address_space<workgroup>>
        gpu.barrier
        %partial = scf.for %k = %c0 to %c16 step %c1 iter_args(%p = %acc) -> (f32) {
          %x = memref.load %tileA[%ty, %k] : memref<16x16xf32, #gpu.address_space<workgroup>>
          %y = memref.load %tileB[%k, %tx] : memref<16x16xf32, #gpu.address_space<workgroup>>
          %m = arith.mulf %x, %y : f32
          %s = arith.addf %p, %m : f32
          scf.yield %s : f32
        }
        gpu.barrier
        scf.yield %partial : f32
      }
      memref.store %sum, %C[%row, %col] : memref<128x128xf32>
      gpu.terminator
    }
    return
  }
}
