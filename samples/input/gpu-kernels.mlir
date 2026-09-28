// Two elementwise kernels launched from the host, for the GPU kernels sample:
// outlined into gpu.module @saxpy_kernel and @relu_kernel, lowered to NVVM,
// and serialized to PTX.
module attributes {gpu.container_module} {
  func.func @saxpy(%a: f32, %x: memref<1024xf32>, %y: memref<1024xf32>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c4, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c256, %sy = %c1, %sz = %c1) {
      %base = arith.muli %bx, %c256 : index
      %i = arith.addi %base, %tx : index
      %xv = memref.load %x[%i] : memref<1024xf32>
      %yv = memref.load %y[%i] : memref<1024xf32>
      %ax = arith.mulf %a, %xv : f32
      %r = arith.addf %ax, %yv : f32
      memref.store %r, %y[%i] : memref<1024xf32>
      gpu.terminator
    }
    return
  }

  func.func @relu(%x: memref<1024xf32>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant 0.0 : f32
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c4, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c256, %sy = %c1, %sz = %c1) {
      %base = arith.muli %bx, %c256 : index
      %i = arith.addi %base, %tx : index
      %v = memref.load %x[%i] : memref<1024xf32>
      %r = arith.maximumf %v, %zero : f32
      memref.store %r, %x[%i] : memref<1024xf32>
      gpu.terminator
    }
    return
  }
}
