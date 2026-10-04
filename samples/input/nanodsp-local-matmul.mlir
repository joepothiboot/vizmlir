// A 128x256 by 256x128 f32 matmul in nano-dsp-mlir's `dsp` dialect, the
// kernel of nano-dsp's test/Hexagon/kernels-local-f32.mlir (RUN lines
// dropped). nanodsp-opt, not mlir-opt, reads it: the hexagon-hvx128 schedule
// cuts k into two 128-wide cache tiles, and -nanodsp-promote-local stages the
// A and B tiles in local memory (#dsp.local) with double-buffered DMAs.
module {
  func.func @matmul_local(%a: tensor<128x256xf32>, %b: tensor<256x128xf32>) -> tensor<128x128xf32>
      attributes {llvm.emit_c_interface} {
    %r = dsp.matmul %a, %b : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<128x128xf32>
    return %r : tensor<128x128xf32>
  }
}
