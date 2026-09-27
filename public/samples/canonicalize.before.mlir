#map = affine_map<(d0) -> (d0)>
module {
  func.func @scale_bias(%arg0: tensor<64xf32>, %arg1: index) -> (tensor<64xf32>, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = arith.addi %arg1, %c0 : index
    %1 = arith.muli %0, %c1 : index
    %2 = arith.muli %c4, %c4 : index
    %3 = arith.addi %1, %2 : index
    %4 = arith.muli %3, %3 : index
    %5 = tensor.empty() : tensor<64xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<64xf32>) outs(%5 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.mulf %in, %cst : f32
      %8 = arith.addf %7, %cst_0 : f32
      %9 = arith.mulf %in, %cst : f32
      %10 = arith.addf %8, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<64xf32>
    return %6, %3 : tensor<64xf32>, index
  }
}

