#map = affine_map<(d0) -> (d0)>
module {
  func.func @scale_bias(%arg0: tensor<64xf32>, %arg1: index) -> (tensor<64xf32>, index) {
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %0 = arith.addi %arg1, %c16 : index
    %1 = tensor.empty() : tensor<64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<64xf32>) outs(%1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %cst : f32
      %4 = arith.addf %3, %in : f32
      linalg.yield %4 : f32
    } -> tensor<64xf32>
    return %2, %0 : tensor<64xf32>, index
  }
}

