// A matmul + bias + ReLU layer, the payload for the tiling and lowering samples.
module {
  func.func @dense(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>, %bias: tensor<64xf32>) -> tensor<128x64xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128x64xf32>
    %acc = linalg.fill ins(%zero : f32) outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
    %mm = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>)
                        outs(%acc : tensor<128x64xf32>) -> tensor<128x64xf32>
    %out = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (j)>, affine_map<(i, j) -> (i, j)>],
                           iterator_types = ["parallel", "parallel"]}
        ins(%mm, %bias : tensor<128x64xf32>, tensor<64xf32>) outs(%init : tensor<128x64xf32>) {
    ^bb0(%m: f32, %b: f32, %o: f32):
      %s = arith.addf %m, %b : f32
      %r = arith.maximumf %s, %zero : f32
      linalg.yield %r : f32
    } -> tensor<128x64xf32>
    func.return %out : tensor<128x64xf32>
  }
}
