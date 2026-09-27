#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @dense(%arg0: tensor<128x256xf32>, %arg1: tensor<256x64xf32>, %arg2: tensor<64xf32>) -> tensor<128x64xf32> {
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %2 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %1) -> (tensor<128x64xf32>) {
      %4 = scf.for %arg5 = %c0 to %c64 step %c32 iter_args(%arg6 = %arg4) -> (tensor<128x64xf32>) {
        %5 = scf.for %arg7 = %c0 to %c256 step %c64 iter_args(%arg8 = %arg6) -> (tensor<128x64xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg7] [32, 64] [1, 1] : tensor<128x256xf32> to tensor<32x64xf32>
          %extracted_slice_0 = tensor.extract_slice %arg1[%arg7, %arg5] [64, 32] [1, 1] : tensor<256x64xf32> to tensor<64x32xf32>
          %extracted_slice_1 = tensor.extract_slice %arg8[%arg3, %arg5] [32, 32] [1, 1] : tensor<128x64xf32> to tensor<32x32xf32>
          %6 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<32x64xf32>, tensor<64x32xf32>) outs(%extracted_slice_1 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %inserted_slice = tensor.insert_slice %6 into %arg8[%arg3, %arg5] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x64xf32>
          scf.yield %inserted_slice : tensor<128x64xf32>
        }
        scf.yield %5 : tensor<128x64xf32>
      }
      scf.yield %4 : tensor<128x64xf32>
    }
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<128x64xf32>, tensor<64xf32>) outs(%0 : tensor<128x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      %5 = arith.maximumf %4, %cst : f32
      linalg.yield %5 : f32
    } -> tensor<128x64xf32>
    return %3 : tensor<128x64xf32>
  }
}

