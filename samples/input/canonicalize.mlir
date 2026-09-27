// Redundant arithmetic that canonicalize + cse fold away.
module {
  func.func @scale_bias(%x: tensor<64xf32>, %n: index) -> (tensor<64xf32>, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %one = arith.constant 1.0 : f32
    %zero = arith.constant 0.0 : f32
    %n0 = arith.addi %n, %c0 : index
    %n1 = arith.muli %n0, %c1 : index
    %k = arith.muli %c4, %c4 : index
    %m = arith.addi %n1, %k : index
    %dead = arith.muli %m, %m : index
    %init = tensor.empty() : tensor<64xf32>
    %y = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                         iterator_types = ["parallel"]}
        ins(%x : tensor<64xf32>) outs(%init : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %a = arith.mulf %in, %one : f32
      %b = arith.addf %a, %zero : f32
      %c = arith.mulf %in, %one : f32
      %d = arith.addf %b, %c : f32
      linalg.yield %d : f32
    } -> tensor<64xf32>
    func.return %y, %m : tensor<64xf32>, index
  }
}
