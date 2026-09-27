// Tiles the matmul 32x32 and the reduction by 64, like a CPU codegen pipeline.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %mm tile_sizes [32, 32, 64]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
