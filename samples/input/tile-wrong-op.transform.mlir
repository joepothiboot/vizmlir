// A common mistake: matching the scalar ReLU op instead of its linalg.generic.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %relu = transform.structured.match ops{["arith.maximumf"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %relu tile_sizes [32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
