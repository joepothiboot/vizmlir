module attributes {transform.with_named_sequence} {
  func.func @f(%a: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %0 = arith.addi %a, %c0 : i32
    return %0 : i32
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["arith.addi"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %f, "matched addi" : !transform.any_op
    %t, %l = transform.structured.tile_using_for %f tile_sizes [2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
