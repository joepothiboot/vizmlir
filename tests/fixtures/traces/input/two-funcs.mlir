#map = affine_map<(d0) -> (d0)>
module {
  func.func @f(%a: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %0 = arith.addi %a, %c0 : i32
    %1 = arith.addi %a, %c0 : i32
    %2 = arith.muli %0, %1 : i32
    return %2 : i32
  }
  func.func @g(%a: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %0 = arith.muli %a, %c1 : i32
    return %0 : i32
  }
}
