// RUN: mlir-tensorrt-opt %s --plan-outline-constant-foldable-subgraphs --split-input-file --canonicalize -allow-unregistered-dialect | FileCheck %s

func.func @simple_case1(%arg0: tensor<4xf32>) -> tensor<4xf32>{
  %c0 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %c1 = stablehlo.constant dense<3.0> : tensor<4xf32>
  %add0 = stablehlo.add %c0, %c1 : tensor<4xf32>
  %add1 = stablehlo.add %arg0, %add0 : tensor<4xf32>
  return %add1 : tensor<4xf32>
}

// CHECK-LABEL: @simple_case1
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>) -> tensor<4xf32>
//  CHECK-NEXT: %[[v0:.+]] = call @constant_subgraph() : () -> tensor<4xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.add %[[arg0]], %[[v0]] : tensor<4xf32>
//  CHECK-NEXT: return %[[v1]] : tensor<4xf32>
//       CHECK: func.func private @constant_subgraph() -> tensor<4xf32> attributes {plan.constant_foldable}

// -----

func.func @simple_case2(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>{
  %c0 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %c1 = stablehlo.constant dense<3.0> : tensor<4xf32>
  %add0 = stablehlo.add %c0, %arg0 : tensor<4xf32>
  %add1 = stablehlo.add %c0, %c1 : tensor<4xf32>
  %add2 = stablehlo.add %arg1, %add1 : tensor<4xf32>
  %sub0 = stablehlo.subtract %c0, %c1 : tensor<4xf32>
  %add3 = stablehlo.add %add2, %add0 : tensor<4xf32>
  %sub1 = stablehlo.subtract %add3, %sub0 : tensor<4xf32>
  return %sub1 : tensor<4xf32>
}

// CHECK-LABEL: @simple_case2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>, %[[arg1:.+]]: tensor<4xf32>) -> tensor<4xf32>
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
//  CHECK-NEXT: %[[v1:.+]] = call @constant_subgraph_0() : () -> tensor<4xf32>
//  CHECK-NEXT: %[[v2:.+]] = call @constant_subgraph() : () -> tensor<4xf32>
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.add %[[v0]], %[[arg0]] : tensor<4xf32>
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.add %[[arg1]], %[[v2]] : tensor<4xf32>
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.add %[[v4]], %[[v3]] : tensor<4xf32>
//  CHECK-NEXT: %[[v6:.+]] = stablehlo.subtract %[[v5]], %[[v1]] : tensor<4xf32>
//  CHECK-NEXT: return %[[v6]] : tensor<4xf32>
//       CHECK: func.func private @constant_subgraph() -> tensor<4xf32> attributes {plan.constant_foldable}
//       CHECK: func.func private @constant_subgraph_0() -> tensor<4xf32> attributes {plan.constant_foldable}

// -----

func.func @simple_case3(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>{
  %c0 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %add0 = stablehlo.add %arg0, %c0 : tensor<4xf32>
  %c1 = stablehlo.constant dense<3.0> : tensor<4xf32>
  %add1 = stablehlo.add %c0, %arg1 : tensor<4xf32>
  %add2 = stablehlo.add %add0, %c1 : tensor<4xf32>
  %add3 = stablehlo.add %arg0, %add2 : tensor<4xf32>
  return %add3 : tensor<4xf32>
}

// CHECK-LABEL: @simple_case3
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>, %[[arg1:.+]]: tensor<4xf32>) -> tensor<4xf32>
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.add %[[arg0]], %[[v1]] : tensor<4xf32>
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.add %[[v2]], %[[v0]] : tensor<4xf32>
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.add %[[arg0]], %[[v3]] : tensor<4xf32>
//  CHECK-NEXT: return %[[v4]] : tensor<4xf32>

// -----

func.func @scf_for_case1(%arg0: tensor<4xf32>) -> tensor<4xf32>{
  %c0 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %c1 = stablehlo.constant dense<3.0> : tensor<4xf32>
  %add0 = stablehlo.add %c0, %arg0 : tensor<4xf32>
  %for_start = arith.constant 0 : index
  %for_end = arith.constant 5 : index
  %for_step = arith.constant 1 : index
  %sum = scf.for %iv = %for_start to %for_end step %for_step
    iter_args(%sum_iter = %add0) -> (tensor<4xf32>) {
    %add1 = stablehlo.add %c0, %c1 : tensor<4xf32>
    %sub1 = stablehlo.subtract %add1, %c1 : tensor<4xf32>
    %sum_next = stablehlo.add %add0, %sub1 : tensor<4xf32>
    scf.yield %sum_next : tensor<4xf32>
  }
  return %sum : tensor<4xf32>
}

//  CHECK-LABEL: @scf_for_case1
//   CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>) -> tensor<4xf32>
//        CHECK: %[[cst:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
//        CHECK: %[[v0:.+]] = stablehlo.add %[[cst]], %[[arg0]] : tensor<4xf32>
//        CHECK: %[[v1:.+]] = scf.for
//   CHECK-NEXT: func.call @constant_subgraph() : () -> tensor<4xf32>
//   CHECK-NEXT: stablehlo.add
//   CHECK-NEXT: scf.yield
//        CHECK: return %[[v1]] : tensor<4xf32>
//        CHECK: func.func private @constant_subgraph() -> tensor<4xf32> attributes {plan.constant_foldable}

// -----

func.func @scf_for_case2(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>{
  %c0 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %c1 = stablehlo.constant dense<3.0> : tensor<4xf32>
  %add0 = stablehlo.add %c0, %arg0 : tensor<4xf32>
  %for_start = arith.constant 0 : index
  %for_end = arith.constant 5 : index
  %for_step = arith.constant 1 : index
  %sum = scf.for %iv = %for_start to %for_end step %for_step
    iter_args(%sum_iter = %add0) -> (tensor<4xf32>) {
    %add1 = stablehlo.add %c0, %c1 : tensor<4xf32>
    %add2 = stablehlo.add %add1, %arg1 : tensor<4xf32>
    %sub1 = stablehlo.subtract %add2, %c1 : tensor<4xf32>
    %sum_next = stablehlo.add %sum_iter, %sub1 : tensor<4xf32>
    scf.yield %sum_next : tensor<4xf32>
  }
  return %sum : tensor<4xf32>
}

// CHECK-LABEL: @scf_for_case2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>, %[[arg1:.+]]: tensor<4xf32>) -> tensor<4xf32>
//       CHECK: %[[cst:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
//       CHECK: %[[v0:.+]] = stablehlo.add %[[cst]], %[[arg0]] : tensor<4xf32>
//       CHECK: %[[v1:.+]] = scf.for
//  CHECK-NEXT: func.call @constant_subgraph() : () -> tensor<4xf32>
//  CHECK-NEXT: stablehlo.add
//  CHECK-NEXT: stablehlo.subtract
//  CHECK-NEXT: stablehlo.add
//  CHECK-NEXT: scf.yield
//       CHECK: return %[[v1]] : tensor<4xf32>
//       CHECK: func.func private @constant_subgraph() -> tensor<4xf32> attributes {plan.constant_foldable}

// -----

func.func @scf_for_case3(%arg0: tensor<4xf32>) -> tensor<4xf32>{
  %c0 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %add0 = stablehlo.add %c0, %arg0 : tensor<4xf32>
  %for_start = arith.constant 0 : index
  %for_end = arith.constant 5 : index
  %for_step = arith.constant 1 : index
  %sum = scf.for %iv = %for_start to %for_end step %for_step
    iter_args(%sum_iter = %add0) -> (tensor<4xf32>) {
    %sum_next = stablehlo.add %sum_iter, %c0 : tensor<4xf32>
    scf.yield %sum_next : tensor<4xf32>
  }
  return %sum : tensor<4xf32>
}

// CHECK-LABEL: @scf_for_case3
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>) -> tensor<4xf32>
//       CHECK: %[[cst:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
//       CHECK: %[[v0:.+]] = stablehlo.add %[[cst]], %[[arg0]] : tensor<4xf32>
//       CHECK: %[[v1:.+]] = scf.for
//  CHECK-NEXT: stablehlo.add
//  CHECK-NEXT: scf.yield
//       CHECK: return %[[v1]] : tensor<4xf32>

// -----

func.func @scf_for_case4(%arg0: tensor<1111x1345xf32>) -> tensor<1111x1345xf32> {
  %c10_i64 = arith.constant 10 : i64
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<1345x1111xf16>
  %c = stablehlo.constant dense<1> : tensor<i64>
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  %c1_i64 = arith.constant 1 : i64
  %0:2 = scf.for %arg1 = %c1_i64 to %c10_i64 step %c1_i64 iter_args(%arg2 =
    %c_0, %arg3 = %arg0) -> (tensor<i64>, tensor<1111x1345xf32>)  : i64 {
    %1 = stablehlo.add %arg2, %c : tensor<i64>
    %2 = stablehlo.convert %cst : (tensor<1345x1111xf16>) ->tensor<1345x1111xf32>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<1345x1111xf32>) ->tensor<1111x1345xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<1111x1345xf32>
    scf.yield %1, %4 : tensor<i64>,tensor<1111x1345xf32>
  }
  return %0#1 : tensor<1111x1345xf32>
}

// CHECK-LABEL: @scf_for_case4
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1111x1345xf32>) -> tensor<1111x1345xf32>
//       CHECK: %[[v1:.+]]:2 = scf.for
//  CHECK-NEXT: func.call @constant_subgraph() : () -> tensor<1111x1345xf32>
//  CHECK-NEXT: stablehlo.add
//  CHECK-NEXT: stablehlo.add
//  CHECK-NEXT: scf.yield
//       CHECK: return %[[v1]]#1 : tensor<1111x1345xf32>
//       CHECK: func.func private @constant_subgraph() -> tensor<1111x1345xf32> attributes {plan.constant_foldable}

// -----

func.func @scf_while_case1(%init: tensor<4xf32>) -> tensor<4xf32>{
  %cst0 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %cst1 = stablehlo.constant dense<3.0> : tensor<4xf32>
  %cst2 = stablehlo.constant dense<4.0> : tensor<4xf32>
  %count = arith.constant 5: i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r, %c = scf.while(%arg0 = %init, %arg1 = %count) : (tensor<4xf32>, i32) -> (tensor<4xf32>, i32){
    %0 = arith.subi %arg1, %c1 : i32
    %1 = arith.cmpi eq, %0, %c0 : i32
    scf.condition(%1) %arg0, %0 : tensor<4xf32>, i32
  } do {
    ^bb0(%base: tensor<4xf32>, %new_count: i32):
      %add0 = stablehlo.add %cst0, %cst1 : tensor<4xf32>
      %sub0 = stablehlo.subtract %cst2, %add0 : tensor<4xf32>
      %new = stablehlo.add %base, %sub0 : tensor<4xf32>
      scf.yield %new, %new_count : tensor<4xf32>, i32
  }
  return %r : tensor<4xf32>
}

//  CHECK-LABEL: @scf_while_case1
//        CHECK: %[[v0:.+]]:2 = scf.while
//        CHECK: scf.condition
//        CHECK: ^bb0(%[[arg1:.+]]: tensor<4xf32>, %[[arg2:.+]]: i32)
//   CHECK-NEXT: stablehlo.add
//   CHECK-NEXT: stablehlo.subtract
//   CHECK-NEXT: stablehlo.add
//   CHECK-NEXT: scf.yield
//        CHECK: return %[[v0]]#0

// -----

func.func @scf_while_case2(%init: tensor<4xf32>) -> tensor<4xf32>{
  %cst0 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %cst1 = stablehlo.constant dense<3.0> : tensor<4xf32>
  %cst2 = stablehlo.constant dense<4.0> : tensor<4xf32>
  %count = arith.constant 2: i32
  %c1 = arith.constant 1 : i32
  %r, %c = scf.while(%arg0 = %init, %arg1 = %count) : (tensor<4xf32>, i32) -> (tensor<4xf32>, i32){
    %0 = arith.subi %arg1, %c1 : i32
    %1 = arith.cmpi eq, %0, %c1 : i32
    scf.condition(%1) %arg0, %0 : tensor<4xf32>, i32
  } do {
    ^bb0(%base: tensor<4xf32>, %new_count: i32):
      %add0 = stablehlo.add %cst0, %cst1 : tensor<4xf32>
      %sub0 = stablehlo.subtract %cst2, %add0 : tensor<4xf32>
      %new = stablehlo.add %base, %sub0 : tensor<4xf32>
      scf.yield %new, %new_count : tensor<4xf32>, i32
  }
  return %r : tensor<4xf32>
}

//  CHECK-LABEL: @scf_while_case2
//        CHECK: %[[v0:.+]]:2 = scf.while
//        CHECK: scf.condition
//        CHECK: ^bb0(%[[arg1:.+]]: tensor<4xf32>, %[[arg2:.+]]: i32)
//   CHECK-NEXT: %[[v1:.+]] = func.call @constant_subgraph() : () -> tensor<4xf32>
//   CHECK-NEXT: %[[v2:.+]] = stablehlo.add %[[arg1]], %[[v1]] : tensor<4xf32>
//   CHECK-NEXT: scf.yield %[[v2]], %[[arg2]] : tensor<4xf32>, i32
//        CHECK: return %[[v0]]#0
//        CHECK: func.func private @constant_subgraph() -> tensor<4xf32> attributes {plan.constant_foldable}

// -----

func.func @scf_if_case1(%arg0: tensor<4xf32>, %arg1: i1) -> tensor<4xf32>{
  %c0 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %c1 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %r = scf.if %arg1 -> (tensor<4xf32>){
    %c2 = stablehlo.constant dense<3.0> : tensor<4xf32>
    %add0 = stablehlo.add %c0, %c1 : tensor<4xf32>
    %sub0 = stablehlo.subtract %c2, %add0 : tensor<4xf32>
    scf.yield %sub0 : tensor<4xf32>
  } else {
    %add0 = stablehlo.add %arg0, %c0 : tensor<4xf32>
    %sub0 = stablehlo.subtract %add0, %c1 : tensor<4xf32>
    scf.yield %sub0 : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: @scf_if_case1
//       CHECK: %[[v0:.+]] = scf.if
//  CHECK-NEXT: func.call @constant_subgraph() : () -> tensor<4xf32>
//  CHECK-NEXT: scf.yield
//  CHECK-NEXT: } else {
//  CHECK-NEXT: stablehlo.add
//  CHECK-NEXT: stablehlo.subtract
//  CHECK-NEXT: scf.yield
//       CHECK: return %[[v0]] : tensor<4xf32>
//       CHECK: func.func private @constant_subgraph() -> tensor<4xf32> attributes {plan.constant_foldable}

// -----

func.func @scf_if_case2(%arg0: tensor<4xf32>) -> tensor<4xf32>{
  %cond = arith.constant 1 : i1
  %c0 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %c1 = stablehlo.constant dense<2.0> : tensor<4xf32>
  %r = scf.if %cond -> (tensor<4xf32>){
    %c2 = stablehlo.constant dense<3.0> : tensor<4xf32>
    %add0 = stablehlo.add %c0, %c1 : tensor<4xf32>
    %sub0 = stablehlo.subtract %c2, %add0 : tensor<4xf32>
    scf.yield %sub0 : tensor<4xf32>
  } else {
    %add0 = stablehlo.add %arg0, %c0 : tensor<4xf32>
    %sub0 = stablehlo.subtract %add0, %c1 : tensor<4xf32>
    scf.yield %sub0 : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: @scf_if_case2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>) -> tensor<4xf32>
//  CHECK-NEXT: %[[v0:.+]] = call @constant_subgraph() : () -> tensor<4xf32>
//  CHECK-NEXT: return %[[v0]] : tensor<4xf32>
//       CHECK: func.func private @constant_subgraph() -> tensor<4xf32> attributes {plan.constant_foldable}

// -----

func.func @foldable_terminator() -> (tensor<4xf32>, tensor<4xf32>){
  %c0 = stablehlo.constant dense_resource<__elided__> : tensor<4xf32>
  %c1 = stablehlo.constant dense_resource<__elided__> : tensor<4xf32>
  %0 = stablehlo.add %c0, %c1 : tensor<4xf32>
  %2 = stablehlo.subtract %0, %c1 : tensor<4xf32>
  return %2, %0 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: @foldable_terminator
//  CHECK-NEXT: %[[v0:.+]]:2 = call @constant_subgraph() : () -> (tensor<4xf32>, tensor<4xf32>)
//  CHECK-NEXT: return %[[v0]]#1, %[[v0]]#0 : tensor<4xf32>, tensor<4xf32>
//       CHECK: func.func private @constant_subgraph() -> (tensor<4xf32>, tensor<4xf32>) attributes {plan.constant_foldable}

// -----

func.func @skip_outlining() -> (tensor<4xf32>, tensor<4xf32>){
  %c0 = stablehlo.constant dense_resource<__elided__> : tensor<4xf32>
  %c1 = stablehlo.constant dense_resource<__elided__> : tensor<4xf32>
  %0 = stablehlo.add %c0, %c1 : tensor<4xf32>
  %1 = "some.op"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = stablehlo.subtract %0, %c1 : tensor<4xf32>
  return %1, %2 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: @skip_outlining
//  CHECK-NEXT: %[[cst:.+]] = stablehlo.constant
//  CHECK-NEXT: %[[v0:.+]] = call @constant_subgraph() : () -> tensor<4xf32>
//  CHECK-NEXT: %[[v1:.+]] = "some.op"(%[[v0]]) : (tensor<4xf32>) -> tensor<4xf32>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.subtract %[[v0]], %[[cst]] : tensor<4xf32>
//  CHECK-NEXT: return %[[v1]], %[[v2]] : tensor<4xf32>, tensor<4xf32>
//       CHECK: func.func private @constant_subgraph() -> tensor<4xf32> attributes {plan.constant_foldable}

// -----

func.func @reduce_const_foldable_negative(%arg0: tensor<f32>) -> tensor<1x10xf32> {
  %cst_0 = stablehlo.constant dense<1.0> : tensor<1x10x20xf32>
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %cst_1 = stablehlo.constant dense<2.0> : tensor<f32>
  %0 = stablehlo.reduce(%cst_0 init: %cst)
      across dimensions = [2] : (tensor<1x10x20xf32>, tensor<f32>) -> tensor<1x10xf32>
    reducer(%accum: tensor<f32>, %curr: tensor<f32>)  {
      %first = stablehlo.add %accum, %curr : tensor<f32>
      %second = stablehlo.multiply %first, %cst_1 : tensor<f32>
      %third = stablehlo.subtract %second, %arg0 : tensor<f32>
      stablehlo.return %third : tensor<f32>
  }
  return %0 : tensor<1x10xf32>
}

// CHECK-LABEL: @reduce_const_foldable_negative
//  CHECK-NEXT: stablehlo.constant
//  CHECK-NEXT: stablehlo.constant
//  CHECK-NEXT: stablehlo.constant
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.reduce
//       CHECK: return %[[v1]] : tensor<1x10xf32>

// -----

func.func @reduce_const_foldable(%arg0: tensor<f32>) -> tensor<1x10xf32> {
  %cst_0 = stablehlo.constant dense<1.0> : tensor<1x10x20xf32>
  %add_0 = stablehlo.add %cst_0, %cst_0 : tensor<1x10x20xf32>
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %cst_1 = stablehlo.constant dense<2.0> : tensor<f32>
  %0 = stablehlo.reduce(%cst_0 init: %cst)
      across dimensions = [2] : (tensor<1x10x20xf32>, tensor<f32>) -> tensor<1x10xf32>
    reducer(%accum: tensor<f32>, %curr: tensor<f32>)  {
      %first = stablehlo.add %accum, %curr : tensor<f32>
      %second = stablehlo.multiply %first, %cst_1 : tensor<f32>
      stablehlo.return %second : tensor<f32>
  }
  return %0 : tensor<1x10xf32>
}

// CHECK-LABEL: @reduce_const_foldable
//  CHECK-NEXT: %[[v0:.+]] = call @constant_subgraph() : () -> tensor<1x10xf32>
//  CHECK-NEXT: return %[[v0]] : tensor<1x10xf32>
//       CHECK: func.func private @constant_subgraph() -> tensor<1x10xf32> attributes {plan.constant_foldable}

// -----

#map = affine_map<(d0)->(d0)>
func.func @linalg_generic_neg(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %empty = tensor.empty () : tensor<10xf32>
  %0 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map]
  } ins(%arg0: tensor<10xf32>) outs(%empty: tensor<10xf32>) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.negf %a : f32
      linalg.yield %r : f32
  } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @linalg_generic_neg
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>) -> tensor<10xf32>
//  CHECK-NEXT: %[[v0:.+]] = tensor.empty() : tensor<10xf32>
//  CHECK-NEXT: %[[v1:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<10xf32>) outs(%[[v0]] : tensor<10xf32>)
//       CHECK: return %[[v1]] : tensor<10xf32>

// -----

#map = affine_map<(d0)->(d0)>
func.func @linalg_generic() -> tensor<10xf32> {
  %empty = tensor.empty () : tensor<10xf32>
  %cst = stablehlo.constant dense<4.0> : tensor<10xf32>
  %cst_1 = stablehlo.constant dense<4.0> : tensor<10xf32>
  %add = stablehlo.add %cst, %cst_1 : tensor<10xf32>
  %0 = linalg.generic {
    iterator_types = ["parallel"],
    indexing_maps = [#map, #map]
  } ins(%add: tensor<10xf32>) outs(%empty: tensor<10xf32>) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.negf %a : f32
      linalg.yield %r : f32
  } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @linalg_generic
//  CHECK-NEXT: %[[v0:.+]] = call @constant_subgraph() : () -> tensor<10xf32>
//  CHECK-NEXT: return %[[v0]] : tensor<10xf32>
//       CHECK: func.func private @constant_subgraph() -> tensor<10xf32> attributes {plan.constant_foldable}