// RUN: mlir-tensorrt-opt %s -split-input-file -convert-stablehlo-to-scf | FileCheck %s

func.func @stablehlo_while_to_scf_for() -> (tensor<i64>, tensor<i64>) {
    %init_i = stablehlo.constant dense<1> :tensor<i64>
    %init_sum = stablehlo.constant dense<0> :tensor<i64>
    %one = stablehlo.constant dense<1> :tensor<i64>
    %ten = stablehlo.constant dense<10> :tensor<i64>

    %results0, %results1 = "stablehlo.while"(%init_i, %init_sum) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %cond = "stablehlo.compare"(%arg0, %ten) {
          comparison_direction = #stablehlo<comparison_direction LT>
        } : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %cond : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %new_sum = stablehlo.add %arg1, %one : tensor<i64>
        %new_i = stablehlo.add %arg0, %one : tensor<i64>
        stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
    }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
    return %results0, %results1 : tensor<i64>, tensor<i64>
}

// CHECK-LABEL: func.func @stablehlo_while_to_scf_for
//   CHECK-DAG:     %[[cst:.+]] = arith.constant dense<9> : tensor<i64>
//   CHECK-DAG:     %[[c10_i64:.+]] = arith.constant 10 : i64
//   CHECK-DAG:     %[[c:.+]] = stablehlo.constant dense<0> : tensor<i64>
//   CHECK-DAG:     %[[c1_i64:.+]] = arith.constant 1 : i64
//   CHECK-DAG:     %[[cst1:.+]] = stablehlo.constant dense<1> : tensor<i64>
//       CHECK:     %[[v0:.+]] = scf.for %[[arg0:.+]] = %[[c1_i64]] to %[[c10_i64]] step %[[c1_i64]]
//  CHECK-SAME:       iter_args(%[[arg1:.+]] = %[[c]]) -> (tensor<i64>)
//       CHECK-DAG:       %[[v1:.+]] = stablehlo.add %[[arg1]], %[[cst1]] : tensor<i64>
//       CHECK-DAG:       scf.yield %[[v1]] : tensor<i64>
//       CHECK:     return %[[cst]], %[[v0]] : tensor<i64>, tensor<i64>

// -----

func.func private @condition(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i1>

func.func @stablehlo_while_to_scf_while(%arg0: tensor<i64>, %arg1: tensor<i64>) -> (tensor<i64>, tensor<i64>) {
  %one = stablehlo.constant dense<1> :tensor<i64>
  %results0, %results1 = "stablehlo.while"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
      %cond = call @condition(%arg2, %arg3) : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
  }, {
  ^bb0(%arg4: tensor<i64>, %arg5: tensor<i64>):
      %new_sum = stablehlo.add %arg5, %one : tensor<i64>
      %new_i = stablehlo.add %arg4, %one : tensor<i64>
      stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
  }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
  return %results0, %results1 : tensor<i64>, tensor<i64>
}

// CHECK-LABEL: func.func @stablehlo_while_to_scf_while
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i64>, %[[arg1:.+]]: tensor<i64>)
//       CHECK:     %[[c1_i64:.+]] = stablehlo.constant dense<1> : tensor<i64>
//       CHECK:     %[[v0:.+]]:2 = scf.while (%[[arg2:.+]] = %[[arg0]], %[[arg3:.+]] = %[[arg1]])
//   CHECK-DAG:       %[[v1:.+]] = func.call @condition(%[[arg2]], %[[arg3]]) : (tensor<i64>, tensor<i64>) -> tensor<i1>
//   CHECK-DAG:       %[[extracted:.+]] = tensor.extract %[[v1]][] : tensor<i1>
//   CHECK-DAG:       scf.condition(%[[extracted]]) %[[arg2]], %[[arg3]] : tensor<i64>, tensor<i64>
//       CHECK:     } do {
//       CHECK:     ^bb0(%[[arg2:.+]]: tensor<i64>, %[[arg3:.+]]: tensor<i64>):
//   CHECK-DAG:       %[[v1:.+]] = stablehlo.add %[[arg3]], %[[c1_i64]]
//   CHECK-DAG:       %[[v2:.+]] = stablehlo.add %[[arg2]], %[[c1_i64]]
//   CHECK-DAG:       scf.yield %[[v2]], %[[v1]] : tensor<i64>, tensor<i64>
//       CHECK:     return %[[v0]]#0, %[[v0]]#1 : tensor<i64>, tensor<i64>

// -----

func.func private @some_compute(tensor<f32>) -> tensor<1xf32>

func.func @stablehlo_while_single_iteration(%arg0: tensor<1xf32>, %arg1: tensor<f32>) -> tensor<1xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %5:2 = stablehlo.while(%iterArg = %c0, %iterArg_34 = %cst) : tensor<i32>, tensor<1xf32>
    cond {
    %6 = stablehlo.compare  LT, %iterArg, %c1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %6 : tensor<i1>
  } do {
    %6 = stablehlo.compare  LT, %iterArg, %c0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %7 = stablehlo.add %iterArg, %c1 : tensor<i32>
    %8 = stablehlo.select %6, %7, %iterArg : tensor<i1>, tensor<i32>
    %10 = stablehlo.dynamic_slice %arg0, %8, sizes = [1] : (tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %11 = stablehlo.reshape %10 : (tensor<1xf32>) -> tensor<f32>

    %12 = stablehlo.compare  GE, %11, %cst_0,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %35 = stablehlo.select %12, %cst_0, %arg1 : tensor<i1>, tensor<f32>
    %39 = func.call @some_compute(%35) : (tensor<f32>) -> tensor<1xf32>
    stablehlo.return %7, %39 : tensor<i32>, tensor<1xf32>
  }
  return %5#1 : tensor<1xf32>
}

// CHECK-LABEL: func.func @stablehlo_while_single_iteration
//   CHECK-NOT: scf.while
//   CHECK-NOT: scf.for
//       CHECK:  stablehlo.compare
//       CHECK:  stablehlo.add
//       CHECK:  stablehlo.dynamic_slice
//       CHECK:  call @some_compute
//       CHECK:  return

// -----

func.func @if_op_not_foldable(%arg0: tensor<i64>, %arg1: tensor<i64>) -> (tensor<2xi64>, tensor<2xi64>) {
  %cond = stablehlo.compare LT, %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %result0, %result1 = "stablehlo.if"(%cond) ({
    %0 = stablehlo.constant dense<0> : tensor<2xi64>
    %1 = stablehlo.constant dense<2> : tensor<2xi64>
    %3 = stablehlo.add %0, %1 : tensor<2xi64>
    stablehlo.return %3, %3 : tensor<2xi64>, tensor<2xi64>
  }, {
    %1 = stablehlo.constant dense<1> : tensor<2xi64>
    stablehlo.return %1, %1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
  func.return %result0, %result1 : tensor<2xi64>, tensor<2xi64>
}

// CHECK-LABEL: func.func @if_op_not_foldable
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i64>, %[[arg1:.+]]: tensor<i64>) -> (tensor<2xi64>, tensor<2xi64>) {
//   CHECK-DAG:     %[[c:.+]] = stablehlo.constant dense<1> : tensor<2xi64>
//   CHECK-DAG:     %[[c_0:.+]] = stablehlo.constant dense<2> : tensor<2xi64>
//   CHECK-DAG:     %[[c_1:.+]] = stablehlo.constant dense<0> : tensor<2xi64>
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][] : tensor<i64>
//   CHECK-DAG:     %[[extracted_2:.+]] = tensor.extract %[[arg1]][] : tensor<i64>
//   CHECK-DAG:     %[[v0:.+]] = arith.cmpi slt, %[[extracted]], %[[extracted_2]] : i64
//       CHECK:     %[[v1]]:2 = scf.if %[[v0]] -> (tensor<2xi64>, tensor<2xi64>) {
//       CHECK:       %[[v2:.+]] = stablehlo.add %[[c_1]], %[[c_0]] : tensor<2xi64>
//       CHECK:       scf.yield %[[v2]], %[[v2]] : tensor<2xi64>, tensor<2xi64>
//       CHECK:     } else {
//       CHECK:       scf.yield %[[c]], %[[c]] : tensor<2xi64>, tensor<2xi64>
//       CHECK:     return %[[v1]]#0, %[[v1]]#1 : tensor<2xi64>, tensor<2xi64>


// -----

func.func @if_ops_true_foldable() -> (tensor<2xi64>, tensor<2xi64>) {
  %pred = stablehlo.constant dense<true> : tensor<i1>
  %result0, %result1 = "stablehlo.if"(%pred) ({
    %0 = stablehlo.constant dense<0> : tensor<2xi64>
    %1 = stablehlo.constant dense<2> : tensor<2xi64>
    %3 = stablehlo.add %0, %1 : tensor<2xi64>
    stablehlo.return %3, %3 : tensor<2xi64>, tensor<2xi64>
  }, {
    %1 = stablehlo.constant dense<1> : tensor<2xi64>
    stablehlo.return %1, %1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
  func.return %result0, %result1 : tensor<2xi64>, tensor<2xi64>
}

// CHECK-LABEL: func.func @if_ops_true_foldable
//  CHECK-DAG:     %[[c:.+]] = stablehlo.constant dense<2> : tensor<2xi64>
//  CHECK-DAG:     %[[c_0:.+]] = stablehlo.constant dense<0> : tensor<2xi64>
//  CHECK-DAG:     %[[v0:.+]] = stablehlo.add %[[c_0]], %[[c]] : tensor<2xi64>
//  CHECK-DAG:     return %[[v0]], %[[v0]] : tensor<2xi64>, tensor<2xi64>

// -----

func.func @case_one_branch() -> (tensor<2xi64>, tensor<2xi64>) {
  %index = stablehlo.constant dense<0> : tensor<i32>
  %result_branch0 = stablehlo.constant dense<0> : tensor<2xi64>
  %result0, %result1 = "stablehlo.case"(%index) ({
    stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
  func.return %result0, %result1 : tensor<2xi64>, tensor<2xi64>
}

// CHECK-LABEL: func.func @case_one_branch
//       CHECK:   %[[c:.+]] = stablehlo.constant dense<0> : tensor<2xi64>
//       CHECK:   return %[[c]], %[[c]] : tensor<2xi64>, tensor<2xi64>

// -----

func.func @case_two_branches(%arg0: tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>) {
  %result_branch0 = stablehlo.constant dense<0> : tensor<2xi64>
  %result_branch1 = stablehlo.constant dense<1> : tensor<2xi64>
  %result0, %result1 = "stablehlo.case"(%arg0) ({
    stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
  },{stablehlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
  func.return %result0, %result1 : tensor<2xi64>, tensor<2xi64>
}

// CHECK-LABEL: func.func @case_two_branches
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>)
//   CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[c:.+]] = stablehlo.constant dense<0> : tensor<2xi64>
//   CHECK-DAG:     %[[c_0:.+]] = stablehlo.constant dense<1> : tensor<2xi64>
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][] : tensor<i32>
//   CHECK-DAG:     %[[v0:.+]] = arith.cmpi eq, %[[extracted]], %[[c0_i32]] : i32
//   CHECK-DAG:     %[[v1:.+]] = arith.select %[[v0]], %[[c]], %[[c_0]] : tensor<2xi64>
//   CHECK-DAG:     %[[v2:.+]] = arith.select %[[v0]], %[[c]], %[[c_0]] : tensor<2xi64>
//       CHECK:     return %[[v1]], %[[v2]] : tensor<2xi64>, tensor<2xi64>

// -----

func.func @case_three_branches(
      %index: tensor<i32>, %arg0: tensor<2xi64>, %arg1: tensor<2xi64>, %arg2: tensor<2xi64>)
   -> (tensor<2xi64>) {
  %result = "stablehlo.case"(%index) ({
    %0 = stablehlo.add %arg0, %arg1 : tensor<2xi64>
    %1 = stablehlo.multiply %0, %arg2 : tensor<2xi64>
    stablehlo.return %1 : tensor<2xi64>
  },{
    %0 = stablehlo.add %arg1, %arg2 : tensor<2xi64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<2xi64>
    stablehlo.return %1 : tensor<2xi64>
  },{
    %0 = stablehlo.add %arg2, %arg0 : tensor<2xi64>
    %1 = stablehlo.multiply %0, %arg1 : tensor<2xi64>
    stablehlo.return %1 : tensor<2xi64>
  }) : (tensor<i32>) -> (tensor<2xi64>)
  func.return %result : tensor<2xi64>
}

// CHECK-LABEL: func.func @case_three_branches
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>, %[[arg1:.+]]: tensor<2xi64>, %[[arg2:.+]]: tensor<2xi64>, %[[arg3:.+]]: tensor<2xi64>)
//   CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][] : tensor<i32>
//   CHECK-DAG:     %[[v0:.+]] = arith.cmpi eq, %[[extracted]], %[[c0_i32]] : i32
//   CHECK-DAG:     %[[v1:.+]] = scf.if %[[v0]] -> (tensor<2xi64>) {
//       CHECK:       %[[v2:.+]] = stablehlo.add %[[arg1]], %[[arg2]] : tensor<2xi64>
//       CHECK:       %[[v3:.+]] = stablehlo.multiply %[[v2]], %[[arg3]] : tensor<2xi64>
//       CHECK:       scf.yield %[[v3]] : tensor<2xi64>
//       CHECK:     } else {
//   CHECK-DAG:       %[[extracted_0:.+]] = tensor.extract %[[arg0]][] : tensor<i32>
//   CHECK-DAG:       %[[v2:.+]] = arith.cmpi eq, %[[extracted_0]], %[[c1_i32]] : i32
//       CHECK:       %[[v3:.+]] = scf.if %[[v2]] -> (tensor<2xi64>) {
//   CHECK-DAG:         %[[v4:.+]] = stablehlo.add %[[arg2]], %[[arg3]] : tensor<2xi64>
//   CHECK-DAG:         %[[v5:.+]] = stablehlo.multiply %[[v4]], %[[arg1]] : tensor<2xi64>
//       CHECK:         scf.yield %[[v5]] : tensor<2xi64>
//       CHECK:       } else {
//   CHECK-DAG:         %[[v4:.+]] = stablehlo.add %[[arg3]], %[[arg1]] : tensor<2xi64>
//   CHECK-DAG:         %[[v5:.+]] = stablehlo.multiply %[[v4]], %[[arg2]] : tensor<2xi64>
//   CHECK-DAG:         scf.yield %[[v5]] : tensor<2xi64>
//       CHECK:       scf.yield %[[v3]] : tensor<2xi64>
//       CHECK:     return %[[v1]] : tensor<2xi64>
