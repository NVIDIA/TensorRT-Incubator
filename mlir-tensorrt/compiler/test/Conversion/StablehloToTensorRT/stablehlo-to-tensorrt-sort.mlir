// RUN: mlir-tensorrt-opt -split-input-file -convert-stablehlo-to-tensorrt %s | FileCheck %s

func.func @key_value_sort_unsupported(%arg0: tensor<3x4xi32>, %arg1: tensor<3x4xi32>) -> (tensor<3x4xi32>, tensor<3x4xi32>) {
  %0:2 = "stablehlo.sort"(%arg0, %arg1) <{dimension = 1 : i64, is_stable = true}> ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
    %1 = stablehlo.compare  LT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  }) : (tensor<3x4xi32>, tensor<3x4xi32>) -> (tensor<3x4xi32>, tensor<3x4xi32>)
  return %0#0, %0#1 : tensor<3x4xi32>, tensor<3x4xi32>
}

// CHECK: func.func @key_value_sort_unsupported
// CHECK:  stablehlo.sort
