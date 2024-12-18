// RUN: mlir-tensorrt-opt -split-input-file %s --convert-stablehlo-to-tensorrt="trt-major-version=8" -verify-diagnostics | FileCheck %s

func.func @hlo_add_i64(%arg0: tensor<?xi64>) -> tensor<?xi64> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<?xi64>
  return %0 : tensor<?xi64>
}

// CHECK-LABEL: hlo_add_i64
//  CHECK-NEXT: stablehlo.add

// -----

func.func @hlo_iota_i64() -> tensor<128xi64> {
  %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<128xi64>
  return %0 : tensor<128xi64>
}

// CHECK-LABEL: hlo_iota_i64
//  CHECK-NEXT: stablehlo.iota

// -----

func.func @hlo_logistic(%arg0: tensor<?xbf16>) -> tensor<?xbf16> {
  %0 = "stablehlo.logistic"(%arg0) {} : (tensor<?xbf16>) -> tensor<?xbf16>
  return %0 : tensor<?xbf16>
}

// CHECK-LABEL: hlo_logistic
//  CHECK-NEXT: stablehlo.logistic

// -----

func.func @hlo_sort2() -> tensor<3xi64> {
    %cst = stablehlo.constant dense<[3, 1, 2]> : tensor<3xi64>
    %1 = "stablehlo.sort"(%cst) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %2 = stablehlo.compare  LT, %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = false} : (tensor<3xi64>) -> tensor<3xi64>
    return %1 : tensor<3xi64>
}

// CHECK-LABEL: hlo_sort2
//  CHECK-NEXT: stablehlo.constant
//  CHECK-NEXT: stablehlo.sort