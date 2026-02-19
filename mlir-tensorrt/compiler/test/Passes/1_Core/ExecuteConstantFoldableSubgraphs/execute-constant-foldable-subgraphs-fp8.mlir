// REQUIRES: all-gpus-support-fp8
// RUN: mlir-tensorrt-opt %s --plan-execute-constant-foldable-subgraphs --inline --canonicalize --split-input-file | FileCheck %s

func.func @test_f8() -> tensor<4xf8E4M3FN> {
  %0 = call @constant_subgraph() : () -> tensor<4xf8E4M3FN>
  return %0 : tensor<4xf8E4M3FN>
}
func.func private @constant_subgraph() -> tensor<4xf8E4M3FN> attributes {plan.constant_foldable} {
  %cst = stablehlo.constant dense<[3.820310e+00, 4.950000e+01, 1.899410e-01, 1.079690e+01]> : tensor<4xf8E4M3FN>
  %cst_0 = stablehlo.constant dense<[2.859380e+00, 2.023440e+01, 2.232500e+02, 3.310000e+02]> : tensor<4xf8E4M3FN>
  %0 = stablehlo.add %cst, %cst_0 : tensor<4xf8E4M3FN>
  return %0 : tensor<4xf8E4M3FN>
}

// CHECK-LABEL: @test_f8
//  CHECK-NEXT: %[[c:.+]] = arith.constant dense<[6.500000e+00, 6.400000e+01, 2.240000e+02, 3.200000e+02]> : tensor<4xf8E4M3FN>
//  CHECK-NEXT: return %[[c]] : tensor<4xf8E4M3FN>