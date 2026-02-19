// RUN: mlir-tensorrt-opt %s -split-input-file -convert-stablehlo-to-arith | FileCheck %s

func.func @test_stablehlo_constant_to_arith() -> tensor<10xui32> {
  %0 = stablehlo.constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xui32>
  return %0 : tensor<10xui32>
}

// CHECK-LABEL: func.func @test_stablehlo_constant_to_arith
//   CHECK-DAG: %[[cst:.+]] = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi32>
//   CHECK-DAG: %[[v0:.+]] = builtin.unrealized_conversion_cast %[[cst]] : tensor<10xi32> to tensor<10xui32>
//   CHECK-DAG: return %[[v0]] : tensor<10xui32>

// -----

func.func @test_stablehlo_constant_to_arith() -> tensor<10xui32> {
  %0 = stablehlo.constant dense_resource<__elided__> : tensor<10xui32>
  return %0 : tensor<10xui32>
}

// CHECK-LABEL: func.func @test_stablehlo_constant_to_arith
//   CHECK-DAG: %[[cst:.+]] = arith.constant dense_resource<__elided__> : tensor<10xi32>
//   CHECK-DAG: %[[v0:.+]] = builtin.unrealized_conversion_cast %[[cst]] : tensor<10xi32> to tensor<10xui32>
//   CHECK-DAG: return %[[v0]] : tensor<10xui32>
