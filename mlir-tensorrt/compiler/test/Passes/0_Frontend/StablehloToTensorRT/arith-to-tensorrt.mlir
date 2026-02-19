// RUN: mlir-tensorrt-opt -split-input-file %s --convert-stablehlo-to-tensorrt | FileCheck %s

func.func @arith_constant() -> tensor<1x128x64xf32> {
  %0 = "arith.constant"() {value = dense<1.0> : tensor<1x128x64xf32>} : () -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// CHECK-LABEL: @arith_constant
//       CHECK:   tensorrt.constant
//  CHECK-SAME:       dense<1.0{{.+}}> : tensor<1x128x64xf32>

// -----

func.func @arith_constant_bool() -> tensor<1x128x64xi1> {
  %0 = "arith.constant"() {value = dense<true> : tensor<1x128x64xi1>} : () -> tensor<1x128x64xi1>
  return %0 : tensor<1x128x64xi1>
}

// CHECK-LABEL: @arith_constant_bool
//       CHECK:   tensorrt.constant
//  CHECK-SAME:     : tensor<1x128x64xi32>
//       CHECK:   tensorrt.identity
//  CHECK-SAME:     to tensor<1x128x64xi1>
