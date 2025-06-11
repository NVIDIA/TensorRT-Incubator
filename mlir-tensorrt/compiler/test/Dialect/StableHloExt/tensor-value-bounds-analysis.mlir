// RUN: mlir-tensorrt-opt %s -test-tensor-value-bounds-analysis -split-input-file 2>&1 | FileCheck %s

func.func @stablehlo_convert_bool_to_int() -> tensor<2xi32> {
  %cst = arith.constant dense<[true, false]> : tensor<2xi1>
  %0 = stablehlo.convert %cst {tag = "convert"} : (tensor<2xi1>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: func stablehlo_convert_bool_to_int:
//      CHECK: test_tag: convert:
// CHECK-NEXT:  operand #0: <[-1, -1], [0, 0]>
// CHECK-NEXT:  result #0: <[1, 1], [0, 0]>

// -----

func.func @stablehlo_convert_int_to_bool() -> tensor<2xi1> {
  %cst = arith.constant dense<[0, -53]> : tensor<2xi32>
  %0 = stablehlo.convert %cst {tag = "convert"} : (tensor<2xi32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK-LABEL: func stablehlo_convert_int_to_bool:
//      CHECK: test_tag: convert:
// CHECK-NEXT:  operand #0: <[0, 0], [-53, -53]>
// CHECK-NEXT:  result #0: <[0, 0], [-1, -1]>

// -----

func.func @stablehlo_convert_float_to_int() -> tensor<2xi32> {
  %cst = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = stablehlo.convert %cst {tag = "convert"} : (tensor<2xf32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: func stablehlo_convert_float_to_int:
//      CHECK: test_tag: convert:
// CHECK-NEXT:  operand #0: <<uninitialized>>
// CHECK-NEXT:  result #0: <<uninitialized>>

// -----

// Semantic for signed -> unsigned is currently undefined in stablehlo spec.

func.func @stablehlo_convert_int_to_uint() -> tensor<2xui32> {
  %cst = arith.constant dense<[2, -1]> : tensor<2xi32>
  %0 = stablehlo.convert %cst {tag = "convert"} : (tensor<2xi32>) -> tensor<2xui32>
  return %0 : tensor<2xui32>
}

// CHECK-LABEL: func stablehlo_convert_int_to_uint:
//      CHECK: test_tag: convert:
// CHECK-NEXT:  operand #0: <[2, 2], [-1, -1]>
// CHECK-NEXT:  result #0: <<uninitialized>>
