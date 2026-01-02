// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-ext-lower-check-custom-calls | FileCheck %s

func.func @test_expect_close(%arg0: tensor<20x20xf32>, %arg1: tensor<20x20xf32>) -> tensor<20x20xf32> {
  stablehlo.custom_call @check.expect_close(%arg0, %arg1) {has_side_effect = true} : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
  return %arg0 : tensor<20x20xf32>
}

// CHECK-LABEL: func.func @test_expect_close
// CHECK-SAME: %[[INPUT0:.+]]: tensor<20x20xf32>, %[[INPUT1:.+]]: tensor<20x20xf32>
// CHECK: %[[CAST0:.+]] = tensor.cast %[[INPUT0]] : tensor<20x20xf32> to tensor<?x?xf32>
// CHECK: %[[CAST1:.+]] = tensor.cast %[[INPUT1]] : tensor<20x20xf32> to tensor<?x?xf32>
// CHECK: %[[MIN:.+]] = stablehlo.constant dense<0>
// CHECK: %[[MAX:.+]] = stablehlo.constant dense<3>
// CHECK: call @check_expect_close_0(%[[CAST0]], %[[CAST1]], %[[MIN]], %[[MAX]])
// CHECK: return %[[INPUT0]]

// CHECK-LABEL: func.func private @check_expect_close_0
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[MIN_ULP:.+]]: tensor<ui64>, %[[MAX_ULP:.+]]: tensor<ui64>)
// CHECK-SAME: attributes {no_inline}
// CHECK: %[[ACTUAL_BITS:.+]] = stablehlo.bitcast_convert %[[ARG0]] : (tensor<?x?xf32>) -> tensor<?x?xui32>
// CHECK: %[[EXPECTED_BITS:.+]] = stablehlo.bitcast_convert %[[ARG1]] : (tensor<?x?xf32>) -> tensor<?x?xui32>
// CHECK: %[[BITWISE_EQ:.+]] = stablehlo.compare
// CHECK: %[[MIN_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[MIN_ULP]], %{{.+}}
// CHECK: %[[MAX_BCAST:.+]] = stablehlo.dynamic_broadcast_in_dim %[[MAX_ULP]], %{{.+}}
// CHECK: %[[TRUE:.+]] = stablehlo.constant dense<true>
// CHECK: %[[REDUCE:.+]] = stablehlo.reduce(
// CHECK: %[[EXTRACTED:.+]] = tensor.extract %[[REDUCE]]
// CHECK: cf.assert %[[EXTRACTED]], "check_expect_close failed"
// CHECK: return

// -----

func.func @test_with_attributes(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) {
  stablehlo.custom_call @check.expect_close(%arg0, %arg1) {
    has_side_effect = true,
    min_ulp_difference = 0 : i64,
    max_ulp_difference = 5 : i64
  } : (tensor<10xf32>, tensor<10xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @test_with_attributes
// CHECK: %[[MIN:.+]] = stablehlo.constant dense<0>
// CHECK: %[[MAX:.+]] = stablehlo.constant dense<5>
// CHECK: call @check_expect_close_{{[0-9]+}}(%{{.+}}, %{{.+}}, %[[MIN]], %[[MAX]])

// -----

// Test with f16 type
func.func @test_f16(%arg0: tensor<5x5xf16>, %arg1: tensor<5x5xf16>) {
  stablehlo.custom_call @check.expect_close(%arg0, %arg1) {has_side_effect = true} : (tensor<5x5xf16>, tensor<5x5xf16>) -> ()
  return
}

// CHECK-LABEL: func.func @test_f16
// CHECK: call @check_expect_close_{{[0-9]+}}

// -----

// Test with f64 type
func.func @test_f64(%arg0: tensor<3x3xf64>, %arg1: tensor<3x3xf64>) {
  stablehlo.custom_call @check.expect_close(%arg0, %arg1) {has_side_effect = true} : (tensor<3x3xf64>, tensor<3x3xf64>) -> ()
  return
}

// CHECK-LABEL: func.func @test_f64
// CHECK: call @check_expect_close_{{[0-9]+}}

// -----

// Test that non-check custom calls are not affected
func.func @test_other_custom_call(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = stablehlo.custom_call @some_other_op(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: func.func @test_other_custom_call
// CHECK: stablehlo.custom_call @some_other_op
// CHECK-NOT: func.call @check_expect_close
