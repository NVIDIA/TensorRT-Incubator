// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-ext-lower-check-custom-calls | FileCheck %s

// expect_eq (int)

func.func @expect_eq_i32(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>) {
  stablehlo.custom_call @check.expect_eq(%a, %b) {has_side_effect = true} : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  return
}

// CHECK-LABEL: func.func @expect_eq_i32
// CHECK: tensor.cast
// CHECK: tensor.cast
// CHECK: call @check_expect_eq_

// CHECK-LABEL: func.func private @check_expect_eq_
// CHECK: stablehlo.compare  EQ
// CHECK: stablehlo.reduce(
// CHECK: cf.assert

// -----
// expect_almost_eq (complex)

func.func @expect_almost_eq_c32(%a: tensor<4xcomplex<f32>>, %b: tensor<4xcomplex<f32>>) {
  stablehlo.custom_call @check.expect_almost_eq(%a, %b) {has_side_effect = true} : (tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>) -> ()
  return
}

// CHECK-LABEL: func.func @expect_almost_eq_c32
// CHECK: call @check_expect_almost_eq_

// CHECK-LABEL: func.func private @check_expect_almost_eq_
// CHECK: stablehlo.real
// CHECK: stablehlo.imag
// CHECK: stablehlo.compare
// CHECK: cf.assert

// -----
// expect_close (complex)

func.func @expect_close_c64(%a: tensor<2xcomplex<f64>>, %b: tensor<2xcomplex<f64>>) {
  stablehlo.custom_call @check.expect_close(%a, %b) {has_side_effect = true} : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> ()
  return
}

// CHECK-LABEL: func.func @expect_close_c64
// CHECK: stablehlo.constant dense<0> : tensor<ui64>
// CHECK: stablehlo.constant dense<3> : tensor<ui64>
// CHECK: call @check_expect_close_

// CHECK-LABEL: func.func private @check_expect_close_
// CHECK: stablehlo.real
// CHECK: stablehlo.imag
// CHECK: stablehlo.maximum
// CHECK: cf.assert

// -----
// expect_eq_const (i1)

func.func @expect_eq_const_i1(%a: tensor<4xi1>, %b: tensor<4xi1>) {
  stablehlo.custom_call @check.expect_eq_const(%a, %b) {has_side_effect = true} : (tensor<4xi1>, tensor<4xi1>) -> ()
  return
}

// CHECK-LABEL: func.func @expect_eq_const_i1
// CHECK: call @check_expect_eq_

// -----
// expect_almost_eq_const (f32, explicit tolerance attr)

func.func @expect_almost_eq_const_f32(%a: tensor<2xf32>, %b: tensor<2xf32>) {
  stablehlo.custom_call @check.expect_almost_eq_const(%a, %b) {has_side_effect = true, tolerance = 1.000000e-04 : f64} : (tensor<2xf32>, tensor<2xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @expect_almost_eq_const_f32
// CHECK: call @check_expect_almost_eq_

// -----
// expect_serialized_eq (noop lowering)

func.func @expect_serialized_eq_noop(%a: tensor<3xcomplex<f32>>) {
  stablehlo.custom_call @check.expect_serialized_eq(%a) {has_side_effect = true, probe_id = "probe_c32", iter = 0 : i32} : (tensor<3xcomplex<f32>>) -> ()
  return
}

// CHECK-LABEL: func.func @expect_serialized_eq_noop
// CHECK: call @check_expect_serialized_eq_noop_
// CHECK-NOT: stablehlo.custom_call @check.expect_serialized_eq

// -----
// check.eq (returns tensor<i1> result, no internal assertion)

func.func @check_eq_scalar(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<i1> {
  %0 = stablehlo.custom_call @check.eq(%a, %b) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
  return %0 : tensor<i1>
}

// CHECK-LABEL: func.func @check_eq_scalar
// CHECK: tensor.cast
// CHECK: tensor.cast
// CHECK: %[[RESULT:.+]] = call @check_eq_{{.*}}({{.*}}) : (tensor<?xf32>, tensor<?xf32>) -> tensor<i1>
// CHECK: return %[[RESULT]] : tensor<i1>
// CHECK-NOT: stablehlo.custom_call @check.eq

// CHECK-LABEL: func.func private @check_eq_
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?xf32>, %[[ARG1:.+]]: tensor<?xf32>) -> tensor<i1>
// CHECK-SAME: attributes {no_inline}
// CHECK: stablehlo.reduce(
// CHECK-NOT: cf.assert
// CHECK: return
