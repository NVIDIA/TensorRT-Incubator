// RUN: mlir-tensorrt-opt %s -mtrt-scf-float-strength-reduce -split-input-file | FileCheck %s

// Test: Basic countdown loop with float counter - empty body gets optimized away
// This matches the pattern from out.mlir: start=1.0, step=-0.1, limit>=0.05

// CHECK-LABEL: func.func @countdown_loop_simple
// CHECK-NOT: scf.while
func.func @countdown_loop_simple(%arg0: tensor<1x10x32xf32>) -> (tensor<1x10x32xf32>, f32) {
  %cst_step = arith.constant -1.000000e-01 : f32
  %cst_init = arith.constant 1.000000e+00 : f32
  %cst_limit = arith.constant 5.000000e-02 : f32
  %result:2 = scf.while (%arg1 = %arg0, %f = %cst_init) : (tensor<1x10x32xf32>, f32) -> (tensor<1x10x32xf32>, f32) {
    %cond = arith.cmpf oge, %f, %cst_limit : f32
    scf.condition(%cond) %arg1, %f : tensor<1x10x32xf32>, f32
  } do {
  ^bb0(%arg1: tensor<1x10x32xf32>, %f: f32):
    %next_f = arith.addf %f, %cst_step : f32
    %acc = arith.addf %arg1, %arg1 : tensor<1x10x32xf32>
    scf.yield %acc, %next_f : tensor<1x10x32xf32>, f32
  }
  return %result#0, %result#1 : tensor<1x10x32xf32>, f32
}

// -----

// Test: Count-up loop with positive step - empty body gets optimized away

// CHECK-LABEL: func.func @countup_loop_simple
// CHECK-NEXT: %[[c10:.+]] = arith.constant 1.{{0+}}e+01 : f32
// CHECK-NEXT: return %[[c10]]
func.func @countup_loop_simple() -> f32 {
  %cst_step = arith.constant 5.000000e-01 : f32
  %cst_init = arith.constant 0.000000e+00 : f32
  %cst_limit = arith.constant 1.000000e+01 : f32
  %result = scf.while (%f = %cst_init) : (f32) -> f32 {
    %cond = arith.cmpf olt, %f, %cst_limit : f32
    scf.condition(%cond) %f : f32
  } do {
  ^bb0(%f: f32):
    %next_f = arith.addf %f, %cst_step : f32
    scf.yield %next_f : f32
  }
  return %result : f32
}

// -----

// Test: Loop with subtraction (equivalent to negative addition) - empty body gets optimized away

// CHECK-LABEL: func.func @loop_with_subf
// CHECK-NEXT: %[[c0:.+]] = arith.constant 0.{{0+}}e+00 : f32
// CHECK-NEXT: return %[[c0]]
func.func @loop_with_subf() -> f32 {
  %cst_step = arith.constant 2.500000e-01 : f32
  %cst_init = arith.constant 5.000000e+00 : f32
  %cst_limit = arith.constant 0.000000e+00 : f32
  %result = scf.while (%f = %cst_init) : (f32) -> f32 {
    %cond = arith.cmpf ogt, %f, %cst_limit : f32
    scf.condition(%cond) %f : f32
  } do {
  ^bb0(%f: f32):
    %next_f = arith.subf %f, %cst_step : f32
    scf.yield %next_f : f32
  }
  return %result : f32
}

// -----

// Test: Loop that cannot be transformed (non-constant step)

// CHECK-LABEL: func.func @loop_non_constant_step
// CHECK: scf.while
// CHECK-NOT: scf.for
func.func @loop_non_constant_step(%step: f32) -> f32 {
  %cst_init = arith.constant 1.000000e+00 : f32
  %cst_limit = arith.constant 0.000000e+00 : f32
  %result = scf.while (%f = %cst_init) : (f32) -> f32 {
    %cond = arith.cmpf ogt, %f, %cst_limit : f32
    scf.condition(%cond) %f : f32
  } do {
  ^bb0(%f: f32):
    %next_f = arith.subf %f, %step : f32
    scf.yield %next_f : f32
  }
  return %result : f32
}

// -----

// Test: Loop that cannot be transformed (non-constant init)

// CHECK-LABEL: func.func @loop_non_constant_init
// CHECK: scf.while
func.func @loop_non_constant_init(%init: f32) -> f32 {
  %cst_step = arith.constant 1.000000e-01 : f32
  %cst_limit = arith.constant 0.000000e+00 : f32
  %result = scf.while (%f = %init) : (f32) -> f32 {
    %cond = arith.cmpf ogt, %f, %cst_limit : f32
    scf.condition(%cond) %f : f32
  } do {
  ^bb0(%f: f32):
    %next_f = arith.subf %f, %cst_step : f32
    scf.yield %next_f : f32
  }
  return %result : f32
}

// -----

// Test: Loop with integer step that doesn't need scaling (step = 1.0) - empty body gets optimized away

// CHECK-LABEL: func.func @loop_integer_step
// CHECK-NEXT: %[[c10:.+]] = arith.constant 1.{{0+}}e+01 : f32
// CHECK-NEXT: return %[[c10]]
func.func @loop_integer_step() -> f32 {
  %cst_step = arith.constant 1.000000e+00 : f32
  %cst_init = arith.constant 0.000000e+00 : f32
  %cst_limit = arith.constant 1.000000e+01 : f32
  %result = scf.while (%f = %cst_init) : (f32) -> f32 {
    %cond = arith.cmpf olt, %f, %cst_limit : f32
    scf.condition(%cond) %f : f32
  } do {
  ^bb0(%f: f32):
    %next_f = arith.addf %f, %cst_step : f32
    scf.yield %next_f : f32
  }
  return %result : f32
}

// -----

// Test: Loop with uses of the float value in the body - transformed to scf.for with scaling

// CHECK-LABEL: func.func @loop_with_float_uses
// CHECK-NOT: scf.while
// CHECK: scf.for
// CHECK: arith.index_cast {{.*}} : index to i64
// CHECK: arith.sitofp {{.*}} : i64 to f32
// CHECK: arith.mulf
// CHECK: arith.addf
func.func @loop_with_float_uses(%arg0: f32) -> f32 {
  %cst_step = arith.constant -5.000000e-01 : f32
  %cst_init = arith.constant 5.000000e+00 : f32
  %cst_limit = arith.constant 0.000000e+00 : f32
  %result:2 = scf.while (%f = %cst_init, %acc = %arg0) : (f32, f32) -> (f32, f32) {
    %cond = arith.cmpf ogt, %f, %cst_limit : f32
    scf.condition(%cond) %f, %acc : f32, f32
  } do {
  ^bb0(%f: f32, %acc: f32):
    // Use the float value in computation
    %sum = arith.addf %acc, %f : f32
    %next_f = arith.addf %f, %cst_step : f32
    scf.yield %next_f, %sum : f32, f32
  }
  return %result#1 : f32
}

// -----

// Test: Float counter is used in comparison but NOT passed to after region.
// The condition passes a derived value instead. This should NOT be transformed.

// CHECK-LABEL: func.func @loop_float_not_passed_to_after
// CHECK: scf.while
func.func @loop_float_not_passed_to_after() -> f32 {
  %cst_step = arith.constant 1.000000e-01 : f32
  %cst_init = arith.constant 0.000000e+00 : f32
  %cst_limit = arith.constant 1.000000e+00 : f32
  // The before region uses %f for comparison, but passes a derived value to after
  %result = scf.while (%f = %cst_init) : (f32) -> f32 {
    %cond = arith.cmpf olt, %f, %cst_limit : f32
    // Pass a derived value (doubled), not the original %f
    %doubled = arith.addf %f, %f : f32
    scf.condition(%cond) %doubled : f32
  } do {
  ^bb0(%derived: f32):
    // The after block receives %derived, not the original counter
    %next = arith.subf %derived, %cst_step : f32
    scf.yield %next : f32
  }
  return %result : f32
}

// -----

// Test: Before region has more args than after region receives.
// Float counter (arg 0) is passed but arg 1 is not passed to after region.
// This verifies correct index tracking between before and after regions.

// CHECK-LABEL: func.func @loop_different_before_after_args
// CHECK-NEXT: %[[c5:.+]] = arith.constant 5.{{0+}}e+00 : f32
// CHECK-NEXT: return %[[c5]]
func.func @loop_different_before_after_args(%extra: f32) -> f32 {
  %cst_step = arith.constant 5.000000e-01 : f32
  %cst_init = arith.constant 0.000000e+00 : f32
  %cst_limit = arith.constant 5.000000e+00 : f32
  // Before region has 2 args: %f and %unused
  // Only %f is passed to after region (after has 1 arg)
  %result = scf.while (%f = %cst_init, %unused = %extra) : (f32, f32) -> f32 {
    %cond = arith.cmpf olt, %f, %cst_limit : f32
    // Only pass %f to after region, not %unused
    scf.condition(%cond) %f : f32
  } do {
  ^bb0(%f_after: f32):
    // After block only has 1 arg
    %next_f = arith.addf %f_after, %cst_step : f32
    // Need to yield 2 values back to before region
    scf.yield %next_f, %f_after : f32, f32
  }
  return %result : f32
}
