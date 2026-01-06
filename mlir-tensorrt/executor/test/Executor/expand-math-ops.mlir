// RUN: executor-opt --executor-expand-math-ops -split-input-file %s | FileCheck %s
// RUN: executor-opt --executor-expand-math-ops -convert-std-to-executor -split-input-file %s

//===----------------------------------------------------------------------===//
// Test that math.sinh is expanded to (exp(x) - exp(-x)) / 2
// sinh is NOT supported by Executor, so it must be expanded.
//===----------------------------------------------------------------------===//

func.func @test_sinh_f32(%arg0: f32) -> f32 {
  %0 = math.sinh %arg0 : f32
  return %0 : f32
}
// CHECK-LABEL: func.func @test_sinh_f32
//   CHECK-NOT:   math.sinh
//   CHECK-DAG:   math.exp
//   CHECK-DAG:   arith.negf
//   CHECK-DAG:   arith.subf
//       CHECK:   return

func.func @test_sinh_f16(%arg0: f16) -> f16 {
  %0 = math.sinh %arg0 : f16
  return %0 : f16
}
// CHECK-LABEL: func.func @test_sinh_f16
//   CHECK-NOT:   math.sinh
//       CHECK:   return

func.func @test_sinh_bf16(%arg0: bf16) -> bf16 {
  %0 = math.sinh %arg0 : bf16
  return %0 : bf16
}
// CHECK-LABEL: func.func @test_sinh_bf16
//   CHECK-NOT:   math.sinh
//       CHECK:   return

func.func @test_sinh_f8e4m3fn(%arg0: f8E4M3FN) -> f8E4M3FN {
  %0 = math.sinh %arg0 : f8E4M3FN
  return %0 : f8E4M3FN
}
// CHECK-LABEL: func.func @test_sinh_f8e4m3fn
//   CHECK-NOT:   math.sinh
//       CHECK:   return

// -----

//===----------------------------------------------------------------------===//
// Test that math.cosh is expanded to (exp(x) + exp(-x)) / 2
// cosh is NOT supported by Executor, so it must be expanded.
//===----------------------------------------------------------------------===//

func.func @test_cosh_f32(%arg0: f32) -> f32 {
  %0 = math.cosh %arg0 : f32
  return %0 : f32
}
// CHECK-LABEL: func.func @test_cosh_f32
//   CHECK-NOT:   math.cosh
//   CHECK-DAG:   math.exp
//   CHECK-DAG:   arith.negf
//   CHECK-DAG:   arith.addf
//       CHECK:   return

func.func @test_cosh_f16(%arg0: f16) -> f16 {
  %0 = math.cosh %arg0 : f16
  return %0 : f16
}
// CHECK-LABEL: func.func @test_cosh_f16
//   CHECK-NOT:   math.cosh
//       CHECK:   return

func.func @test_cosh_bf16(%arg0: bf16) -> bf16 {
  %0 = math.cosh %arg0 : bf16
  return %0 : bf16
}
// CHECK-LABEL: func.func @test_cosh_bf16
//   CHECK-NOT:   math.cosh
//       CHECK:   return

func.func @test_cosh_f8e4m3fn(%arg0: f8E4M3FN) -> f8E4M3FN {
  %0 = math.cosh %arg0 : f8E4M3FN
  return %0 : f8E4M3FN
}
// CHECK-LABEL: func.func @test_cosh_f8e4m3fn
//   CHECK-NOT:   math.cosh
//       CHECK:   return

// -----

//===----------------------------------------------------------------------===//
// Test that math.rsqrt is expanded to 1 / sqrt(x)
// rsqrt is NOT supported by Executor, so it must be expanded.
//===----------------------------------------------------------------------===//

func.func @test_rsqrt_f32(%arg0: f32) -> f32 {
  %0 = math.rsqrt %arg0 : f32
  return %0 : f32
}
// CHECK-LABEL: func.func @test_rsqrt_f32
//   CHECK-NOT:   math.rsqrt
//       CHECK:   math.sqrt
//       CHECK:   arith.divf
//       CHECK:   return

func.func @test_rsqrt_f16(%arg0: f16) -> f16 {
  %0 = math.rsqrt %arg0 : f16
  return %0 : f16
}
// CHECK-LABEL: func.func @test_rsqrt_f16
//   CHECK-NOT:   math.rsqrt
//       CHECK:   return

func.func @test_rsqrt_bf16(%arg0: bf16) -> bf16 {
  %0 = math.rsqrt %arg0 : bf16
  return %0 : bf16
}
// CHECK-LABEL: func.func @test_rsqrt_bf16
//   CHECK-NOT:   math.rsqrt
//       CHECK:   return

func.func @test_rsqrt_f8e4m3fn(%arg0: f8E4M3FN) -> f8E4M3FN {
  %0 = math.rsqrt %arg0 : f8E4M3FN
  return %0 : f8E4M3FN
}
// CHECK-LABEL: func.func @test_rsqrt_f8e4m3fn
//   CHECK-NOT:   math.rsqrt
//       CHECK:   return

// -----

//===----------------------------------------------------------------------===//
// Test that math.fma is expanded to a * b + c
// fma is NOT supported by Executor, so it must be expanded.
//===----------------------------------------------------------------------===//

func.func @test_fma_f32(%a: f32, %b: f32, %c: f32) -> f32 {
  %0 = math.fma %a, %b, %c : f32
  return %0 : f32
}
// CHECK-LABEL: func.func @test_fma_f32
//   CHECK-NOT:   math.fma
//       CHECK:   arith.mulf
//       CHECK:   arith.addf
//       CHECK:   return

func.func @test_fma_f16(%a: f16, %b: f16, %c: f16) -> f16 {
  %0 = math.fma %a, %b, %c : f16
  return %0 : f16
}
// CHECK-LABEL: func.func @test_fma_f16
//   CHECK-NOT:   math.fma
//       CHECK:   arith.mulf
//       CHECK:   arith.addf
//       CHECK:   return

func.func @test_fma_bf16(%a: bf16, %b: bf16, %c: bf16) -> bf16 {
  %0 = math.fma %a, %b, %c : bf16
  return %0 : bf16
}
// CHECK-LABEL: func.func @test_fma_bf16
//   CHECK-NOT:   math.fma
//       CHECK:   arith.mulf
//       CHECK:   arith.addf
//       CHECK:   return

func.func @test_fma_f8e4m3fn(%a: f8E4M3FN, %b: f8E4M3FN, %c: f8E4M3FN) -> f8E4M3FN {
  %0 = math.fma %a, %b, %c : f8E4M3FN
  return %0 : f8E4M3FN
}
// CHECK-LABEL: func.func @test_fma_f8e4m3fn
//   CHECK-NOT:   math.fma
//       CHECK:   arith.mulf
//       CHECK:   arith.addf
//       CHECK:   return

// -----

//===----------------------------------------------------------------------===//
// Test that math.clampf is expanded to max(min(x, max), min)
// clampf is NOT supported by Executor, so it must be expanded.
//===----------------------------------------------------------------------===//

func.func @test_clampf_f32(%value: f32, %min: f32, %max: f32) -> f32 {
  %0 = math.clampf %value to [%min, %max] : f32
  return %0 : f32
}
// CHECK-LABEL: func.func @test_clampf_f32
//   CHECK-NOT:   math.clampf
//       CHECK:   arith.minimumf
//       CHECK:   arith.maximumf
//       CHECK:   return

func.func @test_clampf_f16(%value: f16, %min: f16, %max: f16) -> f16 {
  %0 = math.clampf %value to [%min, %max] : f16
  return %0 : f16
}
// CHECK-LABEL: func.func @test_clampf_f16
//   CHECK-NOT:   math.clampf
//       CHECK:   arith.minimumf
//       CHECK:   arith.maximumf
//       CHECK:   return

func.func @test_clampf_bf16(%value: bf16, %min: bf16, %max: bf16) -> bf16 {
  %0 = math.clampf %value to [%min, %max] : bf16
  return %0 : bf16
}
// CHECK-LABEL: func.func @test_clampf_bf16
//   CHECK-NOT:   math.clampf
//       CHECK:   arith.minimumf
//       CHECK:   arith.maximumf
//       CHECK:   return

func.func @test_clampf_f8e4m3fn(%value: f8E4M3FN, %min: f8E4M3FN, %max: f8E4M3FN) -> f8E4M3FN {
  %0 = math.clampf %value to [%min, %max] : f8E4M3FN
  return %0 : f8E4M3FN
}
// CHECK-LABEL: func.func @test_clampf_f8e4m3fn
//   CHECK-NOT:   math.clampf
//       CHECK:   arith.minimumf
//       CHECK:   arith.maximumf
//       CHECK:   return

// -----

//===----------------------------------------------------------------------===//
// Test that math.asinh is expanded
// asinh is NOT supported by Executor, so it must be expanded.
//===----------------------------------------------------------------------===//

func.func @test_asinh_f32(%arg0: f32) -> f32 {
  %0 = math.asinh %arg0 : f32
  return %0 : f32
}
// CHECK-LABEL: func.func @test_asinh_f32
//   CHECK-NOT:   math.asinh
//       CHECK:   return

func.func @test_asinh_f16(%arg0: f16) -> f16 {
  %0 = math.asinh %arg0 : f16
  return %0 : f16
}
// CHECK-LABEL: func.func @test_asinh_f16
//   CHECK-NOT:   math.asinh
//       CHECK:   return

func.func @test_asinh_bf16(%arg0: bf16) -> bf16 {
  %0 = math.asinh %arg0 : bf16
  return %0 : bf16
}
// CHECK-LABEL: func.func @test_asinh_bf16
//   CHECK-NOT:   math.asinh
//       CHECK:   return

func.func @test_asinh_f8e4m3fn(%arg0: f8E4M3FN) -> f8E4M3FN {
  %0 = math.asinh %arg0 : f8E4M3FN
  return %0 : f8E4M3FN
}
// CHECK-LABEL: func.func @test_asinh_f8e4m3fn
//   CHECK-NOT:   math.asinh
//       CHECK:   return

// -----

//===----------------------------------------------------------------------===//
// Test that math.acosh is expanded
// acosh is NOT supported by Executor, so it must be expanded.
//===----------------------------------------------------------------------===//

func.func @test_acosh_f32(%arg0: f32) -> f32 {
  %0 = math.acosh %arg0 : f32
  return %0 : f32
}
// CHECK-LABEL: func.func @test_acosh_f32
//   CHECK-NOT:   math.acosh
//       CHECK:   return

func.func @test_acosh_f16(%arg0: f16) -> f16 {
  %0 = math.acosh %arg0 : f16
  return %0 : f16
}
// CHECK-LABEL: func.func @test_acosh_f16
//   CHECK-NOT:   math.acosh
//       CHECK:   return

func.func @test_acosh_bf16(%arg0: bf16) -> bf16 {
  %0 = math.acosh %arg0 : bf16
  return %0 : bf16
}
// CHECK-LABEL: func.func @test_acosh_bf16
//   CHECK-NOT:   math.acosh
//       CHECK:   return

func.func @test_acosh_f8e4m3fn(%arg0: f8E4M3FN) -> f8E4M3FN {
  %0 = math.acosh %arg0 : f8E4M3FN
  return %0 : f8E4M3FN
}
// CHECK-LABEL: func.func @test_acosh_f8e4m3fn
//   CHECK-NOT:   math.acosh
//       CHECK:   return

// -----

//===----------------------------------------------------------------------===//
// Test that math.atanh is expanded
// atanh is NOT supported by Executor, so it must be expanded.
//===----------------------------------------------------------------------===//

func.func @test_atanh_f32(%arg0: f32) -> f32 {
  %0 = math.atanh %arg0 : f32
  return %0 : f32
}
// CHECK-LABEL: func.func @test_atanh_f32
//   CHECK-NOT:   math.atanh
//       CHECK:   return

func.func @test_atanh_f16(%arg0: f16) -> f16 {
  %0 = math.atanh %arg0 : f16
  return %0 : f16
}
// CHECK-LABEL: func.func @test_atanh_f16
//   CHECK-NOT:   math.atanh
//       CHECK:   return

func.func @test_atanh_bf16(%arg0: bf16) -> bf16 {
  %0 = math.atanh %arg0 : bf16
  return %0 : bf16
}
// CHECK-LABEL: func.func @test_atanh_bf16
//   CHECK-NOT:   math.atanh
//       CHECK:   return

func.func @test_atanh_f8e4m3fn(%arg0: f8E4M3FN) -> f8E4M3FN {
  %0 = math.atanh %arg0 : f8E4M3FN
  return %0 : f8E4M3FN
}
// CHECK-LABEL: func.func @test_atanh_f8e4m3fn
//   CHECK-NOT:   math.atanh
//       CHECK:   return

// -----

//===----------------------------------------------------------------------===//
// Test that math.powf is expanded
// powf is NOT supported by Executor, so it must be expanded.
//===----------------------------------------------------------------------===//

func.func @test_powf_f32(%x: f32, %y: f32) -> f32 {
  %0 = math.powf %x, %y : f32
  return %0 : f32
}
// CHECK-LABEL: func.func @test_powf_f32
//   CHECK-NOT:   math.powf
//       CHECK:   return

func.func @test_powf_f16(%x: f16, %y: f16) -> f16 {
  %0 = math.powf %x, %y : f16
  return %0 : f16
}
// CHECK-LABEL: func.func @test_powf_f16
//   CHECK-NOT:   math.powf
//       CHECK:   return

func.func @test_powf_bf16(%x: bf16, %y: bf16) -> bf16 {
  %0 = math.powf %x, %y : bf16
  return %0 : bf16
}
// CHECK-LABEL: func.func @test_powf_bf16
//   CHECK-NOT:   math.powf
//       CHECK:   return

func.func @test_powf_f8e4m3fn(%x: f8E4M3FN, %y: f8E4M3FN) -> f8E4M3FN {
  %0 = math.powf %x, %y : f8E4M3FN
  return %0 : f8E4M3FN
}
// CHECK-LABEL: func.func @test_powf_f8e4m3fn
//   CHECK-NOT:   math.powf
//       CHECK:   return

// -----

//===----------------------------------------------------------------------===//
// Test that operations SUPPORTED by Executor are NOT expanded.
// These should remain as math.* ops and will be converted to executor.*
// ops by the convert-std-to-executor pass.
//===----------------------------------------------------------------------===//

func.func @test_supported_ops_not_expanded_f32(%arg0: f32) -> (f32, f32, f32, f32, f32) {
  %0 = math.sin %arg0 : f32
  %1 = math.cos %arg0 : f32
  %2 = math.exp %arg0 : f32
  %3 = math.tanh %arg0 : f32
  %4 = math.sqrt %arg0 : f32
  return %0, %1, %2, %3, %4 : f32, f32, f32, f32, f32
}
// CHECK-LABEL: func.func @test_supported_ops_not_expanded_f32
//       CHECK:   math.sin {{.*}} : f32
//       CHECK:   math.cos {{.*}} : f32
//       CHECK:   math.exp {{.*}} : f32
//       CHECK:   math.tanh {{.*}} : f32
//       CHECK:   math.sqrt {{.*}} : f32

func.func @test_supported_ops_not_expanded_f16(%arg0: f16) -> (f16, f16, f16, f16, f16) {
  %0 = math.sin %arg0 : f16
  %1 = math.cos %arg0 : f16
  %2 = math.exp %arg0 : f16
  %3 = math.tanh %arg0 : f16
  %4 = math.sqrt %arg0 : f16
  return %0, %1, %2, %3, %4 : f16, f16, f16, f16, f16
}
// CHECK-LABEL: func.func @test_supported_ops_not_expanded_f16
//       CHECK:   math.sin {{.*}} : f16
//       CHECK:   math.cos {{.*}} : f16
//       CHECK:   math.exp {{.*}} : f16
//       CHECK:   math.tanh {{.*}} : f16
//       CHECK:   math.sqrt {{.*}} : f16

func.func @test_supported_ops_not_expanded_bf16(%arg0: bf16) -> (bf16, bf16, bf16, bf16, bf16) {
  %0 = math.sin %arg0 : bf16
  %1 = math.cos %arg0 : bf16
  %2 = math.exp %arg0 : bf16
  %3 = math.tanh %arg0 : bf16
  %4 = math.sqrt %arg0 : bf16
  return %0, %1, %2, %3, %4 : bf16, bf16, bf16, bf16, bf16
}
// CHECK-LABEL: func.func @test_supported_ops_not_expanded_bf16
//       CHECK:   math.sin {{.*}} : bf16
//       CHECK:   math.cos {{.*}} : bf16
//       CHECK:   math.exp {{.*}} : bf16
//       CHECK:   math.tanh {{.*}} : bf16
//       CHECK:   math.sqrt {{.*}} : bf16

func.func @test_supported_ops_not_expanded_f8e4m3fn(%arg0: f8E4M3FN) -> (f8E4M3FN, f8E4M3FN, f8E4M3FN, f8E4M3FN, f8E4M3FN) {
  %0 = math.sin %arg0 : f8E4M3FN
  %1 = math.cos %arg0 : f8E4M3FN
  %2 = math.exp %arg0 : f8E4M3FN
  %3 = math.tanh %arg0 : f8E4M3FN
  %4 = math.sqrt %arg0 : f8E4M3FN
  return %0, %1, %2, %3, %4 : f8E4M3FN, f8E4M3FN, f8E4M3FN, f8E4M3FN, f8E4M3FN
}
// CHECK-LABEL: func.func @test_supported_ops_not_expanded_f8e4m3fn
//       CHECK:   math.sin {{.*}} : f8E4M3FN
//       CHECK:   math.cos {{.*}} : f8E4M3FN
//       CHECK:   math.exp {{.*}} : f8E4M3FN
//       CHECK:   math.tanh {{.*}} : f8E4M3FN
//       CHECK:   math.sqrt {{.*}} : f8E4M3FN

// -----

//===----------------------------------------------------------------------===//
// Test that log operations are NOT expanded (supported by Executor)
//===----------------------------------------------------------------------===//

func.func @test_log_ops_not_expanded_f32(%arg0: f32) -> (f32, f32, f32, f32) {
  %0 = math.log %arg0 : f32
  %1 = math.log2 %arg0 : f32
  %2 = math.log10 %arg0 : f32
  %3 = math.log1p %arg0 : f32
  return %0, %1, %2, %3 : f32, f32, f32, f32
}
// CHECK-LABEL: func.func @test_log_ops_not_expanded_f32
//       CHECK:   math.log {{.*}} : f32
//       CHECK:   math.log2 {{.*}} : f32
//       CHECK:   math.log10 {{.*}} : f32
//       CHECK:   math.log1p {{.*}} : f32

func.func @test_log_ops_not_expanded_f16(%arg0: f16) -> (f16, f16, f16, f16) {
  %0 = math.log %arg0 : f16
  %1 = math.log2 %arg0 : f16
  %2 = math.log10 %arg0 : f16
  %3 = math.log1p %arg0 : f16
  return %0, %1, %2, %3 : f16, f16, f16, f16
}
// CHECK-LABEL: func.func @test_log_ops_not_expanded_f16
//       CHECK:   math.log {{.*}} : f16
//       CHECK:   math.log2 {{.*}} : f16
//       CHECK:   math.log10 {{.*}} : f16
//       CHECK:   math.log1p {{.*}} : f16

func.func @test_log_ops_not_expanded_bf16(%arg0: bf16) -> (bf16, bf16, bf16, bf16) {
  %0 = math.log %arg0 : bf16
  %1 = math.log2 %arg0 : bf16
  %2 = math.log10 %arg0 : bf16
  %3 = math.log1p %arg0 : bf16
  return %0, %1, %2, %3 : bf16, bf16, bf16, bf16
}
// CHECK-LABEL: func.func @test_log_ops_not_expanded_bf16
//       CHECK:   math.log {{.*}} : bf16
//       CHECK:   math.log2 {{.*}} : bf16
//       CHECK:   math.log10 {{.*}} : bf16
//       CHECK:   math.log1p {{.*}} : bf16

func.func @test_log_ops_not_expanded_f8e4m3fn(%arg0: f8E4M3FN) -> (f8E4M3FN, f8E4M3FN, f8E4M3FN, f8E4M3FN) {
  %0 = math.log %arg0 : f8E4M3FN
  %1 = math.log2 %arg0 : f8E4M3FN
  %2 = math.log10 %arg0 : f8E4M3FN
  %3 = math.log1p %arg0 : f8E4M3FN
  return %0, %1, %2, %3 : f8E4M3FN, f8E4M3FN, f8E4M3FN, f8E4M3FN
}
// CHECK-LABEL: func.func @test_log_ops_not_expanded_f8e4m3fn
//       CHECK:   math.log {{.*}} : f8E4M3FN
//       CHECK:   math.log2 {{.*}} : f8E4M3FN
//       CHECK:   math.log10 {{.*}} : f8E4M3FN
//       CHECK:   math.log1p {{.*}} : f8E4M3FN

// -----

//===----------------------------------------------------------------------===//
// Test that erf is NOT expanded (supported by Executor)
// but erfc IS expanded (not supported by Executor)
//===----------------------------------------------------------------------===//

func.func @test_erf_erfc_f32(%arg0: f32) -> (f32, f32) {
  %0 = math.erf %arg0 : f32
  %1 = math.erfc %arg0 : f32
  return %0, %1 : f32, f32
}
// CHECK-LABEL: func.func @test_erf_erfc_f32
//       CHECK:   math.erf {{.*}} : f32
//   CHECK-NOT:   math.erfc
//       CHECK:   return

func.func @test_erf_erfc_f16(%arg0: f16) -> (f16, f16) {
  %0 = math.erf %arg0 : f16
  %1 = math.erfc %arg0 : f16
  return %0, %1 : f16, f16
}
// CHECK-LABEL: func.func @test_erf_erfc_f16
//       CHECK:   math.erf {{.*}} : f16
//   CHECK-NOT:   math.erfc
//       CHECK:   return

func.func @test_erf_erfc_bf16(%arg0: bf16) -> (bf16, bf16) {
  %0 = math.erf %arg0 : bf16
  %1 = math.erfc %arg0 : bf16
  return %0, %1 : bf16, bf16
}
// CHECK-LABEL: func.func @test_erf_erfc_bf16
//       CHECK:   math.erf {{.*}} : bf16
//   CHECK-NOT:   math.erfc
//       CHECK:   return

func.func @test_erf_erfc_f8e4m3fn(%arg0: f8E4M3FN) -> (f8E4M3FN, f8E4M3FN) {
  %0 = math.erf %arg0 : f8E4M3FN
  %1 = math.erfc %arg0 : f8E4M3FN
  return %0, %1 : f8E4M3FN, f8E4M3FN
}
// CHECK-LABEL: func.func @test_erf_erfc_f8e4m3fn
//       CHECK:   math.erf {{.*}} : f8E4M3FN
//   CHECK-NOT:   math.erfc
//       CHECK:   return

// -----

//===----------------------------------------------------------------------===//
// Test that atan2 is NOT expanded (supported by Executor)
//===----------------------------------------------------------------------===//

func.func @test_atan2_not_expanded_f32(%x: f32, %y: f32) -> f32 {
  %0 = math.atan2 %x, %y : f32
  return %0 : f32
}
// CHECK-LABEL: func.func @test_atan2_not_expanded_f32
//       CHECK:   math.atan2 {{.*}} : f32
//       CHECK:   return

func.func @test_atan2_not_expanded_f16(%x: f16, %y: f16) -> f16 {
  %0 = math.atan2 %x, %y : f16
  return %0 : f16
}
// CHECK-LABEL: func.func @test_atan2_not_expanded_f16
//       CHECK:   math.atan2 {{.*}} : f16
//       CHECK:   return

func.func @test_atan2_not_expanded_bf16(%x: bf16, %y: bf16) -> bf16 {
  %0 = math.atan2 %x, %y : bf16
  return %0 : bf16
}
// CHECK-LABEL: func.func @test_atan2_not_expanded_bf16
//       CHECK:   math.atan2 {{.*}} : bf16
//       CHECK:   return

func.func @test_atan2_not_expanded_f8e4m3fn(%x: f8E4M3FN, %y: f8E4M3FN) -> f8E4M3FN {
  %0 = math.atan2 %x, %y : f8E4M3FN
  return %0 : f8E4M3FN
}
// CHECK-LABEL: func.func @test_atan2_not_expanded_f8e4m3fn
//       CHECK:   math.atan2 {{.*}} : f8E4M3FN
//       CHECK:   return

// -----

//===----------------------------------------------------------------------===//
// Test that round, ceil, floor are NOT expanded (supported by Executor)
//===----------------------------------------------------------------------===//

func.func @test_rounding_ops_not_expanded_f32(%arg0: f32) -> (f32, f32, f32) {
  %0 = math.round %arg0 : f32
  %1 = math.ceil %arg0 : f32
  %2 = math.floor %arg0 : f32
  return %0, %1, %2 : f32, f32, f32
}
// CHECK-LABEL: func.func @test_rounding_ops_not_expanded_f32
//       CHECK:   math.round {{.*}} : f32
//       CHECK:   math.ceil {{.*}} : f32
//       CHECK:   math.floor {{.*}} : f32

func.func @test_rounding_ops_not_expanded_f16(%arg0: f16) -> (f16, f16, f16) {
  %0 = math.round %arg0 : f16
  %1 = math.ceil %arg0 : f16
  %2 = math.floor %arg0 : f16
  return %0, %1, %2 : f16, f16, f16
}
// CHECK-LABEL: func.func @test_rounding_ops_not_expanded_f16
//       CHECK:   math.round {{.*}} : f16
//       CHECK:   math.ceil {{.*}} : f16
//       CHECK:   math.floor {{.*}} : f16

func.func @test_rounding_ops_not_expanded_bf16(%arg0: bf16) -> (bf16, bf16, bf16) {
  %0 = math.round %arg0 : bf16
  %1 = math.ceil %arg0 : bf16
  %2 = math.floor %arg0 : bf16
  return %0, %1, %2 : bf16, bf16, bf16
}
// CHECK-LABEL: func.func @test_rounding_ops_not_expanded_bf16
//       CHECK:   math.round {{.*}} : bf16
//       CHECK:   math.ceil {{.*}} : bf16
//       CHECK:   math.floor {{.*}} : bf16

func.func @test_rounding_ops_not_expanded_f8e4m3fn(%arg0: f8E4M3FN) -> (f8E4M3FN, f8E4M3FN, f8E4M3FN) {
  %0 = math.round %arg0 : f8E4M3FN
  %1 = math.ceil %arg0 : f8E4M3FN
  %2 = math.floor %arg0 : f8E4M3FN
  return %0, %1, %2 : f8E4M3FN, f8E4M3FN, f8E4M3FN
}
// CHECK-LABEL: func.func @test_rounding_ops_not_expanded_f8e4m3fn
//       CHECK:   math.round {{.*}} : f8E4M3FN
//       CHECK:   math.ceil {{.*}} : f8E4M3FN
//       CHECK:   math.floor {{.*}} : f8E4M3FN

// -----

//===----------------------------------------------------------------------===//
// Test that cbrt is NOT expanded (supported by Executor)
//===----------------------------------------------------------------------===//

func.func @test_cbrt_not_expanded_f32(%arg0: f32) -> f32 {
  %0 = math.cbrt %arg0 : f32
  return %0 : f32
}
// CHECK-LABEL: func.func @test_cbrt_not_expanded_f32
//       CHECK:   math.cbrt {{.*}} : f32
//       CHECK:   return

func.func @test_cbrt_not_expanded_f16(%arg0: f16) -> f16 {
  %0 = math.cbrt %arg0 : f16
  return %0 : f16
}
// CHECK-LABEL: func.func @test_cbrt_not_expanded_f16
//       CHECK:   math.cbrt {{.*}} : f16
//       CHECK:   return

func.func @test_cbrt_not_expanded_bf16(%arg0: bf16) -> bf16 {
  %0 = math.cbrt %arg0 : bf16
  return %0 : bf16
}
// CHECK-LABEL: func.func @test_cbrt_not_expanded_bf16
//       CHECK:   math.cbrt {{.*}} : bf16
//       CHECK:   return

func.func @test_cbrt_not_expanded_f8e4m3fn(%arg0: f8E4M3FN) -> f8E4M3FN {
  %0 = math.cbrt %arg0 : f8E4M3FN
  return %0 : f8E4M3FN
}
// CHECK-LABEL: func.func @test_cbrt_not_expanded_f8e4m3fn
//       CHECK:   math.cbrt {{.*}} : f8E4M3FN
//       CHECK:   return

// -----

//===----------------------------------------------------------------------===//
// Test that exp2 and expm1 are NOT expanded (supported by Executor)
//===----------------------------------------------------------------------===//

func.func @test_exp_variants_not_expanded_f32(%arg0: f32) -> (f32, f32) {
  %0 = math.exp2 %arg0 : f32
  %1 = math.expm1 %arg0 : f32
  return %0, %1 : f32, f32
}
// CHECK-LABEL: func.func @test_exp_variants_not_expanded_f32
//       CHECK:   math.exp2 {{.*}} : f32
//       CHECK:   math.expm1 {{.*}} : f32

func.func @test_exp_variants_not_expanded_f16(%arg0: f16) -> (f16, f16) {
  %0 = math.exp2 %arg0 : f16
  %1 = math.expm1 %arg0 : f16
  return %0, %1 : f16, f16
}
// CHECK-LABEL: func.func @test_exp_variants_not_expanded_f16
//       CHECK:   math.exp2 {{.*}} : f16
//       CHECK:   math.expm1 {{.*}} : f16

func.func @test_exp_variants_not_expanded_bf16(%arg0: bf16) -> (bf16, bf16) {
  %0 = math.exp2 %arg0 : bf16
  %1 = math.expm1 %arg0 : bf16
  return %0, %1 : bf16, bf16
}
// CHECK-LABEL: func.func @test_exp_variants_not_expanded_bf16
//       CHECK:   math.exp2 {{.*}} : bf16
//       CHECK:   math.expm1 {{.*}} : bf16

func.func @test_exp_variants_not_expanded_f8e4m3fn(%arg0: f8E4M3FN) -> (f8E4M3FN, f8E4M3FN) {
  %0 = math.exp2 %arg0 : f8E4M3FN
  %1 = math.expm1 %arg0 : f8E4M3FN
  return %0, %1 : f8E4M3FN, f8E4M3FN
}
// CHECK-LABEL: func.func @test_exp_variants_not_expanded_f8e4m3fn
//       CHECK:   math.exp2 {{.*}} : f8E4M3FN
//       CHECK:   math.expm1 {{.*}} : f8E4M3FN
