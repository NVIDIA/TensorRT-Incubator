
// RUN: executor-opt %s -split-input-file -convert-std-to-executor | FileCheck %s
// RUN: executor-opt %s -split-input-file -convert-std-to-executor -executor-lower-to-runtime-builtins | FileCheck %s --check-prefix=EXEC


func.func @test_absf_f16(%arg0: f16) -> f16 {
  %0 = math.absf %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_absf_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.absf
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_absf_f16(
//       EXEC: func.func @test_absf_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_absf_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_absf_f32(%arg0: f32) -> f32 {
  %0 = math.absf %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_absf_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.absf
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_absf_f32(
//       EXEC: func.func @test_absf_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_absf_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_absf_f64(%arg0: f64) -> f64 {
  %0 = math.absf %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_absf_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.absf
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_absf_f64(
//       EXEC: func.func @test_absf_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_absf_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_cbrt_f16(%arg0: f16) -> f16 {
  %0 = math.cbrt %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_cbrt_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.cbrt
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_cbrt_f16(
//       EXEC: func.func @test_cbrt_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_cbrt_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_cbrt_f32(%arg0: f32) -> f32 {
  %0 = math.cbrt %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_cbrt_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.cbrt
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_cbrt_f32(
//       EXEC: func.func @test_cbrt_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_cbrt_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_cbrt_f64(%arg0: f64) -> f64 {
  %0 = math.cbrt %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_cbrt_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.cbrt
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_cbrt_f64(
//       EXEC: func.func @test_cbrt_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_cbrt_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_ceil_f16(%arg0: f16) -> f16 {
  %0 = math.ceil %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_ceil_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.ceil
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_ceil_f16(
//       EXEC: func.func @test_ceil_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_ceil_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_ceil_f32(%arg0: f32) -> f32 {
  %0 = math.ceil %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_ceil_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.ceil
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_ceil_f32(
//       EXEC: func.func @test_ceil_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_ceil_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_ceil_f64(%arg0: f64) -> f64 {
  %0 = math.ceil %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_ceil_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.ceil
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_ceil_f64(
//       EXEC: func.func @test_ceil_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_ceil_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_cos_f16(%arg0: f16) -> f16 {
  %0 = math.cos %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_cos_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.cos
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_cos_f16(
//       EXEC: func.func @test_cos_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_cos_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_cos_f32(%arg0: f32) -> f32 {
  %0 = math.cos %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_cos_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.cos
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_cos_f32(
//       EXEC: func.func @test_cos_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_cos_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_cos_f64(%arg0: f64) -> f64 {
  %0 = math.cos %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_cos_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.cos
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_cos_f64(
//       EXEC: func.func @test_cos_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_cos_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_erf_f16(%arg0: f16) -> f16 {
  %0 = math.erf %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_erf_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.erf
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_erf_f16(
//       EXEC: func.func @test_erf_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_erf_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_erf_f32(%arg0: f32) -> f32 {
  %0 = math.erf %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_erf_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.erf
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_erf_f32(
//       EXEC: func.func @test_erf_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_erf_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_erf_f64(%arg0: f64) -> f64 {
  %0 = math.erf %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_erf_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.erf
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_erf_f64(
//       EXEC: func.func @test_erf_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_erf_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_exp_f16(%arg0: f16) -> f16 {
  %0 = math.exp %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_exp_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.exp
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_exp_f16(
//       EXEC: func.func @test_exp_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_exp_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_exp_f32(%arg0: f32) -> f32 {
  %0 = math.exp %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_exp_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.exp
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_exp_f32(
//       EXEC: func.func @test_exp_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_exp_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_exp_f64(%arg0: f64) -> f64 {
  %0 = math.exp %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_exp_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.exp
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_exp_f64(
//       EXEC: func.func @test_exp_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_exp_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_exp2_f16(%arg0: f16) -> f16 {
  %0 = math.exp2 %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_exp2_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.exp2
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_exp2_f16(
//       EXEC: func.func @test_exp2_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_exp2_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_exp2_f32(%arg0: f32) -> f32 {
  %0 = math.exp2 %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_exp2_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.exp2
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_exp2_f32(
//       EXEC: func.func @test_exp2_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_exp2_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_exp2_f64(%arg0: f64) -> f64 {
  %0 = math.exp2 %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_exp2_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.exp2
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_exp2_f64(
//       EXEC: func.func @test_exp2_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_exp2_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_expm1_f16(%arg0: f16) -> f16 {
  %0 = math.expm1 %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_expm1_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.expm1
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_expm1_f16(
//       EXEC: func.func @test_expm1_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_expm1_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_expm1_f32(%arg0: f32) -> f32 {
  %0 = math.expm1 %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_expm1_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.expm1
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_expm1_f32(
//       EXEC: func.func @test_expm1_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_expm1_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_expm1_f64(%arg0: f64) -> f64 {
  %0 = math.expm1 %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_expm1_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.expm1
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_expm1_f64(
//       EXEC: func.func @test_expm1_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_expm1_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_floor_f16(%arg0: f16) -> f16 {
  %0 = math.floor %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_floor_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.floor
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_floor_f16(
//       EXEC: func.func @test_floor_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_floor_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_floor_f32(%arg0: f32) -> f32 {
  %0 = math.floor %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_floor_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.floor
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_floor_f32(
//       EXEC: func.func @test_floor_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_floor_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_floor_f64(%arg0: f64) -> f64 {
  %0 = math.floor %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_floor_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.floor
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_floor_f64(
//       EXEC: func.func @test_floor_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_floor_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_log_f16(%arg0: f16) -> f16 {
  %0 = math.log %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_log_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.log
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_log_f16(
//       EXEC: func.func @test_log_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_log_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_log_f32(%arg0: f32) -> f32 {
  %0 = math.log %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_log_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.log
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_log_f32(
//       EXEC: func.func @test_log_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_log_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_log_f64(%arg0: f64) -> f64 {
  %0 = math.log %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_log_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.log
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_log_f64(
//       EXEC: func.func @test_log_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_log_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_log10_f16(%arg0: f16) -> f16 {
  %0 = math.log10 %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_log10_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.log10
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_log10_f16(
//       EXEC: func.func @test_log10_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_log10_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_log10_f32(%arg0: f32) -> f32 {
  %0 = math.log10 %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_log10_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.log10
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_log10_f32(
//       EXEC: func.func @test_log10_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_log10_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_log10_f64(%arg0: f64) -> f64 {
  %0 = math.log10 %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_log10_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.log10
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_log10_f64(
//       EXEC: func.func @test_log10_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_log10_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_log1p_f16(%arg0: f16) -> f16 {
  %0 = math.log1p %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_log1p_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.log1p
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_log1p_f16(
//       EXEC: func.func @test_log1p_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_log1p_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_log1p_f32(%arg0: f32) -> f32 {
  %0 = math.log1p %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_log1p_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.log1p
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_log1p_f32(
//       EXEC: func.func @test_log1p_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_log1p_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_log1p_f64(%arg0: f64) -> f64 {
  %0 = math.log1p %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_log1p_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.log1p
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_log1p_f64(
//       EXEC: func.func @test_log1p_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_log1p_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_log2_f16(%arg0: f16) -> f16 {
  %0 = math.log2 %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_log2_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.log2
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_log2_f16(
//       EXEC: func.func @test_log2_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_log2_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_log2_f32(%arg0: f32) -> f32 {
  %0 = math.log2 %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_log2_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.log2
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_log2_f32(
//       EXEC: func.func @test_log2_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_log2_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_log2_f64(%arg0: f64) -> f64 {
  %0 = math.log2 %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_log2_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.log2
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_log2_f64(
//       EXEC: func.func @test_log2_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_log2_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_negf_f16(%arg0: f16) -> f16 {
  %0 = arith.negf %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_negf_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.negf
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_negf_f16(
//       EXEC: func.func @test_negf_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_negf_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_negf_f32(%arg0: f32) -> f32 {
  %0 = arith.negf %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_negf_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.negf
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_negf_f32(
//       EXEC: func.func @test_negf_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_negf_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_negf_f64(%arg0: f64) -> f64 {
  %0 = arith.negf %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_negf_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.negf
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_negf_f64(
//       EXEC: func.func @test_negf_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_negf_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_sin_f16(%arg0: f16) -> f16 {
  %0 = math.sin %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_sin_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.sin
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_sin_f16(
//       EXEC: func.func @test_sin_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_sin_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_sin_f32(%arg0: f32) -> f32 {
  %0 = math.sin %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_sin_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.sin
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_sin_f32(
//       EXEC: func.func @test_sin_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_sin_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_sin_f64(%arg0: f64) -> f64 {
  %0 = math.sin %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_sin_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.sin
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_sin_f64(
//       EXEC: func.func @test_sin_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_sin_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_sqrt_f16(%arg0: f16) -> f16 {
  %0 = math.sqrt %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_sqrt_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.sqrt
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_sqrt_f16(
//       EXEC: func.func @test_sqrt_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_sqrt_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_sqrt_f32(%arg0: f32) -> f32 {
  %0 = math.sqrt %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_sqrt_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.sqrt
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_sqrt_f32(
//       EXEC: func.func @test_sqrt_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_sqrt_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_sqrt_f64(%arg0: f64) -> f64 {
  %0 = math.sqrt %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_sqrt_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.sqrt
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_sqrt_f64(
//       EXEC: func.func @test_sqrt_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_sqrt_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_tan_f16(%arg0: f16) -> f16 {
  %0 = math.tan %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_tan_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.tan
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_tan_f16(
//       EXEC: func.func @test_tan_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_tan_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_tan_f32(%arg0: f32) -> f32 {
  %0 = math.tan %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_tan_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.tan
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_tan_f32(
//       EXEC: func.func @test_tan_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_tan_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_tan_f64(%arg0: f64) -> f64 {
  %0 = math.tan %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_tan_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.tan
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_tan_f64(
//       EXEC: func.func @test_tan_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_tan_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_tanh_f16(%arg0: f16) -> f16 {
  %0 = math.tanh %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_tanh_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.tanh
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_tanh_f16(
//       EXEC: func.func @test_tanh_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_tanh_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_tanh_f32(%arg0: f32) -> f32 {
  %0 = math.tanh %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_tanh_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.tanh
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_tanh_f32(
//       EXEC: func.func @test_tanh_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_tanh_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_tanh_f64(%arg0: f64) -> f64 {
  %0 = math.tanh %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_tanh_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.tanh
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_tanh_f64(
//       EXEC: func.func @test_tanh_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_tanh_f64(
//  EXEC-NEXT:   return %[[v0]] : f64

// -----

func.func @test_round_f16(%arg0: f16) -> f16 {
  %0 = math.round %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @test_round_f16
//  CHECK-NEXT:   %[[v0:.+]] = executor.round
//  CHECK-NEXT:   return %[[v0]] : f16

// EXEC-LABEL: executor.func private @_round_f16(
//       EXEC: func.func @test_round_f16
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_round_f16(
//  EXEC-NEXT:   return %[[v0]] : f16

// -----

func.func @test_round_f32(%arg0: f32) -> f32 {
  %0 = math.round %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_round_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.round
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_round_f32(
//       EXEC: func.func @test_round_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_round_f32(
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_round_f64(%arg0: f64) -> f64 {
  %0 = math.round %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @test_round_f64
//  CHECK-NEXT:   %[[v0:.+]] = executor.round
//  CHECK-NEXT:   return %[[v0]] : f64

// EXEC-LABEL: executor.func private @_round_f64(
//       EXEC: func.func @test_round_f64
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_round_f64(
//  EXEC-NEXT:   return %[[v0]] : f64


// -----

func.func @test_atan2_f32(%arg0: f32, %arg1: f32) -> f32 {
  %0 = math.atan2 %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_atan2_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.atan2
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_atan2_f32(
//       EXEC: func.func @test_atan2_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_atan2_f32
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @test_copysign_f32(%arg0: f32, %arg1: f32) -> f32 {
  %0 = math.copysign %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_copysign_f32
//  CHECK-NEXT:   %[[v0:.+]] = executor.copysign
//  CHECK-NEXT:   return %[[v0]] : f32

// EXEC-LABEL: executor.func private @_copysign_f32(
//       EXEC: func.func @test_copysign_f32
//  EXEC-NEXT:   %[[v0:.+]] = executor.call @_copysign_f32
//  EXEC-NEXT:   return %[[v0]] : f32

// -----

func.func @cast_fp_to_si(%arg0: f16) -> i32 {
  %0 = executor.fptosi %arg0 : f16 to i32
  return %0 : i32
}

// EXEC-LABEL: func.func @cast_fp_to_si
//  EXEC-SAME: (%[[arg0:.+]]: f16) -> i32 {
//       EXEC:     %[[v0:.+]] = executor.call @_fptosi_f16_i32(%[[arg0]]) : (f16) -> i32
//       EXEC:     return %[[v0]] : i32
// -----

func.func @extf(%arg0: f16) -> f32 {
  %0 = executor.extf %arg0 : f16 to f32
  return %0 : f32
}

// EXEC-LABEL: func.func @extf
//  EXEC-SAME: (%[[arg0:.+]]: f16) -> f32 {
//       EXEC:     %[[v0:.+]] = executor.call @_extf_f16_f32(%[[arg0]]) : (f16) -> f32
//       EXEC:     return %[[v0]] : f32

// -----

func.func @truncf(%arg0: f32) -> f16 {
  %0 = executor.truncf %arg0 : f32 to f16
  return %0 : f16
}

// EXEC-LABEL: func.func @truncf
//  EXEC-SAME: (%[[arg0:.+]]: f32) -> f16 {
//       EXEC:     %[[v0:.+]] = executor.call @_truncf_f32_f16(%[[arg0]]) : (f32) -> f16
//       EXEC:     return %[[v0]] : f16

// -----

func.func @arith_remsi(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.remsi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: @arith_remsi
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> i32 {
//       CHECK:     %[[v0:.+]] = executor.sremi %[[arg0]], %[[arg1]] : i32
//       CHECK:     return %[[v0]] : i32
