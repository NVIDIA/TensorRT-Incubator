// REQUIRES: all-gpus-support-fp4
// REQUIRES: has-support-for-ptx-gte-86
// RUN: kernel-opt %s -split-input-file -pass-pipeline="builtin.module(convert-vector-to-scf,gpu.module(kernel-expand-memref-args,kernel-lower-to-nvvm,reconcile-unrealized-casts))" | FileCheck %s

gpu.module @arith_emulations [#nvvm.target<chip = "sm_120">] {
  func.func @arith_extf_f4e2m1_f16(%arg0: f4E2M1FN) -> f16 attributes {gpu.kernel} {
    %0 = arith.extf %arg0 : f4E2M1FN to f16
    return %0 : f16
  }
}

// CHECK-LABEL: @arith_extf_f4e2m1_f16
//  CHECK-SAME: (%[[arg0:.+]]: i4) -> f16 attributes {nvvm.kernel}
//   CHECK-DAG: %[[v0:.+]] = llvm.zext %[[arg0]] : i4 to i16
//   CHECK-DAG: %[[v1:.+]] = llvm.inline_asm asm_dialect = att "{\0A.reg .b8 byte_input;\0Acvt.u8.u16 byte_input, $1;\0Acvt.rn.f16x2.e2m1x2 $0, byte_input;\0A}", "=r,h" %[[v0]] : (i16) -> i32
//   CHECK-DAG: %[[v2:.+]] = llvm.trunc %[[v1]] : i32 to i16
//   CHECK-DAG: %[[v3:.+]] = llvm.bitcast %[[v2]] : i16 to f16
//   CHECK-DAG: llvm.return %[[v3]] : f16

// -----

gpu.module @arith_emulations2 [#nvvm.target<chip = "sm_120">] {
  func.func @arith_truncf_f16_f4e2m1(%arg0: f16) -> f4E2M1FN attributes {gpu.kernel} {
    %0 = arith.truncf %arg0 : f16 to f4E2M1FN
    return %0 : f4E2M1FN
  }
}

// CHECK-LABEL: @arith_truncf_f16_f4e2m1
//  CHECK-SAME: (%[[arg0:.+]]: f16) -> i4 attributes {nvvm.kernel}
//   CHECK-DAG: %[[v0:.+]] = llvm.fpext %[[arg0]] : f16 to f32
//   CHECK-DAG: %[[v1:.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//   CHECK-DAG: %[[v2:.+]] = llvm.inline_asm asm_dialect = att "{\0A.reg .b8 byte_result;\0Acvt.rn.satfinite.e2m1x2.f32 byte_result, $1, $2;\0Amov.b16 $0, {byte_result, 0};\0A}", "=h,r,r" %[[v1]], %[[v0]] : (f32, f32) -> i16
//   CHECK-DAG: %[[v3:.+]] = llvm.trunc %[[v2]] : i16 to i4
//   CHECK-DAG: llvm.return %[[v3]] : i4
