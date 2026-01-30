// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(func.func(tensorrt-legalize-int8),translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

// CHECK-LABEL: @constant_splat_fp16
//  CHECK-SAME: tensorrt.engine
func.func @constant_splat_fp16() -> (tensor<10xf16>, tensor<10xf16>) {
  %cst_f16 = tensorrt.constant dense<5.000000e-01> : tensor<10xf16>
  %cst_f16_elided = tensorrt.constant dense_resource<__elided__> : tensor<10xf16>
  return %cst_f16, %cst_f16_elided : tensor<10xf16>, tensor<10xf16>
}

// CHECK-LABEL: @constant_splat_int8_no_qdq
//  CHECK-SAME: tensorrt.engine
func.func @constant_splat_int8_no_qdq() -> tensor<10xi8> {
  %cst_i8 = tensorrt.constant dense<5> : tensor<10xi8>
  return %cst_i8 : tensor<10xi8>
}

