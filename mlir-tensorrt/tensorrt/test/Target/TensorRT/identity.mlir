// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

// CHECK-LABEL: @trt_identity_noop
//  CHECK-SAME: tensorrt.engine
func.func @trt_identity_noop(%arg0: tensor<10x128x64xf32>) -> tensor<10x128x64xf32> {
  %0 = tensorrt.identity %arg0 : tensor<10x128x64xf32> to tensor<10x128x64xf32>
  return %0 : tensor<10x128x64xf32>
}


// CHECK-LABEL: @trt_identity_cast_f32_i32_f32
//  CHECK-SAME: tensorrt.engine
func.func @trt_identity_cast_f32_i32_f32(%arg0: tensor<10x128x64xf32>) -> tensor<10x128x64xf32> {
  %0 = tensorrt.identity %arg0 : tensor<10x128x64xf32> to tensor<10x128x64xi32>
  %1 = tensorrt.identity %0 : tensor<10x128x64xi32> to tensor<10x128x64xf32>
  return %1 : tensor<10x128x64xf32>
}


// CHECK-LABEL: @trt_identity_cast_ui8_f32
//  CHECK-SAME: tensorrt.engine
func.func @trt_identity_cast_ui8_f32(%arg0: tensor<10xui8>) -> tensor<10xf32> {
  %0 = tensorrt.identity %arg0 : tensor<10xui8> to tensor<10xf32>
  return %0 : tensor<10xf32>
}
