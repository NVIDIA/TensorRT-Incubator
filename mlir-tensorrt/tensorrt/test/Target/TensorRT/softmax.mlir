// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(tensorrt-expand-ops,translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s

// CHECK-LABEL: @trt_softmax_f32
//  CHECK-SAME: tensorrt.engine
func.func @trt_softmax_f32(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
  %0 = tensorrt.softmax {
    axis = 1
  } %arg0 : tensor<1x3x224x224xf32>
  return %0 : tensor<1x3x224x224xf32>
}

// -----

// CHECK-LABEL: @trt_softmax_f16
//  CHECK-SAME: tensorrt.engine
func.func @trt_softmax_f16(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
  %0 = tensorrt.softmax {
    axis = 1
  } %arg0 : tensor<1x3x224x224xf16>
  return %0 : tensor<1x3x224x224xf16>
}

// -----

// CHECK-LABEL: @trt_ragged_softmax_f32
//  CHECK-SAME: tensorrt.engine
func.func @trt_ragged_softmax_f32(%arg0: tensor<1x3x5xf32>, %bounds: tensor<1x3x1xi32>) -> tensor<1x3x5xf32> {
  %0 = tensorrt.ragged_softmax ins (%arg0, %bounds : tensor<1x3x5xf32>, tensor<1x3x1xi32>) -> tensor<1x3x5xf32>
  return %0 : tensor<1x3x5xf32>
}
