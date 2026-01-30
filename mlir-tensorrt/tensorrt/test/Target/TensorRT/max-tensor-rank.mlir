// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

!tensor_type = tensor<8x8x8x8x8x8x8x8xf32>

// Check that we can convert a network with tensor ranks of max allowed by TensorRT (rank 8).
func.func @trt_max_rank(%arg1: !tensor_type, %arg2: !tensor_type) -> (!tensor_type) {
  %1 = tensorrt.element_wise <kSUM> (%arg1, %arg2 : !tensor_type, !tensor_type)
    -> !tensor_type
  return %1 : !tensor_type
}

// CHECK-LABEL: @trt_max_rank
//  CHECK-SAME: tensorrt.engine
