// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s


// CHECK-LABEL: @trt_shuffle_infer_fp8
//  CHECK-sAME: tensorrt.engine
func.func @trt_shuffle_infer_fp8(%arg0: tensor<30x20x10x1xf8E4M3FN>) -> tensor<30x20x10xf8E4M3FN> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: -1, 0, 0>,
    second_transpose = array<i64: 0, 1, 2>,
    zero_is_placeholder = true
  } ins(%arg0 : tensor<30x20x10x1xf8E4M3FN>) -> tensor<30x20x10xf8E4M3FN>
  return %0 : tensor<30x20x10xf8E4M3FN>
}

// -----

// CHECK-LABEL: @trt_shuffle_infer_bf16
//  CHECK-sAME: tensorrt.engine
func.func @trt_shuffle_infer_bf16(%arg0: tensor<30x20x10x1xbf16>) -> tensor<30x20x10xbf16> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: -1, 0, 0>,
    second_transpose = array<i64: 0, 1, 2>,
    zero_is_placeholder = true
  } ins(%arg0 : tensor<30x20x10x1xbf16>) -> tensor<30x20x10xbf16>
  return %0 : tensor<30x20x10xbf16>
}
