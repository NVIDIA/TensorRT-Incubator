// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32 %s | FileCheck %s

// CHECK-LABEL: @trt_matrix_multiply
//  CHECK-SAME: tensorrt.engine
func.func @trt_matrix_multiply(%arg0: tensor<10x128x64xf32>, %arg1: tensor<10x64x256xf32>)
            -> tensor<10x128x256xf32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<10x128x64xf32>, tensor<10x64x256xf32>) -> tensor<10x128x256xf32>
  return %0 : tensor<10x128x256xf32>
}


// CHECK-LABEL: @trt_matrix_vector
//  CHECK-SAME:   tensorrt.engine
func.func @trt_matrix_vector(%arg0: tensor<10x20xf32>, %arg1: tensor<20xf32>) -> tensor<10xf32> {
  %0 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kVECTOR>} ins(%arg0, %arg1 : tensor<10x20xf32>, tensor<20xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}


// CHECK-LABEL: @trt_matrix_multiply_f16
//  CHECK-SAME:   tensorrt.engine
func.func @trt_matrix_multiply_f16(%arg0: tensor<128x64xf16>, %arg1: tensor<64x128xf16>) -> tensor<128x128xf16> {
  %0 = "tensorrt.matrix_multiply" (%arg0, %arg1) {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  }: (tensor<128x64xf16>, tensor<64x128xf16>) -> tensor<128x128xf16>
  return %0 : tensor<128x128xf16>
}
