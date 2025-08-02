// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: func.func @trt_host_input
//  CHECK-SAME: tensorrt.engine
func.func @trt_host_input(
        %arg0: tensor<?x4xf32> {tensorrt.dimension_names = {}, tensorrt.shape_profile = #tensorrt.shape_profile<min = [2, 4], opt = [4, 4], max = [6, 4]>}, 
        %arg1: tensor<i32> {tensorrt.host_tensor, tensorrt.value_bounds = #tensorrt.shape_profile<min = [1], opt = [2], max = [3]>}) 
        -> tensor<?x?xf32> {
  %cst_i32 = tensorrt.constant dense<1> : tensor<i32>
  %0 = tensorrt.element_wise <kSUM>(%arg0, %arg0 : tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %1 = tensorrt.shape %0 : tensor<?x4xf32> -> tensor<2xi32>
  %2 = tensorrt.slice %1[0][1][1] : tensor<2xi32> to tensor<1xi32>
  %3 = tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64>, second_transpose = array<i64>, zero_is_placeholder = false} ins(%2 : tensor<1xi32>) -> tensor<i32>
  %4 = tensorrt.element_wise <kPROD>(%3, %cst_i32 : tensor<i32>, tensor<i32>) -> tensor<i32>
  %5 = tensorrt.slice %1[1][1][1] : tensor<2xi32> to tensor<1xi32>
  %6 = tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64>, second_transpose = array<i64>, zero_is_placeholder = false} ins(%5 : tensor<1xi32>) -> tensor<i32>
  %7 = tensorrt.element_wise <kPROD>(%4, %6 : tensor<i32>, tensor<i32>) -> tensor<i32>
  %8 = tensorrt.element_wise <kPROD>(%arg1, %cst_i32 : tensor<i32>, tensor<i32>) -> tensor<i32>
  %9 = tensorrt.element_wise <kFLOOR_DIV>(%7, %8 : tensor<i32>, tensor<i32>) -> tensor<i32>
  %10 = tensorrt.shuffle {first_transpose = array<i64>, reshape = array<i64: 1>, second_transpose = array<i64: 0>, zero_is_placeholder = false} ins(%9 : tensor<i32>) -> tensor<?xi32>
  %11 = tensorrt.shuffle {first_transpose = array<i64>, reshape = array<i64: 1>, second_transpose = array<i64: 0>, zero_is_placeholder = false} ins(%arg1 : tensor<i32>) -> tensor<?xi32>
  %12 = tensorrt.concatenation {axis = 0 : i32} ins(%10, %11 : tensor<?xi32>, tensor<?xi32>) -> tensor<2xi32>
  %13 = tensorrt.shuffle {first_transpose = array<i64: 0, 1>, second_transpose = array<i64: 0, 1>, zero_is_placeholder = false} ins(%0, %12 : tensor<?x4xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %13 : tensor<?x?xf32>
}
