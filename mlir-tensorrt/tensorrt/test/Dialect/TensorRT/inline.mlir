// RUN: tensorrt-opt -inline -split-input-file %s | FileCheck %s

func.func @outlined()->tensor<2x2xf32> {
    %0 = tensorrt.constant dense<2.0> : tensor<2x2xf32>
    %1 = tensorrt.element_wise <kSUM>(%0, %0 : tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

func.func @valid_inline() -> tensor<2x2xf32> {
    %0 = call @outlined() : () -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

//  CHECK-LABEL: @valid_inline
//   CHECK-NEXT: %[[v0:.+]] = tensorrt.constant
//   CHECK-NEXT: %[[v1:.+]] = tensorrt.constant
//   CHECK-NEXT: %[[v2:.+]] = tensorrt.element_wise <kSUM>(%[[v0]], %[[v1]] : {{.*}})
//   CHECK-NEXT: return %[[v2]] : tensor<2x2xf32>

// -----

tensorrt.module @engines {
  func.func @trt_callee(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    return %arg0: tensor<?xf32>
  }
}

func.func @invalid_inline(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = tensor.empty() : tensor<10xf32>
  %1 = tensorrt.call @engines::@trt_callee(%arg0 : tensor<10xf32>) outs(%0: tensor<10xf32>)
    -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

//  CHECK-LABEL: @invalid_inline
//   CHECK-NEXT: tensor.empty()
//   CHECK-NEXT: %[[v1:.+]] = tensorrt.call
//   CHECK-NEXT: return %[[v1]] : tensor<10xf32>

// -----

func.func @trt_callee(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  return %arg0: tensor<?xf32>
}

func.func @invalid_inline_same_sym_table(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = tensor.empty() : tensor<10xf32>
  %1 = tensorrt.call @trt_callee(%arg0 : tensor<10xf32>) outs(%0: tensor<10xf32>)
    -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

//  CHECK-LABEL: @invalid_inline_same_sym_table
//   CHECK-NEXT: tensor.empty()
//   CHECK-NEXT: %[[v1:.+]] = tensorrt.call
//   CHECK-NEXT: return %[[v1]] : tensor<10xf32>