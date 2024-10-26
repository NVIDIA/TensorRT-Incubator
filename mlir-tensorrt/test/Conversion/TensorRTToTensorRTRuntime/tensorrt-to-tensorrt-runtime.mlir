// RUN: mlir-tensorrt-opt -split-input-file -convert-tensorrt-to-runtime -canonicalize %s | FileCheck %s

tensorrt.module @trt_engines {
  func.func @trt_func(%arg0: tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32> attributes {
    "tensorrt.engine" = dense<0> : vector<8xi8>
  } {
    %cst_f32 = tensorrt.constant dense<0.00392156886> : tensor<1xf32>
    %0 = tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64: 1, 1, 1, 1>, second_transpose = array<i64: 0, 1, 2, 3>, zero_is_placeholder = false} ins(%cst_f32 : tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %1 = tensorrt.element_wise <kPROD>(%arg0, %0 : tensor<1x3x256x256xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x256x256xf32>
    return %1 : tensor<1x3x256x256xf32>
  }
}
func.func @main(%arg0: tensor<1x3x256x256xf32>, %arg1: tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32> {
  %0 = tensor.empty() : tensor<1x3x256x256xf32>
  %1 = tensorrt.call @trt_engines::@trt_func(%arg0 : tensor<1x3x256x256xf32>) outs(%0 : tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32>
  return %1 : tensor<1x3x256x256xf32>
}

// CHECK-LABEL: @main
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x3x256x256xf32>, %[[arg1:.+]]: tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32> {
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<1x3x256x256xf32>
//       CHECK:     %[[v1:.+]] = trtrt.compile @trt_engines::@trt_func : !trtrt.context
//       CHECK:     %[[v2:.+]] = cuda.get_global_stream 0
//       CHECK:     %[[v3:.+]] = trtrt.enqueue %[[v1]] stream(%[[v2]]) (%[[arg0]]) outs(%[[v0]]) : (tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32>
//       CHECK:     return %[[v3]] : tensor<1x3x256x256xf32>

// -----

tensorrt.module @trt_engines {
  func.func @trt_func(%arg0: tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32> attributes {
    "tensorrt.engine" = dense<0> : vector<8xi8>
  } {
    %cst_f32 = tensorrt.constant dense<0.00392156886> : tensor<1xf32>
    %0 = tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64: 1, 1, 1, 1>, second_transpose = array<i64: 0, 1, 2, 3>, zero_is_placeholder = false} ins(%cst_f32 : tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %1 = tensorrt.element_wise <kPROD>(%arg0, %0 : tensor<1x3x256x256xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x256x256xf32>
    return %1 : tensor<1x3x256x256xf32>
  }
}
func.func @main(%arg0: tensor<1x3x256x256xf32>, %arg1: tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32> {
  %1 = tensorrt.call_alloc @trt_engines::@trt_func(%arg0 : tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32>
  return %1 : tensor<1x3x256x256xf32>
}

// CHECK-LABEL: @main
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x3x256x256xf32>, %[[arg1:.+]]: tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32> {
//       CHECK:     %[[v1:.+]] = trtrt.compile @trt_engines::@trt_func : !trtrt.context
//       CHECK:     %[[v2:.+]] = cuda.get_global_stream 0
//       CHECK:     %[[v3:.+]] = trtrt.enqueue_alloc %[[v1]] stream(%[[v2]]) (%[[arg0]]) : (tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32>
//       CHECK:     return %[[v3]] : tensor<1x3x256x256xf32>

// -----

// CHECK-LABEL: @convert_tensorrt_const
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   return
func.func @convert_tensorrt_const() -> tensor<10xf32> {
  %0 = tensorrt.constant dense<0.0> : tensor<10xf32>
  return %0 : tensor<10xf32>
}


// -----

tensorrt.module @trt_engines {
  func.func @main_region(%arg0: tensor<i32>) -> tensor<i1> {
    %cst_i32 = tensorrt.constant dense<10> : tensor<i32>
    %0 = tensorrt.element_wise <kLESS>(%arg0, %cst_i32 : tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
  func.func @main_region_0(%arg0: tensor<i32>, %arg1: tensor<10xf16>, %arg2: tensor<f16>) -> (tensor<f16>, tensor<i32>) {
    %cst_i32 = tensorrt.constant dense<0> : tensor<i32>
    %cst_i32_0 = tensorrt.constant dense<10> : tensor<i32>
    %cst_i32_1 = tensorrt.constant dense<1> : tensor<i32>
    %0 = tensorrt.element_wise <kLESS>(%arg0, %cst_i32 : tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = tensorrt.element_wise <kSUM>(%arg0, %cst_i32_0 : tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tensorrt.select ins(%0, %1, %arg0 : tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tensorrt.expand_rank %2 : tensor<i32> to tensor<1xi32>
    %4 = tensorrt.slice %arg1[%3: tensor<1xi32>][1][1] : tensor<10xf16> to tensor<1xf16>
    %5 = tensorrt.collapse_rank %4 : tensor<1xf16> to tensor<f16>
    %6 = tensorrt.element_wise <kSUM>(%arg2, %5 : tensor<f16>, tensor<f16>) -> tensor<f16>
    %7 = tensorrt.element_wise <kSUM>(%arg0, %cst_i32_1 : tensor<i32>, tensor<i32>) -> tensor<i32>
    return %6, %7 : tensor<f16>, tensor<i32>
  }
}
func.func public @main(%arg0: tensor<10xf16> {jax.arg_info = "input_tensor", mhlo.sharding = "{replicated}"}) -> (tensor<i32> {jax.result_info = "[0]"}, tensor<f16> {jax.result_info = "[1]"}) {
  %cst_i32 = tensorrt.constant dense<0> : tensor<i32>
  %cst_f16 = tensorrt.constant dense<0.000000e+00> : tensor<f16>
  %0:2 = scf.while (%arg1 = %cst_i32, %arg2 = %cst_f16) : (tensor<i32>, tensor<f16>) -> (tensor<i32>, tensor<f16>) {
    %1 = tensor.empty() : tensor<i1>
    %2 = tensorrt.call @trt_engines::@main_region(%arg1 : tensor<i32>) outs(%1 : tensor<i1>) -> tensor<i1>
    %extracted = tensor.extract %2[] : tensor<i1>
    scf.condition(%extracted) %arg1, %arg2 : tensor<i32>, tensor<f16>
  } do {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<f16>):
    %1 = tensor.empty() : tensor<f16>
    %2 = tensor.empty() : tensor<i32>
    %3:2 = tensorrt.call @trt_engines::@main_region_0(%arg1, %arg0, %arg2 : tensor<i32>, tensor<10xf16>, tensor<f16>) outs(%1, %2 : tensor<f16>, tensor<i32>) -> tensor<f16>, tensor<i32>
    scf.yield %3#1, %3#0 : tensor<i32>, tensor<f16>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<f16>
}

//       CHECK:   tensorrt.module @trt_engines {
// CHECK-LABEL:   func.func public @main
//       CHECK:     scf.while
//       CHECK:       %[[v1:.+]] = tensor.empty
//       CHECK:       %[[v2:.+]] = trtrt.compile
//       CHECK:       %[[v3:.+]] = cuda.get_global_stream 0
//       CHECK:       trtrt.enqueue %[[v2]] stream(%[[v3]]) (%{{.+}}) outs(%[[v1]]) : (tensor<i32>) -> tensor<i1>
//       CHECK:     } do {
//       CHECK:       %[[v1:.+]] = tensor.empty
//       CHECK:       %[[v2:.+]] = tensor.empty
//       CHECK:       %[[v3:.+]] = trtrt.compile
//       CHECK:       %[[v4:.+]] = cuda.get_global_stream 0
//       CHECK:       trtrt.enqueue %[[v3]] stream(%[[v4]]) host_tensor_args [0] (%{{.+}}, %{{.+}}, %{{.+}}) outs(%[[v1]], %[[v2]])
