// RUN: mlir-tensorrt-opt -split-input-file -convert-tensorrt-to-runtime -canonicalize %s | FileCheck %s


func.func @test_call(%arg0: tensor<1x3x256x256xf32>, %arg1: tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32> {
  %0 = tensor.empty() : tensor<1x3x256x256xf32>
  %1 = tensorrt.call @trt_engines::@trt_func(%arg0 : tensor<1x3x256x256xf32>) outs(%0 : tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32>
  %2 = tensorrt.call @trt_engines::@trt_func(%1 : tensor<1x3x256x256xf32>) outs(%0 : tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32>
  return %2 : tensor<1x3x256x256xf32>
}

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

//   CHECK-DAG:   trtrt.compiled_func @trt_func_engine_data dense<0> : vector<8xi8>
// CHECK-LABEL: func.func @test_call
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x3x256x256xf32>, %[[arg1:.+]]: tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32> {
//   CHECK-DAG:     %[[v0:.+]] = tensor.empty() : tensor<1x3x256x256xf32>
//   CHECK-DAG:     %[[v1:.+]] = trtrt.get_function @trt_func_engine_data : !trtrt.context
//   CHECK-DAG:     %[[device:.+]] = cuda.get_active_device
//   CHECK-DAG:     %[[v2:.+]] = cuda.get_global_stream device(%[[device]]) [0]
//   CHECK-DAG:     %[[v3:.+]] = trtrt.enqueue %[[v1]] stream(%[[v2]]) (%[[arg0]]) outs(%[[v0]]) : (tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32>
//   CHECK-DAG:     %[[v4:.+]] = trtrt.get_function @trt_func_engine_data : !trtrt.context
//   CHECK-DAG:     %[[device:.+]] = cuda.get_active_device
//   CHECK-DAG:     %[[v5:.+]] = cuda.get_global_stream device(%[[device]]) [0]
//   CHECK-DAG:     %[[v6:.+]] = trtrt.enqueue %[[v4]] stream(%[[v5]]) (%[[v3]]) outs(%[[v0]]) : (tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32>
//   CHECK-DAG:     return %[[v6]] : tensor<1x3x256x256xf32>
//   CHECK-NOT:   tensorrt.module

// -----


func.func @test_alloc_call(%arg0: tensor<1x3x256x256xf32>, %arg1: tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32> {
  %1 = tensorrt.call_alloc @trt_engines::@trt_func(%arg0 : tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32>
  return %1 : tensor<1x3x256x256xf32>
}

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

//       CHECK:   trtrt.compiled_func @trt_func_engine_data dense<0> : vector<8xi8>
// CHECK-LABEL: func.func @test_alloc_call
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x3x256x256xf32>, %[[arg1:.+]]: tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32> {
//   CHECK-DAG:     %[[v0:.+]] = trtrt.get_function @trt_func_engine_data : !trtrt.context
//   CHECK-DAG:     %[[device:.+]] = cuda.get_active_device
//   CHECK-DAG:     %[[v1:.+]] = cuda.get_global_stream device(%[[device]]) [0]
//   CHECK-DAG:     %[[v2:.+]] = trtrt.enqueue_alloc %[[v0]] stream(%[[v1]]) (%[[arg0]]) : (tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32>
//   CHECK-DAG:     return %[[v2]] : tensor<1x3x256x256xf32>
//   CHECK-NOT:   tensorrt.module


// -----

// CHECK-LABEL: @convert_tensorrt_const
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   return
func.func @convert_tensorrt_const() -> tensor<10xf32> {
  %0 = tensorrt.constant dense<0.0> : tensor<10xf32>
  return %0 : tensor<10xf32>
}


// -----

// This test checks that the TensorKindAnalysis is correctly used to populate the 'host_tensor_args' attribute.
func.func @test_tensor_kind_analysis(%arg0: tensor<10xf16>) -> (tensor<i32>, tensor<f16>) {
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

tensorrt.module @trt_engines {
  func.func @main_region(%arg0: tensor<i32>) -> tensor<i1> attributes {
    "tensorrt.engine" = dense<0> : vector<8xi8>
  } {
    %cst_i32 = tensorrt.constant dense<10> : tensor<i32>
    %0 = tensorrt.element_wise <kLESS>(%arg0, %cst_i32 : tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
  func.func @main_region_0(%arg0: tensor<i32>, %arg1: tensor<10xf16>, %arg2: tensor<f16>) -> (tensor<f16>, tensor<i32>) attributes {
    "tensorrt.engine" = dense<0> : vector<8xi8>
  } {
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

//   CHECK-DAG:   trtrt.compiled_func @main_region_engine_data dense<0> : vector<8xi8>
//   CHECK-DAG:   trtrt.compiled_func @main_region_0_engine_data dense<0> : vector<8xi8>
// CHECK-LABEL: func.func @test_tensor_kind_analysis
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf16>) -> (tensor<i32>, tensor<f16>)
//   CHECK-DAG:     %[[cst:.+]] = arith.constant dense<0> : tensor<i32>
//   CHECK-DAG:     %[[cst_0:.+]] = arith.constant dense<0.000000e+00> : tensor<f16>
//       CHECK:     %[[v0:.+]]:2 = scf.while (%[[arg1]] = %[[cst:.+]], %[[arg2:.+]] = %[[cst_0]])
//   CHECK-DAG:       %[[v1:.+]] = tensor.empty() : tensor<i1>
//   CHECK-DAG:       %[[v2:.+]] = trtrt.get_function @main_region_engine_data : !trtrt.context
//   CHECK-DAG:       %[[device:.+]] = cuda.get_active_device
//   CHECK-DAG:       %[[v3:.+]] = cuda.get_global_stream device(%[[device]]) [0]
//   CHECK-DAG:       %[[v4:.+]] = trtrt.enqueue %[[v2]] stream(%[[v3]]) (%[[arg1]]) outs(%[[v1]]) : (tensor<i32>) -> tensor<i1>
//   CHECK-DAG:       %[[extracted:.+]] = tensor.extract %[[v4]][] : tensor<i1>
//   CHECK-DAG:       scf.condition(%[[extracted]]) %[[arg1]], %[[arg2]] : tensor<i32>, tensor<f16>
//       CHECK:     } do {
//       CHECK:     ^bb0(%[[arg1:.+]]: tensor<i32>, %[[arg2:.+]]: tensor<f16>):
//   CHECK-DAG:       %[[v1:.+]] = tensor.empty() : tensor<f16>
//   CHECK-DAG:       %[[v2:.+]] = tensor.empty() : tensor<i32>
//   CHECK-DAG:       %[[v3:.+]] = trtrt.get_function @main_region_0_engine_data : !trtrt.context
//   CHECK-DAG:       %[[device:.+]] = cuda.get_active_device
//   CHECK-DAG:       %[[v4:.+]] = cuda.get_global_stream device(%[[device]]) [0]
//   CHECK-DAG:       %[[v5:.+]]:2 = trtrt.enqueue %[[v3]] stream(%[[v4]]) host_tensor_args [0] (%[[arg1]], %[[arg0]], %[[arg2]]) outs(%[[v1]], %[[v2]])
//   CHECK-DAG:       scf.yield %[[v5]]#1, %[[v5]]#0
//   CHECK-DAG:     return %[[v0]]#0, %[[v0]]#1
//   CHECK-NOT:   tensorrt.module