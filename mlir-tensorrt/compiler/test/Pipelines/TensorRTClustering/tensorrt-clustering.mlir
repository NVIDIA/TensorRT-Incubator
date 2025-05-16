// RUN: mlir-tensorrt-opt %s -tensorrt-clustering-pipeline -split-input-file -verify-diagnostics | FileCheck %s

func.func @trt_relu(%arg0: tensor<?xf16> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[1],opt =[4],max=[7]>}) -> (tensor<?xf16>) {
  %0 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %arg0 : tensor<?xf16>
  return %0: tensor<?xf16>
}

// CHECK-LABEL: @trt_relu
// CHECK-DAG:   %[[v0:.+]] = tensorrt.call_alloc @trt_engines::@tensorrt_cluster
// CHECK-DAG: return %[[v0]]
// CHECK-DAG: @tensorrt_cluster(%arg0: tensor<?xf16> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1], opt = [4], max = [7]>})
// CHECK-DAG: %[[v1:.+]] = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>}
// CHECK-DAG: return %[[v1]]

// -----

func.func @trt_dynamic_out_with_no_profile(%arg0: tensor<3xf16>) -> tensor<?xf32> {
  %0 = tensorrt.cast %arg0 : tensor<3xf16> to tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL: @trt_dynamic_out_with_no_profile
// CHECK-DAG:   %[[v0:.+]] = tensorrt.call_alloc @trt_engines::@tensorrt_cluster
// CHECK-DAG: return %[[v0]]
// CHECK-DAG: @tensorrt_cluster(%arg0: tensor<3xf16>)
// CHECK-DAG: %[[v1:.+]] = tensorrt.cast
// CHECK-DAG: return %[[v1]]

// -----

func.func @trt_unused_argument(
  %arg0: tensor<?xf16> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[1],opt =[4],max=[7]>},
  %arg1: tensor<?xf16> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[1],opt =[4],max=[7]>}
) -> (tensor<?xf16>) {
  %0 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %arg0 : tensor<?xf16>
  return %0: tensor<?xf16>
}

// CHECK-LABEL: @trt_unused_argument
// CHECK-DAG:   %[[v0:.+]] = tensorrt.call_alloc @trt_engines::@tensorrt_cluster
// CHECK-DAG: return %[[v0]]
// CHECK-DAG: @tensorrt_cluster(%arg0: tensor<?xf16> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1], opt = [4], max = [7]>})
// CHECK-DAG: %[[v1:.+]] = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>}
// CHECK-DAG: return %[[v1]]

// -----

func.func @trt_argument_ordering(
    %arg0: tensor<?xf16> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[1],opt =[4],max=[7]>},
    %arg1: tensor<?xf16> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[3],opt =[9],max=[17]>}
) -> (tensor<?xf16>) {
  %0 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %arg1 : tensor<?xf16>
  %1 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %arg0 : tensor<?xf16>
  %2 = tensorrt.element_wise <kSUM>(%1, %0 : tensor<?xf16>, tensor<?xf16>) -> tensor<?xf16>
  return  %2: tensor<?xf16>
}

// CHECK-LABEL: @trt_argument_ordering
// CHECK-DAG:   %[[v0:.+]] = tensorrt.call_alloc @trt_engines::@tensorrt_cluster
// CHECK-DAG: return %[[v0]]
// CHECK-DAG: @tensorrt_cluster(%arg0: tensor<?xf16> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1], opt = [4], max = [7]>}, %arg1: tensor<?xf16> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [3], opt = [9], max = [17]>})

// -----

// expected-error @below {{Profile attribute (tensorrt.shape_profile) of argument 0 is not set}}
func.func @no_shape_profile(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %arg0 : tensor<?xf32>
  return %0: tensor<?xf32>
}

// -----

func.func @reorder_engine_arguments(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4x2xf32>) -> tensor<2x3x?xf32> {
  %cst_i32 = tensorrt.constant dense<1> : tensor<1xi32>
  %0 = tensorrt.shape %arg1 : tensor<4x2xf32> -> tensor<2xi32>
  %1 = tensorrt.slice %0[0][1][1] : tensor<2xi32> to tensor<1xi32>
  %2 = tensorrt.collapse_rank %1 : tensor<1xi32> to tensor<i32>
  %cst_i32_0 = tensorrt.constant dense<1> : tensor<1xi32>
  %3 = tensorrt.reshape %2 shape(%cst_i32_0: tensor<1xi32>) : tensor<i32> to tensor<?xi32>
  %4 = tensorrt.slice %0[1][1][1] : tensor<2xi32> to tensor<1xi32>
  %5 = tensorrt.collapse_rank %4 : tensor<1xi32> to tensor<i32>
  %cst_i32_1 = tensorrt.constant dense<1> : tensor<1xi32>
  %6 = tensorrt.reshape %5 shape(%cst_i32_1: tensor<1xi32>) : tensor<i32> to tensor<?xi32>
  %7 = tensorrt.concatenation {axis = 0 : i32} ins(%cst_i32, %3, %6 : tensor<1xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<3xi32>
  %8 = tensorrt.reshape %arg1 shape(%7: tensor<3xi32>) : tensor<4x2xf32> to tensor<?x?x?xf32>
  %9 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%arg0, %8 : tensor<2x3x4xf32>, tensor<?x?x?xf32>) -> tensor<2x3x?xf32>
  return %9 : tensor<2x3x?xf32>
}

// CHECK-LABEL: @reorder_engine_arguments
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x4xf32>, %[[arg1:.+]]: tensor<4x2xf32>) -> tensor<2x3x?xf32>
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.call_alloc @trt_engines::@tensorrt_cluster(%[[arg0]], %[[arg1]] : tensor<2x3x4xf32>, tensor<4x2xf32>) -> tensor<2x3x?xf32>
//  CHECK-NEXT: return %[[v0:.+]] : tensor<2x3x?xf32>
// CHECK-LABEL: tensorrt.module @trt_engines
// CHECK-LABEL: func.func @tensorrt_cluster
//  CHECK-SAME: (%[[arg2:.+]]: tensor<2x3x4xf32>, %[[arg3:.+]]: tensor<4x2xf32>) -> tensor<2x3x?xf32>