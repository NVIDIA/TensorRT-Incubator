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

// -----

func.func @maintain_output_order() -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
    %cst_f32 = tensorrt.constant dense<0.000000e+00> : tensor<f32>
    %cst_i32 = tensorrt.constant dense<[1, 2]> : tensor<2xi32>
    %0 = tensorrt.broadcast %cst_f32 broadcast_dims<> shape(%cst_i32 : tensor<2xi32>) : tensor<f32> to tensor<?x?xf32>
    %cst_f32_0 = tensorrt.constant dense<1.000000e+00> : tensor<f32>
    %cst_i32_1 = tensorrt.constant dense<[3, 4]> : tensor<2xi32>
    %1 = tensorrt.broadcast %cst_f32_0 broadcast_dims<> shape(%cst_i32_1 : tensor<2xi32>) : tensor<f32> to tensor<?x?xf32>
    %cst_f32_2 = tensorrt.constant dense<0.000000e+00> : tensor<f32>
    %cst_i32_3 = tensorrt.constant dense<[5, 6]> : tensor<2xi32>
    %2 = tensorrt.broadcast %cst_f32_2 broadcast_dims<> shape(%cst_i32_3 : tensor<2xi32>) : tensor<f32> to tensor<?x?xf32>
    %cst_f32_4 = tensorrt.constant dense<1.000000e+00> : tensor<f32>
    %cst_i32_5 = tensorrt.constant dense<[7, 8]> : tensor<2xi32>
    %3 = tensorrt.broadcast %cst_f32_4 broadcast_dims<> shape(%cst_i32_5 : tensor<2xi32>) : tensor<f32> to tensor<?x?xf32>
    return %0, %1, %2, %3 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

// CHECK-LABEL: @maintain_output_order
//  CHECK-SAME: () -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-NEXT: %[[v0:.+]]:4 = tensorrt.call_alloc @trt_engines::@tensorrt_cluster()
//  CHECK-NEXT: return %[[v0]]#0, %[[v0]]#1, %[[v0]]#2, %[[v0]]#3
// CHECK-LABEL: tensorrt.module @trt_engines
// CHECK-LABEL: func.func @tensorrt_cluster
//   CHECK-NEXT: %[[v0:.+]] = tensorrt.constant dense<0.000000e+00> : tensor<f32>
//   CHECK-NEXT: %[[v1:.+]] = tensorrt.constant dense<[1, 2]> : tensor<2xi32>
//   CHECK-NEXT: %[[v2:.+]] = tensorrt.broadcast %[[v0]] broadcast_dims<> shape(%[[v1]] : tensor<2xi32>)
//   CHECK-NEXT: %[[v3:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<f32>
//   CHECK-NEXT: %[[v4:.+]] = tensorrt.constant dense<[3, 4]> : tensor<2xi32>
//   CHECK-NEXT: %[[v5:.+]] = tensorrt.broadcast %[[v3]] broadcast_dims<> shape(%[[v4]] : tensor<2xi32>)
//   CHECK-NEXT: %[[v6:.+]] = tensorrt.constant dense<0.000000e+00> : tensor<f32>
//   CHECK-NEXT: %[[v7:.+]] = tensorrt.constant dense<[5, 6]> : tensor<2xi32>
//   CHECK-NEXT: %[[v8:.+]] = tensorrt.broadcast %[[v6]] broadcast_dims<> shape(%[[v7]] : tensor<2xi32>)
//   CHECK-NEXT: %[[v9:.+]] = tensorrt.constant dense<1.000000e+00> : tensor<f32>
//   CHECK-NEXT: %[[v10:.+]] = tensorrt.constant dense<[7, 8]> : tensor<2xi32>
//   CHECK-NEXT: %[[v11:.+]] = tensorrt.broadcast %[[v9]] broadcast_dims<> shape(%[[v10]] : tensor<2xi32>)
//   return %[[v2]], %[[v5]], %[[v8]], %[[v11]]