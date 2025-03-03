// RUN: mlir-tensorrt-opt %s -tensorrt-clustering-pipeline -split-input-file | FileCheck %s

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
