// RUN: mlir-tensorrt-opt %s -tensorrt-clustering-pipeline -split-input-file | FileCheck %s

func.func @trt_relu(%arg0: tensor<2x10xf16>) -> (tensor<2x10xf16>) {
  %0 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %arg0 : tensor<2x10xf16>
  return %0: tensor<2x10xf16>
}

// CHECK-LABEL: @trt_relu
// CHECK-DAG:   %[[v0:.+]] = tensorrt.call_alloc @trt_engines::@tensorrt_cluster
// CHECK-DAG: return %[[v0]]
// CHECK-DAG: @tensorrt_cluster
// CHECK-DAG: %[[v1:.+]] = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>}
// CHECK-DAG: return %[[v1]]
