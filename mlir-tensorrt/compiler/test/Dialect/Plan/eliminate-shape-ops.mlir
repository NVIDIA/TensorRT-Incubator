// RUN: mlir-tensorrt-opt -split-input-file -plan-eliminate-shape-ops %s | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 10)>
#profile = #tensorrt.shape_profile<min = [10, 10], opt = [105, 10], max = [200, 10]>
func.func @test_simple(%arg0: tensor<?x10xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [10, 10], opt = [100, 10], max = [200, 10]>}) -> tensor<?x10xf32> {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x10xf32>
  %0 = tensor.empty() : tensor<2000xf32>
  %1 = affine.apply #map()[%dim]
  %extracted_slice = tensor.extract_slice %0[0] [%1] [1] : tensor<2000xf32> to tensor<?xf32>
  %from_elements = tensor.from_elements %dim, %c10 : tensor<2xindex>
  %reshape = tensor.reshape %extracted_slice(%from_elements) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x10xf32>
  %2 = tensorrt.call @trt_engines::@tensorrt_cluster(%arg0, %dim, %c10 : tensor<?x10xf32>, index, index) outs(%reshape : tensor<?x10xf32>) -> tensor<?x10xf32>
  return %2 : tensor<?x10xf32>
}

tensorrt.module @trt_engines {
  func.func @tensorrt_cluster(%arg0: tensor<?x10xf32> {tensorrt.shape_profile = #profile},
        %arg1: index, %arg2: index)
      -> (tensor<?x10xf32> {tensorrt.shape_profile = #profile}) {
    %0 = stablehlo.exponential %arg0 : tensor<?x10xf32>
    %1 = plan.with_shape %0(%arg1, %arg2) : (tensor<?x10xf32>, index, index) -> tensor<?x10xf32>
    return %1 : tensor<?x10xf32>
  }
}

//       CHECK: #[[$profile:.+]] = #tensorrt.shape_profile
// CHECK-LABEL: @test_simple
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x10xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<{{.+}}>}) -> tensor<?x10xf32> {
//       CHECK:     tensorrt.call @trt_engines::@tensorrt_cluster(%{{.+}} : tensor<?x10xf32>) outs(%{{.+}} : tensor<?x10xf32>)
//  CHECK-NEXT:     return

//       CHECK: tensorrt.module @trt_engines
//       CHECK: @tensorrt_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x10xf32> {tensorrt.shape_profile = #[[$profile]]}) -> (tensor<?x10xf32> {tensorrt.shape_profile = #[[$profile]]})
//  CHECK-NEXT:       %[[v0:.+]] = stablehlo.exponential %[[arg0]] : tensor<?x10xf32>
//  CHECK-NEXT:       return %[[v0]] : tensor<?x10xf32>

