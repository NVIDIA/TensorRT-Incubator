// RUN: mlir-tensorrt-opt -split-input-file -post-clustering-validation -verify-diagnostics %s

#profile = #tensorrt.shape_profile<min = [4, 4, 3, 3], opt = [4, 4, 3, 3], max = [4, 4, 3, 3]>
module attributes {executor.process_grid_shape = array<i64: 1, 1>} {
  func.func @main() -> tensor<4x4x7x3xf32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<4x4x5x5xf32>
    %1 = tensor.empty() : tensor<4x4x3x3xf32>
    %2 = tensorrt.call @trt_engines::@tensorrt_cluster() outs(%1 : tensor<4x4x3x3xf32>) -> tensor<4x4x3x3xf32>
    // expected-error-re @below {{op: {{.*}} "stablehlo.convolution"{{.*}} from function main is invalid, post clustering.}}
    %3 = stablehlo.convolution(%0, %2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [2, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x4x5x5xf32>, tensor<4x4x3x3xf32>) -> tensor<4x4x7x3xf32>
    return %3 : tensor<4x4x7x3xf32>
  }
  tensorrt.module @trt_engines {
    func.func @tensorrt_cluster() -> (tensor<4x4x3x3xf32> {tensorrt.shape_profile = #profile}) attributes {cluster.tensorrt} {
      %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<144xf32>
      %1 = stablehlo.reshape %0 : (tensor<144xf32>) -> tensor<4x4x3x3xf32>
      return %1 : tensor<4x4x3x3xf32>
    }
  }
}
