// RUN: mlir-tensorrt-opt -split-input-file \
// RUN:  -pass-pipeline="builtin.module(plan-segmentation-pipeline,cse)" %s | FileCheck %s

module attributes {
  plan.backends = [
    #plan.tensorrt_backend<benefit = 2, disallow_shape_tensor_calculations = true, tensorrt_major_version = 10>,
    #plan.kernel_backend<benefit = 1>,
    #plan.host_backend<benefit = 0>
  ]
} {

func.func @chlo_erf_to_trt_cluster(%arg0: tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32> {
  %0 = chlo.erf %arg0 : tensor<1x197x3072xf32> -> tensor<1x197x3072xf32>
  return %0 : tensor<1x197x3072xf32>
}

}

//       CHECK: #[[$profile:.+]] = #tensorrt.shape_profile<min = [1, 197, 3072], opt = [1, 197, 3072], max = [1, 197, 3072]>
// CHECK-LABEL: @chlo_erf_to_trt_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x197x3072xf32>)
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<1x197x3072xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.call @trt_engines::@tensorrt_cluster(%[[arg0]] : tensor<1x197x3072xf32>) outs(%[[v0]] : tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
//       CHECK:     return %[[v1]] : tensor<1x197x3072xf32>

//       CHECK: tensorrt.module @trt_engines
// CHECK-LABEL: func.func @tensorrt_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x197x3072xf32>) -> (tensor<1x197x3072xf32> {tensorrt.shape_profile = #[[$profile]]}) attributes {cluster.tensorrt}
//       CHECK:       %[[v0:.+]] = chlo.erf %[[arg0]]
//       CHECK:       return %[[v0]]

// -----

module attributes {
  plan.backends = [
    #plan.tensorrt_backend<benefit = 2, disallow_shape_tensor_calculations = true, tensorrt_major_version = 10>,
    #plan.kernel_backend<benefit = 1>,
    #plan.host_backend<benefit = 0>
  ]
} {

func.func @reduce(%arg0: tensor<4xi32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i1>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %2 = stablehlo.constant dense<0> : tensor<i32>
  %3 = stablehlo.compare EQ, %2, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %4 = stablehlo.reduce(%arg0 init: %2) across dimensions = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
    reducer(%arg6: tensor<i32>, %arg7: tensor<i32>)  {
    %27 = stablehlo.add %arg6, %arg7 : tensor<i32>
    stablehlo.return %27 : tensor<i32>
  }

  return %4, %3 : tensor<i32>, tensor<i1>
}

}

//       CHECK: #[[$profile:.+]] = #tensorrt.shape_profile<min = [], opt = [], max = []>
// CHECK-LABEL: func.func @reduce
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>, %[[arg1:.+]]: tensor<i32>) -> (tensor<i32>, tensor<i1>) {
//   CHECK-DAG:     %[[v1:.+]] = tensor.empty() : tensor<i1>
//   CHECK-DAG:     %[[v0:.+]] = tensor.empty() : tensor<i32>
//       CHECK:     %[[v2:.+]]:2 = tensorrt.call @trt_engines::@tensorrt_cluster(%[[arg0]], %[[arg1]] : {{.*}}) outs(%[[v0]], %[[v1]] : {{.*}})
//       CHECK:     return %[[v2]]#0, %[[v2]]#1 :
//       CHECK: tensorrt.module @trt_engines
// CHECK-LABEL: func.func @tensorrt_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>, %[[arg1:.+]]: tensor<i32>) -> (tensor<i32> {{.*}}, tensor<i1> {{.*}})
//       CHECK:       stablehlo.constant
//       CHECK:       stablehlo.compare
//       CHECK:       stablehlo.reduce
//       CHECK:       return

// -----

module attributes {
  plan.backends = [
    #plan.tensorrt_backend<benefit = 2, disallow_shape_tensor_calculations = true, tensorrt_major_version = 10>,
    #plan.kernel_backend<benefit = 1>,
    #plan.host_backend<benefit = 0>
  ]
} {

func.func @small_reduce_host(%arg0: tensor<4xi32>, %arg1: tensor<i32>)
    -> (tensor<i32> {tensorrt.host_tensor}, tensor<i1> {tensorrt.host_tensor}) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %2 = stablehlo.constant dense<0> : tensor<i32>
  %3 = stablehlo.compare EQ, %2, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %4 = stablehlo.reduce(%arg0 init: %2) across dimensions = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
    reducer(%arg6: tensor<i32>, %arg7: tensor<i32>)  {
    %27 = stablehlo.add %arg6, %arg7 : tensor<i32>
    stablehlo.return %27 : tensor<i32>
  }
  return %4, %3 : tensor<i32>, tensor<i1>
}

}

// CHECK-LABEL: func.func @small_reduce_host
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>, %[[arg1:.+]]: tensor<i32>)
//  CHECK-NEXT:     %[[v0]]:2 = call @host_backend(%[[arg0]], %[[arg1]])
//  CHECK-NEXT:     return %[[v0]]#0, %[[v0]]#1 : tensor<i32>, tensor<i1>

// CHECK-LABEL: func.func private @host_backend
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>, %[[arg1:.+]]: tensor<i32>) -> (tensor<i32>, tensor<i1>)
//  CHECK-NEXT:     %[[c:.+]] = stablehlo.constant dense<0> : tensor<i32>
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.compare  EQ, %[[c]], %[[arg1]]
//  CHECK-NEXT:     %[[v1:.+]] = stablehlo.reduce(%[[arg0]] init: %[[c]])
//  CHECK-NEXT:     return %[[v1]], %[[v0]] : tensor<i32>, tensor<i1>

// -----

module attributes {
  plan.backends = [
    #plan.tensorrt_backend<benefit = 2, disallow_shape_tensor_calculations = true, tensorrt_major_version = 10>,
    #plan.kernel_backend<benefit = 1>,
    #plan.host_backend<benefit = 0>
  ]
} {

func.func @interior_padding_test(%arg0: tensor<1x2xi32>) -> tensor<1x2xi32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = "stablehlo.slice"(%arg0) {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 2>} : (tensor<1x2xi32>) -> tensor<1x1xi32>
  %2 = "stablehlo.slice"(%arg0) {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 2>} : (tensor<1x2xi32>) -> tensor<1x1xi32>
  %3 = stablehlo.add %1, %2 : tensor<1x1xi32>
  %4 = "stablehlo.slice"(%arg0) {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x2xi32>) -> tensor<1x1xi32>
  %5 = "stablehlo.pad"(%4, %0) {edge_padding_high = array<i64: 0, 1>, edge_padding_low = array<i64:0, 0>, interior_padding = array<i64: 0, 1>} : (tensor<1x1xi32>, tensor<i32>) -> tensor<1x2xi32>
  %6 = "stablehlo.pad"(%3, %0) {edge_padding_high = array<i64: 0, 0>, edge_padding_low = array<i64: 0, 1>, interior_padding = array<i64: 0, 1>} : (tensor<1x1xi32>, tensor<i32>) -> tensor<1x2xi32>
  %7 = stablehlo.add %5, %6 : tensor<1x2xi32>
  return %7 : tensor<1x2xi32>
}

}

//   CHECK-DAG: #[[$profile:.+]] = #tensorrt.shape_profile<min = [1, 1], opt = [1, 1], max = [1, 1]>
//   CHECK-DAG: #[[$profile1:.+]] = #tensorrt.shape_profile<min = [1, 2], opt = [1, 2], max = [1, 2]>
// CHECK-LABEL: func.func @interior_padding_test
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x2xi32>) -> tensor<1x2xi32>
//   CHECK-DAG:     %[[v0:.+]] = call @codegen_cluster(%[[arg0]]) :
//   CHECK-DAG:     %[[v1:.+]] = call @codegen_cluster_0(%[[arg0]])
//   CHECK-DAG:     %[[v2:.+]] = tensor.empty() : tensor<1x1xi32>
//   CHECK-DAG:     %[[v3:.+]] = tensorrt.call @trt_engines::@tensorrt_cluster(%[[v0]], %[[v1]] :
//   CHECK-DAG:     %[[v4:.+]] = call @codegen_cluster_1(%[[arg0]])
//   CHECK-DAG:     %[[v5:.+]] = call @codegen_cluster_2(%[[v3]])
//   CHECK-DAG:     %[[v6:.+]] = tensor.empty() : tensor<1x2xi32>
//   CHECK-DAG:     %[[v7:.+]] = tensorrt.call @trt_engines::@tensorrt_cluster_0(%[[v4]], %[[v5]] :
//   CHECK-DAG:     return %[[v7]] : tensor<1x2xi32>

// CHECK-LABEL: func.func private @codegen_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x2xi32>)
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.slice %[[arg0]] [0:1, 0:1:2]
//   CHECK-DAG:     return %[[v0]]

// CHECK-LABEL: func.func private @codegen_cluster_0
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x2xi32>)
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.slice %[[arg0]] [0:1, 1:2:2]
//   CHECK-DAG:     return %[[v0]]

//       CHECK:   tensorrt.module @trt_engines
// CHECK-LABEL: func.func @tensorrt_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x1xi32>, %[[arg1:.+]]: tensor<1x1xi32>)
//   CHECK-DAG:       %[[v0:.+]] = stablehlo.add %[[arg0]], %[[arg1]]
//   CHECK-DAG:       return %[[v0]]

// CHECK-LABEL: func.func @tensorrt_cluster_0
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x2xi32>, %[[arg1:.+]]: tensor<1x2xi32>)
//   CHECK-DAG:       %[[v0:.+]] = stablehlo.add %[[arg0]], %[[arg1]]
//   CHECK-DAG:       return %[[v0]]


// CHECK-LABEL: func.func private @codegen_cluster_1
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x2xi32>) -> tensor<1x2xi32> attributes {plan.cluster_kind = #plan.kernel_backend<benefit = 1>} {
//   CHECK-DAG:     %[[c:.+]] = stablehlo.constant dense<0>
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.slice %[[arg0]] [0:1, 0:1]
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.pad %[[v0]], %[[c]], low = [0, 0], high = [0, 1], interior = [0, 1]
//   CHECK-DAG:     return %[[v1]]

// CHECK-LABEL: func.func private @codegen_cluster_2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x1xi32>)
//   CHECK-DAG:     %[[c:.+]] = stablehlo.constant dense<0>
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.pad %[[arg0]], %[[c]], low = [0, 1], high = [0, 0], interior = [0, 1]
//   CHECK-DAG:     return %[[v0]]

// -----

module attributes {
  plan.backends = [
    #plan.tensorrt_backend<benefit = 2, disallow_shape_tensor_calculations = true, tensorrt_major_version = 10>,
    #plan.kernel_backend<benefit = 1>,
    #plan.host_backend<benefit = 0>
  ]
} {

func.func @concat_i64(%arg0: tensor<128xi64>, %arg1: tensor<128xi64>) -> tensor<256xi64> {
  %2  = stablehlo.concatenate %arg0, %arg1, dim=0 : (tensor<128xi64>, tensor<128xi64>) -> tensor<256xi64>
  return %2 : tensor<256xi64>
}

}

// CHECK-LABEL: func.func @concat_i64
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xi64>, %[[arg1:.+]]: tensor<128xi64>) -> tensor<256xi64>
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<256xi64>
//       CHECK:     %[[v1:.+]] = tensorrt.call @trt_engines::@tensorrt_cluster(%[[arg0]], %[[arg1]] : tensor<128xi64>, tensor<128xi64>) outs(%[[v0]] : tensor<256xi64>) -> tensor<256xi64>
//       CHECK:    return %[[v1]] : tensor<256xi64>
// CHECK-LABEL: tensorrt.module @trt_engines
// CHECK-LABEL: func.func @tensorrt_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xi64>, %[[arg1:.+]]: tensor<128xi64>) -> (tensor<256xi64> {tensorrt.shape_profile = #profile}) attributes {cluster.tensorrt}
//       CHECK:     %[[v0:.+]] = stablehlo.concatenate %[[arg0]], %[[arg1]], dim = 0 :
//       CHECK:     return %[[v0]] : tensor<256xi64>

// -----

module attributes {
  plan.backends = [
    #plan.tensorrt_backend<benefit = 2, disallow_shape_tensor_calculations = true, tensorrt_major_version = 10>,
    #plan.kernel_backend<benefit = 1>,
    #plan.host_backend<benefit = 0>
  ]
} {

func.func @maximum_i64(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
  %2  = stablehlo.maximum %arg0, %arg1 : tensor<i64>
  return %2 : tensor<i64>
}

}

// CHECK-LABEL: func.func @maximum_i64
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i64>, %[[arg1:.+]]: tensor<i64>) -> tensor<i64>
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<i64>
//       CHECK:     %[[v1:.+]] = tensorrt.call @trt_engines::@tensorrt_cluster(%[[arg0]], %[[arg1]] : tensor<i64>, tensor<i64>) outs(%[[v0]] : tensor<i64>) -> tensor<i64>
//       CHECK:    return %[[v1]] : tensor<i64>
// CHECK-LABEL: tensorrt.module @trt_engines
// CHECK-LABEL: func.func @tensorrt_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i64>, %[[arg1:.+]]: tensor<i64>) -> (tensor<i64> {tensorrt.shape_profile = #profile}) attributes {cluster.tensorrt}
//       CHECK:     %[[v0:.+]] = stablehlo.maximum %[[arg0]], %[[arg1]] :
//       CHECK:     return %[[v0]] : tensor<i64>

// -----

module attributes {
  plan.backends = [
    #plan.tensorrt_backend<benefit = 2, disallow_shape_tensor_calculations = true, tensorrt_major_version = 10>,
    #plan.kernel_backend<benefit = 1>,
    #plan.host_backend<benefit = 0>
  ]
} {

func.func @reduce_maximum_i64(%arg0: tensor<128xi64>) -> tensor<i64> {
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  %0 = stablehlo.reduce(%arg0 init: %c_0) applies stablehlo.maximum across dimensions = [0] : (tensor<128xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}

}

// CHECK-LABEL: func.func @reduce_maximum_i64
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xi64>) -> tensor<i64>
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<i64>
//       CHECK:     %[[v1:.+]] = tensorrt.call @trt_engines::@tensorrt_cluster(%[[arg0]] : tensor<128xi64>) outs(%[[v0]] : tensor<i64>) -> tensor<i64>
//       CHECK:    return %[[v1]] : tensor<i64>
// CHECK-LABEL: tensorrt.module @trt_engines
// CHECK-LABEL: func.func @tensorrt_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xi64>) -> (tensor<i64> {tensorrt.shape_profile = #profile}) attributes {cluster.tensorrt}
//       CHECK:     %[[c:.+]] = stablehlo.constant dense<0> : tensor<i64>
//       CHECK:     %[[v0:.+]] = stablehlo.reduce(%[[arg0]] init: %[[c]]) applies stablehlo.maximum across dimensions = [0] : (tensor<128xi64>, tensor<i64>) -> tensor<i64>
//       CHECK:     return %[[v0]] : tensor<i64>

// -----

module attributes {
  plan.backends = [
    #plan.tensorrt_backend<benefit = 2, disallow_shape_tensor_calculations = true, tensorrt_major_version = 10>,
    #plan.kernel_backend<benefit = 1>,
    #plan.host_backend<benefit = 0>
  ]
} {

// Quantize f32 -> int8
func.func @main(%arg0: tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8> {
  %0 = stablehlo.composite "tensorrt.pt_q" %arg0 {composite_attributes = {axis = -1 : i32, scale = dense<8.000000e-01> : tensor<f32>}, decomposition = @pt_q} : (tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
  return %0 : tensor<2x3x300x300xi8>
}
func.func private @pt_q(%arg0: tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8> attributes {plan.decomposition} {
  %cst = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  %cst_0 = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  %cst_1 = stablehlo.constant dense<8.000000e-01> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x3x300x300xf32>
  %1 = stablehlo.divide %arg0, %0 : tensor<2x3x300x300xf32>
  %2 = stablehlo.round_nearest_even %1 : tensor<2x3x300x300xf32>
  %3 = stablehlo.clamp %cst, %2, %cst_0 : (tensor<f32>, tensor<2x3x300x300xf32>, tensor<f32>) -> tensor<2x3x300x300xf32>
  %4 = stablehlo.convert %3 : (tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
  return %4 : tensor<2x3x300x300xi8>
}

}

// CHECK-LABEL: func.func @main
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
//  CHECK-NEXT: %[[v0:.+]] = tensor.empty() : tensor<2x3x300x300xi8>
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.call @trt_engines::@tensorrt_cluster(%[[arg0]] : tensor<2x3x300x300xf32>) outs(%[[v0]] : tensor<2x3x300x300xi8>) -> tensor<2x3x300x300xi8>
//  CHECK-NEXT: return %[[v1]] : tensor<2x3x300x300xi8>

// CHECK-LABEL: tensorrt.module @trt_engines
// CHECK-LABEL: func.func @tensorrt_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x300x300xf32>
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.composite "tensorrt.pt_q" %[[arg0]] {composite_attributes = {axis = -1 : i32, scale = dense<8.000000e-01> : tensor<f32>}, decomposition = @pt_q} : (tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
//  CHECK-NEXT: return %[[v0]] : tensor<2x3x300x300xi8>

// CHECK-LABEL: func.func private @pt_q
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x300x300xf32>)
//  CHECK-SAME: attributes {plan.decomposition}
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.constant dense<1.270000e+02> : tensor<f32>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.constant dense<8.000000e-01> : tensor<f32>
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.broadcast_in_dim %[[v2]], dims = [] : (tensor<f32>) -> tensor<2x3x300x300xf32>
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.divide %[[arg0]], %[[v3]] : tensor<2x3x300x300xf32>
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.round_nearest_even %[[v4]] : tensor<2x3x300x300xf32>
//  CHECK-NEXT: %[[v6:.+]] = stablehlo.clamp %[[v0]], %[[v5]], %[[v1]]
//  CHECK-NEXT: %[[v7:.+]] = stablehlo.convert %[[v6]]
//  CHECK-NEXT: return %[[v7]] : tensor<2x3x300x300xi8>
