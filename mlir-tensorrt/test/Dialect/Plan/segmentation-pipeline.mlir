// RUN: mlir-tensorrt-opt -split-input-file \
// RUN:  -plan-segmentation-pipeline -cse -verify-diagnostics %s | FileCheck %s

builtin.module attributes {
  plan.cluster_kinds = [
    #plan.tensorrt_cluster<benefit = 1, disallow_shape_tensor_calculations=true>,
    #plan.host_cluster<benefit = 0>
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

builtin.module attributes {
  plan.cluster_kinds = [
    #plan.tensorrt_cluster<benefit = 1, disallow_shape_tensor_calculations=true>,
    #plan.host_cluster<benefit = 0>
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
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<i1>
//       CHECK:     %[[v1:.+]] = tensor.empty() : tensor<i32>
//       CHECK:     %[[v2:.+]]:2 = tensorrt.call @trt_engines::@tensorrt_cluster(%[[arg1]], %[[arg0]] : tensor<i32>, tensor<4xi32>) outs(%[[v0]], %[[v1]] :
//       CHECK:     return %[[v2]]#1, %[[v2]]#0 :
//       CHECK: tensorrt.module @trt_engines
// CHECK-LABEL: func.func @tensorrt_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>, %[[arg1:.+]]: tensor<4xi32>) -> (tensor<i1> {tensorrt.shape_profile = #[[$profile]]}, tensor<i32> {tensorrt.shape_profile = #[[$profile]]}) attributes {cluster.tensorrt}
//       CHECK:       stablehlo.constant
//       CHECK:       stablehlo.compare
//       CHECK:       stablehlo.reduce
//       CHECK:       return

// -----
builtin.module attributes {
  plan.cluster_kinds = [
    #plan.tensorrt_cluster<benefit = 1, disallow_shape_tensor_calculations=true>,
    #plan.host_cluster<benefit = 0>
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
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>, %[[arg1:.+]]: tensor<i32>) -> (tensor<i32> {tensorrt.host_tensor}, tensor<i1> {tensorrt.host_tensor})

//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg1]][] : tensor<i32>
//   CHECK-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<4xi32>
//   CHECK-DAG:     %[[extracted_1:.+]] = tensor.extract %[[arg0]][%[[c1]]] : tensor<4xi32>
//   CHECK-DAG:     %[[extracted_2:.+]] = tensor.extract %[[arg0]][%[[c2]]] : tensor<4xi32>
//   CHECK-DAG:     %[[extracted_3:.+]] = tensor.extract %[[arg0]][%[[c3]]] : tensor<4xi32>
//       CHECK:     %[[v0:.+]]:2 = call @host_cluster(%[[extracted]], %[[extracted_0]], %[[extracted_1]], %[[extracted_2]], %[[extracted_3]]) :
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v0]]#0 : tensor<i1>
//       CHECK:     %[[from_elements_4:.+]] = tensor.from_elements %[[v0]]#1 : tensor<i32>
//       CHECK:     return %[[from_elements_4]], %[[from_elements]] : tensor<i32>, tensor<i1>
// CHECK-LABEL: private @host_cluster
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: i32, %[[arg2:.+]]: i32, %[[arg3:.+]]: i32, %[[arg4:.+]]: i32) -> (i1, i32) attributes {cluster.host}
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.constant dense<0> : tensor<i32>
//   CHECK-DAG:     %[[from_elements:.+]] = tensor.from_elements %[[arg0]] : tensor<i32>
//   CHECK-DAG:     %[[from_elements_0:.+]] = tensor.from_elements %[[arg1]], %[[arg2]], %[[arg3]], %[[arg4]] : tensor<4xi32>
//       CHECK:     %[[v1:.+]] = stablehlo.compare  EQ, %[[v0]]
//       CHECK:     %[[v2:.+]] = stablehlo.reduce(%[[from_elements_0]] init: %[[v0]]) applies stablehlo.add across dimensions = [0] :
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[v1]][]
//       CHECK:     %[[extracted_1:.+]] = tensor.extract %[[v2]][]
//       CHECK:     return %[[extracted]], %[[extracted_1]]

// -----

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

// -----

builtin.module @simple_gather_dynamic {
func.func @simple_gather_dynamic(%arg0: tensor<?x?x256x256xi32>, %arg1: tensor<?xi32>) -> tensor<?x?x256x256xi32> {
  %c1 = stablehlo.constant dense<1> : tensor<1xi32>
  %c256 = stablehlo.constant dense<256> : tensor<1xi32>
  %dim = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<?x?x256x256xi32>) -> tensor<i32>
  %dim.1 = stablehlo.reshape %dim : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %c1, %dim.1, %c256, %c256, dim = 0 :
    (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %shape) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    indices_are_sorted = false, slice_sizes = array<i64: 1>
  } : (tensor<?x?x256x256xi32>, tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x256x256xi32>
  return %0 : tensor<?x?x256x256xi32>
}
}

//       CHECK: #[[$map:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) * 65536)>
// CHECK-LABEL: func.func @simple_gather_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x256x256xi32>, %[[arg1:.+]]: tensor<?xi32>) -> tensor<?x?x256x256xi32>
//   CHECK-DAG:     %[[c256:.+]] = arith.constant 256 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg1]], %[[c0]] : tensor<?xi32>
//   CHECK-DAG:     %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c1]] : tensor<?x?x256x256xi32>
//   CHECK-DAG:     %[[v0:.+]] = arith.index_cast %[[dim_0]] : index to i32
//   CHECK-DAG:     %[[v1:.+]] = arith.index_cast %[[v0]] : i32 to index
//   CHECK-DAG:     %[[v2:.+]] = tensor.empty() : tensor<65536xi32>
//   CHECK-DAG:     %[[v3:.+]] = affine.apply #[[$map]]()[%[[dim]], %[[v1]]]
//   CHECK-DAG:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v2]][0] [%[[v3]]] [1] : tensor<65536xi32> to tensor<?xi32>
//   CHECK-DAG:     %[[from_elements:.+]] = tensor.from_elements %[[dim]], %[[v1]], %[[c256]], %[[c256]] : tensor<4xindex>
//   CHECK-DAG:     %[[reshape:.+]] = tensor.reshape %[[extracted_slice]](%[[from_elements]]) : (tensor<?xi32>, tensor<4xindex>) -> tensor<?x?x256x256xi32>
//   CHECK-DAG:     %[[v4:.+]] = tensorrt.call @trt_engines::@tensorrt_cluster(%[[arg1]], %[[arg0]]
//   CHECK-DAG:     return %[[v4]] : tensor<?x?x256x256xi32>

// CHECK-LABEL: func.func @tensorrt_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xi32>{{.*}}, %[[arg1:.+]]: tensor<?x?x256x256xi32>{{.*}})
//   CHECK-DAG:       %[[c:.+]] = stablehlo.constant dense<1> : tensor<1xi32>
//   CHECK-DAG:       %[[c_0:.+]] = stablehlo.constant dense<256> : tensor<1xi32>
//   CHECK-DAG:       %[[v0:.+]] = stablehlo.get_dimension_size %[[arg1]], dim = 1
//   CHECK-DAG:       %[[v1:.+]] = stablehlo.reshape %[[v0]] : (tensor<i32>) -> tensor<1xi32>
//   CHECK-DAG:       %[[v2:.+]] = stablehlo.concatenate %[[c]], %[[v1]], %[[c_0]], %[[c_0]]
//   CHECK-DAG:       %[[v3:.+]] = "stablehlo.dynamic_gather"(%[[arg1]], %[[arg0]], %[[v2]])
//   CHECK-DAG:       return %[[v3]] : tensor<?x?x256x256xi32>