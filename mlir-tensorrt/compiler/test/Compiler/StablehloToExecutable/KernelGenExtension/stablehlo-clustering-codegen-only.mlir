// RUN: mlir-tensorrt-opt %s \
// RUN: -pass-pipeline="builtin.module(plan-clustering)" -split-input-file | FileCheck %s
// RUN: mlir-tensorrt-opt %s \
// RUN: -pass-pipeline="builtin.module(plan-segmentation-pipeline)" -split-input-file | FileCheck %s --check-prefix=SEGMENT


// TODO: add codegen, host constraints
builtin.module attributes {
  plan.backends = [
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
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>, %[[arg1:.+]]: tensor<i32>) -> (tensor<i32> {tensorrt.host_tensor}, tensor<i1> {tensorrt.host_tensor}) {
//   CHECK-DAG:     %[[cst:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
//   CHECK-DAG:     %[[cst_0:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
//   CHECK-DAG:     %[[c:.+]] = stablehlo.constant dense<0> : tensor<i32>
//   CHECK-DAG:     %[[v0:.+]]:2 = plan.inline_group target(#plan.host_backend<{{.*}}>) attributes {__cluster_target__ = #plan.host_backend<{{.*}}>}
//   CHECK-DAG:       %[[v1:.+]] = stablehlo.compare  EQ, %[[c]], %[[arg1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
//   CHECK-DAG:       %[[v2:.+]] = stablehlo.reduce(%[[arg0]] init: %[[c]]) applies stablehlo.add across dimensions = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
//   CHECK-DAG:       yield %[[v2]], %[[v1]] : tensor<i32>, tensor<i1>
//   CHECK-DAG:     return %[[v0]]#0, %[[v0]]#1 : tensor<i32>, tensor<i1>


// -----

builtin.module attributes {
  plan.backends = [
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

// CHECK-LABEL: func.func @interior_padding_test
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x2xi32>) -> tensor<1x2xi32> {
//       CHECK:     %[[c:.+]] = stablehlo.constant dense<0> : tensor<i32>
//       CHECK:     %[[v0:.+]] = plan.inline_group target(#plan.kernel_backend<benefit = 1>) attributes {__cluster_target__ = #plan.kernel_backend<benefit = 1>} -> tensor<1x2xi32> {
//       CHECK:       %[[v1:.+]] = stablehlo.slice %[[arg0]] [0:1, 0:1:2] : (tensor<1x2xi32>) -> tensor<1x1xi32>
//       CHECK:       %[[v2:.+]] = stablehlo.slice %[[arg0]] [0:1, 1:2:2] : (tensor<1x2xi32>) -> tensor<1x1xi32>
//       CHECK:       %[[v3:.+]] = stablehlo.add %[[v1]], %[[v2]] : tensor<1x1xi32>
//       CHECK:       %[[v4:.+]] = stablehlo.slice %[[arg0]] [0:1, 0:1] : (tensor<1x2xi32>) -> tensor<1x1xi32>
//       CHECK:       %[[v5:.+]] = stablehlo.pad %[[v4]], %[[c]], low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<1x1xi32>, tensor<i32>) -> tensor<1x2xi32>
//       CHECK:       %[[v6:.+]] = stablehlo.pad %[[v3]], %[[c]], low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<1x1xi32>, tensor<i32>) -> tensor<1x2xi32>
//       CHECK:       %[[v7:.+]] = stablehlo.add %[[v5]], %[[v6]] : tensor<1x2xi32>
//       CHECK:       yield %[[v7]] : tensor<1x2xi32>
//       CHECK:     return %[[v0]] : tensor<1x2xi32>

// -----

builtin.module attributes {
  plan.backends = [
    #plan.kernel_backend<benefit = 1>,
    #plan.host_backend<benefit = 0>
  ]
} {

func.func @concat_must_be_copy(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128x4xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<128xf32>) -> tensor<128x2xf32>
  %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<128xf32>) -> tensor<128x2xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<128x2xf32>, tensor<128x2xf32>) -> tensor<128x4xf32>
  return %2 : tensor<128x4xf32>
}

}

// CHECK-LABEL: func.func @concat_must_be_copy
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xf32>, %[[arg1:.+]]: tensor<128xf32>) -> tensor<128x4xf32> {
//       CHECK:     %[[v0:.+]] = plan.inline_group target(#plan.kernel_backend<benefit = 1>) attributes {__cluster_target__ = #plan.kernel_backend<benefit = 1>} -> tensor<128x4xf32> {
//       CHECK:       %[[v1:.+]] = stablehlo.broadcast_in_dim %[[arg0]], dims = [0] : (tensor<128xf32>) -> tensor<128x2xf32>
//       CHECK:       %[[v2:.+]] = stablehlo.broadcast_in_dim %[[arg1]], dims = [0] : (tensor<128xf32>) -> tensor<128x2xf32>
//       CHECK:       %[[v3:.+]] = stablehlo.concatenate %[[v1]], %[[v2]], dim = 1 : (tensor<128x2xf32>, tensor<128x2xf32>) -> tensor<128x4xf32>
//       CHECK:       yield %[[v3]] : tensor<128x4xf32>
//       CHECK:     }
//       CHECK:     return %[[v0]] : tensor<128x4xf32>

// -----

builtin.module attributes {
  plan.backends = [
    #plan.kernel_backend<benefit = 1>,
    #plan.host_backend<benefit = 0>
  ]
} {

// This test checks that the 'KernelBackend' clustering configuration correctly handles
// non-int/float scalar types correctly.

func.func @complex_add(%arg0: tensor<128x2xcomplex<f32>>) -> (tensor<128x2xcomplex<f32>>, tensor<128x2xcomplex<f32>>) {
  %cst = stablehlo.constant dense<(0.0, 0.0)> : tensor<128x2xcomplex<f32>>
  %0 = stablehlo.add %arg0, %cst : tensor<128x2xcomplex<f32>>
  return %0, %cst : tensor<128x2xcomplex<f32>>, tensor<128x2xcomplex<f32>>
}

}

// CHECK-LABEL: func.func @complex_add

// SEGMENT-LABEL: func.func @complex_add
//  SEGMENT-SAME: (%[[arg0:.+]]:
//   SEGMENT-DAG:     %[[cst:.+]] = stablehlo.constant
//   SEGMENT-DAG:     %[[v0:.+]] = call @codegen_cluster(%[[arg0]])
//   SEGMENT-DAG:     return %[[v0]], %[[cst]]
// SEGMENT-LABEL: func.func private @codegen_cluster
//  SEGMENT-SAME: (%[[arg0:.+]]: tensor<128x2xcomplex<f32>>)
//   SEGMENT-DAG:     %[[cst:.+]] = stablehlo.constant
//   SEGMENT-DAG:     %[[v0:.+]] = stablehlo.add %[[arg0]], %[[cst]] :
//   SEGMENT-DAG:     return %[[v0]] : tensor<128x2xcomplex<f32>>
