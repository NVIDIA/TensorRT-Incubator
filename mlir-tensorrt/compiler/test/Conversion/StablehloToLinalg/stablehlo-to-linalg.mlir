// RUN: mlir-tensorrt-opt %s -convert-stablehlo-to-linalg -split-input-file | FileCheck %s

func.func @reverse(%input: tensor<2048xf32>) -> tensor<2048xf32> {
  %result = "stablehlo.reverse"(%input) {
    dimensions = array<i64: 0>
  } : (tensor<2048xf32>) -> tensor<2048xf32>
  func.return %result : tensor<2048xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0) -> (d0)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<()[s0] -> (-s0 + 2047)>
// CHECK-LABEL: func.func @reverse
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2048xf32>) -> tensor<2048xf32>
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<2048xf32>
//       CHECK:     %[[v1:.+]] = linalg.generic
// CHECK-SAME:        indexing_maps = [#[[$map]]],
// CHECK-SAME:        iterator_types = ["parallel"]
// CHECK-SAME:        outs(%[[v0]] : tensor<2048xf32>
//       CHECK:     ^bb0(%[[out:.+]]: f32):
//   CHECK-DAG:       %[[v2:.+]] = linalg.index 0 : index
//   CHECK-DAG:       %[[v3:.+]] = affine.apply #[[$map1]]()[%[[v2]]]
//   CHECK-DAG:       %[[extracted:.+]] = tensor.extract %[[arg0]][%[[v3]]]
//   CHECK-DAG:       linalg.yield %[[extracted]] : f32
//       CHECK:     return %[[v1]] : tensor<2048xf32>
