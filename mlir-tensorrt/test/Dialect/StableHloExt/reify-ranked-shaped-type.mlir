// RUN: mlir-tensorrt-opt %s -split-input-file -test-tensorrt-shape-inference | FileCheck %s

func.func @refine_convolution(%arg0 : tensor<100x26x26x32xf32>, %arg1 : tensor<3x3x1x32xf32>)
        -> (index, index, index, index) {
  %result = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f],
    window = {
      stride = [1, 1],
      pad = [[2, 2], [2, 2]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    } {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) -> tensor<?x?x?x?xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?x?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?x?x?xf32>
  %d2 = tensor.dim %result, %c2 : tensor<?x?x?x?xf32>
  %d3 = tensor.dim %result, %c3 : tensor<?x?x?x?xf32>
  return %d0, %d1, %d2, %d3 : index, index, index, index
}

// CHECK-LABEL: @refine_convolution
// CHECK-DAG:     %[[c100:.+]] = arith.constant 100 : index
// CHECK-DAG:     %[[c28:.+]] = arith.constant 28 : index
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:     return %[[c100]], %[[c28]], %[[c28]], %[[c1]] :

// -----

func.func @convolution_empty_spatial_dimensions(%arg0: tensor<3x2xf16>,
    %arg1: tensor<2x2xf16>) -> (index, index) {
  %0 = stablehlo.convolution(%arg0, %arg1)
         dim_numbers = [b, f]x[i, o]->[b, f],
         window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [],
           reverse = []}
         {
           batch_group_count = 1 : i64,
           feature_group_count = 1 : i64,
           precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
         }
       : (tensor<3x2xf16>, tensor<2x2xf16>) -> tensor<?x?xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %0, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %0, %c1 : tensor<?x?xf16>
  func.return %d0, %d1 : index, index
}

// CHECK-LABEL: @convolution_empty_spatial_dimensions
// CHECK-DAG:  %[[c3:.+]] = arith.constant 3 : index
// CHECK-DAG:  %[[c2:.+]] = arith.constant 2 : index
// CHECK-DAG:  return %[[c3]], %[[c2]] : index, index

// -----

func.func @dynamic_input(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<1x1x1024x1024xf32>) -> (index, index, index, index) {
  %cst = stablehlo.constant dense_resource<__elided__> : tensor<256x256x1x1xf32>
  %0 = stablehlo.convolution(%arg0, %cst) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
        window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]}
        {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
        : (tensor<?x?x?x?xf32>, tensor<256x256x1x1xf32>) -> tensor<?x?x?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %0, %c0 : tensor<?x?x?x?xf32>
  %d1 = tensor.dim %0, %c1 : tensor<?x?x?x?xf32>
  %d2 = tensor.dim %0, %c2 : tensor<?x?x?x?xf32>
  %d3 = tensor.dim %0, %c3 : tensor<?x?x?x?xf32>
  return %d0, %d1, %d2, %d3 : index, index, index, index
}

// CHECK-LABEL: func.func @dynamic_input
// CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x?x?xf32>, %[[arg1:.+]]: tensor<1x1x1024x1024xf32>)
//   CHECK-DAG: %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[cn1:.+]] = arith.constant -1 : index
//   CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG: %[[c256:.+]] = arith.constant 256 : index
//   CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x?x?x?xf32>
//   CHECK-DAG: %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c2]] : tensor<?x?x?x?xf32>
//   CHECK-DAG: %[[v0:.+]] = arith.maxsi %[[dim_0]], %[[c0]]
//   CHECK-DAG: %[[v1:.+]] = arith.addi %[[v0]], %[[cn1]] : index
//   CHECK-DAG: %[[v2:.+]] = arith.addi %[[v1]], %[[c1]] : index
//   CHECK-DAG: %[[v3:.+]] = arith.maxsi %[[v2]], %[[c0]] : index
//   CHECK-DAG: %[[dim_1:.+]] = tensor.dim %[[arg0]], %[[c3]] : tensor<?x?x?x?xf32>
//   CHECK-DAG: %[[v4:.+]] = arith.maxsi %[[dim_1]], %[[c0]] : index
//   CHECK-DAG: %[[v5:.+]] = arith.addi %[[v4]], %[[cn1]] : index
//   CHECK-DAG: %[[v6:.+]] = arith.addi %[[v5]], %[[c1]] : index
//   CHECK-DAG: %[[v7:.+]] = arith.maxsi %[[v6]], %[[c0]] : index
//   CHECK-DAG: return %[[dim]], %[[c256]], %[[v3]], %[[v7]] :
