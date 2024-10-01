// RUN: tensorrt-opt %s -split-input-file -test-tensorrt-shape-inference | FileCheck %s

func.func @test_resize_linear(%arg0: tensor<10x10xf32>) -> (index, index) {
  %result = tensorrt.resize_linear {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>,
    selectorForSinglePixel = #tensorrt.resize_selector<kUPPER>
  } %arg0 : (tensor<10x10xf32>) -> tensor<20x20xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<20x20xf32>
  %d1 = tensor.dim %result, %c1 : tensor<20x20xf32>
  return %d0, %d1 : index, index
}

// CHECK-LABEL: test_resize_linear
// CHECK-NEXT: %[[c20:.+]] = arith.constant 20 : index
// CHECK-NEXT: return %[[c20]], %[[c20]] : index, index

// -----

func.func @test_resize_dynamic_batch(%arg0: tensor<?x1x10x10xf32>) -> (index, index, index, index) {
  %result = tensorrt.resize_linear {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>,
    selectorForSinglePixel = #tensorrt.resize_selector<kUPPER>
  } %arg0 : (tensor<?x1x10x10xf32>) -> tensor<?x1x20x20xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x1x20x20xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x1x20x20xf32>
  %d2 = tensor.dim %result, %c2 : tensor<?x1x20x20xf32>
  %d3 = tensor.dim %result, %c3 : tensor<?x1x20x20xf32>
  return %d0, %d1, %d2, %d3 : index, index, index, index
}

// CHECK-LABEL: func.func @test_resize_dynamic_batch
// CHECK-SAME: (%[[arg0:.+]]: tensor<?x1x10x10xf32>)
//   CHECK-DAG:   %[[c20:.+]] = arith.constant 20 : index
//   CHECK-DAG:   %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[c0:.+]] = arith.constant 0 : index
//       CHECK:   %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x1x10x10xf32>
//       CHECK:   return %[[dim]], %[[c1]], %[[c20]], %[[c20]]

// -----

func.func @test_resize_output_shape(%arg0: tensor<4x4xf32>, %arg1: tensor<2xi32>) -> (index, index) {
  %result = tensorrt.resize_linear {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>,
    selectorForSinglePixel = #tensorrt.resize_selector<kUPPER>
  } %arg0, %arg1 : (tensor<4x4xf32>, tensor<2xi32>) -> tensor<?x?xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %result, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %result, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}

// CHECK-LABEL: func.func @test_resize_output_shape
// CHECK-SAME: (%[[arg0:.+]]: tensor<4x4xf32>, %[[arg1:.+]]: tensor<2xi32>)
//   CHECK-DAG:   %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[extracted:.+]] = tensor.extract %arg1[%c0] : tensor<2xi32>
//   CHECK-DAG:   %[[v0:.+]] = arith.index_cast %extracted : i32 to index
//   CHECK-DAG:   %[[extracted_0:.+]] = tensor.extract %arg1[%c1] : tensor<2xi32>
//   CHECK-DAG:   %[[v1:.+]] = arith.index_cast %extracted_0 : i32 to index
//       CHECK:   return %[[v0]], %[[v1]]
