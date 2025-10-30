// RUN: mlir-tensorrt-opt %s -split-input-file -plan-legalize-io-bounds-attributes -mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: func.func @test_legalize_io_bounds_attributes
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32> {plan.shape_bounds = #plan.bounds<shape, [10], [30]>},
//  CHECK-SAME: %[[arg1:.+]]: tensor<1xi32> {plan.value_bounds = #plan.bounds<value, dense<1> : tensor<1xi32>, dense<3> : tensor<1xi32>>})
//  CHECK-SAME: -> (tensor<?xf32> {plan.shape_bounds = #plan.bounds<shape, [1], [6]>}) {

func.func @test_legalize_io_bounds_attributes(
      %arg0: tensor<?xf32> {
        tensorrt.shape_profile = #tensorrt.shape_profile<min=[10], opt=[20], max=[30]>
      },
      %arg1: tensor<1xi32> {
        tensorrt.value_bounds = #tensorrt.shape_profile<min=[1], opt=[2], max=[3]>
      }) -> (tensor<?xf32> {
        tensorrt.shape_profile = #tensorrt.shape_profile<min=[1], opt=[2], max=[6]>
      }) {
  %c0 = arith.constant 0 : index
  %0 = tensor.extract %arg1[%c0] : tensor<1xi32>
  %idx = arith.index_cast %0 : i32 to index
  %1 = tensor.empty(%idx) : tensor<?xf32>
  return %1 : tensor<?xf32>
}
