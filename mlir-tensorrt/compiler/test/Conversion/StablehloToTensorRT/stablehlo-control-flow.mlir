// RUN: mlir-tensorrt-opt -split-input-file %s --convert-stablehlo-to-tensorrt="convert-loops=true trt-major-version=10" | FileCheck %s

func.func @while() -> tensor<i32> {
  %arg0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i32>):
    %1 = "stablehlo.constant"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
    %2 = "stablehlo.compare"(%arg1, %1) {comparison_direction = #stablehlo<comparison_direction LT>}: (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i32>):
    %1 = "stablehlo.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %2 = "stablehlo.add"(%arg1, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%2) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> (tensor<i32>)
  return %0 : tensor<i32>
}

// CHECK-LABEL: @while
//  CHECK-SAME: () -> tensor<i32> {
//       CHECK:   %[[cst_i32:.+]] = tensorrt.constant dense<1> : tensor<i32>
//       CHECK:   %[[cst_i32_0:.+]] = tensorrt.constant dense<3> : tensor<i32>
//       CHECK:   %[[cst_i32_1:.+]] = tensorrt.constant dense<0> : tensor<i32>
//       CHECK:   %[[v0:.+]] = tensorrt.while(%[[cst_i32_1]] : tensor<i32>) -> tensor<i32> {
//       CHECK:   ^bb0(%[[arg0:.+]]: tensor<i32>):
//       CHECK:     %[[v1:.+]] = tensorrt.element_wise <kLESS>(%[[arg0]], %[[cst_i32_0]] : tensor<i32>, tensor<i32>) -> tensor<i1>
//       CHECK:     tensorrt.condition(%[[v1]] : tensor<i1>) %[[arg0]] : tensor<i32>
//       CHECK:   }, {
//       CHECK:   ^bb0(%[[arg0:.+]]: tensor<i32>):
//       CHECK:     %[[v1:.+]] = tensorrt.element_wise <kSUM>(%[[arg0]], %[[cst_i32]] : tensor<i32>, tensor<i32>) -> tensor<i32>
//       CHECK:     tensorrt.yield %[[v1]] : tensor<i32>
//       CHECK:   }
//       CHECK:   return %[[v0]] : tensor<i32>

// -----

func.func @case_to_if_bool_convert(%arg0: tensor<i1>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
  %index = stablehlo.convert %arg0 : (tensor<i1>) -> tensor<i32>
  %0 = "stablehlo.case"(%index) ({
    stablehlo.return %arg1 : tensor<1xf32>
  }, {
    stablehlo.return %arg2 : tensor<1xf32>
  }) : (tensor<i32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK-LABEL: @case_to_if
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i1>, %[[arg1:.+]]: tensor<1xf32>, %[[arg2:.+]]: tensor<1xf32>) -> tensor<1xf32> {
//       CHECK:     %[[v1:.+]] = tensorrt.if(%[[arg0]] : tensor<i1>) -> tensor<1xf32> {
//       CHECK:       tensorrt.yield %[[arg2]] : tensor<1xf32>
//       CHECK:     } else {
//       CHECK:       tensorrt.yield %[[arg1]] : tensor<1xf32>
//       CHECK:     }
//       CHECK:     return %[[v1]] : tensor<1xf32>

// -----

func.func @case_to_if(%arg0: tensor<i32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<1xf32>
  }, {
    stablehlo.return %arg2 : tensor<1xf32>
  }) : (tensor<i32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK-LABEL: @case_to_if
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>, %[[arg1:.+]]: tensor<1xf32>, %[[arg2:.+]]: tensor<1xf32>) -> tensor<1xf32> {
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<0> : tensor<i32>
//       CHECK:     %[[v0:.+]] = tensorrt.element_wise <kEQUAL>(%[[arg0]], %[[cst_i32]] : tensor<i32>, tensor<i32>) -> tensor<i1>
//       CHECK:     %[[v1:.+]] = tensorrt.if(%[[v0]] : tensor<i1>) -> tensor<1xf32> {
//       CHECK:       tensorrt.yield %[[arg1]] : tensor<1xf32>
//       CHECK:     } else {
//       CHECK:       tensorrt.yield %[[arg2]] : tensor<1xf32>
//       CHECK:     }
//       CHECK:     return %[[v1]] : tensor<1xf32>

// -----

func.func @case_one_region(%arg0: tensor<i32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "stablehlo.case"(%arg0) ({
    %1 = stablehlo.add %arg1, %arg2 : tensor<1xf32>
    stablehlo.return %1 : tensor<1xf32>
  }) : (tensor<i32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK-LABEL: @case_one_region
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>, %[[arg1:.+]]: tensor<1xf32>, %[[arg2:.+]]: tensor<1xf32>) -> tensor<1xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.element_wise <kSUM>(%[[arg1]], %[[arg2]] : tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
//       CHECK:     return %[[v0]] : tensor<1xf32>