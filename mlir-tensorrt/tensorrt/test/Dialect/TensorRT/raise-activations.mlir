// RUN: tensorrt-opt %s -split-input-file -tensorrt-raise-activations=target-tensorrt-version=10.0 | FileCheck %s
// RUN: tensorrt-opt %s -split-input-file -tensorrt-raise-activations=target-tensorrt-version=8.6 | FileCheck %s --check-prefix=TRT8

func.func @raise_gelu(%arg0: tensor<12x128x4x12x1xf32>) -> (tensor<12x128x4x12x1xf32>)  {
  %cst_f32 = tensorrt.constant dense<5.000000e-01> : tensor<1x1x1x1x1xf32>
  %cst_f32_0 = tensorrt.constant dense<1.000000e+00> : tensor<1x1x1x1x1xf32>
  %cst_f32_1 = tensorrt.constant dense<0.797884583> : tensor<1x1x1x1x1xf32>
  %cst_f32_2 = tensorrt.constant dense<4.471500e-02> : tensor<1x1x1x1x1xf32>
  %0 = tensorrt.element_wise <kPROD>(%arg0, %arg0 : tensor<12x128x4x12x1xf32>, tensor<12x128x4x12x1xf32>) -> tensor<12x128x4x12x1xf32>
  %1 = tensorrt.element_wise <kPROD>(%0, %arg0 : tensor<12x128x4x12x1xf32>, tensor<12x128x4x12x1xf32>) -> tensor<12x128x4x12x1xf32>
  %2 = tensorrt.element_wise <kPROD>(%1, %cst_f32_2 : tensor<12x128x4x12x1xf32>, tensor<1x1x1x1x1xf32>) -> tensor<12x128x4x12x1xf32>
  %3 = tensorrt.element_wise <kSUM>(%arg0, %2 : tensor<12x128x4x12x1xf32>, tensor<12x128x4x12x1xf32>) -> tensor<12x128x4x12x1xf32>
  %4 = tensorrt.element_wise <kPROD>(%3, %cst_f32_1 : tensor<12x128x4x12x1xf32>, tensor<1x1x1x1x1xf32>) -> tensor<12x128x4x12x1xf32>
  %5 = tensorrt.activation {activationType = #tensorrt.activation_type<kTANH>} %4 : tensor<12x128x4x12x1xf32>
  %6 = tensorrt.element_wise <kSUM>(%5, %cst_f32_0 : tensor<12x128x4x12x1xf32>, tensor<1x1x1x1x1xf32>) -> tensor<12x128x4x12x1xf32>
  %7 = tensorrt.element_wise <kPROD>(%6, %cst_f32 : tensor<12x128x4x12x1xf32>, tensor<1x1x1x1x1xf32>) -> tensor<12x128x4x12x1xf32>
  %8 = tensorrt.element_wise <kPROD>(%arg0, %7 : tensor<12x128x4x12x1xf32>, tensor<12x128x4x12x1xf32>) -> tensor<12x128x4x12x1xf32>
  return %8 : tensor<12x128x4x12x1xf32>
}

// CHECK-LABEL: func.func @raise_gelu
//  CHECK-SAME: (%[[arg0:.+]]: tensor<12x128x4x12x1xf32>) -> tensor<12x128x4x12x1xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.activation {activationType = #tensorrt.activation_type<kGELU_TANH>} %[[arg0]] : tensor<12x128x4x12x1xf32>
//       CHECK:     return %[[v0]] : tensor<12x128x4x12x1xf32>

// TRT8-LABEL: func.func @raise_gelu
//   TRT8-NOT:  kGELU_TANH
