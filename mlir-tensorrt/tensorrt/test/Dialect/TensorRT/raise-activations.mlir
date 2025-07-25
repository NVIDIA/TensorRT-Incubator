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

// -----

// CHECK-LABEL: func.func @raise_gelu2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<16x1024x1024xbf16>) -> tensor<16x1024x1024xbf16> {
//       CHECK:     %[[v0:.+]] = tensorrt.activation {activationType = #tensorrt.activation_type<kGELU_TANH>} %[[arg0]] : tensor<16x1024x1024xbf16>
//       CHECK:     return %[[v0]] : tensor<16x1024x1024xbf16>

// TRT8-LABEL: func.func @raise_gelu2
//   TRT8-NOT:  kGELU_TANH

func.func @raise_gelu2(%arg0: tensor<16x1024x1024xbf16>) -> tensor<16x1024x1024xbf16> {
    %cst_bf16 = tensorrt.constant dense<5.000000e-01> : tensor<1x1x1xbf16>
    %cst_bf16_0 = tensorrt.constant dense<3.000000e+00> : tensor<1x1x1xbf16>
    %cst_bf16_2 = tensorrt.constant dense<6.367190e-01> : tensor<1x1x1xbf16>
    %cst_bf16_3 = tensorrt.constant dense<4.467770e-02> : tensor<1x1x1xbf16>
    %cst_bf16_5 = tensorrt.constant dense<1.000000e+00> : tensor<1x1x1xbf16>
    %0 = tensorrt.slice %cst_bf16[0, 0, 0][16, 1024, 1024][1, 1, 1] {mode = #tensorrt.slice_mode<kWRAP>} : tensor<1x1x1xbf16> to tensor<16x1024x1024xbf16>
    %5 = tensorrt.element_wise <kPROD>(%arg0, %cst_bf16 : tensor<16x1024x1024xbf16>, tensor<1x1x1xbf16>) -> tensor<16x1024x1024xbf16>
    %6 = tensorrt.element_wise <kPOW>(%cst_bf16_2, %0 : tensor<1x1x1xbf16>, tensor<16x1024x1024xbf16>) -> tensor<16x1024x1024xbf16>
    %7 = tensorrt.element_wise <kPOW>(%arg0, %cst_bf16_0 : tensor<16x1024x1024xbf16>, tensor<1x1x1xbf16>) -> tensor<16x1024x1024xbf16>
    %8 = tensorrt.element_wise <kPROD>(%7, %cst_bf16_3 : tensor<16x1024x1024xbf16>, tensor<1x1x1xbf16>) -> tensor<16x1024x1024xbf16>
    %9 = tensorrt.element_wise <kSUM>(%arg0, %8 : tensor<16x1024x1024xbf16>, tensor<16x1024x1024xbf16>) -> tensor<16x1024x1024xbf16>
    %10 = tensorrt.element_wise <kPROD>(%6, %9 : tensor<16x1024x1024xbf16>, tensor<16x1024x1024xbf16>) -> tensor<16x1024x1024xbf16>
    %11 = tensorrt.activation {activationType = #tensorrt.activation_type<kTANH>} %10 : tensor<16x1024x1024xbf16>
    %12 = tensorrt.element_wise <kSUM>(%11, %cst_bf16_5 : tensor<16x1024x1024xbf16>, tensor<1x1x1xbf16>) -> tensor<16x1024x1024xbf16>
    %13 = tensorrt.element_wise <kPROD>(%5, %12 : tensor<16x1024x1024xbf16>, tensor<16x1024x1024xbf16>) -> tensor<16x1024x1024xbf16>
    return %13 : tensor<16x1024x1024xbf16>
}

// -----

// CHECK-LABEL: func.func @raise_gelu_erf
//  CHECK-SAME: (%[[arg0:.+]]: tensor<16x1024x1024xbf16>) -> tensor<16x1024x1024xbf16> {
//       CHECK:     %[[v0:.+]] = tensorrt.activation {activationType = #tensorrt.activation_type<kGELU_ERF>} %[[arg0]] : tensor<16x1024x1024xbf16>
//       CHECK:     return %[[v0]] : tensor<16x1024x1024xbf16>

// TRT8-LABEL: func.func @raise_gelu_erf
//   TRT8-NOT:  kGELU_ERF

func.func @raise_gelu_erf(%arg0: tensor<16x1024x1024xbf16>) -> tensor<16x1024x1024xbf16> {
    %cst_bf16_1 = tensorrt.constant dense<1.000000e+00> : tensor<1x1x1xbf16>
    %cst_bf16_2 = tensorrt.constant dense<5.000000e-01> : tensor<1x1x1xbf16>
    %cst_bf16_3 = tensorrt.constant dense<7.070310e-01> : tensor<1x1x1xbf16>
    %5 = tensorrt.element_wise <kPROD>(%arg0, %cst_bf16_3 : tensor<16x1024x1024xbf16>, tensor<1x1x1xbf16>) -> tensor<16x1024x1024xbf16>
    %6 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kERF>} %5 : tensor<16x1024x1024xbf16>
    %7 = tensorrt.element_wise <kSUM>(%6, %cst_bf16_1 : tensor<16x1024x1024xbf16>, tensor<1x1x1xbf16>) -> tensor<16x1024x1024xbf16>
    %8 = tensorrt.element_wise <kPROD>(%7, %cst_bf16_2 : tensor<16x1024x1024xbf16>, tensor<1x1x1xbf16>) -> tensor<16x1024x1024xbf16>
    %9 = tensorrt.element_wise <kPROD>(%arg0, %8 : tensor<16x1024x1024xbf16>, tensor<16x1024x1024xbf16>) -> tensor<16x1024x1024xbf16>
    return %9 : tensor<16x1024x1024xbf16>
}

// -----

// CHECK: @raise_min_max(%[[arg0:.+]]: tensor<16x1024x1024xbf16>)
// CHECK: #tensorrt.activation_type<kCLIP>
func.func @raise_min_max(%arg0: tensor<16x1024x1024xbf16>) -> tensor<16x1024x1024xbf16> {
    %cst_f32 = tensorrt.constant dense<6.000000e+00> : tensor<f32>
    %cst_f32_1 = tensorrt.constant dense<0.000000e+00> : tensor<f32>
    %5 = tensorrt.cast %cst_f32_1 : tensor<f32> to tensor<bf16>
    %6 = tensorrt.expand_rank %5 : tensor<bf16> to tensor<1x1x1xbf16>
    %8 = tensorrt.cast %cst_f32 : tensor<f32> to tensor<bf16>
    %9 = tensorrt.expand_rank %8 : tensor<bf16> to tensor<1x1x1xbf16>
    %15 = tensorrt.element_wise <kMAX>(%arg0, %6 : tensor<16x1024x1024xbf16>, tensor<1x1x1xbf16>) -> tensor<16x1024x1024xbf16>
    %16 = tensorrt.element_wise <kMIN>(%9, %15 : tensor<1x1x1xbf16>, tensor<16x1024x1024xbf16>) -> tensor<16x1024x1024xbf16>
    return %16 : tensor<16x1024x1024xbf16>
}
