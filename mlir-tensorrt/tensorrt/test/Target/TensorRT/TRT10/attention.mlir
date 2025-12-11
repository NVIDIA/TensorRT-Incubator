// REQUIRES: tensorrt-version-ge-10.14
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_attention_f16
//  CHECK-SAME: tensorrt.engine
func.func @trt_attention_f16(%arg0: tensor<2x8x128x64xf16>, 
                               %arg1: tensor<2x8x128x64xf16>, 
                               %arg2: tensor<2x8x128x64xf16>) 
                               -> tensor<2x8x128x64xf16> {
  %0 = tensorrt.attention ins(%arg0, %arg1, %arg2 : 
      tensor<2x8x128x64xf16>, tensor<2x8x128x64xf16>, tensor<2x8x128x64xf16>)
      -> tensor<2x8x128x64xf16>
  return %0 : tensor<2x8x128x64xf16>
}

// CHECK-LABEL: @trt_attention_causal_f16
//  CHECK-SAME: tensorrt.engine
func.func @trt_attention_causal_f16(%arg0: tensor<2x8x128x64xf16>, 
                                      %arg1: tensor<2x8x128x64xf16>, 
                                      %arg2: tensor<2x8x128x64xf16>) 
                                      -> tensor<2x8x128x64xf16> {
  %0 = tensorrt.attention {causal = true} ins(%arg0, %arg1, %arg2 : 
      tensor<2x8x128x64xf16>, tensor<2x8x128x64xf16>, tensor<2x8x128x64xf16>)
      -> tensor<2x8x128x64xf16>
  return %0 : tensor<2x8x128x64xf16>
}

// CHECK-LABEL: @trt_attention_with_mask_f16
//  CHECK-SAME: tensorrt.engine
func.func @trt_attention_with_mask_f16(%arg0: tensor<2x8x128x64xf16>, 
                                         %arg1: tensor<2x8x128x64xf16>, 
                                         %arg2: tensor<2x8x128x64xf16>,
                                         %mask: tensor<2x8x128x128xf16>) 
                                         -> tensor<2x8x128x64xf16> {
  %0 = tensorrt.attention ins(%arg0, %arg1, %arg2, mask = %mask : 
      tensor<2x8x128x64xf16>, tensor<2x8x128x64xf16>, tensor<2x8x128x64xf16>, tensor<2x8x128x128xf16>)
      -> tensor<2x8x128x64xf16>
  return %0 : tensor<2x8x128x64xf16>
}

// CHECK-LABEL: @trt_attention_with_quantization_f16
//  CHECK-SAME: tensorrt.engine
func.func @trt_attention_with_quantization_f16(%arg0: tensor<2x8x128x64xf16>, 
                                                 %arg1: tensor<2x8x128x64xf16>, 
                                                 %arg2: tensor<2x8x128x64xf16>) 
                                                 -> tensor<2x8x128x64xf16> {
  %scale = tensorrt.constant dense<1.0> : tensor<f32>
  %0 = tensorrt.attention {
      normalization_quantize_to_type = #tensorrt.data_type<kFP8>
    } ins(%arg0, %arg1, %arg2, 
          normalization_quantize_scale = %scale :
          tensor<2x8x128x64xf16>, tensor<2x8x128x64xf16>, 
          tensor<2x8x128x64xf16>, tensor<f32>)
      -> tensor<2x8x128x64xf16>
  return %0 : tensor<2x8x128x64xf16>
}

// CHECK-LABEL: @trt_attention_decomposable_f16
//  CHECK-SAME: tensorrt.engine
func.func @trt_attention_decomposable_f16(%arg0: tensor<2x8x128x64xf16>, 
                                            %arg1: tensor<2x8x128x64xf16>, 
                                            %arg2: tensor<2x8x128x64xf16>) 
                                            -> tensor<2x8x128x64xf16> {
  %0 = tensorrt.attention {decomposable = true} ins(%arg0, %arg1, %arg2 : 
      tensor<2x8x128x64xf16>, tensor<2x8x128x64xf16>, tensor<2x8x128x64xf16>)
      -> tensor<2x8x128x64xf16>
  return %0 : tensor<2x8x128x64xf16>
}

