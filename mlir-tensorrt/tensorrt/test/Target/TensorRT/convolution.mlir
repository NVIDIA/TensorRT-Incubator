// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

// CHECK-LABEL: @trt_2d_convolution
//  CHECK-SAME: tensorrt.engine
func.func @trt_2d_convolution(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x64x128x128xf32> {
  %0 = tensorrt.convolution {
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    stride = array<i64: 1, 1>,
    biasStatic = dense<0.1>:tensor<64xf32>,
    kernelStatic = dense<0.1>:tensor<64x32x3x3xf32>
  } in (%arg0: tensor<1x32x128x128xf32>) -> tensor<1x64x128x128xf32>
  return %0 : tensor<1x64x128x128xf32>
}

// -----

// CHECK-LABEL: @trt_2d_f16_convolution
//  CHECK-SAME: tensorrt.engine
func.func @trt_2d_f16_convolution(%arg0: tensor<1x32x128x128xf16>) -> tensor<1x64x128x128xf16> {
  %0 = tensorrt.convolution {
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    stride = array<i64: 1, 1>,
    kernelStatic = dense<0.1>:tensor<64x32x3x3xf16>
  } in (%arg0 : tensor<1x32x128x128xf16>) -> tensor<1x64x128x128xf16>
  return %0 : tensor<1x64x128x128xf16>
}

// -----

// CHECK-LABEL: @conv_blob_weights
//  CHECK-SAME: tensorrt.engine
func.func @conv_blob_weights(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x6x224x224xf32> {
    %cst_f32 = tensorrt.constant dense_resource<torch_tensor_6_3_1_1_torch.float32> : tensor<6x3x1x1xf32>
    %0 = tensorrt.convolution {dilation = array<i64: 1, 1>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>} in(%arg0 : tensor<1x3x224x224xf32>) kernel(%cst_f32 : tensor<6x3x1x1xf32>) -> tensor<1x6x224x224xf32>
    return %0 : tensor<1x6x224x224xf32>
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_6_3_1_1_torch.float32: "0x040000004AAA07BFF08FC5BE8B3FA23EEDBE51BE3D60FE3DFF5B613E480FE23D09DA433D1A65F43D411FACBE0A7B0BBFDAA3833E33A5943E6446BFBE771970BE2E9F0F3E39B0C5BE0F22ABBD",
      torch_tensor_6_torch.float32: "0x040000002EA9ADBE425CBABE2FCBE43EC8108D3E55C627BED2AF97BD"
    }
  }
#-}
