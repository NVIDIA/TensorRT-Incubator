// RUN: tensorrt-opt -allow-unregistered-dialect -split-input-file --verify-diagnostics %s

// This file contains all negative tests for TensorRT dialect operations. The `tensorrt-opt` flag `--verify-diagnostics`
// allows `tensorrt-opt` to check for particular diagnostic messages.

func.func @trt_einsum(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>)->tensor<?xf32> {
  // expected-error @below {{einsum op may only have 1 or 2 inputs}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "i,i,i->i"
  } ins(%arg0, %arg1, %arg2: tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<?x?x128x64xf32>, %arg1: tensor<64x256xf32>) -> tensor<?x?x128x256xf32> {
  // expected-error @below {{einsum equation syntax is invalid. TensorRT only supports ASCII lower-case letters with no ellipses.}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "ABcd,de->ABce"
  } ins(%arg0, %arg1 : tensor<?x?x128x64xf32>, tensor<64x256xf32>) -> tensor<?x?x128x256xf32>
  return %0 : tensor<?x?x128x256xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>)->tensor<?x?xf32> {
  // expected-error @below {{einsum equation syntax is invalid. TensorRT only supports ASCII lower-case letters with no ellipses.}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "...i,...i->i"
  } ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>)->tensor<?xf32> {
  // expected-error @below {{each tensor input should have a subscript. Received 2 tensor operands and 1 input subscripts}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "i->i"
  } ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // expected-error @below {{'tensorrt.einsum' op inferred type(s) 'tensor<2x3xf32>' are incompatible with return type(s) of operation 'tensor<3x2xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "ij"
  } ins(%arg0 : tensor<2x3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<5x5xf32>) -> tensor<5xf32> {
  // expected-error @below {{'tensorrt.einsum' op inferred type(s) 'tensor<f32>' are incompatible with return type(s) of operation 'tensor<5xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "ii"
  } ins(%arg0 : tensor<5x5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<2x3xf32>) -> tensor<f32> {
  // expected-error @below {{'tensorrt.einsum' op inferred type(s) 'tensor<2x3xf32>' are incompatible with return type(s) of operation 'tensor<f32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "ij"
  } ins(%arg0 : tensor<2x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @trt_einsum(%arg0: tensor<4x7x8x9xf32>) -> tensor<7x4x8x9xf32> {
  // expected-error @below {{'tensorrt.einsum' op inferred type(s) 'tensor<7x4x9x8xf32>' are incompatible with return type(s) of operation 'tensor<7x4x8x9xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "ijkl->jilk"
  } ins(%arg0: tensor<4x7x8x9xf32>) ->tensor<7x4x8x9xf32>
  return %0 : tensor<7x4x8x9xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<4x7xf32>, %arg1: tensor<7x9xf32>) -> tensor<4x9xf32> {
  // expected-error @below {{'tensorrt.einsum' op inferred type(s) 'tensor<9x4xf32>' are incompatible with return type(s) of operation 'tensor<4x9xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "zj,jl"
  } ins(%arg0, %arg1 : tensor<4x7xf32>, tensor<7x9xf32>) ->tensor<4x9xf32>
  return %0 : tensor<4x9xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<2x3xf32>) -> tensor<f32> {
  // expected-error @below {{'tensorrt.einsum' op inferred type(s) 'tensor<2xf32>' are incompatible with return type(s) of operation 'tensor<f32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "ij->i"
  } ins(%arg0 : tensor<2x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @trt_einsum(%arg0: tensor<4x7xf32>, %arg1: tensor<8x9xf32>) -> tensor<4x9xf32> {
  // expected-error @below {{label `j` is repeated between inputs but dimensions are not same}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "ij,jk"
  } ins(%arg0, %arg1 : tensor<4x7xf32>, tensor<8x9xf32>) ->tensor<4x9xf32>
  return %0 : tensor<4x9xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<5x4x7xf32>, %arg1: tensor<5x7x9xf32>) -> tensor<5x4x9xf32> {
  // expected-error @below {{each tensor dimension must have a label. Tensor input 0 has rank of 3 but subscript size is 2}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "ij,bjk -> bik"
  } ins(%arg0, %arg1 : tensor<5x4x7xf32>, tensor<5x7x9xf32>) ->tensor<5x4x9xf32>
  return %0 : tensor<5x4x9xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<5x4x7xf32>, %arg1: tensor<5x7x9xf32>) -> tensor<5x4x9xf32> {
  // expected-error @below {{output label `z` does not appear in the input subscript string}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "bij,bjk -> zik"
  } ins(%arg0, %arg1 : tensor<5x4x7xf32>, tensor<5x7x9xf32>) ->tensor<5x4x9xf32>
  return %0 : tensor<5x4x9xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<4x7xf32>, %arg1: tensor<7x9xf32>) -> tensor<4x9xf32> {
  // expected-error @below {{label `j` is repeated in the output substring}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "ij,jk->jj"
  } ins(%arg0, %arg1 : tensor<4x7xf32>, tensor<7x9xf32>) ->tensor<4x9xf32>
  return %0 : tensor<4x9xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<5x4x7xf32>, %arg1: tensor<5x7x9xf32>) -> tensor<5x7x9xf32> {
  // expected-error @below {{'tensorrt.einsum' op inferred type(s) 'tensor<5x4x9xf32>' are incompatible with return type(s) of operation 'tensor<5x7x9xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "bij,bjk->bik"
  } ins(%arg0, %arg1 : tensor<5x4x7xf32>, tensor<5x7x9xf32>) ->tensor<5x7x9xf32>
  return %0 : tensor<5x7x9xf32>
}

// -----

func.func @trt_einsum(%arg0: tensor<2x3xf32>) -> tensor<3xf32> {
  // expected-error @below {{'tensorrt.einsum' op inferred type(s) 'tensor<2xf32>' are incompatible with return type(s) of operation 'tensor<3xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.einsum {
    equation = "ij->i"
  } ins(%arg0 : tensor<2x3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x128x128xf32> {
  // expected-error @below {{'tensorrt.convolution' op stride/pre_padding/post_padding should have size equal to the number of spatial dimensions}}
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1, 1, 1>,
    pre_padding = array<i64: 1, 1, 1, 1>,
    post_padding = array<i64: 1, 1, 1, 1>
  } in(%arg0: tensor<?x32x128x128xf32>) kernel(%arg1:tensor<64x32x3x3xf32>) bias(%arg2:tensor<64xf32>) -> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<1x64x1x1xf32>) -> tensor<?x64x128x128xf32> {
  // expected-error @below {{'tensorrt.convolution' op bias type should be a rank-1 tensor type with size equal to the number of channels (dim 1) of the result tensor type}}
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0:tensor<?x32x128x128xf32>) kernel(%arg1:tensor<64x32x3x3xf32>) bias(%arg2:tensor<1x64x1x1xf32>) -> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>) -> tensor<?x64x128x128xf32> {
  // expected-error @below {{'tensorrt.convolution' op kernel operand or kernelStatic attribute must be specified}}
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0:tensor<?x32x128x128xf32>)-> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>) -> tensor<?x64x128x128xf32> {
  // expected-error @below {{'tensorrt.convolution' op only one of kernel operand or kernelStatic attribute can be specified}}
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    kernelStatic = dense<0.1>:tensor<64x32x3x3xf32>
  } in(%arg0:tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x3x3xf32>)-> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}

// -----

// Checks that AND, OR, and XOR should only support boolean inputs and results

func.func @trt_element_wise_AND_bool_only(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  // expected-error @below {{'tensorrt.element_wise' op ElementWiseOperation type kAND expected all input types to be i1 (bool)}}
  %0 = tensorrt.element_wise <kAND>(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @trt_element_wise_OR_bool_only(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @below {{'tensorrt.element_wise' op ElementWiseOperation type kOR expected all input types to be i1 (bool)}}
  %0 = tensorrt.element_wise <kOR>(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @trt_element_wise_XOR_bool_only(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @below {{'tensorrt.element_wise' op ElementWiseOperation type kXOR expected all input types to be i1 (bool)}}
  %0 = tensorrt.element_wise <kXOR>(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// Two shapes must have the same rank to be broadcastable. Elementwise operand types must have broadcastable shapes.

func.func @trt_element_wise_not_broadcastable(%arg0: tensor<?xf32>, %arg1: tensor<10x1xf32>) -> tensor<10x?xf32> {
  // expected-error @below {{'tensorrt.element_wise' op failed to verify that all of {input1, input2, result} have same rank}}
  %0 = tensorrt.element_wise <kSUM>(%arg0, %arg1 : tensor<?xf32>, tensor<10x1xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @trt_element_wise_compare_gt(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'tensorrt.element_wise' op result element type expected to be i1 (bool)}}
  %0 = tensorrt.element_wise <kGREATER>(%arg0, %arg1: tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @trt_element_wise_compare_lt(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'tensorrt.element_wise' op result element type expected to be i1 (bool)}}
  %0 = tensorrt.element_wise <kLESS>(%arg0, %arg1: tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @trt_element_wise_compare_equal(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'tensorrt.element_wise' op result element type expected to be i1 (bool)}}
  %0 = tensorrt.element_wise <kEQUAL>(%arg0, %arg1: tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @broadcast_ewise_double(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<128xf32> {
  // expected-error @below {{'tensorrt.element_wise' op inferred type(s) 'tensor<1xf32>' are incompatible with return type(s) of operation 'tensor<128xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.element_wise <kSUM>(%arg0, %arg1 : tensor<1xf32>, tensor<1xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// -----

func.func @trt_fill_linspace(%arg0: tensor<2xi32>) -> tensor<?x?xf32> {
  // expected-error @below {{'tensorrt.linspace' op dynamic `step` must be specified if the result is greater than rank 1}}
  %0 = tensorrt.linspace [0.0][%arg0: tensor<2xi32>][1.0] : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @trt_fill_linspace(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  // expected-error @below {{'tensorrt.linspace' op `step` element type and result element type should be the same}}
  %0 = tensorrt.linspace [0.0][%arg0: tensor<2xi32>][%arg1: tensor<2xi32>] : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @trt_fill_linspace(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<?x?x?xi32> {
  // expected-error @below {{'tensorrt.linspace' op result 0 has type tensor<?x?x?xi32> but inferred tensor of shape <?x?>}}
  %0 = tensorrt.linspace [0.0][%arg0: tensor<2xi32>][%arg1: tensor<2xi32>] : tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
}

// -----

func.func @trt_expand_rank(%arg0: tensor<100x100xf32>) -> tensor<100x100xf32> {
  // expected-error @below {{'tensorrt.expand_rank' op the reshape is not a valid rank expansion produced from inserting 1's}}
  %0 = tensorrt.expand_rank %arg0 : tensor<100x100xf32> to tensor<100x100xf32>
  return %0 : tensor<100x100xf32>
}

// -----

// You cannot remove or collapse dimensions with `tensorrt.expand_rank`.

func.func @trt_expand_rank1(%arg0: tensor<100x100xf32>) -> tensor<100xf32> {
  // expected-error @below {{'tensorrt.expand_rank' op result rank should be greater than or equal to input rank}}
  %0 = tensorrt.expand_rank %arg0 : tensor<100x100xf32> to tensor<100xf32>
  return %0 : tensor<100xf32>
}

// -----

func.func @trt_expand_rank2(%arg0: tensor<1x100xf32>) -> tensor<100xf32> {
  // expected-error @below {{'tensorrt.expand_rank' op result rank should be greater than or equal to input rank}}
  %0 = tensorrt.expand_rank %arg0 : tensor<1x100xf32> to tensor<100xf32>
  return %0 : tensor<100xf32>
}

// -----

func.func @trt_expand_rank3(%arg0: tensor<100xf32>) -> tensor<10x10xf32> {
  // expected-error @below {{'tensorrt.expand_rank' op the reshape is not a valid rank expansion produced from inserting 1's}}
  %0 = tensorrt.expand_rank %arg0 : tensor<100xf32> to tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// -----

func.func @trt_collapse_rank(%arg0: tensor<100x1x1x100xf32>) -> tensor<100x1x100x1xf32> {
  // expected-error @below {{'tensorrt.collapse_rank' op failed to compute a reassociation map from input to output type}}
  %0 = tensorrt.collapse_rank %arg0 : tensor<100x1x1x100xf32> to tensor<100x1x100x1xf32>
  return %0 : tensor<100x1x100x1xf32>
}

// -----

// Collapse rank can only remove unit dims.

func.func @trt_collapse_rank(%arg0: tensor<10x10x1xf32>) -> tensor<100x1xf32> {
  // expected-error @below {{'tensorrt.collapse_rank' op expected to only drop unit-dims from the shape}}
  %0 = tensorrt.collapse_rank %arg0 : tensor<10x10x1xf32> to tensor<100x1xf32>
  return %0 : tensor<100x1xf32>
}

// -----

func.func @trt_collapse_rank(%arg0: tensor<100x1x1x100xf32>) -> tensor<100x1x1x1x100xf32> {
  // expected-error @below {{'tensorrt.collapse_rank' op input type rank should be greater than or equal to result type rank}}
  %0 = tensorrt.collapse_rank %arg0 : tensor<100x1x1x100xf32> to tensor<100x1x1x1x100xf32>
  return %0 : tensor<100x1x1x1x100xf32>
}

// -----

func.func @trt_reshape(%arg0: tensor<100x100xf32>) -> tensor<100x2xf32> {
  // expected-error @below {{'tensorrt.reshape' op input and result tensor types should have the same number of elements}}
  %0 = tensorrt.reshape %arg0 : tensor<100x100xf32> to tensor<100x2xf32>
  return %0 : tensor<100x2xf32>
}

// -----

func.func @trt_reshape(%arg0: tensor<?x?xf32>, %arg1: tensor<?xi32>) -> tensor<?x?xf32> {
  // expected-error @below {{'tensorrt.reshape' op dynamic reshape must be a 1D tensor of known shape}}
  %0 = tensorrt.reshape %arg0 shape(%arg1: tensor<?xi32>) : tensor<?x?xf32> to tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @trt_reshape(%arg0: tensor<?x?xf32>, %arg1: tensor<3xi32>) -> tensor<?x?xf32> {
  // expected-error @below {{'tensorrt.reshape' op reshape tensor size (3) does not match result type rank (1)}}
  %0 = tensorrt.reshape %arg0 shape(%arg1: tensor<3xi32>) : tensor<?x?xf32> to tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @trt_reshape(%arg0: tensor<?xf32>) -> tensor<?x2x?xf32> {
  // expected-error @below {{'tensorrt.reshape' op result type may have at most one dynamic dimension if the 'shape' operand is not provided}}
  %0 = tensorrt.reshape %arg0  : tensor<?xf32> to tensor<?x2x?xf32>
  return %0 : tensor<?x2x?xf32>
}


// -----

func.func @trt_topk_wrong_rank(%arg0: tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?xi32>) {
  // expected-error @below {{'tensorrt.top_k' op failed to verify that all of {input, values, indices} have same rank}}
  %0, %1 = tensorrt.top_k <kMAX> {
    k = 1 : i64,
    axis = 1 : i64
  } %arg0 : tensor<?x?xf32> -> tensor<?xf32>, tensor<?xi32>
  return %0, %1 : tensor<?xf32>, tensor<?xi32>
}

// -----

func.func @trt_topk_wrong_shape(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // expected-error @below {{'tensorrt.top_k' op expected result shapes dim 1 to equal 1}}
  %0, %1 = tensorrt.top_k <kMAX> {
    k = 1 : i64,
    axis = 1 : i64
  } %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>, tensor<?x?xi32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

func.func @trt_topk_mask_oob(%arg0: tensor<?x?xf32>) -> (tensor<?x1xf32>, tensor<?x1xi32>) {
  // expected-error @below {{'tensorrt.top_k' op "axis" attribute is 2, but this is out of bounds for input of rank 2}}
  %0, %1 = tensorrt.top_k <kMAX> {
      k = 1 : i64,
      axis = 2 : i64
    } %arg0 : tensor<?x?xf32> -> tensor<?x1xf32>, tensor<?x1xi32>
  return %0, %1 : tensor<?x1xf32>, tensor<?x1xi32>
}

// -----

func.func @trt_topk_mask_oob(%arg0: tensor<?x?xf32>) -> (tensor<?x1xf32>, tensor<?x1xi32>) {
  // expected-error @below {{'tensorrt.top_k' op "axis" attribute is -1, but this is out of bounds for input of rank 2}}
  %0, %1 = tensorrt.top_k <kMAX> {
    k = 1 : i64,
    axis = -1 : i64
  } %arg0 : tensor<?x?xf32> -> tensor<?x1xf32>, tensor<?x1xi32>
  return %0, %1 : tensor<?x1xf32>, tensor<?x1xi32>
}

// -----

func.func @trt_topk(%arg0: tensor<?x10xf32>) -> (tensor<?x11xf32>, tensor<?x11xi32>) {
  // expected-error @below {{'tensorrt.top_k' op expected K attribute value to be smaller than the dimension specified by "axis"}}
  %0, %1 = tensorrt.top_k <kMAX> {
    k = 11 : i64,
    axis = 1 : i64
  } %arg0 : tensor<?x10xf32> -> tensor<?x11xf32>, tensor<?x11xi32>
  return %0, %1 : tensor<?x11xf32>, tensor<?x11xi32>
}

// -----
func.func @trt_topk_mismatched_dims(%arg0: tensor<10x5xf32>) -> (tensor<9x1xf32>, tensor<9x1xi32>) {
  // expected-error @below {{'tensorrt.top_k' op inferred type(s) 'tensor<10x1xf32>', 'tensor<10x1xi32>' are incompatible with return type(s) of operation 'tensor<9x1xf32>', 'tensor<9x1xi32>'}}
  // expected-error @below {{'tensorrt.top_k' op failed to infer returned types}}
  %0, %1 = tensorrt.top_k <kMAX> {
    k = 1 : i64,
    axis = 1 : i64
  } %arg0 : tensor<10x5xf32> -> tensor<9x1xf32>, tensor<9x1xi32>
  return %0, %1 : tensor<9x1xf32>, tensor<9x1xi32>
}

// -----

func.func @trt_argmin(%arg0: tensor<?x10xf32>) -> (tensor<?x1xf32>, tensor<?x1xi32>) {
  // expected-error @below {{'tensorrt.argmin' op "axis" attribute is -1, but this is out of bounds for input of rank 2}}
  %0, %1 = tensorrt.argmin {
    axis = -1 : i64
  } %arg0 : tensor<?x10xf32> -> tensor<?x1xf32>, tensor<?x1xi32>
  return %0, %1 : tensor<?x1xf32>, tensor<?x1xi32>
}

// -----

func.func @trt_reduce(%arg0: tensor<?x2xf32>) -> (tensor<?x2x1xf32>) {
  // expected-error @below {{'tensorrt.reduce' op expected each element of reduceAxes to be smaller than input type rank (2)}}
  %0 = tensorrt.reduce <kMIN> %arg0 {keepDimensions = false, reduceAxes = array<i64: 2>} : tensor<?x2xf32> -> tensor<?x2x1xf32>
  return %0 : tensor<?x2x1xf32>
}
// -----

func.func @concat(%arg0: tensor<5x5x1x32xf32>, %arg1: tensor<5x5x3x32xf32>, %arg2: tensor<5x5x5x32xf32>) -> tensor<5x5x9x32xf32> {
  // expected-error @below {{'tensorrt.concatenation' op concat axis exceeds the dimension of input tensor.}}
  %0 = tensorrt.concatenation {axis = 4 : i32, nbInputs = 3 : i32} ins(%arg0, %arg1, %arg2 : tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>) -> tensor<5x5x9x32xf32>
  return %0 : tensor<5x5x9x32xf32>
}

// -----

func.func @trt_reduce(%arg0: tensor<?x2xf32>) -> (tensor<?xf32>) {
  // expected-error @below {{'tensorrt.reduce' op inferred type(s) 'tensor<?x1xf32>' are incompatible with return type(s) of operation 'tensor<?xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.reduce <kMIN> %arg0 {keepDimensions = true, reduceAxes = array<i64: 1>} : tensor<?x2xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @trt_reduce(%arg0: tensor<?x2xf32>) -> (tensor<?xf32>) {
  // expected-error @below {{'tensorrt.reduce' op expected each element of reduceAxes to be smaller than input type rank (2)}}
  %0 = tensorrt.reduce <kMIN> %arg0 {keepDimensions = false, reduceAxes = array<i64: 2>} : tensor<?x2xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @trt_reduce(%arg0: tensor<?x2x2xf32>) -> (tensor<?xf32>) {
  // expected-error @below {{'tensorrt.reduce' op inferred type(s) 'tensor<?x2xf32>' are incompatible with return type(s) of operation 'tensor<?xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.reduce <kMIN> %arg0 {keepDimensions = false, reduceAxes = array<i64: 1>} : tensor<?x2x2xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @trt_reduce(%arg0: tensor<?x3x2xf32>) -> (tensor<?x3x2xf32>) {
  // expected-error @below {{'tensorrt.reduce' op inferred type(s) 'tensor<?x1x2xf32>' are incompatible with return type(s) of operation 'tensor<?x3x2xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.reduce <kMIN> %arg0 {keepDimensions = true, reduceAxes = array<i64: 1>} : tensor<?x3x2xf32> -> tensor<?x3x2xf32>
  return %0 : tensor<?x3x2xf32>
}

// -----

func.func @trt_reduce(%arg0: tensor<?x3x2xf32>) -> (tensor<?x3xf32>) {
  // expected-error @below {{'tensorrt.reduce' op inferred type(s) 'tensor<?x2xf32>' are incompatible with return type(s) of operation 'tensor<?x3xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.reduce <kMIN> %arg0 {keepDimensions = false, reduceAxes = array<i64: 1>} : tensor<?x3x2xf32> -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// -----

func.func @concat(%arg0: tensor<5x5x1x32xf32>, %arg1: tensor<5x5x3x32xf32>, %arg2: tensor<5x5x32xf32>) -> tensor<5x5x9x32xf32> {
  // expected-error @below {{'tensorrt.concatenation' op input rank at input[2] is 3 which is different from the rank of input tensor at index 0}}
  %0 = tensorrt.concatenation {axis = 2 : i32} ins(%arg0, %arg1, %arg2 : tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x32xf32>) -> tensor<5x5x9x32xf32>
  return %0 : tensor<5x5x9x32xf32>
}
// -----

func.func @concat(%arg0: tensor<5x5x1x32xf32>, %arg1: tensor<5x5x3x32xf32>, %arg2: tensor<5x3x5x32xf32>) -> tensor<5x5x9x32xf32> {
  // expected-error @below {{'tensorrt.concatenation' op input tensor[2] has size 3 at dimension 1, while the value should be 5}}
  %0 = tensorrt.concatenation {axis = 2 : i32} ins(%arg0, %arg1, %arg2 : tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x3x5x32xf32>) -> tensor<5x5x9x32xf32>
  return %0 : tensor<5x5x9x32xf32>
}

// -----

func.func @trt_concatenation_different_types(%arg0: tensor<1x10xf32>, %arg1: tensor<3x10xf16>) -> tensor<4x10xf32> {
  // expected-error @below {{'tensorrt.concatenation' op requires the same element type for all operands and results}}
  %0 = tensorrt.concatenation {axis = 0 : i32} ins(%arg0, %arg1: tensor<1x10xf32>, tensor<3x10xf16>)
    -> tensor<4x10xf32>
  return %0 : tensor<4x10xf32>
}

// -----

func.func @concat_infer_test(%arg0: tensor<5x5x1x32xf32>, %arg1: tensor<5x5x3x32xf32>, %arg2: tensor<5x5x5x32xf32>) -> tensor<5x5x12x32xf32> {
  // expected-error @below {{'tensorrt.concatenation' op inferred type(s) 'tensor<5x5x9x32xf32>' are incompatible with return type(s) of operation 'tensor<5x5x12x32xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.concatenation {axis = 2 : i32}
    ins(%arg0, %arg1, %arg2 : tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>) -> tensor<5x5x12x32xf32>
  return %0 : tensor<5x5x12x32xf32>
}

// -----

func.func @trt_slice_static(%arg0: tensor<?x?xf32>) -> tensor<2x2xf32> {
// expected-error @below {{'tensorrt.slice' op the size of all static index arrays (start, size, stride) should be equal to the rank of the input tensor type}}
  %0 = tensorrt.slice %arg0 [0, 0][1, 1, 2][2, 2] : tensor<?x?xf32> to tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

func.func @trt_slice_static(%arg0: tensor<?x?xbf16>) -> tensor<2x2xbf16> {
  // expected-error @below {{'tensorrt.slice' op kCLAMP, kFILL and kREFLECT modes do not support bfloat16 type}}
  %0 = tensorrt.slice %arg0 [0, 0][2, 2][1, 1] { mode = #tensorrt.slice_mode<kFILL>} : tensor<?x?xbf16> to tensor<2x2xbf16>
  return %0 : tensor<2x2xbf16>
}
// -----

func.func @trt_slice_dynamic(%arg0: tensor<?x?xf32>, %start: tensor<2xi32>, %stride: tensor<2xi32>, %size: tensor<2xi32>) -> tensor<2xf32> {
  // expected-error @below {{'tensorrt.slice' op inferred type(s) 'tensor<?x?xf32>' are incompatible with return type(s) of operation 'tensor<2xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.slice %arg0 [%start: tensor<2xi32>][%size: tensor<2xi32>][%stride: tensor<2xi32>] : tensor<?x?xf32> to tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

func.func @trt_slice_dynamic(%arg0: tensor<?x?xf32>, %start: tensor<2xi32>, %stride: tensor<2xi32>) -> tensor<10x3xf32> {
  // expected-error @below {{'tensorrt.slice' op inferred type(s) 'tensor<10x2xf32>' are incompatible with return type(s) of operation 'tensor<10x3xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.slice %arg0 [%start: tensor<2xi32>][10, 2][%stride: tensor<2xi32>] : tensor<?x?xf32> to tensor<10x3xf32>
  return %0 : tensor<10x3xf32>
}

// -----

func.func @trt_slice_dynamic(%arg0: tensor<?x?xf32>, %start: tensor<1xi32>, %stride: tensor<1xi32>, %size: tensor<1xi32>) -> tensor<2x2xf32> {
  // expected-error @below {{'tensorrt.slice' op all dynamic index tensors should have a static size equal to the rank of the input tensor type}}
  %0 = tensorrt.slice %arg0 [%start: tensor<1xi32>][%stride: tensor<1xi32>][%size: tensor<1xi32>] : tensor<?x?xf32> to tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

func.func @trt_slice_static(%arg0: tensor<?x?xf32>) -> tensor<1x1xf32> {
  // expected-error @below {{'tensorrt.slice' op inferred type(s) 'tensor<2x2xf32>' are incompatible with return type(s) of operation 'tensor<1x1xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.slice %arg0 [0, 0][2, 2][1, 1] : tensor<?x?xf32> to tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}

// -----

func.func @trt_slice_pad(%arg0: tensor<128x128xf32>) -> tensor<130x130xf32> {
  %cst = tensorrt.constant dense<0.0>:tensor<1xf32>
  // expected-error @below {{'tensorrt.slice' op when a fill value is provided, the slice "mode" must be kFILL}}
  %0 = tensorrt.slice %arg0[-1, -1][130, 130][1, 1] fill(%cst : tensor<1xf32>) : tensor<128x128xf32> to tensor<130x130xf32>
  return %0 : tensor<130x130xf32>
}

// -----

func.func @trt_pooling(%arg0: tensor<10x64x112x112xf32>) -> tensor<10x64x56x56xf32> {
  // expected-error @below {{'tensorrt.pooling' op "stride" array size should be equal to the size of the "windowSize" array}}
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kMAX>,
    windowSize = array<i64: 3, 3>,
    stride = array<i64: 2, 2, 2, 2>,
    prePadding = array<i64: 1, 1>,
    postPadding = array<i64: 1, 1>
  } ins(%arg0 : tensor<10x64x112x112xf32>) -> tensor<10x64x56x56xf32>
  return %0 : tensor<10x64x56x56xf32>
}
// -----

func.func @trt_pooling(%arg0: tensor<10x1x3x3xf32>) -> tensor<10x1xf32> {
  // expected-error @below {{'tensorrt.pooling' op input tensor type and result tensor type should have equal rank}}
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kMAX>,
    windowSize = array<i64: 3, 3>,
    stride = array<i64: 1, 1>,
    prePadding = array<i64: 0, 0>,
    postPadding = array<i64: 0, 0>
  } ins(%arg0 : tensor<10x1x3x3xf32>) -> tensor<10x1xf32>
  return %0 : tensor<10x1xf32>
}

// -----

func.func @trt_max_pool(%arg0: tensor<10x1x3x3xf32>) -> tensor<10x1x1x1xf32> {
  // expected-error @below {{'tensorrt.pooling' op "averageCountExcludesPadding" must be provided when pooling type is "kMAX_AVERAGE_BLEND" or "kMAX", otherwise it should not be provided}}
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kAVERAGE>,
    windowSize = array<i64: 3, 3>,
    stride = array<i64: 1, 1>,
    prePadding = array<i64: 0, 0>,
    postPadding = array<i64: 0, 0>
  } ins(%arg0 : tensor<10x1x3x3xf32>) -> tensor<10x1x1x1xf32>
  return %0 : tensor<10x1x1x1xf32>
}

// -----

func.func @trt_max_pool(%arg0: tensor<10x1x3x3xf32>) -> tensor<10x1x1x1xf32> {
  // expected-error @below {{'tensorrt.pooling' op blendFactor is required when pooling type is "kMAX_AVERAGE_BLEND", otherwise it should not be present}}
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kMAX_AVERAGE_BLEND>,
    windowSize = array<i64: 3, 3>,
    stride = array<i64: 1, 1>,
    prePadding = array<i64: 0, 0>,
    postPadding = array<i64: 0, 0>,
    averageCountExcludesPadding = true
  } ins(%arg0 : tensor<10x1x3x3xf32>) -> tensor<10x1x1x1xf32>
  return %0 : tensor<10x1x1x1xf32>
}

// -----

func.func @trt_max_pool_wrong(%arg0: tensor<10x1x3x3xf16>) -> tensor<10x1x1x1xf32> {
  // expected-error @below {{'tensorrt.pooling' op requires the same element type for all operands and results}}
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kMAX_AVERAGE_BLEND>,
    windowSize = array<i64: 3, 3>,
    stride = array<i64: 1, 1>,
    prePadding = array<i64: 0, 0>,
    postPadding = array<i64: 0, 0>,
    averageCountExcludesPadding = true
  } ins(%arg0 : tensor<10x1x3x3xf16>) -> tensor<10x1x1x1xf32>
  return %0 : tensor<10x1x1x1xf32>
}


// -----

func.func @trt_broadcast(%arg0: tensor<1024xf32>) -> tensor<10x1024x512x2xf32> {
  // expected-error @below {{'tensorrt.broadcast' op expected "broadcast_dims" size to equal input type rank (1)}}
  %0 = tensorrt.broadcast %arg0 broadcast_dims<1, 2> : tensor<1024xf32> to tensor<10x1024x512x2xf32>
  return %0 : tensor<10x1024x512x2xf32>
}

// -----

func.func @trt_broadcast(%arg0: tensor<1024x1024xf32>) -> tensor<10x1024x2048xf32> {
  // expected-error @below {{'tensorrt.broadcast' op expected input shape dimension 0 (1024) to be broadcastable to result type dimension 0 (10)}}
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> : tensor<1024x1024xf32> to tensor<10x1024x2048xf32>
  return %0 : tensor<10x1024x2048xf32>
}

// -----

func.func @trt_broadcast(%arg0: tensor<10x10xf32>) -> tensor<10x1024x2048xf32> {
  // expected-error @below {{'tensorrt.broadcast' op "broadcast_dims" contains duplicate values}}
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 0> : tensor<10x10xf32> to tensor<10x1024x2048xf32>
  return %0 : tensor<10x1024x2048xf32>
}

// -----

func.func @trt_broadcast(%arg0: tensor<10x10xf32>) -> tensor<10x1024x2048xf32> {
  // expected-error @below {{'tensorrt.broadcast' op all "broadcast_dims" values must be in the range [0,3)}}
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 3> : tensor<10x10xf32> to tensor<10x1024x2048xf32>
  return %0 : tensor<10x1024x2048xf32>
}

// -----

func.func @trt_broadcast(%arg0: tensor<1x1xf32>) -> tensor<?x?xf32> {
  // expected-error @below {{'tensorrt.broadcast' op if the result type has unknown dimensions, a shape operand must be provided}}
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> : tensor<1x1xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @trt_broadcast(%arg0: tensor<?x?x?xf32>) -> tensor<10x1x?xf32> {
  // expected-error @below {{'tensorrt.broadcast' op if the result type has unknown dimensions, a shape operand must be provided}}
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1, 2> : tensor<?x?x?xf32> to tensor<10x1x?xf32>
  return %0 : tensor<10x1x?xf32>
}

// -----

func.func @trt_broadcast(%arg0: tensor<?x?x?xf32>, %arg1: tensor<2xi32>) -> tensor<10x1x?xf32> {
  // expected-error @below {{'tensorrt.broadcast' op result 0 has type tensor<10x1x?xf32> but inferred tensor of shape <?x?>}}
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1, 2> shape(%arg1: tensor<2xi32>) : tensor<?x?x?xf32> to tensor<10x1x?xf32>
  return %0 : tensor<10x1x?xf32>
}

// -----

func.func @trt_broadcast(%arg0: tensor<1x1x1xf32>, %arg1: tensor<2xi32>) -> tensor<10x1x10xf32> {
  // expected-error @below {{'tensorrt.broadcast' op result 0 has type tensor<10x1x10xf32> but inferred tensor of shape <?x?>}}
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1, 2> shape(%arg1: tensor<2xi32>) : tensor<1x1x1xf32> to tensor<10x1x10xf32>
  return %0 : tensor<10x1x10xf32>
}

// -----

func.func @trt_broadcast(%arg0: tensor<1x1x1xf32>, %arg1: tensor<?xi32>) -> tensor<10x1x10xf32> {
  // expected-error @below {{'tensorrt.broadcast' op operand #1 must be 1D static i32 tensor representing a shape of 32-bit signless integer values, but got 'tensor<?xi32>'}}
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1, 2> shape(%arg1: tensor<?xi32>) : tensor<1x1x1xf32> to tensor<10x1x10xf32>
  return %0 : tensor<10x1x10xf32>
}

// -----

func.func @trt_identity(%arg0: tensor<10xui8>) -> tensor<10xi32> {
  // expected-error @below {{'tensorrt.identity' op if input element type is ui8, result element type must be f32 or f16}}
  %0 = tensorrt.identity %arg0 : tensor<10xui8> to tensor<10xi32>
  return %0 : tensor<10xi32>
}

// -----

func.func @trt_identity(%arg0: tensor<10xi32>) -> tensor<10xui8> {
  // expected-error @below {{'tensorrt.identity' op if result element type is ui8, input element type must be f32 or f16}}
  %0 = tensorrt.identity %arg0 : tensor<10xi32> to tensor<10xui8>
  return %0 : tensor<10xui8>
}

// -----

func.func @trt_identity(%arg0: tensor<?x1xi32>) -> tensor<?x10xi32> {
  // expected-error @below {{'tensorrt.identity' op result 0 has type tensor<?x10xi32> but inferred tensor of shape <?x1>}}
  %0 = tensorrt.identity %arg0 : tensor<?x1xi32> to tensor<?x10xi32>
  return %0 : tensor<?x10xi32>
}

// -----

func.func @trt_identity(%arg0: tensor<10x2xi32>) -> tensor<10x2x1xi8> {
  // expected-error @below {{'tensorrt.identity' op result 0 has type tensor<10x2x1xi8> but inferred tensor of shape <10x2>}}
  %0 = tensorrt.identity %arg0 : tensor<10x2xi32> to tensor<10x2x1xi8>
  return %0 : tensor<10x2x1xi8>
}

// -----

// TensorRT 8.4 version doesn't have uint8 data type.

func.func @trt_identity84(%arg0: tensor<10xui8>) -> tensor<10xf16> {
  // expected-error @below {{'tensorrt.identity84' op operand #0 must be 0D/1D/2D/3D/4D/5D/6D/7D/8D tensor of 32-bit float or 16-bit float or 32-bit signless integer or 1-bit signless integer values, but got 'tensor<10xui8>'}}
  %0 = tensorrt.identity84 %arg0 : tensor<10xui8> to tensor<10xf16>
  return %0 : tensor<10xf16>
}

// -----

func.func @trt_select(%arg0: tensor<10x10xi1>, %arg1: tensor<1x10xf16>, %arg2: tensor<10x1xf32>) -> tensor<10x10xf32> {
  // expected-error @below {{'tensorrt.select' op thenInput and elseInput must have the same element type}}
  %0 = tensorrt.select ins(%arg0, %arg1, %arg2: tensor<10x10xi1>, tensor<1x10xf16>, tensor<10x1xf32>)
    -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// -----

func.func @trt_select(%arg0: tensor<10x10xi1>, %arg1: tensor<2x10xf32>, %arg2: tensor<10x1xf32>) -> tensor<10x10xf32> {
  // expected-error @below {{failed to determine expected shape}}
  // expected-error @below {{'tensorrt.select' op failed to infer returned types}}
  %0 = tensorrt.select ins(%arg0, %arg1, %arg2: tensor<10x10xi1>, tensor<2x10xf32>, tensor<10x1xf32>) -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// -----

func.func @trt_select(%arg0: tensor<10x10xi1>, %arg1: tensor<1x10xf32>, %arg2: tensor<10x1xf32>) -> tensor<10x11xf32> {
  // expected-error @below {{'tensorrt.select' op inferred type(s) 'tensor<10x10xf32>' are incompatible with return type(s) of operation 'tensor<10x11xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.select ins(%arg0, %arg1, %arg2: tensor<10x10xi1>, tensor<1x10xf32>, tensor<10x1xf32>)
    -> tensor<10x11xf32>
  return %0 : tensor<10x11xf32>
}

// -----

func.func @trt_softmax(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // expected-error @below {{'tensorrt.softmax' op expected axis to be non-negative and less than 2}}
  %0 = tensorrt.softmax {axis = 2 : i64} %arg0 : tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// -----

func.func @trt_softmax(%arg0: tensor<10x10xi8>) -> tensor<10x10xi8> {
  // expected-error @below {{'tensorrt.softmax' op operand #0 must be 0D/1D/2D/3D/4D/5D/6D/7D/8D tensor of 32-bit float or 16-bit float or bfloat16 type values, but got 'tensor<10x10xi8>'}}
  %0 = tensorrt.softmax {axis = 1 : i64} %arg0 : tensor<10x10xi8>
  return %0 : tensor<10x10xi8>
}

// -----

func.func @trt_onehot(%indices: tensor<3xi32>, %values: tensor<2xi32>, %depth: tensor<i32>) -> tensor<3x?xi32> {
  // expected-error @below {{expected axis to be in the range [-rank(indices)-1, rank(indices)]}}
  // expected-error @below {{'tensorrt.one_hot' op failed to infer returned types}}
  %0 = tensorrt.one_hot {
    axis = 2 : si64
  } ins(%indices, %values, %depth : tensor<3xi32>, tensor<2xi32>, tensor<i32>) -> tensor<3x?xi32>

  return %0 : tensor<3x?xi32>
}

// -----

func.func @trt_onehot(%indices: tensor<3xi32>, %values: tensor<2x2xi32>, %depth: tensor<i32>) -> tensor<3x?xi32> {
  // expected-error @below {{expected values to be of rank 1}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.one_hot {
    axis = -1 : si64
  } ins(%indices, %values, %depth : tensor<3xi32>, tensor<2x2xi32>, tensor<i32>) -> tensor<3x?xi32>

  return %0 : tensor<3x?xi32>
}

// -----

func.func @trt_onehot(%indices: tensor<3xi32>, %values: tensor<1xi32>, %depth: tensor<i32>) -> tensor<3x?xi32> {
  // expected-error @below {{expected values to have two elements}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.one_hot {
    axis = -1 : si64
  } ins(%indices, %values, %depth : tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3x?xi32>

  return %0 : tensor<3x?xi32>
}

// -----

func.func @trt_onehot(%indices: tensor<3xi32>, %values: tensor<2xi32>, %depth: tensor<1xi32>) -> tensor<3x?xi32> {
  // expected-error @below {{expected depth to be of rank 0}}
  %0 = tensorrt.one_hot {
    axis = -1 : si64
  } ins(%indices, %values, %depth : tensor<3xi32>, tensor<2xi32>, tensor<1xi32>) -> tensor<3x?xi32>

  return %0 : tensor<3x?xi32>
}

// -----

func.func @trt_onehot(%indices: tensor<3xi32>, %values: tensor<2xi32>, %depth: tensor<i32>) -> tensor<3x?xf32> {
  // expected-error @below {{'tensorrt.one_hot' op failed to verify that all of {values, result} have same element type}}
  %0 = tensorrt.one_hot {
    axis = -1 : si64
  } ins(%indices, %values, %depth : tensor<3xi32>, tensor<2xi32>, tensor<i32>) -> tensor<3x?xf32>

  return %0 : tensor<3x?xf32>
}

// -----

func.func @trt_ragged_softmax(%input: tensor<1x3x5xf32>, %bounds: tensor<1x3x1xf32>) -> tensor<1x3x5xf32> {
  // expected-error @below {{'tensorrt.ragged_softmax' op operand #1 must be 0D/1D/2D/3D/4D/5D/6D/7D/8D tensor of 32-bit signless integer values, but got 'tensor<1x3x1xf32>'}}
  %0 = tensorrt.ragged_softmax ins(%input, %bounds : tensor<1x3x5xf32>, tensor<1x3x1xf32>) -> tensor<1x3x5xf32>
  return %0 : tensor<1x3x5xf32>
}

// -----

func.func @trt_ragged_softmax(%input: tensor<1x3x3x5xf32>, %bounds: tensor<1x3x3x1xi32>) -> tensor<1x3x3x5xf32> {
  // expected-error @below {{expected input and bounds to be of rank 3}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.ragged_softmax ins(%input, %bounds : tensor<1x3x3x5xf32>, tensor<1x3x3x1xi32>) -> tensor<1x3x3x5xf32>
  return %0 : tensor<1x3x3x5xf32>
}

// -----

func.func @trt_padding(%arg0 : tensor<10x10xf32>) -> tensor<8x8xf32> {
  // expected-error @below {{input rank should be greater or equal than 4}}
  %0 = tensorrt.padding {
    prePadding = array<i64: -1, -1>,
    postPadding = array<i64: -1, -1>
  } ins(%arg0 : tensor<10x10xf32>) -> tensor<8x8xf32>

  return %0 : tensor<8x8xf32>
}

// -----

func.func @trt_padding(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x1x8x8xf32> {
  // expected-error @below {{shape error on dimension 2 : pre padding amount: -8; post padding amount: -4; origin dimension: 10}}
  %0 = tensorrt.padding {
    prePadding = array<i64: -8, 0>,
    postPadding = array<i64: -4, 0>
  } ins(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x1x0x8xf32>

  return %0 : tensor<1x1x0x8xf32>
}

// -----

func.func @trt_transpose(%arg0 : tensor<10x20x30xf32>) -> tensor<20x10x30xf32> {
  // expected-error @below {{'tensorrt.transpose' op expected "permutation" to be a permutation of rank 3}}
  %0 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2)->(d1, d0)>}
        %arg0 : tensor<10x20x30xf32> to tensor<20x10x30xf32>
  return %0 : tensor<20x10x30xf32>
}

// -----

func.func @trt_transpose(%arg0 : tensor<10x20x30xf32>) -> tensor<30x20x10xf32> {
  // expected-error @below {{'tensorrt.transpose' op inferred type(s) 'tensor<20x10x30xf32>' are incompatible with return type(s) of operation 'tensor<30x20x10xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2)->(d1, d0, d2)>}
        %arg0 : tensor<10x20x30xf32> to tensor<30x20x10xf32>
  return %0 : tensor<30x20x10xf32>
}

// -----

func.func @trt_matmul_precision_mismatch(%arg0: tensor<10x20xf32>, %arg1: tensor<20x10xf16>) -> tensor<10x10xf32> {
  // expected-error @below {{'tensorrt.matrix_multiply' op requires the same element type for all operands and results}}
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<10x20xf32>, tensor<20x10xf16>) -> tensor<10x10xf32>

  return %0 : tensor<10x10xf32>
}

// -----

func.func @trt_matmul_vector_pass(%arg0: tensor<20xf32>, %arg1: tensor<20xf32>) -> tensor<f32> {
  // expected-error @below {{'tensorrt.matrix_multiply' op Input 0 has rank one. Expected TRT MatOp kVCETOR }}
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<20xf32>, tensor<20xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @trt_matmul_unsupported_dtype(%arg0: tensor<5x10xi32>, %arg1: tensor<10x5xi32>) -> tensor<5x5xi32> {
  // expected-error @below {{'tensorrt.matrix_multiply' op operand #0 must be 0D/1D/2D/3D/4D/5D/6D/7D/8D tensor of 32-bit float or 16-bit float or bfloat16 type or allowed TensorRT tensor i8 element types or 1-bit signless integer values, but got 'tensor<5x10xi32>'}}
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<5x10xi32>, tensor<10x5xi32>) -> tensor<5x5xi32>

  return %0 : tensor<5x5xi32>
}

// -----

func.func @trt_matmul_collection_dims_mismatch(%arg0: tensor<3x5x10x20xf32>, %arg1: tensor<3x7x20x10xf32>) -> tensor<3x7x20x10xf32> {
  // expected-error @below {{'tensorrt.matrix_multiply' op collection (batch) dimensions of "input0" and "input1" are not broadcastable}}
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<3x5x10x20xf32>, tensor<3x7x20x10xf32>) -> tensor<3x7x20x10xf32>

  return %0 : tensor<3x7x20x10xf32>
}

// -----
func.func @trt_matmul_shape_mismatch(%arg0: tensor<20x30xf32>, %arg1: tensor<10x30xf32>) -> tensor<20x10xf32> {
  // expected-error @below {{'tensorrt.matrix_multiply' op operand shapes do not have consistent sizes for the contraction dimension}}
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<20x30xf32>, tensor<10x30xf32>) -> tensor<20x10xf32>

  return %0 : tensor<20x10xf32>
}

// -----

func.func @trt_matrix_multiply(%arg0: tensor<64xf32>, %arg1: tensor<128x64xf32>)
            -> tensor<128xf32> {
  // expected-error @below {{'tensorrt.matrix_multiply' op operand shapes do not have consistent sizes for the contraction dimension}}
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kVECTOR>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<64xf32>, tensor<128x64xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// -----

func.func @trt_yield(%arg0: tensor<i1>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10xf32>) -> tensor<10x20xf32> {
  // expected-error @below {{'tensorrt.yield' op expects parent op to be one of 'tensorrt.if, tensorrt.for, tensorrt.while, tensorrt.opaque_plugin'}}
  tensorrt.yield %arg0 : tensor<i1>
  return %arg1 : tensor<10x10xf32>
}

// -----

func.func @trt_gather(%arg0: tensor<10x20x30xf32>, %arg1: tensor<5xi32>) -> tensor<10x5x30xf32> {
  // expected-error @below {{'tensorrt.gather' op expected "axis" to must be in the range [0, 3)}}
  %0 = tensorrt.gather {
    axis = 3 : i64,
    numBroadcastDims = 0 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<5xi32>) -> tensor<10x5x30xf32>
  return %0 : tensor<10x5x30xf32>
}

// -----

func.func @trt_gather(%arg0: tensor<10x20x30xf32>, %arg1: tensor<5xi32>) -> tensor<10x5x30xf32> {
  // expected-error @below {{'tensorrt.gather' op when numBroadcastDims = 1, first dimension of data and indices must be broadcastable}}
  %0 = tensorrt.gather {
    axis = 1 : i64,
    numBroadcastDims = 1 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<5xi32>) -> tensor<10x5x30xf32>
  return %0 : tensor<10x5x30xf32>
}

// -----

func.func @trt_gather(%arg0: tensor<10x20x30xf32>, %arg1: tensor<5xi32>) -> tensor<10x5x30xf32> {
  // expected-error @below {{'tensorrt.gather' op inferred type(s) 'tensor<5x20x30xf32>' are incompatible with return type(s) of operation 'tensor<10x5x30xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.gather {
    axis = 0 : i64,
    numBroadcastDims = 0 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<5xi32>) -> tensor<10x5x30xf32>
  return %0 : tensor<10x5x30xf32>
}

// -----

func.func @trt_gather(%arg0: tensor<10x20x30xf32>, %arg1: tensor<5x10xi32>) -> tensor<10x5x30x30xf32> {
  // expected-error @below {{'tensorrt.gather' op inferred type(s) 'tensor<10x5x10x30xf32>' are incompatible with return type(s) of operation 'tensor<10x5x30x30xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.gather {
    axis = 1 : i64,
    numBroadcastDims = 0 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<5x10xi32>) -> tensor<10x5x30x30xf32>
  return %0 : tensor<10x5x30x30xf32>
}

// -----

func.func @trt_gather_elements(%arg0: tensor<10x20x30xf32>, %arg1: tensor<10x20x10xi32>) -> tensor<10x20x5xf32> {
  // expected-error @below {{'tensorrt.gather_elements' op inferred type(s) 'tensor<10x20x10xf32>' are incompatible with return type(s) of operation 'tensor<10x20x5xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.gather_elements {
    axis = 2 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<10x20x10xi32>) -> tensor<10x20x5xf32>
  return %0 : tensor<10x20x5xf32>
}

// -----

func.func @trt_nonzero(%arg0: tensor<?x?x?x?xf32>) -> tensor<3x?xi32> {
  // expected-error @below {{'tensorrt.non_zero' op inferred type(s) 'tensor<4x?xi32>' are incompatible with return type(s) of operation 'tensor<3x?xi32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.non_zero %arg0 : tensor<?x?x?x?xf32> -> tensor<3x?xi32>
  return %0 : tensor<3x?xi32>
}

// -----

func.func @trt_shuffle_wrong_transpose(%arg0: tensor<30x20x10x1xf32>) -> tensor<6000x1xf32> {
  // expected-error @below {{second transpose array is not a permutation of reshaped rank 2}}
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: -1, 0>,
    second_transpose = array<i64: 0, 1, 2>
  } ins(%arg0 : tensor<30x20x10x1xf32>) -> tensor<6000x1xf32>

  return %0 : tensor<6000x1xf32>
}

// -----

// The below shuffle tests verify the TRT inference procedure for identifying which
// dims correspond to '0' in the reshape spec when "zero_is_placeholder" is true.

func.func @trt_shuffle_infer(%arg0: tensor<30x20x10x1xf32>) -> tensor<600x5x2xf32> {
  // This is invalid because the "0, 0" in reshape must correspond to
  // "20, 10". The 600 is only valid if "0, 0"  correspond to "10, 1".

  // expected-error @below {{'tensorrt.shuffle' op result 0 has type tensor<600x5x2xf32> but inferred tensor of shape <600x20x10>}}
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: 600, 0, 0>,
    second_transpose = array<i64: 0, 1, 2>
  } ins(%arg0 : tensor<30x20x10x1xf32>) -> tensor<600x5x2xf32>
  return %0 : tensor<600x5x2xf32>
}

// -----

func.func @trt_shuffle_infer2(%arg0: tensor<30x20x10x1xf32>) -> tensor<600x10xf32> {
  // expected-error @below {{'tensorrt.shuffle' op result 0 has type tensor<600x10xf32> but inferred tensor of shape <300x20>}}
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: -1, 0>,
    second_transpose = array<i64: 0, 1>
  } ins(%arg0 : tensor<30x20x10x1xf32>) -> tensor<600x10xf32>
  return %0 : tensor<600x10xf32>
}

// -----

func.func @trt_shuffle_infer3(%arg0: tensor<1x2x3x4x5x6xf32>) -> tensor<2x3x4x30xf32> {
  // expected-error @below {{'tensorrt.shuffle' op result 0 has type tensor<2x3x4x30xf32> but inferred tensor of shape <1x8x3x30>}}
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3, 4, 5>,
    reshape = array<i64: 0, -1, 0, 30>,
    second_transpose = array<i64: 0, 1, 2, 3>
  } ins(%arg0 : tensor<1x2x3x4x5x6xf32>) -> tensor<2x3x4x30xf32>
  return %0 : tensor<2x3x4x30xf32>
}


// -----

func.func @trt_shuffle_infer4(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x1x2x3x4xf32> {
  // expected-error @below {{invalid reshape specification - 0 placeholder maps to out-of-bounds index of input shape}}
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: 0, -1, 0, 0, 0>,
    second_transpose = array<i64: 0, 1, 2, 3, 4>
  } ins(%arg0 : tensor<1x2x3x4xf32>) -> tensor<1x1x2x3x4xf32>
  return %0 : tensor<1x1x2x3x4xf32>
}

// -----

func.func @trt_if(%arg0: tensor<i1>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error @below {{'tensorrt.if' op region #0 should have no arguments}}
  %result = tensorrt.if (%arg0: tensor<i1>) -> tensor<10xf32> {
    ^bb0(%x: tensor<f32>):
      tensorrt.yield %arg1: tensor<10xf32>
    } else {
    ^bb0(%u: tensor<f32>):
      tensorrt.yield %arg1: tensor<10xf32>
    }
  return %result: tensor<10xf32>
}

// -----

func.func @trt_shuffle_wrong_reshape(%arg0: tensor<1x30x20x10xf32>) -> tensor<30x20x10xf32> {
  // expected-error @below {{invalid reshape specification - at most one '-1' is allowed}}
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: -1, -1, 10>,
    second_transpose = array<i64: 0, 1, 2>
  } ins(%arg0 : tensor<1x30x20x10xf32>) -> tensor<30x20x10xf32>

  return %0 : tensor<30x20x10xf32>
}

// -----

func.func @trt_if(%arg0: tensor<i1>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error @below {{number of output tensors in true and false regions must be same}}
  // expected-error @below {{'tensorrt.if' op failed to infer returned types}}
  %result = tensorrt.if (%arg0: tensor<i1>) -> tensor<10xf32> {
      tensorrt.yield %arg1: tensor<10xf32>
    } else {
      tensorrt.yield
    }
  return %result: tensor<10xf32>
}

// -----

func.func @trt_if(%arg0: tensor<i1>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error @below {{true and false regions must yield equivalent types}}
  // expected-error @below {{'tensorrt.if' op failed to infer returned types}}
  %result = tensorrt.if (%arg0: tensor<i1>) -> tensor<10xf32> {
      tensorrt.yield %arg1: tensor<10xf32>
    } else {
      tensorrt.yield %arg0: tensor<i1>
    }
  return %result: tensor<10xf32>
}

// -----

func.func @trt_if(%arg0: tensor<i1>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %cond = arith.constant dense<1> : tensor<i1>
  // expected-error @below {{true and false regions must yield equivalent types}}
  // expected-error @below {{'tensorrt.if' op failed to infer returned types}}
  %result = tensorrt.if (%cond: tensor<i1>) -> tensor<10xf32> {
      tensorrt.yield %arg1: tensor<10xf32>
    } else {
      tensorrt.yield %arg0: tensor<i1>
    }
  return %result: tensor<10xf32>
}

// -----

func.func @trt_shuffle_num_elements_mismatch(%arg0: tensor<1x1x768xf32>) -> tensor<f32> {
    // expected-error @below {{'tensorrt.shuffle' op result 0 has type tensor<f32> but inferred tensor of shape <768>}}
    %1 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 1, 2>,
        reshape = array<i64: 768>,
        second_transpose = array<i64: 0>
    } ins(%arg0 : tensor<1x1x768xf32>) -> tensor<f32>
    return %1 : tensor<f32>
}

// -----

func.func @trt_shuffle_not_permutation(%arg0: tensor<224x224x3xf32>) -> tensor<3x224x224xf32> {
  // expected-error @below {{first transpose array is not a permutation}}
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 1>,
    reshape = array<i64: 3, 224, 224>,
    second_transpose = array<i64: 0, 1, 2>,
    zero_is_placeholder = true
  } ins(%arg0 : tensor<224x224x3xf32>) -> tensor<3x224x224xf32>

  return %0 : tensor<3x224x224xf32>
}

// -----

// expected-error @below {{shape min/opt/max arrays should have equal size}}
func.func @trt_shape_bounds(%arg1: tensor<10xf32> {tensorrt.profile = #tensorrt.shape_profile<min=[],opt =[10],max=[10]>})
                              -> tensor<10xf32> {
  return %arg1: tensor<10xf32>
}

// -----

// expected-error @below {{profile dimension 0 min=11 should be less than or equal to opt=9}}
func.func @trt_shape_bounds(%arg1: tensor<10xf32> {tensorrt.profile = #tensorrt.shape_profile<min=[11],opt=[9],max=[10]>})
                              -> tensor<10xf32> {
  return %arg1: tensor<10xf32>
}

// -----


// expected-error @below {{profile dimension 0 opt=11 should be less than or equal to max=10}}
func.func @trt_shape_bounds(%arg1: tensor<10xf32> {tensorrt.profile = #tensorrt.shape_profile<min=[5],opt=[11],max=[10]>})
                              -> tensor<10xf32> {
  return %arg1: tensor<10xf32>
}

// -----

func.func @trt_activation(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // expected-error @below {{'tensorrt.activation' op expected no `alpha` attribute and expected no `beta` attribute for `activationType=kRELU}}
  %0 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>,
    alpha = 1. : f32,
    beta = 1. : f32
  } %arg0 : tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @trt_shape_op(%arg0: tensor<?x?x?x?xf32>) -> tensor<2xi32> {
  // expected-error @below {{'tensorrt.shape' op inferred type(s) 'tensor<4xi32>' are incompatible with return type(s) of operation 'tensor<2xi32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.shape %arg0 : tensor<?x?x?x?xf32> -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

func.func @trt_shape_op(%arg0: tensor<f32>) -> tensor<1xi32> {
  // expected-error @below {{'tensorrt.shape' op inferred type(s) 'tensor<0xi32>' are incompatible with return type(s) of operation 'tensor<1xi32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.shape %arg0 : tensor<f32> -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// -----

func.func @trt_parametric_relu_op(%arg0: tensor<2x10xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x10xf32> {
  // expected-error @below {{'tensorrt.parametric_relu' op expected dimensions of slope tensor should be either 1 or equal to correspond dimension of the input tensor}}
  %0 = tensorrt.parametric_relu ins(%arg0, %arg1 : tensor<2x10xf32>, tensor<2x2xf32>) -> tensor<2x10xf32>
  return %0 : tensor<2x10xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x128x128xf32> {
  // expected-error @below {{'tensorrt.convolution' op stride/pre_padding/post_padding should have size equal to the number of spatial dimensions}}
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1, 1, 1>,
    pre_padding = array<i64: 1, 1, 1, 1>,
    post_padding = array<i64: 1, 1, 1, 1>
  } in(%arg0: tensor<?x32x128x128xf32>) kernel(%arg1:tensor<64x32x3x3xf32>) bias(%arg2:tensor<64xf32>) -> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>, %arg2: tensor<1x64x1x1xf32>) -> tensor<?x64x128x128xf32> {
  // expected-error @below {{'tensorrt.deconvolution' op bias type should be a rank-1 tensor type with size equal to the number of channels (dim 1) of the result tensor type}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0:tensor<?x32x128x128xf32>) kernelWeights(%arg1:tensor<32x64x3x3xf32>) biasWeights(%arg2:tensor<1x64x1x1xf32>) -> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x2x1x1xf32>, %arg2: tensor<2xf32>) -> tensor<?x2x130x130xf32> {
  // expected-error @below {{op failed to infer returned types}}
  // expected-error @below {{'tensorrt.deconvolution' op inferred type(s) 'tensor<?x64x126x126xf32>' are incompatible with return type(s) of operation 'tensor<?x2x130x130xf32>'}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    num_groups = 32 : ui32
  } in(%arg0 : tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x2x1x1xf32>) biasWeights(%arg2 : tensor<2xf32>) -> tensor<?x2x130x130xf32>
  return %0 : tensor<?x2x130x130xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>) -> tensor<?x64x128x128xf32> {
  // expected-error @below {{'tensorrt.deconvolution' op kernelWeights operand or kernelWeightsStatic attribute must be specified}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0:tensor<?x32x128x128xf32>)-> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>) -> tensor<?x64x128x128xf32> {
  // expected-error @below {{'tensorrt.deconvolution' op only one of kernelWeights operand or kernelWeightsStatic attribute can be specified}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    kernelWeightsStatic = dense<0.1>:tensor<64x32x3x3xf32>
  } in(%arg0:tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>)-> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}

// -----

func.func @trt_unary(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  // expected-error @below {{'tensorrt.unary' op expected element type to be one of the following: I1}}
  %0 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNOT>} %arg0 : tensor<10xi32>
  return %0 : tensor<10xi32>
}

// -----

func.func @trt_unary_sqrt(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @below {{'tensorrt.unary' op TensorRT Unary ops need at least 1D input}}
  %0 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @trt_unary_sqrt(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  // expected-error @below {{'tensorrt.unary' op expected element type to be one of the following: F16, F32}}
  %0 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %arg0 : tensor<10xi32>
  return %0 : tensor<10xi32>
}

// -----

func.func @resize_nearest_static(%arg0: tensor<2x2x2x2x2xf32>) -> tensor<2x4x4x4x4xf32> {
  // expected-error @below {{'tensorrt.resize_nearest' op only supports resizing on the innermost min(3, rank(input)) dimensions}}
  %0 = tensorrt.resize_nearest {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    nearestRounding = #tensorrt.resize_round_mode<kFLOOR>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>} %arg0 : (tensor<2x2x2x2x2xf32>) -> tensor<2x4x4x4x4xf32>
  return %0 : tensor<2x4x4x4x4xf32>
}

// -----

func.func @resize_nearest_static(%arg0: tensor<2x2x2x2xf32>) -> tensor<2x4x4x4xf32> {
  // expected-error @below {{scales parameter must have same number of dimensions as input/output}}
  %0 = tensorrt.resize_nearest {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    nearestRounding = #tensorrt.resize_round_mode<kFLOOR>,
    scales = array<f32: 2.0, 3.0>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>} %arg0 : (tensor<2x2x2x2xf32>) -> tensor<2x4x4x4xf32>
  return %0 : tensor<2x4x4x4xf32>
}

// -----

func.func @resize_nearest_dynamic(%arg0: tensor<2x2x2x?x?xf32>) -> tensor<2x2x2x?x?xf32> {
  // expected-error @below {{'tensorrt.resize_nearest' op input innermost min(3, rank(input)) dimension that resize on cannot be dynamic when output_shape parameter is NOT specified and it cannot be inferred statically}}
  %0 = tensorrt.resize_nearest {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    nearestRounding = #tensorrt.resize_round_mode<kFLOOR>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>} %arg0 : (tensor<2x2x2x?x?xf32>) -> tensor<2x2x2x?x?xf32>
  return %0 : tensor<2x2x2x?x?xf32>
}

// -----

func.func @resize_nearest_dynamic(%arg0: tensor<2x2x2x?x?xf32>) -> tensor<2x2x2x?x?xf32> {
  // expected-error @below {{'tensorrt.resize_nearest' op result 0 has type tensor<2x2x2x?x?xf32> but inferred tensor of shape <2x2x4x?x?>}}
  %0 = tensorrt.resize_nearest {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    nearestRounding = #tensorrt.resize_round_mode<kFLOOR>,
    scales = array<f32: 1.0, 1.0, 2.0, 2.0, 2.0>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>} %arg0 : (tensor<2x2x2x?x?xf32>) -> tensor<2x2x2x?x?xf32>
  return %0 : tensor<2x2x2x?x?xf32>
}

// -----

func.func @resize_nearest_dynamic(%arg0: tensor<2x2x2x?x?xf32>) -> tensor<2x4x4x?x?xf32> {
  // expected-error @below {{all scale values except innermost min(3, rank(input)) must be 1}}
  %0 = tensorrt.resize_nearest {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    nearestRounding = #tensorrt.resize_round_mode<kFLOOR>,
    scales = array<f32: 1.0, 2.0, 2.0, 2.0, 2.0>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>} %arg0 : (tensor<2x2x2x?x?xf32>) -> tensor<2x4x4x?x?xf32>
  return %0 : tensor<2x4x4x?x?xf32>
}

// -----

func.func @resize_nearest_dynamic(%arg0: tensor<2x3xf32>) -> tensor<4x6xf16> {
  // expected-error @below {{'tensorrt.resize_nearest' op result 0 has element type 'f16' but inferred element type'f32'}}
  %0 = tensorrt.resize_nearest {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    nearestRounding = #tensorrt.resize_round_mode<kFLOOR>,
    scales = array<f32: 2.0, 2.0>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>} %arg0 : (tensor<2x3xf32>) -> tensor<4x6xf16>
  return %0 : tensor<4x6xf16>
}
// -----

func.func @resize_linear_static(%arg0: tensor<2x2x2x2x2xf32>) -> tensor<2x4x4x4x4xf32> {
  // expected-error @below {{'tensorrt.resize_linear' op only supports resizing on the innermost min(3, rank(input)) dimensions}}
  %0 = tensorrt.resize_linear {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>} %arg0 : (tensor<2x2x2x2x2xf32>) -> tensor<2x4x4x4x4xf32>
  return %0 : tensor<2x4x4x4x4xf32>
}

// -----

func.func @resize_linear_static(%arg0: tensor<2x2x2x2x2xf32>) -> tensor<2x2x4x4x4xf32> {
  // expected-error @below {{scales parameter must have same number of dimensions as input/output}}
  %0 = tensorrt.resize_linear {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>,
    scales = array<f32: 2.0, 3.0>} %arg0 : (tensor<2x2x2x2x2xf32>) -> tensor<2x2x4x4x4xf32>
  return %0 : tensor<2x2x4x4x4xf32>
}

// -----

func.func @resize_linear_dynamic(%arg0: tensor<2x2x2x?x?xf32>) -> tensor<2x2x2x?x?xf32> {
  // expected-error @below {{'tensorrt.resize_linear' op input innermost min(3, rank(input)) dimension that resize on cannot be dynamic when output_shape parameter is NOT specified and it cannot be inferred statically}}
  %0 = tensorrt.resize_linear {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>} %arg0 : (tensor<2x2x2x?x?xf32>) -> tensor<2x2x2x?x?xf32>
  return %0 : tensor<2x2x2x?x?xf32>
}

// -----

func.func @resize_linear_dynamic(%arg0: tensor<2x2x2x?x?xf32>) -> tensor<2x2x2x?x?xf32> {
  // expected-error @below {{'tensorrt.resize_linear' op result 0 has type tensor<2x2x2x?x?xf32> but inferred tensor of shape <2x2x4x?x?>}}
  %0 = tensorrt.resize_linear {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>,
    scales = array<f32: 1.0, 1.0, 2.0, 3.0, 3.0>} %arg0 : (tensor<2x2x2x?x?xf32>) -> tensor<2x2x2x?x?xf32>
  return %0 : tensor<2x2x2x?x?xf32>
}

// -----

func.func @resize_linear_dynamic(%arg0: tensor<2x2x2x3x3xf32>) -> tensor<2x4x4x9x9xf32> {
  // expected-error @below {{all scale values except innermost min(3, rank(input)) must be 1}}
  %0 = tensorrt.resize_linear {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>,
    scales = array<f32: 1.0, 2.0, 2.0, 3.0, 3.0>} %arg0 : (tensor<2x2x2x3x3xf32>) -> tensor<2x4x4x9x9xf32>
  return %0 : tensor<2x4x4x9x9xf32>
}

// -----

func.func @resize_linear_dynamic(%arg0: tensor<2x?xf32>) -> tensor<10x?xf32> {
  // expected-error @below {{'tensorrt.resize_linear' op result 0 has type tensor<10x?xf32> but inferred tensor of shape <8x?>}}
  %0 = tensorrt.resize_linear {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>,
    scales = array<f32: 4.0, 5.0>} %arg0 : (tensor<2x?xf32>) -> tensor<10x?xf32>
  return %0 : tensor<10x?xf32>
}

// -----

func.func @resize_cubic_rank1(%arg0: tensor<2xf32>) -> tensor<4xf32> {
  // expected-error @below {{does not support resizing on a tensor that has rank < 2}}
  %0 = tensorrt.resize_cubic {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>,
    cubicCoeff = -0.75 : f32,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>} %arg0 : (tensor<2xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @resize_cubic_static(%arg0: tensor<2x2x2xf32>) -> tensor<4x4x4xf32> {
  // expected-error @below {{'tensorrt.resize_cubic' op only supports resizing on the innermost 2 dimensions}}
  %0 = tensorrt.resize_cubic {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>,
    cubicCoeff = -0.5 : f32,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>} %arg0 : (tensor<2x2x2xf32>) -> tensor<4x4x4xf32>
  return %0 : tensor<4x4x4xf32>
}

// -----

func.func @resize_cubic_static(%arg0: tensor<2x2x2xf32>) -> tensor<4x4x4xf32> {
  // expected-error @below {{scales parameter must have same number of dimensions as input/output}}
  %0 = tensorrt.resize_cubic {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>,
    scales = array<f32: 1.0, 1.0, 2.0, 3.0, 3.0>,
    cubicCoeff = -0.5 : f32} %arg0 : (tensor<2x2x2xf32>) -> tensor<4x4x4xf32>
  return %0 : tensor<4x4x4xf32>
}

// -----

func.func @resize_cubic_dynamic(%arg0: tensor<2x?x?xf32>) -> tensor<2x?x?xf32> {
  // expected-error @below {{'tensorrt.resize_cubic' op input innermost 2 dimensions that resize on cannot be dynamic when output_shape parameter is NOT specified and it cannot be inferred statically}}
  %0 = tensorrt.resize_cubic {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>,
    cubicCoeff = -1.0 : f32} %arg0 : (tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  return %0 : tensor<2x?x?xf32>
}

// -----

func.func @resize_cubic_dynamic(%arg0: tensor<2x2x?xf32>) -> tensor<2x4x?xf32> {
  // expected-error @below {{'tensorrt.resize_cubic' op result 0 has type tensor<2x4x?xf32> but inferred tensor of shape <2x6x?>}}
  %0 = tensorrt.resize_cubic {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>,
    scales = array<f32: 1.0, 3.0, 3.0>,
    cubicCoeff = -1.0 : f32} %arg0 : (tensor<2x2x?xf32>) -> tensor<2x4x?xf32>
  return %0 : tensor<2x4x?xf32>
}

// -----

func.func @resize_cubic_dynamic(%arg0: tensor<2x2x3x?xf32>) -> tensor<2x4x6x?xf32> {
  // expected-error @below {{all scale values except 2 innermost must be 1}}
  %0 = tensorrt.resize_cubic {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>,
    scales = array<f32: 1.0, 2.0, 2.0, 2.0>,
    cubicCoeff = -1.0 : f32} %arg0 : (tensor<2x2x3x?xf32>) -> tensor<2x4x6x?xf32>
  return %0 : tensor<2x4x6x?xf32>
}

// -----

func.func @trt_quantize_per_axis(%arg0: tensor<10x10xf32>, %arg1: tensor<10xf32>) -> tensor<10x10xi8> {
  // expected-error @below {{'tensorrt.quantize' op expected axis to be non-negative and less than 2}}
  %result = tensorrt.quantize {
    axis = 2 : i32
  } in(%arg0 : tensor<10x10xf32>) scale(%arg1 : tensor<10xf32>) -> tensor<10x10xi8>
  return %result : tensor<10x10xi8>
}

// -----

func.func @trt_quantize_per_axis(%arg0: tensor<10x10xf32>, %arg1: tensor<11xf32>) -> tensor<10x10xi8> {
  // expected-error @below {{'tensorrt.quantize' op expected the scales size to match the quantization axis of input tensor}}
  %result = tensorrt.quantize {
    axis = 1 : i32
  } in(%arg0 : tensor<10x10xf32>) scale(%arg1 : tensor<11xf32>) -> tensor<10x10xi8>
  return %result : tensor<10x10xi8>
}

// -----

func.func @trt_quantize(%arg0: tensor<10x10xf32>, %arg1: tensor<2xf32>) -> tensor<10x10xi8> {
  // expected-error @below {{'tensorrt.quantize' op if no axis is provided and input is not INT4, quantization is per-tensor. In this case, `scale` must be a scalar i.e. 0 dim tensor.}}
  %result = tensorrt.quantize in(%arg0 : tensor<10x10xf32>) scale(%arg1 : tensor<2xf32>) -> tensor<10x10xi8>
  return %result : tensor<10x10xi8>
}

// -----

func.func @trt_quantize_i4(%arg0: tensor<10x10xf32>, %arg1: tensor<2x10xf32>) -> tensor<10x10xi8> {
  // expected-error @below {{'tensorrt.quantize' op 2D scale is supported only for quantizing INT4 output}}
  %result = tensorrt.quantize in(%arg0 : tensor<10x10xf32>) scale(%arg1 : tensor<2x10xf32>) -> tensor<10x10xi8>
  return %result : tensor<10x10xi8>
}

// -----

func.func @trt_dequantize_per_axis(%arg0: tensor<10x10xi8>, %arg1: tensor<10xf32>) -> tensor<10x10xf32> {
  // expected-error @below {{'tensorrt.dequantize' op expected axis to be non-negative and less than 2}}
  %result = tensorrt.dequantize {
    axis = 2 : i32
  } in(%arg0 : tensor<10x10xi8>) scale(%arg1 : tensor<10xf32>) -> tensor<10x10xf32>
  return %result : tensor<10x10xf32>
}

// -----

func.func @trt_dequantize_per_axis(%arg0: tensor<10x10xi8>, %arg1: tensor<11xf32>) -> tensor<10x10xf32> {
  // expected-error @below {{'tensorrt.dequantize' op expected the scales size to match the dequantization axis of input tensor}}
  %result = tensorrt.dequantize {
    axis = 1 : i32
  } in(%arg0 : tensor<10x10xi8>) scale(%arg1 : tensor<11xf32>) -> tensor<10x10xf32>
  return %result : tensor<10x10xf32>
}

// -----

func.func @trt_dequantize(%arg0: tensor<10x10xi8>, %arg1: tensor<2xf32>) -> tensor<10x10xf32> {
  // expected-error @below {{'tensorrt.dequantize' op if no axis is provided and input is not INT4, dequantization is per-tensor. In this case, `scale` must be a scalar i.e. 0 dim tensor.}}
  %result = tensorrt.dequantize in(%arg0 : tensor<10x10xi8>) scale(%arg1 : tensor<2xf32>) -> tensor<10x10xf32>
  return %result : tensor<10x10xf32>
}

// -----


func.func @trt_matrix_multiply_trans_vec(%arg0: tensor<1x1x1x50x10xf32>, %arg1: tensor<1x4x240x50xf32>) -> tensor<1x1x240x10xf32> {
  // expected-error @below {{'tensorrt.matrix_multiply' op inferred type(s) 'tensor<1x4x240x10xf32>' are incompatible with return type(s) of operation 'tensor<1x1x240x10xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kTRANSPOSE>, op1 = #tensorrt.matrix_operation<kVECTOR>}
    ins(%arg0, %arg1 : tensor<1x1x1x50x10xf32>, tensor<1x4x240x50xf32>) -> tensor<1x1x240x10xf32>
  return %0 : tensor<1x1x240x10xf32>
}

// -----

func.func @trt_gather(%arg0: tensor<1x1x5x4xf32>, %arg1: tensor<4x3xi32>) -> tensor<2x1x3x4xf32> {
  // expected-error @below {{'tensorrt.gather' op inferred type(s) 'tensor<4x1x3x4xf32>' are incompatible with return type(s) of operation 'tensor<2x1x3x4xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.gather {
    axis = 2 : i64,
    numBroadcastDims = 1 : i64
  } ins(%arg0, %arg1 : tensor<1x1x5x4xf32>, tensor<4x3xi32>) -> tensor<2x1x3x4xf32>
  return %0 : tensor<2x1x3x4xf32>
}

// -----

func.func @trt_for_loop(%lb: tensor<i32>, %ub: tensor<i32>, %step: tensor<i32>, %arg0: tensor<10xf16>, %arg1: tensor<10xf16>)
    -> (tensor<10xf16>, tensor<10xf16>) {
  // expected-error @below {{'tensorrt.for' op  along control flow edge from Region #0 to Region #0: source type #1 'tensor<9xf16>' should match input type #1 'tensor<10xf16>'}}
  %0, %1 = tensorrt.for %i = %lb to %ub step %step init(%iter0 = %arg0, %iter1 = %arg1) -> (tensor<10xf16>, tensor<10xf16>) {
    %cst = tensorrt.constant dense<0.0> : tensor<9xf16>
    tensorrt.yield %iter0, %cst : tensor<10xf16>, tensor<9xf16>
  }
  return %0, %1 : tensor<10xf16>, tensor<10xf16>
}

// -----

// expected-note @below {{prior use here}}
func.func @trt_for_loop(%lb: tensor<i32>, %ub: tensor<i32>, %step: tensor<i32>, %arg0: tensor<10xf16>, %arg1: tensor<10xf16>)
    -> (tensor<10xf16>, tensor<9xf16>) {
  // expected-error @below {{use of value '%arg1' expects different type than prior uses: 'tensor<9xf16>' vs 'tensor<10xf16>'}}
  %0, %1 = tensorrt.for %i = %lb to %ub step %step init(%iter0 = %arg0, %iter1 = %arg1) -> (tensor<10xf16>, tensor<9xf16>) {
    tensorrt.yield %iter0, %iter1 : tensor<10xf16>, tensor<10xf16>
  }
  return %0, %1 : tensor<10xf16>, tensor<9xf16>
}

// -----

func.func @argmax_infer_test(%arg0: tensor<8x10xf32>) -> (tensor<9x1xf32>, tensor<9x1xi32>) {
  // expected-error @below {{'tensorrt.argmax' op inferred type(s) 'tensor<8x1xf32>', 'tensor<8x1xi32>' are incompatible with return type(s) of operation 'tensor<9x1xf32>', 'tensor<9x1xi32>'}}
  // expected-error @below {{'tensorrt.argmax' op failed to infer returned types}}
  %0, %1 = tensorrt.argmax {
    axis = 1 : i64
  } %arg0 : tensor<8x10xf32> -> tensor<9x1xf32>, tensor<9x1xi32>
  return %0, %1 : tensor<9x1xf32>, tensor<9x1xi32>
}

// -----

func.func @trt_prelu(%arg0: tensor<10x20xf32>, %arg1: tensor<1x20xf32>) -> tensor<1x20xf32> {
    // expected-error @below {{'tensorrt.parametric_relu' op inferred type(s) 'tensor<10x20xf32>' are incompatible with return type(s) of operation 'tensor<1x20xf32>'}}
    // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.parametric_relu ins(%arg0, %arg1: tensor<10x20xf32>, tensor<1x20xf32>) -> tensor<1x20xf32>
    return %0 : tensor<1x20xf32>
}

// -----

func.func @trt_padding(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x1x13x11xf32> {
  // expected-error @below {{'tensorrt.padding' op inferred type(s) 'tensor<1x1x7x11xf32>' are incompatible with return type(s) of operation 'tensor<1x1x13x11xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.padding {
    prePadding = array<i64: -2, 0>,
    postPadding = array<i64: -1, 1>
  } ins(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x1x13x11xf32>
  return %0 : tensor<1x1x13x11xf32>
}

// -----

func.func @trt_padding(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x5x11x13xf32> {
  // expected-error @below {{'tensorrt.padding' op padding exactly two innermost dimensions is supported but received "prePadding" input of size: 3 and "postPadding" input of size: 3}}
  %0 = tensorrt.padding {
    prePadding = array<i64: 2, 2, 0>,
    postPadding = array<i64: 2, -1, 3>
  } ins(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x5x11x13xf32>
  return %0 : tensor<1x5x11x13xf32>
}

// -----

func.func @trt_padding(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x1x10x12xf32> {
  // expected-error @below {{'tensorrt.padding' op padding exactly two innermost dimensions is supported but received "prePadding" input of size: 1 and "postPadding" input of size: 1}}
  %0 = tensorrt.padding {
    prePadding = array<i64: 2>,
    postPadding = array<i64: 2>
  } ins(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x1x10x12xf32>
  return %0 : tensor<1x1x10x12xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{op failed to infer returned types}}
  // expected-error @below {{'tensorrt.deconvolution' op inferred type(s) 'tensor<?x64x255x255xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 2, 2>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{op failed to infer returned types}}
  // expected-error @below {{'tensorrt.deconvolution' op inferred type(s) 'tensor<?x64x255x128xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 2, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{op failed to infer returned types}}
  // expected-error @below {{'tensorrt.deconvolution' op inferred type(s) 'tensor<?x64x130x130xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    dilation = array<i64: 2, 2>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{op failed to infer returned types}}
  // expected-error @below {{'tensorrt.deconvolution' op inferred type(s) 'tensor<?x64x127x124xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 2, 4>,
    post_padding = array<i64: 1, 2>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{op failed to infer returned types}}
  // expected-error @below {{'tensorrt.deconvolution' op inferred type(s) 'tensor<?x64x256x128xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 2, 1>,
    pre_padding = array<i64: 2, 4>,
    post_padding = array<i64: 1, 2>,
    dilation = array<i64: 2, 3>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x1x1xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{op failed to infer returned types}}
  // expected-error @below {{'tensorrt.deconvolution' op inferred type(s) 'tensor<?x64x252x122xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 2, 1>,
    pre_padding = array<i64: 2, 4>,
    post_padding = array<i64: 1, 2>,
    dilation = array<i64: 2, 3>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x1x1xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x1x1xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{op failed to infer returned types}}
  // expected-error @below {{'tensorrt.deconvolution' op inferred type(s) 'tensor<?x2048x126x126xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    num_groups = 32 : ui32
  } in(%arg0 : tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x1x1xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x128x128xf32> {
  // expected-error @below {{'tensorrt.deconvolution' op dilation should have size equal to the number of spatial dimensions}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    dilation = array<i64: 1, 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x8x128x128xf32>, %arg1: tensor<8x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x128x128xf32> {
  // expected-error @below {{'tensorrt.deconvolution' op input channels must be divisible by "num_groups"}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    num_groups = 16 : ui32
  } in(%arg0 : tensor<?x8x128x128xf32>) kernelWeights(%arg1: tensor<8x64x3x3xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x?x?xf32>, %arg1: tensor<32x2x1x1xf32>, %arg2: tensor<2xf32>) -> tensor<?x2x129x129xf32> {
  // expected-error @below {{'tensorrt.deconvolution' op inferred type(s) 'tensor<?x64x?x?xf32>' are incompatible with return type(s) of operation 'tensor<?x2x129x129xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    num_groups = 32 : ui32
  } in(%arg0 : tensor<?x32x?x?xf32>) kernelWeights(%arg1: tensor<32x2x1x1xf32>) biasWeights(%arg2 : tensor<2xf32>) -> tensor<?x2x129x129xf32>
  return %0 : tensor<?x2x129x129xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x?x?xf32>, %arg1: tensor<32x64x1x1xf32>, %arg2: tensor<64xf32>) -> tensor<?x63x129x129xf32> {
  // expected-error @below {{'tensorrt.deconvolution' op bias type should be a rank-1 tensor type with size equal to the number of channels (dim 1) of the result tensor type}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x?x?xf32>) kernelWeights(%arg1: tensor<32x64x1x1xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x63x129x129xf32>
  return %0 : tensor<?x63x129x129xf32>
}

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{'tensorrt.deconvolution' op inferred type(s) 'tensor<?x64x128x128xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_prelu(%arg0: tensor<10x20xf32>, %arg1: tensor<1x20xf32>) -> tensor<10x20xf16> {
    // expected-error @below {{'tensorrt.parametric_relu' op inferred type(s) 'tensor<10x20xf32>' are incompatible with return type(s) of operation 'tensor<10x20xf16>'}}
    // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.parametric_relu ins(%arg0, %arg1: tensor<10x20xf32>, tensor<1x20xf32>) -> tensor<10x20xf16>
    return %0 : tensor<10x20xf16>
}

// -----

func.func @trt_pooling_static(%arg0: tensor<10x64x112x112xf32>) -> tensor<10x64x57x57xf32> {
  // expected-error @below {{'tensorrt.pooling' op inferred type(s) 'tensor<10x64x56x56xf32>' are incompatible with return type(s) of operation 'tensor<10x64x57x57xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kMAX>,
    windowSize = array<i64: 3, 3>,
    stride = array<i64: 2, 2>,
    prePadding = array<i64: 1, 1>,
    postPadding = array<i64: 1, 1>
  } ins(%arg0 : tensor<10x64x112x112xf32>) -> tensor<10x64x57x57xf32>
  return %0 : tensor<10x64x57x57xf32>
}
// -----

func.func @trt_pooling_static(%arg0: tensor<10x64x112x112xf32>) -> tensor<10x64x57x57xf32> {
  // expected-error @below {{'tensorrt.pooling' op inferred type(s) 'tensor<10x64x110x110xf32>' are incompatible with return type(s) of operation 'tensor<10x64x57x57xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kMAX>,
    windowSize = array<i64: 3, 3>,
    stride = array<i64: 1, 1>,
    prePadding = array<i64: 0, 0>,
    postPadding = array<i64: 0, 0>
  } ins(%arg0 : tensor<10x64x112x112xf32>) -> tensor<10x64x57x57xf32>
  return %0 : tensor<10x64x57x57xf32>
}

// -----

func.func @trt_pooling_dynamic(%arg0: tensor<?x64x?x112xf32>) -> tensor<10x64x57x57xf32> {
  // expected-error @below {{'tensorrt.pooling' op inferred type(s) 'tensor<?x64x?x110xf32>' are incompatible with return type(s) of operation 'tensor<10x64x57x57xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kMAX>,
    windowSize = array<i64: 3, 3>,
    stride = array<i64: 1, 1>,
    prePadding = array<i64: 0, 0>,
    postPadding = array<i64: 0, 0>
  } ins(%arg0 : tensor<?x64x?x112xf32>) -> tensor<10x64x57x57xf32>
  return %0 : tensor<10x64x57x57xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{'tensorrt.convolution' op inferred type(s) 'tensor<?x64x64x64xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.convolution {
    stride = array<i64: 2, 2>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{'tensorrt.convolution' op inferred type(s) 'tensor<?x64x64x128xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.convolution {
    stride = array<i64: 2, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{'tensorrt.convolution' op inferred type(s) 'tensor<?x64x126x126xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    dilation = array<i64: 2, 2>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{'tensorrt.convolution' op inferred type(s) 'tensor<?x64x129x132xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 2, 4>,
    post_padding = array<i64: 1, 2>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{'tensorrt.convolution' op inferred type(s) 'tensor<?x64x64x128xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.convolution {
    stride = array<i64: 2, 1>,
    pre_padding = array<i64: 2, 4>,
    post_padding = array<i64: 1, 2>,
    dilation = array<i64: 2, 3>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x1x1xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{'tensorrt.convolution' op inferred type(s) 'tensor<?x64x66x134xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.convolution {
    stride = array<i64: 2, 1>,
    pre_padding = array<i64: 2, 4>,
    post_padding = array<i64: 1, 2>,
    dilation = array<i64: 2, 3>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x1x1xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x1x1xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{'tensorrt.convolution' op for "num_groups" = 32 and input channels = 32, second (idx = 1) kernel dimension should be 1}}
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    num_groups = 32 : ui32
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x1x1xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x128x128xf32> {
  // expected-error @below {{'tensorrt.convolution' op dilation should have size equal to the number of spatial dimensions}}
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    dilation = array<i64: 1, 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<32x64x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x8x128x128xf32>, %arg1: tensor<8x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x128x128xf32> {
  // expected-error @below {{'tensorrt.convolution' op both input channels and output channels must be divisible by "num_groups"}}
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    num_groups = 16 : ui32
  } in(%arg0 : tensor<?x8x128x128xf32>) kernel(%arg1: tensor<8x64x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x?x?xf32>, %arg1: tensor<64x32x1x1xf32>, %arg2: tensor<64xf32>) -> tensor<?x63x129x129xf32> {
  // expected-error @below {{'tensorrt.convolution' op bias type should be a rank-1 tensor type with size equal to the number of channels (dim 1) of the result tensor type}}
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x?x?xf32>) kernel(%arg1: tensor<64x32x1x1xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x63x129x129xf32>
  return %0 : tensor<?x63x129x129xf32>
}

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x129x129xf32> {
  // expected-error @below {{'tensorrt.convolution' op inferred type(s) 'tensor<?x32x128x128xf32>' are incompatible with return type(s) of operation 'tensor<?x64x129x129xf32>'}}
  // expected-error @below {{op failed to infer returned types}}
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<32x64x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x129x129xf32>
  return %0 : tensor<?x64x129x129xf32>
}

// -----

func.func @trt_if(%arg1: tensor<20xf32>, %arg2: tensor<20xf32>, %arg3: tensor<10xf32>, %arg4: tensor<10xf32>) -> tensor<10xf32> {
  %cond = arith.constant dense<1> : tensor<i1>
  // expected-error @below {{true and false regions must yield equivalent types}}
  // expected-error @below {{'tensorrt.if' op failed to infer returned types}}
  %result = tensorrt.if (%cond: tensor<i1>) -> tensor<10xf32> {
      %add = tensorrt.element_wise <kSUM>(%arg1, %arg2 : tensor<20xf32>, tensor<20xf32>)
          -> tensor<20xf32>
      tensorrt.yield %add: tensor<20xf32>
    } else {
      %sub = tensorrt.element_wise <kSUB>(%arg3, %arg4 : tensor<10xf32>, tensor<10xf32>)
          -> tensor<10xf32>
      tensorrt.yield %sub: tensor<10xf32>
    }
  return %result: tensor<10xf32>
}

// -----

func.func @trt_if(%arg1: tensor<20xf32>, %arg2: tensor<20xf32>) -> tensor<10xf32> {
  %cond = arith.constant dense<1> : tensor<i1>
  // expected-error @below {{number of output tensors in true and false regions must be same}}
  // expected-error @below {{'tensorrt.if' op failed to infer returned types}}
  %result = tensorrt.if (%cond: tensor<i1>) -> tensor<10xf32> {
      %add = tensorrt.element_wise <kSUM>(%arg1, %arg2 : tensor<20xf32>, tensor<20xf32>)
          -> tensor<20xf32>
      tensorrt.yield %arg1: tensor<20xf32>
    } else {
      %values, %indices = tensorrt.top_k <kMAX> {
            k = 1 : i64,
            axis = 0 : i64
        } %arg1 : tensor<20xf32> -> tensor<1xf32>, tensor<1xi32>

      tensorrt.yield %values, %indices: tensor<1xf32>, tensor<1xi32>
    }
  return %result: tensor<10xf32>
}

// -----

func.func @trt_if(%arg1: tensor<10xf32>, %arg2: tensor<10xf32>) -> tensor<20xf32> {
  %cond = arith.constant dense<1> : tensor<i1>
  // expected-error @below {{'tensorrt.if' op inferred type(s) 'tensor<10xf32>' are incompatible with return type(s) of operation 'tensor<20xf32>'}}
  // expected-error @below {{'tensorrt.if' op failed to infer returned types}}
  %result = tensorrt.if (%cond: tensor<i1>) -> tensor<20xf32> {
      %add = tensorrt.element_wise <kSUM>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
          -> tensor<10xf32>
      tensorrt.yield %arg1: tensor<10xf32>
    } else {
      %sub = tensorrt.element_wise <kSUB>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
          -> tensor<10xf32>
      tensorrt.yield %arg1: tensor<10xf32>
    }
  return %result: tensor<20xf32>
}

// -----

func.func @trt_if(%arg1: tensor<10xf32>, %arg2: tensor<10xf32>) -> tensor<20xf32> {
  %cond = arith.constant dense<1> : tensor<i1>
  // expected-error @below {{'tensorrt.if' op inferred type(s) 'tensor<10xf32>', 'tensor<10xf32>' are incompatible with return type(s) of operation 'tensor<10xf32>', 'tensor<20xf32>'}}
  // expected-error @below {{'tensorrt.if' op failed to infer returned types}}
  %a, %b = tensorrt.if (%cond: tensor<i1>) -> tensor<10xf32>, tensor<20xf32> {
      %add = tensorrt.element_wise <kSUM>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
          -> tensor<10xf32>
      %sub = tensorrt.element_wise <kSUB>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
          -> tensor<10xf32>
      tensorrt.yield %add, %sub: tensor<10xf32>, tensor<10xf32>
    } else {
      %sub = tensorrt.element_wise <kSUB>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
          -> tensor<10xf32>
      %add = tensorrt.element_wise <kSUM>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
          -> tensor<10xf32>
      tensorrt.yield %sub, %add: tensor<10xf32>, tensor<10xf32>
    }
  return %a, %b: tensor<10xf32>, tensor<20xf32>
}

// -----

func.func @trt_scatter_nd_incorrect_output_shape(%arg0: tensor<4x4x4xf32>, %arg1: tensor<2x1xi32>, %arg2: tensor<2x4x4xf32>) -> tensor<2x4x4xf32> {
  // expected-note @below {{prior use here}}
  %result = tensorrt.scatter_nd
      data(%arg0 : tensor<4x4x4xf32>)
      indices(%arg1: tensor<2x1xi32>)
      updates(%arg2: tensor<2x4x4xf32>)
  // expected-error @below {{use of value '%result' expects different type than prior uses: 'tensor<2x4x4xf32>' vs 'tensor<4x4x4xf32>'}}
  return %result : tensor<2x4x4xf32>
}

// -----

func.func @trt_scatter_nd_mismatched_types_1(%arg0: tensor<4x4x4xf32>, %arg1: tensor<2x1xi32>, %arg2: tensor<2x4x4xf16>) -> tensor<4x4x4xf32> {
  // expected-error @below {{'tensorrt.scatter_nd' op failed to verify that all of {data, updates} have same element type}}
  %0 = tensorrt.scatter_nd
    data(%arg0 : tensor<4x4x4xf32>)
    indices(%arg1: tensor<2x1xi32>)
    updates(%arg2: tensor<2x4x4xf16>)
  return %0 : tensor<4x4x4xf32>
}

// -----

func.func @trt_scatter_nd_unexpected_indices_type(%arg0: tensor<4x4x4xf32>, %arg1: tensor<2x1xi8>, %arg2: tensor<2x4x4xf32>) -> tensor<4x4x4xf32> {
  // expected-error @below {{'tensorrt.scatter_nd' op operand #1 must be 0D/1D/2D/3D/4D/5D/6D/7D/8D tensor of 32-bit signless integer values, but got 'tensor<2x1xi8>'}}
  %0 = tensorrt.scatter_nd
    data(%arg0 : tensor<4x4x4xf32>)
    indices(%arg1: tensor<2x1xi8>)
    updates(%arg2: tensor<2x4x4xf32>)
  return %0 : tensor<4x4x4xf32>
}

// -----

func.func @trt_scatter_nd_incorrect_updates_rank(%arg0: tensor<4xf32>, %arg1: tensor<1xi32>, %arg2: tensor<2xf32>) -> tensor<4xf32> {
  // expected-error @below {{'tensorrt.scatter_nd' op expected updates tensor rank to be 0}}
  %0 = tensorrt.scatter_nd
    data(%arg0 : tensor<4xf32>)
    indices(%arg1: tensor<1xi32>)
    updates(%arg2: tensor<2xf32>)
  return %0 : tensor<4xf32>
}

// -----

func.func @trt_scatter_nd_incorrect_indices_rank(%arg0: tensor<4xf32>, %arg1: tensor<i32>, %arg2: tensor<f32>) -> tensor<4xf32> {
  // expected-error @below {{'tensorrt.scatter_nd' op expected indices to have rank >= 1}}
  %0 = tensorrt.scatter_nd
    data(%arg0 : tensor<4xf32>)
    indices(%arg1: tensor<i32>)
    updates(%arg2: tensor<f32>)
  return %0 : tensor<4xf32>
}

// -----

func.func @trt_scatter_elements_axis_out_of_bounds(%arg0: tensor<3x3xf32>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xf32>) -> tensor<3x3xf32> {
  // expected-error @below {{'tensorrt.scatter_elements' op expected axis to be in the range [0, 2)}}
  %0 = tensorrt.scatter_elements {
    axis = 3: i64
    }
    data(%arg0: tensor<3x3xf32>)
    indices(%arg1: tensor<2x3xi32>)
    updates(%arg2: tensor<2x3xf32>)

    return %0: tensor<3x3xf32>
}

// -----

func.func @trt_scatter_elements_zero_rank_input(%arg0: tensor<f32>, %arg1: tensor<i32>, %arg2: tensor<f32>) -> tensor<f32> {
  // expected-error @below {{'tensorrt.scatter_elements' op expected data to have rank >= 1, got 0}}
  %0 = tensorrt.scatter_elements
    data(%arg0: tensor<f32>)
    indices(%arg1: tensor<i32>)
    updates(%arg2: tensor<f32>)

    return %0: tensor<f32>
}

// -----

func.func @trt_assertion(%condition: tensor<1x3xi1>) {
  // expected-error @below {{expected condition to be of rank 0 or 1}}
  tensorrt.assertion {
    message = "One or more conditions fail."
  } ins(%condition : tensor<1x3xi1>)

  return
}

// -----

func.func @trt_normalize_oob_index(%inp: tensor<1x3x100x100xf16>, %scale: tensor<1x3x1x1xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<1x3x100x100xf16> {
    // expected-error @below {{'tensorrt.normalization' op `axis` value 5 is out of the bound for input tensor of rank 4}}
    %0 = tensorrt.normalization {
        axis = array<i64: 5>
    } (%inp: tensor<1x3x100x100xf16>, %scale: tensor<1x3x1x1xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<1x3x100x100xf16>
    return %0: tensor<1x3x100x100xf16>
}

// -----

func.func @trt_batch_normalize_wrong_scale(%inp: tensor<2x3x2x2xf16>, %scale: tensor<1x3x1x2xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<2x3x2x2xf16> {
    // expected-error @below {{'tensorrt.normalization' op In case of batch normalization (axis=0), scale and bias shape is expected to be [1, C, 1, 1, ..] where input is in the form [N, C, H, W, ...]}}
    %0 = tensorrt.normalization {
        axis = array<i64: 0>
    } (%inp: tensor<2x3x2x2xf16>, %scale: tensor<1x3x1x2xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<2x3x2x2xf16>
    return %0: tensor<2x3x2x2xf16>
}

// -----

func.func @trt_group_normalize_channels_non_divisible_by_groups(%inp: tensor<1x24x100x100xf16>, %scale: tensor<1x6x1x1xf16>, %bias: tensor<1x6x1x1xf16>) -> tensor<1x24x100x100xf16> {
    // expected-error @below {{'tensorrt.normalization' op It is an error to set `num_groups` to a value that does not evenly divide into the number of channels of the input tensor.}}
    %0 = tensorrt.normalization {
        axis = array<i64: 2, 3>, num_groups = 5 : i32
    } (%inp: tensor<1x24x100x100xf16>, %scale: tensor<1x6x1x1xf16>, %bias: tensor<1x6x1x1xf16>) -> tensor<1x24x100x100xf16>
    return %0: tensor<1x24x100x100xf16>
}

// -----

func.func @trt_group_normalize_wrong_axis(%inp: tensor<2x24x2x2xf16>, %scale: tensor<1x4x1x1xf16>, %bias: tensor<1x4x1x1xf16>) -> tensor<2x24x2x2xf16> {
    // expected-error @below {{'tensorrt.normalization' op If num_groups != 1, it is expected that axis array contains all dimensions after the channel dimension (which is 1).}}
    %0 = tensorrt.normalization {
        axis = array<i64: 2>, num_groups = 4 : i32
    } (%inp: tensor<2x24x2x2xf16>, %scale: tensor<1x4x1x1xf16>, %bias: tensor<1x4x1x1xf16>) -> tensor<2x24x2x2xf16>
    return %0: tensor<2x24x2x2xf16>
}

// -----

func.func @trt_group_normalize_wrong_scale(%inp: tensor<2x24x2x2xf16>, %scale: tensor<1x24x1x1xf16>, %bias: tensor<1x4x1x1xf16>) -> tensor<2x24x2x2xf16> {
    // expected-error @below {{'tensorrt.normalization' op If num_groups != 1, scale and bias shape is expected to be [1, num_groups, 1, 1, ... N] where N is rank of input tensor i.e. 4}}
    %0 = tensorrt.normalization {
        axis = array<i64: 2, 3>, num_groups = 4 : i32
    } (%inp: tensor<2x24x2x2xf16>, %scale: tensor<1x24x1x1xf16>, %bias: tensor<1x4x1x1xf16>) -> tensor<2x24x2x2xf16>
    return %0: tensor<2x24x2x2xf16>
}

// -----

func.func @trt_instance_normalize_wrong_axis(%inp: tensor<2x3x2x2x2xf16>, %scale: tensor<1x3x1x1x1xf16>, %bias: tensor<1x3x1x1x1xf16>) -> tensor<2x3x2x2x2xf16> {
    // expected-error @below {{'tensorrt.normalization' op If more than one axis is provided and scale/bias are in the form [1, C, 1, 1, .. N], this is the case for instance normalization. Array `axis` should contain all the axis after channel.}}
    %0 = tensorrt.normalization {
        axis = array<i64: 2, 3>
    } (%inp: tensor<2x3x2x2x2xf16>, %scale: tensor<1x3x1x1x1xf16>, %bias: tensor<1x3x1x1x1xf16>) -> tensor<2x3x2x2x2xf16>
    return %0: tensor<2x3x2x2x2xf16>
}

// -----

func.func @random_uniform_all_dynamic(%low: tensor<f16>, %high: tensor<f32>, %shape: tensor<4xi32>) -> tensor<?x?x?x?xf16> {
  // expected-error @below {{'tensorrt.random_uniform' op `low` and `high` tensor types must have the same element type}}
  %0 = tensorrt.random_uniform low(%low: tensor<f16>) high(%high: tensor<f32>) shape(%shape: tensor<4xi32>) ->  tensor<?x?x?x?xf16>
}

// -----

func.func @random_normal_all_dynamic(%mean: tensor<f16>, %std: tensor<f32>, %shape: tensor<4xi32>) -> tensor<?x?x?x?xf16> {
  // expected-error @below {{'tensorrt.random_normal' op `mean` and `std` tensor types must have the same element type}}
  %0 = tensorrt.random_normal mean(%mean: tensor<f16>) std(%std: tensor<f32>) shape(%shape: tensor<4xi32>) ->  tensor<?x?x?x?xf16>
  return %0 : tensor<?x?x?x?xf16>
}

// -----

func.func @random_uniform_all_dynamic(%low: tensor<f16>, %high: tensor<f16>, %shape: tensor<4xi32>) -> tensor<?x?x?x?xf32> {
  // expected-error @below {{'tensorrt.random_uniform' op `low`, `high` and `result` element type should be the same}}
  %0 = tensorrt.random_uniform low(%low: tensor<f16>) high(%high: tensor<f16>) shape(%shape: tensor<4xi32>) ->  tensor<?x?x?x?xf32>
}

// -----

func.func @random_normal_all_dynamic(%mean: tensor<f16>, %std: tensor<f16>, %shape: tensor<4xi32>) -> tensor<?x?x?x?xf32> {
  // expected-error @below {{'tensorrt.random_normal' op `mean`, `std` and `result` element type should be the same}}
  %0 = tensorrt.random_normal mean(%mean: tensor<f16>) std(%std: tensor<f16>) shape(%shape: tensor<4xi32>) ->  tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @random_uniform_all_dynamic(%low: tensor<f16>, %high: tensor<f16>) -> tensor<?x?x?x?xf16> {
  // expected-error @below {{'tensorrt.random_uniform' op If `result` has dynamic dims, `shape` tensor must be present}}
  %0 = tensorrt.random_uniform low(%low: tensor<f16>) high(%high: tensor<f16>) ->  tensor<?x?x?x?xf16>
}

// -----

func.func @random_normal_all_dynamic(%mean: tensor<f16>, %std: tensor<f16>) -> tensor<?x?x?x?xf16> {
  // expected-error @below {{'tensorrt.random_normal' op If `result` has dynamic dims, `shape` tensor must be present}}
  %0 = tensorrt.random_normal mean(%mean: tensor<f16>) std(%std: tensor<f16>) ->  tensor<?x?x?x?xf16>
  return %0 : tensor<?x?x?x?xf16>
}

// -----

func.func @trt_fill_linspace_i32(%arg0: tensor<i32>, %arg1: tensor<4xi32>) -> tensor<1x2x?x4xi32> {
  // expected-error @below {{'tensorrt.linspace' op If `result` has dynamic dims, `shape` tensor must be present}}
  %0 = tensorrt.linspace [%arg0: tensor<i32>][static][%arg1: tensor<4xi32>] : tensor<1x2x?x4xi32>
  return %0 : tensor<1x2x?x4xi32>
}

// -----

func.func @trt_fill_linspace_dynamic_f16() -> tensor<1024x1024xf16> {
  %shape = tensorrt.constant dense<[1024, 1024]>:tensor<2xi32>
  %start = tensorrt.constant dense<0.0>:tensor<f16>
  %step = tensorrt.constant dense<[1.0,1.0]>:tensor<2xf16>
  // expected-error @below {{'tensorrt.linspace' op operand #1 must be 0D tensor of 32-bit float or 32-bit signless integer values, but got 'tensor<f16>'}}
  %0 = tensorrt.linspace [%start:tensor<f16>][%shape:tensor<2xi32>][%step:tensor<2xf16>] : tensor<1024x1024xf16>
  return %0 : tensor<1024x1024xf16>
}

// -----

func.func @gather_nd_incorrect_shape(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<16x17x2xi32>) -> tensor<16x17x3x4xf32> {
  // expected-error @below {{'tensorrt.gather_nd' op inferred type(s) 'tensor<16x17x3x4xf32>' are incompatible with return type(s) of operation 'tensor<16x17x?xf32>'}}
  // expected-error @below {{'tensorrt.gather_nd' op failed to infer returned types}}
  %0 = tensorrt.gather_nd data(%arg0) indices(%arg1) : (tensor<1x2x3x4xf32>, tensor<16x17x2xi32>) -> tensor<16x17x?xf32>
  return %0 : tensor<16x17x?xf32>
}
// -----

func.func @gather_nd_invalid_index_size(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<16x17x5xi32>) -> tensor<16x17xf32> {
  // expected-error @below {{the extent of the last dimension of 'indices' shape must be greater than zero and less-than-or-equal-to 'data' rank}}
  // expected-error @below {{'tensorrt.gather_nd' op failed to infer returned types}}
  %0 = tensorrt.gather_nd data(%arg0) indices(%arg1) : (tensor<1x2x3x4xf32>, tensor<16x17x5xi32>) -> tensor<16x17xf32>
  return %0 : tensor<16x17xf32>
}

// -----

func.func @gather_nd_invalid_index_size2(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<16x17x?xi32>) -> tensor<16x17xf32> {

  // expected-error @below {{the extent of the last dimension of 'indices' shape must be greater than zero and less-than-or-equal-to 'data' rank}}
  // expected-error @below {{'tensorrt.gather_nd' op failed to infer returned types}}
  %0 = tensorrt.gather_nd data(%arg0) indices(%arg1) : (tensor<1x2x3x4xf32>, tensor<16x17x?xi32>) -> tensor<16x17xf32>
  return %0 : tensor<16x17xf32>
}

// -----

func.func @trt_constant_op_rank_more_than_8() -> (tensor<10x2x2x2x2x2x2x2x2xf32>) {
  // expected-error @below {{'tensorrt.constant' op result #0 must be 0D/1D/2D/3D/4D/5D/6D/7D/8D tensor}}
  %zeros = tensorrt.constant dense<0.0> : tensor<10x2x2x2x2x2x2x2x2xf32>
  return %zeros : tensor<10x2x2x2x2x2x2x2x2xf32>
}

// -----

func.func @trt_elementwise_op_rank_more_than_8(%zeros: tensor<10x2x2x2x2x2x2x2x2xf32>, %ones: tensor<10x2x2x2x2x2x2x2x2xf32>) -> (tensor<10x2x2x2x2x2x2x2x2xf32>) {
  // expected-error @below {{'tensorrt.element_wise' op operand #0 must be 0D/1D/2D/3D/4D/5D/6D/7D/8D tensor}}
  %add = tensorrt.element_wise <kSUM>(%zeros, %ones : tensor<10x2x2x2x2x2x2x2x2xf32>, tensor<10x2x2x2x2x2x2x2x2xf32>) -> tensor<10x2x2x2x2x2x2x2x2xf32>
  return %add : tensor<10x2x2x2x2x2x2x2x2xf32>
}

// -----

func.func @test_shape_region_wrong_num_block_args(%arg0: tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // expected-error @below {{'tensorrt.opaque_plugin' op expected 4 i64 block arguments but got 3 arguments  of types 'i64', 'i64', 'i64'}}
  %0 = tensorrt.opaque_plugin {
      creator_params = {},
      dso_path = "libTensorRTTestPlugins.so",
      plugin_name = "TestInferShapePlugin",
      plugin_namespace = "",
      plugin_version = "0"}(%arg0) : (tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  ^bb0(%arg1: i64, %arg3: i64, %arg4: i64):
    tensorrt.yield %arg1, %arg3, %arg4, %arg4: i64, i64, i64, i64
  }
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @test_shape_region_wrong_type_block_args(%arg0: tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // expected-error @below {{'tensorrt.opaque_plugin' op expected 4 i64 block arguments but got 4 arguments  of types 'i64', 'index', 'i64', 'i64'}}
  %0 = tensorrt.opaque_plugin {
      creator_params = {},
      plugin_name = "TestInferShapePlugin",
      plugin_namespace = "",
      plugin_version = "0"}(%arg0) : (tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  ^bb0(%arg1: i64, %arg2: index, %arg3: i64, %arg4: i64):
    tensorrt.yield %arg1, %arg2, %arg4, %arg4 : i64, index, i64, i64
  }
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @test_shape_region_wrong_num_yields(%arg0: tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // expected-error @below {{'tensorrt.opaque_plugin' op expected 4 i64 values to be yielded from the 'shapes' region but got 3 values of types 'i64', 'i64', 'i64'}}
  %0 = tensorrt.opaque_plugin {
      creator_params = {},
      plugin_name = "TestInferShapePlugin",
      plugin_namespace = "",
      plugin_version = "0"}(%arg0) : (tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  ^bb0(%arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64):
    tensorrt.yield %arg1, %arg2, %arg3 : i64, i64, i64
  }
  return %0 : tensor<?x?x?x?xf32>
}


// -----

func.func @test_shape_region_wrong_yield_types(%arg0: tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // expected-error @below {{'tensorrt.opaque_plugin' op expected 4 i64 values to be yielded from the 'shapes' region but got 4 values of types 'index', 'i64', 'i64', 'i64'}}
  %0 = tensorrt.opaque_plugin {
      creator_params = {},
      plugin_name = "TestInferShapePlugin",
      plugin_namespace = "",
      plugin_version = "0"}(%arg0) : (tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  ^bb0(%arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64):
    %1 = arith.index_cast %arg1 : i64 to index
    tensorrt.yield %1, %arg2, %arg3, %arg4 : index, i64, i64, i64
  }
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @test_shape_region_disallowed_ops(%arg0: tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // expected-error @below {{'tensorrt.opaque_plugin' op expected only 'arith' dialect ops and 'tensorrt.yield' (terminator) in the 'shapes' region, but an op of type 'affine.apply' is present}}
  %0 = tensorrt.opaque_plugin {
      creator_params = {},
      plugin_name = "TestInferShapePlugin",
      plugin_namespace = "",
      plugin_version = "0"}(%arg0) : (tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  ^bb0(%arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64):
    %1 = arith.index_cast %arg1 : i64 to index
    %2 = affine.apply affine_map<(d0)->(d0+1)>(%1)
    %3 = arith.index_cast %2 : index to i64
    tensorrt.yield %3, %arg2, %arg3, %arg4 : i64, i64, i64, i64
  }
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @test_plugin_wrong_attributes(%arg0: tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // expected-error @below {{'tensorrt.opaque_plugin' op "creator_func" is provided but "dso_path" was not specified}}
  %0 = tensorrt.opaque_plugin {
      creator_params = {},
      creator_func = "getPluginCreator",
      plugin_name = "TestInferShapePlugin",
      plugin_namespace = "",
      plugin_version = "0"}(%arg0) : (tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32>

  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @test_plugin_shape_verification(%arg0: tensor<?x4x?x?xf32>) -> tensor<41x?x?x16xf32> {
  // expected-error @below {{'tensorrt.opaque_plugin' op result 0 has type tensor<41x?x?x16xf32> but inferred tensor of shape <42x?x?x?>}}
  %0 = tensorrt.opaque_plugin {
      creator_params = {},
      plugin_name = "TestInferShapePlugin",
      plugin_namespace = "",
      plugin_version = "0"}(%arg0) : (tensor<?x4x?x?xf32>) -> tensor<41x?x?x16xf32> {
  ^bb0(%arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64):
    %1 = arith.constant 42 : i64
    tensorrt.yield %1, %arg2, %arg3, %arg4 : i64, i64, i64, i64
  }
  return %0 : tensor<41x?x?x16xf32>
}
