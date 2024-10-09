// RUN: mlir-tensorrt-opt -split-input-file -tensorrt-stablehlo-input-preprocessing -convert-stablehlo-to-tensorrt %s | FileCheck %s
func.func @conv2d_nhwc_rsck_no_padding_dilated(
    %arg0: tensor<1x32x64x2xf32>,
    %arg1: tensor<3x3x2x128xf32>)
  -> tensor<1x28x62x128xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 2, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x32x64x2xf32>, tensor<3x3x2x128xf32>) -> tensor<1x28x62x128xf32>
  func.return %0 : tensor<1x28x62x128xf32>
}

// CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0, d1)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @conv2d_nhwc_rsck_no_padding_dilated
//  CHECK-SAME:     (%[[arg0:.+]]: tensor<{{.+}}xf32>, %[[arg1:.+]]: tensor<{{.+}}xf32>)
//       CHECK:   %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]] :
//       CHECK:   %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[arg1]] :
//       CHECK:   %[[v2:.+]] = tensorrt.convolution
//  CHECK-SAME:       dilation = array<i64: 2, 1>
//  CHECK-SAME:      post_padding = array<i64: 0, 0>
//  CHECK-SAME:      pre_padding = array<i64: 0, 0>
//  CHECK-SAME:      stride = array<i64: 1, 1>
//  CHECK-sAME:        in(%[[v0]] : tensor<1x2x32x64xf32>) kernel(%[[v1]] : tensor<128x2x3x3xf32>) -> tensor<1x128x28x62xf32>
//       CHECK:   %[[v3:.+]] = tensorrt.transpose {permutation = #[[$map2]]} %[[v2]]
//       CHECK:   return %[[v3]] : tensor<1x28x62x128xf32>

// -----

func.func @conv2d_nchw_kcrs_padded(
    %arg0: tensor<1x2x32x64xf32>,
    %arg1: tensor<128x2x3x3xf32>)
  -> tensor<1x128x28x62xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 1,
      input_spatial_dimensions = [2, 3],
      kernel_input_feature_dimension = 1,
      kernel_output_feature_dimension = 0,
      kernel_spatial_dimensions = [2, 3],
      output_batch_dimension = 0,
      output_feature_dimension = 1,
      output_spatial_dimensions = [2, 3]
    >,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 2, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x2x32x64xf32>, tensor<128x2x3x3xf32>)
    -> tensor<1x128x28x62xf32>
  func.return %0 : tensor<1x128x28x62xf32>
}

// CHECK-LABEL: @conv2d_nchw_kcrs_padded(
//  CHECK-SAME:       %[[arg0:.+]]: tensor<1x2x32x64xf32>, %[[arg1:.+]]: tensor<128x2x3x3xf32>)
//       CHECK:   %[[v0:.+]] = tensorrt.convolution {dilation = array<i64: 2, 1>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 1>}
//  CHECK-SAME:     in(%[[arg0]] : tensor<1x2x32x64xf32>) kernel(%[[arg1]] : tensor<128x2x3x3xf32>) -> tensor<1x128x28x62xf32>
//       CHECK:   return %[[v0]]


// -----

func.func @conv3d_ndhwc_drsck(%arg0: tensor<?x16x16x16x?xf32>, %arg1: tensor<2x2x2x?x?xf32>)
  -> tensor<?x15x15x15x?xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 4,
      input_spatial_dimensions = [1, 2, 3],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 4,
      kernel_spatial_dimensions = [0, 1, 2],
      output_batch_dimension = 0,
      output_feature_dimension = 4,
      output_spatial_dimensions = [1, 2, 3]
    >,
    feature_group_count = 1 : i64,
    padding = dense<[[0, 0], [0, 0], [0, 0]]> : tensor<3x2xi64>,
    rhs_dilation = array<i64: 1, 1, 1>,
    window_strides = array<i64: 1, 1, 1>
  } : (tensor<?x16x16x16x?xf32>, tensor<2x2x2x?x?xf32>) -> tensor<?x15x15x15x?xf32>
  func.return %0 : tensor<?x15x15x15x?xf32>
}

// CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2, d3)>
// CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d3, d0, d1, d2)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>

// CHECK-LABEL: @conv3d_ndhwc_drsck
//  CHECK-SAME:     (%[[arg0:.+]]: tensor<{{.+}}xf32>, %[[arg1:.+]]: tensor<{{.+}}xf32>)
//       CHECK:   %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]] :
//       CHECK:   %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[arg1]] :
//       CHECK:   %[[v2:.+]] = tensorrt.convolution
//  CHECK-SAME:       dilation = array<i64: 1, 1, 1>
//  CHECK-SAME:      post_padding = array<i64: 0, 0, 0>
//  CHECK-SAME:      pre_padding = array<i64: 0, 0, 0>
//  CHECK-SAME:      stride = array<i64: 1, 1, 1>
//  CHECK-sAME:        in(%[[v0]] : tensor<?x?x16x16x16xf32>) kernel(%[[v1]] : tensor<?x?x2x2x2xf32>) -> tensor<?x?x15x15x15xf32>
//       CHECK:   %[[v3:.+]] = tensorrt.transpose {permutation = #[[$map2]]} %[[v2]]
//       CHECK:   return %[[v3]] : tensor<?x15x15x15x?xf32>

// -----

func.func @conv2d_chwb_crsk_hwnc(
    %arg0: tensor<4x6x7x1xf32>,
    %arg1: tensor<2x6x3x2xf32>)
  -> tensor<1x2x1x2xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, -1]],
              rhs_dilate = [1, 2],
              reverse = [0, 0]}
    {
      batch_group_count = 1 : i64,
      feature_group_count = 2 : i64
    } : (tensor<4x6x7x1xf32>, tensor<2x6x3x2xf32>) -> tensor<1x2x1x2xf32>
  return %0 : tensor<1x2x1x2xf32>
}

// CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
// CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>

// CHECK-LABEL: @conv2d_chwb_crsk_hwnc
//  CHECK-SAME:     (%[[arg0:.+]]: tensor<{{.+}}xf32>, %[[arg1:.+]]: tensor<{{.+}}xf32>)
//       CHECK:   %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]] :
//       CHECK:   %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg1]] :
//       CHECK:   %[[v2:.+]] = tensorrt.convolution
//  CHECK-SAME:       dilation = array<i64: 1, 2>
//  CHECK-SAME:        num_groups = 2 : ui32
//  CHECK-SAME:      post_padding = array<i64: 0, -1>
//  CHECK-SAME:      pre_padding = array<i64: 0, 0>
//  CHECK-SAME:      stride = array<i64: 1, 1>
//  CHECK-sAME:        in(%[[v0]] : tensor<1x4x6x7xf32>) kernel(%[[v1]] : tensor<2x2x6x3xf32>) -> tensor<1x2x1x2xf32>
//       CHECK:   %[[v3:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[v2]]
//       CHECK:   return %[[v3]] : tensor<1x2x1x2xf32>

// -----

func.func @conv2d_nhwc_rsck_lhs_dilate(
  %arg0: tensor<2x9x10x3xf32>,
  %arg1: tensor<4x4x3x3xf32>)
  -> tensor<2x15x25x3xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1],
      pad = [[6, 6], [6, 6]],
      lhs_dilate = [1, 2],
      rhs_dilate = [2, 2]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<2x9x10x3xf32>, tensor<4x4x3x3xf32>) -> tensor<2x15x25x3xf32>
  return %0 : tensor<2x15x25x3xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0, d1)>
//   CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
//   CHECK-DAG: #[[$map3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @conv2d_nhwc_rsck_lhs_dilate
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}) -> {{.*}}
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[arg1]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.slice %[[v1]][0, 0, 3, 3][3, 3, 4, 4][1, 1, -1, -1]
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.transpose {permutation = #[[$map2]]} %[[v2]]
//  CHECK-NEXT: %[[v4:.+]] = tensorrt.deconvolution
//  CHECK-SAME: {dilation = array<i64: 2, 2>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 2>}
//  CHECK-SAME: in(%[[v0]] : tensor<2x3x9x10xf32>) kernelWeights(%[[v3]] : tensor<3x3x4x4xf32>) -> {{.*}}
//  CHECK-NEXT: %[[v5:.+]] = tensorrt.transpose {permutation = #[[$map3]]} %[[v4]]
//  CHECK-NEXT: return %[[v5]] : {{.*}}

// -----

func.func @conv2d_nchw_kcrs_lhs_dilate_group(
  %arg0: tensor<1x72x8x14xf32>,
  %arg1: tensor<72x24x4x4xf32>)
  -> tensor<1x72x16x28xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {
      stride = [1, 1],
      pad = [[2, 2], [2, 2]],
      lhs_dilate = [2, 2],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 3 : i64
    } : (tensor<1x72x8x14xf32>, tensor<72x24x4x4xf32>) -> tensor<1x72x16x28xf32>
  return %0 : tensor<1x72x16x28xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3, d4)>

// CHECK-LABEL: @conv2d_nchw_kcrs_lhs_dilate_group
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}) -> {{.*}}
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.slice %[[arg1]][0, 0, 3, 3][72, 24, 4, 4][1, 1, -1, -1]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.reshape %[[v0]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v1]]
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.reshape %[[v2]]
//  CHECK-NEXT: %[[v4:.+]] = tensorrt.deconvolution
//  CHECK-SAME: {dilation = array<i64: 1, 1>, num_groups = 3 : ui32, post_padding = array<i64: 1, 1>, pre_padding = array<i64: 1, 1>, stride = array<i64: 2, 2>}
//  CHECK-SAME: in(%[[arg0]] : tensor<1x72x8x14xf32>) kernelWeights(%[[v3]] : tensor<72x24x4x4xf32>) -> {{.*}}
//  CHECK-NEXT: return %[[v4:.+]] : {{.*}}

// -----

func.func @conv2d_nchw_kcrs_lhs_dilate_group_2(
  %arg0: tensor<1x4x3x3xf32>,
  %arg1: tensor<16x2x1x1xf32>)
  -> tensor<1x16x5x5xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {
      stride = [1, 1],
      pad = [[0, 0], [0, 0]],
      lhs_dilate = [2, 2],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 2 : i64
    } : (tensor<1x4x3x3xf32>, tensor<16x2x1x1xf32>) -> tensor<1x16x5x5xf32>
  return %0 : tensor<1x16x5x5xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3, d4)>

// CHECK-LABEL: @conv2d_nchw_kcrs_lhs_dilate_group_2
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}) -> {{.*}}
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.slice %[[arg1]][0, 0, 0, 0][16, 2, 1, 1][1, 1, -1, -1]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.reshape %[[v0]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v1]]
//  CHECK-NEXT: %[[v3:.+]] = tensorrt.reshape %[[v2]]
//  CHECK-NEXT: %[[v4:.+]] = tensorrt.deconvolution
//  CHECK-SAME: {dilation = array<i64: 1, 1>, num_groups = 2 : ui32, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 2, 2>}
//  CHECK-SAME: in(%[[arg0]] : tensor<1x4x3x3xf32>) kernelWeights(%[[v3]] : tensor<4x8x1x1xf32>) -> {{.*}}
//  CHECK-NEXT: return %[[v4:.+]] : {{.*}}

// -----

func.func @conv2d_nchw_kcrs_lhs_dilate(
  %arg0: tensor<1x1x3x3xf32>,
  %arg1: tensor<1x1x3x3xf32>)
  -> tensor<1x1x5x7xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
  dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
  window = {
    stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 2], rhs_dilate = [1, 1], reverse = [0, 0]
    }
    {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x5x7xf32>
  return %0 : tensor<1x1x5x7xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @conv2d_nchw_kcrs_lhs_dilate
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}) -> {{.*}}
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.slice %[[arg1]][0, 0, 2, 2][1, 1, 3, 3][1, 1, -1, -1]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v0]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.deconvolution
//  CHECK-SAME: {dilation = array<i64: 1, 1>, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 2>}
//  CHECK-SAME: in(%[[arg0]] : tensor<1x1x3x3xf32>) kernelWeights(%[[v1]] : tensor<1x1x3x3xf32>) -> {{.*}}
//  CHECK-NEXT: return %[[v2:.+]] : {{.*}}

// -----

func.func @conv2d_nchw_kcrs_lhs_dilate_2(
  %arg0: tensor<1x2x5x5xf32>,
  %arg1: tensor<1x2x2x2xf32>)
  -> tensor<1x1x2x2xf32> {
    %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {
      stride = [1, 1],
      pad = [[-1, -1], [-1, -1]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1],
      reverse = [0, 0]
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x2x5x5xf32>, tensor<1x2x2x2xf32>) -> tensor<1x1x2x2xf32>
    return %0 : tensor<1x1x2x2xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @conv2d_nchw_kcrs_lhs_dilate_2
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}) -> {{.*}}
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.slice %[[arg1]][0, 0, 1, 1][1, 2, 2, 2][1, 1, -1, -1]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v0]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.deconvolution
//  CHECK-SAME: {dilation = array<i64: 1, 1>, post_padding = array<i64: 2, 2>, pre_padding = array<i64: 2, 2>, stride = array<i64: 1, 1>}
//  CHECK-SAME: in(%[[arg0]] : tensor<1x2x5x5xf32>) kernelWeights(%[[v1]] : tensor<2x1x2x2xf32>) -> {{.*}}
//  CHECK-NEXT: return %[[v2:.+]] : {{.*}}

// -----

func.func @conv2d_nchw_kcrs_lhs_dilate_3(
  %arg0: tensor<1x2x5x5xf32>,
  %arg1: tensor<1x2x2x2xf32>)
  -> tensor<1x1x2x1xf32> {
    %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {
      stride = [1, 1],
      pad = [[-1, -1], [-1, -1]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 2],
      reverse = [0, 0]
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x2x5x5xf32>, tensor<1x2x2x2xf32>) -> tensor<1x1x2x1xf32>
    return %0 : tensor<1x1x2x1xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @conv2d_nchw_kcrs_lhs_dilate_3
//  CHECK-SAME: (%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}}) -> {{.*}}
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.slice %[[arg1]][0, 0, 1, 1][1, 2, 2, 2][1, 1, -1, -1]
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v0]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.deconvolution
//  CHECK-SAME: {dilation = array<i64: 1, 2>, post_padding = array<i64: 2, 3>, pre_padding = array<i64: 2, 3>, stride = array<i64: 1, 1>}
//  CHECK-SAME: in(%[[arg0]] : tensor<1x2x5x5xf32>) kernelWeights(%[[v1]] : tensor<2x1x2x2xf32>) -> {{.*}}
//  CHECK-NEXT: return %[[v2:.+]] : {{.*}}

// -----

func.func @conv2d_nchw_lhs_dilate_dynamic_kernel_negative(
  %arg0: tensor<1x72x8x14xf32>,
  %arg1: tensor<72x24x?x?xf32>)
  -> tensor<1x72x16x28xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {
      stride = [1, 1],
      pad = [[2, 2], [2, 2]],
      lhs_dilate = [2, 2],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 3 : i64
    } : (tensor<1x72x8x14xf32>, tensor<72x24x?x?xf32>) -> tensor<1x72x16x28xf32>
  return %0 : tensor<1x72x16x28xf32>
}

// CHECK-LABEL: @conv2d_nchw_lhs_dilate_dynamic_kernel_negative
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.convolution
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @conv2d_nchw_kcrs_neg_effective_pad_negative(%arg0: tensor<1x1x3x3xf32>,
  %arg1: tensor<1x1x1x1xf32>) -> tensor<1x1x5x7xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
  dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
  window = {
    stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 2], rhs_dilate = [1, 1], reverse = [0, 0]
    }
    {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<1x1x3x3xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x5x7xf32>
  return %0 : tensor<1x1x5x7xf32>
}

// CHECK-LABEL: @conv2d_nchw_kcrs_neg_effective_pad_negative
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.convolution
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @conv2d_fully_seperable(%arg0: tensor<1x64x96x128xf16>) -> tensor<1x64x96x128xf16> {
  %0 = stablehlo.constant dense<1.0> : tensor<5x5x1x128xf16>
  %1 = stablehlo.convolution(%arg0, %0)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1],
    pad = [[2, 2], [2, 2]],
    rhs_dilate = [1, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 128 : i64
    } : (tensor<1x64x96x128xf16>, tensor<5x5x1x128xf16>) -> tensor<1x64x96x128xf16>
  return %1 : tensor<1x64x96x128xf16>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
//       CHECK: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0, d1)>
//       CHECK: #[[$map2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//       CHECK: module {
// CHECK-LABEL: @conv2d_fully_seperable
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x64x96x128xf16>) -> tensor<1x64x96x128xf16> {
//       CHECK:     %[[cst_f16:.+]] = tensorrt.constant dense<1.0{{.*}}> : tensor<5x5x1x128xf16>
//       CHECK:     %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]] : tensor<1x64x96x128xf16> to tensor<1x128x64x96xf16>
//       CHECK:     %[[w:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[cst_f16]] : tensor<5x5x1x128xf16> to tensor<128x1x5x5xf16>
//       CHECK:     %[[v1:.+]] = tensorrt.convolution {
//  CHECK-SAME:        dilation = array<i64: 1, 1>
//  CHECK-SAME:        num_groups = 128 : ui32
//  CHECK-SAME:        post_padding = array<i64: 2, 2>
//  CHECK-SAME:        pre_padding = array<i64: 2, 2>
//  CHECK-SAME:        stride = array<i64: 1, 1>
//  CHECK-SAME:        in(%[[v0]] : tensor<1x128x64x96xf16>)
//  CHECK-SAME:        kernel(%[[w]] : tensor<128x1x5x5xf16>) -> tensor<1x128x64x96xf16>
//       CHECK:     %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map2]]} %[[v1]] : tensor<1x128x64x96xf16> to tensor<1x64x96x128xf16>
//       CHECK:     return %[[v2]] : tensor<1x64x96x128xf16>

// -----

func.func @conv2d_partially_seperable(%arg0: tensor<1x64x96x128xf16>) -> tensor<1x64x96x128xf16> {
  %0 = stablehlo.constant dense<1.0> : tensor<5x5x2x128xf16>
  %1 = stablehlo.convolution(%arg0, %0)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1],
    pad = [[2, 2], [2, 2]],
    rhs_dilate = [1, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 64 : i64
    } : (tensor<1x64x96x128xf16>, tensor<5x5x2x128xf16>) -> tensor<1x64x96x128xf16>
  return %1 : tensor<1x64x96x128xf16>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
//       CHECK: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0, d1)>
//       CHECK: #[[$map2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//       CHECK: module {
// CHECK-LABEL: @conv2d_partially_seperable
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x64x96x128xf16>) -> tensor<1x64x96x128xf16> {
//       CHECK:     %[[cst_f16:.+]] = tensorrt.constant dense<1.0{{.*}}> :
//       CHECK:     %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]] : tensor<1x64x96x128xf16> to tensor<1x128x64x96xf16>
//       CHECK:     %[[w:.+]] = tensorrt.transpose {permutation = #[[$map1]]} %[[cst_f16]] : tensor<5x5x2x128xf16> to tensor<128x2x5x5xf16>
//       CHECK:     %[[v1:.+]] = tensorrt.convolution
//  CHECK-SAME:        dilation = array<i64: 1, 1>
//  CHECK-SAME:        num_groups = 64 : ui32
//  CHECK-SAME:        post_padding = array<i64: 2, 2>
//  CHECK-SAME:        pre_padding = array<i64: 2, 2>
//  CHECK-SAME:        stride = array<i64: 1, 1>
//  CHECK-SAME:        in(%[[v0]] : tensor<1x128x64x96xf16>)
//  CHECK-SAME:        kernel(%[[w]] : tensor<128x2x5x5xf16>) -> tensor<1x128x64x96xf16>
//       CHECK:     %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map2]]} %[[v1]] : tensor<1x128x64x96xf16> to tensor<1x64x96x128xf16>
//       CHECK:     return %[[v2]] : tensor<1x64x96x128xf16>
