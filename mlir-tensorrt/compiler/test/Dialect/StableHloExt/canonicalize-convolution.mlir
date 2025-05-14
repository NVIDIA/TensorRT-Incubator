// RUN: mlir-tensorrt-opt %s -stablehlo-ext-canonicalize-convolution -split-input-file | FileCheck %s

func.func @conv2d_nhwc_rsck_no_padding_dilated(
    %arg0: tensor<1x32x64x2xf32>,
    %arg1: tensor<3x3x2x128xf32>)
  -> tensor<1x28x62x128xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [2, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<1x32x64x2xf32>, tensor<3x3x2x128xf32>) -> tensor<1x28x62x128xf32>
  func.return %0 : tensor<1x28x62x128xf32>
}

// CHECK-LABEL: @conv2d_nhwc_rsck_no_padding_dilated
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x32x64x2xf32>, %[[arg1:.+]]: tensor<3x3x2x128xf32>) -> tensor<1x28x62x128xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 3, 1, 2] : (tensor<1x32x64x2xf32>) -> tensor<1x2x32x64xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.transpose %[[arg1]], dims = [3, 2, 0, 1] : (tensor<3x3x2x128xf32>) -> tensor<128x2x3x3xf32>
//       CHECK:     %[[v2:.+]] = stablehlo.convolution(%[[v0]], %[[v1]])
//  CHECK-SAME:        dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
//  CHECK-SAME:        window = {stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]], rhs_dilate = [2, 1]}
//  CHECK-SAME:        {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<128x2x3x3xf32>) -> tensor<1x128x28x62xf32>
//       CHECK:     %[[v3:.+]] = stablehlo.transpose %[[v2]], dims = [0, 2, 3, 1] : (tensor<1x128x28x62xf32>) -> tensor<1x28x62x128xf32>
//       CHECK:     return %[[v3]] : tensor<1x28x62x128xf32>

// -----

func.func @conv2d_nchw_kcrs_padded(
    %arg0: tensor<1x2x32x64xf32>,
    %arg1: tensor<128x2x3x3xf32>)
  -> tensor<1x128x28x62xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], pad=[[0, 0], [0, 0]], rhs_dilate = [2, 1]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<1x2x32x64xf32>, tensor<128x2x3x3xf32>)
    -> tensor<1x128x28x62xf32>
  func.return %0 : tensor<1x128x28x62xf32>
}

// CHECK-LABEL: @conv2d_nchw_kcrs_padded
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x2x32x64xf32>, %[[arg1:.+]]: tensor<128x2x3x3xf32>) -> tensor<1x128x28x62xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.convolution(%[[arg0]], %[[arg1]]) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
//  CHECK-SAME:       window = {stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]], rhs_dilate = [2, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
//  CHECK-SAME:       : (tensor<1x2x32x64xf32>, tensor<128x2x3x3xf32>) -> tensor<1x128x28x62xf32>
//       CHECK:     return %[[v0]] : tensor<1x128x28x62xf32>

// -----

func.func @conv1d_nhc_rcf(
    %arg0: tensor<1x32x2xf32>,
    %arg1: tensor<3x2x128xf32>)
  -> tensor<1x28x128xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
    window = {stride = [1], pad = [[0, 0]], rhs_dilate = [2]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<1x32x2xf32>, tensor<3x2x128xf32>) -> tensor<1x28x128xf32>
  func.return %0 : tensor<1x28x128xf32>
}

// CHECK-LABEL: @conv1d_nhc_rcf
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x32x2xf32>, %[[arg1:.+]]: tensor<3x2x128xf32>) -> tensor<1x28x128xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 2, 1] : (tensor<1x32x2xf32>) -> tensor<1x2x32xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.transpose %[[arg1]], dims = [2, 1, 0] : (tensor<3x2x128xf32>) -> tensor<128x2x3xf32>
//       CHECK:     %[[v2:.+]] = stablehlo.reshape %[[v0]] : (tensor<1x2x32xf32>) -> tensor<1x2x1x32xf32>
//       CHECK:     %[[v3:.+]] = stablehlo.reshape %[[v1]] : (tensor<128x2x3xf32>) -> tensor<128x2x1x3xf32>
//       CHECK:     %[[v4:.+]] = stablehlo.convolution(%[[v2]], %[[v3]]) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
//  CHECK-SAME:       window = {stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]], rhs_dilate = [1, 2]}
//  CHECK-SAME:       {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
//  CHECK-SAME:       : (tensor<1x2x1x32xf32>, tensor<128x2x1x3xf32>) -> tensor<1x128x1x28xf32>
//       CHECK:     %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<1x128x1x28xf32>) -> tensor<1x128x28xf32>
//       CHECK:     %[[v6:.+]] = stablehlo.transpose %[[v5]], dims = [0, 2, 1] : (tensor<1x128x28xf32>) -> tensor<1x28x128xf32>
//       CHECK:     return %[[v6]] : tensor<1x28x128xf32>

// -----


func.func public @conv1d_whisper_jax(%arg0: tensor<1x3000x80xf32>, %arg1: tensor<3x80x384xf32>) -> tensor<1x3000x384xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
    window = {stride = [1], pad = [[1, 1]], lhs_dilate = [1], rhs_dilate = [1], reverse = [0]} {
      batch_group_count = 1 : i64, feature_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<1x3000x80xf32>, tensor<3x80x384xf32>) -> tensor<1x3000x384xf32>
  return %0 : tensor<1x3000x384xf32>
}

// CHECK-LABEL: @conv1d_whisper_jax
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x3000x80xf32>, %[[arg1:.+]]: tensor<3x80x384xf32>) -> tensor<1x3000x384xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 2, 1] : (tensor<1x3000x80xf32>) -> tensor<1x80x3000xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.transpose %[[arg1]], dims = [2, 1, 0] : (tensor<3x80x384xf32>) -> tensor<384x80x3xf32>
//       CHECK:     %[[v2:.+]] = stablehlo.reshape %[[v0]] : (tensor<1x80x3000xf32>) -> tensor<1x80x1x3000xf32>
//       CHECK:     %[[v3:.+]] = stablehlo.reshape %[[v1]] : (tensor<384x80x3xf32>) -> tensor<384x80x1x3xf32>
//       CHECK:     %[[v4:.+]] = stablehlo.convolution(%[[v2]], %[[v3]]) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
//  CHECK-SAME:       window = {stride = [1, 1], pad = {{\[}}[0, 0], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]}
//  CHECK-SAME:       {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x80x1x3000xf32>, tensor<384x80x1x3xf32>) -> tensor<1x384x1x3000xf32>
//       CHECK:     %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<1x384x1x3000xf32>) -> tensor<1x384x3000xf32>
//       CHECK:     %[[v6:.+]] = stablehlo.transpose %[[v5]], dims = [0, 2, 1] : (tensor<1x384x3000xf32>) -> tensor<1x3000x384xf32>
//       CHECK:     return %[[v6]] : tensor<1x3000x384xf32>


// -----

func.func @conv3d_ncdhw_kcdrs_padded(
    %arg0: tensor<1x2x32x64x32xf32>,
    %arg1: tensor<128x2x3x3x3xf32>)
  -> tensor<1x128x28x62x30xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1, 2]x[o, i, 0, 1, 2]->[b, f, 0, 1, 2],
    window = {stride = [1, 1, 1], pad=[[0, 0], [0, 0], [0, 0]], rhs_dilate = [2, 1, 1]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<1x2x32x64x32xf32>, tensor<128x2x3x3x3xf32>)
    -> tensor<1x128x28x62x30xf32>
  func.return %0 : tensor<1x128x28x62x30xf32>
}

// CHECK-LABEL: @conv3d_ncdhw_kcdrs_padded
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x2x32x64x32xf32>, %[[arg1:.+]]: tensor<128x2x3x3x3xf32>) -> tensor<1x128x28x62x30xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.convolution(%[[arg0]], %[[arg1]]) dim_numbers = [b, f, 0, 1, 2]x[o, i, 0, 1, 2]->[b, f, 0, 1, 2],
//  CHECK-SAME:       window = {stride = [1, 1, 1], pad = {{\[}}[0, 0], [0, 0], [0, 0]], rhs_dilate = [2, 1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
//  CHECK-SAME:       : (tensor<1x2x32x64x32xf32>, tensor<128x2x3x3x3xf32>) -> tensor<1x128x28x62x30xf32>
//       CHECK:     return %[[v0]] : tensor<1x128x28x62x30xf32>

// -----

func.func @conv3d_ndhwc_drsck_no_padding_dilated(
    %arg0: tensor<1x32x64x32x2xf32>,
    %arg1: tensor<3x3x3x2x128xf32>)
  -> tensor<1x28x62x30x128xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f],
    window = {stride = [1, 1, 1], pad = [[0, 0], [0, 0], [0, 0]], rhs_dilate = [2, 1, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<1x32x64x32x2xf32>, tensor<3x3x3x2x128xf32>) -> tensor<1x28x62x30x128xf32>
  func.return %0 : tensor<1x28x62x30x128xf32>
}

// CHECK-LABEL: @conv3d_ndhwc_drsck_no_padding_dilated
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x32x64x32x2xf32>, %[[arg1:.+]]: tensor<3x3x3x2x128xf32>) -> tensor<1x28x62x30x128xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 4, 1, 2, 3] : (tensor<1x32x64x32x2xf32>) -> tensor<1x2x32x64x32xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.transpose %[[arg1]], dims = [4, 3, 0, 1, 2] : (tensor<3x3x3x2x128xf32>) -> tensor<128x2x3x3x3xf32>
//       CHECK:     %[[v2:.+]] = stablehlo.convolution(%[[v0]], %[[v1]])
//  CHECK-SAME:        dim_numbers = [b, f, 0, 1, 2]x[o, i, 0, 1, 2]->[b, f, 0, 1, 2],
//  CHECK-SAME:        window = {stride = [1, 1, 1], pad = {{\[}}[0, 0], [0, 0], [0, 0]], rhs_dilate = [2, 1, 1]}
//  CHECK-SAME:        {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x32x64x32xf32>, tensor<128x2x3x3x3xf32>) -> tensor<1x128x28x62x30xf32>
//       CHECK:     %[[v3:.+]] = stablehlo.transpose %[[v2]], dims = [0, 2, 3, 4, 1] : (tensor<1x128x28x62x30xf32>) -> tensor<1x28x62x30x128xf32>
//       CHECK:     return %[[v3]] : tensor<1x28x62x30x128xf32>

// -----

// TODO: the pattern should be updated to handle this case

func.func @conv2d_permuted_spatial_dims(%arg0: tensor<4x7x2x6xbf16>,
              %arg1: tensor<5x2x3x4xbf16>) -> tensor<2x2x5x5xbf16> {
  %0 = stablehlo.convolution(%arg0, %arg1)
     dim_numbers = [f, 0, b, 1]x[o, 0, 1, i]->[1, b, f, 0],
     window = {stride = [1, 2], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [2, 1],
      reverse = [false, false]}
      {
       batch_group_count = 1 : i64, feature_group_count = 1 : i64,
       precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision HIGHEST>]
      } : (tensor<4x7x2x6xbf16>, tensor<5x2x3x4xbf16>) -> tensor<2x2x5x5xbf16>
  return %0 : tensor<2x2x5x5xbf16>
}

// CHECK-LABEL: func.func @conv2d_permuted_spatial_dims
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4x7x2x6xbf16>, %[[arg1:.+]]: tensor<5x2x3x4xbf16>) -> tensor<2x2x5x5xbf16> {
//       CHECK:     %[[v0:.+]] = stablehlo.convolution(%[[arg0]], %[[arg1]])
//  CHECK-SAME:      dim_numbers = [f, 0, b, 1]x[o, 0, 1, i]->[1, b, f, 0]
//       CHECK:     return %[[v0]] : tensor<2x2x5x5xbf16>

// -----

/// Convolution without spatial dimensions is `stablehlo.dot`.
func.func @no_spatial_dims(%arg0: tensor<10x5xf32>, %arg1: tensor<5x7xf32>)
    -> tensor<10x7xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f]x[i, o]->[b, f], 
    window = {
      stride = [], 
      pad = [], 
      lhs_dilate = [], 
      rhs_dilate = [], 
      reverse = []
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64, 
       precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<10x5xf32>, tensor<5x7xf32>) -> tensor<10x7xf32>
  return %0 : tensor<10x7xf32>
}

// CHECK-LABEL: func.func @no_spatial_dims
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x5xf32>, %[[arg1:.+]]: tensor<5x7xf32>) -> tensor<10x7xf32>
//       CHECK:     %[[v0:.+]] = stablehlo.dot_general %[[arg0]], %[[arg1]], contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<10x5xf32>, tensor<5x7xf32>) -> tensor<10x7xf32>
//       CHECK:     return %[[v0]] : tensor<10x7xf32>

// -----

/// Convolution without spatial dimensions is `stablehlo.dot`.
func.func @reversed_no_spatial_dims(%arg0: tensor<5x10xf32>, %arg1: tensor<7x5xf32>)
    -> tensor<10x7xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [f, b]x[o, i]->[b, f], 
    window = {
      stride = [], 
      pad = [], 
      lhs_dilate = [], 
      rhs_dilate = [], 
      reverse = []
    } {batch_group_count = 1 : i64, feature_group_count = 1 : i64, 
       precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<5x10xf32>, tensor<7x5xf32>) -> tensor<10x7xf32>
  return %0 : tensor<10x7xf32>
}


// CHECK-LABEL: func.func @reversed_no_spatial_dims
//  CHECK-SAME: (%[[arg0:.+]]: tensor<5x10xf32>, %[[arg1:.+]]: tensor<7x5xf32>)
//       CHECK:     %[[v0:.+]] = stablehlo.dot_general %[[arg0]], %[[arg1]], contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x10xf32>, tensor<7x5xf32>) -> tensor<10x7xf32>
//       CHECK:     return %[[v0]] : tensor<10x7xf32>
