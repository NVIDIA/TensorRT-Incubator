// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-ext-constant-folding | FileCheck %s

func.func @transpose_fold_splat() -> tensor<2x4xf32> {
  %0 = stablehlo.constant dense<2.0> : tensor<4x2xf32>
  %1 = "stablehlo.transpose"(%0) {permutation = array<i64: 1, 0>} :
    (tensor<4x2xf32>) -> tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}

// CHECK-LABEL: @transpose_fold_splat
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<2x4xf32>
//       CHECK:     return %[[v0]] : tensor<2x4xf32>

// -----

func.func @transpose_fold_2d() -> tensor<3x2xf32> {
  %0 = stablehlo.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  %1 = "stablehlo.transpose"(%0) {permutation = array<i64: 1, 0>} :
    (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// CHECK-LABEL: @transpose_fold_2d
//       CHECK:     %[[v0:.+]] = stablehlo.constant
//  CHECK-SAME: dense<{{\[}}[0.000000e+00, 3.000000e+00],
//  CHECK-SAME: [1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00]{{\]}}> : tensor<3x2xf32>
//       CHECK:     return %[[v0]] : tensor<3x2xf32>

// -----

func.func @transpose_fold_2d_requires_cast() -> tensor<?x?xf32> {
  %0 = stablehlo.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  %1 = "stablehlo.transpose"(%0) {permutation = array<i64: 1, 0>} :
    (tensor<2x3xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @transpose_fold_2d_requires_cast
//       CHECK:     %[[v0:.+]] = stablehlo.constant
//  CHECK-SAME: dense<{{\[}}[0.000000e+00, 3.000000e+00],
//  CHECK-SAME: [1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00]{{\]}}> : tensor<3x2xf32>
//       CHECK:     %[[cast:.+]] = tensor.cast %[[v0]] : tensor<3x2xf32> to tensor<?x?xf32>
//       CHECK:     return %[[cast]] : tensor<?x?xf32>

// -----

func.func @transpose_fold_4d_i16() -> tensor<3x1x4x2xi16> {
  %0 = stablehlo.constant dense<[[
    [[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]],
    [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
  ]]> : tensor<1x2x3x4xi16>
  %1 = "stablehlo.transpose"(%0) {permutation = array<i64: 2, 0, 3, 1>} :
    (tensor<1x2x3x4xi16>) -> tensor<3x1x4x2xi16>
  return %1 : tensor<3x1x4x2xi16>
}

// CHECK-LABEL: @transpose_fold_4d_i16
//       CHECK:     %[[v0:.+]] = stablehlo.constant
//  CHECK-SAME:       {{\[\[\[}}
//  CHECK-SAME:       [0, 12], [1, 13], [2, 14], [3, 15]{{\]\]}},
//  CHECK-SAME:       {{\[\[}}
//  CHECK-SAME:       [4, 16], [5, 17], [6, 18], [7, 19]{{\]\]}},
//  CHECK-SAME:       {{\[\[}}
//  CHECK-SAME:       [8, 20], [9, 21], [10, 22], [11, 23]{{\]\]\]}}> : tensor<3x1x4x2xi16>
//       CHECK:     return %[[v0]] : tensor<3x1x4x2xi16>

// -----

func.func @convert_folder() -> (tensor<4xi32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf32>) {
  %0 = stablehlo.constant dense<[1., 2., 3., 4.]> : tensor<4xf32>
  %1 = stablehlo.constant dense<[1., 2., 3., 4.]> : tensor<4xf16>
  %ci32 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %2 = stablehlo.convert %0 : (tensor<4xf32>) -> tensor<4xf16>
  %3 = stablehlo.convert %1 : (tensor<4xf16>) -> tensor<4xf32>
  %4 = stablehlo.convert %1 : (tensor<4xf16>) -> tensor<4xi32>
  %5 = stablehlo.convert %ci32 : (tensor<4xi32>) -> tensor<4xf32>
  return %4, %3, %2, %5 : tensor<4xi32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf32>
}

// CHECK-LABEL: @convert_folder
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf16>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
//   CHECK-DAG:     %[[v2:.+]] = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
//       CHECK:     return %[[v2]], %[[v1]], %[[v0]], %[[v1]] : tensor<4xi32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf32>

// -----

func.func @convert_int_to_float_negative() -> tensor<2xf32> {
  %c = stablehlo.constant dense<[-2, -1]> : tensor<2xi32>
  %0 = stablehlo.convert %c : (tensor<2xi32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @convert_int_to_float_negative
//       CHECK:     %[[cst:.+]] = stablehlo.constant dense<[-2.000000e+00, -1.000000e+00]> :
//       CHECK:     return %[[cst]] : tensor<2xf32>

// -----

func.func @fold_multi_user_non_splat(%arg0: tensor<2x3xf32>) -> (tensor<3x2xf32>, tensor<2x3xf32>) {
  %0 = stablehlo.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  %1 = "stablehlo.transpose"(%0) {permutation = array<i64: 1, 0>} :
    (tensor<2x3xf32>) -> tensor<3x2xf32>
  %2 = stablehlo.add %arg0, %0 : tensor<2x3xf32>
  return %1, %2 : tensor<3x2xf32>, tensor<2x3xf32>
}

// CHECK-LABEL: @fold_multi_user_non_splat
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3xf32>)
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.constant dense<{{\[\[}}0.000000e+00, 3.000000e+00],
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.constant dense<{{\[\[}}0.000000e+00, 1.000000e+00,
//       CHECK:     %[[v2:.+]] = stablehlo.add %[[arg0]], %[[v1]]
//       CHECK:     return %[[v0]], %[[v2]]

// -----

func.func @fp32_to_fp16_splat_convert() -> (tensor<f16>, tensor<f16>) {
    %0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %2 = stablehlo.convert %0 : (tensor<f32>) -> tensor<f16>
    %3 = stablehlo.convert %1 : (tensor<f32>) -> tensor<f16>
    return %2, %3 : tensor<f16>, tensor<f16>
}

// CHECK-LABEL: @fp32_to_fp16_splat_convert
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<{{.*}}> : tensor<f16>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.constant dense<{{.*}}> : tensor<f16>
//  CHECK-NEXT: return %[[v0]], %[[v1]] : tensor<f16>, tensor<f16>

// -----

func.func @convert_folder_dynamic() -> tensor<?xf16> {
    %0 = stablehlo.constant dense<9.99999974E-6> : tensor<4xf32>
    %1 = stablehlo.convert %0 : (tensor<4xf32>) -> tensor<?xf16>
    return %1 : tensor<?xf16>
}

// CHECK-LABEL: func.func @convert_folder_dynamic
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<1.001360e-05> : tensor<4xf16>
//       CHECK:     %[[cast:.+]] = tensor.cast %[[v0]] : tensor<4xf16> to tensor<?xf16>
//       CHECK:     return %[[cast]] : tensor<?xf16>

// -----

func.func @convert_fold_limit() -> (tensor<262144xf16>) {
    %0 = stablehlo.constant dense_resource<__elided__> : tensor<262144xf32>
    %1 = stablehlo.convert %0 : (tensor<262144xf32>) -> tensor<262144xf16>
    return %1 : tensor<262144xf16>
}

// CHECK-LABEL: func.func @convert_fold_limit
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<262144xf32>
//  CHECK-NEXT:     %[[v1:.+]] = stablehlo.convert %[[v0]] : (tensor<262144xf32>) -> tensor<262144xf16>
//  CHECK-NEXT:     return %[[v1]] : tensor<262144xf16>

// -----

func.func @slice_const_fold_zero_size() -> tensor<?x?xi64> {
  %0 = stablehlo.constant dense<[[0, 1, 2, 3],
                            [4, 5, 6, 7],
                            [8, 9, 10, 11],
                            [12, 13, 14, 15]]> : tensor<4x4xi64>
  %1 = stablehlo.slice %0 [1:1:1, 2:2:1] : (tensor<4x4xi64>) -> (tensor<?x?xi64>)
  func.return %1 : tensor<?x?xi64>
}

// CHECK-LABEL: func.func @slice_const_fold_zero_size
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<0x0xi64>
//       CHECK:     %[[cast:.+]] = tensor.cast %[[v0]] : tensor<0x0xi64> to tensor<?x?xi64>
//       CHECK:     return %[[cast]] : tensor<?x?xi64>

// -----

func.func @slice_const_fold_i64() -> tensor<2x2xi64> {
  %0 = stablehlo.constant dense<[[0, 1, 2, 3],
                            [4, 5, 6, 7],
                            [8, 9, 10, 11],
                            [12, 13, 14, 15]]> : tensor<4x4xi64>
  %1 = stablehlo.slice %0 [1:3:1, 2:4:1] : (tensor<4x4xi64>) -> (tensor<2x2xi64>)
  func.return %1 : tensor<2x2xi64>
}

// CHECK-LABEL: @slice_const_fold_i64
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<{{\[}}[6, 7], [10, 11]]
//       CHECK:     return %[[v0]] : tensor<2x2xi64>

// -----

func.func @slice_fold_limit() -> tensor<262144xi64> {
  %0 = stablehlo.constant dense_resource<__elided__> : tensor<2621440xi64>
  %1 = stablehlo.slice %0 [0:262144:1] : (tensor<2621440xi64>) -> (tensor<262144xi64>)
  func.return %1 : tensor<262144xi64>
}

// CHECK-LABEL: func.func @slice_fold_limit
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<2621440xi64>
//  CHECK-NEXT:     %[[v1:.+]] = stablehlo.slice %[[v0]] [0:262144] : (tensor<2621440xi64>) -> tensor<262144xi64>
//  CHECK-NEXT:     return %[[v1]] : tensor<262144xi64>

// -----

func.func @slice_const_fold_i4() -> tensor<2x2xi4> {
  %0 = stablehlo.constant dense<[[0, 1, 2, 3],
                              [-1, 0, -3, -2],
                              [2, 1, 0, -3],
                              [3, -1, -3, 0]]> : tensor<4x4xi4>
  %1 = stablehlo.slice %0 [1:3:1, 2:4:1] : (tensor<4x4xi4>) -> (tensor<2x2xi4>)
  func.return %1 : tensor<2x2xi4>
}

// CHECK-LABEL: func.func @slice_const_fold_i4
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<{{\[}}[-3, -2], [0, -3]]> : tensor<2x2xi4>
//       CHECK:     return %[[v0]] : tensor<2x2xi4>

// -----

func.func @slice_const_fold_f32() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[0., 1., 2., 3.],
                                 [4., 5., 6., 7.],
                                 [8., 9., 10., 11.],
                                 [12., 13., 14., 15.]]> : tensor<4x4xf32>
  %1 = stablehlo.slice %0 [1:3:1, 2:4:1] : (tensor<4x4xf32>) -> (tensor<2x2xf32>)
  func.return %1 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @slice_const_fold_f32
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<{{\[}}[6.000000e+00, 7.000000e+00], [1.000000e+01, 1.100000e+01]]> : tensor<2x2xf32>
//       CHECK:     return %[[v0]] : tensor<2x2xf32>

// -----

func.func @slice_const_fold_f16() -> tensor<2x2xf16> {
  %0 = stablehlo.constant dense<[[0., 1., 2., 3.],
                                 [4., 5., 6., 7.],
                                 [8., 9., 10., 11.],
                                 [12., 13., 14., 15.]]> : tensor<4x4xf16>
  %1 = stablehlo.slice %0 [1:3:1, 2:4:1] : (tensor<4x4xf16>) -> (tensor<2x2xf16>)
  func.return %1 : tensor<2x2xf16>
}

// CHECK-LABEL: func.func @slice_const_fold_f16
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<{{\[}}[6.000000e+00, 7.000000e+00], [1.000000e+01, 1.100000e+01]]> : tensor<2x2xf16>
//       CHECK:     return %[[v0]] : tensor<2x2xf16>

// -----

func.func @slice_const_fold_f8() -> tensor<2x2xf8E4M3FN> {
  %0 = stablehlo.constant dense<[[0., 1., 2., 3.],
                                 [4., 5., 6., 7.],
                                 [8., 9., 10., 11.],
                                 [12., 13., 14., 15.]]> : tensor<4x4xf8E4M3FN>
  %1 = stablehlo.slice %0 [1:3:1, 2:4:1] : (tensor<4x4xf8E4M3FN>) -> (tensor<2x2xf8E4M3FN>)
  func.return %1 : tensor<2x2xf8E4M3FN>
}

// CHECK-LABEL: func.func @slice_const_fold_f8
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<{{\[}}[6.000000e+00, 7.000000e+00], [1.000000e+01, 1.100000e+01]]> : tensor<2x2xf8E4M3FN>
//       CHECK:     return %[[v0]] : tensor<2x2xf8E4M3FN>


// -----

func.func @slice_const_fold_unsupported() -> tensor<2xcomplex<f32>> {
  %0 = stablehlo.constant dense<(0.0, 1.0)> : tensor<10xcomplex<f32>>
  %1 = stablehlo.slice %0 [1:3:1] : (tensor<10xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)
  func.return %1 : tensor<2xcomplex<f32>>
}

// CHECK-LABEL: @slice_const_fold_unsupported
//   CHECK-NEXT: %[[const:.+]] = stablehlo.constant
//   CHECK-NEXT: %[[slice:.+]] = stablehlo.slice %[[const]]
//   CHECK-NEXT: return %[[slice]] :

// -----

func.func @slice_const_fold_zero_size() -> tensor<?x?xi64> {
  %0 = stablehlo.constant dense<[[0, 1, 2, 3],
                            [4, 5, 6, 7],
                            [8, 9, 10, 11],
                            [12, 13, 14, 15]]> : tensor<4x4xi64>
  %1 = stablehlo.slice %0 [1:1:1, 2:2:1] : (tensor<4x4xi64>) -> (tensor<?x?xi64>)
  func.return %1 : tensor<?x?xi64>
}

// CHECK-LABEL: func.func @slice_const_fold_zero_size
//  CHECK-SAME: () -> tensor<?x?xi64> {
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<0x0xi64>
//       CHECK:     %[[cast:.+]] = tensor.cast %[[v0]] : tensor<0x0xi64> to tensor<?x?xi64>
//       CHECK:     return %[[cast]] : tensor<?x?xi64>

// -----

func.func @slice_const_fold_type_erased() -> tensor<?x?xi64> {
  %0 = stablehlo.constant dense<[[0, 1, 2, 3],
                            [4, 5, 6, 7],
                            [8, 9, 10, 11],
                            [12, 13, 14, 15]]> : tensor<4x4xi64>
  %1 = stablehlo.slice %0 [1:3:1, 2:4:1] : (tensor<4x4xi64>) -> (tensor<?x?xi64>)
  func.return %1 : tensor<?x?xi64>
}

// CHECK-LABEL: @slice_const_fold_type_erased
//  CHECK-SAME: () -> tensor<?x?xi64> {
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<[{{\[}}6, 7], [10, 11]]>
//       CHECK:     %[[cast:.+]] = tensor.cast %[[v0]] : tensor<2x2xi64> to tensor<?x?xi64>
//       CHECK:     return %[[cast]] : tensor<?x?xi64>

// -----

func.func @slice_const_fold_strided() -> tensor<2x2xi64> {
  %0 = stablehlo.constant dense<[[0, 1, 2, 3],
                            [4, 5, 6, 7],
                            [8, 9, 10, 11],
                            [12, 13, 14, 15]]> : tensor<4x4xi64>
  %1 = stablehlo.slice %0 [1:4:2, 1:4:2] : (tensor<4x4xi64>) -> (tensor<2x2xi64>)
  func.return %1 : tensor<2x2xi64>
}

// CHECK-LABEL: @slice_const_fold_strided
//  CHECK-SAME: () -> tensor<2x2xi64> {
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<{{\[}}[5, 7], [13, 15]]>
//       CHECK:     return %[[v0]] : tensor<2x2xi64>

// -----

func.func @slice_const_fold_1d() -> tensor<1xi32> {
  %0 = stablehlo.constant dense<[4, 6]> : tensor<2xi32>
  %3 = stablehlo.slice %0 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
  return %3 : tensor<1xi32>
}

// CHECK-LABEL: func.func @slice_const_fold_1d
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<6> : tensor<1xi32>
//       CHECK:     return %[[v0]] : tensor<1xi32>

// -----

func.func @slice_const_fold_splat() -> tensor<1x12xi32> {
  %0 = stablehlo.constant dense<1> : tensor<12x12xi32>
  %3 = stablehlo.slice %0 [0:1, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
  return %3 : tensor<1x12xi32>
}

// CHECK-LABEL: func.func @slice_const_fold_splat
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.constant dense<1> : tensor<1x12xi32>
//  CHECK-NEXT:     return %[[v0]] : tensor<1x12xi32>

// -----

// Constant folding for concat is tested upstream in
// 'stablehlo/tests/stablehlo_aggressive_simplification.mlir'. This
// pattern just tests that we can absorb `tensor.cast` operations.

func.func @concat_absorb_cast() -> tensor<?xi32> {
  %0 = stablehlo.constant dense<[6]> : tensor<1xi32>
  %1 = stablehlo.constant dense<1> : tensor<1xi32>
  %2 = tensor.cast %0 : tensor<1xi32> to tensor<?xi32>
  %3 = stablehlo.concatenate %2, %0, dim = 0 : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  return %3 : tensor<?xi32>
}

// CHECK-LABEL: func.func @concat_absorb_cast
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.constant dense<6> : tensor<1xi32>
//  CHECK-NEXT:     %[[v1:.+]] = stablehlo.concatenate %[[v0]], %[[v0]], dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
//  CHECK-NEXT:     return %[[v1]] : tensor<?xi32>


// -----

func.func @concat_remove_zero_extent_segments(%arg0: tensor<?xi32>, %arg1: tensor<0xi32>) -> tensor<?xi32> {
  %0 = stablehlo.concatenate %arg0, %arg0, %arg1, dim = 0 : (tensor<?xi32>, tensor<?xi32>, tensor<0xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @concat_remove_zero_extent_segments
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xi32>, %[[arg1:.+]]: tensor<0xi32>)
//       CHECK:     %[[v0:.+]] = stablehlo.concatenate %[[arg0]], %[[arg0]], dim = 0 : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
//       CHECK:     return %[[v0]] : tensor<?xi32>

// -----

func.func @concat_simplify_single_operand(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = stablehlo.concatenate %arg0, dim = 0 : (tensor<?xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @concat_simplify_single_operand
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xi32>)
//  CHECK-NEXT:     return %[[arg0]]

// -----

func.func @concat_simplify_single_operand_requires_cast(%arg0: tensor<4xi32>) -> tensor<?xi32> {
  %0 = stablehlo.concatenate %arg0, dim = 0 : (tensor<4xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @concat_simplify_single_operand_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>)
//  CHECK-NEXT:     %[[cast:.+]] = tensor.cast %[[arg0]] : tensor<4xi32> to tensor<?xi32>
//  CHECK-NEXT:     return %[[cast]]

// -----

func.func @bitwise_or_fold_lhs(%arg0: tensor<5xi8>, %arg1: tensor<5xi1>, %arg2: tensor<5xi32>) -> (tensor<5xi8>, tensor<5xi1>, tensor<5xi32>, tensor<5xi32>){
    %0 = stablehlo.constant dense<[255, 255, 255, 255, 255]> : tensor<5xi8>
    %1 = stablehlo.or %0, %arg0 : tensor<5xi8>
    %2 = stablehlo.constant dense<[1, 1, 1, 1, 1]> : tensor<5xi1>
    %3 = stablehlo.or %2, %arg1 : tensor<5xi1>
    %4 = stablehlo.constant dense<[4294967295, 4294967295, 4294967295, 4294967295, 4294967295]> : tensor<5xi32>
    %5 = stablehlo.or %4, %arg2 : tensor<5xi32>
    %6 = stablehlo.constant dense<[0, 0, 0, 0, 0]> : tensor<5xi32>
    %7 = stablehlo.or %arg2, %6 : tensor<5xi32>
    return %1, %3, %5, %7 : tensor<5xi8>, tensor<5xi1>, tensor<5xi32>, tensor<5xi32>
}
// CHECK-LABEL: bitwise_or_fold_lhs
//  CHECK-SAME: (%[[arg0:.+]]: tensor<5xi8>, %[[arg1:.+]]: tensor<5xi1>, %[[arg2:.+]]: tensor<5xi32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<-1>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.constant dense<true>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.constant dense<-1>
//  CHECK-NEXT: return %[[v2]], %[[v1]], %[[v0]], %[[arg2]]

// -----

func.func @bitwise_or_fold_rhs(%arg0: tensor<5xi8>, %arg1: tensor<5xi1>, %arg2: tensor<5xi32>) -> (tensor<5xi8>, tensor<5xi1>, tensor<5xi32>, tensor<5xi32>){
    %0 = stablehlo.constant dense<[255, 255, 255, 255, 255]> : tensor<5xi8>
    %1 = stablehlo.or %arg0, %0 : tensor<5xi8>
    %2 = stablehlo.constant dense<[1, 1, 1, 1, 1]> : tensor<5xi1>
    %3 = stablehlo.or %arg1, %2 : tensor<5xi1>
    %4 = stablehlo.constant dense<[4294967295, 4294967295, 4294967295, 4294967295, 4294967295]> : tensor<5xi32>
    %5 = stablehlo.or %arg2, %4 : tensor<5xi32>
    %6 = stablehlo.constant dense<[0, 0, 0, 0, 0]> : tensor<5xi32>
    %7 = stablehlo.or %arg2, %6 : tensor<5xi32>
    return %1, %3, %5, %7 : tensor<5xi8>, tensor<5xi1>, tensor<5xi32>, tensor<5xi32>
}
// CHECK-LABEL: bitwise_or_fold_rhs
//  CHECK-SAME: (%[[arg0:.+]]: tensor<5xi8>, %[[arg1:.+]]: tensor<5xi1>, %[[arg2:.+]]: tensor<5xi32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<-1>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.constant dense<true>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.constant dense<-1>
//  CHECK-NEXT: return %[[v2]], %[[v1]], %[[v0]], %[[arg2]]


// -----

func.func @bitwise_and_fold_lhs(%arg0: tensor<5xi8>, %arg1: tensor<5xi1>, %arg2: tensor<5xi32>) -> (tensor<5xi8>, tensor<5xi1>, tensor<5xi32>, tensor<5xi32>){
    %0 = stablehlo.constant dense<[255, 255, 255, 255, 255]> : tensor<5xi8>
    %1 = stablehlo.and %0, %arg0 : tensor<5xi8>
    %2 = stablehlo.constant dense<[1, 1, 1, 1, 1]> : tensor<5xi1>
    %3 = stablehlo.and %2, %arg1 : tensor<5xi1>
    %4 = stablehlo.constant dense<[4294967295, 4294967295, 4294967295, 4294967295, 4294967295]> : tensor<5xi32>
    %5 = stablehlo.and %4, %arg2 : tensor<5xi32>
    %6 = stablehlo.constant dense<[0, 0, 0, 0, 0]> : tensor<5xi32>
    %7 = stablehlo.and %arg2, %6 : tensor<5xi32>
    return %1, %3, %5, %7 : tensor<5xi8>, tensor<5xi1>, tensor<5xi32>, tensor<5xi32>
}
// CHECK-LABEL: bitwise_and_fold_lhs
//  CHECK-SAME: (%[[arg0:.+]]: tensor<5xi8>, %[[arg1:.+]]: tensor<5xi1>, %[[arg2:.+]]: tensor<5xi32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<0>
//  CHECK-NEXT: return %[[arg0]], %[[arg1]], %[[arg2]], %[[v0]]

// -----

func.func @bitwise_and_fold_rhs(%arg0: tensor<5xi8>, %arg1: tensor<5xi1>, %arg2: tensor<5xi32>) -> (tensor<5xi8>, tensor<5xi1>, tensor<5xi32>, tensor<5xi32>){
    %0 = stablehlo.constant dense<[255, 255, 255, 255, 255]> : tensor<5xi8>
    %1 = stablehlo.and %arg0, %0 : tensor<5xi8>
    %2 = stablehlo.constant dense<[1, 1, 1, 1, 1]> : tensor<5xi1>
    %3 = stablehlo.and %arg1, %2 : tensor<5xi1>
    %4 = stablehlo.constant dense<[4294967295, 4294967295, 4294967295, 4294967295, 4294967295]> : tensor<5xi32>
    %5 = stablehlo.and %arg2, %4 : tensor<5xi32>
    %6 = stablehlo.constant dense<[0, 0, 0, 0, 0]> : tensor<5xi32>
    %7 = stablehlo.and %arg2, %6 : tensor<5xi32>
    return %1, %3, %5, %7 : tensor<5xi8>, tensor<5xi1>, tensor<5xi32>, tensor<5xi32>
}
// CHECK-LABEL: bitwise_and_fold_rhs
//  CHECK-SAME: (%[[arg0:.+]]: tensor<5xi8>, %[[arg1:.+]]: tensor<5xi1>, %[[arg2:.+]]: tensor<5xi32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<0>
//  CHECK-NEXT: return %[[arg0]], %[[arg1]], %[[arg2]], %[[v0]]

// -----

func.func @bitwise_fold_both() -> (tensor<5xi8>, tensor<5xi8>){
    %0 = stablehlo.constant dense<[10, 12, 13, 45, 8]> : tensor<5xi8>
    %1 = stablehlo.constant dense<[5, 9, 13, 5, 22]> : tensor<5xi8>
    %2 = stablehlo.and %0, %1 : tensor<5xi8>
    %3 = stablehlo.or %0, %1 : tensor<5xi8>
    return %2, %3 : tensor<5xi8>, tensor<5xi8>
}
// CHECK-LABEL: bitwise_fold_both
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<[0, 8, 13, 5, 0]>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.constant dense<[15, 13, 13, 45, 30]>
//  CHECK-NEXT: return %[[v0]], %[[v1]]

// -----

func.func @bitwise_and_or_fold_negative(%arg: tensor<5xi8>) -> tensor<5xi8>{
    %0 = stablehlo.constant dense_resource<__elided__> : tensor<5xi8>
    %1 = stablehlo.constant dense_resource<__elided__> : tensor<5xi8>
    %2 = stablehlo.or %0, %1 : tensor<5xi8>
    %3 = stablehlo.and %2, %arg : tensor<5xi8>
    return %3 : tensor<5xi8>
}
// CHECK-LABEL: bitwise_and_or_fold_negative
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense_resource<__elided__>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.or
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.and
//  CHECK-NEXT: return %[[v2]]

// -----

func.func @bitwise_and_or_fold_negative_2(%arg0: tensor<5xi8>) -> (tensor<5xi8>, tensor<5xi8>){
    %0 = stablehlo.constant dense<[10, 0, 11, 21, 41]> : tensor<5xi8>
    %1 = stablehlo.and %arg0, %0 : tensor<5xi8>
    %2 = stablehlo.constant dense<[14, 1, 26, 1, 4]> : tensor<5xi8>
    %3 = stablehlo.or %arg0, %2 : tensor<5xi8>
    return %1, %3 : tensor<5xi8>, tensor<5xi8>
}
// CHECK-LABEL: bitwise_and_or_fold_negative_2
//  CHECK-NEXT: stablehlo.constant
//  CHECK-NEXT: stablehlo.constant
//  CHECK-NEXT: stablehlo.and
//  CHECK-NEXT: stablehlo.or
//  CHECK-NEXT: return

// -----

func.func @cascaded_convert(%arg0: tensor<i1>) -> tensor<i32>{
    %0 = stablehlo.convert %arg0: (tensor<i1>) -> tensor<i8>
    %1 = stablehlo.convert %0: (tensor<i8>) -> tensor<i32>
    return %1 : tensor<i32>
}
// CHECK-LABEL: cascaded_convert
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i1>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.convert %[[arg0]] : (tensor<i1>) -> tensor<i32>
//  CHECK-NEXT: return %[[v0]] : tensor<i32>

// -----

func.func @cascaded_convert_negative(%arg0: tensor<i8>) -> tensor<i32>{
    %0 = stablehlo.convert %arg0: (tensor<i8>) -> tensor<i1>
    %1 = stablehlo.convert %0: (tensor<i1>) -> tensor<i32>
    return %1 : tensor<i32>
}
// CHECK-LABEL: cascaded_convert_negative
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i8>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.convert
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.convert
//  CHECK-NEXT: return %[[v1]] : tensor<i32>

// -----

func.func @compare_fold_float() -> (tensor<5xi1>, tensor<5xi1>, tensor<5xi1>, tensor<5xi1>, tensor<5xi1>, tensor<5xi1>){
    %const_0 = stablehlo.constant dense<[2.3, 0.0, 1.3, 8.928, 7.1]> : tensor<5xf32>
    %const_1 = stablehlo.constant dense<[2.1, 0.0, 2.3, 8.92889, 4.1]> : tensor<5xf32>
    %0 = stablehlo.compare LT, %const_0, %const_1 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
    %1 = stablehlo.compare EQ, %const_0, %const_1 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
    %2 = stablehlo.compare NE, %const_0, %const_1 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
    %3 = stablehlo.compare LE, %const_0, %const_1 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
    %4 = stablehlo.compare GT, %const_0, %const_1 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
    %5 = stablehlo.compare GE, %const_0, %const_1 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
    return %0, %1, %2, %3, %4, %5 : tensor<5xi1>, tensor<5xi1>, tensor<5xi1>, tensor<5xi1>, tensor<5xi1>, tensor<5xi1>
}

// CHECK-LABEL: compare_fold_float
//  CHECK-NEXT: stablehlo.constant dense<[false, false, true, true, false]>
//  CHECK-NEXT: stablehlo.constant dense<[false, true, false, false, false]>
//  CHECK-NEXT: stablehlo.constant dense<[true, false, true, true, true]>
//  CHECK-NEXT: stablehlo.constant dense<[false, true, true, true, false]>
//  CHECK-NEXT: stablehlo.constant dense<[true, false, false, false, true]>
//  CHECK-NEXT: stablehlo.constant dense<[true, true, false, false, true]>
//  CHECK-NEXT: return

// -----

func.func @compare_fold_float_negative() -> (tensor<5xi1>){
    %0 = stablehlo.constant dense_resource<__elided__> : tensor<5xf32>
    %1 = stablehlo.constant dense_resource<__elided__> : tensor<5xf32>
    %2 = stablehlo.compare LT, %0, %1 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
    return %2 : tensor<5xi1>
}

// CHECK-LABEL: compare_fold_float_negative
//  CHECK-NEXT: stablehlo.constant
//  CHECK-NEXT: stablehlo.compare  LT
//  CHECK-NEXT: return

// -----

func.func @compare_fold_float_big_splat() -> (tensor<1024x1024xi1>){
    %0 = stablehlo.constant dense<0.0> : tensor<1024x1024xf32>
    %1 = stablehlo.constant dense<1.0> : tensor<1024x1024xf32>
    %2 = stablehlo.compare LT, %0, %1 : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xi1>
    return %2 : tensor<1024x1024xi1>
}

// CHECK-LABEL: compare_fold_float_big_splat
//  CHECK-NEXT: %[[c:.+]] = stablehlo.constant dense<true> : tensor<1024x1024xi1>
//  CHECK-NEXT: return %[[c]]

// -----

func.func @canonicalize_iota() -> tensor<18x12xi32>{
    %0 = stablehlo.iota dim = 1 : tensor<18x12xi32>
    return %0 : tensor<18x12xi32>
}

// CHECK-LABEL: canonicalize_iota
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.iota dim = 0
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.broadcast_in_dim %[[v0]], dims = [1]
//  CHECK-NEXT: return %[[v1]]

// -----

func.func @fold_sqrt() -> (tensor<6xf32>, tensor<6xf16>) {
    %0 = stablehlo.constant dense<[2.0, 4.0, 9.0, 16.0, 25.0, 34.0]> : tensor<6xf32>
    %1 = stablehlo.constant dense<[2.0, 4.0, 9.0, 16.0, 25.0, 34.0]> : tensor<6xf16>
    %2 = stablehlo.sqrt %0 : tensor<6xf32>
    %3 = stablehlo.sqrt %1 : tensor<6xf16>
    return %2, %3 : tensor<6xf32>, tensor<6xf16>
}
// CHECK-LABEL: fold_sqrt
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<[1.41421354, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 5.83095169]> : tensor<6xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.constant dense<[1.414060e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 5.832030e+00]> : tensor<6xf16>
//  CHECK-NEXT: return %[[v0]], %[[v1]]

// -----

// test the smallest possible positive number for each data type.
// You can check the result by using double precision to try to
// recover the inverse of rsqrt using NumPy:
//  `1/np.asarray([2.67137384e22, 4096.0, 1.043390e+20, 22.])**2`.
func.func @fold_rsqrt_small() -> (tensor<f32>, tensor<f16>, tensor<bf16>, tensor<f8E4M3FN>) {

    %c = stablehlo.constant dense<1.401300e-45> : tensor<f32>
    %r = stablehlo.rsqrt %c : tensor<f32>

    %c16 = stablehlo.constant dense<5.960460e-08> : tensor<f16>
    %r16 = stablehlo.rsqrt %c16 : tensor<f16>

    %cbf16 = stablehlo.constant dense<9.183550e-41> : tensor<bf16>
    %rbf16 = stablehlo.rsqrt %cbf16 : tensor<bf16>

    %c8 = stablehlo.constant dense<1.953130e-03> : tensor<f8E4M3FN>
    %r8 = stablehlo.rsqrt %c8 : tensor<f8E4M3FN>
    return %r, %r16, %rbf16, %r8 : tensor<f32>, tensor<f16>, tensor<bf16>, tensor<f8E4M3FN>
}

// CHECK-LABEL: @fold_rsqrt_small
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<2.67137384E+22> : tensor<f32>
//       CHECK:     %[[v1:.+]] = stablehlo.constant dense<4.096000e+03> : tensor<f16>
//       CHECK:     %[[v2:.+]] = stablehlo.constant dense<1.043390e+20> : tensor<bf16>
//       CHECK:     %[[v3:.+]] = stablehlo.constant dense<2.200000e+01> : tensor<f8E4M3FN>
//       CHECK:     return %[[v0]], %[[v1]], %[[v2]], %[[v3]] : tensor<f32>, tensor<f16>, tensor<bf16>, tensor<f8E4M3FN>

// -----

// Don't fold negatives
func.func @fold_rsqrt_neg() -> (tensor<f32>) {
  %c = stablehlo.constant dense<-1.401300e-45> : tensor<f32>
  %r = stablehlo.rsqrt %c : tensor<f32>
  return %r : tensor<f32>
}

// CHECK-LABEL: func.func @fold_rsqrt_neg
//       CHECK:     %[[v0:.+]] = stablehlo.constant
//       CHECK:     %[[v1:.+]] = stablehlo.rsqrt %[[v0]] : tensor<f32>
//       CHECK:     return %[[v1]] : tensor<f32>

// -----

func.func @transpose_consecutive(%arg0: tensor<10x2x1xf32>) -> tensor<1x10x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<10x2x1xf32>) -> tensor<10x1x2xf32>
  %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<10x1x2xf32>) -> tensor<1x10x2xf32>
  return %1 : tensor<1x10x2xf32>
}

// CHECK-LABEL: transpose_consecutive
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x2x1xf32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.reshape %[[arg0]]
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @transpose_consecutive_identity(%arg0: tensor<10x2x1xf32>) -> tensor<10x2x1xf32> {
  %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<10x2x1xf32>) -> tensor<10x1x2xf32>
  %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<10x1x2xf32>) -> tensor<10x2x1xf32>
  return %1 : tensor<10x2x1xf32>
}

// CHECK-LABEL: transpose_consecutive
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x2x1xf32>)
//  CHECK-NEXT: return %[[arg0]]

// -----

func.func @broadcast_in_dim_is_reshape(%arg0: tensor<10xf32>) -> (tensor<1x10xf32>, tensor<10x10xf32>) {
  %0 = stablehlo.broadcast_in_dim %arg0, dims=[1] : (tensor<10xf32>) -> tensor<1x10xf32>
  %1 = stablehlo.broadcast_in_dim %arg0, dims=[0] : (tensor<10xf32>) -> tensor<10x10xf32>
  return %0, %1 : tensor<1x10xf32>, tensor<10x10xf32>
}

// CHECK-LABEL: @broadcast_in_dim_is_reshape
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>) -> (tensor<1x10xf32>, tensor<10x10xf32>) {
//       CHECK:     %[[v0:.+]] = stablehlo.reshape %[[arg0]] : (tensor<10xf32>) -> tensor<1x10xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.broadcast_in_dim %[[arg0]], dims = [0] : (tensor<10xf32>) -> tensor<10x10xf32>
//       CHECK:     return %[[v0]], %[[v1]] : tensor<1x10xf32>, tensor<10x10xf32>

// -----

// This is another case where our pattern applies but the upstream does not.
func.func @broadcast_is_reshape_multi_dim(%arg0: tensor<4x5xf32>) -> tensor<1x4x1x5x1xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims=[1, 3] : (tensor<4x5xf32>) -> tensor<1x4x1x5x1xf32>
  return %0 : tensor<1x4x1x5x1xf32>
}

// CHECK-LABEL: func.func @broadcast_is_reshape_multi_dim
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4x5xf32>)
//       CHECK:     %[[v0:.+]] = stablehlo.reshape %[[arg0]] : (tensor<4x5xf32>) -> tensor<1x4x1x5x1xf32>
//       CHECK:     return %[[v0]] : tensor<1x4x1x5x1xf32>

// -----

// Our `broadcast_in_dim` simplifier doesn't handle this, but a combination of
// upstream patterns do.
func.func @broadcast_is_transpose_is_reshape(%arg0: tensor<1x10xf32>) -> tensor<10x1xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims=[1, 0] : (tensor<1x10xf32>) -> tensor<10x1xf32>
  return %0 : tensor<10x1xf32>
}

// CHECK-LABEL: func.func @broadcast_is_transpose_is_reshape
//       CHECK:     %[[v0:.+]] = stablehlo.reshape %[[arg0]] : (tensor<1x10xf32>) -> tensor<10x1xf32>
//       CHECK:     return %[[v0]] : tensor<10x1xf32>

// -----

func.func @not_reshape(%arg0: tensor<2x1x3xf32>) -> tensor<2x4x3xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims=[0, 1, 2] : (tensor<2x1x3xf32>) -> tensor<2x4x3xf32>
  return %0 : tensor<2x4x3xf32>
}

// CHECK-LABEL: func.func @not_reshape
//   CHECK-NOT:     stablehlo.reshape

// -----

func.func @dynamic_iota_requires_cast(%arg0: tensor<?xi32>) -> tensor<?xf32> {
  %0 = stablehlo.concatenate %arg0, dim = 0 : (tensor<?xi32>) -> tensor<1xi32>
  %1 = stablehlo.dynamic_iota %0, dim=0 : (tensor<1xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func.func @dynamic_iota_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xi32>)
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg0]] : tensor<?xi32> to tensor<1xi32>
//       CHECK:     %[[v0:.+]] = stablehlo.dynamic_iota %[[cast]], dim = 0
//       CHECK:     return %[[v0]] : tensor<?xf32>

// -----

func.func @dynamic_broadcast_in_dim_requires_cast(%arg0: tensor<?xf32>, %arg1: tensor<?xi32>) -> tensor<?xf32> {
  %0 = stablehlo.concatenate %arg1, dim = 0 : (tensor<?xi32>) -> tensor<1xi32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims=[0] : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func.func @dynamic_broadcast_in_dim_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<?xi32>)
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg1]] : tensor<?xi32> to tensor<1xi32>
//       CHECK:     %[[v0:.+]] = stablehlo.dynamic_broadcast_in_dim %[[arg0]], %[[cast]], dims = [0]
//       CHECK:     return %[[v0]] : tensor<?xf32>

// -----

func.func @dynamic_reshape_requires_cast(%arg0: tensor<?xf32>, %arg1: tensor<?xi32>) -> tensor<?xf32> {
  %0 = stablehlo.concatenate %arg1, dim = 0 : (tensor<?xi32>) -> tensor<1xi32>
  %1 = stablehlo.dynamic_reshape %arg0, %0 : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func.func @dynamic_reshape_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<?xi32>)
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg1]] : tensor<?xi32> to tensor<1xi32>
//       CHECK:     %[[v0:.+]] = stablehlo.dynamic_reshape %[[arg0]], %[[cast]]
//       CHECK:     return %[[v0]] : tensor<?xf32>

// -----

func.func @real_dynamic_slice_param_requires_cast(%arg0: tensor<?xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<?xi32>) -> tensor<?xf32> {
  %0 = stablehlo.concatenate %arg3, dim = 0 : (tensor<?xi32>) -> tensor<1xi32>
  %1 = stablehlo.real_dynamic_slice %arg0, %0, %arg1, %arg2 : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func.func @real_dynamic_slice_param_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<1xi32>, %[[arg2:.+]]: tensor<1xi32>, %[[arg3:.+]]: tensor<?xi32>)
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg3]] : tensor<?xi32> to tensor<1xi32>
//       CHECK:     %[[v0:.+]] = stablehlo.real_dynamic_slice %[[arg0]], %[[cast]], %[[arg1]], %[[arg2]]
//       CHECK:     return %[[v0]] : tensor<?xf32>


// -----

func.func @dynamic_pad_requires_cast(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>, %arg2: tensor<?xi32>, %arg3: tensor<2xi32>) -> tensor<?x?xf32> {
  %cst_0 = stablehlo.constant dense<[0, 0]> : tensor<2xi32>
  %0 = stablehlo.concatenate %arg2, dim = 0 : (tensor<?xi32>) -> tensor<2xi32>
  %1 = "stablehlo.dynamic_pad"(%arg0, %arg1, %0, %arg3, %cst_0) : (tensor<?x?xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %1: tensor<?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_pad_requires_cast
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<f32>, %[[arg2:.+]]: tensor<?xi32>, %[[arg3:.+]]: tensor<2xi32>) -> tensor<?x?xf32> {
//       CHECK:     %[[constant:.+]] = stablehlo.constant dense<0> : tensor<2xi32>
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg2]] : tensor<?xi32> to tensor<2xi32>
//       CHECK:     %[[v0:.+]] = stablehlo.dynamic_pad %[[arg0]], %[[arg1]], %[[cast]], %[[arg3]], %[[constant]]
//       CHECK:     return %[[v0]] : tensor<?x?xf32>

// -----

func.func @dynamic_gather_requires_cast(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?xf32> {
  %0 = stablehlo.iota dim = 0 : tensor<4x6xf32>
  %1 = stablehlo.concatenate %arg1, dim = 0 : (tensor<?xi32>) -> tensor<2xi32>
  %2 = "stablehlo.dynamic_gather"(%0, %arg0, %1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>
    } : (tensor<4x6xf32>, tensor<?xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_gather_requires_cast
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<?xi32>, %[[arg1:.+]]: tensor<?xi32>) -> tensor<?x?xf32> {
//       CHECK:     %[[iota:.+]] = stablehlo.iota dim = 0 : tensor<4xf32>
//       CHECK:     %[[broadcast:.+]] = stablehlo.broadcast_in_dim %[[iota]], dims = [0] : (tensor<4xf32>) -> tensor<4x6xf32>
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg1]] : tensor<?xi32> to tensor<2xi32>
//       CHECK:     %[[v0:.+]] = "stablehlo.dynamic_gather"(%[[broadcast]], %[[arg0]], %[[cast]]) {{.*}} : (tensor<4x6xf32>, tensor<?xi32>, tensor<2xi32>) -> tensor<?x?xf32>

// -----

func.func @simplify_trivial_slice(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.slice %arg0 [0:2] : (tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_slice
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xi32>)
//  CHECK-NEXT:     return %[[arg0]] : tensor<2xi32>

// -----

func.func @simplify_trivial_slice_nd(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32> {
  %0 = stablehlo.slice %arg0 [0:2, 0:4:1] : (tensor<2x4xi32>) -> tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_slice_nd
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x4xi32>)
//  CHECK-NEXT:     return %[[arg0]] : tensor<2x4xi32>

// -----

func.func @simplify_trivial_slice_requires_cast(%arg0: tensor<2xi32>) -> tensor<?xi32> {
  %0 = stablehlo.slice %arg0 [0:2] : (tensor<2xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_slice_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xi32>) -> tensor<?xi32> {
//  CHECK-NEXT:     %[[cast:.+]] = tensor.cast %[[arg0]] : tensor<2xi32> to tensor<?xi32>
//  CHECK-NEXT:     return %[[cast]] : tensor<?xi32>

// -----

func.func @simplify_trivial_slice_negative(%arg0: tensor<2xi32>) -> tensor<?xi32> {
  %0 = stablehlo.slice %arg0 [0:2:2] : (tensor<2xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_slice_negative
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xi32>)
//  CHECK-NEXT:   %[[v0:.+]] = stablehlo.slice %[[arg0]] [0:2:2]
//  CHECK-NEXT:   return %[[v0]]

// -----

func.func @simplify_trivial_slice_0d(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.slice %arg0 [] : (tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @simplify_trivial_slice_0d
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>) -> tensor<i32> {
//  CHECK-NEXT:     return %[[arg0]] : tensor<i32>

// -----

func.func @simplify_trivial_slice_empty(%arg0: tensor<0xi32>) -> tensor<0xi32> {
  %0 = stablehlo.slice %arg0 [0:0] : (tensor<0xi32>) -> tensor<0xi32>
  return %0 : tensor<0xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_slice_empty
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<0xi32>
//       CHECK:     return %[[v0]] : tensor<0xi32>

// -----

func.func @simplify_trivial_min(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = stablehlo.minimum %arg0, %arg0 :  tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_min
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>) -> tensor<1xi32> {
//       CHECK:     return %[[arg0]] : tensor<1xi32>

// -----


func.func @simplify_trivial_max(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = stablehlo.minimum %arg0, %arg0 :  tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_max
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>) -> tensor<1xi32> {
//       CHECK:     return %[[arg0]] : tensor<1xi32>

// -----

func.func @simplify_trivial_min_requires_cast(%arg0: tensor<1xi32>) -> tensor<?xi32> {
  %0 = stablehlo.minimum %arg0, %arg0 :  (tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_min_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>) -> tensor<?xi32> {
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg0]] : tensor<1xi32> to tensor<?xi32>
//       CHECK:     return %[[cast]] : tensor<?xi32>

// -----

// Verifies that redundant `tensor.cast` are eliminated.

func.func @simplify_cast_cancel(%arg0: tensor<1xi32>, %arg1: tensor<?xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<?xi32> {
  %0 = stablehlo.minimum %arg0, %arg0 :  (tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1 = tensor.cast %0 : tensor<?xi32> to tensor<1xi32>
  %2 = stablehlo.real_dynamic_slice %arg1, %arg2, %1, %arg3 : (tensor<?xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  return %2 : tensor<?xi32>
}

// CHECK-LABEL: func.func @simplify_cast_cancel
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>, %[[arg1:.+]]: tensor<?xi32>, %[[arg2:.+]]: tensor<1xi32>, %[[arg3:.+]]: tensor<1xi32>) -> tensor<?xi32> {
//       CHECK:     %[[v0:.+]] = stablehlo.real_dynamic_slice %[[arg1]], %[[arg2]], %[[arg0]], %[[arg3]] : (tensor<?xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
//       CHECK:     return %[[v0]] : tensor<?xi32>

// -----

func.func @fold_compare_dynamic_result() -> tensor<1xi32>{
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
    %cst_5 = stablehlo.constant dense<1.85714912> : tensor<1xf32>
    %0 = stablehlo.compare  GE, %cst_4, %cst_5 : (tensor<1xf32>, tensor<1xf32>) -> tensor<?xi1>
    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xi1>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    return %2 : tensor<1xi32>
}

// CHECK-LABEL: fold_compare_dynamic_result
//       CHECK: stablehlo.compare
//  CHECK-NEXT: stablehlo.get_dimension_size
//  CHECK-NEXT: stablehlo.reshape
//  CHECK-NEXT: return

// -----

func.func @simplify_reshape_broadcastindim_reshape(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x8x256xf16> {
  %0 = stablehlo.reshape %arg0 : (tensor<1x1x1x256xf16>) -> tensor<1x1x1x1x1x1x1x1x256xf16>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3, 4, 5, 7, 8, 9] : (tensor<1x1x1x1x1x1x1x1x256xf16>) -> tensor<1x1x1x1x1x1x8x1x1x256xf16>
  %2 = stablehlo.reshape %1 : (tensor<1x1x1x1x1x1x8x1x1x256xf16>) -> tensor<1x1x8x256xf16>
  return %2 : tensor<1x1x8x256xf16>
}

// CHECK-LABEL: func.func @simplify_reshape_broadcastindim_reshape
//       CHECK: stablehlo.broadcast_in_dim
//  CHECK-SAME: dims = [0, 1, 2, 3]
//  CHECK-SAME: (tensor<1x1x1x256xf16>) -> tensor<1x1x8x256xf16>
//  CHECK-NEXT: return

// -----

func.func @simplify_reshape_broadcastindim_reshape2(%arg0: tensor<1x1xf32>) -> tensor<8x1xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<1x1xf32>) -> tensor<1x1x1x1x1xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 3, 4, 5] : (tensor<1x1x1x1x1xf32>) -> tensor<1x1x8x1x1x1xf32>
  %2 = stablehlo.reshape %1 : (tensor<1x1x8x1x1x1xf32>) -> tensor<8x1xf32>
  return %2 : tensor<8x1xf32>
}

// CHECK-LABEL: func.func @simplify_reshape_broadcastindim_reshape2
//       CHECK: stablehlo.broadcast_in_dim
//  CHECK-SAME: dims = [0, 1]
//  CHECK-SAME: (tensor<1x1xf32>) -> tensor<8x1xf32>
//  CHECK-NEXT: return