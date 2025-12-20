// RUN: kernel-opt %s -split-input-file -kernel-prepare-linalg-for-codegen | FileCheck %s

func.func @rewrite_map(%arg0: f32) -> tensor<10xf32> {
  %0 = tensor.empty() : tensor<10xf32>
  %1 = linalg.map outs(%0 : tensor<10xf32>) () {
    linalg.yield %arg0 : f32
  }
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: func.func @rewrite_map
//       CHECK:   linalg.generic

// -----

func.func @tensor_pad(%arg0: tensor<1x10x10x1xf32>, %arg1: f32) -> tensor<1x14x14x1xf32> {
  %0 = tensor.pad %arg0 low[0, 2, 2, 0] high[0, 2, 2, 0] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
    tensor.yield %arg1: f32
  } : tensor<1x10x10x1xf32> to tensor<1x14x14x1xf32>
  return %0 : tensor<1x14x14x1xf32>
}

// CHECK-LABEL: func.func @tensor_pad
//       CHECK:   linalg.generic

// -----


func.func @matmul_mk_kn_mn_no_reorder(%arg0: tensor<1024x512xf32>, %arg1: tensor<512x2048xf32>,
                          %arg2: tensor<1024x2048xf32>) -> tensor<1024x2048xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x512xf32>, tensor<512x2048xf32>)
    outs(%arg2: tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
  return %0 : tensor<1024x2048xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//   CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//   CHECK-DAG: #[[$map3:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//   CHECK-DAG: #[[$map4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_mk_kn_mn_no_reorder
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1024x512xf32>, %[[arg1:.+]]: tensor<512x2048xf32>, %[[arg2:.+]]: tensor<1024x2048xf32>) -> tensor<1024x2048xf32> {
//       CHECK:  %[[v0:.+]] = tensor.empty() : tensor<2048x512xf32>
//       CHECK:  %[[v1:.+]] = linalg.generic {indexing_maps = [#[[$map1]], #[[$map]]], iterator_types = ["parallel", "parallel"]} ins(%[[arg1]] : tensor<512x2048xf32>) outs(%[[v0]] : tensor<2048x512xf32>) {
//       CHECK:   ^bb0(%[[in:.+]]: f32, %[[out:.+]]: f32):
//       CHECK:     linalg.yield %[[in]] : f32
//       CHECK:  %[[v2:.+]] = linalg.generic {indexing_maps = [#[[$map2]], #[[$map3]], #[[$map4]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[arg0]], %[[v1]] : tensor<1024x512xf32>, tensor<2048x512xf32>) outs(%[[arg2]] : tensor<1024x2048xf32>) {
//       CHECK:   ^bb0(%[[in:.+]]: f32, %[[in_0:.+]]: f32, %[[out:.+]]: f32):
//       CHECK:     %[[v3:.+]] = arith.mulf %[[in]], %[[in_0]] : f32
//       CHECK:     %[[v4:.+]] = arith.addf %[[out]], %[[v3]] : f32
//       CHECK:     linalg.yield %[[v4]] : f32
//       CHECK:   } -> tensor<1024x2048xf32>
//       CHECK:     return %[[v2]] : tensor<1024x2048xf32>

// -----


func.func @matmul_mk_kn_mn_reorder(%arg0: tensor<1024x512xf32>, %arg1: tensor<512x2048xf32>,
                          %arg2: tensor<1024x2048xf32>) -> tensor<1024x2048xf32> {
  %0 = linalg.generic {
    iterator_types = ["reduction", "parallel", "parallel"],
    indexing_maps = [
      affine_map<(d0, d1, d2)-> (d1, d0)>,
      affine_map<(d0, d1, d2)-> (d0, d2)>,
      affine_map<(d0, d1, d2)-> (d1, d2)>
    ]
  } ins(%arg0, %arg1 : tensor<1024x512xf32>, tensor<512x2048xf32>)
    outs(%arg2: tensor<1024x2048xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %1 = arith.mulf %a, %b : f32
    %2 = arith.addf %1, %c : f32
    linalg.yield %2 : f32
  } -> tensor<1024x2048xf32>
  return %0 : tensor<1024x2048xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//   CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//   CHECK-DAG: #[[$map3:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//   CHECK-DAG: #[[$map4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_mk_kn_mn_reorder
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1024x512xf32>, %[[arg1:.+]]: tensor<512x2048xf32>, %[[arg2:.+]]: tensor<1024x2048xf32>) -> tensor<1024x2048xf32> {
//       CHECK:  %[[v0:.+]] = tensor.empty() : tensor<2048x512xf32>
//       CHECK:  %[[v1:.+]] = linalg.generic {indexing_maps = [#[[$map1]], #[[$map]]], iterator_types = ["parallel", "parallel"]} ins(%[[arg1]] : tensor<512x2048xf32>) outs(%[[v0]] : tensor<2048x512xf32>) {
//       CHECK:   ^bb0(%[[in:.+]]: f32, %[[out:.+]]: f32):
//       CHECK:     linalg.yield %[[in]] : f32
//       CHECK:  %[[v2:.+]] = linalg.generic {indexing_maps = [#[[$map2]], #[[$map3]], #[[$map4]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[arg0]], %[[v1]] : tensor<1024x512xf32>, tensor<2048x512xf32>) outs(%[[arg2]] : tensor<1024x2048xf32>) {
//       CHECK:   ^bb0(%[[in:.+]]: f32, %[[in_0:.+]]: f32, %[[out:.+]]: f32):
//       CHECK:     %[[v3:.+]] = arith.mulf %[[in]], %[[in_0]] : f32
//       CHECK:     %[[v4:.+]] = arith.addf %[[v3]], %[[out]] : f32
//       CHECK:     linalg.yield %[[v4]] : f32
//       CHECK:   return %[[v2]] : tensor<1024x2048xf32>


// -----


func.func @matmul_bmk_bkn_bmn(%arg0: tensor<10x1024x512xf32>, %arg1: tensor<10x512x2048xf32>,
                              %arg2: tensor<10x1024x2048xf32>) -> tensor<10x1024x2048xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<10x1024x512xf32>, tensor<10x512x2048xf32>)
    outs(%arg2: tensor<10x1024x2048xf32>) -> tensor<10x1024x2048xf32>
  return %0 : tensor<10x1024x2048xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
//   CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//   CHECK-DAG: #[[$map3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//   CHECK-DAG: #[[$map4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: @matmul_bmk_bkn_bmn
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x1024x512xf32>, %[[arg1:.+]]: tensor<10x512x2048xf32>, %[[arg2:.+]]: tensor<10x1024x2048xf32>) -> tensor<10x1024x2048xf32> {
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<10x2048x512xf32>
//       CHECK:     %[[v1:.+]] = linalg.generic {indexing_maps = [#[[$map1]], #[[$map]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[arg1]] : tensor<10x512x2048xf32>) outs(%[[v0]] : tensor<10x2048x512xf32>) {
//       CHECK:     ^bb0(%[[in]]: f32, %[[out]]: f32):
//       CHECK:       linalg.yield %[[in]] : f32
//       CHECK:     %[[v2:.+]] = linalg.generic {indexing_maps = [#[[$map2]], #[[$map3]], #[[$map4]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[arg0]], %[[v1]] : tensor<10x1024x512xf32>, tensor<10x2048x512xf32>) outs(%[[arg2]] : tensor<10x1024x2048xf32>) {
//       CHECK:     ^bb0(%[[in]]: f32, %[[in_0]]: f32, %[[out]]: f32):
//       CHECK:       %[[v3:.+]] = arith.mulf %[[in]], %[[in_0]] : f32
//       CHECK:       %[[v4:.+]] = arith.addf %[[out]], %[[v3]] : f32
//       CHECK:       linalg.yield %[[v4]] : f32
//       CHECK:     return %[[v2]] : tensor<10x1024x2048xf32>


// -----


func.func @bmm_strange(%arg0: tensor<10x1024x512xf32>, %arg1: tensor<512x10x2048xf32>,
                              %arg2: tensor<10x1024x2048xf32>) -> tensor<10x1024x2048xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d3, d0, d2)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ],
    iterator_types = [
      "parallel",
      "parallel",
      "parallel",
      "reduction"
    ]
  } ins(%arg0, %arg1 : tensor<10x1024x512xf32>, tensor<512x10x2048xf32>) outs(%arg2: tensor<10x1024x2048xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %2 = arith.mulf %a, %b : f32
    %3 = arith.addf %2, %c : f32
    linalg.yield %3: f32
  } -> tensor<10x1024x2048xf32>
  return %0 : tensor<10x1024x2048xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1, d0)>
//   CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//   CHECK-DAG: #[[$map3:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3)>
//   CHECK-DAG: #[[$map4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: @bmm_strange
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x1024x512xf32>, %[[arg1:.+]]: tensor<512x10x2048xf32>, %[[arg2:.+]]: tensor<10x1024x2048xf32>)
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<2048x10x512xf32>
//       CHECK:     %[[v1:.+]] = linalg.generic {indexing_maps = [#[[$map1]], #[[$map]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[arg1]] : tensor<512x10x2048xf32>) outs(%[[v0]] : tensor<2048x10x512xf32>)
//       CHECK:     ^bb0(%[[in]]: f32, %[[out]]: f32):
//       CHECK:       linalg.yield %[[in]] : f32
//       CHECK:     %[[v2:.+]] = linalg.generic {indexing_maps = [#[[$map2]], #[[$map3]], #[[$map4]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[arg0]], %[[v1]] : tensor<10x1024x512xf32>, tensor<2048x10x512xf32>) outs(%[[arg2]] : tensor<10x1024x2048xf32>)
//       CHECK:     ^bb0(%[[in]]: f32, %[[in_0]]: f32, %[[out]]: f32):
//       CHECK:       %[[v3:.+]] = arith.mulf %[[in]], %[[in_0]] : f32
//       CHECK:       %[[v4:.+]] = arith.addf %[[v3]], %[[out]] : f32
//       CHECK:       linalg.yield %[[v4]] : f32
//       CHECK:     return %[[v2]] : tensor<10x1024x2048xf32>


// -----

func.func @conv2d_to_matmul(%arg0: tensor<10x28x28x128xf32>, %arg1: tensor<256x3x3x128xf32>, %arg2: tensor<10x26x26x256xf32>) -> tensor<10x26x26x256xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc ins(%arg0, %arg1: tensor<10x28x28x128xf32>, tensor<256x3x3x128xf32>) outs(%arg2: tensor<10x26x26x256xf32>) -> tensor<10x26x26x256xf32>
  return %0 : tensor<10x26x26x256xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//   CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//   CHECK-DAG: #[[$map3:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
//   CHECK-DAG: #[[$map4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: @conv2d_to_matmul
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x28x28x128xf32>, %[[arg1:.+]]: tensor<256x3x3x128xf32>, %[[arg2:.+]]: tensor<10x26x26x256xf32>) -> tensor<10x26x26x256xf32> {
//       CHECK:     %[[collapsed:.+]] = tensor.collapse_shape %[[arg1]] {{\[}}[0], [1, 2, 3]] : tensor<256x3x3x128xf32> into tensor<256x1152xf32>
//       CHECK:     %[[collapsed_0:.+]] = tensor.collapse_shape %[[arg2]] {{\[}}[0], [1, 2], [3]] : tensor<10x26x26x256xf32> into tensor<10x676x256xf32>
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<10x26x26x3x3x128xf32>
//       CHECK:     %[[v1:.+]] = linalg.generic {indexing_maps = [#[[$map]], #[[$map1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[arg0]] : tensor<10x28x28x128xf32>) outs(%[[v0]] : tensor<10x26x26x3x3x128xf32>) {
//       CHECK:     ^bb0(%[[in:.+]]: f32, %[[out:.+]]: f32):
//       CHECK:       linalg.yield %[[in]] : f32
//       CHECK:     } -> tensor<10x26x26x3x3x128xf32>
//       CHECK:     %[[collapsed_1:.+]] = tensor.collapse_shape %[[v1]] {{\[}}[0], [1, 2], [3, 4, 5]] : tensor<10x26x26x3x3x128xf32> into tensor<10x676x1152xf32>
//       CHECK:     %[[v2:.+]] = linalg.generic {indexing_maps = [#[[$map2]], #[[$map3]], #[[$map4]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[collapsed_1]], %[[collapsed]] : tensor<10x676x1152xf32>, tensor<256x1152xf32>) outs(%[[collapsed_0]] : tensor<10x676x256xf32>) {
//       CHECK:     ^bb0(%[[in:.+]]: f32, %[[in_2:.+]]: f32, %[[out:.+]]: f32):
//       CHECK:       %[[v3:.+]] = arith.mulf %[[in]], %[[in_2]] : f32
//       CHECK:       %[[v4:.+]] = arith.addf %[[v3]], %[[out]] : f32
//       CHECK:       linalg.yield %[[v4]] : f32
//       CHECK:     } -> tensor<10x676x256xf32>
//       CHECK:     %[[expanded:.+]] = tensor.expand_shape %[[v2]] {{\[}}[0], [1, 2], [3]] output_shape [10, 26, 26, 256] : tensor<10x676x256xf32> into tensor<10x26x26x256xf32>
//       CHECK:     return %[[expanded]] : tensor<10x26x26x256xf32>

// -----

func.func @conv2d_hwcf_filter(%arg0: tensor<10x28x28x128xf32>, %arg1: tensor<3x3x128x256xf32>, %arg2: tensor<10x26x26x256xf32>) -> tensor<10x26x26x256xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: tensor<10x28x28x128xf32>, tensor<3x3x128x256xf32>) outs(%arg2: tensor<10x26x26x256xf32>) -> tensor<10x26x26x256xf32>
  return %0 : tensor<10x26x26x256xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
//   CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>
//   CHECK-DAG: #[[$map3:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//   CHECK-DAG: #[[$map4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//   CHECK-DAG: #[[$map5:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
//   CHECK-DAG: #[[$map6:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @conv2d_hwcf_filter
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x28x28x128xf32>, %[[arg1:.+]]: tensor<3x3x128x256xf32>, %[[arg2:.+]]: tensor<10x26x26x256xf32>) -> tensor<10x26x26x256xf32> {
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<256x3x3x128xf32>
//       CHECK:     %[[v1:.+]] = linalg.generic {indexing_maps = [#[[$map1]], #[[$map]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[arg1]] : tensor<3x3x128x256xf32>) outs(%[[v0]] : tensor<256x3x3x128xf32>) {
//       CHECK:     ^bb0(%[[in:.+]]: f32, %[[out:.+]]: f32):
//       CHECK:       linalg.yield %[[in]] : f32
//       CHECK:     } -> tensor<256x3x3x128xf32>
//       CHECK:     %[[collapsed:.+]] = tensor.collapse_shape %[[v1]] {{\[}}[0], [1, 2, 3]] : tensor<256x3x3x128xf32> into tensor<256x1152xf32>
//       CHECK:     %[[collapsed_0:.+]] = tensor.collapse_shape %[[arg2]] {{\[}}[0], [1, 2], [3]] : tensor<10x26x26x256xf32> into tensor<10x676x256xf32>
//       CHECK:     %[[v2:.+]] = tensor.empty() : tensor<10x26x26x3x3x128xf32>
//       CHECK:     %[[v3:.+]] = linalg.generic {indexing_maps = [#[[$map2]], #[[$map3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[arg0]] : tensor<10x28x28x128xf32>) outs(%[[v2]] : tensor<10x26x26x3x3x128xf32>) {
//       CHECK:     ^bb0(%[[in:.+]]: f32, %[[out:.+]]: f32):
//       CHECK:       linalg.yield %[[in]] : f32
//       CHECK:     } -> tensor<10x26x26x3x3x128xf32>
//       CHECK:     %[[collapsed_1:.+]] = tensor.collapse_shape %[[v3]] {{\[}}[0], [1, 2], [3, 4, 5]] : tensor<10x26x26x3x3x128xf32> into tensor<10x676x1152xf32>
//       CHECK:     %[[v4:.+]] = linalg.generic {indexing_maps = [#[[$map4]], #[[$map5]], #[[$map6]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[collapsed_1]], %[[collapsed]] : tensor<10x676x1152xf32>, tensor<256x1152xf32>) outs(%[[collapsed_0]] : tensor<10x676x256xf32>) {
//       CHECK:     ^bb0(%[[in:.+]]: f32, %[[in_2:.+]]: f32, %[[out:.+]]: f32):
//       CHECK:       %[[v5:.+]] = arith.mulf %[[in]], %[[in_2]] : f32
//       CHECK:       %[[v6:.+]] = arith.addf %[[v5]], %[[out]] : f32
//       CHECK:       linalg.yield %[[v6]] : f32
//       CHECK:     } -> tensor<10x676x256xf32>
//       CHECK:     %[[expanded:.+]] = tensor.expand_shape %[[v4]] {{\[}}[0], [1, 2], [3]] output_shape [10, 26, 26, 256] : tensor<10x676x256xf32> into tensor<10x26x26x256xf32>
//       CHECK:     return %[[expanded]] : tensor<10x26x26x256xf32>

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (0, 0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


func.func @simlify_transpose_gather(%arg0: tensor<4x4x4xi32>, %arg1: tensor<6x1xi32>, %arg2: tensor<6x1x4x4xi32>) -> tensor<6x1x4x4xi32> {
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<4x4x4xi32>
  %1 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x4x4xi32>) outs(%0 : tensor<4x4x4xi32>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  } -> tensor<4x4x4xi32>
  %2 = tensor.empty() : tensor<6x1x4x4xi32>
  %3 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%arg2 : tensor<6x1x4x4xi32>) {
  ^bb0(%out: i32):
    %4 = linalg.index 0 : index
    %5 = linalg.index 1 : index
    %6 = linalg.index 2 : index
    %7 = linalg.index 3 : index
    %extracted = tensor.extract %arg1[%4, %c0] : tensor<6x1xi32>
    %8 = arith.index_cast %extracted : i32 to index
    %9 = arith.maxsi %8, %c0 : index
    %10 = arith.minsi %9, %c3 : index
    %11 = arith.addi %10, %5 : index
    %extracted_0 = tensor.extract %1[%11, %6, %7] : tensor<4x4x4xi32>
    linalg.yield %extracted_0 : i32
  } -> tensor<6x1x4x4xi32>
  return %3 : tensor<6x1x4x4xi32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @simlify_transpose_gather
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<4x4x4xi32>, %[[arg1:.+]]: tensor<6x1xi32>, %[[arg2:.+]]: tensor<6x1x4x4xi32>)
//       CHECK:     %[[c3:.+]] = arith.constant 3 : index
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[v0:.+]] = linalg.generic {indexing_maps = [#[[$map]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%[[arg2]] : tensor<6x1x4x4xi32>) {
//       CHECK:     ^bb0(%[[out:.+]]: i32):
//       CHECK:       %[[v1:.+]] = linalg.index 0 : index
//       CHECK:       %[[v3:.+]] = linalg.index 2 : index
//       CHECK:       %[[v4:.+]] = linalg.index 3 : index
//       CHECK:       %[[extracted:.+]] = tensor.extract %[[arg1]][%[[v1]], %[[c0]]] : tensor<6x1xi32>
//       CHECK:       %[[v5:.+]] = arith.index_cast %[[extracted]] : i32 to index
//       CHECK:       %[[v6:.+]] = arith.maxsi %[[v5]], %[[c0]] : index
//       CHECK:       %[[v7:.+]] = arith.minsi %[[v6]], %[[c3]] : index
//       CHECK:       %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[v3]], %[[v7]], %[[v4]]] : tensor<4x4x4xi32>
//       CHECK:       linalg.yield %[[extracted_0]] : i32
//       CHECK:     return %[[v0]] : tensor<6x1x4x4xi32>

// -----

#map = affine_map<(d0, d1)->(d0, d1)>

func.func @simplify_extract_of_elwise(%arg0: tensor<128x64xf32>, %arg1: tensor<128x64xf16>) -> tensor<128x64xf16> {
  %empty = tensor.empty() : tensor<128x64xf16>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<128x64xf32>)
    outs(%empty : tensor<128x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %6 = arith.truncf %in : f32 to f16
      linalg.yield %6 : f16
  } -> tensor<128x64xf16>
  %5 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
      ins(%arg0: tensor<128x64xf32>)
      outs(%arg1: tensor<128x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %6 = linalg.index 0 : index
      %7 = linalg.index 1 : index
      %extracted = tensor.extract %1[%6, %7] : tensor<128x64xf16>
      linalg.yield %extracted: f16
  } -> tensor<128x64xf16>
  return %5 : tensor<128x64xf16>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @simplify_extract_of_elwise
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128x64xf32>, %[[arg1:.+]]: tensor<128x64xf16>) -> tensor<128x64xf16> {
//       CHECK:     %[[v0:.+]] = linalg.generic {indexing_maps = [#[[$map]], #[[$map]]], iterator_types = ["parallel", "parallel"]} ins(%[[arg0]] : tensor<128x64xf32>) outs(%[[arg1]] : tensor<128x64xf16>) {
//       CHECK:     ^bb0(%[[in]]: f32, %[[out]]: f16):
//       CHECK:       %[[v1:.+]] = linalg.index 0 : index
//       CHECK:       %[[v2:.+]] = linalg.index 1 : index
//       CHECK:       %[[extracted:.+]] = tensor.extract %[[arg0]][%[[v1]], %[[v2]]] : tensor<128x64xf32>
//       CHECK:       %[[v3:.+]] = arith.truncf %[[extracted]] : f32 to f16
//       CHECK:       linalg.yield %[[v3]] : f16
//       CHECK:     return %[[v0]] : tensor<128x64xf16>
