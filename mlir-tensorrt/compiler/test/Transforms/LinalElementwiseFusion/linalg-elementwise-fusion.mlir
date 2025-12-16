// Note that we use named ops in the test case below for conciseness, but we have to
// rewrite to generics before running the elementwise fusion pass.

// RUN: mlir-tensorrt-opt %s -split-input-file \
// RUN:   --linalg-generalize-named-ops \
// RUN:   --mtrt-linalg-elementwise-fusion | FileCheck %s

func.func @fuse_pure_transpose(%arg0 : tensor<100x100xf16>, %arg1 : tensor<100x100xf16>) -> (tensor<100x100xf16>) {
  %0 = tensor.empty() : tensor<100x100xf16>

  %1 = linalg.transpose ins(%arg0 : tensor<100x100xf16>) outs(%0 : tensor<100x100xf16>) permutation = [1, 0]
  %2 = tensor.empty() : tensor<100x100xf16>
  %3 = linalg.transpose ins(%arg1 : tensor<100x100xf16>) outs(%2 : tensor<100x100xf16>) permutation = [1, 0]
  %5 = tensor.empty() : tensor<100x100xf16>
  %4 = linalg.matmul_transpose_b
    ins(%1, %3 :tensor<100x100xf16>, tensor<100x100xf16>)
    outs(%5 : tensor<100x100xf16>) -> tensor<100x100xf16>
  return  %4 : tensor<100x100xf16>
}

//  CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
//  CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @fuse_pure_transpose
//  CHECK-SAME: (%[[arg0:.+]]: tensor<100x100xf16>, %[[arg1:.+]]: tensor<100x100xf16>)
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<100x100xf16>
//       CHECK:     %[[v1:.+]] = linalg.generic {indexing_maps = [#[[$map]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:        iterator_types = ["parallel", "parallel", "reduction"]}
//  CHECK-SAME:        ins(%[[arg0]], %[[arg1]] : tensor<100x100xf16>, tensor<100x100xf16>) outs(%[[v0]]
//       CHECK:     ^bb0(%[[in:.+]]: f16, %[[in_0:.+]]: f16, %[[out:.+]]: f16):
//       CHECK:       %[[v2:.+]] = arith.mulf %[[in]], %[[in_0]] : f16
//       CHECK:       %[[v3:.+]] = arith.addf %[[out]], %[[v2]] : f16
//       CHECK:       linalg.yield %[[v3]] : f16
//       CHECK:     return %[[v1]] : tensor<100x100xf16>

// -----

// Don't fuse pointwise to ops with reductions if the iteration spaces
// don't match or if op implements contraction interface.

func.func @dont_fuse_to_contract(%arg0 : tensor<100x100xf32>, %arg1 : tensor<100x100xf16>) -> (tensor<100x100xf16>) {
  %0 = tensor.empty() : tensor<100x100xf16>

  %1 = linalg.generic {
    indexing_maps =
      [affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<100x100xf32>)
    outs(%0 : tensor<100x100xf16>) {
  ^bb0(%in: f32, %out: f16):
    %1 = arith.truncf %in : f32 to f16
    linalg.yield %1 : f16
  } -> tensor<100x100xf16>

  %5 = tensor.empty() : tensor<100x100xf16>
  %4 = linalg.matmul_transpose_b
    ins(%1, %arg1 :tensor<100x100xf16>, tensor<100x100xf16>)
    outs(%5 : tensor<100x100xf16>) -> tensor<100x100xf16>
  return  %4 : tensor<100x100xf16>
}

// CHECK-LABEL: func.func @dont_fuse_to_contract
// CHECK: linalg.generic
// CHECK: arith.truncf
// CHECK: linalg.generic
// CHECK:  arith.mulf
// CHECK:  arith.addf

// -----

// Don't fuse pointwise to ops with reductions if the iteration spaces
// don't match or if op implements contraction interface.

func.func @dont_fuse_to_reduce(%arg0 : tensor<100x100x100xf32>,
                              %arg1 : tensor<100x100x100xf16>)
                              -> tensor<100x100xf16> {
  %0 = tensor.empty() : tensor<100x100x100xf16>

  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<100x100x100xf32>)
    outs(%0 : tensor<100x100x100xf16>) {
  ^bb0(%in: f32, %out: f16):
    %1 = arith.truncf %in : f32 to f16
    linalg.yield %1 : f16
  } -> tensor<100x100x100xf16>

  %5 = tensor.empty() : tensor<100x100xf16>
  %6 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%1, %arg1 : tensor<100x100x100xf16>, tensor<100x100x100xf16>)
   outs(%5 : tensor<100x100xf16>) {
   ^bb0(%in: f16, %in_0: f16, %out: f16):
      %3 = arith.mulf %in, %in_0 : f16
      %4 = arith.addf %out, %3 : f16
      linalg.yield %4 : f16
  } -> tensor<100x100xf16>
  return  %6 : tensor<100x100xf16>
}

// CHECK-LABEL: func.func @dont_fuse_to_reduce
// CHECK: linalg.generic
// CHECK: arith.truncf
// CHECK: linalg.generic
// CHECK:  arith.mulf
// CHECK:  arith.addf

// -----

func.func @fuse_through_extract(
  %11 :tensor<128256x4096xf16>, %12 : tensor<4x?xi64>,
  %13 : tensor<4x?x4096xf32>, %14 : tensor<128256x4096xf32>)
    -> tensor<4x?x4096xf32>{

  %15 = linalg.generic {
    indexing_maps = [ affine_map<(d0, d1) -> (d0, d1)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%11 : tensor<128256x4096xf16>)
    outs(%14 : tensor<128256x4096xf32>) {
      ^bb0(%in: f16, %out: f32):
        %17 = arith.extf %in : f16 to f32
        linalg.yield %17 : f32
    } -> tensor<128256x4096xf32>
  %16 = linalg.generic {
    indexing_maps = [ affine_map<(d0, d1, d2) -> (d0, d1)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%12 : tensor<4x?xi64>)
    outs(%13 : tensor<4x?x4096xf32>) {
      ^bb0(%in: i64, %out: f32):
        %17 = arith.index_cast %in : i64 to index
        %18 = linalg.index 2 : index
        %extracted = tensor.extract %15[%17, %18] : tensor<128256x4096xf32>
        linalg.yield %extracted : f32
      } -> tensor<4x?x4096xf32>
  return %16 : tensor<4x?x4096xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//       CHECK: #[[$map1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @fuse_through_extract
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128256x4096xf16>, %[[arg1:.+]]: tensor<4x?xi64>, %[[arg2:.+]]: tensor<4x?x4096xf32>, %[[arg3:.+]]: tensor<128256x4096xf32>)
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[dim:.+]] = tensor.dim %[[arg2]], %[[c1]] : tensor<4x?x4096xf32>
//       CHECK:     %[[v0:.+]] = tensor.empty(%[[dim]]) : tensor<4x?x4096xf32>
//       CHECK:     %[[v1:.+]] = linalg.generic {indexing_maps = [#[[$map]], #[[$map1]]],
//  CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel"]}
//  CHECK-SAME:        ins(%[[arg1]] : tensor<4x?xi64>) outs(%[[v0]]
//       CHECK:     ^bb0(%[[in:.+]]: i64, %[[out:.+]]: f32):
//       CHECK:       %[[v2:.+]] = arith.index_cast %[[in]] : i64 to index
//       CHECK:       %[[v3:.+]] = linalg.index 2 : index
//       CHECK:       %[[extracted:.+]] = tensor.extract %[[arg0]][%[[v2]], %[[v3]]]
//       CHECK:       %[[v4:.+]] = arith.extf %[[extracted]] : f16 to f32
//       CHECK:       linalg.yield %[[v4]] : f32
//       CHECK:     return %[[v1]] : tensor<4x?x4096xf32>

// -----

// Check im2col won't be combined with matmul, which messes up the codegen stage.

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<()[s0] -> (s0 mod 4)>
#map2 = affine_map<()[s0, s1] -> (s0 floordiv 14 + s1 floordiv 12)>
#map3 = affine_map<()[s0, s1] -> (s0 mod 14 + (s1 mod 12) floordiv 4)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @conv_16433136(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %collapsed = tensor.collapse_shape %arg1 [[0, 1, 2], [3]] : tensor<3x3x4x16xf32> into tensor<36x16xf32>
  %collapsed_0 = tensor.collapse_shape %arg2 [[0], [1, 2], [3]] : tensor<1x14x14x16xf32> into tensor<1x196x16xf32>
  %0 = tensor.empty() : tensor<1x196x36xf32>
  %1 = linalg.generic {indexing_maps = [#map],
      iterator_types = ["parallel", "parallel", "parallel"]}
      outs(%0 : tensor<1x196x36xf32>) {
  ^bb0(%out: f32):
    %3 = linalg.index 0 : index
    %4 = linalg.index 1 : index
    %5 = linalg.index 2 : index
    %6 = affine.apply #map1()[%5]
    %7 = affine.apply #map2()[%4, %5]
    %8 = affine.apply #map3()[%4, %5]
    %extracted = tensor.extract %arg0[%3, %7, %8, %6] : tensor<1x16x16x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x196x36xf32>
  %2 = linalg.generic {indexing_maps = [#map4, #map5, #map6],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%1, %collapsed : tensor<1x196x36xf32>, tensor<36x16xf32>)
    outs(%collapsed_0 : tensor<1x196x16xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    %4 = arith.addf %3, %out : f32
    linalg.yield %4 : f32
  } -> tensor<1x196x16xf32>
  %expanded = tensor.expand_shape %2 [[0], [1, 2], [3]] output_shape [1, 14, 14, 16] : tensor<1x196x16xf32> into tensor<1x14x14x16xf32>
  return %expanded : tensor<1x14x14x16xf32>
}

// CHECK-LABEL: func.func @conv_16433136
// CHECK: linalg.generic
// CHECK: linalg.generic

// -----

func.func @fuse_through_extract_2(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<1xi32>{
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %36 = tensor.empty() : tensor<1x1xi32>
  %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1xi32>) outs(%36 : tensor<1x1xi32>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  } -> tensor<1x1xi32>
  %38 = tensor.empty() : tensor<1xi32>
  %39 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%38 : tensor<1xi32>) {
  ^bb0(%out: i32):
    %46 = linalg.index 0 : index
    %extracted = tensor.extract %37[%46, %c0] : tensor<1x1xi32>
    %47 = arith.index_cast %extracted : i32 to index
    %48 = arith.maxsi %47, %c0 : index
    %49 = arith.minsi %48, %c1 : index
    %extracted_4 = tensor.extract %arg1[%49] : tensor<2xi32>
    linalg.yield %extracted_4 : i32
  } -> tensor<1xi32>
  return %39 : tensor<1xi32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @fuse_through_extract_2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>, %[[arg1:.+]]: tensor<2xi32>) -> tensor<1xi32>
//  CHECK-NEXT: %[[c1:.+]] = arith.constant 1 : index
//  CHECK-NEXT: %[[c0:.+]] = arith.constant 0 : index
//  CHECK-NEXT: %[[v0:.+]] = tensor.empty() : tensor<1xi32>
//  CHECK-NEXT: %[[v1:.+]] = linalg.generic {indexing_maps = [#[[$map]]], iterator_types = ["parallel"]} outs(%[[v0]] : tensor<1xi32>)
//       CHECK: %[[v2:.+]] = linalg.index 0 : index
//       CHECK: %[[extracted:.+]] = tensor.extract %[[arg0]][%[[v2]]] : tensor<1xi32>
//       CHECK: %[[v3:.+]] = arith.index_cast %[[extracted]] : i32 to index
//       CHECK: %[[v4:.+]] = arith.maxsi %[[v3]], %[[c0]] : index
//       CHECK: %[[v5:.+]] = arith.minsi %[[v4]], %[[c1]] : index
//       CHECK: %[[extracted_0:.+]] = tensor.extract %[[arg1]][%[[v5]]] : tensor<2xi32>
//       CHECK: linalg.yield %[[extracted_0]] : i32
//       CHECK: return %[[v1]] : tensor<1xi32>
