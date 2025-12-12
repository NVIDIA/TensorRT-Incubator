// RUN: kernel-opt %s -split-input-file -transform-interpreter -canonicalize -cse | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
func.func @test_fuse_through_shared_outs_non_dps(%arg0: tensor<30x16xf32>, %arg1: tensor<16x64xf32>) -> tensor<30x64xf32> {

  %0 = tensor.empty() : tensor<30x64xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32

  %01 = linalg.fill ins(%cst : f32) outs(%0 : tensor<30x64xf32>) -> tensor<30x64xf32> // to be fused
  %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<30x16xf32>, tensor<16x64xf32>) outs(%01 : tensor<30x64xf32>) attrs = {kernel.root} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %3, %out : f32
    linalg.yield %4 : f32
  } -> tensor<30x64xf32>
  return %2 : tensor<30x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops{["linalg.generic"]} attributes {kernel.root} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0   num_threads [] tile_sizes [15, 32, 0] :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %3 = transform.kernel.fuse_greedily_op %tiled_op
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0) -> (d0 * 15)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0) -> (d0 * 32)>
//   CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//   CHECK-DAG: #[[$map3:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//   CHECK-DAG: #[[$map4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: @test_fuse_through_shared_outs_non_dps
//  CHECK-SAME: (%[[arg0:.+]]: tensor<30x16xf32>, %[[arg1:.+]]: tensor<16x64xf32>) -> tensor<30x64xf32> {
//       CHECK:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<30x64xf32>
//       CHECK:     %[[v1:.+]] = scf.forall (%[[arg2:.+]], %[[arg3:.+]]) in (2, 2) shared_outs(%[[arg4:.+]] = %[[v0]]) -> (tensor<30x64xf32>) {
//       CHECK:       %[[v2:.+]] = affine.apply #[[$map]](%[[arg2]])
//       CHECK:       %[[v3:.+]] = affine.apply #[[$map1]](%[[arg3]])
//       CHECK:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg0]][%[[v2]], 0] [15, 16] [1, 1] : tensor<30x16xf32> to tensor<15x16xf32>
//       CHECK:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg1]][0, %[[v3]]] [16, 32] [1, 1] : tensor<16x64xf32> to tensor<16x32xf32>
//       CHECK:       %[[extracted_slice_1:.+]] = tensor.extract_slice %[[arg4]][%[[v2]], %[[v3]]] [15, 32] [1, 1] : tensor<30x64xf32> to tensor<15x32xf32>
//       CHECK:       %[[v4:.+]] = linalg.fill ins(%[[cst]] : f32) outs(%[[extracted_slice_1]] : tensor<15x32xf32>) -> tensor<15x32xf32>
//       CHECK:       %[[v5:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[extracted_slice]], %[[extracted_slice_0]] : tensor<15x16xf32>, tensor<16x32xf32>) outs(%[[v4]] : tensor<15x32xf32>)
//       CHECK:       scf.forall.in_parallel
//       CHECK:         tensor.parallel_insert_slice %[[v5]] into %[[arg4]][%[[v2]], %[[v3]]] [15, 32] [1, 1] : tensor<15x32xf32> into tensor<30x64xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#matmul_params = {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
func.func @two_matmul(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>, %arg3: tensor<1024x32xf32>) -> tensor<1024x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<1024x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%empty : tensor<1024x32xf32>) -> tensor<1024x32xf32>
  %2 = linalg.generic #matmul_params
    ins(%arg0, %arg1 : tensor<1024x32xf32>, tensor<32x32xf32>)
    outs(%1 : tensor<1024x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %3, %out : f32
    linalg.yield %4 : f32
  } -> tensor<1024x32xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%arg3 : tensor<1024x32xf32>) -> tensor<1024x32xf32>
  %3 = linalg.generic #matmul_params
    ins(%2, %arg2 : tensor<1024x32xf32>, tensor<32x32xf32>)
    outs(%filled : tensor<1024x32xf32>) attrs = {kernel.root} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %3, %out : f32
    linalg.yield %4 : f32
  } -> tensor<1024x32xf32>
  return %3 : tensor<1024x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops{["linalg.generic"]} attributes {kernel.root} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0   num_threads [] tile_sizes [128, 32, 32] :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %1 = transform.kernel.fuse_greedily_op %tiled_op
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0) -> (d0 * 128)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//   CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//   CHECK-DAG: #[[$map3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: @two_matmul
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1024x32xf32>, %[[arg1:.+]]: tensor<32x32xf32>, %[[arg2:.+]]: tensor<32x32xf32>, %[[arg3:.+]]: tensor<1024x32xf32>) -> tensor<1024x32xf32> {
//       CHECK:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<1024x32xf32>
//       CHECK:     %[[v1:.+]] = scf.forall (%[[arg4:.+]]) in (8) shared_outs(%[[arg5:.+]] = %[[arg3]]) -> (tensor<1024x32xf32>) {
//       CHECK:       %[[v2:.+]] = affine.apply #[[$map]](%[[arg4]])
//       CHECK:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg0]][%[[v2]], 0] [128, 32] [1, 1] : tensor<1024x32xf32> to tensor<128x32xf32>
//       CHECK:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[v0]][%[[v2]], 0] [128, 32] [1, 1] : tensor<1024x32xf32> to tensor<128x32xf32>
//       CHECK:       %[[v3:.+]] = linalg.fill ins(%[[cst]] : f32) outs(%[[extracted_slice_0]] : tensor<128x32xf32>) -> tensor<128x32xf32>
//       CHECK:       %[[v4:.+]] = linalg.generic {{.*}} : tensor<128x32xf32>, tensor<32x32xf32>) outs(%[[v3]] : tensor<128x32xf32>) {
//       CHECK:       %[[extracted_slice_1:.+]] = tensor.extract_slice %[[arg5]][%[[v2]], 0] [128, 32] [1, 1] : tensor<1024x32xf32> to tensor<128x32xf32>
//       CHECK:       %[[v5:.+]] = linalg.fill ins(%[[cst]] : f32) outs(%[[extracted_slice_1]] : tensor<128x32xf32>) -> tensor<128x32xf32>
//       CHECK:       %[[v6:.+]] = linalg.generic {indexing_maps = [#[[$map1]], #[[$map2]], #[[$map3]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[v4]], %[[arg2]] : tensor<128x32xf32>, tensor<32x32xf32>) outs(%[[v5]] : tensor<128x32xf32>)
//       CHECK:       scf.forall.in_parallel
//       CHECK:         tensor.parallel_insert_slice %[[v6]] into %[[arg5]][%[[v2]], 0] [128, 32] [1, 1] : tensor<128x32xf32> into tensor<1024x32xf32>

// -----

func.func @test_fuse_through_shared_outs_dps(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<10xf32>) -> tensor<10xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
    ins(%arg0: tensor<10xf32>) outs(%1: tensor<10xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out :f32
    linalg.yield %3 : f32
  } -> tensor<10xf32>
  return %2 : tensor<10xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0   num_threads [] tile_sizes [2] :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %2 = transform.kernel.fuse_greedily_op %tiled_op
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0) -> (d0 * 2)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @test_fuse_through_shared_outs_dps
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>) -> tensor<10xf32> {
//       CHECK:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:     %[[v0:.+]] = scf.forall (%[[arg2:.+]]) in (5) shared_outs(%[[arg3:.+]] = %[[arg1]]) -> (tensor<10xf32>) {
//       CHECK:       %[[v1:.+]] = affine.apply #[[$map]](%[[arg2]])
//       CHECK:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg0]][%[[v1]]] [2] [1] : tensor<10xf32> to tensor<2xf32>
//       CHECK:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg3]][%[[v1]]] [2] [1] : tensor<10xf32> to tensor<2xf32>
//       CHECK:       %[[v2:.+]] = linalg.fill ins(%[[cst]] : f32) outs(%[[extracted_slice_0]] : tensor<2xf32>) -> tensor<2xf32>
//       CHECK:       %[[v3:.+]] = linalg.generic {indexing_maps = [#[[$map1]], #[[$map1]]], iterator_types = ["parallel"]} ins(%[[extracted_slice]] : tensor<2xf32>) outs(%[[v2]] : tensor<2xf32>) {
//       CHECK:       ^bb0(%[[in:.+]]: f32, %[[out:.+]]: f32):
//       CHECK:         %[[v4:.+]] = arith.addf %[[in]], %[[out]] : f32
//       CHECK:         linalg.yield %[[v4]] : f32
//       CHECK:       scf.forall.in_parallel {
//       CHECK:         tensor.parallel_insert_slice %[[v3]] into %[[arg3]][%[[v1]]] [2] [1] : tensor<2xf32> into tensor<10xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @fuse_through_expand_shape(%arg0: tensor<64x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<2x32x2x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<60x70xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<60x70xf32>) -> tensor<60x70xf32>
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [2, 32, 4, 16]: tensor<64x64xf32> into tensor<2x32x4x16xf32>
  %expanded_1 = tensor.expand_shape %arg1 [[0, 1], [2, 3]] output_shape [4, 16, 2, 64] : tensor<64x128xf32> into tensor<4x16x2x64xf32>
  %3 = tensor.empty() : tensor<2x32x2x64xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2x32x2x64xf32>) -> tensor<2x32x2x64xf32>
  %5 = scf.forall (%arg2, %arg3) in (2, 2) shared_outs(%arg4 = %4) -> (tensor<2x32x2x64xf32>) {
    %extracted_slice = tensor.extract_slice %expanded[%arg2, 0, 0, 0] [1, 32, 4, 16] [1, 1, 1, 1] : tensor<2x32x4x16xf32> to tensor<32x4x16xf32>
    %extracted_slice_2 = tensor.extract_slice %expanded_1[0, 0, %arg3, 0] [4, 16, 1, 64] [1, 1, 1, 1] : tensor<4x16x2x64xf32> to tensor<4x16x64xf32>
    %outA = tensor.empty() : tensor<4x32x16xf32>
    %outC = tensor.extract_slice %arg4[%arg2, 0, %arg3, 0] [1, 32, 1, 64][1, 1, 1, 1] : tensor<2x32x2x64xf32> to tensor<32x64xf32>
    %a = linalg.transpose ins(%extracted_slice : tensor<32x4x16xf32>) outs(%outA: tensor<4x32x16xf32>) permutation = [1, 0, 2]
    %10 = linalg.batch_reduce_matmul ins(%a, %extracted_slice_2 : tensor<4x32x16xf32>, tensor<4x16x64xf32>)
      outs(%outC: tensor<32x64xf32>) -> tensor<32x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %10 into %arg4[%arg2, %arg3, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<32x64xf32> into tensor<2x32x2x64xf32>
    }
  }
  return %5 : tensor<2x32x2x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops{["linalg.batch_reduce_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.kernel.fuse_greedily_op %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<()[s0] -> (s0 * 32)>
//       CHECK: #[[$map1:.+]] = affine_map<()[s0] -> (s0 * 64)>
// CHECK-LABEL: @fuse_through_expand_shape
//  CHECK-SAME: (%[[arg0:.+]]: tensor<64x64xf32>, %[[arg1:.+]]: tensor<64x128xf32>) -> tensor<2x32x2x64xf32> {
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<2x32x2x64xf32>
//       CHECK:     %[[v1:.+]] = scf.forall (%[[arg2:.+]], %[[arg3:.+]]) in (2, 2) shared_outs(%[[arg4:.+]] = %[[v0]]) -> (tensor<2x32x2x64xf32>) {
//       CHECK:       %[[v2:.+]] = affine.apply #[[$map]]()[%[[arg2]]]
//       CHECK:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg0]][%[[v2]], 0] [32, 64] [1, 1] : tensor<64x64xf32> to tensor<32x64xf32>
//       CHECK:       %[[expanded:.+]] = tensor.expand_shape %[[extracted_slice]] {{\[}}[0], [1, 2]] output_shape [32, 4, 16] : tensor<32x64xf32> into tensor<32x4x16xf32>
//       CHECK:       %[[v3:.+]] = affine.apply #[[$map1]]()[%[[arg3]]]
//       CHECK:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg1]][0, %[[v3]]] [64, 64] [1, 1] : tensor<64x128xf32> to tensor<64x64xf32>
//       CHECK:       %[[expanded_1:.+]] = tensor.expand_shape %[[extracted_slice_0]] {{\[}}[0, 1], [2]] output_shape [4, 16, 64] : tensor<64x64xf32> into tensor<4x16x64xf32>
//       CHECK:       %[[v4:.+]] = tensor.empty() : tensor<4x32x16xf32>
//       CHECK:       %[[extracted_slice_2:.+]] = tensor.extract_slice %[[arg4]][%[[arg2]], 0, %[[arg3]], 0] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<2x32x2x64xf32>
//       CHECK:       %[[collapsed:.+]] = tensor.collapse_shape %[[extracted_slice_2]] {{\[}}[0, 1], [2, 3]] : tensor<1x32x1x64xf32> into tensor<32x64xf32>
//       CHECK:       %[[v5:.+]] = linalg.fill ins(%[[cst]] : f32) outs(%[[collapsed]] : tensor<32x64xf32>) -> tensor<32x64xf32>
//       CHECK:       %[[transposed:.+]] = linalg.transpose ins(%[[expanded]] : tensor<32x4x16xf32>) outs(%[[v4]] : tensor<4x32x16xf32>) permutation = [1, 0, 2]
//       CHECK:       %[[v6:.+]] = linalg.batch_reduce_matmul ins(%[[transposed]], %[[expanded_1]] : tensor<4x32x16xf32>, tensor<4x16x64xf32>) outs(%[[v5]] : tensor<32x64xf32>) -> tensor<32x64xf32>
//       CHECK:       scf.forall.in_parallel {
//       CHECK:         tensor.parallel_insert_slice %[[v6]] into %[[arg4]][%[[arg2]], %[[arg3]], 0, 0] [1, 32, 1, 64] [1, 1, 1, 1]
//       CHECK:     return %[[v1]] : tensor<2x32x2x64xf32>

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fuse_through_expand_shape2(%arg0: tensor<16xi32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %cst = arith.constant -0.99999994 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %c1065353216_i32 = arith.constant 1065353216 : i32
  %c9_i32 = arith.constant 9 : i32
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [4, 4] : tensor<16xi32> into tensor<4x4xi32>
  %c4 = arith.constant 4 : index
  %0 = scf.forall (%arg2) in (4) shared_outs(%arg3 = %arg1) -> (tensor<4x4xf32>) {
    %extracted_slice = tensor.extract_slice %expanded[%arg2, 0] [1, 4] [1, 1] : tensor<4x4xi32> to tensor<1x4xi32>
    %extracted_slice_2 = tensor.extract_slice %arg3[%arg2, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
    ins(%extracted_slice : tensor<1x4xi32>) outs(%extracted_slice_2 : tensor<1x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %2 = arith.shrui %in, %c9_i32 : i32
      %3 = arith.ori %2, %c1065353216_i32 : i32
      %4 = arith.bitcast %3 : i32 to f32
      %5 = arith.subf %4, %cst_1 : f32
      %6 = arith.mulf %5, %cst_0 : f32
      %7 = arith.addf %6, %cst : f32
      %8 = arith.maximumf %7, %cst : f32
      linalg.yield %8 : f32
    } -> tensor<1x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %1 into %arg3[%arg2, 0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<4x4xf32>
    }
  }
  return %0 : tensor<4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %2 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.kernel.fuse_greedily_op %2 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}


//       CHECK: #[[$map:.+]] = affine_map<()[s0] -> (s0 * 4)>
//       CHECK: #[[$map1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @fuse_through_expand_shape2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<16xi32>, %[[arg1:.+]]: tensor<4x4xf32>) -> tensor<4x4xf32> {
//       CHECK:     %[[cst:.+]] = arith.constant -0.99999994 : f32
//       CHECK:     %[[cst_0:.+]] = arith.constant 2.000000e+00 : f32
//       CHECK:     %[[cst_1:.+]] = arith.constant 1.000000e+00 : f32
//       CHECK:     %[[c1065353216_i32:.+]] = arith.constant 1065353216 : i32
//       CHECK:     %[[c9_i32:.+]] = arith.constant 9 : i32
//       CHECK:     %[[v0:.+]] = scf.forall (%[[arg2:.+]]) in (4) shared_outs(%[[arg3]] = %[[arg1]]) -> (tensor<4x4xf32>) {
//       CHECK:       %[[v1:.+]] = affine.apply #[[$map]]()[%[[arg2]]]
//       CHECK:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg0]][%[[v1]]] [4] [1] : tensor<16xi32> to tensor<4xi32>
//       CHECK:       %[[expanded:.+]] = tensor.expand_shape %[[extracted_slice]] {{\[}}[0, 1]] output_shape [1, 4] : tensor<4xi32> into tensor<1x4xi32>
//       CHECK:       %[[extracted_slice_2:.+]] = tensor.extract_slice %[[arg3]][%[[arg2]], 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
//       CHECK:       linalg.generic {indexing_maps = [#[[$map1]], #[[$map1]]], iterator_types = ["parallel", "parallel"]}
//  CHECK-SAME:         ins(%[[expanded]] : tensor<1x4xi32>) outs(%[[extracted_slice_2]] : tensor<1x4xf32>)

// -----

#map = affine_map<(d0) -> (-d0 + 7)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0)>

// In this test, '%2' is the result of 'linalg.generic' operation with
// an indexing map that contains a reversal. Such linalg operations are
// not currently tileable, and upstream may crash when attempting to tile
// them.
//


// CHECK-LABEL: @dont_tile_reversed_dimensions
func.func @dont_tile_reversed_dimensions(%arg0: tensor<3x8xi8>) -> tensor<3xi32> {
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi8>
  %c0_i32 = arith.constant 0 : i32
  %c0_i8 = arith.constant 0 : i8
  %c8_i8 = arith.constant 8 : i8
  %0 = tensor.empty() : tensor<3xi32>
  %1 = tensor.empty() : tensor<8xi8>
  // CHECK: linalg.generic
  %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%cst : tensor<8xi8>) outs(%1 : tensor<8xi8>) {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  } -> tensor<8xi8>
  // CHECK: scf.forall
  %3 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<3xi32>) -> tensor<3xi32>
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: scf.forall.in_parallel
  %4 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]}
    ins(%arg0, %2 : tensor<3x8xi8>, tensor<8xi8>) outs(%3 : tensor<3xi32>)
    attrs = {kernel.root} {
  ^bb0(%in: i8, %in_0: i8, %out: i32):
    %5 = arith.shli %in, %in_0 : i8
    %6 = arith.cmpi ult, %in_0, %c8_i8 : i8
    %7 = arith.select %6, %5, %c0_i8 : i8
    %8 = arith.extsi %7 : i8 to i32
    %9 = arith.addi %out, %8 : i32
    linalg.yield %9 : i32
  } -> tensor<3xi32>
  return %4 : tensor<3xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops{["linalg.generic"]} attributes {kernel.root} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0 num_threads [1, 1](mapping = [#gpu.block<x>, #gpu.block<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %1 = transform.kernel.fuse_greedily_op %tiled_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2, d2 * 2 + d4, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
func.func @test_multiple_fuse_shared_outs(%arg0: tensor<f32>, %arg1: tensor<f32>,
                                     %arg2: tensor<5x2x10x11xf32>, %arg3: tensor<5x2x10x11xf32>)
                                     -> (tensor<5x1x5x11xf32>, tensor<5x1x5x11xf32>) {
  %0 = tensor.empty() : tensor<2xf32>
  %1 = tensor.empty() : tensor<5x1x5x11xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<f32>) outs(%1 : tensor<5x1x5x11xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<5x1x5x11xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg1 : tensor<f32>) outs(%1 : tensor<5x1x5x11xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<5x1x5x11xf32>
  %4:2 = linalg.generic {indexing_maps = [#map2, #map2, #map3, #map4, #map4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
    ins(%arg2, %arg3, %0 : tensor<5x2x10x11xf32>, tensor<5x2x10x11xf32>, tensor<2xf32>)
    outs(%2, %3 : tensor<5x1x5x11xf32>, tensor<5x1x5x11xf32>) attrs =  {kernel.root} {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32, %out_2: f32):
    %5 = arith.addf %out, %in : f32
    %6 = arith.addf %out_2, %in_0 : f32
    linalg.yield %5, %6 : f32, f32
  } -> (tensor<5x1x5x11xf32>, tensor<5x1x5x11xf32>)
  return %4#0, %4#1 : tensor<5x1x5x11xf32>, tensor<5x1x5x11xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %4 = transform.structured.match interface{LinalgOp} attributes {kernel.root} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %4 num_threads [1, 1, 1, 1, 0](
      mapping = [#gpu.block<linear_dim_0>, #gpu.block<linear_dim_1>, #gpu.block<linear_dim_2>,
                 #gpu.block<linear_dim_3>]
    ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %5 = transform.kernel.fuse_greedily_op %tiled_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0, d1, d2, d3) -> ()>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//   CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2, d2 * 2 + d4, d3)>
//   CHECK-DAG: #[[$map3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
//   CHECK-DAG: #[[$map4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @test_multiple_fuse_shared_outs
//  CHECK-SAME: (%[[arg0:.+]]: tensor<f32>, %[[arg1:.+]]: tensor<f32>, %[[arg2:.+]]: tensor<5x2x10x11xf32>, %[[arg3:.+]]: tensor<5x2x10x11xf32>)
//   CHECK-DAG:     %[[v0:.+]] = tensor.empty() : tensor<2xf32>
//   CHECK-DAG:     %[[v1:.+]] = tensor.empty() : tensor<5x1x5x11xf32>
//       CHECK:     scf.forall ({{.*}}) in (1, 1, 1, 1) shared_outs(%[[arg9:.+]] = %[[v1]], %[[arg10:.+]] = %[[v1]])
//   CHECK-DAG:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg2]][0, 0, 0, 0] [5, 1, 10, 11] [1, 1, 1, 1] : tensor<5x2x10x11xf32> to tensor<5x1x10x11xf32>
//   CHECK-DAG:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg3]][0, 0, 0, 0] [5, 1, 10, 11] [1, 1, 1, 1] : tensor<5x2x10x11xf32> to tensor<5x1x10x11xf32>
//   CHECK-DAG:       %[[v3:.+]] = linalg.generic {{.*}} ins(%[[arg0]] : tensor<f32>) outs(%[[arg9]] : tensor<5x1x5x11xf32>)
//   CHECK-DAG:       %[[v4:.+]] = linalg.generic {{.*}} ins(%[[arg1]] : tensor<f32>) outs(%[[arg10]] : tensor<5x1x5x11xf32>)
//   CHECK-DAG:       %[[v5]]:2 = linalg.generic {{.*}} ins(%[[extracted_slice]], %[[extracted_slice_0]], %[[v0]] : {{.*}}) outs(%[[v3]], %[[v4]] :
//   CHECK-DAG:       scf.forall.in_parallel {
//   CHECK-DAG:         tensor.parallel_insert_slice %[[v5]]#0 into %[[arg9]][0, 0, 0, 0] [5, 1, 5, 11] [1, 1, 1, 1] : tensor<5x1x5x11xf32> into tensor<5x1x5x11xf32>
//   CHECK-DAG:         tensor.parallel_insert_slice %[[v5]]#1 into %[[arg10]][0, 0, 0, 0] [5, 1, 5, 11] [1, 1, 1, 1] : tensor<5x1x5x11xf32> into tensor<5x1x5x11xf32>

