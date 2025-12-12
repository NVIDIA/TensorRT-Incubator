// RUN: kernel-opt %s -split-input-file -transform-interpreter | FileCheck %s


func.func @kernel(%arg0: tensor<128x32xf32>,
                  %arg1: tensor<32x128xf32>,
                  %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c2 = arith.constant 2: index
  %0 = scf.forall (%i, %j) in (%c2, %c2) shared_outs(%out = %arg2) -> tensor<128x128xf32> {
    %row = affine.apply affine_map<()[s0]->(s0 * 64)>()[%i]
    %col = affine.apply affine_map<()[s0]->(s0 * 64)>()[%j]
    %a = tensor.extract_slice %arg0[%row, 0][64, 32][1, 1] : tensor<128x32xf32> to tensor<64x32xf32>
    %b = tensor.extract_slice %arg1[0, %col][32, 64][1, 1] : tensor<32x128xf32> to tensor<32x64xf32>
    %c = tensor.extract_slice %arg2[%row, %col][64, 64][1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
    %d = linalg.matmul ins(%a, %b : tensor<64x32xf32>, tensor<32x64xf32>) outs(%c: tensor<64x64xf32>)
      -> tensor<64x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %d into %out[%row, %col][64, 64][1, 1] :
        tensor<64x64xf32> into tensor<128x128xf32>
    }
  }
  return %0 : tensor<128x128xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %forall_op = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %warp_id = transform.kernel.forall_to_subgroups %forall_op  subgroup_size(32)
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<()[s0] -> ((s0 mod 128) floordiv 32)>
//       CHECK: #[[$map1:.+]] = affine_map<()[s0] -> (s0 * 64)>
// CHECK-LABEL: @kernel
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128x32xf32>, %[[arg1:.+]]: tensor<32x128xf32>, %[[arg2:.+]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
//       CHECK:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[v0:.+]] = gpu.thread_id  x
//       CHECK:     %[[v1:.+]] = affine.apply #[[$map]]()[%[[v0]]]
//       CHECK:     %[[v2:.+]]:2 = affine.delinearize_index %[[v1]] into (2, 2) : index, index
//       CHECK:     %[[v3:.+]] = affine.apply #[[$map1]]()[%[[v2]]#0]
//       CHECK:     %[[v4:.+]] = affine.apply #[[$map1]]()[%[[v2]]#1]
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[arg0]][%[[v3]], 0] [64, 32] [1, 1] : tensor<128x32xf32> to tensor<64x32xf32>
//       CHECK:     %[[extracted_slice_2:.+]] = tensor.extract_slice %[[arg1]][0, %[[v4]]] [32, 64] [1, 1] : tensor<32x128xf32> to tensor<32x64xf32>
//       CHECK:     %[[extracted_slice_3:.+]] = tensor.extract_slice %[[arg2]][%[[v3]], %[[v4]]] [64, 64] [1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
//       CHECK:     %[[v5:.+]] = linalg.matmul ins(%[[extracted_slice]], %[[extracted_slice_2]] : tensor<64x32xf32>, tensor<32x64xf32>) outs(%[[extracted_slice_3]] : tensor<64x64xf32>) -> tensor<64x64xf32>
//       CHECK:     %[[inserted_slice:.+]] = tensor.insert_slice %[[v5]] into %[[arg2]][%[[v3]], %[[v4]]] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x128xf32>
//       CHECK:     return %[[inserted_slice]] : tensor<128x128xf32>

// -----

func.func @kernel(%arg0: tensor<?xf32>,
                  %arg1: tensor<?xf32>,
                  %arg2: index) -> tensor<?xf32> {
  %c2 = arith.constant 2: index
  %limit = affine.min affine_map<(d0)[]->(d0 floordiv 64)>(%arg2)
  %0 = scf.forall (%i) in (%limit) shared_outs(%out = %arg1) -> tensor<?xf32> {
    %offt = affine.apply affine_map<(d0)->(d0*64)>(%i)
    %d = tensor.extract_slice %arg0[%offt][64][1] : tensor<?xf32> to tensor<64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %d into %out[%offt][64][1] :
        tensor<64xf32> into tensor<?xf32>
    }
  }
  return %0 : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %forall_op = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %warp_id = transform.kernel.forall_to_subgroups %forall_op  subgroup_size(32)
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<(d0) -> (d0 floordiv 64)>
//       CHECK: #[[$map1:.+]] = affine_map<()[s0, s1] -> ((s0 mod (s1 * 32)) floordiv 32)>
//       CHECK: #[[$map2:.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-LABEL: @kernel
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<?xf32>, %[[arg2:.+]]: index) -> tensor<?xf32>
//       CHECK:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[v0:.+]] = affine.min #[[$map]](%[[arg2]])
//       CHECK:     %[[v1:.+]] = gpu.thread_id  x
//       CHECK:     %[[v2:.+]] = affine.apply #[[$map1]]()[%[[v1]], %[[v0]]]
//       CHECK:     %[[v3:.+]] = affine.delinearize_index %[[v2]] into (%[[v0]]) : index
//       CHECK:     %[[v4:.+]] = affine.apply #[[$map2]](%[[v3]])
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[arg0]][%[[v4]]] [64] [1] : tensor<?xf32> to tensor<64xf32>
//       CHECK:     %[[inserted_slice:.+]] = tensor.insert_slice %[[extracted_slice]] into %[[arg1]][%[[v4]]] [64] [1] : tensor<64xf32> into tensor<?xf32>
//       CHECK:     return %[[inserted_slice]] : tensor<?xf32>

// -----

#map = affine_map<(d0) -> (d0 * 512)>
#map1 = affine_map<(d0) -> (d0 * 128)>
#map2 = affine_map<(d0) -> (d0 * 32)>
#map3 = affine_map<(d0) -> (d0 * 4)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
gpu.module @kernels {
  func.func @trt_unary_multi_dim_kernel(%arg0: tensor<2x1024x1024x32xf32>, %arg1: tensor<2x1024x1024x32xf32>) -> tensor<2x1024x1024x32xf32> {
    %0 = gpu.block_id  x
    %1 = gpu.block_id  y
    %2 = gpu.block_id  z
    %cst = arith.constant 5.000000e+00 : f32
    %3 = affine.apply #map(%2)
    %extracted_slice = tensor.extract_slice %arg1[%0, %1, %3, 0] [1, 1, 512, 32] [1, 1, 1, 1] : tensor<2x1024x1024x32xf32> to tensor<1x1x512x32xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%extracted_slice : tensor<1x1x512x32xf32>) -> tensor<1x1x512x32xf32>
    %5 = scf.forall (%arg2) in (4) shared_outs(%arg3 = %extracted_slice) -> (tensor<1x1x512x32xf32>) {
      %6 = affine.apply #map1(%arg2)
      %extracted_slice_0 = tensor.extract_slice %4[0, 0, %6, 0] [1, 1, 128, 32] [1, 1, 1, 1] : tensor<1x1x512x32xf32> to tensor<1x1x128x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[0, 0, %6, 0] [1, 1, 128, 32] [1, 1, 1, 1] : tensor<1x1x512x32xf32> to tensor<1x1x128x32xf32>
      %7 = scf.forall (%arg4, %arg5) in (4, 8) shared_outs(%arg6 = %extracted_slice_1) -> (tensor<1x1x128x32xf32>) {
        %8 = affine.apply #map2(%arg4)
        %9 = affine.apply #map3(%arg5)
        %extracted_slice_2 = tensor.extract_slice %extracted_slice_0[0, 0, %8, %9] [1, 1, 32, 4] [1, 1, 1, 1] : tensor<1x1x128x32xf32> to tensor<1x1x32x4xf32>
        %extracted_slice_3 = tensor.extract_slice %arg6[0, 0, %8, %9] [1, 1, 32, 4] [1, 1, 1, 1] : tensor<1x1x128x32xf32> to tensor<1x1x32x4xf32>
        %10 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_2 : tensor<1x1x32x4xf32>) outs(%extracted_slice_3 : tensor<1x1x32x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %11 = arith.negf %in : f32
          linalg.yield %11 : f32
        } -> tensor<1x1x32x4xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %10 into %arg6[0, 0, %8, %9] [1, 1, 32, 4] [1, 1, 1, 1] : tensor<1x1x32x4xf32> into tensor<1x1x128x32xf32>
        }
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %7 into %arg3[0, 0, %6, 0] [1, 1, 128, 32] [1, 1, 1, 1] : tensor<1x1x128x32xf32> into tensor<1x1x512x32xf32>
      }
    }
    %inserted_slice = tensor.insert_slice %5 into %arg1[%0, %1, %3, 0] [1, 1, 512, 32] [1, 1, 1, 1] : tensor<1x1x512x32xf32> into tensor<2x1024x1024x32xf32>
    return %inserted_slice : tensor<2x1024x1024x32xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %4 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %5 = transform.kernel.forall_to_subgroups %4 subgroup_size(32)
       : (!transform.any_op) -> !transform.any_op
    %6 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %7 = transform.kernel.forall_to_subgroups %6 subgroup_size(1)
       : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//       CHECK: #[[$map2:.+]] = affine_map<()[s0] -> ((s0 mod 128) floordiv 32)>
//       CHECK: #[[$map5:.+]] = affine_map<()[s0] -> (s0 mod 32)>
// CHECK-LABEL: @trt_unary_multi_dim_kernel
//       CHECK:    %[[v1:.+]] = gpu.thread_id  x
//       CHECK:    %[[v2:.+]] = affine.apply #[[$map2]]()[%[[v1]]]
//       CHECK:    %[[v3:.+]] = affine.delinearize_index %[[v2]] into (4) : index

//       CHECK:    %[[v4:.+]] = gpu.thread_id  x
//       CHECK:    %[[v5:.+]] = affine.apply #[[$map5]]()[%[[v4]]]
//       CHECK:    %[[v6:.+]] = affine.delinearize_index %[[v5]] into (4, 8) :
