// RUN: kernel-opt -split-input-file %s -kernel-initial-transform-schedule="generator-benefit={fallback:100} device-compute-capability=80 device-max-smem-per-block=48 device-max-registers-per-block=65536" -verify-diagnostics | FileCheck %s
// RUN: kernel-opt -split-input-file %s -kernel-linalg-codegen-pipeline="generator-benefit={fallback:100} device-compute-capability=80 device-max-smem-per-block=48 device-max-registers-per-block=65536" | FileCheck %s --check-prefix=E2E
!tensor_input_lg = tensor<32x32768xf32>
!tensor_reduced = tensor<32xf32>

module @reduction_one_parallel_dim_small {
  func.func @kernel(%arg0: !tensor_input_lg,
                                        %arg1: !tensor_reduced) -> !tensor_reduced {
    %cst = arith.constant 0.0 : f32
    %out = linalg.fill ins(%cst: f32) outs(%arg1: !tensor_reduced) -> !tensor_reduced
    %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%arg0 : !tensor_input_lg)
    outs(%out: !tensor_reduced)
    attrs =  {kernel.root} {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.mulf %arg2, %arg2 : f32
      %3 = arith.addf %arg3, %2 : f32
      linalg.yield %3 : f32
    } -> !tensor_reduced
    return %1 : !tensor_reduced
  }
}

// CHECK-LABEL: module @reduction_one_parallel_dim_small
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.generic
//  CHECK-SAME:  kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [32, 32768], [32, 64], [1, 64], [1, 1], [32, 1]>

// E2E-LABEL: @reduction_one_parallel_dim_small
//       E2E: gpu.module.kernels.ptx_data

// -----

!tensor_input_lg = tensor<32768x32768xf32>
!tensor_reduced = tensor<32768xf32>

module @reduction_one_parallel_dim_large {
  func.func @kernel(%arg0: !tensor_input_lg,
                    %arg1: !tensor_reduced) -> !tensor_reduced {
    %cst = arith.constant 0.0 : f32
    %out = linalg.fill ins(%cst: f32) outs(%arg1: !tensor_reduced) -> !tensor_reduced
    %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%arg0 : !tensor_input_lg)
    outs(%out: !tensor_reduced)
    attrs =  {kernel.root} {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.mulf %arg2, %arg2 : f32
      %3 = arith.addf %arg3, %2 : f32
      linalg.yield %3 : f32
    } -> !tensor_reduced
    return %1 : !tensor_reduced
  }
}


// CHECK-LABEL: module @reduction_one_parallel_dim_large
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.generic
//  CHECK-SAME:  kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [128, 32768], [128, 64], [1, 64], [256, 1], [128, 1]>

// E2E-LABEL: @reduction_one_parallel_dim_large
//       E2E: gpu.module.kernels.ptx_data

// -----

#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 + d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

module @reduce_window {
  func.func @kernel(%arg0: tensor<1024x1024x133xf32>,
                           %arg1: tensor<6xf32>,
                           %arg2: tensor<1024x1024x128xf32>)
                        -> tensor<1024x1024x128xf32> {
    %3 = linalg.generic {
      indexing_maps = [#map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    }
      ins(%arg0, %arg1 : tensor<1024x1024x133xf32>, tensor<6xf32>)
      outs(%arg2 : tensor<1024x1024x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %out, %in : f32
      linalg.yield %4 : f32
    } -> tensor<1024x1024x128xf32>
    return %3 : tensor<1024x1024x128xf32>
  }
}

// CHECK-LABEL: module @reduce_window
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.generic
//  CHECK-SAME:  kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [16, 16, 16, 6], [16, 16, 16, 6], [2, 2, 2, 6], [64, 64, 8, 1], [8, 8, 8, 1]>

// E2E-LABEL: @reduce_window
//       E2E: gpu.module.kernels.ptx_data

// -----

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0)>

module @tile_multiple_reduction {
  func.func @kernel(%arg0: tensor<86x128xf32>, %arg1: tensor<4096x86x128xf32>, %arg2: tensor<4096xf32>) -> tensor<4096xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1 : tensor<86x128xf32>, tensor<4096x86x128xf32>) outs(%arg2 : tensor<4096xf32>) attrs =  {kernel.root} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %1, %out : f32
      linalg.yield %2 : f32
    } -> tensor<4096xf32>
    return %0 : tensor<4096xf32>
  }
}

// CHECK-LABEL: module @tile_multiple_reduction {
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.generic
//  CHECK-SAME:  kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [64, 86, 128], [64, 2, 64], [1, 2, 64], [64, 1, 1], [64, 1, 1]>

// E2E-LABEL: @tile_multiple_reduction
//       E2E: gpu.module.kernels.ptx_data

// -----

module @tile_multiple_reduction2 {
  func.func @kernel(%collapsed_37: tensor<80x3002xf32>, %collapsed: tensor<384x80x3xf32>, %1 : tensor<384x3000xf32>) -> tensor<384x3000xf32>{
    %2 = linalg.generic
      {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d2 + d3)>,
          affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d1, d2)>
        ], iterator_types = ["reduction", "parallel", "parallel", "reduction"]
      }
      ins(%collapsed_37, %collapsed : tensor<80x3002xf32>, tensor<384x80x3xf32>) outs(%1 : tensor<384x3000xf32>) attrs =  {cluster.root = 0 : i64, kernel.root} {
    ^bb0(%in: f32, %in_53: f32, %out: f32):
      %171 = arith.mulf %in, %in_53 : f32
      %172 = arith.addf %out, %171 : f32
      linalg.yield %172 : f32
    } -> tensor<384x3000xf32>
    return %2 : tensor<384x3000xf32>
  }
}

// CHECK-LABEL: module @tile_multiple_reduction2 {
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.generic
//  CHECK-SAME:  kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [80, 16, 15, 3], [8, 16, 15, 3], [8, 1, 1, 3], [1, 24, 200, 1], [1, 16, 15, 1]>

// E2E-LABEL: @tile_multiple_reduction2
//       E2E: gpu.module.kernels.ptx_data

// -----

#map = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1)>

module @tile_multiple_reduction_3 {
  func.func @kernel(%arg0: tensor<4096x86x128xf32>, %arg2: tensor<4096xf32>)
       -> tensor<4096xf32> {
    %0 = linalg.generic {
      indexing_maps = [#map, #map2],
      iterator_types = ["reduction", "parallel", "reduction"]}
      ins(%arg0 : tensor<4096x86x128xf32>)
      outs(%arg2 : tensor<4096xf32>) attrs =  {kernel.root} {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
    } -> tensor<4096xf32>
    return %0 : tensor<4096xf32>
  }
}

// CHECK-LABEL: module @tile_multiple_reduction_3
//       CHECK: linalg.generic
//  CHECK-SAME: kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [86, 32, 128], [2, 32, 16], [2, 1, 16], [1, 128, 1], [1, 32, 1]>
// E2E-LABEL: @tile_multiple_reduction_3
//       E2E: gpu.module.kernels.ptx_data

// -----

#map4 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0)>

module @small_multi_reduction {
  func.func @kernel(%arg0: tensor<3x2x3xf32>) -> tensor<2xf32> {
    %0 = tensor.empty() : tensor<2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2xf32>) -> tensor<2xf32>
    %2 = linalg.generic {
      indexing_maps = [#map4, #map5],
      iterator_types = ["parallel", "reduction", "reduction"]
    } ins(%arg0 : tensor<3x2x3xf32>) outs(%1 : tensor<2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<2xf32>
    return %2 : tensor<2xf32>
  }
}

// CHECK-LABEL: module @small_multi_reduction {
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.generic
//  CHECK-SAME:  kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [2, 3, 3], [2, 3, 3], [1, 3, 3], [1, 1, 1], [2, 1, 1]>

// E2E-LABEL: @small_multi_reduction
//       E2E: gpu.module.kernels.ptx_data


// -----

module @vec_mat_fusion_case {
  func.func @kernel(%arg0: tensor<512xf32>, %arg1: tensor<1024x512xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<512xf32>
    %cst = arith.constant 24.0 : f32
    %1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
      ins(%arg0 : tensor<512xf32>)
      outs (%0 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.divf %in, %cst : f32
      linalg.yield %2 : f32
    } -> tensor<512xf32>

    %e = tensor.empty() : tensor<1024xf32>
    %2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%1, %arg1 : tensor<512xf32>, tensor<1024x512xf32>)
      outs(%e : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<1024xf32>

    return %2 : tensor<1024xf32>
  }
}

// CHECK-LABEL: module @vec_mat_fusion_case
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.generic
//       CHECK:  linalg.generic
//  CHECK-SAME:  kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [64, 512], [64, 64], [1, 64], [16, 1], [64, 1]>

// E2E-LABEL: @vec_mat_fusion_case
//       E2E: gpu.module.kernels.ptx_data
