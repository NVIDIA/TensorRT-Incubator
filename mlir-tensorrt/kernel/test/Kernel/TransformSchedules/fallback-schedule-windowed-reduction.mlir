// RUN: kernel-opt -split-input-file %s -kernel-initial-transform-schedule="generator-benefit={fallback:100} device-compute-capability=80 device-max-smem-per-block=48 device-max-registers-per-block=65536" -verify-diagnostics | FileCheck %s
// RUN: kernel-opt -split-input-file %s -kernel-linalg-codegen-pipeline="generator-benefit={fallback:100} device-compute-capability=80 device-max-smem-per-block=48 device-max-registers-per-block=65536" | FileCheck %s --check-prefix=E2E

!input_type = tensor<2x128x128x3xf32>
!out_type = tensor<2x64x64x3xf32>

module @pooling_nhwc_max_simple {
func.func @kernel(%input: !input_type) -> !out_type {
  %fake = tensor.empty() : tensor<3x3xf32>
  %init = tensor.empty() : !out_type
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : !out_type) -> !out_type
  %res = linalg.pooling_nhwc_max {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>,
    kernel.root}
    ins(%input, %fake: !input_type, tensor<3x3xf32>)
    outs(%fill: !out_type) -> !out_type
  return %res : !out_type
}
}

// CHECK-LABEL: module @pooling_nhwc_max_simple
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.pooling_nhwc_max
//  CHECK-SAME:  kernel.parameters =
//  CHECK-SAME:    #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [2, 16, 16, 3, 3, 3], [2, 16, 16, 3, 3, 3], [2, 1, 1, 1, 3, 3], [1, 4, 4, 1, 1, 1], [1, 16, 16, 3, 1, 1]>

// E2E-LABEL: module @pooling_nhwc_max_simple
//       E2E: gpu.module.kernels.ptx_data

// -----

!input_type = tensor<2x128x128x3xf32>
!out_type = tensor<2x128x128x3xf32>

module @pooling_nhwc_contrived {
func.func @kernel(%input: !input_type) -> !out_type {
  %fake = tensor.empty() : tensor<1x1xf32>
  %init = tensor.empty() : !out_type
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : !out_type) -> !out_type
  %res = linalg.pooling_nhwc_max {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>,
    kernel.root}
    ins(%input, %fake: !input_type, tensor<1x1xf32>)
    outs(%fill: !out_type) -> !out_type
  return %res : !out_type
}
}

// CHECK-LABEL: module @pooling_nhwc_contrived
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.pooling_nhwc_max
//  CHECK-SAME:  kernel.parameters =
//  CHECK-SAME:    #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [2, 16, 16, 3, 1, 1], [2, 16, 16, 3, 1, 1], [2, 1, 1, 1, 1, 1], [1, 8, 8, 1, 1, 1], [1, 16, 16, 3, 1, 1]>

// E2E-LABEL: module @pooling_nhwc_contrived
//       E2E: gpu.module.kernels.ptx_data

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2, d3 * 2 + d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>

!input_type = tensor<3x6x3x15x13xf32>
!out_type = tensor<3x6x2x7x13xf32>

module @window_min_pool {
func.func @kernel(%arg0: !input_type) -> !out_type {
  %0 = tensor.empty() : tensor<2xf32>
  %1 = tensor.empty() : !out_type
  %cst = arith.constant 0x7F800000 : f32
  %2 = linalg.generic {
      indexing_maps = [#map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    outs(%1 : !out_type) {
  ^bb0(%out: f32):
      linalg.yield %cst : f32
  } -> !out_type
  %3 = linalg.generic {
      indexing_maps = [#map1, #map2, #map3],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]}
    ins(%arg0, %0 : !input_type, tensor<2xf32>)
    outs(%2 : !out_type) attrs =  {kernel.root} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.minimumf %out, %in : f32
      linalg.yield %4 : f32
  } -> !out_type
  return %3 : !out_type
}
}

// CHECK-LABEL: module @window_min_pool
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.generic
//       CHECK:  kernel.parameters =
//  CHECK-SAME:    #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [3, 6, 2, 7, 13, 2], [3, 6, 2, 7, 13, 2], [3, 2, 1, 1, 1, 2], [1, 1, 1, 1, 1, 1], [1, 3, 2, 7, 13, 1]

// E2E-LABEL: module @window_min_pool
//       E2E: gpu.module.kernels.ptx_data

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#map11 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 7 + d4, d3 * 7 + d5)>

builtin.module @fuse_input_chain {
  func.func @kernel(%arg0: tensor<1x512x7x7xf32>, %arg1: tensor<512x1x1xf32>, %arg2: tensor<512x1x1xf32>, %arg3: tensor<512x1x1xf32>, %arg4: tensor<512x1x1xf32>, %arg5: tensor<1x512x7x7xf32>) -> tensor<1x512x1x1xf32> {
    %0 = tensor.empty() : tensor<7x7xf32>
    %1 = tensor.empty() : tensor<1x512x1x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = tensor.empty() : tensor<1x512x7x7xf32>
    %3 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%2 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %11 = arith.subf %in, %in_0 : f32
      linalg.yield %11 : f32
    } -> tensor<1x512x7x7xf32>
    %4 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %arg2 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%2 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %11 = arith.mulf %in, %in_0 : f32
      linalg.yield %11 : f32
    } -> tensor<1x512x7x7xf32>
    %5 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %arg3 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%2 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %11 = arith.mulf %in, %in_0 : f32
      linalg.yield %11 : f32
    } -> tensor<1x512x7x7xf32>
    %6 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %arg4 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%2 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %11 = arith.addf %in, %in_0 : f32
      linalg.yield %11 : f32
    } -> tensor<1x512x7x7xf32>
    %7 = linalg.generic {indexing_maps = [#map5, #map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %arg5 : tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) outs(%2 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %11 = arith.addf %in, %in_0 : f32
      linalg.yield %11 : f32
    } -> tensor<1x512x7x7xf32>
    %8 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<1x512x7x7xf32>) outs(%2 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.cmpf ugt, %in, %cst : f32
      %12 = arith.select %11, %in, %cst : f32
      linalg.yield %12 : f32
    } -> tensor<1x512x7x7xf32>
    %9 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<1x512x1x1xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<1x512x1x1xf32>
    %10 = linalg.generic {indexing_maps = [#map11, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%8, %0 : tensor<1x512x7x7xf32>, tensor<7x7xf32>) outs(%9 : tensor<1x512x1x1xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %11 = arith.addf %out, %in : f32
      linalg.yield %11 : f32
    } -> tensor<1x512x1x1xf32>
    return %10 : tensor<1x512x1x1xf32>
  }
}

// CHECK-LABEL: module @fuse_input_chain
// CHECK-LABEL: func.func @kernel

// E2E-LABEL: @fuse_input_chain
//       E2E: gpu.module.kernels.ptx_data

// -----

// Regression test for a bug where we were not promoting allowing the thread-level
// tile to be promoted using alloca under previous heuristics.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#map11 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 7 + d4, d3 * 7 + d5)>

builtin.module @windowed_reduction_local_alloc {
func.func @kernel(%arg0: tensor<1x512x7x7xf32>, %arg1: tensor<512x1x1xf32>,
      %arg2: tensor<512x1x1xf32>, %arg3: tensor<512x1x1xf32>,
      %arg4: tensor<512x1x1xf32>, %arg5: tensor<1x512x7x7xf32>)
      -> tensor<1x512x1x1xf32> {
  %0 = tensor.empty() : tensor<7x7xf32>
  %1 = tensor.empty() : tensor<1x512x1x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %2 = tensor.empty() : tensor<1x512x7x7xf32>
  %3 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%2 : tensor<1x512x7x7xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %11 = arith.subf %in, %in_0 : f32
    linalg.yield %11 : f32
  } -> tensor<1x512x7x7xf32>
  %4 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%3, %arg2 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%2 : tensor<1x512x7x7xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %11 = arith.mulf %in, %in_0 : f32
    linalg.yield %11 : f32
  } -> tensor<1x512x7x7xf32>
  %5 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%4, %arg3 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%2 : tensor<1x512x7x7xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %11 = arith.mulf %in, %in_0 : f32
    linalg.yield %11 : f32
  } -> tensor<1x512x7x7xf32>
  %6 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%5, %arg4 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%2 : tensor<1x512x7x7xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %11 = arith.addf %in, %in_0 : f32
    linalg.yield %11 : f32
  } -> tensor<1x512x7x7xf32>
  %7 = linalg.generic {indexing_maps = [#map5, #map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%6, %arg5 : tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) outs(%2 : tensor<1x512x7x7xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %11 = arith.addf %in, %in_0 : f32
    linalg.yield %11 : f32
  } -> tensor<1x512x7x7xf32>
  %8 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%7 : tensor<1x512x7x7xf32>) outs(%2 : tensor<1x512x7x7xf32>) {
  ^bb0(%in: f32, %out: f32):
    %11 = arith.cmpf ugt, %in, %cst : f32
    %12 = arith.select %11, %in, %cst : f32
    linalg.yield %12 : f32
  } -> tensor<1x512x7x7xf32>
  %9 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    outs(%1 : tensor<1x512x1x1xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst : f32
  } -> tensor<1x512x1x1xf32>
  %10 = linalg.generic {
      indexing_maps = [#map11, #map8, #map9],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%8, %0 : tensor<1x512x7x7xf32>, tensor<7x7xf32>) outs(%9 : tensor<1x512x1x1xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %11 = arith.addf %out, %in : f32
    linalg.yield %11 : f32
  } -> tensor<1x512x1x1xf32>
  return %10 : tensor<1x512x1x1xf32>
}
}

// CHECK-LABEL: module @windowed_reduction_local_alloc
// CHECK-LABEL: func.func @kernel

// E2E-LABEL: @windowed_reduction_local_alloc
//       E2E: gpu.module.kernels.ptx_data
