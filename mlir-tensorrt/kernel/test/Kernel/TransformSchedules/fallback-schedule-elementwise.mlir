// RUN: kernel-opt -split-input-file %s -kernel-initial-transform-schedule="generator-benefit={fallback:100} device-compute-capability=80 device-max-smem-per-block=48 device-max-registers-per-block=65536" -verify-diagnostics | FileCheck %s
// RUN: kernel-opt -split-input-file %s -kernel-linalg-codegen-pipeline="generator-benefit={fallback:100} device-compute-capability=80 device-max-smem-per-block=48 device-max-registers-per-block=65536" | FileCheck %s --check-prefix=E2E

builtin.module @fallback_elementwise_unary_neg_small {
  func.func @kernel(%arg0: tensor<1024xf32>,
                      %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %1 = linalg.map {arith.negf}
      ins(%arg0 : tensor<1024xf32>)
      outs(%arg1 : tensor<1024xf32>)

    return %1 : tensor<1024xf32>
  }
}

// CHECK-LABEL: module @fallback_elementwise_unary_neg_small
// CHECK-LABEL: func.func @kernel
//       CHECK:     linalg.map
// CHECK-SAME: kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [1024], [1024], [8], [1], [128]>

// E2E-LABEL: @fallback_elementwise_unary_neg_small
//       E2E: gpu.module.kernels.ptx_data

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @unary_medium {
  func.func @kernel(%arg0: tensor<2x1024x1024x32xf32>,
                     %arg1: tensor<2x1024x1024x32xf32>) -> tensor<2x1024x1024x32xf32> {
    %0 = tensor.empty() : tensor<2x1024x1024x32xf32>
    %1 = tensor.empty() : tensor<2x1024x1024x32xf32>
    %cst = arith.constant 5.000000e+00 : f32
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x1024x1024x32xf32>) outs(%arg1 : tensor<2x1024x1024x32xf32>) attrs =  {kernel.root} {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.negf %in : f32
      linalg.yield %3 : f32
    } -> tensor<2x1024x1024x32xf32>
  return %3 : tensor<2x1024x1024x32xf32>
  }
}

// CHECK-LABEL: module @unary_medium

// E2E-LABEL: @unary_medium
//       E2E: gpu.module.kernels.ptx_data

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module @unary_odd_dims {
  func.func @kernel() -> tensor<1024x40x3x5x3x32xf32> {
    %0 = tensor.empty() : tensor<1024x40x3x5x3x32xf32>
    %1 = tensor.empty() : tensor<1024x40x3x5x3x32xf32>
  %cst = arith.constant 5.000000e+00 : f32
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1024x40x3x5x3x32xf32>) -> tensor<1024x40x3x5x3x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2: tensor<1024x40x3x5x3x32xf32>) outs(%0 : tensor<1024x40x3x5x3x32xf32>) attrs =  {kernel.root} {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.negf %in : f32
    linalg.yield %3 : f32
  } -> tensor<1024x40x3x5x3x32xf32>
  return %3 : tensor<1024x40x3x5x3x32xf32>
 }
}

// CHECK-LABEL: module @unary_odd_dims
//       CHECK: #kernel.fallback_parameters<
//  CHECK-SAME: [16, 10, 3, 5, 3, 32], [16, 10, 3, 5, 3, 32], [4, 2, 3, 1, 3, 4], [64, 4, 1, 1, 1, 1], [4, 5, 1, 5, 1, 8]

// E2E-LABEL: @unary_odd_dims
//       E2E: gpu.module.kernels.ptx_data

// -----

!tensor_type_large = tensor<1024x2048x16768x8192xf32>

builtin.module @fallback_elementwise_add_large {
  func.func @kernel(%arg0: !tensor_type_large,
                    %arg1: !tensor_type_large,
                    %arg2: !tensor_type_large) -> !tensor_type_large {
    %1 = linalg.map {arith.addf}
      ins(%arg0, %arg1 : !tensor_type_large, !tensor_type_large)
      outs(%arg2 : !tensor_type_large)
    return %1 : !tensor_type_large
  }
}

// CHECK-LABEL: module @fallback_elementwise_add_large
// CHECK-LABEL: func.func @kernel
//       CHECK:     linalg.map
//  CHECK-SAME: kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [16, 16, 16, 32], [16, 16, 16, 32], [4, 4, 4, 8], [64, 128, 1048, 256], [4, 4, 4, 4]>

// E2E-LABEL: @fallback_elementwise_add_large
//       E2E: gpu.module.kernels.ptx_data

// -----

!tensor_type_med = tensor<1024x2048xf32>

builtin.module @fallback_elementwise_add_med {
  func.func @kernel(%arg0: !tensor_type_med,
                    %arg1: !tensor_type_med,
                    %arg2: !tensor_type_med) -> !tensor_type_med {
    %1 = linalg.map {arith.addf}
      ins(%arg0, %arg1 : !tensor_type_med, !tensor_type_med)
      outs(%arg2 : !tensor_type_med)
    return %1 : !tensor_type_med
  }
}

// CHECK-LABEL: module @fallback_elementwise_add_med
// CHECK-LABEL: func.func @kernel
//       CHECK:   linalg.map
//  CHECK-SAME:   kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [32, 64], [32, 64], [2, 4], [32, 32], [16, 16]>

// E2E-LABEL: @fallback_elementwise_add_med
//       E2E: gpu.module.kernels.ptx_data

// -----

!tensor_type_scalar = tensor<f32>

builtin.module @fallback_elementwise_add_scalar {
  func.func @kernel(%arg0: !tensor_type_scalar,
                    %arg1: !tensor_type_scalar,
                    %arg2: !tensor_type_scalar) -> !tensor_type_scalar {
    %1 = linalg.map {arith.addf}
      ins(%arg0, %arg1 : !tensor_type_scalar, !tensor_type_scalar)
      outs(%arg2 : !tensor_type_scalar)
    return %1 : !tensor_type_scalar
  }
}

// CHECK-LABEL: module @fallback_elementwise_add_scalar
// CHECK-LABEL: func.func @kernel
//       CHECK:   linalg.map
// CHECK-SAME:       kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [], [], [], [], []>, kernel.root}

// E2E-LABEL: @fallback_elementwise_add_scalar
//       E2E: gpu.module.kernels.ptx_data

// -----

#map = affine_map<(d0) -> (d0)>
!complex_tensor_type = tensor<1024xcomplex<f32>>
builtin.module @fallback_unary_complex {
  func.func @kernel(%arg0: !complex_tensor_type,
                      %arg1: !complex_tensor_type) -> !complex_tensor_type {
    %1 = linalg.map {complex.neg}
      ins(%arg0 : !complex_tensor_type)
      outs(%arg1 : !complex_tensor_type)
    return %1 : !complex_tensor_type
  }
}

// CHECK-LABEL:   module @fallback_unary_complex
// CHECK-LABEL: func.func @kernel
//       CHECK:   linalg.map
// CHECK-SAME:       kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [1024], [1024], [8], [1], [128]>, kernel.root}

// E2E-LABEL: fallback_unary_complex
//       E2E: gpu.module.kernels.ptx_data

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>

builtin.module @many_broadcast {
func.func @kernel(%arg0: tensor<1x64x112x112xf32>, %arg1: tensor<64x1x1xf32>,
    %arg2: tensor<64x1x1xf32>, %arg3: tensor<64x1x1xf32>, %arg4: tensor<64x1x1xf32>)
      -> tensor<1x64x112x112xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x64x112x112xf32>
  %1 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.subf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x64x112x112xf32>
  %2 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %arg2 : tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.mulf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x64x112x112xf32>
  %3 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %arg3 : tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.mulf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x64x112x112xf32>
  %4 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %arg4 : tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.addf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x64x112x112xf32>
  %5 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<1x64x112x112xf32>) outs(%0 : tensor<1x64x112x112xf32>) attrs =  {kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_60">, [1, 16, 16, 28], [1, 16, 16, 28], [1, 2, 2, 4], [1, 4, 7, 4], [1, 8, 8, 7]>, kernel.root} {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.cmpf ugt, %in, %cst : f32
    %7 = arith.select %6, %in, %cst : f32
    linalg.yield %7 : f32
  } -> tensor<1x64x112x112xf32>
  return %5 : tensor<1x64x112x112xf32>
}
}

// CHECK-LABEL: module @many_broadcast
// E2E-LABEL: @many_broadcast
//       E2E: gpu.module.kernels.ptx_data

// -----

#map = affine_map<(d0, d1) -> (d0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

module @broadcast_2d {
  func.func @kernel(%arg0: tensor<128x1xi32>, %arg1: tensor<128x128xi32>) -> tensor<128x128xi32> {
    %0 = tensor.empty() : tensor<16xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<128x1xi32>)
      outs(%arg1 : tensor<128x128xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<128x128xi32>
    return %1 : tensor<128x128xi32>
  }
}

// CHECK-LABEL: module @broadcast_2d

// E2E-LABEL: @broadcast_2d
//       E2E: gpu.module.kernels.ptx_data

// -----

#map = affine_map<(d0)->(d0)>
module @small_1d {
  func.func @kernel(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xi8>, %arg3: tensor<8xf32>) -> tensor<8xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i8 = arith.constant 0 : i8
    %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg2, %arg0, %arg1 : tensor<8xi8>, tensor<8xf32>, tensor<8xf32>) outs(%arg3 : tensor<8xf32>) {
    ^bb0(%in: i8, %in_1: f32, %in_2: f32, %out: f32):
      %1 = math.sin %in_2 : f32
      %2 = arith.divf %1, %in_2 : f32
      %3 = arith.mulf %in_1, %cst_0 : f32
      %4 = arith.addf %3, %cst : f32
      %5 = arith.cmpi ne, %in, %c0_i8 : i8
      %6 = arith.select %5, %4, %2 : f32
      linalg.yield %6 : f32
    } -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// CHECK-LABEL: module @small_1d

// E2E-LABEL: @small_1d
//       E2E: gpu.module.kernels.ptx_data


// -----

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

builtin.module @redundant_producer_operand {
func.func @kernel() -> tensor<2x2xi1> {
  %arg0 = tensor.empty() : tensor<2xi32>
  %arg1 = tensor.empty() : tensor<2x2xi1>
  %0 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel"]}
    outs(%arg0 : tensor<2xi32>) {
  ^bb0(%out: i32):
    %2 = linalg.index 0 : index
    %3 = arith.index_cast %2 : index to i32
    linalg.yield %3 : i32
  } -> tensor<2xi32>
  %1 = linalg.generic {
       indexing_maps = [#map1, #map2, #map3],
      iterator_types = ["parallel", "parallel"]}
    ins(%0, %0 : tensor<2xi32>, tensor<2xi32>)
    outs(%arg1 : tensor<2x2xi1>) attrs =  {
      kernel.root
    } {
  ^bb0(%in: i32, %in_0: i32, %out: i1):
    %2 = arith.cmpi eq, %in, %in_0 : i32
    linalg.yield %2 : i1
  } -> tensor<2x2xi1>
  return %1 : tensor<2x2xi1>
}
}

// CHECK-LABEL: module @redundant_producer_operand

// E2E-LABEL: @redundant_producer_operand
//       E2E: gpu.module.kernels.ptx_data


// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>

builtin.module @many_producers {
func.func @kernel(%arg0: tensor<1x64x112x112xf32>, %arg1: tensor<64x1x1xf32>,
    %arg2: tensor<64x1x1xf32>, %arg3: tensor<64x1x1xf32>, %arg4: tensor<64x1x1xf32>)
      -> tensor<1x64x112x112xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x64x112x112xf32>
  %1 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.subf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x64x112x112xf32>
  %2 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %arg2 : tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.mulf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x64x112x112xf32>
  %3 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %arg3 : tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.mulf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x64x112x112xf32>
  %4 = linalg.generic {indexing_maps = [#map5, #map6, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %arg4 : tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.addf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x64x112x112xf32>
  %5 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<1x64x112x112xf32>)
    outs(%0 : tensor<1x64x112x112xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.cmpf ugt, %in, %cst : f32
    %7 = arith.select %6, %in, %cst : f32
    linalg.yield %7 : f32
  } -> tensor<1x64x112x112xf32>
  return %5 : tensor<1x64x112x112xf32>
}
}

// CHECK-LABEL: module @many_producers
// CHECK-LABEL: func.func @kernel

// E2E-LABEL: module @many_producers
//       E2E: gpu.module.kernels.ptx_data


// -----

// -----

#map = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

builtin.module @elementwise_large_f32 {
  func.func @kernel(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<1x3x1280x800xf32> {
    %0 = tensor.empty() : tensor<1x3x1280x800xf32>
    %c1065353216_i32 = arith.constant 1065353216 : i32
    %c9_i32 = arith.constant 9 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c24_i32 = arith.constant 24 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c29_i32 = arith.constant 29 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c466688986_i32 = arith.constant 466688986 : i32
    %c26_i32 = arith.constant 26 : i32
    %c6_i32 = arith.constant 6 : i32
    %c15_i32 = arith.constant 15 : i32
    %c17_i32 = arith.constant 17 : i32
    %c13_i32 = arith.constant 13 : i32
    %c19_i32 = arith.constant 19 : i32
    %c32_i64 = arith.constant 32 : i64
    %c1024000_i64 = arith.constant 1024000 : i64
    %c800_i64 = arith.constant 800 : i64
    %1 = linalg.generic {
        indexing_maps = [#map, #map, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
       ins(%arg0, %arg1 : tensor<i32>, tensor<i32>)
       outs(%0 : tensor<1x3x1280x800xf32>) {
    ^bb0(%in: i32, %in_0: i32, %out: f32):
      %2 = linalg.index 1 : index
      %3 = linalg.index 2 : index
      %4 = linalg.index 3 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = arith.index_cast %3 : index to i64
      %7 = arith.muli %6, %c800_i64 : i64
      %8 = arith.index_cast %2 : index to i64
      %9 = arith.muli %8, %c1024000_i64 : i64
      %10 = arith.addi %9, %7 : i64
      %11 = arith.addi %10, %5 : i64
      %12 = arith.trunci %11 : i64 to i32
      %13 = arith.addi %12, %in_0 : i32
      %14 = arith.shrui %11, %c32_i64 : i64
      %15 = arith.trunci %14 : i64 to i32
      %16 = arith.addi %15, %in : i32
      %17 = arith.addi %16, %13 : i32
      %18 = arith.shrui %13, %c19_i32 : i32
      %19 = arith.shli %13, %c13_i32 : i32
      %20 = arith.ori %19, %18 : i32
      %21 = arith.xori %17, %20 : i32
      %22 = arith.addi %17, %21 : i32
      %23 = arith.shrui %21, %c17_i32 : i32
      %24 = arith.shli %21, %c15_i32 : i32
      %25 = arith.ori %24, %23 : i32
      %26 = arith.xori %22, %25 : i32
      %27 = arith.addi %22, %26 : i32
      %28 = arith.shrui %26, %c6_i32 : i32
      %29 = arith.shli %26, %c26_i32 : i32
      %30 = arith.ori %29, %28 : i32
      %31 = arith.xori %27, %30 : i32
      %32 = arith.xori %in, %in_0 : i32
      %33 = arith.xori %32, %c466688986_i32 : i32
      %34 = arith.addi %27, %31 : i32
      %35 = arith.shrui %31, %c26_i32 : i32
      %36 = arith.shli %31, %c6_i32 : i32
      %37 = arith.ori %36, %35 : i32
      %38 = arith.xori %34, %37 : i32
      %39 = arith.addi %38, %33 : i32
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.addi %34, %in_0 : i32
      %42 = arith.addi %41, %40 : i32
      %43 = arith.shrui %40, %c15_i32 : i32
      %44 = arith.shli %40, %c17_i32 : i32
      %45 = arith.ori %44, %43 : i32
      %46 = arith.xori %42, %45 : i32
      %47 = arith.addi %42, %46 : i32
      %48 = arith.shrui %46, %c3_i32 : i32
      %49 = arith.shli %46, %c29_i32 : i32
      %50 = arith.ori %49, %48 : i32
      %51 = arith.xori %47, %50 : i32
      %52 = arith.addi %47, %51 : i32
      %53 = arith.shrui %51, %c16_i32 : i32
      %54 = arith.shli %51, %c16_i32 : i32
      %55 = arith.ori %54, %53 : i32
      %56 = arith.xori %52, %55 : i32
      %57 = arith.addi %52, %56 : i32
      %58 = arith.shrui %56, %c8_i32 : i32
      %59 = arith.shli %56, %c24_i32 : i32
      %60 = arith.ori %59, %58 : i32
      %61 = arith.xori %57, %60 : i32
      %62 = arith.addi %61, %in : i32
      %63 = arith.addi %62, %c2_i32 : i32
      %64 = arith.addi %57, %33 : i32
      %65 = arith.addi %64, %63 : i32
      %66 = arith.shrui %63, %c19_i32 : i32
      %67 = arith.shli %63, %c13_i32 : i32
      %68 = arith.ori %67, %66 : i32
      %69 = arith.xori %65, %68 : i32
      %70 = arith.addi %65, %69 : i32
      %71 = arith.shrui %69, %c17_i32 : i32
      %72 = arith.shli %69, %c15_i32 : i32
      %73 = arith.ori %72, %71 : i32
      %74 = arith.xori %70, %73 : i32
      %75 = arith.addi %70, %74 : i32
      %76 = arith.shrui %74, %c6_i32 : i32
      %77 = arith.shli %74, %c26_i32 : i32
      %78 = arith.ori %77, %76 : i32
      %79 = arith.xori %75, %78 : i32
      %80 = arith.addi %75, %79 : i32
      %81 = arith.shrui %79, %c26_i32 : i32
      %82 = arith.shli %79, %c6_i32 : i32
      %83 = arith.ori %82, %81 : i32
      %84 = arith.xori %80, %83 : i32
      %85 = arith.addi %84, %in_0 : i32
      %86 = arith.addi %85, %c3_i32 : i32
      %87 = arith.addi %80, %in : i32
      %88 = arith.addi %87, %86 : i32
      %89 = arith.shrui %86, %c15_i32 : i32
      %90 = arith.shli %86, %c17_i32 : i32
      %91 = arith.ori %90, %89 : i32
      %92 = arith.xori %88, %91 : i32
      %93 = arith.addi %88, %92 : i32
      %94 = arith.shrui %92, %c3_i32 : i32
      %95 = arith.shli %92, %c29_i32 : i32
      %96 = arith.ori %95, %94 : i32
      %97 = arith.xori %93, %96 : i32
      %98 = arith.addi %93, %97 : i32
      %99 = arith.shrui %97, %c16_i32 : i32
      %100 = arith.shli %97, %c16_i32 : i32
      %101 = arith.ori %100, %99 : i32
      %102 = arith.xori %98, %101 : i32
      %103 = arith.addi %98, %102 : i32
      %104 = arith.shrui %102, %c8_i32 : i32
      %105 = arith.shli %102, %c24_i32 : i32
      %106 = arith.ori %105, %104 : i32
      %107 = arith.xori %103, %106 : i32
      %108 = arith.addi %107, %33 : i32
      %109 = arith.addi %108, %c4_i32 : i32
      %110 = arith.addi %103, %in_0 : i32
      %111 = arith.addi %110, %109 : i32
      %112 = arith.shrui %109, %c19_i32 : i32
      %113 = arith.shli %109, %c13_i32 : i32
      %114 = arith.ori %113, %112 : i32
      %115 = arith.xori %111, %114 : i32
      %116 = arith.addi %111, %115 : i32
      %117 = arith.shrui %115, %c17_i32 : i32
      %118 = arith.shli %115, %c15_i32 : i32
      %119 = arith.ori %118, %117 : i32
      %120 = arith.xori %116, %119 : i32
      %121 = arith.addi %116, %120 : i32
      %122 = arith.shrui %120, %c6_i32 : i32
      %123 = arith.shli %120, %c26_i32 : i32
      %124 = arith.ori %123, %122 : i32
      %125 = arith.xori %121, %124 : i32
      %126 = arith.addi %121, %125 : i32
      %127 = arith.shrui %125, %c26_i32 : i32
      %128 = arith.shli %125, %c6_i32 : i32
      %129 = arith.ori %128, %127 : i32
      %130 = arith.xori %126, %129 : i32
      %131 = arith.addi %130, %in : i32
      %132 = arith.addi %131, %c5_i32 : i32
      %133 = arith.addi %126, %33 : i32
      %134 = arith.xori %133, %132 : i32
      %135 = arith.shrui %134, %c9_i32 : i32
      %136 = arith.ori %135, %c1065353216_i32 : i32
      %137 = arith.bitcast %136 : i32 to f32
      linalg.yield %137 : f32
    } -> tensor<1x3x1280x800xf32>
    return %1 : tensor<1x3x1280x800xf32>
  }
}

// CHECK-LABEL: module @elementwise_large_f32
// CHECK-LABEL: func.func @kernel

// E2E-LABEL: elementwise_large_f32
//       E2E: gpu.module.kernels.ptx_data


// -----

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<()[s0] -> (-s0 + 2047)>
module @reverse_tensor {
  func.func @kernel(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %0 = tensor.empty() : tensor<2048xf32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<2048xf32>) {
    ^bb0(%out: f32):
      %2 = linalg.index 0 : index
      %3 = affine.apply #map1()[%2]
      %extracted = tensor.extract %arg0[%3] : tensor<2048xf32>
      linalg.yield %extracted : f32
    } -> tensor<2048xf32>
    return %1 : tensor<2048xf32>
  }
}

// CHECK-LABEL: module @reverse_tensor
// CHECK-LABEL: func.func @kernel

// E2E-LABEL: @reverse_tensor
//       E2E: gpu.module.kernels.ptx_data
