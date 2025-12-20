// RUN: kernel-opt -split-input-file %s -kernel-initial-transform-schedule="generator-benefit={fallback:100} device-compute-capability=80 device-max-smem-per-block=48 device-max-registers-per-block=65536" -verify-diagnostics | FileCheck %s
// RUN: kernel-opt -split-input-file %s -kernel-linalg-codegen-pipeline="generator-benefit={fallback:100} device-compute-capability=80 device-max-smem-per-block=48 device-max-registers-per-block=65536" | FileCheck %s --check-prefix=E2E

!lhs_type = tensor<1024x1024xf32>
!rhs_type = tensor<1024x1024xf32>
!result_type = tensor<1024x1024xf32>

builtin.module @matmul_1024x1024x1024 {
  func.func @kernel(%arg0: !lhs_type,
                    %arg1: !rhs_type,
                    %arg2: !result_type) -> !result_type {
    %cst = arith.constant 0.0 : f32
    %0 = tensor.empty() : !result_type
    %out = linalg.fill ins(%cst: f32) outs(%0: !result_type) -> !result_type
    %1 = linalg.matmul ins(%arg0, %arg1: !lhs_type, !rhs_type) outs(%out: !result_type)
      -> !result_type
    return %1 : !result_type
  }
}

// CHECK-LABEL: module @matmul_1024x1024x1024
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.matmul
//  CHECK-SAME:  kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [64, 128, 1024], [64, 128, 32], [8, 8, 32], [16, 8, 1], [8, 16, 1]>

// E2E-LABEL: @matmul_1024x1024x1024
//       E2E: gpu.module.kernels.ptx_data

// -----

!lhs_type = tensor<9x9xf32>
!rhs_type = tensor<9x9xf32>
!result_type = tensor<9x9xf32>

builtin.module @matmul_9x9x9 {
  func.func @kernel(%arg0: !lhs_type,
                    %arg1: !rhs_type,
                    %arg2: !result_type) -> !result_type {
    %cst = arith.constant 0.0 : f32
    %out = linalg.fill ins(%cst: f32) outs(%arg2: !result_type) -> !result_type
    %1 = linalg.matmul ins(%arg0, %arg1: !lhs_type, !rhs_type) outs(%out: !result_type)
      -> !result_type
    return %1 : !result_type
  }
}

// CHECK-LABEL: module @matmul_9x9x9
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.matmul
//  CHECK-SAME:  kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [9, 9, 9], [9, 9, 9], [1, 1, 9], [1, 1, 1], [9, 9, 1]>

// E2E-LABEL: @matmul_9x9x9
//       E2E: gpu.module.kernels.ptx_data

// -----

!lhs_type = tensor<15x72xf32>
!rhs_type = tensor<72x17xf32>
!result_type = tensor<15x17xf32>

builtin.module @matmul_15x17x72 {
  func.func @kernel(%arg0: !lhs_type,
                    %arg1: !rhs_type,
                    %arg2: !result_type) -> !result_type {
    %cst = arith.constant 0.0 : f32
    %out = linalg.fill ins(%cst: f32) outs(%arg2: !result_type) -> !result_type
    %1 = linalg.matmul ins(%arg0, %arg1: !lhs_type, !rhs_type) outs(%out: !result_type)
      -> !result_type
    return %1 : !result_type
  }
}

// CHECK-LABEL: module @matmul_15x17x72
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.matmul
//  CHECK-SAME:  kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [15, 17, 72], [15, 17, 36], [1, 1, 36], [1, 1, 1], [15, 17, 1]>

// E2E-LABEL: @matmul_15x17x72
//       E2E: gpu.module.kernels.ptx_data

// -----

!lhs_type = tensor<128x1024xf32>
!rhs_type = tensor<256x1024xf32>
!result_type = tensor<128x256xf32>

builtin.module @matmul_128x256x1024 {
  func.func @kernel(%arg0: !lhs_type,
                    %arg1: !rhs_type,
                    %arg2: !result_type) -> !result_type {
    %cst = arith.constant 0.0 : f32
    %out = linalg.fill ins(%cst: f32) outs(%arg2: !result_type) -> !result_type
    %1 = linalg.matmul
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ]
      ins(%arg0, %arg1: !lhs_type, !rhs_type) outs(%out: !result_type)
      -> !result_type
    return %1 : !result_type
  }
}

// CHECK-LABEL: module @matmul_128x256x1024
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.matmul
//  CHECK-SAME:  kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [64, 64, 1024], [64, 64, 64], [4, 4, 64], [2, 4, 1], [16, 16, 1]>

// E2E-LABEL: @matmul_128x256x1024
//       E2E: gpu.module.kernels.ptx_data

// -----

!lhs_type = tensor<1024x128x32xf32>
!rhs_type = tensor<1024x32x128xf32>
!result_type = tensor<1024x128x128xf32>

builtin.module @batch_matmul_1024x128x128x32 {
  func.func @kernel(%arg0: !lhs_type,
                    %arg1: !rhs_type,
                    %arg2: !result_type) -> !result_type {
    %cst = arith.constant 0.0 : f32
    %out = linalg.fill ins(%cst: f32) outs(%arg2: !result_type) -> !result_type
    %1 = linalg.batch_matmul ins(%arg0, %arg1: !lhs_type, !rhs_type) outs(%out: !result_type)
      -> !result_type
    return %1 : !result_type
  }
}

// CHECK-LABEL: module @batch_matmul_1024x128x128x32
// CHECK-LABEL: func.func @kernel
//       CHECK:  linalg.batch_matmul
//  CHECK-SAME:  kernel.parameters = #kernel.fallback_parameters<#nvvm.target<chip = "sm_80">, [16, 16, 32, 32], [16, 16, 32, 8], [2, 2, 2, 8], [64, 8, 4, 1], [8, 8, 16, 1]>

// E2E-LABEL: @batch_matmul_1024x128x128x32
//       E2E: gpu.module.kernels.ptx_data
