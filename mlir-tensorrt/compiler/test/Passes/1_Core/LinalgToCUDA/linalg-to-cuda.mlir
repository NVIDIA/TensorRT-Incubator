// RUN: mlir-tensorrt-opt -split-input-file -convert-linalg-to-cuda %s | FileCheck %s

func.func @fill_device_f32(%arg0: memref<?xf32, #plan.memory_space<device>>,
                            %val: f32) {
  linalg.fill ins(%val : f32)
      outs(%arg0 : memref<?xf32, #plan.memory_space<device>>)
  return
}

// CHECK-LABEL: @fill_device_f32
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   cuda.memset stream(%[[STREAM]]) %[[ARG0:.+]] with %[[VAL:.+]] : memref<?xf32, #plan.memory_space<device>>, f32
//       CHECK:   return

// -----

func.func @fill_host_f32(%arg0: memref<?xf32, #plan.memory_space<host>>,
                          %val: f32) {
  linalg.fill ins(%val : f32)
      outs(%arg0 : memref<?xf32, #plan.memory_space<host>>)
  return
}

// CHECK-LABEL: @fill_host_f32
//  CHECK-NOT:    cuda.memset
//       CHECK:   linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : memref<?xf32, #plan.memory_space<host>>)
//       CHECK:   return

// -----

func.func @generic_fill_device(
    %arg0: memref<4xf16, #plan.memory_space<device>>) {
  %cst = arith.constant 3.0 : f16
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } outs(%arg0 : memref<4xf16, #plan.memory_space<device>>) {
  ^bb0(%out: f16):
    linalg.yield %cst : f16
  }
  return
}

// CHECK-LABEL: @generic_fill_device
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   cuda.memset stream(%[[STREAM]]) %{{.+}} with %[[CST:.+]] : memref<4xf16, #plan.memory_space<device>>, f16
//       CHECK:   return

// -----

func.func @fill_unified_f32(%arg0: memref<4xf32, #plan.memory_space<unified>>,
                             %val: f32) {
  linalg.fill ins(%val : f32)
      outs(%arg0 : memref<4xf32, #plan.memory_space<unified>>)
  return
}

// CHECK-LABEL: @fill_unified_f32
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   cuda.memset stream(%[[STREAM]]) %[[ARG0:.+]] with %[[VAL:.+]] : memref<4xf32, #plan.memory_space<unified>>, f32
//       CHECK:   return

// -----

func.func @fill_device_noncontig(
    %arg0: memref<4x4xf32, strided<[8, 1]>, #plan.memory_space<device>>,
                                 %val: f32) {
  linalg.fill ins(%val : f32) outs(
      %arg0 : memref<4x4xf32, strided<[8, 1]>, #plan.memory_space<device>>)
  return
}

// CHECK-LABEL: @fill_device_noncontig
//  CHECK-NOT:    cuda.memset
//       CHECK:   linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : memref<4x4xf32, strided<[8, 1]>, #plan.memory_space<device>>)
//       CHECK:   return

// -----

func.func @fill_device_i64(%arg0: memref<4xi64, #plan.memory_space<device>>,
                            %val: i64) {
  linalg.fill ins(%val : i64)
      outs(%arg0 : memref<4xi64, #plan.memory_space<device>>)
  return
}

// CHECK-LABEL: @fill_device_i64
//  CHECK-NOT:    cuda.memset
//       CHECK:   linalg.fill ins(%{{.+}} : i64) outs(%{{.+}} : memref<4xi64, #plan.memory_space<device>>)
//       CHECK:   return
