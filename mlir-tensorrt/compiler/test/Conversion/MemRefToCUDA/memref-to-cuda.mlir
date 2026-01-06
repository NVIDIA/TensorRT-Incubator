// RUN: mlir-tensorrt-opt -split-input-file -convert-memref-to-cuda %s | FileCheck %s


func.func @device_alloc() -> memref<4x8xf32, #plan.memory_space<device>> {
  %0 = memref.alloc() : memref<4x8xf32, #plan.memory_space<device>>
  return %0 : memref<4x8xf32, #plan.memory_space<device>>
}

// CHECK-LABEL: @device_alloc
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   %[[ALLOC:.+]] = cuda.alloc() stream(%[[STREAM]]) : memref<4x8xf32, #plan.memory_space<device>>
//       CHECK:   return %[[ALLOC]]


func.func @device_alloc_dynamic(%arg0: index, %arg1: index) -> memref<?x?xf32, #plan.memory_space<device>> {
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32, #plan.memory_space<device>>
  return %0 : memref<?x?xf32, #plan.memory_space<device>>
}

// CHECK-LABEL: @device_alloc_dynamic
//  CHECK-SAME:   (%[[ARG0:.+]]: index, %[[ARG1:.+]]: index)
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   %[[ALLOC:.+]] = cuda.alloc(%[[ARG0]], %[[ARG1]]) stream(%[[STREAM]]) : memref<?x?xf32, #plan.memory_space<device>>
//       CHECK:   return %[[ALLOC]]

// -----

func.func @device_alloc_aligned() -> memref<4x8xf32, #plan.memory_space<device>> {
  %0 = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32, #plan.memory_space<device>>
  return %0 : memref<4x8xf32, #plan.memory_space<device>>
}

// CHECK-LABEL: @device_alloc_aligned
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   %[[ALLOC:.+]] = cuda.alloc() stream(%[[STREAM]]) align 64 : memref<4x8xf32, #plan.memory_space<device>>
//       CHECK:   return %[[ALLOC]]

// -----

func.func @unified_alloc() -> memref<4x8xf32, #plan.memory_space<unified>> {
  %0 = memref.alloc() : memref<4x8xf32, #plan.memory_space<unified>>
  return %0 : memref<4x8xf32, #plan.memory_space<unified>>
}

// CHECK-LABEL: @unified_alloc
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   %[[ALLOC:.+]] = cuda.alloc() stream(%[[STREAM]]) : memref<4x8xf32, #plan.memory_space<unified>>
//       CHECK:   return %[[ALLOC]]

// -----

func.func @host_pinned_alloc() -> memref<4x8xf32, #plan.memory_space<host_pinned>> {
  %0 = memref.alloc() : memref<4x8xf32, #plan.memory_space<host_pinned>>
  return %0 : memref<4x8xf32, #plan.memory_space<host_pinned>>
}

// CHECK-LABEL: @host_pinned_alloc
//   CHECK-NOT:   cuda.get_program_device
//   CHECK-NOT:   cuda.get_global_stream
//       CHECK:   %[[ALLOC:.+]] = cuda.alloc() : memref<4x8xf32, #plan.memory_space<host_pinned>>
//       CHECK:   return %[[ALLOC]]

// -----

func.func @host_alloc_not_converted() -> memref<4x8xf32, #plan.memory_space<host>> {
  %0 = memref.alloc() : memref<4x8xf32, #plan.memory_space<host>>
  return %0 : memref<4x8xf32, #plan.memory_space<host>>
}

// CHECK-LABEL: @host_alloc_not_converted
//   CHECK-NOT:   cuda.alloc
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<4x8xf32, #plan.memory_space<host>>
//       CHECK:   return %[[ALLOC]]

// -----

func.func @copy_h2d(%src: memref<4x8xf32, #plan.memory_space<host>>,
                    %dst: memref<4x8xf32, #plan.memory_space<device>>) {
  memref.copy %src, %dst : memref<4x8xf32, #plan.memory_space<host>> to memref<4x8xf32, #plan.memory_space<device>>
  return
}

// CHECK-LABEL: @copy_h2d
//  CHECK-SAME:   (%[[SRC:.+]]: memref<4x8xf32, #plan.memory_space<host>>, %[[DST:.+]]: memref<4x8xf32, #plan.memory_space<device>>)
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   cuda.copy_h2d stream(%[[STREAM]]) %[[SRC]], %[[DST]] : memref<4x8xf32, #plan.memory_space<host>> to memref<4x8xf32, #plan.memory_space<device>>
//       CHECK:   return

// -----

func.func @copy_d2h(%src: memref<4x8xf32, #plan.memory_space<device>>,
                    %dst: memref<4x8xf32, #plan.memory_space<host>>) {
  memref.copy %src, %dst : memref<4x8xf32, #plan.memory_space<device>> to memref<4x8xf32, #plan.memory_space<host>>
  return
}

// CHECK-LABEL: @copy_d2h
//  CHECK-SAME:   (%[[SRC:.+]]: memref<4x8xf32, #plan.memory_space<device>>, %[[DST:.+]]: memref<4x8xf32, #plan.memory_space<host>>)
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   cuda.copy_d2h stream(%[[STREAM]]) %[[SRC]], %[[DST]] : memref<4x8xf32, #plan.memory_space<device>> to memref<4x8xf32, #plan.memory_space<host>>
//       CHECK:   cuda.stream.sync %[[STREAM]] : !cuda.stream
//       CHECK:   return

// -----

func.func @copy_d2d(%src: memref<4x8xf32, #plan.memory_space<device>>,
                    %dst: memref<4x8xf32, #plan.memory_space<device>>) {
  memref.copy %src, %dst : memref<4x8xf32, #plan.memory_space<device>> to memref<4x8xf32, #plan.memory_space<device>>
  return
}

// CHECK-LABEL: @copy_d2d
//  CHECK-SAME:   (%[[SRC:.+]]: memref<4x8xf32, #plan.memory_space<device>>, %[[DST:.+]]: memref<4x8xf32, #plan.memory_space<device>>)
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   cuda.copy_d2d stream(%[[STREAM]]) %[[SRC]], %[[DST]] : memref<4x8xf32, #plan.memory_space<device>> to memref<4x8xf32, #plan.memory_space<device>>
//   CHECK-NOT:   cuda.stream.sync
//       CHECK:   return

// -----

func.func @copy_host_pinned_to_device(%src: memref<4x8xf32, #plan.memory_space<host_pinned>>,
                                      %dst: memref<4x8xf32, #plan.memory_space<device>>) {
  memref.copy %src, %dst : memref<4x8xf32, #plan.memory_space<host_pinned>> to memref<4x8xf32, #plan.memory_space<device>>
  return
}

// CHECK-LABEL: @copy_host_pinned_to_device
//  CHECK-SAME:   (%[[SRC:.+]]: memref<4x8xf32, #plan.memory_space<host_pinned>>, %[[DST:.+]]: memref<4x8xf32, #plan.memory_space<device>>)
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   cuda.copy_h2d stream(%[[STREAM]]) %[[SRC]], %[[DST]] : memref<4x8xf32, #plan.memory_space<host_pinned>> to memref<4x8xf32, #plan.memory_space<device>>
//       CHECK:   return

// -----

func.func @copy_device_to_host_pinned(%src: memref<4x8xf32, #plan.memory_space<device>>,
                                      %dst: memref<4x8xf32, #plan.memory_space<host_pinned>>) {
  memref.copy %src, %dst : memref<4x8xf32, #plan.memory_space<device>> to memref<4x8xf32, #plan.memory_space<host_pinned>>
  return
}

// CHECK-LABEL: @copy_device_to_host_pinned
//  CHECK-SAME:   (%[[SRC:.+]]: memref<4x8xf32, #plan.memory_space<device>>, %[[DST:.+]]: memref<4x8xf32, #plan.memory_space<host_pinned>>)
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   cuda.copy_d2h stream(%[[STREAM]]) %[[SRC]], %[[DST]] : memref<4x8xf32, #plan.memory_space<device>> to memref<4x8xf32, #plan.memory_space<host_pinned>>
//       CHECK:   cuda.stream.sync %[[STREAM]] : !cuda.stream
//       CHECK:   return

// -----

func.func @copy_unified(%src: memref<4x8xf32, #plan.memory_space<unified>>,
                        %dst: memref<4x8xf32, #plan.memory_space<device>>) {
  memref.copy %src, %dst : memref<4x8xf32, #plan.memory_space<unified>> to memref<4x8xf32, #plan.memory_space<device>>
  return
}

// CHECK-LABEL: @copy_unified
//  CHECK-SAME:   (%[[SRC:.+]]: memref<4x8xf32, #plan.memory_space<unified>>, %[[DST:.+]]: memref<4x8xf32, #plan.memory_space<device>>)
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   cuda.copy_d2d stream(%[[STREAM]]) %[[SRC]], %[[DST]] : memref<4x8xf32, #plan.memory_space<unified>> to memref<4x8xf32, #plan.memory_space<device>>
//       CHECK:   return

// -----

func.func @copy_host_to_host_not_converted(%src: memref<4x8xf32, #plan.memory_space<host>>,
                                           %dst: memref<4x8xf32, #plan.memory_space<host>>) {
  memref.copy %src, %dst : memref<4x8xf32, #plan.memory_space<host>> to memref<4x8xf32, #plan.memory_space<host>>
  return
}

// CHECK-LABEL: @copy_host_to_host_not_converted
//   CHECK-NOT:   cuda.copy
//       CHECK:   memref.copy %{{.+}}, %{{.+}} : memref<4x8xf32, #plan.memory_space<host>> to memref<4x8xf32, #plan.memory_space<host>>
//       CHECK:   return

// -----

func.func @device_dealloc(%arg0: memref<4x8xf32, #plan.memory_space<device>>) {
  memref.dealloc %arg0 : memref<4x8xf32, #plan.memory_space<device>>
  return
}

// CHECK-LABEL: @device_dealloc
//  CHECK-SAME:   (%[[ARG:.+]]: memref<4x8xf32, #plan.memory_space<device>>)
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   cuda.dealloc stream(%[[STREAM]]) %[[ARG]] : memref<4x8xf32, #plan.memory_space<device>>
//       CHECK:   return

// -----

func.func @unified_dealloc(%arg0: memref<4x8xf32, #plan.memory_space<unified>>) {
  memref.dealloc %arg0 : memref<4x8xf32, #plan.memory_space<unified>>
  return
}

// CHECK-LABEL: @unified_dealloc
//  CHECK-SAME:   (%[[ARG:.+]]: memref<4x8xf32, #plan.memory_space<unified>>)
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   cuda.dealloc stream(%[[STREAM]]) %[[ARG]] : memref<4x8xf32, #plan.memory_space<unified>>
//       CHECK:   return

// -----

func.func @host_pinned_dealloc(%arg0: memref<4x8xf32, #plan.memory_space<host_pinned>>) {
  memref.dealloc %arg0 : memref<4x8xf32, #plan.memory_space<host_pinned>>
  return
}

// CHECK-LABEL: @host_pinned_dealloc
//  CHECK-SAME:   (%[[ARG:.+]]: memref<4x8xf32, #plan.memory_space<host_pinned>>)
//       CHECK:   %[[DEVICE:.+]] = cuda.get_program_device
//       CHECK:   %[[STREAM:.+]] = cuda.get_global_stream device(%[[DEVICE]]) [0]
//       CHECK:   cuda.dealloc stream(%[[STREAM]]) %[[ARG]] : memref<4x8xf32, #plan.memory_space<host_pinned>>
//       CHECK:   return

// -----

func.func @host_dealloc_not_converted(%arg0: memref<4x8xf32, #plan.memory_space<host>>) {
  memref.dealloc %arg0 : memref<4x8xf32, #plan.memory_space<host>>
  return
}

// CHECK-LABEL: @host_dealloc_not_converted
//   CHECK-NOT:   cuda.dealloc
//       CHECK:   memref.dealloc %{{.+}} : memref<4x8xf32, #plan.memory_space<host>>
//       CHECK:   return

// -----

func.func @full_lifecycle(%host_data: memref<4x8xf32, #plan.memory_space<host>>) {
  %device_mem = memref.alloc() : memref<4x8xf32, #plan.memory_space<device>>
  memref.copy %host_data, %device_mem : memref<4x8xf32, #plan.memory_space<host>> to memref<4x8xf32, #plan.memory_space<device>>
  memref.copy %device_mem, %host_data : memref<4x8xf32, #plan.memory_space<device>> to memref<4x8xf32, #plan.memory_space<host>>
  memref.dealloc %device_mem : memref<4x8xf32, #plan.memory_space<device>>
  return
}

// CHECK-LABEL: @full_lifecycle
//  CHECK-SAME:   (%[[HOST_DATA:.+]]: memref<4x8xf32, #plan.memory_space<host>>)
//       CHECK:   %[[DEV1:.+]] = cuda.get_program_device
//       CHECK:   %[[STR1:.+]] = cuda.get_global_stream device(%[[DEV1]]) [0]
//       CHECK:   %[[DEVICE_MEM:.+]] = cuda.alloc() stream(%[[STR1]]) : memref<4x8xf32, #plan.memory_space<device>>
//       CHECK:   cuda.copy_h2d stream(%[[STR1]]) %[[HOST_DATA]], %[[DEVICE_MEM]]
//       CHECK:   cuda.copy_d2h stream(%[[STR1]]) %[[DEVICE_MEM]], %[[HOST_DATA]]
//       CHECK:   cuda.dealloc stream(%[[STR1]]) %[[DEVICE_MEM]]
//       CHECK:   return
