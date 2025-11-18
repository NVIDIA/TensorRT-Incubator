// RUN: executor-translate %s -mlir-to-lua | FileCheck %s
// This test is just a regression test for particular case where the slot assignment algorithm
// would previously crash due to a bug. The specific conditions are difficult to recreate exactly
// by hand with a smaller example. In block 4, we would acquire slot 1 at some point X. Then at a later point Y in the block,
// we would try to re-acquire slot 1 due to an "overlapping but inactive live range" with live
// range Y at the current position. We should ignore this if slot 1 is already in use.
executor.func private @_zext_i1_i8(i1) -> i8
executor.func private @__cuda_memcpy_host_pinned2device(!executor.ptr<host>, !executor.ptr<host_pinned>, i64, !executor.ptr<device>, i64, i64)
executor.func private @__cuda_memcpy_host2device(!executor.ptr<host>, !executor.ptr<host>, i64, !executor.ptr<device>, i64, i64)
executor.func private @__cuda_free_device(!executor.ptr<host>, !executor.ptr<device>)
executor.func private @__cuda_memcpy_device2host(!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<host>, i64, i64)
executor.func private @__cuda_launch(!executor.ptr<host>, i32, i32, i32, i32, i32, i32, i32, !executor.ptr<host>, !executor.ptr<host>)
executor.func private @__cuda_get_function(!executor.ptr<host>, !executor.str_literal) -> !executor.ptr<host>
executor.func private @__cuda_memcpy_device2device(!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<device>, i64, i64)
executor.func private @__cuda_stream_sync(!executor.ptr<host>)
executor.func private @__cuda_memcpy_device2host_pinned(!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<host_pinned>, i64, i64)
executor.func private @__cuda_alloc_device(!executor.ptr<host>, i64, i32) -> !executor.ptr<device>
executor.func private @__cuda_stream_create(i32) -> !executor.ptr<host>
executor.global @stream0 constant : !executor.ptr<host>
executor.func private @__cuda_get_active_device() -> i32
executor.func private @__cuda_alloc_host_pinned(i64, i32) -> !executor.ptr<host_pinned>
executor.func private @__cuda_load_module(i32, !executor.ptr<host>, i64) -> !executor.ptr<host>
executor.global @workspace_1 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
executor.global @workspace_2 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
executor.global @workspace_3 : !executor.table<!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64>
executor.global @workspace_4 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
executor.global @workspace_5 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
executor.global @workspace_6 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
executor.global @workspace_7 : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
executor.global @workspace_8 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64>
executor.global @workspace_9 : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64>
executor.global @workspace_0 : !executor.table<!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64>
executor.data_segment @codegen_cluster_kernel_cuModule_0_ptx_data constant dense_resource<gpu.module.kernels.ptx_data_78> : tensor<349xi8>
executor.global @codegen_cluster_kernel_cuModule_0_cuModule constant : !executor.ptr<host>
executor.global @codegen_cluster_kernel_cuModule_0_cuModule_kernel_cuFunc constant : !executor.ptr<host>

// CHECK-LABEL: function main
func.func public @main(%arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<1xi8, #executor.memory_type<device>>>}, %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<1xi1, #executor.memory_type<device>>>}, %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<1xi8, #executor.memory_type<device>>>, executor.result_slot = 0 : i32}) attributes {executor.func_abi = (memref<1xi8, #executor.memory_type<device>>, memref<1xi1, #executor.memory_type<device>>) -> memref<1xi8, #executor.memory_type<device>>} {
  %c8_i64 = executor.constant 8 : i64
  %c-1_i8 = executor.constant -1 : i8
  %c-128_i8 = executor.constant -128 : i8
  %c1_i8 = executor.constant 1 : i8
  %c0_i8 = executor.constant 0 : i8
  %c16_i32 = executor.constant 16 : i32
  %c0_i32 = executor.constant 0 : i32
  %c1_i32 = executor.constant 1 : i32
  %c1_i64 = executor.constant 1 : i64
  %c0_i64 = executor.constant 0 : i64
  %c-1_i1 = executor.constant true
  %c0_i1 = executor.constant false
  %0 = executor.load %arg2 + %c8_i64 : (!executor.ptr<host>, i64) -> !executor.ptr<device>
  %1 = executor.load %arg0 + %c8_i64 : (!executor.ptr<host>, i64) -> !executor.ptr<device>
  %2 = executor.load %arg1 + %c8_i64 : (!executor.ptr<host>, i64) -> !executor.ptr<device>
  %3 = executor.get_global @workspace_3 : !executor.table<!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64>
  %4 = executor.call @__cuda_get_active_device() : () -> i32
  %5 = executor.get_global @stream0 : !executor.ptr<host>
  %6 = executor.table.get %3[1] : <!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64>
  executor.call @__cuda_memcpy_device2host_pinned(%5, %2, %c0_i64, %6, %c0_i64, %c1_i64) : (!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<host_pinned>, i64, i64) -> ()
  executor.call @__cuda_stream_sync(%5) : (!executor.ptr<host>) -> ()
  %7 = executor.load %6 + %c0_i64 : (!executor.ptr<host_pinned>, i64) -> i1
  %8 = executor.call @_zext_i1_i8(%7) : (i1) -> i8
  %9 = executor.get_global @workspace_7 : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
  %10 = executor.call @__cuda_get_active_device() : () -> i32
  %11 = executor.table.get %9[1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
  executor.call @__cuda_memcpy_device2device(%5, %1, %c0_i64, %11, %c0_i64, %c1_i64) : (!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<device>, i64, i64) -> ()
  %12 = executor.get_global @workspace_0 : !executor.table<!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64>
  %13 = executor.get_global @workspace_9 : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64>
  %14 = executor.get_global @workspace_8 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64>
  %15 = executor.get_global @workspace_5 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  %16 = executor.get_global @workspace_4 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  %17 = executor.get_global @workspace_1 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  %18 = executor.get_global @workspace_2 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  %19 = executor.get_global @workspace_6 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  cf.br ^bb1(%9, %8, %c0_i1 : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>, i8, i1)
^bb1(%20: !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>, %21: i8, %22: i1):  // 2 preds: ^bb0, ^bb4
  %23 = executor.call @__cuda_get_active_device() : () -> i32
  %24 = executor.table.get %20[1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
  %25 = executor.table.get %12[1] : <!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64>
  executor.call @__cuda_memcpy_device2host_pinned(%5, %24, %c0_i64, %25, %c0_i64, %c1_i64) : (!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<host_pinned>, i64, i64) -> ()
  executor.call @__cuda_stream_sync(%5) : (!executor.ptr<host>) -> ()
  %26 = executor.icmp <ne> %21, %c0_i8 : i8
  %27 = executor.get_global @codegen_cluster_kernel_cuModule_0_cuModule_kernel_cuFunc : !executor.ptr<host>
  %28 = executor.table.get %13[1] : <!executor.ptr<device>, !executor.ptr<device>, i64>
  %29 = executor.call @__cuda_get_active_device() : () -> i32
  %30 = executor.alloc %c8_i64 bytes align(%c8_i64) : (i64, i64) -> !executor.ptr<host>
  executor.store %28 to %30 + %c0_i64 : !executor.ptr<device>, !executor.ptr<host>, i64
  %31 = executor.alloc %c8_i64 bytes align(%c8_i64) : (i64, i64) -> !executor.ptr<host>
  executor.store %30 to %31 + %c0_i64 : !executor.ptr<host>, !executor.ptr<host>, i64
  executor.call @__cuda_launch(%27, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c0_i32, %5, %31) : (!executor.ptr<host>, i32, i32, i32, i32, i32, i32, i32, !executor.ptr<host>, !executor.ptr<host>) -> ()
  %32 = executor.call @__cuda_get_active_device() : () -> i32
  %33 = executor.table.get %14[1] : <!executor.ptr<host>, !executor.ptr<host>, i64>
  executor.call @__cuda_memcpy_device2host(%5, %28, %c0_i64, %33, %c0_i64, %c1_i64) : (!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<host>, i64, i64) -> ()
  executor.call @__cuda_stream_sync(%5) : (!executor.ptr<host>) -> ()
  executor.store %26 to %33 + %c0_i64 : i1, !executor.ptr<host>, i64
  %34 = executor.load %33 + %c0_i64 : (!executor.ptr<host>, i64) -> i1
  %35 = executor.load %25 + %c0_i64 : (!executor.ptr<host_pinned>, i64) -> i8
  executor.dealloc %31 : !executor.ptr<host>
  executor.dealloc %30 : !executor.ptr<host>
  cf.cond_br %22, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %36 = executor.call @__cuda_get_active_device() : () -> i32
  executor.call @__cuda_free_device(%5, %24) : (!executor.ptr<host>, !executor.ptr<device>) -> ()
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  cf.cond_br %34, ^bb4, ^bb5
^bb4:  // pred: ^bb3
  %37 = executor.table.get %15[1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  executor.store %26 to %37 + %c0_i64 : i1, !executor.ptr<host>, i64
  %38 = executor.load %37 + %c0_i64 : (!executor.ptr<host>, i64) -> i1
  %39 = executor.select %38, %21, %35 : i8
  %40 = executor.table.get %16[1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  executor.store %39 to %40 + %c0_i64 : i8, !executor.ptr<host>, i64
  %41 = executor.select %26, %21, %35 : i8
  %42 = executor.icmp <eq> %21, %c0_i8 : i8
  %43 = executor.icmp <eq> %35, %c-128_i8 : i8
  %44 = executor.icmp <eq> %21, %c-1_i8 : i8
  %45 = executor.bitwise_andi %43, %44 : i1
  %46 = executor.bitwise_ori %42, %45 : i1
  %47 = executor.select %46, %c1_i8, %21 : i8
  %48 = executor.sremi %35, %47 : i8
  %49 = executor.select %45, %c0_i8, %48 : i8
  %50 = executor.select %42, %35, %49 : i8
  %51 = executor.table.get %17[1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  executor.store %50 to %51 + %c0_i64 : i8, !executor.ptr<host>, i64
  %52 = executor.load %51 + %c0_i64 : (!executor.ptr<host>, i64) -> i8
  %53 = executor.select %38, %52, %c0_i8 : i8
  %54 = executor.table.get %18[1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  executor.store %53 to %54 + %c0_i64 : i8, !executor.ptr<host>, i64
  %55 = executor.select %26, %50, %c0_i8 : i8
  %56 = executor.load %54 + %c0_i64 : (!executor.ptr<host>, i64) -> i8
  %57 = executor.load %40 + %c0_i64 : (!executor.ptr<host>, i64) -> i8
  %58 = executor.smax %56, %57 : i8
  %59 = executor.table.get %19[1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  executor.store %58 to %59 + %c0_i64 : i8, !executor.ptr<host>, i64
  %60 = executor.call @__cuda_get_active_device() : () -> i32
  %61 = executor.call @__cuda_alloc_device(%5, %c1_i64, %c16_i32) : (!executor.ptr<host>, i64, i32) -> !executor.ptr<device>
  %62 = executor.table.create(%61, %61, %c0_i64, %c1_i64, %c1_i64 : !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64) : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
  %63 = executor.call @__cuda_get_active_device() : () -> i32
  executor.call @__cuda_memcpy_host2device(%5, %59, %c0_i64, %61, %c0_i64, %c1_i64) : (!executor.ptr<host>, !executor.ptr<host>, i64, !executor.ptr<device>, i64, i64) -> ()
  %64 = executor.smin %41, %55 : i8
  cf.br ^bb1(%62, %64, %c-1_i1 : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>, i8, i1)
^bb5:  // pred: ^bb3
  executor.store %35 to %25 + %c0_i64 : i8, !executor.ptr<host_pinned>, i64
  %65 = executor.call @__cuda_get_active_device() : () -> i32
  executor.call @__cuda_memcpy_host_pinned2device(%5, %25, %c0_i64, %0, %c0_i64, %c1_i64) : (!executor.ptr<host>, !executor.ptr<host_pinned>, i64, !executor.ptr<device>, i64, i64) -> ()
  return
}
