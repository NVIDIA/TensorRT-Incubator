// RUN: executor-opt %s -inline -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable \
// RUN:   | executor-runner -input-type=rtexe -features=core,cuda | FileCheck %s

!scalar_type = f32
!host_memref_type = memref<4x!scalar_type, #executor.memory_type<host>>
!device_memref_type = memref<4x!scalar_type, #executor.memory_type<device>>

executor.func private @__cuda_stream_create() -> (!executor.ptr<host>)
executor.func private @__cuda_stream_sync(!executor.ptr<host>) -> ()
executor.func private @__cuda_alloc_device(!executor.ptr<host>, i32, i64, i32) -> (!executor.ptr<device>)
executor.func private @__cuda_memset_32(!executor.ptr<device>, i64, i64, i32) -> ()
executor.func private @executor_alloc(i64, i32) -> (!executor.ptr<host>)
executor.func private @__cuda_memcpy_device2host(!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<host>, i64, i64) -> ()

func.func private @print_tensor(
    %arg0: !executor.ptr<host>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %i_i64 = arith.index_cast %i : index to i64
    %offset = executor.getoffset  [%i_i64] : (i64) -> i64, f32
    %el = executor.load %arg0 + %offset : (!executor.ptr<host>, i64) -> f32
    executor.print "[%d] = %.2f"(%i, %el : index, !scalar_type)
  }
  return
}


func.func @main() -> i32 {
  %fill_value = arith.constant 1.1 : !scalar_type
  %fill_value_i32 = arith.bitcast %fill_value : !scalar_type to i32
  %c0_i32 = arith.constant 0 : i32
  %c0_i64 = arith.constant 0 : i64
  %c16 = arith.constant 16 : i64
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 1 : i32
  %stream = executor.call @__cuda_stream_create() : () -> (!executor.ptr<host>)
  %1 = executor.call @__cuda_alloc_device(%stream, %c0_i32, %c16, %c4) : (!executor.ptr<host>, i32, i64, i32) -> (!executor.ptr<device>)
  %host = executor.call @executor_alloc(%c16, %c4) : (i64, i32) -> (!executor.ptr<host>)
  executor.call @__cuda_stream_sync(%stream) : (!executor.ptr<host>) -> ()
  executor.call @__cuda_memset_32(%1, %c0_i64, %c16, %fill_value_i32) : (!executor.ptr<device>, i64, i64, i32) -> ()
  executor.call @__cuda_memcpy_device2host(%stream, %1, %c0_i64, %host, %c0_i64, %c16)
    : (!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<host>, i64, i64) -> ()

  executor.call @__cuda_stream_sync(%stream) : (!executor.ptr<host>) -> ()

  func.call @print_tensor(%host): (!executor.ptr<host>) -> ()

  executor.print "synchronized"()

  return %c0_i32 : i32
}

//      CHECK: [0] = 1.1
// CHECK-NEXT: [1] = 1.1
// CHECK-NEXT: [2] = 1.1
// CHECK-NEXT: [3] = 1.1
