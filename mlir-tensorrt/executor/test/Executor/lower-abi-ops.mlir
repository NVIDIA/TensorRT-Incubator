// RUN: executor-opt %s -executor-lower-abi-ops -canonicalize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @test_scalar_passthrough
func.func @test_scalar_passthrough(%arg0: i32, %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, i32>})
    attributes {executor.func_abi = (i32) -> (i32)} {
  // CHECK: executor.abi.send %arg0 to %arg1 : i32
  executor.abi.send %arg0 to %arg1 : i32
  return
}

// -----

// CHECK-LABEL: @test_complex_passthrough
func.func @test_complex_passthrough(%arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, complex<f32>>},
                                     %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, complex<f32>>})
    attributes {executor.func_abi = (complex<f32>) -> (complex<f32>)} {
  // CHECK: %[[V:.+]] = executor.abi.recv %arg0
  %0 = executor.abi.recv %arg0 : complex<f32>
  // CHECK: executor.abi.send %[[V]] to %arg1 : complex<f32>
  executor.abi.send %0 to %arg1 : complex<f32>
  return
}

// -----

// CHECK-LABEL: @test_memref_recv_send_same_ptr
func.func @test_memref_recv_send_same_ptr(
      %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xf32>>},
      %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32>>})
    attributes {executor.func_abi = (memref<10xf32>) -> (memref<10xf32>)} {
  %0 = executor.abi.recv %arg1 : memref<10xf32>
  %false = arith.constant false
  executor.abi.send %0 to %arg1 ownership(%false) : memref<10xf32>
  // CHECK-NOT: executor.abi.send
  // CHECK: return
  return
}

// -----

// Test error case: unsupported type (neither scalar, complex, nor memref)
func.func @test_unsupported_type(%arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, !executor.table<i32>>},
                                  %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, !executor.table<i32>>})
    attributes {executor.func_abi = (!executor.table<i32>) -> (!executor.table<i32>)} {
  %0 = executor.abi.recv %arg0 : !executor.table<i32>
  // expected-error@+1 {{value type must be scalar, complex, or memref type}}
  executor.abi.send %0 to %arg1 : !executor.table<i32>
  return
}
// -----

func.func private @get_memref() -> (memref<10xf32>, i1)

// CHECK-LABEL: @test_send_undef
func.func @test_send_undef(
   %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32>, undef>})
    attributes {executor.func_abi = () -> (memref<10xf32>)} {
  // CHECK-DAG: %[[ownership:.+]] = arith.constant true
  // CHECK-DAG: %[[V0:.+]]:2 = call @get_memref()
  %0, %1 = call @get_memref() : () -> (memref<10xf32>, i1)
  // CHECK: %[[V1:.+]] = scf.if %[[V0]]#1
  // CHECK:   scf.yield %[[V0]]#0 :
  // CHECK: } else {
  // CHECK:   %[[cloned:.+]] = bufferization.clone %[[V0]]#0
  // CHECK:   scf.yield %[[cloned]] :
  // CHECK: }
  // CHECK: executor.abi.send %[[V1]] to %arg0 ownership(%[[ownership]]) :
  executor.abi.send %0 to %arg0 ownership(%1) : memref<10xf32>
  return
}


// -----

func.func private @get_memref() -> (memref<10xf32>, i1)

// CHECK-LABEL: @test_send_no_undef
func.func @test_send_no_undef(
   %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32>>})
    attributes {executor.func_abi = () -> (memref<10xf32>)} {
  // %[[V0:.+]] = call @get_memref()
  %0, %1 = call @get_memref() : () -> (memref<10xf32>, i1)
  // CHECK-COUNT-2: memref.extract_aligned_pointer
  // CHECK: arith.cmpi eq
  // CHECK: scf.if
  // CHECK:  cf.assert
  // CHECK: else
  // CHECK:  memref.copy
  executor.abi.send %0 to %arg0 ownership(%1) : memref<10xf32>
  // CHECK-NOT: executor.abi.send
  return
  // CHECK: return
}
