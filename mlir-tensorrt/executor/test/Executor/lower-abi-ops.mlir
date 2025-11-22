// RUN: executor-opt %s -executor-lower-abi-ops -canonicalize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @test_scalar_passthrough
func.func @test_scalar_passthrough(%arg0: i32, %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, i32>}) -> i32
    attributes {executor.func_abi = (i32) -> (i32)} {
  // CHECK: executor.abi.send %arg0 to %arg1 : i32
  %0 = executor.abi.send %arg0 to %arg1 : i32
  return %0 : i32
}

// -----

// CHECK-LABEL: @test_complex_passthrough
func.func @test_complex_passthrough(%arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, complex<f32>>},
                                     %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, complex<f32>>}) -> complex<f32>
    attributes {executor.func_abi = (complex<f32>) -> (complex<f32>)} {
  // CHECK: %[[V:.+]] = executor.abi.recv %arg0
  %0 = executor.abi.recv %arg0 : complex<f32>
  // CHECK: executor.abi.send %[[V]] to %arg1 : complex<f32>
  %1 = executor.abi.send %0 to %arg1 : complex<f32>
  return %1 : complex<f32>
}

// -----

// Test error case: unsupported type (neither scalar, complex, nor memref)
func.func @test_unsupported_type(%arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, !executor.table<i32>>},
                                  %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, !executor.table<i32>>}) -> !executor.table<i32>
    attributes {executor.func_abi = (!executor.table<i32>) -> (!executor.table<i32>)} {
  %0 = executor.abi.recv %arg0 : !executor.table<i32>
  %1 = executor.abi.send %0 to %arg1 : !executor.table<i32>
  // expected-error@+1 {{value type must be scalar, complex, or memref type}}
  return %1 : !executor.table<i32>
}
// -----

func.func private @get_memref() -> (memref<10xf32>, i1)

// CHECK-LABEL: @test_send_undef
// CHECK-SAME: (%[[arg0:.+]]:
func.func @test_send_undef(
   %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32>, undef>}) -> memref<10xf32>
    attributes {executor.func_abi = () -> (memref<10xf32>)} {
  // CHECK: %[[v0:.+]]:2 = call @get_memref()
  %0, %1 = call @get_memref() : () -> (memref<10xf32>, i1)
  // CHECK: executor.abi.send %[[v0]]#0 to %[[arg0]]
  return %0 : memref<10xf32>
}
