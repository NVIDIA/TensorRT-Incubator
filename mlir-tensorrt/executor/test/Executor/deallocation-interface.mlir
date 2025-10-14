// RUN: executor-opt -verify-diagnostics --buffer-deallocation-pipeline -split-input-file %s | FileCheck %s

// Test 1: abi.send with unique ownership should drop deallocation
// CHECK-LABEL: func @abi_send_drops_dealloc
func.func @abi_send_drops_dealloc(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  // CHECK: %[[ALLOC:.*]] = memref.alloc
  %c10 = arith.constant 10 : index
  %alloc = memref.alloc(%c10) : memref<?xi32>
  %cast = memref.cast %alloc : memref<?xi32> to memref<10xi32>

  // CHECK: executor.abi.send %[[CAST:.*]] to %arg1
  // CHECK-NOT: bufferization.dealloc
  executor.abi.send %cast to %arg1 : memref<10xi32>
  return
}

// -----

// Test 2: abi.recv and abi.send round-trip
// CHECK-LABEL: func @abi_recv_send_roundtrip
func.func @abi_recv_send_roundtrip(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<5xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<5xf32>>})
      attributes {
        executor.func_abi = (memref<5xf32>) -> (memref<5xf32>)
      } {
  // CHECK: %[[RECV:.*]] = executor.abi.recv %arg0
  %0 = executor.abi.recv %arg0 : memref<5xf32>
  // CHECK: executor.abi.send %[[RECV]] to %arg1
  executor.abi.send %0 to %arg1 : memref<5xf32>
  return
}

// -----

// Test 3: abi.send should handle memref with ownership correctly
// CHECK-LABEL: func @abi_send_with_ownership
func.func @abi_send_with_ownership(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<4x8xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<4x8xf32>>})
      attributes {
        executor.func_abi = (memref<4x8xf32>) -> (memref<4x8xf32>)
      } {
  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  %alloc = memref.alloc() : memref<4x8xf32>

  // CHECK: executor.abi.send %[[ALLOC]] to %arg1
  executor.abi.send %alloc to %arg1 : memref<4x8xf32>

  // CHECK-NOT: bufferization.dealloc
  return
}

// -----

// Test 5: abi.recv should remain unchanged by deallocation pass
// CHECK-LABEL: func @abi_recv_unchanged
func.func @abi_recv_unchanged(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  // CHECK: %[[RECV:.*]] = executor.abi.recv %arg0 : memref<10xi32>
  %0 = executor.abi.recv %arg0 : memref<10xi32>

  // CHECK: executor.abi.send %[[RECV]] to %arg1
  executor.abi.send %0 to %arg1 : memref<10xi32>
  return
}

// -----

// Test 6: Complex control flow with abi.send
// CHECK-LABEL: func @abi_send_control_flow
func.func @abi_send_control_flow(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  // CHECK: %[[ALLOC:.*]] = memref.alloc
  %alloc = memref.alloc() : memref<10xi32>

  // Perform some conditional logic
  %c0 = arith.constant 0 : index
  %load = memref.load %alloc[%c0] : memref<10xi32>
  %c10 = arith.constant 10 : i32
  %cond = arith.cmpi slt, %load, %c10 : i32

  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // CHECK: ^bb1:
  // CHECK: executor.abi.send {{.*}} to %arg1
  // CHECK-NOT: bufferization.dealloc
  executor.abi.send %alloc to %arg1 : memref<10xi32>
  return

^bb2:
  // CHECK: ^bb2:
  // CHECK: executor.abi.send {{.*}} to %arg1
  // CHECK-NOT: bufferization.dealloc
  executor.abi.send %alloc to %arg1 : memref<10xi32>
  return
}

// -----

// Test 8: Multiple abi.send operations
// CHECK-LABEL: func @multiple_abi_send
func.func @multiple_abi_send(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>},
    %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<20xf32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>, memref<20xf32>)
      } {
  // CHECK: %[[ALLOC1:.*]] = memref.alloc
  %alloc1 = memref.alloc() : memref<10xi32>

  // CHECK: %[[ALLOC2:.*]] = memref.alloc
  %alloc2 = memref.alloc() : memref<20xf32>

  // CHECK: executor.abi.send %[[ALLOC1]] to %arg1
  executor.abi.send %alloc1 to %arg1 : memref<10xi32>

  // CHECK: executor.abi.send %[[ALLOC2]] to %arg2
  executor.abi.send %alloc2 to %arg2 : memref<20xf32>

  // CHECK-NOT: bufferization.dealloc
  return
}
