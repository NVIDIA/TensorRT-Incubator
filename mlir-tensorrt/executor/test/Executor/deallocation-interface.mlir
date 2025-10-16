// RUN: executor-opt -verify-diagnostics --buffer-deallocation-pipeline -canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @abi_send_drops_dealloc
func.func @abi_send_drops_dealloc(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  // CHECK-DAG: %[[ALLOC:.*]] = memref.alloc
  // CHECK-DAG: %[[OWN:.*]] = arith.constant true
  %c10 = arith.constant 10 : index
  %alloc = memref.alloc(%c10) : memref<?xi32>
  %cast = memref.cast %alloc : memref<?xi32> to memref<10xi32>
  // CHECK: executor.abi.send %[[CAST:.*]] to %{{.*}} ownership(%[[OWN]])
  // CHECK-NOT: bufferization.dealloc
  executor.abi.send %cast to %arg1 : memref<10xi32>
  return
}

// -----

// CHECK-LABEL: func @abi_recv_send_roundtrip
func.func @abi_recv_send_roundtrip(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<5xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<5xf32>>})
      attributes {
        executor.func_abi = (memref<5xf32>) -> (memref<5xf32>)
      } {
  // CHECK-DAG: %[[RECV:.*]] = executor.abi.recv %arg0
  // CHECK-DAG: %[[OWN:.*]] = arith.constant false
  %0 = executor.abi.recv %arg0 : memref<5xf32>
  // CHECK: executor.abi.send %[[RECV]] to %arg1 ownership(%[[OWN]])
  executor.abi.send %0 to %arg1 : memref<5xf32>
  return
}

// -----

// CHECK-LABEL: func @abi_send_with_ownership
func.func @abi_send_with_ownership(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<4x8xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<4x8xf32>>})
      attributes {
        executor.func_abi = (memref<4x8xf32>) -> (memref<4x8xf32>)
      } {
  // CHECK-DAG: %[[ALLOC:.*]] = memref.alloc()
  // CHECK-DAG: %[[OWN:.*]] = arith.constant true
  %alloc = memref.alloc() : memref<4x8xf32>

  // CHECK: executor.abi.send %[[ALLOC]] to %{{.*}} ownership(%[[OWN]])
  executor.abi.send %alloc to %arg1 : memref<4x8xf32>

  // CHECK-NOT: bufferization.dealloc
  return
}

// -----

// CHECK-LABEL: func @abi_same_ptr
func.func @abi_same_ptr(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  // CHECK-DAG: %[[RECV:.*]] = executor.abi.recv %{{.*}} : memref<10xi32>
  // CHECK-DAG: %[[OWN:.*]] = arith.constant false
  %0 = executor.abi.recv %arg1 : memref<10xi32>
  // CHECK: executor.abi.send %[[RECV]] to %arg1 ownership(%[[OWN]])
  executor.abi.send %0 to %arg1 : memref<10xi32>
  return
}

// -----

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
  // CHECK: executor.abi.send {{.*}} to %arg1 ownership(%[[OWN:.+]])
  // CHECK: scf.if %[[OWN]]
  // CHECK-NEXT: dealloc
  executor.abi.send %alloc to %arg1 : memref<10xi32>
  return

^bb2:
  // CHECK: ^bb2:
  // CHECK: executor.abi.send {{.*}} to %arg1 ownership(%[[OWN:.+]])
  // CHECK: scf.if %[[OWN]]
  // CHECK-NEXT: dealloc
  executor.abi.send %alloc to %arg1 : memref<10xi32>
  return
}


// -----

// CHECK-LABEL: func @abi_send_undef_output_recv
func.func @abi_send_undef_output_recv(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>, undef>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  // CHECK: %[[RECV:.*]] = executor.abi.recv %arg0
  %0 = executor.abi.recv %arg0 : memref<10xi32>

  // With undef=true, we need to ensure unique ownership, so a copy may be created
  // CHECK: executor.abi.send {{.*}} to %arg1 ownership(%[[OWN:.*]])
  // CHECK-NOT: dealloc
  executor.abi.send %0 to %arg1 : memref<10xi32>
  return
}

// -----

// CHECK-LABEL: func @abi_send_undef_output_alloc
func.func @abi_send_undef_output_alloc(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>, undef>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  // CHECK: %[[ALLOC:.*]] = memref.alloc
  %alloc = memref.alloc() : memref<10xi32>

  // With undef=true, we must provide unique ownership
  // CHECK: executor.abi.send {{.*}} to %arg1 ownership(%[[OWN:.*]])
  // CHECK-NOT: dealloc
  executor.abi.send %alloc to %arg1 : memref<10xi32>
  return
}

// -----

// CHECK-LABEL: func @abi_send_undef_shared_ownership
func.func @abi_send_undef_shared_ownership(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>, undef>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  // CHECK: %[[ALLOC:.*]] = memref.alloc
  %alloc = memref.alloc() : memref<10xi32>

  // Use the allocation (creates shared ownership)
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : i32
  memref.store %c42, %alloc[%c0] : memref<10xi32>

  // With undef=true and potentially shared ownership, a copy may be needed
  // CHECK: executor.abi.send {{.*}} to %arg1 ownership(%[[OWN:.*]])
  // CHECK-NOT: dealloc
  executor.abi.send %alloc to %arg1 : memref<10xi32>
  return
}

// -----

// CHECK-LABEL: func @abi_send_undef_control_flow
func.func @abi_send_undef_control_flow(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: i1,
    %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>, undef>})
      attributes {
        executor.func_abi = (memref<10xi32>, i1) -> (memref<10xi32>)
      } {
  // CHECK: %[[ALLOC1:.*]] = memref.alloc
  %alloc1 = memref.alloc() : memref<10xi32>
  // CHECK: %[[ALLOC2:.*]] = memref.alloc
  %alloc2 = memref.alloc() : memref<10xi32>

  cf.cond_br %arg1, ^bb1, ^bb2

^bb1:
  // CHECK: ^bb1:
  // CHECK: executor.abi.send {{.*}} to %arg2 ownership({{.*}})
  // CHECK-NOT: dealloc
  executor.abi.send %alloc1 to %arg2 : memref<10xi32>
  return

^bb2:
  // CHECK: ^bb2:
  // CHECK: executor.abi.send {{.*}} to %arg2 ownership({{.*}})
  // CHECK-NOT: dealloc
  executor.abi.send %alloc2 to %arg2 : memref<10xi32>
  return
}

// -----

// CHECK-LABEL: func @abi_send_mixed_undef
func.func @abi_send_mixed_undef(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>},
    %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<20xf32>, undef>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>, memref<20xf32>)
      } {
  // CHECK: %[[ALLOC1:.*]] = memref.alloc
  %alloc1 = memref.alloc() : memref<10xi32>
  // CHECK: %[[ALLOC2:.*]] = memref.alloc
  %alloc2 = memref.alloc() : memref<20xf32>

  // First output without undef
  // It will get lowered to a copy, so we still deallocate the first allocation.
  // CHECK: executor.abi.send %[[ALLOC1]] to %arg1 ownership(%[[OWN1:.*]])
  executor.abi.send %alloc1 to %arg1 : memref<10xi32>

  // Second output with undef=true
  // CHECK: executor.abi.send %[[ALLOC2]] to %arg2 ownership(%[[OWN2:.*]])
  executor.abi.send %alloc2 to %arg2 : memref<20xf32>
  // CHECK: memref.dealloc %[[ALLOC1]]
  // CHECK-NOT: memref.dealloc
  // CHECK: return
  return
}
