// RUN: mlir-tensorrt-opt %s -split-input-file -cuda-simplify-stream-wait | FileCheck %s

func.func @redundant_wait_in_block() {
  %c0_i32 = arith.constant 0 : i32
  %dev = cuda.get_program_device %c0_i32 : i32
  %s0 = cuda.get_global_stream device(%dev) [0]
  %e0 = cuda.event.create device(%dev)
  cuda.stream.wait_event %s0, %e0
  cuda.stream.wait_event %s0, %e0
  cuda.event.release %e0 : !cuda.event
  return
}

// CHECK-LABEL: func.func @redundant_wait_in_block
// CHECK-COUNT-1: cuda.stream.wait_event
// CHECK-NOT: cuda.stream.wait_event
// CHECK: cuda.event.release

// -----

// Control flow: waits in different branches are not redundant (no dominance).

func.func @waits_in_branches_not_redundant(%cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %dev = cuda.get_program_device %c0_i32 : i32
  %s0 = cuda.get_global_stream device(%dev) [0]
  %e0 = cuda.event.create device(%dev)
  scf.if %cond {
    cuda.stream.wait_event %s0, %e0
  } else {
    cuda.stream.wait_event %s0, %e0
  }
  cuda.event.release %e0 : !cuda.event
  return
}

// CHECK-LABEL: func.func @waits_in_branches_not_redundant
// CHECK-COUNT-2: cuda.stream.wait_event

// -----

func.func @not_redundant_after_if(%cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %dev = cuda.get_program_device %c0_i32 : i32
  %s0 = cuda.get_global_stream device(%dev) [0]
  %e0 = cuda.event.create device(%dev)
  scf.if %cond {
  } else {
    cuda.stream.wait_event %s0, %e0
  }
  cuda.stream.wait_event %s0, %e0
  cuda.event.release %e0 : !cuda.event
  return
}

// CHECK-LABEL: func.func @not_redundant_after_if
// CHECK-COUNT-2: cuda.stream.wait_event
