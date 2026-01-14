// RUN: mlir-tensorrt-opt %s -split-input-file -cuda-schedule-async | FileCheck %s --check-prefix=CHECK

#dev = #plan.memory_space<device>
#host = #plan.memory_space<host>

func.func @schedule_copy_compute_dependency(%devBuf: memref<4xf32, #dev>,
                                           %hostBuf: memref<4xf32, #host>) {
  %device = cuda.get_active_device
  %stream0 = cuda.get_global_stream device(%device) [0]

  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i64 = arith.constant 0 : i64
  %f = builtin.unrealized_conversion_cast %c0_i64 : i64 to !cuda.function

  // Compute-like command (scheduled to stream 0).
  cuda.launch %f(%devBuf : memref<4xf32, #dev>) with
    grid(%c1_i32, %c1_i32, %c1_i32)
    block(%c1_i32, %c1_i32, %c1_i32)
    smem(%c0_i32) stream(%stream0)

  // Copy-like command (scheduled to stream 1). Depends on the launch via
  // shared memory access to %devBuf, so we must insert record/wait across
  // streams.
  cuda.copy_d2h stream(%stream0) %devBuf, %hostBuf : memref<4xf32, #dev> to memref<4xf32, #host>
  return
}

// CHECK-LABEL: func.func @schedule_copy_compute_dependency(
// CHECK: %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK: %[[pDev:.+]] = cuda.get_program_device %[[c0_i32]] : i32
// CHECK: %[[s0:.+]] = cuda.get_global_stream device(%[[pDev]]) [0]
// CHECK: %[[s1:.+]] = cuda.get_global_stream device(%[[pDev]]) [1]
// CHECK: %[[s2:.+]] = cuda.get_global_stream device(%[[pDev]]) [2]

// CHECK-DAG: cuda.stream.record_event %[[s0]], %[[input_ready:.+]]
// CHECK-DAG: cuda.stream.wait_event %[[s1]], %[[input_ready]]
// CHECK-DAG: cuda.stream.wait_event %[[s2]], %[[input_ready]]
//     CHECK: cuda.event.release %[[input_ready]]

// CHECK: cuda.launch %{{.+}}(%{{.+}} : memref<4xf32, #plan.memory_space<device>>) with
// CHECK:   smem(%{{.+}}) stream(%[[s2]])
// CHECK: %[[e0:.+]] = cuda.event.create device(%[[pDev]])
// CHECK: cuda.stream.record_event %[[s2]], %[[e0]]
// CHECK: cuda.stream.wait_event %[[s1]], %[[e0]]
// CHECK: cuda.copy_d2h stream(%[[s1]]) %{{.+}}, %{{.+}} : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
// CHECK: cuda.event.release %[[e0]] : !cuda.event
// CHECK-DAG: cuda.stream.record_event %[[s1]], %[[F1:.+]]
// CHECK-DAG: cuda.stream.record_event %[[s2]], %[[F2:.+]]
// CHECK-DAG: cuda.stream.wait_event %[[s0]], %[[F1]]
// CHECK-DAG: cuda.stream.wait_event %[[s0]], %[[F2]]
// CHECK-DAG: cuda.event.release %[[F1]]
// CHECK-DAG: cuda.event.release %[[F2]]
// CHECK: return

// -----

// Test that events created inside a loop body are released within the same
// loop body, not at the function return. This ensures proper scoping.

#dev = #plan.memory_space<device>
#host = #plan.memory_space<host>

func.func @schedule_inside_loop(%devBuf: memref<4xf32, #dev>,
                                %hostBuf: memref<4xf32, #host>,
                                %n: index) {
  %device = cuda.get_active_device
  %stream0 = cuda.get_global_stream device(%device) [0]

  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f = builtin.unrealized_conversion_cast %c0_i64 : i64 to !cuda.function

  scf.for %i = %c0 to %n step %c1 {
    // Compute on stream 0
    cuda.launch %f(%devBuf : memref<4xf32, #dev>) with
      grid(%c1_i32, %c1_i32, %c1_i32)
      block(%c1_i32, %c1_i32, %c1_i32)
      smem(%c0_i32) stream(%stream0)

    // Copy on stream 1 - needs event sync from stream 0
    cuda.copy_d2h stream(%stream0) %devBuf, %hostBuf : memref<4xf32, #dev> to memref<4xf32, #host>
  }
  return
}

// CHECK-LABEL: func.func @schedule_inside_loop(
// CHECK-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[pDev:.+]] = cuda.get_program_device %[[c0_i32]] : i32
// CHECK-DAG: %[[STREAM0:.+]] = cuda.get_global_stream device(%[[pDev]]) [0]
// CHECK-DAG: %[[STREAM1:.+]] = cuda.get_global_stream device(%[[pDev]]) [1]
// CHECK-DAG: %[[STREAM2:.+]] = cuda.get_global_stream device(%[[pDev]]) [2]

// CHECK-DAG: cuda.stream.record_event %[[s0]], %[[input_ready:.+]]
// CHECK-DAG: cuda.stream.wait_event %[[s1]], %[[input_ready]]
// CHECK-DAG: cuda.stream.wait_event %[[s2]], %[[input_ready]]
//     CHECK: cuda.event.release %[[input_ready]]

// CHECK: scf.for
// CHECK:   cuda.launch
// CHECK:   stream(%[[STREAM2]])
// CHECK:   %[[EVT:.+]] = cuda.event.create device(%[[pDev]])
// CHECK:   cuda.stream.record_event %[[STREAM2]], %[[EVT]]
// CHECK:   cuda.stream.wait_event %[[STREAM1]], %[[EVT]]
// CHECK:   cuda.copy_d2h stream(%[[STREAM1]])
// CHECK:   cuda.event.release %[[EVT]] : !cuda.event
// CHECK: }
// CHECK-DAG: cuda.stream.record_event %[[STREAM1]], %[[F1:.+]]
// CHECK-DAG: cuda.stream.record_event %[[STREAM2]], %[[F2:.+]]
// CHECK-DAG: cuda.stream.wait_event %[[STREAM0]], %[[F1]]
// CHECK-DAG: cuda.stream.wait_event %[[STREAM0]], %[[F2]]
// CHECK-DAG: cuda.event.release %[[F1]]
// CHECK-DAG: cuda.event.release %[[F2]]
// CHECK: return
