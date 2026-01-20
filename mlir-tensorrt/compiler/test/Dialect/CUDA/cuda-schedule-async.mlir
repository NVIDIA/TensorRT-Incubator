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

// CHECK: %[[input_ready:.+]] = cuda.event.create_on_stream %[[s0]]
// CHECK-DAG: cuda.stream.wait_event %[[s1]], %[[input_ready]]
// CHECK-DAG: cuda.stream.wait_event %[[s2]], %[[input_ready]]
//     CHECK: cuda.event.release %[[input_ready]]

// CHECK: cuda.launch %{{.+}}(%{{.+}} : memref<4xf32, #plan.memory_space<device>>) with
// CHECK:   smem(%{{.+}}) stream(%[[s2]])
// CHECK: %[[e0:.+]] = cuda.event.create_on_stream %[[s2]]
// CHECK: cuda.stream.wait_event %[[s1]], %[[e0]]
// CHECK: cuda.copy_d2h stream(%[[s1]]) %{{.+}}, %{{.+}} : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
// CHECK: cuda.event.release %[[e0]] : !cuda.event
// CHECK-DAG: %[[F1:.+]] = cuda.event.create_on_stream %[[s1]]
// CHECK-DAG: %[[F2:.+]] = cuda.event.create_on_stream %[[s2]]
// CHECK-DAG: cuda.stream.wait_event %[[s0]], %[[F1]]
// CHECK-DAG: cuda.stream.wait_event %[[s0]], %[[F2]]
// CHECK-DAG: cuda.event.release %[[F1]]
// CHECK-DAG: cuda.event.release %[[F2]]
// CHECK: return

// -----

// Test that cuda-schedule-async still schedules ops outside the CUDA dialect
// when they are stream-schedulable (e.g. TensorRTRuntime enqueue ops).

#dev = #plan.memory_space<device>

trtrt.compiled_func @engine dense<[0]> : tensor<1xi8>

func.func @schedule_trtrt_enqueue(%input: memref<4xf32, #dev>,
                                 %output: memref<4xf32, #dev>) {
  %device = cuda.get_active_device
  %stream0 = cuda.get_global_stream device(%device) [0]
  %ctx = trtrt.get_function @engine : !trtrt.context

  trtrt.enqueue %ctx stream(%stream0) (%input) outs(%output)
    : (memref<4xf32, #dev>) -> (memref<4xf32, #dev>)
  return
}

// CHECK-LABEL: func.func @schedule_trtrt_enqueue(
// CHECK: %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK: %[[pDev:.+]] = cuda.get_program_device %[[c0_i32]] : i32
// CHECK: %[[s0:.+]] = cuda.get_global_stream device(%[[pDev]]) [0]
// CHECK: %[[s1:.+]] = cuda.get_global_stream device(%[[pDev]]) [1]
// CHECK: %[[s2:.+]] = cuda.get_global_stream device(%[[pDev]]) [2]
// CHECK: trtrt.enqueue %{{.+}} stream(%[[s2]])
// CHECK: return

// -----

// Test that cuda-schedule-async can schedule `executor.call_plugin` when it
// carries a `!cuda.stream` stream operand (via external model registration for
// StreamSchedulableOp).

executor.plugin @my_plugin {
  plugin_name = "test_plugin",
  function_name = "my_func",
  config = {},
  ffi_backend = #executor.ffi_backend<tvm_ffi>
} : () -> ()

func.func @schedule_executor_call_plugin() {
  %device = cuda.get_active_device
  %stream0 = cuda.get_global_stream device(%device) [0]
  %c0_i32 = arith.constant 0 : i32
  executor.call_plugin @my_plugin
    stream(%stream0 : !cuda.stream)
    ins(%c0_i32 : i32) outs(%c0_i32 : i32) {arg_spec = ["args.0", "rets.0"]}
  return
}

// CHECK-LABEL: func.func @schedule_executor_call_plugin(
// CHECK: %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK: %[[pDev:.+]] = cuda.get_program_device %[[c0_i32]] : i32
// CHECK: %[[s0:.+]] = cuda.get_global_stream device(%[[pDev]]) [0]
// CHECK: %[[s1:.+]] = cuda.get_global_stream device(%[[pDev]]) [1]
// CHECK: %[[s2:.+]] = cuda.get_global_stream device(%[[pDev]]) [2]
// CHECK: executor.call_plugin @my_plugin stream(%[[s2]] : !cuda.stream)
// CHECK: return

// -----

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

// CHECK: %[[input_ready:.+]] = cuda.event.create_on_stream %[[s0]]
// CHECK-DAG: cuda.stream.wait_event %[[s1]], %[[input_ready]]
// CHECK-DAG: cuda.stream.wait_event %[[s2]], %[[input_ready]]
//     CHECK: cuda.event.release %[[input_ready]]

// CHECK: scf.for
// CHECK:   cuda.launch
// CHECK:   stream(%[[STREAM2]])
// CHECK:   %[[EVT:.+]] = cuda.event.create_on_stream %[[STREAM2]] : !cuda.stream
// CHECK:   cuda.stream.wait_event %[[STREAM1]], %[[EVT]]
// CHECK:   cuda.copy_d2h stream(%[[STREAM1]])
// CHECK:   cuda.event.release %[[EVT]] : !cuda.event
// CHECK: }
// CHECK-DAG: %[[F1:.+]] = cuda.event.create_on_stream %[[STREAM1]] : !cuda.stream
// CHECK-DAG: %[[F2:.+]] = cuda.event.create_on_stream %[[STREAM2]] : !cuda.stream
// CHECK-DAG: cuda.stream.wait_event %[[STREAM0]], %[[F1]]
// CHECK-DAG: cuda.stream.wait_event %[[STREAM0]], %[[F2]]
// CHECK-DAG: cuda.event.release %[[F1]]
// CHECK-DAG: cuda.event.release %[[F2]]
// CHECK: return
