// RUN: mlir-tensorrt-opt %s -split-input-file -cuda-expand-ops | FileCheck %s

func.func @expand_event_create_on_stream(%device: i32) -> !cuda.event {
  %stream = cuda.stream.create device(%device)
  %event = cuda.event.create_on_stream %stream : !cuda.stream
  return %event : !cuda.event
}

// CHECK-LABEL: func.func @expand_event_create_on_stream
// CHECK: %[[STREAM:.+]] = cuda.stream.create device(%{{.*}})
// CHECK: %[[EVENT:.+]] = cuda.event.create device(%{{.*}})
// CHECK-NEXT: cuda.stream.record_event %[[STREAM]], %[[EVENT]]
// CHECK: return %[[EVENT]] : !cuda.event
