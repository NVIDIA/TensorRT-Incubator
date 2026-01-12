// RUN: mlir-tensorrt-opt %s -split-input-file -cuda-test-side-effects -verify-diagnostics

// Test cuda.event.create has MemAlloc effect on result
func.func @test_event_create(%device: i32) {
  // expected-remark @below {{found an instance of 'allocate' on result #0, on resource '<Default>'}}
  %0 = cuda.event.create device(%device)
  return
}

// -----

// Test cuda.event.release has MemFree effect on operand
func.func @test_event_release(%event: !cuda.event) {
  // expected-remark @below {{found an instance of 'free' on operand #0, on resource '<Default>'}}
  cuda.event.release %event : !cuda.event
  return
}

// -----

// Test cuda.event.sync has read and write effects
func.func @test_event_sync(%event: !cuda.event) {
  // expected-remark @below {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on resource '<Default>'}}
  cuda.event.sync %event : !cuda.event
  return
}

// -----

// Test cuda.event.elapsed has read effects on both event operands
func.func @test_event_elapsed(%start: !cuda.event, %end: !cuda.event) {
  // expected-remark @below {{found an instance of 'read' on operand #0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on operand #1, on resource '<Default>'}}
  %elapsed = cuda.event.elapsed %start, %end : f32
  return
}

// -----

// Test cuda.stream.create has MemAlloc effect on result
func.func @test_stream_create(%device: i32) {
  // expected-remark @below {{found an instance of 'allocate' on result #0, on resource '<Default>'}}
  %0 = cuda.stream.create device(%device)
  return
}

// -----

// Test cuda.stream.destroy has MemFree effect on operand
func.func @test_stream_destroy(%stream: !cuda.stream) {
  // expected-remark @below {{found an instance of 'free' on operand #0, on resource '<Default>'}}
  cuda.stream.destroy %stream : !cuda.stream
  return
}

// -----

// Test cuda.stream.sync has read and write effects
func.func @test_stream_sync(%stream: !cuda.stream) {
  // expected-remark @below {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on resource '<Default>'}}
  cuda.stream.sync %stream : !cuda.stream
  return
}

// -----

// Test cuda.stream.record_event has read and write effects
func.func @test_stream_record_event(%stream: !cuda.stream, %event: !cuda.event) {
  // expected-remark @below {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on resource '<Default>'}}
  cuda.stream.record_event %stream, %event
  return
}

// -----

// Test cuda.stream.wait_event has read and write effects
func.func @test_stream_wait_event(%stream: !cuda.stream, %event: !cuda.event) {
  // expected-remark @below {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on resource '<Default>'}}
  cuda.stream.wait_event %stream, %event
  return
}

// -----

// Test cuda.blas.handle.create has MemAlloc effect on result
func.func @test_blas_handle_create() {
  // expected-remark @below {{found an instance of 'allocate' on result #0, on resource '<Default>'}}
  %0 = cuda.blas.handle.create : !cuda.blas.handle
  return
}

// -----

// Test cuda.blas.handle.destroy has MemFree effect on operand
func.func @test_blas_handle_destroy(%handle: !cuda.blas.handle) {
  // expected-remark @below {{found an instance of 'free' on operand #0, on resource '<Default>'}}
  cuda.blas.handle.destroy %handle : !cuda.blas.handle
  return
}

// -----

// Test cuda.alloc has MemAlloc effect on result
#dev = #plan.memory_space<device>

func.func @test_cuda_alloc(%stream: !cuda.stream) {
  // expected-remark @below {{found an instance of 'allocate' on result #0, on resource '<Default>'}}
  %0 = cuda.alloc() stream(%stream) : memref<4x4xf32, #dev>
  return
}

// -----

// Test cuda.dealloc has MemFree effect on memref operand
#dev = #plan.memory_space<device>

func.func @test_cuda_dealloc(%stream: !cuda.stream, %memref: memref<4x4xf32, #dev>) {
  // expected-remark @below {{found an instance of 'free' on operand #1, on resource '<Default>'}}
  cuda.dealloc stream(%stream) %memref : memref<4x4xf32, #dev>
  return
}

// -----

// Test cuda.copy_d2h has read effect on source and write effect on target
#dev = #plan.memory_space<device>
#host = #plan.memory_space<host>

func.func @test_copy_d2h(%stream: !cuda.stream,
                         %src: memref<4xf32, #dev>,
                         %dst: memref<4xf32, #host>) {
  // expected-remark @below {{found an instance of 'read' on operand #1, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on operand #2, on resource '<Default>'}}
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #dev> to memref<4xf32, #host>
  return
}

// -----

// Test cuda.copy_h2d has read effect on source and write effect on target
#dev = #plan.memory_space<device>
#host = #plan.memory_space<host>

func.func @test_copy_h2d(%stream: !cuda.stream,
                         %src: memref<4xf32, #host>,
                         %dst: memref<4xf32, #dev>) {
  // expected-remark @below {{found an instance of 'read' on operand #1, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on operand #2, on resource '<Default>'}}
  cuda.copy_h2d stream(%stream) %src, %dst : memref<4xf32, #host> to memref<4xf32, #dev>
  return
}

// -----

// Test cuda.copy_d2d has read effect on source and write effect on target
#dev = #plan.memory_space<device>

func.func @test_copy_d2d(%stream: !cuda.stream,
                         %src: memref<4xf32, #dev>,
                         %dst: memref<4xf32, #dev>) {
  // expected-remark @below {{found an instance of 'read' on operand #1, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on operand #2, on resource '<Default>'}}
  cuda.copy_d2d stream(%stream) %src, %dst : memref<4xf32, #dev> to memref<4xf32, #dev>
  return
}

// -----

// Test cuda.memset has write effect on memref
#dev = #plan.memory_space<device>

func.func @test_memset(%memref: memref<4xf32, #dev>, %val: f32) {
  // expected-remark @below {{found an instance of 'write' on operand #0, on resource '<Default>'}}
  cuda.memset %memref with %val : memref<4xf32, #dev>, f32
  return
}

// -----

// Test cuda.launch has write effect on stream and read/write effects on memref args
#dev = #plan.memory_space<device>

func.func @test_launch(%func: !cuda.function, %stream: !cuda.stream,
                       %arg: memref<4xf32, #dev>) {
  // expected-remark @below {{operation has no memory effects}}
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{operation has no memory effects}}
  %c0 = arith.constant 0 : i32
  // expected-remark @below {{found an instance of 'write' on operand #8, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on operand #9, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on operand #9, on resource '<Default>'}}
  cuda.launch %func(%arg : memref<4xf32, #dev>) with
    grid(%c1, %c1, %c1)
    block(%c1, %c1, %c1)
    smem(%c0) stream(%stream)
  return
}
