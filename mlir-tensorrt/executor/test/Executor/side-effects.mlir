// RUN: executor-opt %s -split-input-file -executor-test-side-effects -verify-diagnostics

// Test executor.alloc has MemAlloc effect on result
func.func @test_alloc(%num_bytes: i64, %alignment: i64) {
  // expected-remark @below {{found an instance of 'allocate' on result #0, on resource '<Default>'}}
  %0 = executor.alloc %num_bytes bytes align (%alignment) : (i64, i64) -> !executor.ptr<host>
  return
}

// -----

// Test executor.dealloc has MemFree effect on operand
func.func @test_dealloc(%ptr: !executor.ptr<host>) {
  // expected-remark @below {{found an instance of 'free' on operand #0, on resource '<Default>'}}
  executor.dealloc %ptr : !executor.ptr<host>
  return
}

// -----

// Test executor.load has MemRead effect on ptr operand
func.func @test_load(%ptr: !executor.ptr<host>, %offset: i64) {
  // expected-remark @below {{found an instance of 'read' on operand #0, on resource '<Default>'}}
  %0 = executor.load %ptr + %offset : (!executor.ptr<host>, i64) -> i32
  return
}

// -----

// Test executor.store has MemWrite effect on ptr operand
func.func @test_store(%ptr: !executor.ptr<host>, %offset: i64, %value: i32) {
  // expected-remark @below {{found an instance of 'write' on operand #0, on resource '<Default>'}}
  executor.store %value to %ptr + %offset : i32, !executor.ptr<host>, i64
  return
}

// -----

// Test executor.memcpy has MemRead on src (operand #0) and MemWrite on dest (operand #2)
func.func @test_memcpy(%src: !executor.ptr<host>, %dest: !executor.ptr<host>,
                       %src_offset: i64, %dest_offset: i64, %num_bytes: i64) {
  // expected-remark @below {{found an instance of 'read' on operand #0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on operand #2, on resource '<Default>'}}
  executor.memcpy %src + %src_offset to %dest + %dest_offset size %num_bytes
    : !executor.ptr<host>, i64, !executor.ptr<host>, i64, i64
  return
}

// -----

// Test executor.strided_memref_copy has read and write effects
func.func @test_strided_memref_copy(
    %rank: i32, %elem_size: i64,
    %shape: !executor.ptr<host>,
    %src_strides: !executor.ptr<host>,
    %dst_strides: !executor.ptr<host>,
    %src_aligned_ptr: !executor.ptr<host>,
    %dest_aligned_ptr: !executor.ptr<host>,
    %src_offset: i64,
    %dest_offset: i64) {
  // expected-remark @below {{found an instance of 'read' on operand #2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on operand #3, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on operand #5, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on operand #6, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on operand #8, on resource '<Default>'}}
  executor.strided_memref_copy (
    %rank, %elem_size, %shape,
    %src_aligned_ptr,
    %src_offset, %src_strides,
    %dest_aligned_ptr,
    %dest_offset, %dst_strides)
    : i32, i64, !executor.ptr<host>,
      !executor.ptr<host>, i64, !executor.ptr<host>,
      !executor.ptr<host>, i64, !executor.ptr<host>
  return
}

// -----

// Test executor.print has MemWrite effect (no specific operand)
func.func @test_print(%value: i32) {
  // expected-remark @below {{found an instance of 'write' on resource '<Default>'}}
  executor.print "value: %d" (%value : i32)
  return
}

// -----

// Test executor.alloca has MemAlloc effect on result with AutomaticAllocationScope resource
func.func @test_alloca(%num_elements: i64) {
  // expected-remark @below {{found an instance of 'allocate' on result #0, on resource 'AutomaticAllocationScope'}}
  %0 = executor.alloca %num_elements x i32 : (i64) -> !executor.ptr<host>
  return
}
