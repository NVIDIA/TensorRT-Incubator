// RUN: executor-opt %s -split-input-file -verify-diagnostics -split-input-file

// Test 1: Input argument with byval and scalar type (i32) - should fail
func.func @arg_abi_input_byval_scalar_i32(
    // expected-error @below {{function arg_abi_input_byval_scalar_i32 argument 0 has ABI #executor.arg<byval, i32> but input arguments passed by-val cannot have scalar value types}}
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, i32>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, i32>})
      attributes {
        executor.func_abi = (i32) -> (i32)
      } {
  return
}

// -----

// Test 2: Input argument with byval and scalar float type - should fail
func.func @arg_abi_input_byval_scalar_float(
    // expected-error @below {{function arg_abi_input_byval_scalar_float argument 0 has ABI #executor.arg<byval, f32> but input arguments passed by-val cannot have scalar value types}}
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, f32>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, f32>})
      attributes {
        executor.func_abi = (f32) -> (f32)
      } {
  return
}

// -----

// Test 3: Input argument with byval and index type - should fail
func.func @arg_abi_input_byval_scalar_index(
    // expected-error @below {{function arg_abi_input_byval_scalar_index argument 0 has ABI #executor.arg<byval, index> but input arguments passed by-val cannot have scalar value types}}
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, index>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, index>})
      attributes {
        executor.func_abi = (index) -> (index)
      } {
  return
}

// -----

// Test 4: Input argument with byval and memref type - should pass
func.func @arg_abi_input_byval_memref(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  return
}

// -----

// Test 5: Input argument with byref instead of byval - should fail
func.func @arg_abi_input_byref(
    // expected-error @below {{expected executor.abi input argument 0 to have 'byval' ABI kind but got #executor.arg<byref, memref<10xi32>>}}
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  return
}

// -----

// Test 6: Output argument with byval instead of byref - should fail
func.func @arg_abi_output_byval(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    // expected-error @below {{expected executor.abi output argument 0 to have 'byref' ABI kind but got #executor.arg<byval, memref<10xi32>>}}
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  return
}

// -----

// Test 8: Argument with executor.abi but not a host pointer - should fail
// expected-error @below {{expected executor.abi attribute to be attached to a host pointer type argument but got '!executor.ptr<device>'}}
func.func @arg_abi_not_host_pointer(
    %arg0: !executor.ptr<device> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  return
}

// -----

// Test 9: Argument with executor.abi but missing executor.func_abi - should fail
// expected-error @below {{expected executor.func_abi attribute to be TypeAttr with a FunctionType attached to the function containing arguments decorated with executor.abi}}
func.func @arg_abi_missing_func_abi(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>}) {
  return
}

// -----

// Test 10: Multiple input and output arguments - should pass
func.func @arg_abi_multiple_args(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<5xf32>>},
    %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>},
    %arg3: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<5xf32>>})
      attributes {
        executor.func_abi = (memref<10xi32>, memref<5xf32>) -> (memref<10xi32>, memref<5xf32>)
      } {
  return
}

// -----

// Test 11: abi.recv with correct byval attribute - should pass
func.func @abi_recv_correct(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  %0 = executor.abi.recv %arg0 : memref<10xi32>
  return
}

// -----

// Test 12: abi.send with correct byref attribute - should pass
func.func @abi_send_correct(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %alloc = memref.alloc(%c10) : memref<?xi32>
  %cast = memref.cast %alloc : memref<?xi32> to memref<10xi32>
  executor.abi.send %cast to %arg1 : memref<10xi32>
  return
}

// -----

// Test 13: abi.recv with mismatched type - should fail
func.func @abi_recv_type_mismatch(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  // expected-error @below {{result type 'memref<10xf32>' must match ABI value type 'memref<10xi32>'}}
  %0 = executor.abi.recv %arg0 : memref<10xf32>
  return
}

// -----

// Test 14: abi.recv with byref instead of byval - should fail
func.func @abi_recv_wrong_abi(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  return
}

// -----

// Test 15: abi.send with mismatched type - should fail
func.func @abi_send_type_mismatch(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %alloc = memref.alloc(%c10) : memref<?xf32>
  %cast = memref.cast %alloc : memref<?xf32> to memref<10xf32>
  // expected-error @below {{value type 'memref<10xf32>' must match ABI value type 'memref<10xi32>'}}
  executor.abi.send %cast to %arg1 : memref<10xf32>
  return
}

// -----

// Test 16: abi.send with byval instead of byref - should fail
func.func @abi_send_wrong_abi(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %alloc = memref.alloc(%c10) : memref<?xi32>
  %cast = memref.cast %alloc : memref<?xi32> to memref<10xi32>
  // expected-error @below {{argument must have #executor.arg<byref, ...> ABI}}
  executor.abi.send %cast to %arg0 : memref<10xi32>
  return
}

// -----

// Test 17: abi.recv with non-function-argument pointer - should fail
func.func @abi_recv_non_arg_ptr(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  %c1 = arith.constant 1 : i64
  %ptr = executor.alloca %c1 x memref<10xi32> : (i64) -> !executor.ptr<host>
  // expected-error @below {{ptr operand must be a function argument}}
  %0 = executor.abi.recv %ptr : memref<10xi32>
  return
}

// -----

// Test 18: abi.send with non-function-argument pointer - should fail
func.func @abi_send_non_arg_ptr(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : i64
  %alloc = memref.alloc(%c10) : memref<?xi32>
  %cast = memref.cast %alloc : memref<?xi32> to memref<10xi32>
  %ptr = executor.alloca %c1 x memref<10xi32> : (i64) -> !executor.ptr<host>
  // expected-error @below {{ptr operand must be a function argument}}
  executor.abi.send %cast to %ptr : memref<10xi32>
  return
}

// -----

func.func @abi_recv_no_abi_attr(
    // expected-error @below {{expected executor.abi argument ABI attribute for input argument 0}}
    %arg0: !executor.ptr<host>,
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  %0 = executor.abi.recv %arg0 : memref<10xi32>
  return
}

// -----

// Test 20: abi.send without executor.abi attribute - should fail
func.func @abi_send_no_abi_attr(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    // expected-error @below {{expected executor.abi argument ABI attribute for output argument 0}}
    %arg1: !executor.ptr<host>)
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %alloc = memref.alloc(%c10) : memref<?xi32>
  %cast = memref.cast %alloc : memref<?xi32> to memref<10xi32>
  executor.abi.send %cast to %arg1 : memref<10xi32>
  return
}


// -----

func.func @abi_output_undef(
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>, undef>})
      attributes {
        executor.func_abi = () -> (memref<10xi32>)
      } {
  return
}
