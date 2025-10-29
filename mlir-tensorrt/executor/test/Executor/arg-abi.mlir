// RUN: executor-opt %s -split-input-file -verify-diagnostics -split-input-file

// Input argument with byval and scalar type (i32) - should fail
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

// Input argument with byval and scalar float type - should fail
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

// Input argument with byval and index type - should fail
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

// Input argument with byval and memref type - should pass
func.func @arg_abi_input_byval_memref(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  return
}

// -----

// Input argument with byref instead of byval - should fail
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

// Output argument with byval instead of byref - should fail
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

// Argument with executor.abi but not a host pointer - should fail
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

// Argument with executor.abi but missing executor.func_abi - should fail
// expected-error @below {{expected executor.func_abi attribute to be TypeAttr with a FunctionType attached to the function containing arguments decorated with executor.abi}}
func.func @arg_abi_missing_func_abi(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>}) {
  return
}

// -----

// Multiple input and output arguments - should pass
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

// abi.recv with correct byval attribute - should pass
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

// abi.send with correct byref attribute - should pass
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

// abi.recv with mismatched type - should fail
func.func @abi_recv_type_mismatch(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  // expected-error @below {{result type 'memref<10xf32>' is incompatible with ABI value type 'memref<10xi32>'}}
  %0 = executor.abi.recv %arg0 : memref<10xf32>
  return
}

// -----

// abi.recv with byref instead of byval - should fail
func.func @abi_recv_wrong_abi(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  return
}

// -----

// abi.send with mismatched type - should fail
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
  // expected-error @below {{value type 'memref<10xf32>' is incompatible with ABI value type 'memref<10xi32>'}}
  executor.abi.send %cast to %arg1 : memref<10xf32>
  return
}

// -----

// abi.send with byval instead of byref - should fail
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

// abi.recv with non-function-argument pointer - should fail
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

// abi.send with non-function-argument pointer - should fail
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

// abi.send without executor.abi attribute - should fail
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

// -----

// Signed integer in output func_abi with signless integer in abi.send - should pass
func.func @abi_send_signed_int_compat(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, si32>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (si32)
      } {
  %c42 = arith.constant 42 : i32
  executor.abi.send %c42 to %arg1 : i32
  return
}

// -----

// Unsigned integer in output func_abi with signless integer in abi.send - should pass
func.func @abi_send_unsigned_int_compat(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, ui64>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (ui64)
      } {
  %c100 = arith.constant 100 : i64
  executor.abi.send %c100 to %arg1 : i64
  return
}

// -----

// Index type in output func_abi with i32 in abi.send - should pass
func.func @abi_send_index_i32_compat(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, index>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (index)
      } {
  %c10 = arith.constant 10 : i32
  executor.abi.send %c10 to %arg1 : i32
  return
}

// -----

// Index type in output func_abi with i64 in abi.send - should pass
func.func @abi_send_index_i64_compat(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, index>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (index)
      } {
  %c10 = arith.constant 10 : i64
  executor.abi.send %c10 to %arg1 : i64
  return
}

// -----

// Index type in output func_abi with incompatible i16 in abi.send - should fail
func.func @abi_send_index_i16_incompat(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, index>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (index)
      } {
  %c10 = arith.constant 10 : i16
  // expected-error @below {{value type 'i16' is incompatible with ABI value type 'index'}}
  executor.abi.send %c10 to %arg1 : i16
  return
}

// -----

// Signed integer i8 in output func_abi with signless i8 in abi.send - should pass
func.func @abi_send_si8_compat(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, si8>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (si8)
      } {
  %c5 = arith.constant 5 : i8
  executor.abi.send %c5 to %arg1 : i8
  return
}

// -----

// Unsigned integer ui8 in output func_abi with signless i8 in abi.send - should pass
func.func @abi_send_ui8_compat(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, ui8>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (ui8)
      } {
  %c42 = arith.constant 42 : i8
  executor.abi.send %c42 to %arg1 : i8
  return
}

// -----

// Complex scalar (complex<f32>) in output func_abi with abi.send - should pass
func.func @abi_send_complex_scalar(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, complex<f32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (complex<f32>)
      } {
  %c1 = arith.constant 1.0 : f32
  %complex = complex.create %c1, %c1 : complex<f32>
  executor.abi.send %complex to %arg1 : complex<f32>
  return
}

// -----

// Complex memref in input and output func_abi - should pass
func.func @abi_complex_memref(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<5xcomplex<f32>>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<5xcomplex<f32>>>})
      attributes {
        executor.func_abi = (memref<5xcomplex<f32>>) -> (memref<5xcomplex<f32>>)
      } {
  %0 = executor.abi.recv %arg0 : memref<5xcomplex<f32>>
  executor.abi.send %0 to %arg1 : memref<5xcomplex<f32>>
  return
}


// -----

func.func @f4_scalar_compat(
    %arg0: i8,
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, f4E2M1FN>}) attributes {
      executor.func_abi = (f4E2M1FN) -> (f4E2M1FN)
    } {
  executor.abi.send %arg0 to %arg1 : i8
  return
}

// -----

func.func @f4_scalar_incompat(
    %arg0: i4,
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, f4E2M1FN>}) attributes {
      executor.func_abi = (f4E2M1FN) -> (f4E2M1FN)
    } {
  // expected-error @below {{'executor.abi.send' op value type 'i4' is incompatible with ABI value type 'f4E2M1FN'}}
  executor.abi.send %arg0 to %arg1 : i4
  return
}

// -----

func.func @f4_memref_compat(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi8>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf4E2M1FN>>})
    attributes {
      executor.func_abi = (memref<10xf4E2M1FN>) -> (memref<10xf4E2M1FN>)
    } {
  %0 = executor.abi.recv %arg0 : memref<10xi8>
  executor.abi.send %0 to %arg1 : memref<10xi8>
  return
}

// -----

func.func @f4_memref_incompat(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi4>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf4E2M1FN>>})
    attributes {
      executor.func_abi = (memref<10xf4E2M1FN>) -> (memref<10xf4E2M1FN>)
    } {
  %0 = executor.abi.recv %arg0 : memref<10xi4>
  // expected-error @below {{'executor.abi.send' op value type 'memref<10xi4>' is incompatible with ABI value type 'memref<10xf4E2M1FN>'}}
  executor.abi.send %0 to %arg1 : memref<10xi4>
  return
}
