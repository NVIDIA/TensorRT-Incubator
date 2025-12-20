// RUN: kernel-opt -allow-unregistered-dialect -split-input-file --verify-diagnostics %s

gpu.module @kernels {
  func.func @kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }
}

func.func @caller(%arg0: memref<1xf32>, %arg1: memref<2xf32>, %arg2: index) {
  // expected-error @below {{'kernel.call' op kernel signature '(memref<1xf32>, memref<1xf32>) -> ()' has 2 arguments, but the call operation expects a total of 1 arguments}}
  kernel.call @kernels::@kernel grid[%arg2] block[%arg2] (%arg0) outs() : (memref<1xf32>)->()
  return
}

// -----

gpu.module @kernels {
  func.func @some_kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }
}

func.func @caller(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: index, %arg3: index) {
  // expected-error @below {{'kernel.call' op no valid kernel found with symbol name @kernels::@kernel}}
  kernel.call @kernels::@kernel grid[%arg2] block[%arg3] (%arg0) outs(%arg1) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

gpu.module @kernels {
  func.func @some_kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }
}

func.func @caller(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: index, %arg3: index) {
  // expected-error @below {{custom op 'kernel.call' expected 2 operand types, got 1}}
  kernel.call @kernels::@some_kernel grid[%arg2] block[%arg3] (%arg0) outs(%arg1) : (memref<1xf32>) -> memref<1xf32>
  return
}

// -----

gpu.module @kernels {
  func.func @some_kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: memref<2xf32>) {
    return
  }
}

func.func @caller_too_few_args(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: index) {
  // expected-error @below {{custom op 'kernel.call' expected 2 operand types, got 3}}
  kernel.call @kernels::@some_kernel grid[%arg2] block[%arg2] (%arg0) outs(%arg1) : (memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

gpu.module @kernels {
  func.func @some_kernel(%arg0: memref<1xf32>, %arg1: memref<1xf16>) {
    return
  }
}

func.func @caller_type_mismatch(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: index) {
  // expected-error @below {{'kernel.call' op callee argument #1 of type 'memref<1xf16>' is not compatible with call operand type 'memref<1xf32>'}}
  kernel.call @kernels::@some_kernel grid[%arg2] block[%arg2] (%arg0) outs(%arg1) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

gpu.module @kernels {
  func.func @some_kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }
}

func.func @type_inference(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: index) -> tensor<2xf32> {
  // expected-note @below {{prior use here}}
  %0 = kernel.call @kernels::@some_kernel grid[%arg2] block[%arg2] (%arg0) outs(%arg1) : (tensor<1xf32>, tensor<1xf32>)
    -> tensor<1xf32>
  // expected-error @below {{use of value '%0' expects different type than prior uses: 'tensor<2xf32>' vs 'tensor<1xf32>'}}
  return %0 : tensor<2xf32>
}

// -----

gpu.module @kernels {
  func.func @some_kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }
}

func.func @type_inference(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: index) -> memref<1xf32> {
  // expected-error @below {{'kernel.call' op inferred type(s)  are incompatible with return type(s) of operation 'memref<1xf32>'}}
  // expected-error @below {{'kernel.call' op failed to infer returned types}}
  %0 = kernel.call @kernels::@some_kernel grid[%arg2] block[%arg2] (%arg0) outs(%arg1) : (memref<1xf32>, memref<1xf32>)
    -> memref<1xf32>
  return %0 : memref<1xf32>
}

// -----

gpu.module @kernels {
  func.func @some_kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }
}

func.func @type_inference(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: index) {
  // expected-error @below {{'kernel.call' op block size should have between one and three values, but it has 0 values}}
  kernel.call @kernels::@some_kernel grid[%arg2] block[] (%arg0) outs(%arg1)
    : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

gpu.module @kernels {
  func.func @some_kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }
}

func.func @type_inference(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: index) {
  // expected-error @below {{'kernel.call' op grid size should have between one and three values, but it has 0 values}}
  kernel.call @kernels::@some_kernel grid[] block[%arg2] (%arg0) outs(%arg1)
    : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

gpu.module @kernels {
  func.func @some_kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }
}

func.func @type_inference(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: index) {
  // expected-error @below {{'kernel.call' op grid size should have between one and three values, but it has 4 values}}
  kernel.call @kernels::@some_kernel grid[%arg2, %arg2, %arg2, %arg2] block[%arg2] (%arg0) outs(%arg1)
    : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

gpu.module @kernels {
  func.func @some_kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }
}

func.func @type_inference(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: index) {
  // expected-error @below {{'kernel.call' op block size should have between one and three values, but it has 4 values}}
  kernel.call @kernels::@some_kernel grid[%arg2, %arg2, %arg2] block[%arg2, %arg2, %arg2, %arg2] (%arg0) outs(%arg1)
    : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

func.func @test_associative_even_args(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> (f32, f32) {
  // expected-error @below {{expected an even number of arguments}}
  %0, %1 = kernel.combiner (%arg0, %arg1, %arg2) : f32, f32, f32 {
  ^bb0(%a: f32, %b: f32, %c: f32):
    kernel.yield %a, %b : f32, f32
  }
  return %0, %1 : f32, f32
}

// -----

func.func @test_associative_pair_types(%arg0: f32, %arg1: i32, %arg2: f32, %arg3: f32) -> (f32, f32) {
  // expected-error @below {{expected the types of the last 2 arguments to be the same as the first 2 arguments}}
  %0, %1 = kernel.combiner (%arg0, %arg1, %arg2, %arg3) : f32, i32, f32, f32 {
  ^bb0(%a: f32, %b: i32, %c: f32, %d : f32):
    kernel.yield %a, %b : f32, i32
  }
  return %0, %1 : f32, f32
}

// -----

func.func @test_associative_region_arg(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> (f32, f32) {
  // expected-error @below {{'kernel.combiner' op along control flow edge from parent operands to Region #0: source type #3 'f32' should match input type #3 'i32'}}
  %0, %1 = kernel.combiner (%arg0, %arg1, %arg2, %arg3) : f32, f32, f32, f32 {
  ^bb0(%a: f32, %b: f32, %c: f32, %d: i32):
    kernel.yield %a, %b : f32, f32
  }
  return %0, %1 : f32, f32
}

// -----

func.func @num_threads_verifier_1(%arg0: i32, %arg1: i32) -> i32 {
  // expected-error @below {{'kernel.num_threads' must decorate a function}}
  %0 = arith.addi %arg0, %arg1 {kernel.num_threads = 1} : i32
  return %0 : i32
}

// -----

// expected-error @below {{'kernel.num_threads' must decorate a function nested in a gpu.module}}
func.func @num_threads_verifier_2() attributes {kernel.num_threads = 1} {
  return
}

// -----

gpu.module @kernels {
  // expected-error @below {{'kernel.num_threads' value should be positive integer}}
  func.func @num_threads_verifier_3() attributes {kernel.num_threads = 0} {
    return
  }
}
