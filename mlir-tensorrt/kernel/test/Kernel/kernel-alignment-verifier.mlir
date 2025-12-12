// RUN: kernel-opt -allow-unregistered-dialect -split-input-file --verify-diagnostics %s

gpu.module @kernels {
  func.func @main(%arg0: memref<1xf32>, %arg1: memref<1xf32>) -> memref<1xf32> {
    return %arg0 : memref<1xf32>
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{kernel.alignment must decorate a tensor type, but got 'memref<1xf32>'}}
  func.func @main(%arg0: memref<1xf32> {kernel.alignment = 16 : i64}, %arg1: memref<1xf32>) -> memref<1xf32> {
    return %arg0 : memref<1xf32>
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{kernel.alignment's value should be power of two, but got 11}}
  func.func @main(%arg0: memref<1xf32>, %arg1: tensor<1xf32> {kernel.alignment = 11 : i64}) -> memref<1xf32> {
    return %arg0 : memref<1xf32>
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{kernel.alignment's value should be power of two, but got 12}}
  func.func @main(%arg0: tensor<1xf32> {kernel.other_attr = 1:i32, kernel.alignment = 12 : i64}, %arg1: memref<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{kernel.alignment must decorate a tensor type, but got 'memref<1xf16>'}}
  func.func @main(%arg0: tensor<1xf32>, %arg1: memref<1xf16> {kernel.alignment = 16 : i64}) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }
}

// -----

gpu.module @kernels {
  // expected-error @below {{kernel.alignment's value should have i64 type, but got 'i32'}}
  func.func @main(%arg0: tensor<1xf32> {kernel.alignment = 16 : i64, kernel.other_attr = 1:i32}, %arg1: tensor<1xf32> {kernel.alignment = 16 : i32}) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }
}

// -----

gpu.module @kernels {
  func.func @main(%arg0: tensor<1xf32> {kernel.alignment = 16 : i64}, %arg1: memref<1xf32> {kernel.other_attr = 1:i32}) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }
}

// -----

gpu.module @kernels {
  func.func @kernel(%arg0: tensor<65536xi32> {kernel.alignment = 4 : i64}, %arg1: tensor<65536xi32>) -> tensor<65536xi32> {
    return %arg1 : tensor<65536xi32>
  }
}

// -----

gpu.module @kernels {
  func.func @main(%arg0: tensor<65536xi32> {kernel.alignment = 8 : i64}, %arg1: tensor<65536xi32>) -> tensor<65536xi32> {
    %result = func.call @callee(%arg0, %arg1) : (tensor<65536xi32>, tensor<65536xi32>) -> tensor<65536xi32>
    return %result : tensor<65536xi32>
  }

  // expected-error @below {{kernel.alignment's value should be power of two, but got 3}}
  func.func @callee(%arg0: tensor<65536xi32> {kernel.alignment = 3 : i64}, %arg1: tensor<65536xi32>) -> tensor<65536xi32> {
    return %arg1 : tensor<65536xi32>
  }
}

// -----

gpu.module @kernels {
  func.func @main(%arg0: tensor<65536xi32> {kernel.alignment = 8 : i64}, %arg1: tensor<65536xi32>) -> tensor<65536xi32> {
    %result = func.call @callee(%arg0, %arg1) : (tensor<65536xi32>, tensor<65536xi32>) -> tensor<65536xi32>
    return %result : tensor<65536xi32>
  }

  // expected-error @below {{kernel.alignment's value should have i64 type, but got 'i32'}}
  func.func @main_other(%arg0: tensor<65536xi32>, %arg1: tensor<65536xi32> {kernel.alignment = 16 : i32}) -> tensor<65536xi32> {
    %result = func.call @callee(%arg0, %arg1) : (tensor<65536xi32>, tensor<65536xi32>) -> tensor<65536xi32>
    return %result : tensor<65536xi32>
  }
}

// -----

gpu.module @kernels {
  func.func @main(%arg0: tensor<65536xi32> {kernel.alignment = 8 : i64}, %arg1: tensor<65536xi32>) -> tensor<65536xi32> {
    %result = func.call @callee(%arg0, %arg1) : (tensor<65536xi32>, tensor<65536xi32>) -> tensor<65536xi32>
    return %result : tensor<65536xi32>
  }

  func.func @main_extra(%arg0: tensor<65536xi32> {kernel.alignment = 8 : i64}, %arg1: tensor<65536xi32>) -> tensor<65536xi32> {
    %result = func.call @callee(%arg0, %arg1) : (tensor<65536xi32>, tensor<65536xi32>) -> tensor<65536xi32>
    return %result : tensor<65536xi32>
  }

  // expected-error @below {{kernel.alignment must decorate a tensor type, but got 'memref<1xf32>'}}
  func.func @main_extra(%arg0: memref<1xf32> {kernel.alignment = 16 : i64}, %arg1: memref<1xf32>) -> memref<1xf32> {
    return %arg0 : memref<1xf32>
  }
}

// -----
gpu.module @kernels {
  func.func @main(%arg0: tensor<65536xi32> {kernel.alignment = 8 : i64}, %arg1: tensor<65536xi32>) -> tensor<65536xi32> {
    %result = func.call @callee(%arg0, %arg1) : (tensor<65536xi32>, tensor<65536xi32>) -> tensor<65536xi32>
    return %result : tensor<65536xi32>
  }

  func.func @main_extra(%arg0: tensor<65536xi32> {kernel.alignment = 8 : i64}, %arg1: tensor<65536xi32>) -> tensor<65536xi32> {
    %result = func.call @callee(%arg0, %arg1) : (tensor<65536xi32>, tensor<65536xi32>) -> tensor<65536xi32>
    return %result : tensor<65536xi32>
  }

  func.func @callee(%arg0: tensor<65536xi32> {kernel.alignment = 128 : i64}, %arg1: tensor<65536xi32>) -> tensor<65536xi32> {
    return %arg1 : tensor<65536xi32>
  }
}
