// RUN: executor-opt %s --split-input-file -verify-diagnostics


func.func @valid_identity_complex(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = executor.buffer_bitcast %arg0 : tensor<4xcomplex<f32>> to tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}

// -----

func.func @valid_table_identity(%arg0: tensor<4x!executor.table<i32, f32>>) -> tensor<4x!executor.table<i32, f32>> {
  %0 = executor.buffer_bitcast %arg0 : tensor<4x!executor.table<i32, f32>> to tensor<4x!executor.table<i32, f32>>
  return %0 : tensor<4x!executor.table<i32, f32>>
}

// -----

func.func @invalid_bit_width_i32_i64(%arg0: tensor<4xi32>) -> tensor<4xi64> {
  // expected-error @below {{'executor.buffer_bitcast' op operand type 'tensor<4xi32>' and result type 'tensor<4xi64>' are cast incompatible}}
  %0 = executor.buffer_bitcast %arg0 : tensor<4xi32> to tensor<4xi64>
  return %0 : tensor<4xi64>
}

// -----

func.func @invalid_bit_width_f32_f64(%arg0: memref<4xf32>) -> memref<4xf64> {
  // expected-error @below {{'executor.buffer_bitcast' op operand type 'memref<4xf32>' and result type 'memref<4xf64>' are cast incompatible}}
  %0 = executor.buffer_bitcast %arg0 : memref<4xf32> to memref<4xf64>
  return %0 : memref<4xf64>
}

// -----

func.func @invalid_bit_width_i16_i32(%arg0: tensor<8xi16>) -> tensor<8xi32> {
  // expected-error @below {{'executor.buffer_bitcast' op operand type 'tensor<8xi16>' and result type 'tensor<8xi32>' are cast incompatible}}
  %0 = executor.buffer_bitcast %arg0 : tensor<8xi16> to tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

func.func @incompatible_shape(%arg0: tensor<4xi32>) -> tensor<8xi32> {
  // expected-error @below {{'executor.buffer_bitcast' op operand type 'tensor<4xi32>' and result type 'tensor<8xi32>' are cast incompatible}}
  %0 = executor.buffer_bitcast %arg0 : tensor<4xi32> to tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

func.func @incompatible_rank(%arg0: tensor<4xi32>) -> tensor<2x2xi32> {
  // expected-error @below {{'executor.buffer_bitcast' op operand type 'tensor<4xi32>' and result type 'tensor<2x2xi32>' are cast incompatible}}
  %0 = executor.buffer_bitcast %arg0 : tensor<4xi32> to tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

func.func @incompatible_memref_layout(%arg0: memref<4x4xi32>) -> memref<4x4xi32, strided<[8, 1], offset: 0>> {
  // expected-error @below {{'executor.buffer_bitcast' op operand type 'memref<4x4xi32>' and result type 'memref<4x4xi32, strided<[8, 1]>>' are cast incompatible}}
  %0 = executor.buffer_bitcast %arg0 : memref<4x4xi32> to memref<4x4xi32, strided<[8, 1], offset: 0>>
  return %0 : memref<4x4xi32, strided<[8, 1], offset: 0>>
}

// -----

func.func @valid_same_type(%arg0: tensor<4xi32>) -> tensor<4xf32> {
  %0 = executor.buffer_bitcast %arg0 : tensor<4xi32> to tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @valid_memref_same_type(%arg0: memref<4x4xf32>) -> memref<4x4xi32> {
  %0 = executor.buffer_bitcast %arg0 : memref<4x4xf32> to memref<4x4xi32>
  return %0 : memref<4x4xi32>
}

// -----

func.func @valid_complex_to_float_tensor(%arg0: tensor<4x5xcomplex<f32>>) -> tensor<4x5xi64> {
  %0 = executor.buffer_bitcast %arg0 : tensor<4x5xcomplex<f32>> to tensor<4x5xi64>
  return %0 : tensor<4x5xi64>
}

// -----

func.func @valid_float_to_complex_memref(%arg0: memref<4x5xi64>) -> memref<4x5xcomplex<f32>> {
  %0 = executor.buffer_bitcast %arg0 : memref<4x5xi64> to memref<4x5xcomplex<f32>>
  return %0 : memref<4x5xcomplex<f32>>
}

// -----

func.func @invalid_complex_to_float_wrong_dim(%arg0: tensor<4x5xcomplex<f32>>) -> tensor<4x5xi32> {
  // expected-error @below {{'executor.buffer_bitcast' op operand type 'tensor<4x5xcomplex<f32>>' and result type 'tensor<4x5xi32>' are cast incompatible}}
  %0 = executor.buffer_bitcast %arg0 : tensor<4x5xcomplex<f32>> to tensor<4x5xi32>
  return %0 : tensor<4x5xi32>
}
