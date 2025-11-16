// RUN: executor-opt %s -split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: func @buffer_bitcast_canonicalize
func.func @buffer_bitcast_canonicalize(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = executor.buffer_bitcast %arg0 : tensor<4xi32> to tensor<4xi32>
  // CHECK-NEXT: return
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func @buffer_bitcast_canonicalize_2
func.func @buffer_bitcast_canonicalize_2(%arg0: memref<4xi32>) -> memref<4xi32> {
  %0 = executor.buffer_bitcast %arg0 : memref<4xi32> to memref<4xf32>
  %2 = executor.buffer_bitcast %0 : memref<4xf32> to memref<4xi32>
  // CHECK-NEXT: return
  return %2 : memref<4xi32>
}
