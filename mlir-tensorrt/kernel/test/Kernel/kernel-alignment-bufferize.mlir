// RUN: kernel-opt %s -test-kernel-one-shot-bufferize -split-input-file | FileCheck %s

gpu.module @kernels_multi {
  func.func @main0(%arg0: tensor<65536xi32> {kernel.attr = 9:i32}, %arg1: tensor<65536xi32> {kernel.attr = 1:i32, kernel.alignment = 1 : i64}) -> tensor<65536xi32> {
    return %arg1 : tensor<65536xi32>
  }

  func.func @main1(%arg0: tensor<65536xi32> {kernel.attr = 9:i32, kernel.alignment = 8 : i64}, %arg1: tensor<65536xi32> {kernel.attr = 1:i32}) -> tensor<65536xi32> {
    return %arg0 : tensor<65536xi32>
  }

  func.func @main2(%arg0: tensor<65536xi32> {kernel.attr = 9:i32, kernel.alignment = 16 : i64}, %arg1: tensor<65536xi32> {kernel.attr = 1:i32, kernel.alignment = 64 : i64}) -> (tensor<65536xi32>, tensor<65536xi32>) {
    return %arg0, %arg1 : tensor<65536xi32>, tensor<65536xi32>
  }
}

// CHECK-LABEL: func.func @main0
//  CHECK-SAME: (%[[arg0:.+]]: memref<65536xi32, strided<[?], offset: ?>> {kernel.attr = 9 : i32}, %[[arg1:.+]]: memref<65536xi32, strided<[?], offset: ?>> {kernel.attr = 1 : i32})
//       CHECK:       memref.assume_alignment %[[arg1]], 1 : memref<65536xi32, strided<[?], offset: ?>>

// CHECK-LABEL: func.func @main1
//  CHECK-SAME: (%[[arg0:.+]]: memref<65536xi32, strided<[?], offset: ?>> {kernel.attr = 9 : i32}, %[[arg1:.+]]: memref<65536xi32, strided<[?], offset: ?>> {kernel.attr = 1 : i32})
//       CHECK:       memref.assume_alignment %[[arg0]], 8 : memref<65536xi32, strided<[?], offset: ?>>

// CHECK-LABEL: func.func @main2
//  CHECK-SAME: (%[[arg0:.+]]: memref<65536xi32, strided<[?], offset: ?>> {kernel.attr = 9 : i32}, %[[arg1:.+]]: memref<65536xi32, strided<[?], offset: ?>> {kernel.attr = 1 : i32})
//       CHECK:       memref.assume_alignment %[[arg0]], 16 : memref<65536xi32, strided<[?], offset: ?>>
//       CHECK:       memref.assume_alignment %[[arg1]], 64 : memref<65536xi32, strided<[?], offset: ?>>
