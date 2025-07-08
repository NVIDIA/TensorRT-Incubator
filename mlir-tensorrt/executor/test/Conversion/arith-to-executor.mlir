// RUN: executor-opt %s -split-input-file -convert-std-to-executor=index-bitwidth=64  | \
// RUN: FileCheck %s --check-prefixes=IDX64,BOTH
// RUN: executor-opt %s -split-input-file -convert-std-to-executor=index-bitwidth=32  | \
// RUN: FileCheck %s --check-prefixes=IDX32,BOTH

func.func @index_cast_index_to_i32(%arg0: index) -> i32 {
  %0 = arith.index_cast %arg0: index to i32
  return %0 : i32
}

//  IDX32-LABEL: @index_cast_index_to_i32
//   IDX32-SAME: (%[[arg0:.+]]: i32) -> i32
//        IDX32:     return %[[arg0]]

// IDX64-LABEL: @index_cast_index_to_i32
//  IDX64-SAME: (%[[arg0:.+]]: i64)
//       IDX64:     %[[v0:.+]] = executor.trunc %[[arg0]] : i64 to i32
//       IDX64:     return %[[v0]] : i32

// -----

func.func @index_cast_index_to_i64(%arg0: index) -> i64 {
  %0 = arith.index_cast %arg0: index to i64
  return %0 : i64
}

// IDX32-LABEL: @index_cast_index_to_i64
//  IDX32-SAME: (%[[arg0:.+]]: i32) -> i64
//       IDX32:     %[[v0:.+]] = executor.siext %[[arg0]] : i32 to i64
//       IDX32:     return %[[v0]] : i64

// IDX64-LABEL: @index_cast_index_to_i64
//  IDX64-SAME: (%[[arg0:.+]]: i64)
//       IDX64:     return %[[arg0]] : i64

// -----

func.func @index_cast_i32_to_index(%arg0: i32) -> index {
  %0 = arith.index_cast %arg0: i32 to index
  return %0 : index
}

// IDX32-LABEL: @index_cast_i32_to_index
//  IDX32-SAME: (%[[arg0:.+]]: i32) -> i32
//       IDX32:     return %[[arg0]] : i32

// IDX64-LABEL: @index_cast_i32_to_index
//  IDX64-SAME: (%[[arg0:.+]]: i32) -> i64
//       IDX64:     %[[v0:.+]] = executor.siext %[[arg0]] : i32 to i64
//       IDX64:     return %[[v0]] : i64

// -----

func.func @index_cast_i64_to_index(%arg0: i64) -> index {
  %0 = arith.index_cast %arg0: i64 to index
  return %0 : index
}

// IDX32-LABEL: @index_cast_i64_to_index
//  IDX32-SAME: (%[[arg0:.+]]: i64) -> i32
//       IDX32:     %[[v0:.+]] = executor.trunc %[[arg0]]
//       IDX32:     return %[[v0]] : i32

// IDX64-LABEL: @index_cast_i64_to_index
//  IDX64-SAME: (%[[arg0:.+]]: i64)
//       IDX64:     return %[[arg0]] : i64

// -----

func.func @remf(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.remf %arg0, %arg1 : f32
  return %0 : f32
}

// BOTH-LABEL: @remf
//  BOTH-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: f32) -> f32
//       BOTH:     %[[v0:.+]] = executor.remf %[[arg0]], %[[arg1]]
//       BOTH:     return %[[v0]] : f32
