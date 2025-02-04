// RUN: mlir-tensorrt-opt %s -split-input-file -convert-host-to-llvm | FileCheck %s

func.func @func.func(%arg0: memref<1xf32, #plan.memory_space<host>>) {
  return
}

// CHECK-LABEL: llvm.func @func.func
//  CHECK-SAME: (%[[arg0:.+]]: !llvm.ptr, %[[arg1:.+]]: !llvm.ptr, %[[arg2:.+]]: i64, %[[arg3:.+]]: i64, %[[arg4:.+]]: i64) {
//  CHECK-NEXT:     llvm.return
