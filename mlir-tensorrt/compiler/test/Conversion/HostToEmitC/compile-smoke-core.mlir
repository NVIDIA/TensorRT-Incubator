// REQUIRES: system-linux
//
// RUN: rm -rf %t || true
// RUN: mkdir -p %t
// RUN: mlir-tensorrt-opt %s -convert-host-to-emitc="artifacts-dir=%t" -canonicalize -form-expressions | \
// RUN:   mlir-tensorrt-translate -mlir-to-cpp | tee %t/out.cpp | FileCheck %s
// RUN: %host_cxx -fsyntax-only -I%mtrt_src_dir/executor/lib/Runtime/StandaloneCPP %t/out.cpp
//
// CHECK: #include "MTRTRuntimeCore.h"
// CHECK-NOT: #include "MTRTRuntimeCuda.h"
// CHECK-NOT: #include "MTRTRuntimeTensorRT.h"

module {
  func.func @smoke_memref_load(%arg0: memref<4xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %v = memref.load %arg0[%c0] : memref<4xi32>
    return %v : i32
  }
}


