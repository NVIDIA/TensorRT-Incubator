// REQUIRES: system-linux
// REQUIRES: cuda
//
// RUN: rm -rf %t || true
// RUN: mkdir -p %t
// RUN: mlir-tensorrt-opt %s -convert-host-to-emitc -executor-serialize-artifacts="artifacts-directory=%t create-manifest=true" \
// RUN:   -canonicalize -form-expressions | \
// RUN:   mlir-tensorrt-translate -mlir-to-cpp | tee %t/out.cpp | FileCheck %s
// RUN: %host_cxx -fsyntax-only \
// RUN:   -I%mtrt_src_dir/executor/lib/Runtime/StandaloneCPP \
// RUN:   %cuda_toolkit_linux_cxx_flags \
// RUN:   %t/out.cpp
//
// CHECK: #include "MTRTRuntimeCore.h"
// CHECK: #include "MTRTRuntimeCuda.h"
// CHECK-NOT: #include "MTRTRuntimeTensorRT.h"

module {
  func.func @smoke_cuda_device_and_stream() -> i32 {
    %c0 = arith.constant 0 : i32
    %device = cuda.get_program_device %c0 : i32
    %stream = cuda.get_global_stream device(%device)[0]
    cuda.stream.sync %stream : !cuda.stream
    return %c0 : i32
  }
}
