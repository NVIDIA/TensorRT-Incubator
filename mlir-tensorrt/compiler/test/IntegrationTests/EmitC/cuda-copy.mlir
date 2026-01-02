// REQUIRES: host-has-at-least-1-gpus
// REQUIRES: cuda
// REQUIRES: system-linux

// RUN: rm -rf %t/cpp || true
// RUN: mkdir -p %t/cpp
// RUN: mlir-tensorrt-opt %s -lower-affine -convert-host-to-emitc -cse -canonicalize -form-expressions \
// RUN:   -executor-serialize-artifacts="artifacts-directory=%t/cpp create-manifest=true" \
// RUN:   --mlir-print-op-generic | \
// RUN: mlir-tensorrt-translate -mlir-to-cpp | tee %t/cpp/cuda-copy.cpp
// RUN: %host_cxx \
// RUN:   %t/cpp/cuda-copy.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeStatus.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeCore.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeCuda.cpp \
// RUN:  -I%mtrt_src_dir/executor/lib/Runtime/StandaloneCPP \
// RUN:  %cuda_toolkit_linux_cxx_flags \
// RUN:  -o %t/cpp/cuda-copy-test
// RUN: cd %t/cpp && ./cuda-copy-test | FileCheck %s

#device = #plan.memory_space<device>
#host = #plan.memory_space<host_pinned>

memref.global @gv2 : memref<2x3xf32> = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]>

func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %device = cuda.get_active_device
  %stream = cuda.get_global_stream device(%device)[0]

  %size = arith.constant 16 : i64

  %dev_buffer = cuda.alloc() stream(%stream) : memref<2x3xf32, #device>
  %host_buffer = cuda.alloc() : memref<2x3xf32, #host>

  %src = memref.get_global @gv2 : memref<2x3xf32>
  cuda.copy_h2d stream (%stream) %src, %dev_buffer : memref<2x3xf32> to memref<2x3xf32, #device>
  cuda.copy_d2h stream (%stream) %dev_buffer, %host_buffer : memref<2x3xf32, #device> to memref<2x3xf32, #host>
  cuda.stream.sync %stream : !cuda.stream

  %c0_index = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index

  scf.for %i = %c0_index to %c6 step %c1 {
    %d0 = affine.apply affine_map<(d0)->(d0 floordiv 3)>(%i)
    %d1 = affine.apply affine_map<(d0)->(d0 mod 3)>(%i)
    %result = memref.load %host_buffer[%d0, %d1] : memref<2x3xf32, #host>
    executor.print "pinned host_buffer[%lu, %lu] = %f"(%d0, %d0, %result : index, index, f32)
  }

  return %c0 : i32
}

// CHECK: pinned host_buffer[0, 0] = 0.000000
// CHECK: pinned host_buffer[0, 0] = 1.000000
// CHECK: pinned host_buffer[0, 0] = 2.000000
// CHECK: pinned host_buffer[1, 1] = 3.000000
// CHECK: pinned host_buffer[1, 1] = 4.000000
// CHECK: pinned host_buffer[1, 1] = 5.000000
