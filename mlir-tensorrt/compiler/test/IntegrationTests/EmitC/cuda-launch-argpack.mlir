// REQUIRES: host-has-at-least-1-gpus
// REQUIRES: cuda
// REQUIRES: system-linux
//
// RUN: rm -rf %t || true
// RUN: mkdir -p %t
// RUN: cp %S/cuda-launch-kernel.ptx $PWD/kernels.ptx
// RUN: mlir-tensorrt-opt %s -lower-affine -convert-host-to-emitc -cse -canonicalize -form-expressions \
// RUN:   -executor-serialize-artifacts="artifacts-directory=%t create-manifest=true" \
// RUN:   --mlir-print-op-generic | \
// RUN: mlir-tensorrt-translate -mlir-to-cpp | tee %t/cuda-launch-argpack.cpp
// RUN: %host_cxx \
// RUN:   %t/cuda-launch-argpack.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeStatus.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeCore.cpp \
// RUN:   %mtrt_src_dir/executor/lib/Runtime/StandaloneCPP/MTRTRuntimeCuda.cpp \
// RUN:   %S/cuda-launch-argpack_driver.cpp \
// RUN:  -I%mtrt_src_dir/executor/lib/Runtime/StandaloneCPP \
// RUN:  %cuda_toolkit_linux_cxx_flags \
// RUN:  -o cuda-launch-argpack-test
// RUN: env MTRT_ARTIFACTS_DIR=%t ./cuda-launch-argpack-test | FileCheck %s
//
// This test validates the EmitC lowering for `cuda.launch` argument packing:
// the generated C++ must materialize locals for each parameter and pass a
// `void* argv[]` array to the CUDA driver launch API.

#device = #plan.memory_space<device>
#host = #plan.memory_space<host_pinned>

// A tiny PTX kernel:
//   out[offset + idx] = in[offset + idx] + scalar  (for idx < n)
//
// Parameters:
//   in_ptr  : u64 (pointer)
//   out_ptr : u64 (pointer)
//   offset  : u64 (element offset)
//   n       : u32
//   scalar  : f32
//
// NOTE: vector length must match the string length.
cuda.compiled_module @kernels file "kernels.ptx"

func.func @run() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_index = arith.constant 0 : index
  %c1_index = arith.constant 1 : index
  %c16_index = arith.constant 16 : index
  %c4_index = arith.constant 4 : index
  %c12_index = arith.constant 12 : index
  %c8_index = arith.constant 8 : index
  %c8_i32 = arith.constant 8 : i32
  %scalar = arith.constant 2.5 : f32

  %device = cuda.get_active_device
  %stream = cuda.get_global_stream device(%device)[0]

  %func = cuda.get_function "add_kernel" from @kernels

  // Allocate device buffers.
  %in_dev = cuda.alloc() stream(%stream) : memref<16xf32, #device>
  %out_dev = cuda.alloc() stream(%stream) : memref<16xf32, #device>

  // Fill input on host and copy to device.
  %in_host = cuda.alloc() : memref<16xf32, #host>
  %out_host = cuda.alloc() : memref<16xf32, #host>
  scf.for %i = %c0_index to %c16_index step %c1_index {
    %fi = arith.index_cast %i : index to i32
    %ff = arith.sitofp %fi : i32 to f32
    memref.store %ff, %in_host[%i] : memref<16xf32, #host>
    memref.store %ff, %out_host[%i] : memref<16xf32, #host>
  }
  cuda.copy_h2d stream(%stream) %in_host, %in_dev : memref<16xf32, #host> to memref<16xf32, #device>
  cuda.copy_h2d stream(%stream) %out_host, %out_dev : memref<16xf32, #host> to memref<16xf32, #device>

  // Flattened launch ABI: base pointers + explicit scalar offset.
  // We avoid `memref.subview` here on purpose: the EmitC host backend does not
  // currently lower `memref.subview`, and leaving it around would force
  // unrealized casts that carry `#plan.memory_space` types into translation.
  %in_base, %in_off0, %in_sz, %in_stride =
      memref.extract_strided_metadata %in_dev
      : memref<16xf32, #device>
      -> memref<f32, #device>, index, index, index
  %out_base, %out_off0, %out_sz, %out_stride =
      memref.extract_strided_metadata %out_dev
      : memref<16xf32, #device>
      -> memref<f32, #device>, index, index, index

  // Launch: out[offset+idx] = in[offset+idx] + scalar  (idx < n)
  // Use grid=(1,1,1) block=(32,1,1) so threads 0..7 run.
  %c32_i32 = arith.constant 32 : i32
  cuda.launch %func(%in_base, %out_base, %c4_index, %c8_i32, %scalar
      : memref<f32, #device>, memref<f32, #device>, index, i32, f32) with
    grid(%c1_i32, %c1_i32, %c1_i32)
    block(%c32_i32, %c1_i32, %c1_i32)
    smem(%c0_i32) stream(%stream)

  cuda.copy_d2h stream(%stream) %out_dev, %out_host : memref<16xf32, #device> to memref<16xf32, #host>
  cuda.stream.sync %stream : !cuda.stream

  // Print a few results.
  scf.for %i = %c0_index to %c16_index step %c1_index {
    %v = memref.load %out_host[%i] : memref<16xf32, #host>
    executor.print "out[%lu] = %f"(%i, %v : index, f32)
  }

  return %c0_i32 : i32
}

// Since offset=4, the first 4 elements are not modified by the kernel.
// CHCECK: out[0] = 0.000000
// CHCECK: out[1] = 1.000000
// CHCECK: out[2] = 2.000000
// CHCECK: out[3] = 3.000000

// Since n=8, the next 8 elements are modified by the kernel.
// CHECK: out[4] = 6.500000
// CHECK: out[5] = 7.500000
// CHECK: out[6] = 8.500000
// CHECK: out[7] = 9.500000
// CHECK: out[8] = 10.500000
// CHECK: out[9] = 11.500000
// CHECK: out[10] = 12.500000
// CHECK: out[11] = 13.500000

// The last 4 elements are not modified by the kernel.
// CHECK: out[12] = 12.000000
// CHECK: out[13] = 13.000000
// CHECK: out[14] = 14.000000
// CHECK: out[15] = 15.000000
