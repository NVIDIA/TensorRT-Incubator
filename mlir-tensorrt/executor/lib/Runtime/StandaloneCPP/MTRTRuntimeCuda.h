//===- MTRTRuntimeCuda.h -----------------------------------------*- C++
//-*-===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
/// \file
/// CUDA wrapper declarations used by generated EmitC C++ host code.
//===----------------------------------------------------------------------===//
#ifndef MTRT_RUNTIME_CUDA_H
#define MTRT_RUNTIME_CUDA_H

#include "MTRTRuntimeCore.h"

#include "cuda.h"

namespace mtrt {

//===----------------------------------------------------------------------===//
// CUDA Wrappers
//===----------------------------------------------------------------------===//

/// Return the current CUDA device.
int32_t cuda_get_current_device();

/// Synchronize a CUDA stream.
void cuda_stream_sync(CUstream stream);

/// Perform a CUDA allocation.
void *cuda_alloc(CUstream stream, int64_t size, bool isPinned, bool isManaged);

void cuda_free(CUstream stream, void *ptr, int8_t isHostPinned,
               int8_t isManaged);

void cuda_copy(CUstream stream, void *src, void *dest, int64_t sizeBytes);

CUmodule cuda_module_create_from_ptx_file(const char *filename);

void cuda_module_destroy(CUmodule module);

CUfunction cuda_module_get_func(CUmodule module, const char *name);

/// Push arguments into the array of pointers-to-arguments that will be given to
/// a CUDA kernel launch.
/// Arguments will be pushed into 'array' starting at 0-th offset and the
/// pointer to the ending position is returned.
[[deprecated("EmitC lowering no longer uses this helper; build a local argv[] "
             "of addresses and pass it directly to cuda_launch_kernel.")]]
inline void **cuda_launch_args_push(void **array,
                                    const RankedMemRef<0> &memref) {
  array[0] = const_cast<void **>(&memref.aligned);
  return array + 1;
}

/// Launch a simple CUDA kernel.
void cuda_launch_kernel(CUfunction func, int32_t gridX, int32_t gridY,
                        int32_t gridZ, int32_t blockX, int32_t blockY,
                        int32_t blockZ, int32_t dynamicSharedMemoryBytes,
                        CUstream stream, void **arguments);

/// Copy `src` to `dest` assuming that both can be copied using
/// `cudaMemcpyAsync` (e.g. both are device|host_pinned|unified memory spaces)
/// and also assuming strides are identity layout (canonical generalized
/// row-major packed layout, strides = suffix_product(shape)).
void cuda_copy_using_descriptor(CUstream stream, void *src,
                                UnrankedMemRef srcDesc, void *dest,
                                UnrankedMemRef destDesc);

} // namespace mtrt

#endif // MTRT_RUNTIME_CUDA_H
