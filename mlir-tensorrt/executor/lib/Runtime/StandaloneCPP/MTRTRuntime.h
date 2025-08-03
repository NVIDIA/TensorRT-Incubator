//===- MTRTRuntime.h --------------------------------------------*- C++ -*-===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// This file contains an example implementation of C++ functions required
/// to interact with generated C++ host code.
///
//===----------------------------------------------------------------------===//
#ifndef MTRTRUNTIME
#define MTRTRUNTIME

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <NvInferRuntime.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "cuda.h"
#include "cuda_runtime_api.h"
#include <array>
#include <cstdint>

namespace mtrt {

//===----------------------------------------------------------------------===//
// MemRef
//===----------------------------------------------------------------------===//

/// A RankedMemRef descriptor is a struct containing information about
/// a buffer which has a logical N-d shape.
template <unsigned Rank>
struct RankedMemRef {
  void *allocated;
  void *aligned;
  /// Offset from 'aligned' to start of buffer (in units of 'elements', not
  /// bytes).
  int64_t offset;
  // The first 'Rank' elements are the shape, the last 'Rank' elements are the
  // strides. All sizes are in units of 'elements', not bytes.
  std::array<int64_t, Rank * 2> shapeAndStrides;
};

/// Explicit specialization for 0-rank memrefs.
template <>
struct RankedMemRef<0> {
  void *allocated;
  void *aligned;
  int64_t offset;
};

/// A descriptor containing a pointer directly pointing to the start of the
/// buffer as well as a shape. The layout is assumed to be canonical (strides =
/// suffix_product(shape)). The compiler enforces this condition on I/O
/// boundaries of TensorRT executions, so this descriptor is used to communicate
/// arguments to TRT execution so that the compiler can output how to calculate
/// buffer start explicitly.
template <unsigned Rank>
struct PtrAndShape {
  void *bufferStart;
  std::array<int64_t, Rank> shape;
};

/// Explicit specialization for 0-rank memrefs.
template <>
struct PtrAndShape<0> {
  void *bufferStart;
};

/// An UnrankedMemRef descriptor constains a rank integer and a pointer to the
/// ranked descriptor. The ranked descriptor may be `PtrWithShape` or
/// `RankedMemRef`, use depends on context, compiler enforces the contracts
/// where this is used.
struct UnrankedMemRef {
  int64_t rank;
  const void *rankedDescriptor;
};

/// Construct an unranked descriptor from a ranked descriptor.
template <unsigned Rank>
UnrankedMemRef
make_unranked_descriptor(int64_t rank,
                         const RankedMemRef<Rank> &rankedDescriptor) {
  return UnrankedMemRef{rank, static_cast<const void *>(&rankedDescriptor)};
}

/// Construct an unranked descriptor from a ranked descriptor.
template <unsigned Rank>
UnrankedMemRef
make_unranked_descriptor(int64_t rank,
                         const PtrAndShape<Rank> &rankedDescriptor) {
  return UnrankedMemRef{rank, static_cast<const void *>(&rankedDescriptor)};
}

/// Construct a ranked memref descriptor.
template <unsigned Rank, typename... Args>
auto make_memref_descriptor(void *allocated, void *aligned, int64_t offset,
                            Args &&...args) {
  return RankedMemRef<Rank>{
      allocated, aligned, offset,
      std::array<int64_t, Rank * 2>{std::forward<Args>(args)...}};
}

/// Construct a ranked memref descriptor.
template <>
inline auto make_memref_descriptor<0>(void *allocated, void *aligned,
                                      int64_t offset) {
  return RankedMemRef<0>{allocated, aligned, offset};
}

/// Construct a ranked memref descriptor.
template <unsigned Rank, typename... Args>
auto make_ptr_shape_descriptor(void *start, Args &&...args) {
  return PtrAndShape<Rank>{
      start, std::array<int64_t, Rank>{std::forward<Args>(args)...}};
}

template <>
inline auto make_ptr_shape_descriptor<0>(void *start) {
  return PtrAndShape<0>{start};
}

/// Access the allocated pointer field.
template <typename T>
void *memref_descriptor_get_allocated_ptr(const T &memref) {
  return memref.allocated;
}

/// Access the aligned pointer field.
template <typename T>
void *memref_descriptor_get_aligned_ptr(const T &memref) {
  return memref.aligned;
}

/// Access the element offset field.
template <typename T>
int64_t memref_descriptor_get_offset(const T &memref) {
  return memref.offset;
}

/// Access the shape field.
template <typename T>
int64_t memref_descriptor_get_dim_size(const T &memref, int32_t dim) {
  return memref.shapeAndStrides[dim];
}

/// Access the stride field.
template <unsigned Rank>
int64_t memref_descriptor_get_stride(const RankedMemRef<Rank> &memref,
                                     int32_t dim) {
  return memref.shapeAndStrides[dim + Rank];
}

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

//===----------------------------------------------------------------------===//
// TensorRT Wrappers
//===----------------------------------------------------------------------===//

/// Enqueue the execution of the TRT function with the given inputs/outputs onto
/// a stream.
/// All UnrankedMemRefs here contain pointers to descriptors of 'PtrAndShape'
/// type.
void tensorrt_enqueue(nvinfer1::IExecutionContext *context, CUstream stream,
                      int32_t numInputs, UnrankedMemRef *inputs,
                      int32_t numOutputs, UnrankedMemRef *outputs);

/// Load a TensorRT engine from a serialized plan file.
nvinfer1::ICudaEngine *
tensorrt_engine_create_from_file(nvinfer1::IRuntime *runtime,
                                 const char *filename);

/// Destroy a TensorRT engine.
void tensorrt_engine_destroy(nvinfer1::ICudaEngine *engine);

/// Construct an execution context.
nvinfer1::IExecutionContext *
tensorrt_execution_context_create(nvinfer1::ICudaEngine *engine);

/// Destroy an execution context.
void tensorrt_execution_context_destroy(nvinfer1::IExecutionContext *engine);

//===----------------------------------------------------------------------===//
// Host Memory Management
//===----------------------------------------------------------------------===//

/// Return an aligned allocation.
void *host_alloc(int64_t size, int32_t alignment);

/// Free an allocation from `host_alloc`.
void host_free(void *ptr);

void *constant_load_from_file(const char *filename, int32_t align,
                              int32_t space);

void constant_destroy(void *data, int32_t space);

} // namespace mtrt

#endif // MTRTRUNTIME
