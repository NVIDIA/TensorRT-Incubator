//===- MTRTRuntimeCore.h -----------------------------------------*- C++
//-*-===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
/// \file
/// Core types and helpers used by generated EmitC C++ host code.
///
/// This header is intentionally **TensorRT-free** and **CUDA-free** so that
/// non-TRT / non-CUDA EmitC outputs can be compiled in more environments.
//===----------------------------------------------------------------------===//
#ifndef MTRT_RUNTIME_CORE_H
#define MTRT_RUNTIME_CORE_H

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

/// Construct a pointer-and-shape descriptor.
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

#endif // MTRT_RUNTIME_CORE_H
