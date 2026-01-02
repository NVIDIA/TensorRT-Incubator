//===- MTRTRuntimeCore.h --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
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
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

#include "MTRTRuntimeStatus.h"

namespace mtrt {

//===----------------------------------------------------------------------===//
// Bit Cast (C++17-compatible std::bit_cast alternative)
//===----------------------------------------------------------------------===//

/// C++17-compatible implementation of std::bit_cast (available in C++20).
/// Reinterprets the bits of `src` as type `To`.
template <class To, class From>
To bit_cast(From src) {
  static_assert(sizeof(From) == sizeof(To), "Types must have matching sizes");
  static_assert(std::is_trivially_copyable<From>::value,
                "Source type must be trivially copyable");
  static_assert(std::is_trivially_copyable<To>::value,
                "Destination type must be trivially copyable");

  To dst;
  std::memcpy(std::addressof(dst), std::addressof(src), sizeof(To));
  return dst;
}

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
  /// The logical shape. All sizes are in units of 'elements', not bytes.
  std::array<int64_t, Rank> shape;
  /// The per-dimension strides. All sizes are in units of 'elements', not
  /// bytes.
  std::array<int64_t, Rank> strides;
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

/// A mutable variant of `UnrankedMemRef` for APIs that need to populate ranked
/// descriptors at runtime (e.g. `tensorrt_enqueue_alloc`).
struct UnrankedMemRefMut {
  int64_t rank;
  void *rankedDescriptor;
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

/// Construct a mutable unranked descriptor from a pointer to a ranked
/// descriptor.
inline UnrankedMemRefMut make_unranked_descriptor_mut_ptr(int64_t rank,
                                                          void *rankedDesc) {
  return UnrankedMemRefMut{rank, rankedDesc};
}

/// Construct a ranked memref descriptor.
template <unsigned Rank, typename... Args>
auto make_memref_descriptor(void *allocated, void *aligned, int64_t offset,
                            Args &&...args) {
  static_assert(sizeof...(Args) == Rank * 2,
                "expected 2*Rank arguments: shape..., strides...");
  std::array<int64_t, Rank * 2> shapeAndStrides{
      static_cast<int64_t>(std::forward<Args>(args))...};
  std::array<int64_t, Rank> shape{};
  std::array<int64_t, Rank> strides{};
  for (unsigned i = 0; i < Rank; ++i) {
    shape[i] = shapeAndStrides[i];
    strides[i] = shapeAndStrides[Rank + i];
  }
  return RankedMemRef<Rank>{allocated, aligned, offset, shape, strides};
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
  return memref.shape[dim];
}

/// Access the stride field.
template <unsigned Rank>
int64_t memref_descriptor_get_stride(const RankedMemRef<Rank> &memref,
                                     int32_t dim) {
  return memref.strides[dim];
}

//===----------------------------------------------------------------------===//
// Host Memory Management
//===----------------------------------------------------------------------===//

/// Return an aligned allocation.
/// On success, writes the allocated pointer to `outPtr`.
Status host_aligned_alloc(int64_t sizeBytes, int32_t alignment, void **outPtr);

/// Free an allocation from `host_aligned_alloc`.
void host_free(void *ptr);

/// Load a constant blob from `filename` (searched relative to
/// `MTRT_ARTIFACTS_DIR` and the current working directory), allocating into a
/// memory space implied by `space`. On success, writes the loaded pointer to
/// `outPtr`.
Status constant_load_from_file(const char *filename, int32_t align,
                               int32_t space, void **outPtr);

void constant_destroy(void *data, int32_t space);

} // namespace mtrt

#endif // MTRT_RUNTIME_CORE_H
