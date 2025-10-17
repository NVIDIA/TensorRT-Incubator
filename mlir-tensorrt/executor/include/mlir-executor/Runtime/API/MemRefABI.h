//===- MemRefABI.h ----------------------------------------------*- C++ -*-===//
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
///
/// This file provides runtime data structures that are compatible with the
/// MLIR C runtime ABI.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_RUNTIME_API_MEMREFABI
#define MLIR_EXECUTOR_RUNTIME_API_MEMREFABI

#include "mlir-executor/Runtime/API/API.h"
#include <cstdint>

namespace mtrt {

template <unsigned Rank>
struct MemRefDescriptor;

/// A descriptor for a 0-dimensional memref.
template <>
struct MemRefDescriptor<0> {
  uintptr_t ptr;
  uintptr_t aligned;
  int64_t offset;
};

/// A descriptor for a N-dimensional memref (N>0).
template <unsigned Rank>
struct MemRefDescriptor {
  uintptr_t ptr;
  uintptr_t aligned;
  int64_t offset;
  int64_t shape[Rank];
  int64_t strides[Rank];
};

/// A MemRefDescriptorView is a copy of the information in a MemRefDescriptor,
/// except that the shape/strides refer to the descriptor arrays and the
/// rank is explicitly specified. This should only be used as a temporary
/// method to access a descriptor that is provided as an input argument.
struct MemRefDescriptorView {
  int64_t rank;
  uintptr_t basePtr;
  uintptr_t data;
  int64_t offset;
  const int64_t *shape;
  const int64_t *strides;
};

/// Print a human readable description of the memref descriptor view to the
/// stream.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const MemRefDescriptorView &desc);

/// An unranked memref descriptor contains a rank and a pointer to a ranked
/// memref descriptor.
struct UnrankedMemRefDescriptor {
  int64_t rank;
  uintptr_t rankedDescriptorPtr;
};

/// Print a human readable description of the unranked memref descriptor to the
/// stream.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const UnrankedMemRefDescriptor &desc);

/// Populate the memref descriptor referred to by `desc` with the given
/// information. This version assumes that the aligned data pointer is the same
/// as the base pointer.
template <unsigned Rank>
void populateMemRefDescriptor(MemRefDescriptor<Rank> *desc, uintptr_t allocPtr,
                              uintptr_t alignedPtr, int64_t offset,
                              llvm::ArrayRef<int64_t> shape,
                              llvm::ArrayRef<int64_t> strides) {
  assert(shape.size() == Rank && "rank mismatch");
  assert(strides.size() == Rank && "rank mismatch");
  desc->ptr = allocPtr;
  desc->aligned = alignedPtr;
  desc->offset = offset;
  if constexpr (Rank > 0) {
    for (unsigned i = 0; i < Rank; ++i) {
      desc->shape[i] = shape[i];
      desc->strides[i] = strides[i];
    }
  }
}

/// Populate the memref descriptor referred to by `desc` with the information
/// from `memref`.
template <unsigned Rank>
void populateMemRefDescriptor(MemRefDescriptor<Rank> *desc,
                              const MemRefValue &memref) {
  populateMemRefDescriptor(desc, memref.getMemory(), memref.getMemory(),
                           memref.getLayout().getOffset(), memref.getShape(),
                           memref.getStrides());
}

/// Populate the memref descriptor referred to by `desc` with the information
/// from `memref`.
Status populateMemRefDescriptor(UnrankedMemRefDescriptor desc,
                                const MemRefValue &memref);

/// Populate the memref descriptor referred to by `desc` with the information
/// from `allocPtr`, `alignedPtr`, `offset`, `shape`, and `strides`.
Status populateMemRefDescriptor(UnrankedMemRefDescriptor desc,
                                uintptr_t allocPtr, uintptr_t alignedPtr,
                                int64_t offset, llvm::ArrayRef<int64_t> shape,
                                llvm::ArrayRef<int64_t> strides);

/// Get the memref descriptor info referred to by `desc`.
StatusOr<MemRefDescriptorView>
getMemRefDescriptorInfo(UnrankedMemRefDescriptor desc);

} // namespace mtrt

#endif // MLIR_EXECUTOR_RUNTIME_API_MEMREFABI
