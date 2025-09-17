//===- CommonRuntime.cpp  -------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
#include "mlir-executor/Runtime/Backend/Common/CommonRuntime.h"

using namespace mlirtrt;
namespace mrt = mlirtrt::runtime;
using namespace mrt;

void mrt::executeStridedCopy(
    int64_t elemSize, uintptr_t src, int64_t srcOffset,
    const std::vector<int64_t> &srcShape, std::vector<int64_t> &srcStrides,
    uintptr_t dst, int64_t dstOffset, const std::vector<int64_t> &dstShape,
    std::vector<int64_t> &dstStrides,
    std::function<void(void *dst, void *src, size_t size)> memcpyFunc) {
  const int64_t rank = srcShape.size();
  // Handle edge case of empty source tensor.
  if (rank > 0 && llvm::find(srcShape, 0) != srcShape.end())
    return;

  char *srcPtr = reinterpret_cast<char *>(src + srcOffset * elemSize);
  char *dstPtr = reinterpret_cast<char *>(dst + dstOffset * elemSize);

  // Handle edge case of rank-0 tensor.
  if (rank == 0) {
    memcpyFunc(dstPtr, srcPtr, elemSize);
    return;
  }

  std::vector<int64_t> indices(rank, 0);

  // Initialize index and scale strides.
  for (int64_t i = 0; i < rank; ++i) {
    srcStrides[i] *= elemSize;
    dstStrides[i] *= elemSize;
  }

  // The logic below comes from the original MLIR CRunner implementation of
  // strided memref copy. It could be further optimized if any of the
  // dimensions have unit stride.
  int64_t readIndex = 0, writeIndex = 0;
  while (true) {
    // Copy one element.
    memcpyFunc(dstPtr + writeIndex, srcPtr + readIndex, elemSize);
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      int64_t newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += dstStrides[axis];
      if (srcShape[axis] != newIndex)
        break;
      if (axis == 0)
        return;
      indices[axis] = 0;
      readIndex -= srcShape[axis] * srcStrides[axis];
      writeIndex -= dstShape[axis] * dstStrides[axis];
    }
  }
}

void mrt::executeStridedByteCopy(
    uintptr_t src, int64_t srcOffsetBytes, const std::vector<int64_t> &srcShape,
    const std::vector<int64_t> &srcByteStrides, uintptr_t dst,
    int64_t dstOffsetBytes, const std::vector<int64_t> &dstShape,
    const std::vector<int64_t> &dstByteStrides, size_t elemSizeBytes,
    std::function<void(void *dst, void *src, size_t size)> memcpyFunc) {
  const int64_t rank = srcShape.size();

  // Handle edge case of empty source tensor.
  if (rank > 0 && llvm::find(srcShape, 0) != srcShape.end())
    return;

  char *srcPtr = reinterpret_cast<char *>(src + srcOffsetBytes);
  char *dstPtr = reinterpret_cast<char *>(dst + dstOffsetBytes);

  // Handle edge case of rank-0 tensor.
  if (rank == 0) {
    memcpyFunc(dstPtr, srcPtr, elemSizeBytes);
    return;
  }

  std::vector<int64_t> indices(rank, 0);

  // The logic below comes from the original MLIR CRunner implementation of
  // strided memref copy. It could be further optimized if any of the
  // dimensions have unit stride.
  int64_t readIndex = 0, writeIndex = 0;
  while (true) {
    // Copy one element.
    memcpyFunc(dstPtr + writeIndex, srcPtr + readIndex, elemSizeBytes);
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      int64_t newIndex = ++indices[axis];
      readIndex += srcByteStrides[axis];
      writeIndex += dstByteStrides[axis];
      if (srcShape[axis] != newIndex)
        break;
      if (axis == 0)
        return;
      indices[axis] = 0;
      readIndex -= srcShape[axis] * srcByteStrides[axis];
      writeIndex -= dstShape[axis] * dstByteStrides[axis];
    }
  }
}
