//===- CommonRuntime.h ------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2024 NVIDIA CORPORATION & AFFILIATES.
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
/// Internal utilities common to any runtime implementation (not just Lua
/// runtime).
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_RUNTIME_SUPPORT_STRIDEDCOPY
#define MLIR_EXECUTOR_RUNTIME_SUPPORT_STRIDEDCOPY

#include "llvm/ADT/ArrayRef.h"
#include <vector>

namespace mtrt {

//===----------------------------------------------------------------------===//
// Copy utilities
//===----------------------------------------------------------------------===//

void executeStridedCopy(
    int64_t elemSize, uintptr_t src, int64_t srcOffset,
    const std::vector<int64_t> &srcShape, std::vector<int64_t> &srcStrides,
    uintptr_t dst, int64_t dstOffset, const std::vector<int64_t> &dstShape,
    std::vector<int64_t> &dstStrides,
    std::function<void(void *dst, void *src, size_t size)> memcpyFunc);

/// Execute a strided copy where the strides and offsets are given in bytes.
void executeStridedByteCopy(
    uintptr_t src, int64_t srcOffsetBytes, const std::vector<int64_t> &srcShape,
    const std::vector<int64_t> &srcByteStrides, uintptr_t dst,
    int64_t dstOffsetBytes, const std::vector<int64_t> &dstShape,
    const std::vector<int64_t> &dstByteStrides, size_t elemSizeBytes,
    std::function<void(void *dst, void *src, size_t size)> memcpyFunc);

} // namespace mtrt

#endif // MLIR_EXECUTOR_RUNTIME_SUPPORT_STRIDEDCOPY
