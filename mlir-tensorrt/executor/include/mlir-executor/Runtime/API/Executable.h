//===- Executable.h --------------------------------------------*- C++ -*-===//
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
/// Declares the generated Flatbuffer serialization/deserialization and
/// other helper routines related to the RuntimeExecutable format.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_RUNTIME_API_EXECUTABLE
#define MLIR_EXECUTOR_RUNTIME_API_EXECUTABLE

#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#endif
#include "mlir-executor/Runtime/API/ExecutableFlatbuffer.h"
#if defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace mtrt {
// Alias some objects from the generated Flatbuffer object API class instead of
// using them directly.
using ScalarTypeCode = mtrt::flat::ScalarTypeCode;
using PointerType = mtrt::flat::PointerType;
using PointerOwner = mtrt::flat::PointerOwner;
using TypeCode = mtrt::flat::Type;
using CallingConvention = mtrt::flat::CallingConvention;

//===----------------------------------------------------------------------===//
// ExecutableStorage
//===----------------------------------------------------------------------===//

/// A ExecutableStorage manages storage for the executable. Different concrete
/// implementations may choose to manage the storage using e.g.
/// `llvm::MemoryBuffer` or via a just-encoded flatbuffer-allocated buffer.
class ExecutableStorage {
public:
  ExecutableStorage() = default;
  virtual ~ExecutableStorage() {}
  ExecutableStorage(const ExecutableStorage &) = delete;
  ExecutableStorage &operator=(const ExecutableStorage &) = delete;

  virtual std::unique_ptr<ExecutableStorage> getCopy() const = 0;

  virtual const void *data() const = 0;
  virtual size_t size() const = 0;
};

} // namespace mtrt

namespace mtrt {
// Alias some objects from the generated Flatbuffer object API class instead of
// using them directly.
using ScalarTypeCode = mtrt::ScalarTypeCode;
using PointerType = mtrt::flat::PointerType;
using PointerOwner = mtrt::flat::PointerOwner;
using TypeCode = mtrt::flat::Type;
using CallingConvention = mtrt::flat::CallingConvention;
using CudaStream = uintptr_t;
using CudaEvent = uintptr_t;

using ExecutableStorage = mtrt::ExecutableStorage;

} // namespace mtrt

#endif // MLIR_EXECUTOR_RUNTIME_API_EXECUTABLE
