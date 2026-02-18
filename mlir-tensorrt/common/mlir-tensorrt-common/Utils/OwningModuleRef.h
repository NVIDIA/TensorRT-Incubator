//===- OwningModuleRef.h ---------------------------------------*- C++ -*-===//
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
/// This file contains the declarations for the `OwningModuleRef` class.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_UTILS_OWNINGMODULEREF
#define MLIR_TENSORRT_COMMON_UTILS_OWNINGMODULEREF

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include <functional>
#include <type_traits>
#include <utility>

namespace mlir {
class Operation;
class ModuleOp;

namespace detail {
/// A hook that releases the memory blobs associated with any
/// ResourceElementsAttr referenced by the module.
void releaseReferencedResourceElementsAttrsBlobs(mlir::Operation *op);
} // namespace detail

/// Acts as an owning reference to a top-level module. This is similar to
/// `mlir::OwningOpRef`, but the caller can add additional hooks that are run
/// before and after the op is destroyed. There is a default hook that erases
/// any ResourceElementsAttrs that are referenced by the module.
class OwningModuleRef {
public:
  OwningModuleRef(std::nullptr_t = nullptr);

  OwningModuleRef(ModuleOp op);

  OwningModuleRef(OwningModuleRef &&other);

  ~OwningModuleRef();

  void eraseOp();

  /// Assign from another op reference.
  OwningModuleRef &operator=(OwningModuleRef &&other);

  /// Allow accessing the internal op.
  ModuleOp get() const;
  ModuleOp operator*() const;
  ModuleOp operator->();

  explicit operator bool() const { return op; }

  /// Release the referenced op.
  ModuleOp release();

  std::function<void(mlir::Operation *)> beforeDestroyHook;
  std::function<void()> afterDestroyHook;

private:
  mlir::ModuleOp op;
};

} // namespace mlir

#endif // MLIR_TENSORRT_COMMON_UTILS_OWNINGMODULEREF
