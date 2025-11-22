//===- ModuleUtils.h --------------------------------------------*- C++ -*-===//
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
/// Utilities for querying information about or manipulating module-like ops.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_UTILS_MODULEUTILS
#define MLIR_TENSORRT_UTILS_MODULEUTILS

#include "mlir/IR/Operation.h"

namespace mlir {
class FunctionOpInterface;
/// A ModuleLikeOp is a simple adaptor around Operation* to check whether an
/// operation has SymbolTable, IsolatedFromAbove, and OneRegion op traits. A
/// small number of convenience methods added, otherwise access the underlying
/// Operation* instead of adding additional methods.
class ModuleLikeOp {
public:
  ModuleLikeOp(Operation *op);
  StringRef getSymbolName() const;

  operator Operation *() const { return op; }
  operator bool() const { return op != nullptr; }
  bool operator==(Operation *other) const { return this->op == other; }

  Operation *operator->() { return op; }
  Operation *operator*() { return op; }

  template <typename T>
  auto getOps() const {
    return op->getRegion(0).getOps<T>();
  }
  auto getOps() const { return op->getRegion(0).getOps(); }

private:
  Operation *op;
};

LogicalResult getFuncOpsOrderedByCalls(
    ModuleLikeOp moduleOp, SmallVectorImpl<FunctionOpInterface> &orderedFuncOps,
    SmallVectorImpl<FunctionOpInterface> &remainingFuncOps,
    const std::function<bool(FunctionOpInterface)> &filter = nullptr);
} // namespace mlir

#endif // MLIR_TENSORRT_UTILS_MODULEUTILS
