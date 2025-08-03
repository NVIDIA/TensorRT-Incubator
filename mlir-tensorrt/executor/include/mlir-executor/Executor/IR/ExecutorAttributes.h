//===- ExecutorAttributes.h -------------------------------------*- C++ -*-===//
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
/// Executor dialect attribute declarations
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_EXECUTOR_IR_EXECUTORATTRIBUTES
#define MLIR_EXECUTOR_EXECUTOR_IR_EXECUTORATTRIBUTES

#include "mlir/IR/OpImplementation.h"
#include <optional>

namespace mlir::func {
class FuncOp;
}

//===----------------------------------------------------------------------===//
// Executor Enums
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorEnums.h.inc"

//===----------------------------------------------------------------------===//
// Executor Attributes
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "mlir-executor/Executor/IR/ExecutorAttributes.h.inc"

//===----------------------------------------------------------------------===//
// Executor Arg Attributes Utilities
//===----------------------------------------------------------------------===//

namespace mlir::executor {

/// Extract func argument attribute at an index. Attribute should have a
/// concrete type of `executor::ValueBoundsAttr` or
/// `executor::DimensionBoundsAttr`.
Attribute getFuncArgsBounds(func::FuncOp func, int64_t argIdx);

// Extract func result attribute at an index. Attribute should have a concrete
/// type of `executor::ValueBoundsAttr` or `executor::DimensionBoundsAttr`.
Attribute getFuncResultBounds(func::FuncOp func, int64_t resultIdx);

//===----------------------------------------------------------------------===//
// Module Attributes Utilities
//===----------------------------------------------------------------------===//

/// Return the process grid shape as specified by the attribute attached to the
/// module operation.
FailureOr<ArrayRef<int64_t>> getModuleProcessGridShape(Operation *op);

/// Set the process grid shape on the module.
LogicalResult setModuleProcessGridShape(Operation *op, ArrayRef<int64_t> shape);

/// Returns the name of the attribute attached to the top-level module that
/// holds the function symbol that will initialize all globals after the
/// `executor.global` regions are lowered. This is the only function that is
/// allowed to make `executor.set_global` calls to gloals marked constant.
StringRef getExecutorGlobalInitializerFuncNameAttr();

} // namespace mlir::executor

#endif // MLIR_EXECUTOR_EXECUTOR_IR_EXECUTORATTRIBUTES
