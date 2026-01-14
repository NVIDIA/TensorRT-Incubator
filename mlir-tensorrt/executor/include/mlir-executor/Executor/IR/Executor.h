//===- Executor.h --------------------------------------------------------===//
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
///
/// Executor dialect declarations
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTOR_H
#define MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTOR_H

#include "mlir-executor/Executor/IR/ExecutorAttributes.h" // IWYU pragma: keep
#include "mlir-tensorrt-common/Interfaces/StreamSchedulableOpInterface.h" // IWYU pragma: keep
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace executor {
namespace detail {
LogicalResult verifyRuntimeBuiltinInterface(Operation *op,
                                            const DataLayout &dataLayout);

FailureOr<std::string>
getRuntimeBuiltinFunctionNameImpl(Operation *op, ArrayRef<Type> suffixTypes,
                                  const DataLayout &dataLayout);

FailureOr<CallOpInterface>
lowerToCallDefaultImpl(Operation *op, ArrayRef<Value> operands, ModuleOp module,
                       RewriterBase &rewriter,
                       const TypeConverter &typeConverter,
                       const DataLayout &dataLayout);

} // namespace detail
} // namespace executor
} // namespace mlir

//===----------------------------------------------------------------------===//
// Executor Dialect
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Executor Types
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "mlir-executor/Executor/IR/ExecutorOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// Executor Op Interfaces
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// Executor Ops
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-executor/Executor/IR/ExecutorOps.h.inc"

//===----------------------------------------------------------------------===//
// Executor Utilities
//===----------------------------------------------------------------------===//
namespace mlir::executor {
/// Return an external function declaration within the module, creating a new
/// declaration at the top of the module if necessary.
SymbolRefAttr getOrInsertFuncDeclaration(OpBuilder &rewriter, Location loc,
                                         ModuleOp module, StringRef name,
                                         ExecutorFunctionType sig);

} // namespace mlir::executor

#endif // MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTOR_H
