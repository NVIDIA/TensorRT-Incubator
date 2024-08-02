//===- Executor.h -----------------------------------------------*- C++ -*-===//
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
/// Executor dialect declarations
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTOR_H
#define MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTOR_H

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {
namespace executor {
namespace detail {
/// Verify an operation that has RuntimeBuiltinInterface attached.
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
// Executor Enums
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorEnums.h.inc"

//===----------------------------------------------------------------------===//
// Executor Attributes
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "mlir-executor/Executor/IR/ExecutorAttributes.h.inc"

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
// Executor Traits
//===----------------------------------------------------------------------===//

namespace mlir::executor {

/// A trait that simply indicates an Executor operation should be lowered to
/// a external procedure call.
template <typename ConcreteType>
class LowerToFuncCallTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, LowerToFuncCallTrait> {};

/// Returns the name of the attribute attached to the top-level module that
/// holds the function symbol that will initialize all globals after the
/// `executor.global` regions are lowered. This is the only function that is
/// allowed to make `executor.set_global` calls to gloals marked constant.
StringRef getExecutorGlobalInitializerFuncNameAttr();

} // namespace mlir::executor

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

std::optional<uint64_t>
getUniformWidthOfTableElements(executor::TableType t,
                               const mlir::DataLayout &dataLayout);

/// Return the process grid shape as specified by the attribute attached to the
/// module operation.
FailureOr<ArrayRef<int64_t>> getModuleProcessGridShape(ModuleOp op);

/// Set the process grid shape on the module.
LogicalResult setModuleProcessGridShape(ModuleOp op, ArrayRef<int64_t> shape);

/// Extract func argument attribute at an index. Attribute should have a
/// concrete type of `executor::ValueBoundsAttr` or
/// `executor::DimensionBoundsAttr`.
Attribute getFuncArgsBounds(func::FuncOp func, int64_t argIdx);

// Extract func result attribute at an index. Attribute should have a concrete
/// type of `executor::ValueBoundsAttr` or `executor::DimensionBoundsAttr`.
Attribute getFuncResultBounds(func::FuncOp func, int64_t resultIdx);

} // namespace mlir::executor

#endif // MLIR_TENSORRT_DIALECT_EXECUTOR_IR_EXECUTOR_H
