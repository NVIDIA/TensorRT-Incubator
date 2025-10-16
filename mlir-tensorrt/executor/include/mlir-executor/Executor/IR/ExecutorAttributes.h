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

namespace mlir {
class FunctionOpInterface;
}

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

namespace abi {

/// Get the ABI function type from the `executor.func_abi` attribute attached
/// to the function. Returns failure if the attribute is not present or is not
/// a valid FunctionType.
FailureOr<FunctionType> getABIFunctionType(FunctionOpInterface func);

/// Return true if this function is an Executor runtime compatible ABI wrapper
/// function.
bool isABIWrapperFunction(FunctionOpInterface func);

/// For a given argument of an API wrapper function, return whether this
/// argument is an "input argument" or an "output argument". It is an input
/// argument if it is one of of the first N arguments of the function where N is
/// given by the number of inputs of the ABI function type (attached to the
/// function via the `executor.func_abi` attribute). If it is an input argument,
/// return the index of the corresponding input of the ABI function type (which
/// is really just the argument number).
std::optional<unsigned> isInputArgument(FunctionOpInterface func,
                                        unsigned argIndex);

/// For a given argument of an API wrapper function, return whether this
/// argument is an "output argument". It is an output argument
/// if it is one of of the last M arguments of the function where M is given
/// by the number of results of the ABI function type (attached to the function
/// via the `executor.func_abi` attribute). If it is an output argument, return
/// the index of the corresponding result of the ABI function type.
std::optional<unsigned> isOutputArgument(FunctionOpInterface func,
                                         unsigned argIndex);

/// For the given argument of an API wrapper function, return whether this
/// argument is an "output argument".
std::optional<unsigned> isOutputArgument(FunctionOpInterface func,
                                         BlockArgument arg);

/// Return the ArgumentABIAttr for the given argument.
ArgumentABIAttr getArgumentABIAttr(FunctionOpInterface func, BlockArgument arg);

/// Return the ArgumentABIAttr for the given argument index.
ArgumentABIAttr getArgumentABIAttr(FunctionOpInterface func, unsigned argIndex);

/// Set the ArgumentABIAttr for the given argument.
void setArgumentABIAttr(FunctionOpInterface func, BlockArgument arg,
                        ArgumentABIAttr abiAttr);

/// Get or create an ABIRecvOp for the given function argument.
/// This function asserts that `func` is an ABI wrapper function.
/// If an ABIRecvOp already exists for the argument at `argIndex`, it returns
/// its result (and asserts the type matches `expectedType` if provided).
/// Otherwise, it creates a new ABIRecvOp with the result type derived from
/// the ABI function type (or `expectedType` if provided).
Value getOrCreateABIRecv(OpBuilder &b, FunctionOpInterface func,
                         BlockArgument arg, Type expectedType = nullptr);
Value getOrCreateABIRecv(OpBuilder &b, FunctionOpInterface func,
                         unsigned argIndex, Type expectedType = nullptr);

} // namespace abi

} // namespace mlir::executor

#endif // MLIR_EXECUTOR_EXECUTOR_IR_EXECUTORATTRIBUTES
