//===- ExecutorAttributes.h ----------------------------------------------===//
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
/// Executor dialect attribute declarations
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_EXECUTOR_IR_EXECUTORATTRIBUTES
#define MLIR_EXECUTOR_EXECUTOR_IR_EXECUTORATTRIBUTES

#include "mlir-executor/Executor/IR/Enums.h" // IWYU pragma: keep
#include "mlir/Interfaces/FunctionInterfaces.h"
#include <optional>
#include <variant>

namespace mlir {
class FunctionOpInterface;
namespace func {
class FuncOp;
}
} // namespace mlir

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
Attribute getFuncArgsBounds(FunctionOpInterface func, int64_t argIdx);

// Extract func result attribute at an index. Attribute should have a concrete
/// type of `executor::ValueBoundsAttr` or `executor::DimensionBoundsAttr`.
Attribute getFuncResultBounds(FunctionOpInterface func, int64_t resultIdx);

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

/// Return true if this function is an Executor runtime compatible ABI wrapper
/// function.
bool isABIWrapperFunction(FunctionOpInterface func);

/// Get the ABI function type from the `executor.func_abi` attribute attached
/// to the function. Returns failure if the attribute is not present or is not
/// a valid FunctionType.
FailureOr<FunctionType> getABIFunctionType(FunctionOpInterface func);

/// Return the number of input arguments of the ABI function type.
unsigned getNumInputArguments(FunctionOpInterface func);

/// Return the number of output arguments of the ABI function type.
unsigned getNumOutputArguments(FunctionOpInterface func);

/// Return the index of the corresponding result for the given output argument.
unsigned getOutputArgumentIndex(FunctionOpInterface func, BlockArgument arg);

/// Return the index of the corresponding input for the given input argument.
unsigned getInputArgumentIndex(FunctionOpInterface func, BlockArgument arg);

/// Set the ABI function type on the function.
void setABIFunctionType(FunctionOpInterface func, TypeRange inputTypes,
                        TypeRange resultTypes);

/// Update the value type of an input argument of the ABI function.
/// Updates all required function attributes and argument attributes.
void updateABIInputArgumentValueType(FunctionOpInterface func,
                                     unsigned inputIdx, Type valueType);

/// Update the value type of an output argument of the ABI function.
/// Updates all required function attributes and result attributes.
void updateABIOutputArgumentValueType(FunctionOpInterface func,
                                      unsigned outputIdx, Type valueType);

/// For a given argument of an API wrapper function, return whether this
/// argument is an "input argument" or an "output argument". It is an input
/// argument if it is one of of the first N arguments of the function where N is
/// given by the number of inputs of the ABI function type (attached to the
/// function via the `executor.func_abi` attribute). If it is an input argument,
/// return the index of the corresponding input of the ABI function type (which
/// is really just the argument number).
std::optional<unsigned> isInputArgument(FunctionOpInterface func,
                                        unsigned argIndex);

/// Get the index-th entry block argument of the ABI function.
BlockArgument getInputArgument(FunctionOpInterface func, unsigned index);

/// Get the index-th entry block argument representing an output of the ABI
/// function (the block argument at position `num_block_args - index - 1`).
BlockArgument getOutputArgument(FunctionOpInterface func, unsigned index);

/// Returns true if this is a valid scalar argument type that can be
/// passed-by-value by an ABI wrapper function. If not, then it must be
/// passed-by-pointer.
bool isScalarArgumentType(Type type);

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

/// Set the ArgumentABIAttr for the given input argument index.
void setInputArgumentABIAttr(FunctionOpInterface func, unsigned inputIdx,
                             ArgumentABIAttr abiAttr);

/// Set the ArgumentABIAttr for the given output argument index.
void setOutputArgumentABIAttr(FunctionOpInterface func, unsigned outputIdx,
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

/// Verify public exported ABI functions have no callers.
/// This is has a dedicated function which should be invoked at critical points
/// since checking it in the verifier is expensive.
FailureOr<SmallVector<FunctionOpInterface>>
collectAndValidateABIFuncs(Operation *module);

namespace plugin {
struct DecodeArg {
  unsigned index;
};
struct DecodeRet {
  unsigned index;
};
struct DecodeAttr {
  llvm::StringRef name;
};
struct OptionalNoneTag {};
struct DecodeItem {
  std::variant<DecodeArg, DecodeRet, DecodeAttr, OptionalNoneTag> kind;
  size_t index{0};
  llvm::StringRef spec;
};
struct DecodeSpec {
  std::vector<DecodeItem> items;
};
struct TVMFFIPluginConfig {
  llvm::StringRef pluginName;
  llvm::StringRef functionName;
  llvm::SmallVector<llvm::StringRef> argSpec;
  llvm::SmallVector<int32_t> ioAliasing;
  FunctionType functionType;
  DictionaryAttr immediateArgs;
};

/// Parse an argument specification string into a `abi::plugin::DecodeSpec`.
/// `config` is the dictionary where we can lookup attributes used as immediate
/// arguments. These immediate arguments are placed into `immediateArgs`.
FailureOr<DecodeSpec>
ParseArgSpec(Operation *op, unsigned numInputArgs, unsigned numOutputArgs,
             llvm::StringRef argSpecString, DictionaryAttr config,
             llvm::SmallVectorImpl<llvm::StringRef> &argSpec,
             llvm::SmallVectorImpl<NamedAttribute> &immediateArgs);
/// Parse an argument specification string into a `abi::plugin::DecodeSpec`.
/// `config` is the dictionary where we can lookup attributes used as immediate
/// arguments. These immediate arguments are placed into `immediateArgs`.
FailureOr<DecodeSpec>
ParseArgSpec(Operation *op, unsigned numInputArgs, unsigned numOutputArgs,
             llvm::ArrayRef<llvm::StringRef> argSpecString,
             DictionaryAttr config,
             llvm::SmallVectorImpl<NamedAttribute> &immediateArgs);

} // namespace plugin

} // namespace abi

} // namespace mlir::executor

#endif // MLIR_EXECUTOR_EXECUTOR_IR_EXECUTORATTRIBUTES
