//===- LowerToLLVM.h --------------------------------------------*- C++ -*-===//
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
/// Common utilities for LLVM conversions.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_CONVERSION_LLVMCOMMON_LLVMCOMMON
#define MLIR_TENSORRT_CONVERSION_LLVMCOMMON_LLVMCOMMON

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {

/// The `LLVMOpaqueCallBuilder` assists with constructing calls to externally
/// defined functions. On a call to `create`, the LLVM function declaration with
/// the specified type is inserted into the module if it does not already exist,
/// and a call to that function is inserted at the current insertion point.
///
/// ## Best practices for use:
///
/// If the conversion pass needs to create many function declarations and calls,
/// then try to declare all the `LLVMOpaqueCallBuilder` up at the top of the
/// file in a base class for easy reference. For example:
///
/// ```C++
/// tempalte<typename T>
/// struct XToLLVMOpConverter : public ConvertOpToLLVMPattern<T> {
///   using ConvertOpToLLVMPattern<T>::ConvertOpToLLVMPattern;
///
///   MLIRContext *ctx{getContext()};
///   Type llvmVoidType{...};
///   Type i32Type{...};
///   Type llvmPtrType{...};

///   LLVMOpaqueCallBuilder myFuncCallbuilder = {
///     "my_func", llvmVoidType, {i32Type, llvmPtrType}
///    };
/// };
/// ```
///
/// The in your derived patterns, you have access to `myFuncCallBuilder`
/// in your `matchAndRewrite` functions, and mistakes regarding the
/// function signature are minimized.
///
/// ## Provide the `symbolTables` argument if possible
///
/// Calls to `create` will cause a search for the specified symbol. Without
/// giving a reference to the `symbolTables` argument, this will incur a linear
/// search through the parent module, which can be avoided if SymbolTable is
/// provided. This may not always be possible, e.g. if using pattern-based
/// rewriters where the rewriters are created once during pass initialization.
struct LLVMOpaqueCallBuilder {
  LLVMOpaqueCallBuilder(StringRef functionName, Type returnType,
                        ArrayRef<Type> argumentTypes)
      : functionName(functionName),
        functionType(LLVM::LLVMFunctionType::get(returnType, argumentTypes)) {}

  /// Lookup or insert the function declaration and create a call to that
  /// function at the current insertion point.
  LLVM::CallOp create(Location loc, OpBuilder &builder,
                      ArrayRef<Value> arguments,
                      SymbolTable *symbolTable = nullptr) const;

  StringRef functionName;
  LLVM::LLVMFunctionType functionType;
};

/// Insert a `llvm.global` which holds the given C-string literal data and
/// construct `llvm.addressof` + `llvm.gep` operations to retrieve the pointer
/// to the start of the string.
///
/// The symbol name is only a starting point, it will be uniqued if there is a
/// existing symbol with that name in the table. Provide an existing
/// `symbolTable` to avoid building a new one.
Value insertLLVMStringLiteral(OpBuilder &rewriter, Location loc, StringRef data,
                              StringRef symbolName,
                              SymbolTable *symbolTable = nullptr);

/// Find an LLVM::GlobalOp matching the given properties or construct a new
/// global. Note that the new global's name may be changed/deduplicated if the
/// name is already in the table but the properties don't match.
LLVM::GlobalOp lookupOrInsertGlobal(
    OpBuilder &rewriter, Location loc, StringRef symbolName, bool constant,
    Type type, LLVM::Linkage linkage, Attribute initialValue = {},
    SymbolTable *symbolTable = nullptr,
    llvm::function_ref<Value(OpBuilder &rewriter, Location loc)> initBuilder =
        nullptr);

/// Insert an LLVM::GlobalOp with the given properties.
///
/// The symbol name is only a starting point, it will be uniqued if there is a
/// existing symbol with that name in the table. Provide an existing
/// `symbolTable` to avoid building a new one.
LLVM::GlobalOp insertLLVMGlobal(
    OpBuilder &rewriter, Location loc, StringRef symbolName, bool constant,
    Type type, LLVM::Linkage linkage, Attribute initialValue = {},
    SymbolTable *symbolTable = nullptr,
    llvm::function_ref<Value(OpBuilder &rewriter, Location loc)> initBuilder =
        nullptr);

/// Create an unranked memref descriptor from a ranked memref descriptor and the
/// original ranked type information. This will promote the ranked descriptor to
/// stack storage for reference by the unranked descriptor.
UnrankedMemRefDescriptor getUnrankedLLVMMemRefDescriptor(
    OpBuilder &rewriter, Location loc, const LLVMTypeConverter &typeConverter,
    Value rankedLLVMDescriptor, MemRefType rankedType);

/// Given a set of original values and their LLVM converted values (which may or
/// may not be different), return a new vector containing `originalOperands`
/// where any ranked memref descriptors are replaced with unranked descriptors.
/// Other values are not changed.
SmallVector<Value> promoteLLVMMemRefDescriptorsToUnranked(
    OpBuilder &rewriter, Location loc, const LLVMTypeConverter &typeConverter,
    ValueRange originalOperands, ValueRange convertedOperands);

template <typename ConcreteType, typename RangeType>
auto make_cast_range(RangeType &&range) {
  return llvm::map_range(std::forward<RangeType>(range), [](auto type) {
    assert(isa<ConcreteType>(type) &&
           "type range is not unformly the requested type");
    return llvm::cast<ConcreteType>(type);
  });
}

/// Add a function to the module with the given name (may be altered to
/// de-deuplicate symbol names). An 'llvm.global_ctors' operation is inserted
/// into the module referencing the function with the given priority.
LLVM::LLVMFuncOp insertLLVMCtorFunction(
    OpBuilder &rewriter, Location loc, SymbolTable &symbolTable, StringRef name,
    int32_t priority,
    const std::function<void(OpBuilder &, Location)> &bodyBuilder);

/// Add a function to the module with the given name (may be altered to
/// de-deuplicate symbol names). An 'llvm.global_dtors' operation is inserted
/// into the module referencing the function with the given priority.
LLVM::LLVMFuncOp insertLLVMDtorFunction(
    OpBuilder &rewriter, Location loc, SymbolTable &symbolTable, StringRef name,
    int32_t priority,
    const std::function<void(OpBuilder &, Location)> &bodyBuilder);

/// Serialize the given ElementsAttr to a file.
FailureOr<std::unique_ptr<llvm::ToolOutputFile>>
serializeElementsAttrToFile(Location loc, ElementsAttr denseResourceAttr,
                            StringRef outputDir, StringRef filename);

} // namespace mlir

#endif // MLIR_TENSORRT_CONVERSION_LLVMCOMMON_LLVMCOMMON
