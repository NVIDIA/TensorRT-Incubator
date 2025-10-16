//===- GenerateABIWrappers.cpp
//----------------------------------------------===//
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
/// Generates wrapper functions that are compatible with the Executor ABI
/// for all public functions.
///
/// The existing public functions are made private and only the new wrapper
/// functions are public after the pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::executor {
#define GEN_PASS_DEF_EXECUTORGENERATEABIWRAPPERSPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using namespace mlir::executor;

static bool isScalarType(Type type) {
  return isa<FloatType, IndexType, IntegerType>(type);
}

/// Construct the signature for the ABI wrapper function.
static FailureOr<FunctionType>
createABISignature(FunctionOpInterface func,
                   SmallVectorImpl<Attribute> &argAttrs,
                   bool forceUndefOutputArgs) {
  OpBuilder builder(func.getContext());
  auto funcType = cast<FunctionType>(func.getFunctionType());
  SmallVector<Type> argTypes;
  argTypes.reserve(funcType.getNumInputs() + funcType.getNumResults());
  auto hostPtrType = executor::PointerType::get(funcType.getContext(),
                                                executor::MemoryType::host);
  for (auto &arg : func.getArguments()) {
    if (isScalarType(arg.getType())) {
      argTypes.push_back(arg.getType());
      auto attrs = llvm::to_vector(func.getArgAttrs(arg.getArgNumber()));
      argAttrs.push_back(DictionaryAttr::get(func.getContext(), attrs));
      continue;
    }
    if (!isa<MemRefType, RankedTensorType, TableType>(arg.getType()))
      return emitError(arg.getLoc())
             << "type not supported by the Executor runtime ABI: "
             << arg.getType();
    argTypes.push_back(hostPtrType);

    if (func.getArgAttr(arg.getArgNumber(),
                        executor::ExecutorDialect::kArgABIAttrName))
      return emitError(arg.getLoc())
             << "function " << func.getName() << " argument "
             << arg.getArgNumber() << " is already annotated with a "
             << executor::ExecutorDialect::kArgABIAttrName << " attribute";

    auto attrs = llvm::to_vector(func.getArgAttrs(arg.getArgNumber()));
    attrs.emplace_back(executor::ExecutorDialect::kArgABIAttrName,
                       executor::ArgumentABIAttr::get(
                           executor::ArgABIKind::byval, arg.getType()));
    argAttrs.push_back(DictionaryAttr::get(func.getContext(), attrs));
  }
  // Append the results arguments.
  for (auto [idx, result] : llvm::enumerate(funcType.getResults())) {
    if (!isScalarType(result) &&
        !isa<MemRefType, RankedTensorType, TableType>(result))
      return emitError(func.getLoc())
             << "result type not supported by the Executor runtime ABI: "
             << result;
    // All results are passed by reference
    argTypes.push_back(hostPtrType);
    auto attrs = llvm::to_vector(func.getResultAttrs(idx));
    attrs.emplace_back(executor::ExecutorDialect::kResultArgAttrName,
                       builder.getI32IntegerAttr(idx));
    attrs.emplace_back(
        executor::ExecutorDialect::kArgABIAttrName,
        executor::ArgumentABIAttr::get(executor::ArgABIKind::byref, result,
                                       forceUndefOutputArgs));
    argAttrs.push_back(DictionaryAttr::get(func.getContext(), attrs));
  }
  return FunctionType::get(funcType.getContext(), argTypes, {});
}

namespace {
class ExecutorGenerateABIWrappersPass
    : public executor::impl::ExecutorGenerateABIWrappersPassBase<
          ExecutorGenerateABIWrappersPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    Operation *module = getOperation();

    SymbolTableCollection symbolTables;
    SymbolUserMap symbolUserMap(symbolTables, module);
    SymbolTable &topLevelSymbolTable = symbolTables.getSymbolTable(module);

    SmallVector<func::FuncOp> funcs;
    for (auto func : module->getRegion(0).getOps<func::FuncOp>()) {
      if (!func.isPublic() || func.isDeclaration())
        continue;
      if (!func.getBody().hasOneBlock()) {
        emitError(func.getLoc()) << "the input public function "
                                 << func.getName() << " has multiple blocks, "
                                 << "which is not supported";
        return signalPassFailure();
      }
      funcs.push_back(func);
    }

    SmallVector<std::string> originalFuncNames;
    for (auto func : funcs) {
      originalFuncNames.push_back(func.getName().str());
      if (failed(topLevelSymbolTable.rename(func,
                                            func.getName().str() + "_impl"))) {
        emitError(func.getLoc())
            << "failed to rename function " << func.getName() << " to "
            << func.getName().str() + "_impl";
        return signalPassFailure();
      }
    }

    for (auto [func, originalFuncName] :
         llvm::zip_equal(funcs, originalFuncNames)) {
      func.setPrivate();
      func.setNoInline(false);

      FunctionType funcType = func.getFunctionType();
      SmallVector<Attribute> argAttrs;
      FailureOr<FunctionType> abiFuncType =
          createABISignature(func, argAttrs, forceUndefOutputArgs);
      if (failed(abiFuncType)) {
        emitError(func.getLoc())
            << "failed to create ABI signature for function " << func.getName();
        return signalPassFailure();
      }
      rewriter.setInsertionPointAfter(func);
      func::FuncOp newFunc = rewriter.create<func::FuncOp>(
          func.getLoc(), originalFuncName, *abiFuncType,
          rewriter.getStringAttr("public"), rewriter.getArrayAttr(argAttrs),
          ArrayAttr{},
          /*no_inline=*/false);

      newFunc->setAttr(ExecutorDialect::kFuncABIAttrName,
                       TypeAttr::get(funcType));

      SmallVector<Location> argLocs;
      for (auto arg : func.getArguments())
        argLocs.push_back(arg.getLoc());
      for (Value result : func.getBody().front().getTerminator()->getOperands())
        argLocs.push_back(result.getLoc());

      assert(argLocs.size() == abiFuncType->getNumInputs());
      Block *newBlock =
          rewriter.createBlock(&newFunc.getBody(), newFunc.getBody().end(),
                               abiFuncType->getInputs(), argLocs);
      rewriter.setInsertionPointToStart(newBlock);

      SmallVector<Value> callArgs;
      TypeRange originalArgTypes = funcType.getInputs();
      for (auto [arg, originalType] : llvm::zip_equal(
               newFunc.getArguments().take_front(funcType.getNumInputs()),
               originalArgTypes)) {

        if (arg.getType() == originalType) {
          callArgs.push_back(arg);
          continue;
        }
        Type valueType = originalType;
        if (!isa<ComplexType, MemRefType, RankedTensorType, TableType>(
                originalType)) {
          emitError(arg.getLoc()) << "type not supported: " << originalType;
          return signalPassFailure();
        }
        assert(isa<executor::PointerType>(arg.getType()) &&
               "expected pointer type");

        auto recvOp =
            rewriter.create<executor::ABIRecvOp>(arg.getLoc(), valueType, arg);

        Value callArg = recvOp.getResult();
        if (callArg.getType() != originalType)
          callArg = rewriter
                        .create<UnrealizedConversionCastOp>(
                            arg.getLoc(), originalType, callArg)
                        .getResult(0);
        callArgs.push_back(callArg);
      }

      auto callOp =
          rewriter.create<func::CallOp>(func.getLoc(), func, callArgs);

      for (auto [result, outArg] : llvm::zip_equal(
               callOp.getResults(),
               newFunc.getArguments().take_back(funcType.getNumResults()))) {
        assert(isa<executor::PointerType>(outArg.getType()));
        rewriter.create<executor::ABISendOp>(result.getLoc(), result, outArg);
      }
      rewriter.create<func::ReturnOp>(func.getLoc());
    }
  }
};
} // namespace
