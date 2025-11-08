//===- LowerABIOps.cpp
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
/// Lowers executor.abi.recv and executor.abi.send operations to more primitive
/// operations.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"

namespace mlir::executor {
#define GEN_PASS_DEF_EXECUTORLOWERABIOPSPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using namespace mlir::executor;

static bool isScalarType(Type type) {
  return isa<FloatType, IndexType, IntegerType>(type);
}

static bool isComplexType(Type type) { return isa<ComplexType>(type); }

/// Extract base pointers from the given memrefs and check if they are the same.
static Value createSameBasePointerCheck(OpBuilder &builder, Location loc,
                                        Value memref1, Value memref2) {
  Value ptr1AsIdx =
      builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref1);
  Value ptr2AsIdx =
      builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref2);
  Value doesAlias = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  ptr1AsIdx, ptr2AsIdx);
  return doesAlias;
}

static Operation *findUniqueReturnOp(FunctionOpInterface func) {
  Operation *returnOp{nullptr};
  for (Block &block : func.getBlocks()) {
    Operation *term = block.getTerminator();
    if (term->hasTrait<OpTrait::ReturnLike>()) {
      if (returnOp)
        return nullptr;
      returnOp = term;
    }
  }
  return returnOp;
}

static LogicalResult updateABIFunction(RewriterBase &rewriter,
                                       FunctionOpInterface func) {
  SmallVector<executor::ABISendOp> opsToErase;
  auto result = func->walk([&](executor::ABISendOp sendOp) -> WalkResult {
    Value value = sendOp.getValue();
    Type valueType = value.getType();
    Value ptr = sendOp.getPtr();
    rewriter.setInsertionPoint(sendOp);

    // If the value operand is a scalar or complex type, do nothing.
    if (isScalarType(valueType) || isComplexType(valueType)) {
      return WalkResult::advance();
    }

    // If the value operand is not scalar, complex, or memref, return an error
    if (!isa<MemRefType>(valueType)) {
      return sendOp.emitError(
          "value type must be scalar, complex, or memref type");
    }

    // For memref types, check ownership value
    Value ownership = sendOp.getOwnership();
    if (!ownership) {
      return sendOp.emitError("memref value requires ownership operand");
    }

    // Get the function containing this operation
    auto func = sendOp->getParentOfType<FunctionOpInterface>();
    if (!func) {
      return sendOp.emitError("abi.send must be inside a function");
    }

    // Get the block argument for the ptr operand
    auto blockArg = dyn_cast<BlockArgument>(ptr);
    if (!blockArg) {
      return sendOp.emitError(
          "ptr operand must be a block argument for ABI lowering");
    }

    // Get the ABI attribute for this argument
    executor::ArgumentABIAttr abiAttr =
        executor::abi::getArgumentABIAttr(func, blockArg);
    if (!abiAttr) {
      return sendOp.emitError("ptr argument missing ABI attribute");
    }

    // Check if the argument has 'undef' parameter set
    const bool hasUndefParam = abiAttr.getUndef();
    Type i1Type = rewriter.getIntegerType(1);

    // If we don't have `undef` parameter set, we may need to insert a copy.
    Location loc = sendOp.getLoc();
    if (!hasUndefParam) {
      rewriter.setInsertionPoint(sendOp);
      Value outMemRef =
          abi::getOrCreateABIRecv(rewriter, func, blockArg, value.getType());
      Value aliasCheck =
          createSameBasePointerCheck(rewriter, loc, value, outMemRef);
      rewriter.create<scf::IfOp>(
          sendOp.getLoc(), aliasCheck,
          [&](OpBuilder &builder, Location loc) {
            Value one = builder.create<arith::ConstantOp>(
                loc, builder.getOneAttr(i1Type));
            Value notOwned = builder.create<arith::XOrIOp>(loc, ownership, one);
            builder.create<cf::AssertOp>(loc, notOwned,
                                         "expected ownership to be false");
            builder.create<scf::YieldOp>(loc, ValueRange{});
          },
          [&](OpBuilder &builder, Location loc) {
            // We don't care about ownership here since the deallocation pass
            // will already take care of deallocating if owned.
            builder.create<memref::CopyOp>(loc, value, outMemRef);
            builder.create<scf::YieldOp>(loc, ValueRange{});
          });

      opsToErase.push_back(sendOp);
    }
    // Undef case -- caller requires an owned memref value. If ownership is
    // not statically true, insert a clone at runtime based on the ownership
    // value.
    else {
      rewriter.setInsertionPoint(sendOp);
      auto ifOp = rewriter.create<scf::IfOp>(loc, value.getType(), ownership,
                                             true, true);
      Block *thenBlock = ifOp.thenBlock();
      Block *elseBlock = ifOp.elseBlock();
      rewriter.setInsertionPointToStart(thenBlock);
      rewriter.create<scf::YieldOp>(loc, value);
      rewriter.setInsertionPointToStart(elseBlock);
      auto clone = rewriter.create<bufferization::CloneOp>(loc, value);
      rewriter.create<scf::YieldOp>(loc, clone.getResult());
      rewriter.setInsertionPointAfter(ifOp);
      Value trueVal =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getOneAttr(i1Type));
      sendOp.getOwnershipMutable().assign(trueVal);
      sendOp.getValueMutable().assign(ifOp.getResult(0));
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  // Erase the return and update function type.
  Operation *returnOp = findUniqueReturnOp(func);
  if (!returnOp)
    return func->emitError("expected a single return operation");
  rewriter.setInsertionPoint(returnOp);

  llvm::BitVector resultIndices(func.getNumResults(), true);
  func.eraseResults(resultIndices);
  returnOp->eraseOperands(0, returnOp->getNumOperands());

  // Erase the operations that can be removed
  for (auto op : opsToErase) {
    op.erase();
  }

  FunctionType funcType = cast<FunctionType>(func.getFunctionType());
  auto newFuncType =
      FunctionType::get(funcType.getContext(), funcType.getInputs(), {});
  func.setFunctionTypeAttr(TypeAttr::get(newFuncType));

  return success();
}

/// Update ABI functions to lower `executor.abi.send`. All results are dropped
/// at this point since results are now correctly captured.
static LogicalResult updateABIFunctions(RewriterBase &rewriter,
                                        Operation *module) {
  FailureOr<SmallVector<FunctionOpInterface>> abiFuncs =
      abi::collectAndValidateABIFuncs(module);
  if (failed(abiFuncs))
    return failure();

  for (auto func : *abiFuncs) {
    if (failed(updateABIFunction(rewriter, func)))
      return failure();
  }
  return success();
}

namespace {
class ExecutorLowerABIOpsPass
    : public executor::impl::ExecutorLowerABIOpsPassBase<
          ExecutorLowerABIOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *module = getOperation();
    IRRewriter rewriter(module);
    if (failed(updateABIFunctions(rewriter, module)))
      return signalPassFailure();
  }
};
} // namespace
