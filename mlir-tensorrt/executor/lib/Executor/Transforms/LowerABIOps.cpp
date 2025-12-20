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
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::executor {
#define GEN_PASS_DEF_EXECUTORLOWERABIOPSPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using namespace mlir::executor;

#define DEBUG_TYPE "executor-lower-abi-ops"
#define DBGV(fmt, ...)                                                         \
  llvm::dbgs() << "[" DEBUG_TYPE "] " << llvm::formatv(fmt, __VA_ARGS__)

static llvm::SmallBitVector
resolveResultAliasing(RewriterBase &rewriter, FunctionOpInterface func,
                      MutableArrayRef<OpOperand> returnedValues) {
  BufferOriginAnalysis originAnalysis(func);
  llvm::EquivalenceClasses<OpOperand *> aliasSets;
  for (auto [idx, result] : llvm::enumerate(returnedValues)) {
    if (!isa<MemRefType>(result.get().getType()))
      continue;
    if (aliasSets.findLeader(&result) == aliasSets.member_end())
      aliasSets.insert(&result);
    for (auto [idx2, result2] : llvm::enumerate(returnedValues)) {
      if (idx == idx2 || !isa<MemRefType>(result2.get().getType()))
        continue;
      std::optional<bool> isSameAllocation =
          originAnalysis.isSameAllocation(result.get(), result2.get());
      if (!isSameAllocation.has_value() || *isSameAllocation)
        aliasSets.unionSets(&result, &result2);
    }
  }

  LLVM_DEBUG(DBGV("aliasSets created with {0} equivalence classes\n",
                  aliasSets.getNumClasses()));

  // Within each equivalence class, choose the value *not* to clone.
  llvm::DenseMap<OpOperand *, OpOperand *> doNotClone;
  for (llvm::EquivalenceClasses<OpOperand *>::iterator
           leaderIt = aliasSets.begin(),
           end = aliasSets.end();
       leaderIt != end; ++leaderIt) {
    if (!(*leaderIt)->isLeader())
      continue;
    SmallVector<OpOperand *> ranked;
    for (auto mit = aliasSets.member_begin(**leaderIt),
              meit = aliasSets.member_end();
         mit != meit; ++mit) {
      ranked.push_back(*mit);
    }
    assert(ranked.size() >= 1 && "expected at least one operand");
    llvm::sort(ranked, [](OpOperand *a, OpOperand *b) {
      MemRefType aType = cast<MemRefType>(a->get().getType());
      MemRefType bType = cast<MemRefType>(b->get().getType());
      int64_t aScore = aType.getLayout().isIdentity()
                           ? 0
                           : std::numeric_limits<int64_t>::max();
      int64_t bScore = bType.getLayout().isIdentity()
                           ? 0
                           : std::numeric_limits<int64_t>::max();
      if (aType.hasStaticShape())
        aScore -= aType.getNumElements();
      if (bType.hasStaticShape())
        bScore -= bType.getNumElements();
      return aScore < bScore;
    });
    if (ranked.size() >= 1)
      doNotClone[(*leaderIt)->getData()] = ranked[0];
  }

  llvm::SmallBitVector requiresClone(returnedValues.size(), false);
  for (auto [idx, result] : llvm::enumerate(returnedValues)) {
    if (!isa<MemRefType>(result.get().getType()))
      continue;
    auto leaderIt = aliasSets.findLeader(&result);
    assert(leaderIt != aliasSets.member_end() && "expected leader");
    bool valRequiresClone = doNotClone.lookup(*leaderIt) != &result;
    MemRefType resultType = cast<MemRefType>(result.get().getType());
    LLVM_DEBUG(DBGV("requiresClone[{0}] = {1}\n", idx, valRequiresClone));
    requiresClone[idx] =
        valRequiresClone || !resultType.getLayout().isIdentity();
  }

  return requiresClone;
}

static FailureOr<Value> reallocMemRefValue(OpBuilder &b, Value value,
                                           MemRefType destType) {
  auto srcType = llvm::cast<MemRefType>(value.getType());

  // Element type and rank must match.
  if (srcType.getElementType() != destType.getElementType())
    return failure();
  if (srcType.getRank() != destType.getRank())
    return failure();

  auto loc = value.getLoc();
  SmallVector<Value, 4> dynamicOperands;
  for (int i = 0; i < destType.getRank(); ++i) {
    if (destType.getShape()[i] != ShapedType::kDynamic)
      continue;
    Value size = b.create<memref::DimOp>(loc, value, i);
    dynamicOperands.push_back(size);
  }

  Value newValue = b.create<memref::AllocOp>(loc, destType, dynamicOperands);
  b.create<memref::CopyOp>(loc, value, newValue);
  return newValue;
}

static LogicalResult updateABIFunction(RewriterBase &rewriter,
                                       FunctionOpInterface func) {
  for (Block &block : func.getBlocks()) {
    Operation *returnOp = block.getTerminator();
    assert(returnOp->hasTrait<OpTrait::ReturnLike>() &&
           "expected return like op");

    llvm::SmallBitVector requiresClone =
        resolveResultAliasing(rewriter, func, returnOp->getOpOperands());

    for (auto [idx, value] : llvm::enumerate(returnOp->getOperands())) {
      Type valueType = value.getType();
      BlockArgument ptr = abi::getOutputArgument(func, idx);
      rewriter.setInsertionPoint(returnOp);

      // If the value operand is not scalar, complex, or memref, return an error
      if (!isa<MemRefType>(valueType)) {
        if (!abi::isScalarArgumentType(valueType) &&
            !isa<ComplexType>(valueType))
          return emitError(returnOp->getLoc())
                 << "value type must be scalar, complex, or memref type";
        if (!value.getDefiningOp<executor::ABISendOp>())
          rewriter.create<ABISendOp>(returnOp->getLoc(), value, ptr);
        continue;
      }

      if (requiresClone[idx]) {
        FailureOr<Value> newValue =
            reallocMemRefValue(rewriter, value, cast<MemRefType>(valueType));
        if (failed(newValue))
          return failure();
        value = *newValue;
      }

      // Get the ABI attribute for this argument
      executor::ArgumentABIAttr abiAttr =
          executor::abi::getArgumentABIAttr(func, ptr);
      assert(abiAttr && "expected ABI attribute");

      // Check if the argument has 'undef' parameter set
      const bool hasUndefParam = abiAttr.getUndef();
      if (hasUndefParam) {
        Value trueValue = rewriter.create<arith::ConstantOp>(
            returnOp->getLoc(), rewriter.getBoolAttr(true));
        rewriter.create<ABISendOp>(returnOp->getLoc(), value, ptr, trueValue);
      }
    }

    returnOp->setOperands(ValueRange{});
  }
  llvm::BitVector resultIndices(
      cast<FunctionType>(func.getFunctionType()).getNumResults(), true);
  if (failed(func.eraseResults(resultIndices)))
    return failure();

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
