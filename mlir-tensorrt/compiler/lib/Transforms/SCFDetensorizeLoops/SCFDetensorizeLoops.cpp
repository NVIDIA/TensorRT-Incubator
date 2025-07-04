//===- SCFDetensorizeLoops.cpp --------------------------------------------===//
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
/// Implementation of `scf-detensorize-loops` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Transforms.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir {
#define GEN_PASS_DEF_SCFDETENSORIZELOOPSPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

static bool isHostTensor(Value value, const DataFlowSolver &solver) {
  const TensorKindLattice *lattice =
      solver.lookupState<TensorKindLattice>(value);
  assert(lattice && "expected valid lattice point");
  if (lattice->getValue().isUninitialized())
    return false;
  return lattice->getValue().isHostOnly();
}

namespace {
/// Absorb cast operations into the while loop 'before' region and init types.
struct WhileScalarizeBeforeArgPattern : public OpRewritePattern<scf::WhileOp> {
  WhileScalarizeBeforeArgPattern(
      MLIRContext *ctx,
      ShouldScalarizeWhileBeforeArgFunc shouldScalarizeBeforeArg,
      PatternBenefit benefit)
      : OpRewritePattern(ctx, benefit),
        shouldScalarizeBeforeArg(std::move(shouldScalarizeBeforeArg)) {}

  ShouldScalarizeWhileBeforeArgFunc shouldScalarizeBeforeArg;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> iterArgsToUpdate;
    Region &after = op.getAfter();
    Region &before = op.getBefore();
    auto originalYield = cast<scf::YieldOp>(after.front().getTerminator());

    SmallVector<Value> newOperands(op.getOperands());
    SmallVector<Value> newYieldOperands(originalYield.getOperands());
    SmallVector<std::pair<unsigned, Type>> blockTypeUpdates;
    bool hasUpdate = false;
    for (BlockArgument arg : before.getArguments()) {
      auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
      if (!tensorType || !tensorType.hasStaticShape() ||
          tensorType.getNumElements() != 1)
        continue;
      Value aboveOperand = op.getOperand(arg.getArgNumber());
      Value yieldOperand = originalYield.getOperands()[arg.getArgNumber()];
      if (!shouldScalarizeBeforeArg(arg, aboveOperand, yieldOperand))
        continue;
      rewriter.setInsertionPoint(op);
      newOperands[arg.getArgNumber()] = rewriter.create<tensor::ExtractOp>(
          op.getLoc(), aboveOperand,
          SmallVector<Value>(
              tensorType.getRank(),
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0)));
      rewriter.setInsertionPoint(originalYield);
      newYieldOperands[arg.getArgNumber()] = rewriter.create<tensor::ExtractOp>(
          op.getLoc(), yieldOperand,
          SmallVector<Value>(
              tensorType.getRank(),
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0)));
      blockTypeUpdates.emplace_back(arg.getArgNumber(),
                                    tensorType.getElementType());
      hasUpdate = true;
    }
    if (!hasUpdate)
      return failure();

    rewriter.setInsertionPointToStart(&before.front());
    for (auto [argNumber, type] : blockTypeUpdates) {
      Type originalType = before.getArgument(argNumber).getType();
      auto fromElements = rewriter.create<tensor::FromElementsOp>(
          op.getLoc(), originalType, before.getArgument(argNumber));
      rewriter.replaceAllUsesExcept(before.getArgument(argNumber), fromElements,
                                    fromElements);
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.getInitsMutable().assign(newOperands);
      for (auto [argNumber, type] : blockTypeUpdates)
        before.getArgument(argNumber).setType(type);
    });

    rewriter.setInsertionPointToStart(&before.front());
    rewriter.modifyOpInPlace(originalYield, [&]() {
      originalYield.getResultsMutable().assign(newYieldOperands);
    });
    return success();
  }
};

/// Absorb cast operations into the while loop 'after' region and result types.
struct WhileScalarizeAfterArgPattern : public OpRewritePattern<scf::WhileOp> {
  WhileScalarizeAfterArgPattern(
      MLIRContext *ctx,
      ShouldScalarizeWhileAfterArgFunc shouldScalarizeAfterArg,
      PatternBenefit benefit)
      : OpRewritePattern(ctx, benefit),
        shouldScalarizeAfterArg(std::move(shouldScalarizeAfterArg)) {}

  ShouldScalarizeWhileAfterArgFunc shouldScalarizeAfterArg;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> iterArgsToUpdate;
    Region &after = op.getAfter();
    Region &before = op.getBefore();
    auto originalCond = cast<scf::ConditionOp>(before.front().getTerminator());

    SmallVector<Value> newCondOperands(originalCond.getArgs());
    SmallVector<std::pair<unsigned, Type>> blockTypeUpdates;
    bool hasUpdate = false;
    for (BlockArgument arg : after.getArguments()) {
      auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
      if (!tensorType || !tensorType.hasStaticShape() ||
          tensorType.getNumElements() != 1)
        continue;
      Value condOperand = originalCond.getArgs()[arg.getArgNumber()];
      Value result = op.getResult(arg.getArgNumber());
      if (!shouldScalarizeAfterArg(arg, condOperand, result))
        continue;
      rewriter.setInsertionPoint(originalCond);
      newCondOperands[arg.getArgNumber()] = rewriter.create<tensor::ExtractOp>(
          op.getLoc(), condOperand,
          SmallVector<Value>(
              tensorType.getRank(),
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0)));
      blockTypeUpdates.emplace_back(arg.getArgNumber(),
                                    tensorType.getElementType());
      hasUpdate = true;
    }
    if (!hasUpdate)
      return failure();

    for (auto [argNumber, type] : blockTypeUpdates) {
      rewriter.setInsertionPointToStart(&after.front());
      Type originalType = op.getResult(argNumber).getType();
      auto fromElements = rewriter.create<tensor::FromElementsOp>(
          op.getLoc(), originalType, after.getArgument(argNumber));
      rewriter.replaceAllUsesExcept(after.getArgument(argNumber), fromElements,
                                    fromElements);

      rewriter.setInsertionPointAfter(op);
      auto fromElements2 = rewriter.create<tensor::FromElementsOp>(
          op.getLoc(), originalType, op.getResult(argNumber));
      rewriter.replaceAllUsesExcept(op.getResult(argNumber), fromElements2,
                                    fromElements2);
    }

    rewriter.modifyOpInPlace(op, [&]() {
      for (auto [argNumber, type] : blockTypeUpdates) {
        after.getArgument(argNumber).setType(type);
        op.getResult(argNumber).setType(type);
      }
    });
    rewriter.modifyOpInPlace(originalCond, [&]() {
      originalCond.getArgsMutable().assign(newCondOperands);
    });
    return success();
  }
};
} // namespace

static bool defaultShouldScalarizeBeforeArg(BlockArgument arg,
                                            Value initOperand,
                                            Value yieldOperand) {
  RankedTensorType type = cast<RankedTensorType>(arg.getType());
  return isa<IntegerType, IndexType>(type.getElementType()) &&
         (initOperand.getDefiningOp<tensor::FromElementsOp>() ||
          matchPattern(initOperand, m_Constant())) &&
         (yieldOperand.getDefiningOp<tensor::FromElementsOp>() ||
          matchPattern(yieldOperand, m_Constant()));
}

static bool defaultShouldScalarizeAfterArg(BlockArgument arg, Value condOperand,
                                           Value result) {
  RankedTensorType type = cast<RankedTensorType>(arg.getType());
  return isa<IntegerType, IndexType>(type.getElementType()) &&
         result.hasOneUse() &&
         (isa<BlockArgument>(condOperand) ||
          condOperand.getDefiningOp<tensor::FromElementsOp>());
}

void mlir::populateSCFDetensorizeWhilePatterns(
    RewritePatternSet &patterns,
    ShouldScalarizeWhileBeforeArgFunc shouldScalarizeBeforeArg,
    ShouldScalarizeWhileAfterArgFunc shouldScalarizeAfterArg,
    PatternBenefit benefit) {
  if (!shouldScalarizeBeforeArg)
    shouldScalarizeBeforeArg = defaultShouldScalarizeBeforeArg;
  if (!shouldScalarizeAfterArg)
    shouldScalarizeAfterArg = defaultShouldScalarizeAfterArg;
  patterns.add<WhileScalarizeBeforeArgPattern>(
      patterns.getContext(), shouldScalarizeBeforeArg, benefit);
  patterns.add<WhileScalarizeAfterArgPattern>(patterns.getContext(),
                                              shouldScalarizeAfterArg, benefit);
  scf::ForOp::getCanonicalizationPatterns(patterns, patterns.getContext());
  scf::WhileOp::getCanonicalizationPatterns(patterns, patterns.getContext());
}

namespace {

struct TensorExtractSubsetExtractionOpInterface
    : public SubsetExtractionOpInterface::ExternalModel<
          TensorExtractSubsetExtractionOpInterface, tensor::ExtractOp> {
  OpOperand &getSourceOperand(Operation *op) const {
    return cast<tensor::ExtractOp>(op).getTensorMutable();
  }
};

template <typename OpTy>
struct SubsetOpInterfaceImpl
    : public SubsetOpInterface::ExternalModel<SubsetOpInterfaceImpl<OpTy>,
                                              OpTy> {
  FailureOr<HyperrectangularSlice>
  getAccessedHyperrectangularSlice(Operation *op) const {
    SmallVector<OpFoldResult> indices = cast<OpTy>(op).getIndices();
    SmallVector<OpFoldResult> ones(
        indices.size(), IntegerAttr::get(IndexType::get(op->getContext()), 1));
    return HyperrectangularSlice(indices, ones, ones);
  }
};

struct TensorInsertSubsetInsertionOpInterface
    : public SubsetInsertionOpInterface::ExternalModel<
          TensorInsertSubsetInsertionOpInterface, tensor::InsertOp> {

  OpOperand &getSourceOperand(Operation *op) const {
    return cast<tensor::InsertOp>(op).getScalarMutable();
  }

  OpOperand &getDestinationOperand(Operation *op) const {
    return cast<tensor::InsertOp>(op).getDestMutable();
  }

  OpResult getUpdatedDestination(Operation *op) const {
    return cast<tensor::InsertOp>(op)->getOpResult(0);
  }

  SmallVector<Value>
  getValuesNeededToBuildSubsetExtraction(Operation *op) const {
    return {cast<tensor::InsertOp>(op).getDest()};
  }

  Value buildSubsetExtraction(Operation *op, OpBuilder &builder,
                              Location loc) const {
    return Value{};
  }
};

class SCFDetensorizeLoopsPass
    : public impl::SCFDetensorizeLoopsPassBase<SCFDetensorizeLoopsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    Operation *op = getOperation();

    // Walk through all loops in a function in innermost-loop-first order. This
    // way, we first hoist from the inner loop, and place the ops in the outer
    // loop, which in turn can be further hoisted from.
    IRRewriter rewriter(ctx);
    op->walk([&](LoopLikeOpInterface loopLike) {
      loopLike = mlir::hoistLoopInvariantSubsets(rewriter, loopLike);
    });

    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TensorKindAnalysis>(symbolTable);
    if (failed(solver.initializeAndRun(op))) {
      emitError(op->getLoc()) << "failed to run TensorKindAnalysis";
      return signalPassFailure();
    }
    auto shouldScalarizeBeforeArg = [&](BlockArgument arg, Value initOperand,
                                        Value yieldOperand) {
      return isHostTensor(arg, solver);
    };
    auto shouldScalarizeAfterArg = [&](BlockArgument arg, Value condOperand,
                                       Value result) {
      return isHostTensor(arg, solver);
    };

    populateSCFDetensorizeWhilePatterns(patterns, shouldScalarizeBeforeArg,
                                        shouldScalarizeAfterArg,
                                        /*benefit=*/1);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);

    registry.addExtension(+[](MLIRContext *ctx,
                              tensor::TensorDialect *dialect) {
      tensor::ExtractOp::attachInterface<
          SubsetOpInterfaceImpl<tensor::ExtractOp>>(*ctx);
      tensor::InsertOp::attachInterface<
          SubsetOpInterfaceImpl<tensor::InsertOp>>(*ctx);
      tensor::ExtractOp::attachInterface<
          TensorExtractSubsetExtractionOpInterface>(*ctx);
      tensor::InsertOp::attachInterface<TensorInsertSubsetInsertionOpInterface>(
          *ctx);
    });
  }
};
} // namespace
