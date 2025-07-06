//===- OptimizeMemorySpaces.cpp -------------------------------------------===//
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
///  Implementation of the `plan-optimize-memory-spaces` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::plan {
#define GEN_PASS_DEF_PLANOPTIMIZEMEMORYSPACESPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

constexpr int64_t kConstantDuplicationLimit = 1024;

using namespace mlir;
using namespace mlir::plan;

/// Returns true if the tensor type has a host-visible memory space encoding.
static bool isHostVisible(Value v) {
  auto rtt = dyn_cast<RankedTensorType>(v.getType());
  if (!rtt)
    return false;
  if (auto space = dyn_cast_or_null<MemorySpaceAttr>(rtt.getEncoding()))
    return space.isHostVisible();
  return false;
}

/// Return true if the op is likely in a "compute" region, like the region of
/// `stablehlo.reduce` or `linalg.generic`. For the purposes of this pass, we're
/// defining "compute" region as a region where the normal flow-of-control does
/// not enter from outside. It's only used to define the semantics of the parent
/// operation.a
static bool inComputeRegion(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    // If the parent is a function, then we're in a normal region.
    if (isa<FunctionOpInterface>(parent))
      return false;
    // We are in a region which is not a control flow region and not a function,
    // so it's probably a "compute" region.
    if (!isa<RegionBranchOpInterface>(parent))
      return true;
    // If we're in a control flow region, we may still be nested in a "compute"
    // region. E.g. `scf.if` is allowed in `linalg.generic` region. Keep going
    // up to find the parent function.
    parent = parent->getParentOp();
  }
  return false;
}

namespace {

/// Use an explicit host-visible staging tensor to materialie the
/// 'from_elements' before casting it to the non-host-visible space.
struct FixUpFromElements : public OpRewritePattern<tensor::FromElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    auto space = dyn_cast_or_null<MemorySpaceAttr>(op.getType().getEncoding());
    if (!space || space.isHostVisible())
      return rewriter.notifyMatchFailure(
          op, "skipping no encoding or already host-visible");

    RankedTensorType originalType = op.getType();
    RankedTensorType newType = RankedTensorType::get(
        originalType.getShape(), originalType.getElementType(),
        MemorySpaceAttr::get(originalType.getContext(),
                             plan::MemorySpace::host));
    auto newOp = rewriter.create<tensor::FromElementsOp>(op.getLoc(), newType,
                                                         op.getElements());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, originalType, newOp);
    return success();
  }
};

/// Absorb cast operations into the while loop 'before' and 'after' regions and
/// result types as long as region argument types are changed to the same new
/// type.
struct SCFWhileAbsorbCastPattern : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> iterArgsToUpdate;
    Region &after = op.getAfter();
    Region &before = op.getBefore();
    auto originalCond = cast<scf::ConditionOp>(before.front().getTerminator());
    auto originalYield = cast<scf::YieldOp>(after.front().getTerminator());

    if (after.getArgumentTypes() != before.getArgumentTypes())
      return failure();

    SmallVector<Value> newCondOperands(originalCond.getArgs());
    SmallVector<Value> newOperands(op.getOperands());
    SmallVector<Value> newYieldOperands(originalYield.getOperands());
    SmallVector<std::pair<unsigned, Type>> afterBlockTypeUpdates;
    SmallVector<std::pair<unsigned, Type>> beforeBlockTypeUpdates;
    bool hasUpdate = false;
    for (auto [afterArg, beforeArg] :
         llvm::zip_equal(after.getArguments(), before.getArguments())) {
      // After/before have same type as checked above.
      auto tensorType = dyn_cast<RankedTensorType>(afterArg.getType());
      if (!tensorType)
        continue;
      Value condOperand = originalCond.getArgs()[afterArg.getArgNumber()];
      Value aboveOperand = op.getOperand(beforeArg.getArgNumber());
      Value yieldOperand =
          originalYield.getOperands()[beforeArg.getArgNumber()];
      Value result = op.getResult(afterArg.getArgNumber());
      if (!result.hasOneUse())
        continue;
      auto aboveCast = aboveOperand.getDefiningOp<tensor::CastOp>();
      auto yieldCast = yieldOperand.getDefiningOp<tensor::CastOp>();
      if (!aboveCast || !yieldCast ||
          aboveCast.getOperand().getType() != yieldCast.getOperand().getType())
        continue;
      auto condCast = condOperand.getDefiningOp<tensor::CastOp>();
      auto resultCast = dyn_cast<tensor::CastOp>(*result.user_begin());
      if (!condCast || !resultCast ||
          condCast.getOperand().getType() != resultCast.getType())
        continue;

      // We must be changing the region arg types to the same new type.
      if (aboveCast.getOperand().getType() != condCast.getOperand().getType())
        continue;

      newOperands[beforeArg.getArgNumber()] = aboveCast.getOperand();
      newYieldOperands[beforeArg.getArgNumber()] = yieldCast.getOperand();
      beforeBlockTypeUpdates.emplace_back(beforeArg.getArgNumber(),
                                          aboveCast.getOperand().getType());

      newCondOperands[afterArg.getArgNumber()] = condCast.getOperand();
      afterBlockTypeUpdates.emplace_back(afterArg.getArgNumber(),
                                         condCast.getOperand().getType());
      hasUpdate = true;
    }
    if (!hasUpdate)
      return failure();

    rewriter.modifyOpInPlace(op, [&]() {
      for (auto [argNumber, type] : afterBlockTypeUpdates) {
        after.getArgument(argNumber).setType(type);
        op.getResult(argNumber).setType(type);
      }
    });
    rewriter.modifyOpInPlace(originalCond, [&]() {
      originalCond.getArgsMutable().assign(newCondOperands);
    });

    // Modifications for the "before" region.
    rewriter.modifyOpInPlace(op, [&]() {
      op.getInitsMutable().assign(newOperands);
      for (auto [argNumber, type] : beforeBlockTypeUpdates)
        before.getArgument(argNumber).setType(type);
    });
    rewriter.modifyOpInPlace(originalYield, [&]() {
      originalYield.getResultsMutable().assign(newYieldOperands);
    });
    return success();
  }
};

/// Absorb cast operations into the iteration carried arguments of `scf.for`.
struct SCFForAbsorbCastPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> iterArgsToUpdate;
    Block *body = op.getBody();
    auto originalYield = cast<scf::YieldOp>(body->getTerminator());

    SmallVector<Value> newOperands(op.getInitArgs());
    SmallVector<Value> newYieldOperands(originalYield.getOperands());
    SmallVector<std::pair<unsigned, Type>> blockTypeUpdates;
    bool hasUpdate = false;
    for (auto [iterArgIdx, arg] : llvm::enumerate(op.getRegionIterArgs())) {
      if (!arg.hasOneUse())
        continue;
      auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
      if (!tensorType)
        continue;
      auto argCast = dyn_cast<tensor::CastOp>(*arg.user_begin());
      if (!argCast)
        continue;
      Value yieldOperand = originalYield.getOperands()[iterArgIdx];
      auto yieldCast = yieldOperand.getDefiningOp<tensor::CastOp>();
      if (!yieldCast || argCast.getType() != yieldCast.getOperand().getType())
        continue;
      Value newToOperand = rewriter.create<tensor::CastOp>(
          op.getLoc(), argCast.getType(), op.getInitArgs()[iterArgIdx]);
      newOperands[iterArgIdx] = newToOperand;
      newYieldOperands[iterArgIdx] = yieldCast.getOperand();
      blockTypeUpdates.emplace_back(iterArgIdx, argCast.getType());
      hasUpdate = true;
    }
    if (!hasUpdate)
      return failure();

    rewriter.setInsertionPointAfter(op);
    for (auto [iterArgIdx, type] : blockTypeUpdates) {
      Type originalType = op.getResultTypes()[iterArgIdx];
      auto castOp = rewriter.create<tensor::CastOp>(op.getLoc(), originalType,
                                                    op.getResult(iterArgIdx));
      rewriter.replaceAllUsesExcept(op.getResult(iterArgIdx), castOp, castOp);
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.getInitArgsMutable().assign(newOperands);
      for (auto [iterArgIdx, type] : blockTypeUpdates) {
        op.getRegionIterArg(iterArgIdx).setType(type);
        op->getResult(iterArgIdx).setType(type);
      }
    });
    rewriter.modifyOpInPlace(originalYield, [&]() {
      originalYield.getResultsMutable().assign(newYieldOperands);
    });

    return success();
  }
};

/// For any 'shape' parameter of a 'tensor.reshape', ensure that the shape is
/// host-visible.
struct ReshapeAbsorbDeviceCast : public OpRewritePattern<tensor::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Skip past any explicit host-device transfers or host<->host-pinned
    // transfers
    if (auto matOp =
            op.getShape()
                .getDefiningOp<bufferization::MaterializeInDestinationOp>()) {
      auto source = matOp.getSource();
      if (isHostVisible(source)) {
        rewriter.modifyOpInPlace(
            op, [&]() { op.getShapeMutable().assign(source); });
        return success();
      }
    }
    if (auto castOp = op.getShape().getDefiningOp<tensor::CastOp>()) {
      auto source = castOp.getOperand();
      if (isHostVisible(source)) {
        rewriter.modifyOpInPlace(
            op, [&]() { op.getShapeMutable().assign(source); });
        return success();
      }
    }
    // Don't insert explicit cast if the shape is already host-visible.
    if (isHostVisible(op.getShape()))
      return rewriter.notifyMatchFailure(op, "skipping already host-visible");
    // Otherwise, insert a direct cast-to-host.
    auto castOp = rewriter.create<tensor::CastOp>(
        op.getLoc(),
        op.getShape().getType().cloneWithEncoding(plan::MemorySpaceAttr::get(
            op->getContext(), plan::MemorySpace::host)),
        op.getShape());
    rewriter.modifyOpInPlace(op,
                             [&]() { op.getShapeMutable().assign(castOp); });
    return success();
  }
};

// Abosrb `tensor.cast` into `bufferization.alloc_tensor` (with no copy
// operand).
struct AllocTensorAbsorbCastPattern : public OpRewritePattern<tensor::CastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::CastOp op,
                                PatternRewriter &rewriter) const override {
    auto source = op.getOperand();
    auto allocOp = source.getDefiningOp<bufferization::AllocTensorOp>();
    if (!allocOp || allocOp.getCopy())
      return failure();
    auto allocType = allocOp.getType();
    auto allocMemorySpace =
        llvm::dyn_cast_if_present<MemorySpaceAttr>(allocType.getEncoding());
    if (!allocMemorySpace)
      return failure();
    auto castType = dyn_cast<RankedTensorType>(op.getType());
    if (!castType)
      return failure();
    auto castMemorySpace =
        llvm::dyn_cast_if_present<MemorySpaceAttr>(castType.getEncoding());
    if (!castMemorySpace)
      return failure();
    if (castType.getShape() != allocType.getShape())
      return failure();
    rewriter.replaceOpWithNewOp<bufferization::AllocTensorOp>(
        op, op.getType(), allocOp.getDynamicSizes(), /*copy=*/Value{},
        /*size_hint=*/Value{},
        /*memory_space=*/castMemorySpace);
    return success();
  }
};

/// Absorb `tensor.cast` into `arith.constant` producer by folding in-place or
/// duplicating the  constant (within limits).
struct ConstantAbsorbCastPattern : public OpRewritePattern<tensor::CastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::CastOp op,
                                PatternRewriter &rewriter) const override {
    auto source = op.getOperand();
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "skipping non-ranked-tensor type");
    auto constantOp = source.getDefiningOp<arith::ConstantOp>();
    if (!constantOp)
      return rewriter.notifyMatchFailure(op, "skipping non-constant source");

    bool canUpdateInPlace = constantOp->hasOneUse();
    if (!canUpdateInPlace) {
      if (resultType.getNumElements() > kConstantDuplicationLimit)
        return rewriter.notifyMatchFailure(
            op, "skipping constant with too many elements and multiple users");
    }

    auto constantType = dyn_cast<RankedTensorType>(constantOp.getType());
    if (!constantType)
      return rewriter.notifyMatchFailure(op, "skipping non-ranked-tensor type");
    auto constantMemorySpace =
        llvm::dyn_cast_if_present<MemorySpaceAttr>(constantType.getEncoding());
    if (!constantMemorySpace)
      return rewriter.notifyMatchFailure(op, "skipping non-host-visible");

    ElementsAttr newAttr{};
    if (auto elementsAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue()))
      newAttr = elementsAttr.reshape(resultType);
    if (auto resourceAttr =
            dyn_cast<DenseResourceElementsAttr>(constantOp.getValue())) {
      DenseResourceElementsHandle handle = resourceAttr.getRawHandle();
      newAttr = DenseResourceElementsAttr::get(resultType, handle);
    }
    if (!newAttr)
      return rewriter.notifyMatchFailure(op, "unhandled attribute value type");

    // Modify the constant op in-place if possible.
    if (canUpdateInPlace) {
      rewriter.modifyOpInPlace(constantOp, [&]() {
        constantOp.setValueAttr(newAttr);
        constantOp.getResult().setType(resultType);
      });
      rewriter.replaceOp(op, constantOp);
      return success();
    }

    // Otherwise, duplicate the constant using the new encoding type.
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, newAttr);
    return success();
  }
};

/// Rewrite `memref.load` that acts on device memory to first copy the buffer to
/// the host and load from the host buffer.
struct TensorDeviceExtractRewriter
    : public OpRewritePattern<tensor::ExtractOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto source = op.getTensor();
    if (isHostVisible(source))
      return failure();

    if (inComputeRegion(op))
      return failure();

    // First check if there is an existing `tensor.cast` which can be absorbed.
    if (auto castOp = source.getDefiningOp<tensor::CastOp>()) {
      if (isHostVisible(castOp.getOperand())) {
        rewriter.modifyOpInPlace(
            op, [&]() { op.getTensorMutable().assign(castOp.getOperand()); });
        return success();
      }
    }

    rewriter.setInsertionPointAfterValue(source);
    Value hostTensor = rewriter.create<tensor::CastOp>(
        op.getLoc(),
        RankedTensorType::get(source.getType().getShape(),
                              source.getType().getElementType(),
                              plan::MemorySpaceAttr::get(
                                  op->getContext(), plan::MemorySpace::host)),
        source);

    rewriter.replaceUsesWithIf(op.getTensor(), hostTensor, [&](OpOperand &use) {
      return isa<tensor::ExtractOp>(use.getOwner());
    });

    return success();
  }
};

/// Rewrite `tensor.insert` so that the insertion destination tensor has
/// 'host_pinned' space.
struct DeviceInsertRewriter : public OpRewritePattern<tensor::InsertOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertOp op,
                                PatternRewriter &rewriter) const override {
    auto dest = op.getDest();
    if (isHostVisible(dest))
      return failure();
    if (inComputeRegion(op))
      return failure();
    rewriter.setInsertionPointAfterValue(dest);
    auto newType = RankedTensorType::get(
        dest.getType().getShape(), dest.getType().getElementType(),
        plan::MemorySpaceAttr::get(op->getContext(), plan::MemorySpace::host));
    Value hostTensor =
        rewriter.create<tensor::CastOp>(op.getLoc(), newType, dest);
    rewriter.replaceUsesWithIf(op.getDest(), hostTensor, [&](OpOperand &use) {
      return isa<tensor::InsertOp>(use.getOwner());
    });
    Type originalType = op.getType();
    rewriter.modifyOpInPlace(op, [&]() { op.getResult().setType(newType); });
    rewriter.setInsertionPointAfter(op);
    auto castBack = rewriter.create<tensor::CastOp>(op.getLoc(), originalType,
                                                    op.getResult());
    rewriter.replaceAllUsesExcept(op.getResult(), castBack, castBack);
    return success();
  }
};

/// 'tensor.from_elements' is not a DPS operation, so if we yield it from
/// a loop, the result of bufferization will always be to create and yield a new
/// allocation from the loop, which is highly sub-optimal. This pattern matches
/// any `tensor.from_elements` operation which is being yielded from a loop
/// region. It rewrites it to have an explicit
/// `bufferization.materialize_in_destination` operation to materialize the
/// result into a empty tensor. The advantage of this is that the empty tensor
/// can be bufferized into a memref which is allocated above the loop and
/// doesn't change between iterations.
///
/// Note that you could also use `tensor.insert` to assemble the result, but the
/// BufferizableOpInterface implementation for `tensor.insert` is suboptimal.
struct FromElementsYieldFromLoopPattern
    : public OpRewritePattern<tensor::FromElementsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse())
      return failure();
    auto user = *op->user_begin();
    if (!user->hasTrait<OpTrait::IsTerminator>())
      return failure();

    // This can actually make things worse if the `scf.while` operation's
    // before/after args don't have match.
    auto parentOp = op->getParentOp();
    if (!isa<scf::WhileOp, scf::ForOp>(parentOp))
      return failure();
    if (auto whileOp = dyn_cast_if_present<scf::WhileOp>(parentOp)) {
      Region *before = &whileOp.getBefore();
      Region *after = &whileOp.getAfter();
      if (after->getArgumentTypes() != before->getArgumentTypes())
        return failure();
    }

    // Check to make sure this has a 'host' space, otherwise it will conflict
    // with the FixupFromElements pattern. This pattern doesn't matter if there
    // is a cast between the `from_elements` and the `yield` since an explicit
    // materialization will be applied t here later as well.
    auto space = dyn_cast_or_null<MemorySpaceAttr>(op.getType().getEncoding());
    if (!space || !space.isHostVisible())
      return failure();

    rewriter.setInsertionPointAfter(op);

    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), op.getType().getShape(), op.getType().getElementType(),
        op.getType().getEncoding());
    auto matOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        op.getLoc(), op.getType(), op.getResult(), emptyOp);

    rewriter.replaceAllUsesExcept(op.getResult(), matOp.getResult(), matOp);
    return success();
  }
};

struct OptimizeMemorySpacesPass
    : public plan::impl::PlanOptimizeMemorySpacesPassBase<
          OptimizeMemorySpacesPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    RewritePatternSet patterns(&getContext());
    tensor::CastOp::getCanonicalizationPatterns(patterns, &getContext());
    // clang-format off
    patterns.insert<
      AllocTensorAbsorbCastPattern,
      ConstantAbsorbCastPattern,
      DeviceInsertRewriter,
      FixUpFromElements,
      FromElementsYieldFromLoopPattern,
      ReshapeAbsorbDeviceCast,
      SCFForAbsorbCastPattern,
      SCFWhileAbsorbCastPattern,
      TensorDeviceExtractRewriter
    >(&getContext());
    // clang-format on

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      emitError(func.getLoc()) << "failed to run " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
