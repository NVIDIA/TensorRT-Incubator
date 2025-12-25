//===- CastPatterns.cpp  -----------------------------------------------===//
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
// Where noted below, some code is reproduced from upstream Stablehlo
// "aggressive folder" patterns. They can be removed once we fix upstream and
// can directly use those patterns. The copyright/license is reproduced below:
//
// Copyright 2024 The StableHLO Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of simplification patterns related to tensor.cast and
/// stablehlo operations.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Patterns.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::stablehlo;

namespace {

/// Fold `stablehlo.op(..., tensor.cast(x)... )` to `stablehlo.op(..., x, ...)`
/// if the cast is a generalizing cast (it is removing some static dims of the
/// type of  `x` and replacing them with dynamic dimensions).
struct AbsorbTensorCastProducer : public RewritePattern {
  AbsorbTensorCastProducer(MLIRContext *ctx, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag{}, benefit, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa_and_present<stablehlo::StablehloDialect>(op->getDialect()) ||
        // Composite op types cannot be refined in-place.
        isa<stablehlo::CompositeOp>(op))
      return failure();

    // For each operand, try to absorb the cast operation. For most StableHLO
    // ops, this is legal, but for some operations that have additional
    // constraints, the legality of this depends on which operand is being
    // refined.
    auto hasGeneralizingCast = [](OpOperand &operand) -> tensor::CastOp {
      if (!canUpdateTypeWithoutCast(operand))
        return nullptr;
      Value value = operand.get();
      // Not all stablehlo operands are tensors -- some can have types like
      // 'tuple' or special quantized types.
      auto rtt = dyn_cast<RankedTensorType>(value.getType());
      if (!rtt)
        return nullptr;
      auto castOp = value.getDefiningOp<tensor::CastOp>();
      if (!castOp)
        return nullptr;
      auto operandType =
          dyn_cast<RankedTensorType>(castOp.getOperand().getType());
      if (castOp && operandType &&
          tensorrt::isTargetRefinementOfSource(rtt.getShape(),
                                               operandType.getShape()))
        return castOp;
      return nullptr;
    };
    bool changed = false;
    SmallVector<Value> newInputs;
    for (OpOperand &v : op->getOpOperands()) {
      auto castOp = hasGeneralizingCast(v);
      changed |= castOp != nullptr;
      newInputs.push_back(castOp ? castOp.getOperand() : v.get());
    }
    if (!changed)
      return failure();
    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(newInputs); });
    return success();
  }
};

/// This patterns inserts a tensor.cast between a returned Value if the Value's
/// tensor shape does not match the corresponding function result type. This can
/// happen if the upstream StableHlo aggressive simplification pass does not
/// properly insert casts where it is supposed to. This is a relatively common
/// mistake in the upstream pass which will result in a verification error at
/// the end of the pass if left unhandled. Therefore, we insert this pattern to
/// automatically insert required casts if possible until upstream figures out
/// how to avoid this mistake from occurring.
struct FixInvalidReturnWorkaround : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp)
      return failure();
    auto funcType = funcOp.getFunctionType();
    bool changed = false;
    SmallVector<Value> newOperands(op.getOperands());
    for (auto [idx, value] : llvm::enumerate(op.getOperands())) {
      if (funcType.getResult(idx) == value.getType())
        continue;
      // If it is not a tensor type, then something else has gone horribly
      // wrong, we can't fix it.
      auto tensorType = dyn_cast<RankedTensorType>(value.getType());
      auto desiredType = dyn_cast<RankedTensorType>(funcType.getResult(idx));
      if (!tensorType || !desiredType)
        continue;

      // Check for cast compatibility. If not cast compatible, then something
      // else has gone horribly wrong, we can't fix it.
      if (!tensor::CastOp::areCastCompatible(tensorType, desiredType))
        continue;

      auto castOp =
          rewriter.create<tensor::CastOp>(value.getLoc(), desiredType, value);
      newOperands[idx] = castOp;
      changed = true;
    }
    if (!changed)
      return failure();
    rewriter.modifyOpInPlace(
        op, [&]() { op.getOperandsMutable().assign(newOperands); });
    return success(changed);
  }
};

} // namespace

void stablehlo_ext::populateStableHloAbsorbTensorCastPatterns(
    RewritePatternSet &patterns) {
  patterns.add<AbsorbTensorCastProducer, FixInvalidReturnWorkaround>(
      patterns.getContext());
  tensor::CastOp::getCanonicalizationPatterns(patterns, patterns.getContext());
}
