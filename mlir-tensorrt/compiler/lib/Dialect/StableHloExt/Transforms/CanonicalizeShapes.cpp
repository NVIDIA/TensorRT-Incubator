//===- CanonicalizeShapes.cpp ---------------------------------------------===//
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
/// Implementation of "stablehlo-ext-canonicalize-shapes".
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/StablehloRefineShapes.h"

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_CANONICALIZESHAPESPASS
#define GEN_PASS_DEF_REFINESHAPESPASS
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

using namespace mlir;
using namespace mlir::stablehlo_ext;

namespace {

/// Upstream StableHLO pattern uses `builtin.unrealized_cast` while we prefer to
/// use `tensor.cast` due to better verification. Other passes may create such
/// `tensor.cast`, this pattern absorbs them into function return and refines
/// the function type.
struct AbsorbCastsIntoFuncReturnPattern
    : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    bool needsUpdate = false;
    SmallVector<Value> newOperands(op->getOperands());
    for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
      auto castOp = dyn_cast_or_null<tensor::CastOp>(operand.getDefiningOp());
      if (!castOp || !isa<RankedTensorType>(castOp.getType()) ||
          !isa<RankedTensorType>(castOp.getSource().getType()) ||
          !tensorrt::isTargetRefinementOfSource(
              castOp.getType().getShape(),
              castOp.getSource().getType().getShape()))
        continue;
      newOperands[i] = castOp.getSource();
      needsUpdate = true;
    }
    if (!needsUpdate)
      return failure();

    rewriter.modifyOpInPlace(
        op, [&]() { op.getOperandsMutable().assign(newOperands); });

    // If the type of the enclosing `func.func` needs an update, we simply
    // call setType. We can afford this simplicity because our algorithm
    // currently supports only one function per module.

    auto func = cast<func::FuncOp>(op->getParentOp());
    rewriter.modifyOpInPlace(func, [&]() {
      func.setType(rewriter.getFunctionType(func.getArgumentTypes(),
                                            TypeRange(newOperands)));
    });
    return success();
  }
};

/// Implementation of `stablehlo-ext-refine-shapes`.
class RefineShapesPass
    : public stablehlo_ext::impl::RefineShapesPassBase<RefineShapesPass> {
public:
  using Base::Base;

  LogicalResult initialize(MLIRContext *ctx) override {

    RewritePatternSet patterns_(ctx);
    stablehlo::populateStablehloRefineShapesPatterns(&patterns_, ctx);
    stablehlo::populateStablehloShapeFolderPatterns(&patterns_, ctx);
    patterns_.add<AbsorbCastsIntoFuncReturnPattern>(ctx);
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    // We don't consider failure to converge here an error.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Aggressive;
    config.maxIterations = 4;
    config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
    config.strictMode = GreedyRewriteStrictness::AnyOp;

    Operation *root = getOperation();
    if (failed(applyPatternsAndFoldGreedily(root, patterns, config)))
      emitWarning(root->getLoc()) << getArgument() << " failed to converge in "
                                  << config.maxIterations << " iterations";
  }

private:
  FrozenRewritePatternSet patterns;
};
} // namespace

/// Populate the pipeline that will be dynamically run iteratively during
/// `stablehlo-ext-canonicalize-shapes`.
static void buildShapeCanonicalizationPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloCanonicalizeDynamismPass());
  pm.addPass(stablehlo_ext::createRefineShapesPass());
}

namespace {
/// Implementation of `stablehlo-ext-canonicalize-shapes`.
class CanonicalizeShapesPass
    : public stablehlo_ext::impl::CanonicalizeShapesPassBase<
          CanonicalizeShapesPass> {
public:
  using Base::Base;

  CanonicalizeShapesPass() : Base() {
    assert(getOpName() && "expected pass to be anchored on specific op type");
    dynamicPM = OpPassManager(*getOpName());
    buildShapeCanonicalizationPipeline(dynamicPM);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    dynamicPM.getDependentDialects(registry);
  }

  LogicalResult initialize(MLIRContext *context) override {
    if (maxIterations.getValue() <= 0)
      return emitError(UnknownLoc::get(context))
             << "the 'maxIterations' value must be >= 0, but got "
             << maxIterations.getValue();
    return success();
  }

  void runOnOperation() override {
    ModuleOp op = getOperation();
    OperationFingerPrint fp(op);

    int64_t currIter = 0;
    while (true) {
      if (failed(runPipeline(dynamicPM, op)))
        return signalPassFailure();

      if (currIter++ >= maxIterations.getValue()) {
        emitError(op.getLoc())
            << "StableHLO dynamic shape canonicalization failed to "
               "converge within "
            << maxIterations.getValue() << " iterations";
        return signalPassFailure();
      }

      // Fixed-point was reached, so we can terminate early.
      OperationFingerPrint newFp(op);
      if (newFp == fp)
        break;

      fp = newFp;
    }
  }

private:
  OpPassManager dynamicPM;
};
} // namespace
