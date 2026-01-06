//===- CanonicalizeShapes.cpp ---------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
#include "mlir-tensorrt-common/Utils/ModuleUtils.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/StablehloRefineShapes.h"
#include "stablehlo/transforms/optimization/Passes.h"

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_CANONICALIZESHAPESPASS
#define GEN_PASS_DEF_REFINESHAPESPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

using namespace mlir;
using namespace mlir::stablehlo_ext;

/// Returns the unique entrypoint function of the module if present. Otherwise,
/// it returns nullptr. Stablehlo's `stablehlo-refine-shapes` implements a
/// limited refinement algorithm that only supports a single unique entrypoint
/// function. To be conservative, we additionally check that the callgraph
/// contains no recursion since the Stablehlo documentation does not explicitly
/// state what is supported.
static func::FuncOp getUniqueEntrypointFunction(ModuleOp module) {
  SmallVector<FunctionOpInterface> funcs, remainingFuncs;
  if (failed(mlir::getFuncOpsOrderedByCalls(ModuleLikeOp(module), funcs,
                                            remainingFuncs)))
    return nullptr;
  if (!remainingFuncs.empty())
    return nullptr;
  func::FuncOp entrypoint{nullptr};
  for (FunctionOpInterface func : funcs) {
    if (!func.getFunctionBody().hasOneBlock())
      return nullptr;
    if (!func.isPublic())
      continue;
    if (!isa<func::FuncOp>(func.getOperation()))
      return nullptr;
    if (entrypoint)
      return nullptr;
    entrypoint = cast<func::FuncOp>(func.getOperation());
  }
  return entrypoint;
}

namespace {

struct ReturnCastRewritePattern : public OpRewritePattern<tensor::CastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::CastOp op,
                                PatternRewriter &rewriter) const override {
    if (!llvm::any_of(op->getUsers(), llvm::IsaPred<func::ReturnOp>))
      return failure();
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                            op.getSource());
    return success();
  }
};

/// Implementation of `stablehlo-ext-refine-shapes`.
class RefineShapesPass
    : public stablehlo_ext::impl::RefineShapesPassBase<RefineShapesPass> {
public:
  using Base::Base;

  LogicalResult initialize(MLIRContext *context) override {
    folderOptions =
        std::make_shared<stablehlo::StablehloAggressiveFolderPassOptions>();
    folderOptions->optimizeFloat = false;
    folderOptions->foldOpElementLimit = 128;

    patterns = [&]() {
      RewritePatternSet patterns_(context);
      stablehlo::populateStablehloRefineShapesPatterns(context, &patterns_);
      stablehlo::populateStablehloShapeFolderPatterns(context, &patterns_,
                                                      *folderOptions);
      return FrozenRewritePatternSet(std::move(patterns_));
    }();

    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (interprocedural) {
      if (func::FuncOp entrypoint = getUniqueEntrypointFunction(module)) {
        if (failed(stablehlo::refineEntryFunction(
                getContext(), entrypoint, [](RewritePatternSet *patterns) {
                  patterns->add<ReturnCastRewritePattern>(
                      patterns->getContext());
                })))
          return signalPassFailure();

        return;
      }
    }
    // We can't refine functions, but fallback to refining individual
    // operations.
    if (failed(applyPatternsGreedily(module, patterns))) {
      emitError(module.getLoc())
          << "failed to apply patterns in " << getArgument() << "\n";
      return signalPassFailure();
    }
  }

  FrozenRewritePatternSet patterns;
  std::shared_ptr<stablehlo::StablehloAggressiveFolderPassOptions>
      folderOptions{nullptr};
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
