//===- TestTypeInferencePass.cpp ------------------------------------------===//
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
/// This file contains the implementation of the `test-tensorrt-shape-inference`
/// pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace mlir::tensorrt {
void registerTestTensorRTShapeInferencePass();
}

struct DimOfReifyRankedShapedTypeOpInterface
    : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() {
    OpRewritePattern<tensor::DimOp>::setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    OpResult dimValue = dyn_cast<OpResult>(dimOp.getSource());
    if (!dimValue)
      return failure();
    std::optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex)
      return failure();

    ReifiedRankedShapedTypeDims reifiedResultShapes;
    if (failed(reifyResultShapes(rewriter, dimValue.getOwner(),
                                 reifiedResultShapes)))
      return failure();
    unsigned resultNumber = dimValue.getResultNumber();
    // Do not apply pattern if the IR is invalid (dim out of bounds).
    if ((size_t)(*dimIndex) >= reifiedResultShapes[resultNumber].size())
      return rewriter.notifyMatchFailure(dimOp, "dimension is out of bounds");
    Value replacement = getValueOrCreateConstantIndexOp(
        rewriter, dimOp.getLoc(), reifiedResultShapes[resultNumber][*dimIndex]);
    rewriter.replaceOp(dimOp, replacement);
    return success();
  }
};

namespace {
class TestTypeInferencePass
    : public mlir::PassWrapper<TestTypeInferencePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTypeInferencePass)

  TestTypeInferencePass() = default;

  llvm::StringRef getArgument() const override {
    return "test-tensorrt-shape-inference";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DimOfReifyRankedShapedTypeOpInterface>(ctx);
    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to converge in " << getArgument();
      return signalPassFailure();
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect>();
  }
};
} // namespace

void tensorrt::registerTestTensorRTShapeInferencePass() {
  PassRegistration<TestTypeInferencePass>();
}
