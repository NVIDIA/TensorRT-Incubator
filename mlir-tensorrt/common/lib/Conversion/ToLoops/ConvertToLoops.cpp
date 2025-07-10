//===- ConvertToLoops.cpp ------------------------------------------------===//
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
/// This file contains the implementation of the ConvertToLoops pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Conversion/Passes.h"
#include "mlir-tensorrt-common/Dialect/LinalgExt/Transforms/ToLoopsOpInterfaceImpl.h"
#include "mlir-tensorrt-common/Interfaces/ToLoopsOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOLOOPS
#include "mlir-tensorrt-common/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ConvertToLoopsPattern
    : public OpInterfaceRewritePattern<ToLoopsOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(ToLoopsOpInterface op,
                                PatternRewriter &rewriter) const override {
    if (failed(op.lowerToLoops(rewriter)))
      return failure();
    return success();
  }
};

struct ConvertLinalgOpToLoopsPattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Operation *>> loops =
        linalg_ext::convertLinalgOpToLoops(rewriter, op);
    if (failed(loops))
      return failure();
    rewriter.replaceOp(op, loops->front()->getResults());
    return success();
  }
};

struct ConvertToLoops : public mlir::impl::ConvertToLoopsBase<ConvertToLoops> {
  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    patterns.add<ConvertToLoopsPattern, ConvertLinalgOpToLoopsPattern>(
        op->getContext());

    walkAndApplyPatterns(op, std::move(patterns));
  }
};
} // namespace
