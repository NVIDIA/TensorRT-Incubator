//===- LowerLinalgCopies.cpp ----------------------------------------------===//
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
/// Implementation of `lower-linalg-copies` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LOWERLINALGCOPIESPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct LowerLinalgCopyPattern : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::CopyOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics())
      return rewriter.notifyMatchFailure(op, "expected pure buffer semantics");
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getInputs().front(),
                                                op.getOutputs().front());
    return success();
  }
};

class LowerLinalgCopiesPass
    : public impl::LowerLinalgCopiesPassBase<LowerLinalgCopiesPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<LowerLinalgCopyPattern>(ctx);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
