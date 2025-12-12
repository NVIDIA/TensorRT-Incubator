//===- NormalizeQuantizedConversions.cpp ----------------------------------===//
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
/// Definition of the `kernel-normalize-quantized-conversions` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::kernel {
#define GEN_PASS_DEF_KERNELNORMALIZEQUANTIZEDCONVERSIONSPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace mlir::kernel

using namespace mlir;
using namespace mlir::kernel;

namespace {

/// Decompose `x -> f4/f8` to `x -> f16 -> f4/f8`.
/// Special handling for bf16 is needed as follows because bf16 and f16 have
/// equal bitwidth.
/// bf16 -> f4/f8: bf16 -> f32 -> f16 -> f4/f8
struct DecomposeTruncfToQuantized : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const override {
    Type srcType = op.getIn().getType();
    Type dstType = op.getOut().getType();

    // Check if destination is FP4 or FP8.
    if (!isa<Float4E2M1FNType, Float8E4M3FNType>(dstType))
      return failure();

    // If source is already f16, nothing to decompose.
    if (isa<Float16Type>(srcType))
      return failure();

    // Special handling for bf16.
    if (isa<BFloat16Type>(srcType)) {
      Value f32Val = rewriter.create<arith::ExtFOp>(
          op->getLoc(), rewriter.getF32Type(), op.getIn());
      Value f16Val = rewriter.create<arith::TruncFOp>(
          op->getLoc(), rewriter.getF16Type(), f32Val);
      Value quantized =
          rewriter.create<arith::TruncFOp>(op->getLoc(), dstType, f16Val);

      rewriter.replaceOp(op, quantized);
      return success();
    }

    Value f16Val = rewriter.create<arith::TruncFOp>(
        op->getLoc(), rewriter.getF16Type(), op.getIn());
    Value quantized =
        rewriter.create<arith::TruncFOp>(op->getLoc(), dstType, f16Val);

    rewriter.replaceOp(op, quantized);
    return success();
  }
};

/// Decompose f4/f8 -> x to f4/f8 -> f16 -> x.
/// Special handling for bf16 is needed as follows because bf16 and f16 have
/// equal bitwidth.
/// f4/f8 -> bf16: f4/f8 -> f16 -> f32 -> bf16.
struct DecomposeExtfFromQuantized : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const override {
    Type srcType = op.getIn().getType();
    Type dstType = op.getOut().getType();

    // Check if source is FP4 or FP8.
    if (!isa<Float4E2M1FNType, Float8E4M3FNType>(srcType))
      return failure();

    // If destination is already f16, nothing to decompose.
    if (isa<Float16Type>(dstType))
      return failure();

    // Special handling for bf16.
    if (isa<BFloat16Type>(dstType)) {
      Value f16Val = rewriter.create<arith::ExtFOp>(
          op->getLoc(), rewriter.getF16Type(), op.getIn());
      Value f32Val = rewriter.create<arith::ExtFOp>(
          op->getLoc(), rewriter.getF32Type(), f16Val);
      Value bf16Val =
          rewriter.create<arith::TruncFOp>(op->getLoc(), dstType, f32Val);

      rewriter.replaceOp(op, bf16Val);
      return success();
    }

    Value f16Val = rewriter.create<arith::ExtFOp>(
        op->getLoc(), rewriter.getF16Type(), op.getIn());
    Value extended =
        rewriter.create<arith::ExtFOp>(op->getLoc(), dstType, f16Val);

    rewriter.replaceOp(op, extended);
    return success();
  }
};

class KernelNormalizeQuantizedConversionsPass
    : public kernel::impl::KernelNormalizeQuantizedConversionsPassBase<
          KernelNormalizeQuantizedConversionsPass> {
  using Base::Base;
  void runOnOperation() override {
    gpu::GPUModuleOp gpuModuleOp = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeExtfFromQuantized, DecomposeTruncfToQuantized>(ctx);
    if (failed(applyPatternsGreedily(gpuModuleOp, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace