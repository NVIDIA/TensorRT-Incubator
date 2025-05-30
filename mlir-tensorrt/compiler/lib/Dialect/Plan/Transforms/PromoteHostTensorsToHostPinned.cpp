//===- PromoteHostTensorsToHostPinned.cpp --------------------------------===//
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
///  Implementation of the `plan-promote-host-tensors-to-host-pinned` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "plan-assign-memory-spaces"

namespace mlir::plan {
#define GEN_PASS_DEF_PLANPROMOTEHOSTTENSORSTOHOSTPINNEDPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

namespace {

template <plan::MemorySpace sourceSpace, plan::MemorySpace destSpace>
struct CastPromotionPattern : public OpRewritePattern<tensor::CastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::CastOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<RankedTensorType>(op.getOperand().getType());
    auto destType = dyn_cast<RankedTensorType>(op.getType());
    if (!sourceType || !destType)
      return failure();
    auto sourceSpaceAttr =
        llvm::dyn_cast_if_present<MemorySpaceAttr>(sourceType.getEncoding());
    auto destSpaceAttr =
        llvm::dyn_cast_if_present<MemorySpaceAttr>(destType.getEncoding());
    if (!sourceSpaceAttr || !destSpaceAttr)
      return failure();
    if (sourceSpaceAttr.getValue() != sourceSpace ||
        destSpaceAttr.getValue() != destSpace)
      return failure();
    return handleCast(op, sourceType, destType, rewriter);
  }

  virtual LogicalResult handleCast(tensor::CastOp op,
                                   RankedTensorType sourceType,
                                   RankedTensorType destType,
                                   PatternRewriter &rewriter) const = 0;

  plan::MemorySpaceAttr hostSpaceAttr =
      plan::MemorySpaceAttr::get(getContext(), plan::MemorySpace::host);
  plan::MemorySpaceAttr hostPinnedSpaceAttr =
      plan::MemorySpaceAttr::get(getContext(), plan::MemorySpace::host_pinned);
  plan::MemorySpaceAttr deviceSpaceAttr =
      plan::MemorySpaceAttr::get(getContext(), plan::MemorySpace::device);
};

// Pattern for promoting device->host cast as device->host-pinned cast if all
// the cast's users can be updated in-place.
struct DeviceToHostCastPattern
    : public CastPromotionPattern<plan::MemorySpace::device,
                                  plan::MemorySpace::host> {
  using CastPromotionPattern::CastPromotionPattern;

  LogicalResult handleCast(tensor::CastOp op, RankedTensorType sourceType,
                           RankedTensorType destType,
                           PatternRewriter &rewriter) const override {
    // TODO: this should be replaced with some more general conditions. To
    // handle `tensor.insert`, we would need to cast type of `tensor.insert`
    // result back to original, propogate it forward, etc.
    if (!llvm::all_of(op->getUsers(), llvm::IsaPred<tensor::ExtractOp>))
      return failure();
    auto newType = destType.cloneWithEncoding(hostPinnedSpaceAttr);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newType, op.getOperand());
    return success();
  }
};

// Propogate host->device cast as host->device-pinned cast if the producer is a
// `tensor.from_elements` operation.
struct FromElementsPromotionPattern
    : public CastPromotionPattern<plan::MemorySpace::host,
                                  plan::MemorySpace::device> {
  using CastPromotionPattern::CastPromotionPattern;

  LogicalResult handleCast(tensor::CastOp op, RankedTensorType sourceType,
                           RankedTensorType destType,
                           PatternRewriter &rewriter) const override {
    auto fromElementsOp =
        dyn_cast<tensor::FromElementsOp>(op.getOperand().getDefiningOp());
    if (!fromElementsOp)
      return failure();
    auto hostPinnedType = sourceType.cloneWithEncoding(hostPinnedSpaceAttr);
    rewriter.modifyOpInPlace(fromElementsOp, [&]() {
      fromElementsOp.getResult().setType(hostPinnedType);
    });
    return success();
  };
};

class PromoteHostTensorsToHostPinnedPass
    : public plan::impl::PlanPromoteHostTensorsToHostPinnedPassBase<
          PromoteHostTensorsToHostPinnedPass> {
  using Base::Base;
  void runOnOperation() override {
    auto op = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<DeviceToHostCastPattern, FromElementsPromotionPattern>(
        patterns.getContext());
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
