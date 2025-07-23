//===- MemRefToCUDA.cpp ---------------------------------------------------===//
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
/// Implementation of `convert-memref-to-cuda` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFTOCUDAPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using MemorySpace = plan::MemorySpace;

static std::optional<MemorySpace> getMemorySpace(Type memref) {
  auto memrefType = cast<MemRefType>(memref);
  if (!memrefType)
    return {};
  auto memSpace =
      dyn_cast_or_null<plan::MemorySpaceAttr>(memrefType.getMemorySpace());
  if (!memSpace)
    return {};
  return memSpace.getValue();
}

static bool isHostOnly(MemorySpace t) {
  return t == MemorySpace::host || t == MemorySpace::host_pinned;
}
static bool isHostOnly(BaseMemRefType t) {
  auto memSpace = dyn_cast_or_null<plan::MemorySpaceAttr>(t.getMemorySpace());
  if (!memSpace)
    return false;
  return isHostOnly(memSpace.getValue());
}

static bool isDeviceVisible(MemorySpace t) {
  return t == MemorySpace::device || t == MemorySpace::unified;
}

static bool isPinned(BaseMemRefType t) {
  auto memSpace = dyn_cast_or_null<plan::MemorySpaceAttr>(t.getMemorySpace());
  if (!memSpace)
    return false;
  return memSpace.getValue() == MemorySpace::host_pinned;
}

static bool isDeviceOrHostPinned(BaseMemRefType t) {
  auto memSpace = dyn_cast_or_null<plan::MemorySpaceAttr>(t.getMemorySpace());
  if (!memSpace)
    return false;
  return isPinned(t) || isDeviceVisible(memSpace.getValue());
}

namespace {

struct MemRefCopyToCUDACopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<MemorySpace> sourceSpace =
        getMemorySpace(op.getSource().getType());
    if (!sourceSpace)
      return failure();
    std::optional<MemorySpace> targetSpace =
        getMemorySpace(op.getTarget().getType());
    if (!targetSpace)
      return failure();

    Location loc = op.getLoc();
    Value device = rewriter.create<cuda::GetActiveDeviceOp>(loc);
    Value stream = rewriter.create<cuda::GetGlobalStreamOp>(loc, device, 0);

    if (isDeviceVisible(*sourceSpace) && isDeviceVisible(*targetSpace)) {
      rewriter.replaceOpWithNewOp<cuda::CopyD2DOp>(op, stream, op.getSource(),
                                                   op.getTarget());
      return success();
    }

    if (isHostOnly(*sourceSpace) && isDeviceVisible(*targetSpace)) {
      rewriter.replaceOpWithNewOp<cuda::CopyH2DOp>(op, stream, op.getSource(),
                                                   op.getTarget());
      return success();
    }

    if (isDeviceVisible(*sourceSpace) && isHostOnly(*targetSpace)) {
      rewriter.create<cuda::CopyD2HOp>(loc, stream, op.getSource(),
                                       op.getTarget());
      rewriter.create<cuda::StreamSyncOp>(loc, stream);
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

struct MemRefAllocToCUDAAllocPattern
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<MemorySpace> space = getMemorySpace(op.getType());
    if (!space || *space == MemorySpace::host)
      return failure();
    Location loc = op.getLoc();
    if (*space != MemorySpace::host_pinned) {
      Value device = rewriter.create<cuda::GetActiveDeviceOp>(loc);
      Value stream = rewriter.create<cuda::GetGlobalStreamOp>(loc, device, 0);
      rewriter.replaceOpWithNewOp<cuda::AllocOp>(op, op.getType(), stream,
                                                 op.getDynamicSizes(),
                                                 op.getAlignmentAttr());
      return success();
    }
    rewriter.replaceOpWithNewOp<cuda::AllocOp>(
        op, op.getType(), Value{}, op.getDynamicSizes(), op.getAlignmentAttr());
    return success();
  }
};

struct MemRefDeallocToCUDADeallocPattern
    : public OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<MemorySpace> space = getMemorySpace(op.getMemref().getType());
    if (!space || *space == MemorySpace::host)
      return failure();
    Value device = rewriter.create<cuda::GetActiveDeviceOp>(op.getLoc());
    Value stream =
        rewriter.create<cuda::GetGlobalStreamOp>(op.getLoc(), device, 0);
    rewriter.replaceOpWithNewOp<cuda::DeallocOp>(op, stream, op.getMemref());
    return success();
  }
};

class MemRefToCUDAPass
    : public impl::ConvertMemRefToCUDAPassBase<MemRefToCUDAPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    ConversionTarget target(getContext());
    target.addLegalDialect<cuda::CUDADialect>();
    target.addDynamicallyLegalOp<memref::AllocOp>(
        [](memref::AllocOp op) { return !isDeviceOrHostPinned(op.getType()); });
    target.addDynamicallyLegalOp<memref::CopyOp>([](memref::CopyOp op) {
      if (isHostOnly(op.getTarget().getType()) &&
          isHostOnly(op.getSource().getType()))
        return true;
      return !isDeviceOrHostPinned(op.getSource().getType()) &&
             !isDeviceOrHostPinned(op.getTarget().getType());
    });
    target.addDynamicallyLegalOp<memref::DeallocOp>([](memref::DeallocOp op) {
      return !isDeviceOrHostPinned(op.getMemref().getType());
    });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });

    RewritePatternSet patterns(ctx);
    // clang-format off
    patterns.add<
        MemRefAllocToCUDAAllocPattern,
        MemRefCopyToCUDACopyPattern,
        MemRefDeallocToCUDADeallocPattern
      >(ctx);
    // clang-format on
    Operation *op = getOperation();
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      emitError(op->getLoc())
          << "failed to apply rewrite patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
