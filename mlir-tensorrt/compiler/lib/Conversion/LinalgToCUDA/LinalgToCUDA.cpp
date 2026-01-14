//===- LinalgToCUDA.cpp ---------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Implementation of `convert-linalg-to-cuda` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/Passes.h" // IWYU pragma: keep
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/CUDA/Utils/CUDAUtils.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLINALGTOCUDAPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using MemorySpace = plan::MemorySpace;

static std::optional<MemorySpace> getMemorySpace(Type memrefType) {
  auto rankedType = dyn_cast<MemRefType>(memrefType);
  if (!rankedType)
    return {};
  auto memSpace =
      dyn_cast_or_null<plan::MemorySpaceAttr>(rankedType.getMemorySpace());
  if (!memSpace)
    return {};
  return memSpace.getValue();
}

static bool isDeviceVisible(MemorySpace space) {
  return space == MemorySpace::device || space == MemorySpace::unified;
}

static bool isContiguousImpl(ArrayRef<int64_t> strides,
                             ArrayRef<int64_t> shape) {
  unsigned e = strides.size();
  if (shape.empty() || strides.empty())
    return true;

  auto findNextIndex = [&](unsigned start) -> std::optional<unsigned> {
    for (unsigned i = start; i < e; i++) {
      if (shape[i] != 1)
        return i;
    }
    return {};
  };

  // If no starting index, then this is a scalar shape.
  std::optional<unsigned> index = findNextIndex(0);
  if (!index)
    return true;

  while (*index < e) {
    std::optional<unsigned> next = findNextIndex(*index + 1);
    // If this is the last relevant index, it must be unit stride or unit
    // access.
    if (!next)
      return strides[*index] == 1 || shape[*index] == 1;
    if (ShapedType::isDynamic(strides[*index]) ||
        ShapedType::isDynamic(strides[*next]))
      return false;
    if (strides[*index] != strides[*next] * shape[*next])
      return false;
    index = *next;
  }
  return true;
}

static bool isContiguous(MemRefType type) {
  if (type.getLayout().isIdentity())
    return true;
  if (!type.hasStaticShape())
    return false;
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return false;
  return isContiguousImpl(strides, type.getShape());
}

static bool isSupportedFillType(Type elementType) {
  if (!elementType.isIntOrFloat())
    return false;
  int64_t bitwidth = elementType.getIntOrFloatBitWidth();
  return bitwidth == 8 || bitwidth == 16 || bitwidth == 32;
}

namespace {

struct GenericToFillPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const override {
    if (!generic.hasPureBufferSemantics())
      return failure();
    if (generic.getNumDpsInputs() != 0 || generic.getNumDpsInits() != 1)
      return failure();

    auto &block = generic.getRegion().front();
    if (!llvm::hasSingleElement(block))
      return failure();

    auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return failure();

    Value fillVal = yield.getValues().front();
    if (fillVal.getParentBlock() == &block)
      return failure();

    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        generic, fillVal, generic.getDpsInitOperand(0)->get());
    return success();
  }
};

struct LinalgFillToCUDAMemsetPattern : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics())
      return failure();
    auto memrefType = dyn_cast<MemRefType>(op.getOutputs().front().getType());
    if (!memrefType)
      return failure();
    if (!isContiguous(memrefType))
      return rewriter.notifyMatchFailure(op, "memref type not contiguous");

    std::optional<MemorySpace> space = getMemorySpace(memrefType);
    if (!space || !isDeviceVisible(*space))
      return failure();

    Type elementType = memrefType.getElementType();
    if (op.getInputs().front().getType() != elementType)
      return rewriter.notifyMatchFailure(
          op, "expected fill value type to match memref element type");
    if (!isSupportedFillType(elementType))
      return rewriter.notifyMatchFailure(
          op, "element type bitwidth must be 8, 16, or 32");

    Value stream = cuda::getOrCreateDefaultStream0(rewriter, op);
    rewriter.replaceOpWithNewOp<cuda::MemSetOp>(
        op, stream, op.getOutputs().front(), op.getInputs().front());
    return success();
  }
};

class ConvertLinalgToCUDAPass
    : public impl::ConvertLinalgToCUDAPassBase<ConvertLinalgToCUDAPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();

    {
      RewritePatternSet patterns(ctx);
      patterns.add<GenericToFillPattern>(ctx);
      if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
        emitError(op->getLoc())
            << "failed to simplify linalg.generic to linalg.fill";
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns(ctx);
      patterns.add<LinalgFillToCUDAMemsetPattern>(ctx);
      if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
        emitError(op->getLoc()) << "failed to lower linalg.fill to cuda.memset";
        return signalPassFailure();
      }
    }
  }
};

} // namespace
