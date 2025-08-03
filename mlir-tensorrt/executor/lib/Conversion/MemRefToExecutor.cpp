//===- MemRefToExecutorPatterns.cpp ---------------------------------------===//
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
/// Defines conversion patterns for MemRef dialect to Executor dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Conversion/ConvertToExecutorCommon.h"
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "memref-to-executor"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "]"

namespace mlir::executor {
#define GEN_PASS_DEF_CONVERTMEMREFTOEXECUTORPASS
#include "mlir-executor/Conversion/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using executor::ConvertOpToExecutorPattern;
using executor::ExecutorTypeConverter;
using executor::LowerToExecutorOptions;
using executor::MemoryType;
using executor::MemoryTypeAttr;
using executor::MemRefDescriptor;

/// Return the alignment required by an AllocOp (assuming conversion to
/// Executor's aligned allocation function, which is lowered to a call to
/// `std::aligned_alloc` at runtime).
static FailureOr<uint64_t>
getAlignment(memref::AllocOp op, const ExecutorTypeConverter &typeConverter) {
  if (std::optional<uint64_t> alignment = op.getAlignment())
    return *alignment;
  Type convertedElementType =
      typeConverter.convertType(op.getType().getElementType());
  if (!convertedElementType)
    return failure();
  return typeConverter.getDataLayout().getTypePreferredAlignment(
      convertedElementType);
}

//===----------------------------------------------------------------------===//
// Allocation op lowerings
//===----------------------------------------------------------------------===//

namespace {
/// Convert `memref.alloc` to`executor.allocate`, which has the semantics of an
/// aligned allocation. Replace the result with the descriptor.
class ConvertAlloc : public ConvertOpToExecutorPattern<memref::AllocOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto memrefType = op.getMemref().getType();
    if (!memrefType.hasRank())
      return rewriter.notifyMatchFailure(op, "cannot convert unranked memref");
    Type resultType = getTypeConverter()->convertType(memrefType);
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "could not convert memref type");

    FailureOr<MemRefAllocationInformation> info =
        getMemRefAllocationInformation(b, memrefType,
                                       adaptor.getDynamicSizes());
    if (failed(info))
      return rewriter.notifyMatchFailure(op, "failed to get allocation info");

    // Get the alignement requirement and memory space.
    FailureOr<uint64_t> alignment = getAlignment(op, *getTypeConverter());
    if (failed(alignment))
      return rewriter.notifyMatchFailure(op, "failed to get alignment");
    Value alignmentValue = createIndexConstant(b, *alignment);

    Value alloc;
    if (info->memorySpace == MemoryType::host) {
      alloc = b.create<executor::AllocateOp>(
          getTypeConverter()->getOpaquePointerType(info->memorySpace),
          info->sizeBytes, alignmentValue);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported memory space");
    }

    auto memref = MemRefDescriptor::fromComponents(
        b, *getTypeConverter(), memrefType, alloc, alloc,
        createIndexConstant(b, 0), info->sizes, info->strides);
    rewriter.replaceOp(op, {memref});
    return success();
  }

private:
  /// Default data layout.
  DataLayout defaultLayout;
};

/// Convert `memref.dealloc` to `executor.deallocate`.
struct ConvertMemRefDealloc
    : public ConvertOpToExecutorPattern<memref::DeallocOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MemRefDescriptor desc(adaptor.getMemref(),
                          llvm::cast<MemRefType>(op.getMemref().getType()));

    auto memoryTypeAttr = dyn_cast_or_null<executor::MemoryTypeAttr>(
        op.getMemref().getType().getMemorySpace());
    MemoryType memoryType =
        memoryTypeAttr ? memoryTypeAttr.getValue() : MemoryType::host;

    if (memoryType == MemoryType::host) {
      rewriter.replaceOpWithNewOp<executor::DeallocateOp>(op,
                                                          desc.allocatedPtr(b));
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported memory space");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Load/store/copy lowerings.
//===----------------------------------------------------------------------===//

/// Return true if a load on the given memref type is allowed. In the future, we
/// may want to be more restrictive and require the post-conditions of a pass
/// that decomposes loads into offset arithmetic + load on a 0D `memref<f32>`.
static bool convertLoadStorePreconditions(MemRefType memrefType) {
  return memrefType.hasRank();
}

namespace {
/// Convert `memref.load` to `executor.load`.
struct ConvertLoad : public ConvertOpToExecutorPattern<memref::LoadOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();
    auto memrefType = llvm::cast<MemRefType>(op.getMemref().getType());
    if (!convertLoadStorePreconditions(memrefType) ||
        !isHostVisibleMemoryType(memrefType))
      return failure();

    MemRefDescriptor memref(adaptor.getMemref(), memrefType);
    Value byteOffset = convertOffsetInElementsToBytes(
        b, getLinearizedOffset(b, memref, adaptor.getIndices()), memrefType);
    rewriter.replaceOpWithNewOp<executor::LoadOp>(
        op, resultType, memref.alignedPtr(b), byteOffset);
    return success();
  }
};

/// Convert `memref.store` to `executor.store`. This relies on a preperatory
/// pass that adjusts all `memref.load` operations to be operating at an
/// offset/subview such that the descriptor of the source memref has all unit
/// sizes.
class ConvertStore : public ConvertOpToExecutorPattern<memref::StoreOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto memrefType = llvm::cast<MemRefType>(op.getMemref().getType());
    if (!convertLoadStorePreconditions(memrefType) ||
        !isHostVisibleMemoryType(memrefType))
      return failure();

    MemRefDescriptor memref(adaptor.getMemref(), memrefType);
    Value byteOffset = convertOffsetInElementsToBytes(
        b, getLinearizedOffset(b, memref, adaptor.getIndices()), memrefType);
    rewriter.replaceOpWithNewOp<executor::StoreOp>(
        op, memref.alignedPtr(b), byteOffset, adaptor.getValue());
    return success();
  }
};

/// Convert `memref.copy` to `executor.memcpy` or one of the CUDA variants.
struct ConvertMemRefCopy : public ConvertOpToExecutorPattern<memref::CopyOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // Determine whether the memref is contiguous.
    auto srcType = llvm::cast<MemRefType>(op.getSource().getType());
    auto dstType = llvm::cast<MemRefType>(op.getTarget().getType());
    if (!isHostVisibleOnlyMemoryType(srcType) ||
        !isHostVisibleOnlyMemoryType(dstType))
      return failure();

    MemRefDescriptor src(adaptor.getSource(), srcType);
    MemRefDescriptor dest(adaptor.getTarget(), dstType);
    Value srcOffset = convertOffsetInElementsToBytes(b, src.offset(b), srcType);
    Value dstOffset =
        convertOffsetInElementsToBytes(b, dest.offset(b), dstType);

    if (isCopyStrided(srcType, dstType))
      return failure();

    // By definition, contiguous copies are not strided and thus the copy size
    // is equivalent to the shape volume (stride can be disregarded).
    Value sizeBytes = convertOffsetInElementsToBytes(
        b, src.shapeVolumeInElements(b), srcType);

    rewriter.replaceOpWithNewOp<executor::MemcpyOp>(
        op, src.alignedPtr(b), srcOffset, dest.alignedPtr(b), dstOffset,
        sizeBytes);
    return success();
  }
};

/// Convert `memref.copy` when it cannot be represented by a single contiguous
/// memcpy.
struct ConvertMemRefCopyStrided2D
    : public ConvertOpToExecutorPattern<memref::CopyOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // Determine whether the memref is contiguous.
    auto srcType = llvm::cast<MemRefType>(op.getSource().getType());
    auto dstType = llvm::cast<MemRefType>(op.getTarget().getType());
    if (!isCopyStrided(srcType, dstType) ||
        !isHostVisibleOnlyMemoryType(srcType) ||
        !isHostVisibleOnlyMemoryType(dstType))
      return failure();

    MemRefDescriptor src(adaptor.getSource(), srcType);
    MemRefDescriptor dst(adaptor.getTarget(), dstType);

    SmallVector<Value> operands = {
        createIndexConstant(b, srcType.getRank()),
        createIndexConstant(
            b, getTypeConverter()->getMemRefElementTypeByteSize(srcType))};
    operands.append(src.unpack(b));
    operands.append(dst.unpack(b));

    rewriter.replaceOpWithNewOp<executor::StridedMemrefCopyOp>(
        op, TypeRange{}, ValueRange(operands));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Cast op lowerings
//===----------------------------------------------------------------------===//

/// Convert `memref.reinterpret_cast` to a pack into a `executor.table`
/// descriptor.
struct ReinterpretCastOpLowering
    : public ConvertOpToExecutorPattern<memref::ReinterpretCastOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto memrefSourceType = dyn_cast<MemRefType>(op.getSource().getType());
    if (!memrefSourceType)
      return failure();
    auto memrefType = llvm::cast<MemRefType>(op.getType());
    auto resultDescriptorType = getTypeConverter()->convertType(memrefType);
    if (!resultDescriptorType)
      return failure();

    // The source adaptor is for the 0-rank memref argument. The other
    // shape/stride values are in the other arguments.
    MemRefDescriptor sourceDescriptor(adaptor.getSource(), memrefSourceType);
    Value basePtr = sourceDescriptor.allocatedPtr(b);
    Value alignedPtr = sourceDescriptor.alignedPtr(b);
    OpFoldResult offset =
        op.isDynamicOffset(0)
            ? adaptor.getOffsets()[0]
            : this->createIndexConstant(b, op.getStaticOffsets()[0]);

    // Create mixed static/dynamic values for shape and strides from the adaptor
    // lists.
    SmallVector<OpFoldResult> mixedSizes = mlir::getMixedValues(
        adaptor.getStaticSizes(), adaptor.getSizes(), rewriter);
    SmallVector<OpFoldResult> mixedStrides = mlir::getMixedValues(
        adaptor.getStaticStrides(), adaptor.getStrides(), rewriter);

    rewriter.replaceOp(op, {MemRefDescriptor::fromComponents(
                               b, *getTypeConverter(), memrefType, basePtr,
                               alignedPtr, offset, mixedSizes, mixedStrides)});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Global lowerings
//===----------------------------------------------------------------------===//

/// Convert `memref.global` to `executor.global`.
struct ConvertMemrefGlobal
    : public ConvertOpToExecutorPattern<memref::GlobalOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType memrefType = op.getType();

    /// If there's no initial value, then just perform the allocation.
    ElementsAttr initialValue =
        llvm::dyn_cast_or_null<ElementsAttr>(op.getInitialValueAttr());
    if (!initialValue)
      initialValue =
          cast<ElementsAttr>(rewriter.getZeroAttr(RankedTensorType::get(
              memrefType.getShape(), memrefType.getElementType())));

    std::optional<executor::MemoryType> space = getMemorySpace(memrefType);
    if (!space)
      return failure();

    auto segmentOp = rewriter.create<executor::DataSegmentOp>(
        op.getLoc(), op.getName(), initialValue,
        /*constant=*/op.getConstant(),
        /*uninitialized=*/!op.getInitialValue(),
        /*alignment=*/op.getAlignmentAttr());
    segmentOp.setAddressSpace(*space);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert `memref.get_global` to `executor.get_global`.
struct ConvertMemrefGetGlobal
    : public ConvertOpToExecutorPattern<memref::GetGlobalOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType memrefType = op.getType();

    std::optional<executor::MemoryType> space = getMemorySpace(memrefType);
    if (!space)
      return failure();

    Value ptr = rewriter.create<executor::ConstantResourceLoadOp>(
        op.getLoc(), executor::PointerType::get(rewriter.getContext(), *space),
        op.getName());
    ImplicitLocOpBuilder ib(op.getLoc(), rewriter);

    FailureOr<MemRefAllocationInformation> info =
        getMemRefAllocationInformation(ib, memrefType, {});
    if (failed(info))
      return failure();

    Value zero = createIndexConstant(ib, 0);
    auto memref = MemRefDescriptor::fromComponents(ib, *getTypeConverter(),
                                                   memrefType, ptr, ptr, zero,
                                                   info->sizes, info->strides);
    rewriter.replaceOp(op, Value(memref));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Metadata op lowerings
//===----------------------------------------------------------------------===//

/// Convert `memref.dim` into executor aggregate access.
struct ConvertDim : public ConvertOpToExecutorPattern<memref::DimOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(memref::DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto memrefType = llvm::cast<MemRefType>(op.getSource().getType());
    APInt indexConst;
    if (!matchPattern(op.getIndex(), m_ConstantInt(&indexConst)))
      return failure();
    MemRefDescriptor source(adaptor.getSource(), memrefType);
    rewriter.replaceOp(op, {source.size(b, indexConst.getSExtValue())});
    return success();
  }
};

/// Convert `memref.extract_strided_metadta` to executor operations.
struct ConvertExtractMetadata
    : public ConvertOpToExecutorPattern<memref::ExtractStridedMetadataOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto memrefType = llvm::cast<MemRefType>(op.getSource().getType());
    MemRefDescriptor source(adaptor.getSource(), memrefType);
    Value alignedPtr = source.alignedPtr(b);
    auto baseMemRefType = llvm::cast<MemRefType>(op.getBaseBuffer().getType());
    auto baseBufferDesc = MemRefDescriptor::fromComponents(
        b, *getTypeConverter(), baseMemRefType, alignedPtr, alignedPtr,
        createIndexConstant(b, 0), {}, {});

    SmallVector<Value> replacements;
    replacements.reserve(op->getNumResults());
    replacements.append({baseBufferDesc, source.offset(b)});
    llvm::append_range(replacements, source.sizes(b));
    llvm::append_range(replacements, source.strides(b));
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// Convert `memref.cast` by just forwarding the converted value if we are
/// casting from a ranked memref type to a more general strided memref type.
/// TODO: support more general cast with runtime assertions.
struct ConvertMemRefCast : public ConvertOpToExecutorPattern<memref::CastOp> {
  ConvertMemRefCast(bool allowUncheckedMemrefCastConversion,
                    ExecutorTypeConverter &typeConverter, MLIRContext *ctx,
                    PatternBenefit benefit = 1)
      : ConvertOpToExecutorPattern(typeConverter, ctx, benefit),
        allowUncheckedMemrefCastConversion(allowUncheckedMemrefCastConversion) {
  }

  LogicalResult
  matchAndRewrite(memref::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check that we have ranked memrefs.
    MemRefType src = dyn_cast<MemRefType>(op.getSource().getType());
    MemRefType dst = dyn_cast<MemRefType>(op.getType());
    if (!dst || !src)
      return failure();

    if (allowUncheckedMemrefCastConversion) {
      rewriter.replaceOp(op, adaptor.getSource());
      return success();
    }

    // Get the strides and offsets of each type. According to `memref.cast`
    // documentation, the cast should be lowered to a runtime assertion if the
    // destination type is more specific than the source type. For now we only
    // support "more specific" -> "more general" cast since we don't need to
    // generate conversions in this case.
    auto [srcStrides, srcOffset] = src.getStridesAndOffset();
    auto [dstStrides, dstOffset] = dst.getStridesAndOffset();

    // `lhs` is cast-able to `rhs` without a runtime assertion check if `lhs`
    // is equal to `rhs` or if `rhs` is less specific (more general) than `lhs`,
    // which can only occur if `rhs` is `?` and `lhs` is a static number.
    auto isEqualOrLessGeneralThan = [](int64_t lhs, int64_t rhs) {
      return lhs == rhs || ShapedType::isDynamic(rhs);
    };

    if (!isEqualOrLessGeneralThan(srcOffset, dstOffset) ||
        !llvm::all_of(llvm::zip(srcStrides, dstStrides),
                      [&](const auto &srcDstPair) {
                        auto [src, dst] = srcDstPair;
                        return isEqualOrLessGeneralThan(src, dst);
                      }))
      return rewriter.notifyMatchFailure(
          op, "not a cast from more specific to more general type");

    // Just forward the converted value (with type !executor.table).
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
  bool allowUncheckedMemrefCastConversion;
};

/// Unpack the pointer returned by a memref.extract_aligned_pointer_as_index.
class ConvertExtractAlignedPointerAsIndex
    : public ConvertOpToExecutorPattern<
          memref::ExtractAlignedPointerAsIndexOp> {
public:
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractAlignedPointerAsIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType sourceType = dyn_cast<MemRefType>(op.getSource().getType());
    if (!sourceType)
      return failure();
    MemRefDescriptor desc(adaptor.getSource(), sourceType);
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    rewriter.replaceOpWithNewOp<executor::PtrToIntOp>(
        op, getTypeConverter()->getIndexType(), desc.alignedPtr(b));
    return success();
  }
};
} // namespace

void executor::populateMemRefToExecutorPatterns(
    RewritePatternSet &patterns, ExecutorTypeConverter &typeConverter,
    bool allowUncheckedMemrefCastConversion) {
  patterns.add<ConvertLoad, ConvertStore, ConvertAlloc,
               ReinterpretCastOpLowering, ConvertExtractMetadata, ConvertDim,
               ConvertMemRefCopy, ConvertMemRefCopyStrided2D,
               ConvertMemRefDealloc, ConvertMemrefGlobal,
               ConvertMemrefGetGlobal, ConvertExtractAlignedPointerAsIndex>(
      typeConverter, patterns.getContext());
  patterns.add<ConvertMemRefCast>(allowUncheckedMemrefCastConversion,
                                  typeConverter, patterns.getContext());
}

namespace {
/// Pass to convert `memref` to `executor` dialect operrations.
class ConvertMemRefToExecutorPass
    : public mlir::executor::impl::ConvertMemRefToExecutorPassBase<
          ConvertMemRefToExecutorPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    executor::ExecutorConversionTarget target(*ctx);
    LowerToExecutorOptions opts;
    // We allow index type during memref lowering prior to lowering of certain
    // executor ops to func-calls.
    opts.indexType = IntegerType::get(ctx, indexBitwidth);
    Operation *op = getOperation();
    FailureOr<DataLayout> dataLayout =
        executor::setDataLayoutSpec(op, indexBitwidth, 64);
    if (failed(dataLayout)) {
      emitError(op->getLoc())
          << "failed to set DataLayout; op has DLTI spec that is "
             "inconsistent with provided options";
      return signalPassFailure();
    }
    ExecutorTypeConverter typeConverter(ctx, opts, std::move(*dataLayout));
    // We create executor constants in this pass. Mark them as legal.
    target.addIllegalDialect<memref::MemRefDialect>();
    RewritePatternSet patterns(ctx);
    executor::populateMemRefToExecutorPatterns(
        patterns, typeConverter, allowUncheckedMemrefCastConversion);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
